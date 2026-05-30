#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable, Mapping

from openai import OpenAI

DATASET_DIR = Path(__file__).resolve().parent
DATASET_SUBDIRS = ("External_Function", "Internal_Function", "Hybrid_Function")
GENERATED_DIR_NAMES = {"prompt_call_answer", "function_call_answer"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def discover_inputs(input_path: Path, output_root: Path) -> list[tuple[Path, Path]]:
    input_path = input_path.resolve()
    if input_path.is_file():
        return [(input_path.parent, input_path)]
    if input_path == DATASET_DIR.resolve():
        paths = []
        for subdir in DATASET_SUBDIRS:
            paths.extend(sorted((input_path / subdir).glob("*.jsonl")))
        return [(input_path, path) for path in paths]
    return [
        (input_path, path)
        for path in sorted(input_path.rglob("*.jsonl"))
        if not is_relative_to(path, output_root)
        and not any(part in GENERATED_DIR_NAMES for part in path.parts)
    ]


def safe_name(value: str) -> str:
    cleaned = value.replace(os.sep, "_").replace("/", "_").replace("\\", "_")
    cleaned = re.sub(r"[^A-Za-z0-9._=-]+", "_", cleaned)
    return cleaned.strip("_") or "unnamed"


def output_path(output_root: Path, input_root: Path, data_path: Path, model: str) -> Path:
    rel_path = data_path.relative_to(input_root)
    return output_root / rel_path.parent / f"{data_path.stem}_{safe_name(model)}.jsonl"


def sample_idx(sample: Mapping[str, Any], fallback: int) -> int:
    value = sample.get("index")
    if isinstance(value, bool):
        return fallback
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return fallback
    return fallback


def parse_tool_doc(raw_tool_doc: Any) -> list[dict[str, Any]]:
    if not raw_tool_doc:
        return []
    if isinstance(raw_tool_doc, str):
        return [json.loads(raw_tool_doc)]
    if isinstance(raw_tool_doc, list):
        tools = []
        for item in raw_tool_doc:
            tools.extend(parse_tool_doc(item))
        return tools
    if isinstance(raw_tool_doc, dict) and "name" in raw_tool_doc:
        return [raw_tool_doc]
    if isinstance(raw_tool_doc, dict):
        tools = []
        for item in raw_tool_doc.values():
            tools.extend(parse_tool_doc(item))
        return tools
    return []


def convert_input_schema_to_parameters(input_schema: Mapping[str, Any]) -> dict[str, Any]:
    if not input_schema:
        return {"type": "object", "properties": {}, "required": []}

    parameters = {
        "type": input_schema.get("type", "object"),
        "properties": {},
        "required": input_schema.get("required", []),
    }
    for prop_name, prop_info in input_schema.get("properties", {}).items():
        if not isinstance(prop_info, Mapping):
            continue
        prop = {
            "type": prop_info.get("type", "string"),
            "description": prop_info.get("description", prop_info.get("title", "")),
        }
        if "enum" in prop_info:
            prop["enum"] = prop_info["enum"]
        parameters["properties"][prop_name] = prop
    return parameters


def build_tools_from_sample(sample: Mapping[str, Any], noise_tool: bool) -> list[dict[str, Any]]:
    if noise_tool:
        raise ValueError("--noise-tool is not supported in this self-contained minimal runner")

    tools = []
    for tool_info in parse_tool_doc(sample.get("tool_doc")):
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": tool_info.get("name"),
                    "description": tool_info.get("description", ""),
                    "parameters": convert_input_schema_to_parameters(tool_info.get("input_schema", {})),
                },
            }
        )
    return tools


def build_prompt_call(sample: Mapping[str, Any]) -> str:
    return str(sample["task"]).strip()


def parse_target_tools(sample: Mapping[str, Any]) -> list[str]:
    raw_target = sample.get("target_tool", [])
    if isinstance(raw_target, str):
        return [item.strip().split("::")[-1] for item in raw_target.split(",") if item.strip()]
    if isinstance(raw_target, list):
        return [str(item).strip().split("::")[-1] for item in raw_target if item]
    return []


def normalize_tool_response(sample: Mapping[str, Any]) -> dict[str, Any]:
    raw_response = sample.get("tool_response")
    if isinstance(raw_response, dict):
        return raw_response
    target_tools = parse_target_tools(sample)
    if isinstance(raw_response, list):
        return {
            tool_name: raw_response[index]
            for index, tool_name in enumerate(target_tools)
            if index < len(raw_response)
        }
    if raw_response is not None and len(target_tools) == 1:
        return {target_tools[0]: raw_response}
    return {}


def stringify_tool_result(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def execute_tool_call(tool_call: Mapping[str, Any], sample: Mapping[str, Any]) -> str:
    tool_name = str(tool_call.get("name", ""))
    response_table = normalize_tool_response(sample)
    if tool_name in response_table:
        return stringify_tool_result(response_table[tool_name])
    for key, value in response_table.items():
        if str(key).split("::")[-1] == tool_name:
            return stringify_tool_result(value)
    return ""


def serialize_tool_calls(tool_calls: Any) -> list[dict[str, Any]]:
    serialized = []
    for tool_call in tool_calls or []:
        function = getattr(tool_call, "function", None)
        if function is None:
            continue
        raw_arguments = getattr(function, "arguments", "{}")
        if isinstance(raw_arguments, str):
            try:
                arguments = json.loads(raw_arguments) if raw_arguments.strip() else {}
            except json.JSONDecodeError:
                arguments = {"raw_arguments": raw_arguments}
            arguments_text = raw_arguments
        else:
            arguments = raw_arguments if isinstance(raw_arguments, dict) else {}
            arguments_text = json.dumps(arguments, ensure_ascii=False)
        serialized.append(
            {
                "id": getattr(tool_call, "id", f"call_{len(serialized)}"),
                "name": getattr(function, "name", "unknown"),
                "arguments": arguments,
                "arguments_text": arguments_text,
            }
        )
    return serialized


def assistant_tool_calls_for_messages(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "id": tool_call["id"],
            "type": "function",
            "function": {
                "name": tool_call["name"],
                "arguments": tool_call["arguments_text"],
            },
        }
        for tool_call in tool_calls
    ]


def max_rounds_for(sample: Mapping[str, Any]) -> int:
    return 10


def call_llm_api(
    client: OpenAI,
    *,
    prompt: str,
    model: str,
    tools: list[dict[str, Any]],
    sample: Mapping[str, Any],
) -> dict[str, Any]:
    max_retries = 5
    base_sleep = 1.0
    max_rounds = max_rounds_for(sample)
    all_tool_calls: list[dict[str, Any]] = []
    tool_calls_by_round: list[list[dict[str, Any]]] = []
    conversation_history: list[dict[str, Any]] = []

    if tools:
        conversation_history.append(
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. If you need to use any tools, "
                    "invoke them with the correct parameters according to their documentation."
                ),
            }
        )
    conversation_history.append({"role": "user", "content": prompt})

    final_content = None
    round_num = 0
    conversation_finished = False

    while round_num < max_rounds and not conversation_finished:
        round_num += 1
        api_success = False
        for attempt in range(1, max_retries + 1):
            try:
                api_params: dict[str, Any] = {
                    "model": model,
                    "temperature": 0.0,
                    "max_tokens": 16384,
                    "messages": conversation_history,
                }
                if tools:
                    api_params["tools"] = tools
                    api_params["tool_choice"] = "auto"

                response = client.chat.completions.create(**api_params)
                message = response.choices[0].message
                serialized_tools = serialize_tool_calls(getattr(message, "tool_calls", None))

                assistant_message = {"role": "assistant", "content": message.content}
                if serialized_tools:
                    assistant_message["tool_calls"] = assistant_tool_calls_for_messages(serialized_tools)
                conversation_history.append(assistant_message)

                if serialized_tools:
                    all_tool_calls.extend(serialized_tools)
                    tool_calls_by_round.append(serialized_tools)
                    for tool_call in serialized_tools:
                        conversation_history.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call["id"],
                                "content": execute_tool_call(tool_call, sample),
                            }
                        )
                    api_success = True
                    break

                final_content = message.content
                conversation_finished = True
                api_success = True
                break
            except Exception as exc:
                error_msg = str(exc)
                print(f"API Error (Round {round_num}, Attempt {attempt}): {error_msg}")
                non_retryable = ("invalid", "auth", "api_key", "permission", "quota")
                if any(keyword in error_msg.lower() for keyword in non_retryable):
                    return {
                        "raw_content": final_content,
                        "error": error_msg,
                        "rounds": round_num,
                        "tool_calls": all_tool_calls,
                        "tool_calls_by_round": tool_calls_by_round,
                    }
                if attempt < max_retries:
                    time.sleep(base_sleep * (2 ** (attempt - 1)))
                else:
                    return {
                        "raw_content": final_content,
                        "error": f"Max retries exceeded: {error_msg}",
                        "rounds": round_num,
                        "tool_calls": all_tool_calls,
                        "tool_calls_by_round": tool_calls_by_round,
                    }
        if not api_success:
            break

    return {
        "raw_content": final_content,
        "rounds": round_num,
        "tool_calls": all_tool_calls,
        "tool_calls_by_round": tool_calls_by_round,
    }


def run_one(
    client: OpenAI,
    position: int,
    sample: Mapping[str, Any],
    model: str,
    noise_tool: bool,
) -> dict[str, Any]:
    idx = sample_idx(sample, position)
    try:
        prompt = build_prompt_call(sample)
        tools = build_tools_from_sample(sample, noise_tool)
        response = call_llm_api(client, prompt=prompt, model=model, tools=tools, sample=sample)
        row = {"sample_idx": idx, "model": model, "answer": response.get("raw_content")}
        if response.get("error"):
            row["error"] = response["error"]
        return row
    except Exception as exc:
        return {"sample_idx": idx, "model": model, "answer": None, "error": str(exc)}


def run_dataset(
    *,
    data_path: Path,
    input_root: Path,
    output_root: Path,
    base_url: str,
    api_key: str,
    model: str,
    noise_tool: bool,
    max_workers: int,
) -> None:
    samples = read_jsonl(data_path)
    client = OpenAI(base_url=base_url, api_key=api_key)
    out_path = output_path(output_root, input_root, data_path, model)
    if out_path.exists():
        out_path.unlink()

    if max_workers <= 1:
        rows: Iterable[dict[str, Any]] = [
            run_one(client, position, sample, model, noise_tool)
            for position, sample in enumerate(samples)
        ]
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(run_one, client, position, sample, model, noise_tool)
                for position, sample in enumerate(samples)
            ]
            collected = [future.result() for future in as_completed(futures)]
        collected.sort(key=lambda r: r.get("sample_idx", 0))
        rows = collected

    for row in rows:
        row["dataset_file"] = str(data_path)
        append_jsonl(out_path, row)
    print(f"[done] {data_path} -> {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal function_call runner that records only answer.")
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL"))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", choices=("thinking", "instruct"), default="thinking")
    parser.add_argument("--input", type=Path, default=DATASET_DIR)
    parser.add_argument("--output", type=Path, default=DATASET_DIR / "function_call_answer")
    parser.add_argument("--noise-tool", action="store_true")
    parser.add_argument("--max-workers", type=int, default=1)
    args = parser.parse_args()
    if not args.base_url:
        parser.error("--base-url or OPENAI_BASE_URL is required")
    if not args.api_key:
        parser.error("--api-key or OPENAI_API_KEY is required")
    if args.max_workers < 1:
        parser.error("--max-workers must be >= 1")
    return args


def main() -> None:
    args = parse_args()
    for input_root, data_path in discover_inputs(args.input, args.output):
        run_dataset(
            data_path=data_path,
            input_root=input_root,
            output_root=args.output,
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            noise_tool=args.noise_tool,
            max_workers=args.max_workers,
        )


if __name__ == "__main__":
    main()
