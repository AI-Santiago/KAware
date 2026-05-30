#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable, Mapping

from openai import OpenAI

DATASET_DIR = Path(__file__).resolve().parent
DATASET_SUBDIRS = ("External_Function", "Internal_Function", "Hybrid_Function")
GENERATED_DIR_NAMES = {"prompt_call_answer", "function_call_answer"}
SYSTEM_PROMPT = "You are a helpful assistant."


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


def convert_tool_to_function_block(tool_doc: Any) -> list[dict[str, Any]]:
    blocks = []
    for tool_info in parse_tool_doc(tool_doc):
        blocks.append(
            {
                "type": "function",
                "function": {
                    "name": tool_info.get("name", "unknown_tool"),
                    "description": tool_info.get("description", ""),
                    "parameters": tool_info.get("input_schema", {}),
                },
            }
        )
    return blocks


def build_prompt_call(sample: Mapping[str, Any], noise_tool: bool) -> str:
    if noise_tool:
        raise ValueError("--noise-tool is not supported in this self-contained minimal runner")

    task = sample["task"]
    final_tool_info = convert_tool_to_function_block(sample.get("tool_doc"))
    prompt = f"""
# Role
You are an agent responsible for evaluating if a user's task requires tools.

# Task
{task}

# Available Tools
{final_tool_info}

# Instructions
Assess each tool's necessity according to task specifications.
- Set a tool to `true` if its functionality is essential for task completion.
- Set a tool to `false` if internal capabilities are sufficient.

# Output Format
Return a valid JSON object ONLY in the following format:
{{
  "tool_decisions": {{
    "ToolA": true,
    "ToolB": false,
    ...
  }}
}}
"""
    return prompt.strip()


def call_llm_api(client: OpenAI, prompt: str, model: str) -> tuple[str | None, str | None]:
    max_retries = 3
    base_sleep = 1.0
    last_err: str | None = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=0.0,
                max_tokens=16384,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
            return response.choices[0].message.content, None
        except Exception as exc:
            err = str(exc)
            last_err = err
            print(f"[Error] API call failed (Attempt {attempt}/{max_retries}): {err}")
            if attempt >= max_retries:
                break
            if "429" in err or "Rate limit" in err:
                sleep_time = base_sleep * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
                print(f">>> Rate limited. Sleeping {sleep_time:.2f}s")
                time.sleep(sleep_time)
            else:
                time.sleep(1)
    return None, f"Max retries exceeded: {last_err}" if last_err else "Max retries exceeded"


def run_one(
    client: OpenAI,
    position: int,
    sample: Mapping[str, Any],
    model: str,
    noise_tool: bool,
) -> dict[str, Any]:
    idx = sample_idx(sample, position)
    try:
        prompt = build_prompt_call(sample, noise_tool)
        answer, err = call_llm_api(client, prompt, model)
        row: dict[str, Any] = {"sample_idx": idx, "model": model, "answer": answer}
        if err:
            row["error"] = err
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
    parser = argparse.ArgumentParser(description="Minimal prompt_call runner that records only answer.")
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL"))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", type=Path, default=DATASET_DIR)
    parser.add_argument("--output", type=Path, default=DATASET_DIR / "prompt_call_answer")
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
