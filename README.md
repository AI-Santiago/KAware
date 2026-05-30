# Dataset Preview

This directory contains a minimal preview version.


## Contents

Each `.jsonl` file currently keeps one randomly sampled example from its corresponding bucket. This minimal version is intended for quick inspection of the data format, task structure, and tool-use annotations.

The files are organized by tool/function source and interaction pattern:

- `External_Function/`: samples involving external tools or APIs.
- `Internal_Function/`: samples involving internal computational or utility functions.
- `Hybrid_Function/`: samples combining multiple function sources.
- `single-hop.jsonl`: single-step tool-use tasks.
- `parallel.jsonl`: tasks where multiple tools can be used independently.
- `multi-hop.jsonl`: tasks requiring dependent or sequential tool use.


## Scripts

Two minimal evaluation runners are provided:

- `run_prompt_call_minimal.py` (**prompt_call**): injects each tool's schema into the user prompt and asks the model to decide, via a JSON answer, which tools are needed for the task. The model never actually calls a tool.
- `run_function_call_minimal.py` (**function_call**): registers the tools through the OpenAI tool-use API and lets the model actually invoke them in a multi-round conversation, with tool responses replayed from the dataset.

Run with:

```bash
python run_prompt_call_minimal.py  --model <model_name>
python run_function_call_minimal.py --model <model_name>
```

`--base-url` and `--api-key` can be passed on the CLI or via the `OPENAI_BASE_URL` / `OPENAI_API_KEY` environment variables. Outputs are written to `prompt_call_answer/` and `function_call_answer/` respectively.

