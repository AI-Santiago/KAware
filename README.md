# From Knowing to Acting: Benchmarking Self-Awareness Capability of LLM Agents

KAware targets **self-awareness capability**: an agent's ability to distinguish tasks solvable by internal parametric knowledge from tasks that genuinely require external tools, and to translate that judgment into appropriate tool-use behavior. The benchmark accompanies the paper **"From Knowing to Acting: Benchmarking Self-Awareness Capability of LLM Agents"**.

This directory contains a minimal preview version.

![KAware](./figure/figure.pdf)

## 🔍 Overview

The paper introduces **KAPRO** (*Knowing-Acting Quadrant Probe*), an evaluation protocol that separates two complementary dimensions:

- **Knowing**: the model's explicit metacognitive judgment of which tools are required.
- **Acting**: the model's spontaneous tool-use behavior in a standard agent environment.

KAware organizes tasks into three capability subspaces:

- **External Function**: tasks that strictly require external tools or APIs.
- **Internal Function**: tasks solvable by internal model capability, where tool calls should be withheld.
- **Hybrid Composition**: tasks combining internal reasoning with tool-dependent subtasks.

The full benchmark described in the paper contains **1,076 tasks** across capability boundaries and reasoning topologies: single-hop, parallel, and multi-hop. This repository currently provides a **minimal preview** with one sampled example per released bucket for inspecting task format, tool annotations, and runner behavior.

## 📊 Evaluation

KAPRO reports three main quantities:

- `Acc_know`: Jaccard alignment between the tools predicted as necessary and the ground-truth required tools.
- `Acc_act`: Jaccard alignment between the tools actually invoked and the ground-truth required tools.
- `KAS`: the harmonic mean of `Acc_know` and `Acc_act`, rewarding balanced knowing-acting consistency.

The core finding is that high pass rate can mask poor self-awareness. In internal-capability settings, models may answer correctly while still overusing tools, revealing a boundary-calibration failure that pass rate alone does not capture.

## 📁 Repository Structure

```text
KAware/
|-- External_Function/        # Tool-required tasks
|-- Hybrid_Function/          # Mixed internal and tool-dependent tasks
|-- Internal_Function/        # Internally solvable tasks
|-- figure/
|   `-- figure.pdf           # Main paper figure
|-- run_prompt_call_minimal.py
|-- run_function_call_minimal.py
```

Each `.jsonl` file corresponds to a capability boundary and reasoning topology:

- `single-hop.jsonl`: single-step tool-use tasks.
- `parallel.jsonl`: independent multi-tool subtasks.
- `multi-hop.jsonl`: sequential tasks with dependency between subtasks.

## 🧩 Data Format

Each sample includes the natural-language task, boundary label, tool documentation, target tool set, tool-call parameters, tool responses, and reference answer where available. Hybrid and multi-tool examples additionally include subquestion annotations and dependency metadata.

Common fields include:

- `task`: user-facing query.
- `boundary`: capability boundary label.
- `hop`: reasoning topology.
- `target_tool`: ground-truth required tool or tools.
- `tool_doc`: executable tool schema.
- `tool_response`: replayable response used by the minimal function-call runner.

## 🚀 Scripts

Two runners are included for quick inspection:

- `run_prompt_call_minimal.py`: evaluates **Knowing** by asking the model to mark each tool as necessary or unnecessary without executing tools.
- `run_function_call_minimal.py`: evaluates **Acting** by registering tools through the function-calling API and replaying dataset tool responses during multi-round interaction.

Run with:

```bash
python run_prompt_call_minimal.py --model <model_name>
python run_function_call_minimal.py --model <model_name>
```

`--base-url` and `--api-key` can be passed directly or supplied through `OPENAI_BASE_URL` and `OPENAI_API_KEY`. Outputs are written to `prompt_call_answer/` and `function_call_answer/`.

## 🌟 Citation

If you find this repository useful in your research, please consider
citing our paper:

```bibtex
@misc{li2026knowingactingbenchmarkingselfawareness,
      title={From Knowing to Acting: Benchmarking Self-Awareness Capability of LLM Agents}, 
      author={Yifan Li and Shengbin Yue and Boyu Feng and Jinhu Qi and Bo Ke and Zixing Song and Hongru Wang and Zhongyu Wei and Irwin King},
      year={2026},
      eprint={2606.20661},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2606.20661}, 
}
```


