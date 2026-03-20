# modified_codebase

这个目录用于放置整理后的可扩展代码。

## 结构说明

- `pow_sampling_mcmc/framework.py`
  - 抽离了 `mcmc_power_samp` 主逻辑，仅保留 power-sampling MCMC 所需的核心能力。
  - 提供了统一的数据接口 `DatasetAdapter`。
  - 内置 `JSONListAdapter` 和 `HFDatasetAdapter`，用于接入本地 JSON 或 HuggingFace datasets。
  - 提供 `run_framework` 统一执行入口，方便未来增加新数据集和后处理函数。

- `pow_sampling_mcmc/run_sampling.py`
  - 通用 CLI。
  - 可通过参数切换数据来源（`json` / `hf`），并指定问题字段、答案字段。

## 最小使用示例

### 1) 使用本地 JSON

```bash
python modified_codebase/pow_sampling_mcmc/run_sampling.py \
  --model_name Qwen/Qwen2.5-Math-7B \
  --dataset_source json \
  --dataset_path llm_experiments/data/MATH500.json \
  --question_key prompt \
  --answer_key answer \
  --cot \
  --limit 5
```

### 2) 使用 HuggingFace 数据集

```bash
python modified_codebase/pow_sampling_mcmc/run_sampling.py \
  --model_name Qwen/Qwen2.5-Math-7B \
  --dataset_source hf \
  --dataset_name gsm8k \
  --dataset_subset main \
  --dataset_split test \
  --question_key question \
  --answer_key answer \
  --limit 5
```

## 你可以如何扩展

1. 新数据源：继承 `DatasetAdapter`，实现 `load()`。
2. 新 Prompt：把 `run_framework` 的 `prompt_builder` 参数换成自己的函数。
3. 新后处理：给 `postprocess` 传入解析函数（如提取 boxed answer）。

## Step 1: Data processing for instruction-answer pairs

- Script: `data_process/build_instruction_answer_dataset.py`
- Purpose: Convert raw mode-choice table data into English instruction-answer pairs.
- It resolves ORIGIN/DESTIN admin codes using a region JSON and supports optional strong-LLM polishing.

Example (JSON output):

```bash
python modified_codebase/data_process/build_instruction_answer_dataset.py \
  --input_csv path/to/raw_mode_choice.csv \
  --region_json path/to/regions.json \
  --output_path modified_codebase/data_process/processed/instruction_answer.json \
  --output_format json
```

Example with LLM polishing:

```bash
OPENAI_API_KEY=your_key \
python modified_codebase/data_process/build_instruction_answer_dataset.py \
  --input_csv path/to/raw_mode_choice.csv \
  --region_json path/to/regions.json \
  --output_path modified_codebase/data_process/processed/instruction_answer.csv \
  --output_format csv \
  --use_llm \
  --llm_model deepseek-chat \
  --api_base https://api.deepseek.com
```

## Step 2: External signal models (classification heads)

- Folder: `external_signal/`
- `model.py`
  - `BertTravelModeClassifier`: BERT + classification head (predict travel mode from instruction).
  - `FrozenLLMTravelModeClassifier`: frozen small LLM encoder (e.g., Qwen-1.5B) + trainable classification head.
- `train.py`
  - Trains either `bert` or `frozen_llm` on instruction-answer pairs.
  - Converts `answer` to categorical labels (`Auto`, `Riding`, `Subway`, `Bus`, `Subway&Bus`, `Taxi`, `Cycling`, `Walk`).
- `test.py`
  - Loads checkpoint and reports accuracy + per-class metrics.

Example training:

```bash
python modified_codebase/external_signal/train.py \
  --data_path modified_codebase/data_process/processed/instruction_answer.json \
  --model_type bert \
  --model_name bert-base-uncased
```

Example training with frozen small LLM:

```bash
python modified_codebase/external_signal/train.py \
  --data_path modified_codebase/data_process/processed/instruction_answer.json \
  --model_type frozen_llm \
  --model_name Qwen/Qwen2.5-1.5B-Instruct
```

Example test:

```bash
python modified_codebase/external_signal/test.py \
  --data_path path/to/test_instruction_answer.json \
  --checkpoint_path modified_codebase/external_signal/checkpoints/best_model.pt
```
