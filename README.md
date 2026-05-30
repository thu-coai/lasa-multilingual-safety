# LASA: Language-Agnostic Semantic Alignment at the Semantic Bottleneck for LLM Safety

[中文说明](README.zh-CN.md) | [Paper](https://arxiv.org/abs/2604.12710)

Official code for **LASA: Language-Agnostic Semantic Alignment at the Semantic Bottleneck for LLM Safety**.

LASA addresses multilingual safety failures by moving safety alignment from language-specific text space to a model's language-agnostic semantic space. The paper identifies an intermediate **semantic bottleneck** layer where representations cluster by shared meaning rather than language identity, trains a lightweight **Safety Semantic Interpreter (SSI)** on that layer, and uses the resulting semantic safety signal to condition response generation.

## Highlights

- Semantic bottleneck discovery with layer-wise silhouette scores and t-SNE visualization.
- Lightweight SSI classifier trained on hidden states from the bottleneck layer.
- Semantic-conditioned alignment with a KTO-style training script.
- vLLM inference with optional SSI-triggered conditional generation prompt.
- Bash run scripts and optional generic SLURM submission wrappers.

## Repository Layout

```text
generation/          vLLM generation with optional LASA/SSI conditional prompt injection
models/              SSI loader and vLLM chat wrapper
analysis/            bottleneck, silhouette, t-SNE, SSI training, and classifier evaluation tools
training/            semantic-conditioned Latent KTO entrypoint and custom trainer
scripts/             bash run scripts, optional SLURM wrappers, and env.example
data/                bundled SSI, Latent KTO, LASA safety, and UltraFeedback data
examples/            small JSON prompt example
```

## Method Map

LASA is organized into three stages:

1. **Semantic Bottleneck Identification**
   Compute silhouette scores over hidden states using two partitions: language labels and query IDs. The bottleneck is the layer where query/semantic clustering dominates language clustering. The analysis scripts also produce t-SNE plots.

2. **Safety Semantic Interpreter**
   Freeze the base model, extract hidden states at the bottleneck layer, and train a lightweight MLP to classify benign vs. harmful semantics. The saved SSI directory contains `checkpoint.pt` and `config.json`.

3. **Semantic-Conditioned Alignment**
   Train the model to associate the SSI safety signal with refusal/compliance behavior. This repo keeps the KTO-style implementation used for the released training path.

At inference time, harmful semantics detected by SSI are converted to the conditional generation prompt used in the paper:

```text
Harmful query detected. I should refuse this request and provide a safe response in the user's language.
```

## Setup

```bash
git clone <this-repository-url>
cd lasa-multilingual-safety
pip install -r requirements.txt
```

Optional environment file:

```bash
cp scripts/env.example scripts/env.local
# Edit scripts/env.local with your model and optional cluster paths.
source scripts/env.local
```

All scripts resolve paths relative to the cloned repository. The safety, UltraFeedback, SSI-training, and Latent KTO datasets used by the release are bundled under `data/`; model checkpoints are provided through environment variables such as `MODEL_PATH`.

If you want to use a local TRL source checkout instead of the installed `trl` package:

```bash
export TRL_SOURCE_DIR=/path/to/trl
```

## Checkpoints

Released LASA checkpoints and SSI classifiers are available from Hugging Face:

```text
https://huggingface.co/yangjunxiao2021/LASA_Models
```

Expected SSI classifier directory:

```text
ssi_classifier/
  checkpoint.pt
  config.json        # includes layer_idx, threshold, hidden_dim
```

## Bundled Data

The repository includes the data used by the bottleneck-analysis, SSI-training, and Latent KTO scripts:

```text
data/
  ssi/
    unsafe_train.json
    safety_train_translated.json
    harmbench_training.json
  training/
    mixed_dpo_ultra_safety_converted.json
  safety_train_translated.json
  ultrafeedback/
    monolingual_first1000_en.json
    monolingual_first1000_zh.json
    ...
```

- `ssi/unsafe_train.json`: 11,000 harmful SSI-training examples, built by merging the translated multilingual safety data with HarmBench training prompts.
- `ssi/safety_train_translated.json`: source translated safety data used to build `unsafe_train.json`.
- `ssi/harmbench_training.json`: source HarmBench training data used to build `unsafe_train.json`.
- `training/mixed_dpo_ultra_safety_converted.json`: bundled Latent KTO training data.
- `safety_train_translated.json`: multilingual translated safety data kept for bottleneck-analysis compatibility.
- `ultrafeedback/monolingual_first1000_<lang>.json`: 1,000 benign UltraFeedback prompts per language.
- Languages: `en`, `zh`, `ko`, `th`, `sw`, `bn`, `ar`, `it`, `jw`, `vi`.

The run scripts use these bundled paths by default. Stage 1 uses `data/safety_train_translated.json` and `data/ultrafeedback/`; Stage 2 uses `data/ssi/unsafe_train.json`; Stage 3 uses `data/training/mixed_dpo_ultra_safety_converted.json`. Override the corresponding environment variables only if you want to use a different dataset copy.

## Stage 1: Identify the Semantic Bottleneck

```bash
MODEL_PATH=/path/to/base_model \
LANGUAGES="en zh ko th sw bn ar it jw vi" \
LAYER_INDICES="0 4 8 12 16 20 24 28 32" \
OUTPUT_DIR=outputs/layer_clustering \
bash scripts/run_layer_analysis.sh
```

Optional SLURM submission:

```bash
PARTITION=<partition> GPUS=1 CPUS=4 bash scripts/submit_layer_analysis.sh
```

Leave `PARTITION` empty if your cluster does not require it. The generic wrappers default to `--gres=gpu:${GPUS}`; override with `GPU_REQUEST` if your cluster uses different syntax.

Paper bottleneck examples:

| Model | Bottleneck Layer | Relative Depth |
| --- | ---: | ---: |
| Llama-3.1-8B-Instruct | 14 / 32 | 43.8% |
| Qwen2.5-7B-Instruct | 19 / 28 | 67.9% |
| Qwen2.5-14B-Instruct | 29 / 48 | 60.4% |
| Qwen2.5-32B-Instruct | 29 / 64 | 45.3% |
| Qwen3-8B | 21 / 36 | 58.3% |
| Qwen3-14B | 25 / 40 | 62.5% |
| Qwen3-32B | 42 / 64 | 65.6% |

## Stage 2: Train SSI

By default this stage trains on benign UltraFeedback prompts and the merged harmful SSI dataset in `data/ssi/unsafe_train.json`. Set `MAX_UNSAFE=0` to use all bundled harmful examples, which is the script default.

```bash
MODEL_PATH=/path/to/base_model \
LAYER_IDX=14 \
LANGUAGES="en zh ko" \
OUTPUT_DIR=outputs/ssi_llama31_8b_layer14 \
bash scripts/run_train_safety_classifier.sh
```

Optional SLURM submission:

```bash
PARTITION=<partition> GPUS=1 CPUS=4 bash scripts/submit_train_safety_classifier.sh
```

Training languages in the paper are English, Chinese, and Korean. Evaluation is performed across ten languages: `en`, `zh`, `ko`, `th`, `sw`, `bn`, `ar`, `it`, Javanese, and `vi`; this codebase uses `jw` for Javanese filenames.

## Stage 3: Semantic-Conditioned KTO

The default Latent KTO data is bundled at `data/training/mixed_dpo_ultra_safety_converted.json`. It uses paired preference format:

- `prompt`: prompt text or chat-format messages
- `chosen`: desirable response
- `rejected`: undesirable response
- `source`: usually `safety` or `ultrafeedback`

The trainer also accepts already-unpaired KTO records with `prompt`, `completion`, `label`, and `source`. To use another dataset, set `DATASET_PATH`.

```bash
MODEL_PATH=/path/to/base_model \
OUTPUT_DIR=outputs/latent_kto/run1 \
SAFETY_UNSAFE_RATIO=0.5 \
ULTRAFEEDBACK_UNSAFE_RATIO=0 \
bash scripts/run_latent_kto.sh
```

Optional SLURM submission:

```bash
PARTITION=<partition> GPUS=4 CPUS=8 bash scripts/submit_latent_kto.sh
```

## Inference

Input format:

```json
[
  {"prompt": "Your question here"}
]
```

Normal generation:

```bash
MODEL_PATH=/path/to/aligned_model \
INPUT_FILE=/path/to/input.json \
OUTPUT_FILE=results/output.json \
bash scripts/run_generate.sh
```

LASA generation with SSI:

```bash
MODEL_PATH=/path/to/aligned_model \
CLASSIFIER_DIR=/path/to/ssi_classifier \
INPUT_FILE=/path/to/input.json \
OUTPUT_FILE=results/output.json \
bash scripts/run_generate.sh
```

The output JSON contains `answer`; when SSI is enabled it also contains `safety_prefix_used` and `safety_score`.

## Citation

```bibtex
@misc{yang2026lasa,
  title        = {LASA: Language-Agnostic Semantic Alignment at the Semantic Bottleneck for LLM Safety},
  author       = {Junxiao Yang and Haoran Liu and Jinzhe Tu and Jiale Cheng and Zhexin Zhang and Shiyao Cui and Jiaqi Weng and Jialing Tao and Hui Xue and Hongning Wang and Han Qiu and Minlie Huang},
  year         = {2026},
  eprint       = {2604.12710},
  archivePrefix = {arXiv},
  primaryClass = {cs.LG}
}
```

## Ethical Use

This repository is intended for defensive multilingual safety research. Safety benchmarks contain harmful prompts; handle datasets and generated outputs carefully, and do not use this code to optimize or distribute jailbreak attacks.
