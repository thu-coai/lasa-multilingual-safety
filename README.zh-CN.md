# LASA：面向 LLM 安全的语义瓶颈层语言无关语义对齐

[English](README.md) | [论文](https://arxiv.org/abs/2604.12710)

这是论文 **LASA: Language-Agnostic Semantic Alignment at the Semantic Bottleneck for LLM Safety** 的官方开源代码。

LASA 的核心目标是把安全对齐从语言表层空间移动到模型的语言无关语义空间。论文发现，LLM 的中间层存在一个 **Semantic Bottleneck（语义瓶颈层）**，在该层模型表征主要按语义而不是语言聚类。LASA 在这一层训练轻量级 **Safety Semantic Interpreter (SSI)**，再用 SSI 给出的语义安全信号条件化生成，从而提升多语言安全泛化。

## 代码功能

- 使用逐层 silhouette score 和 t-SNE 发现语义瓶颈层。
- 在语义瓶颈层 hidden states 上训练轻量 SSI 分类器。
- 保留 KTO 风格的 semantic-conditioned alignment 训练脚本。
- 支持 vLLM 推理，并可由 SSI 触发条件生成提示。
- 提供 bash 运行脚本和可选的通用 SLURM 提交 wrapper。

## 目录结构

```text
generation/          vLLM 推理；支持 LASA/SSI 条件提示注入
models/              SSI 加载器和 vLLM chat wrapper
analysis/            语义瓶颈、silhouette、t-SNE、SSI 训练和分类器评估代码
training/            semantic-conditioned Latent KTO 入口和自定义 trainer
scripts/             bash 运行脚本、可选 SLURM wrapper、env.example
data/                随仓库提供的 SSI、Latent KTO、LASA safety 和 UltraFeedback 数据
examples/            小型 JSON prompt 示例
```

## 方法对应关系

LASA 包含三个阶段：

1. **Semantic Bottleneck Identification**
   对每层 hidden states 分别按语言标签和 query ID 计算 silhouette score。当 query/语义聚类相对语言聚类最强时，该层即为语义瓶颈层。分析脚本也会输出 t-SNE 可视化。

2. **Safety Semantic Interpreter**
   冻结基础模型，在语义瓶颈层抽取 hidden states，训练轻量 MLP 判断语义层面的 benign/harmful。保存后的 SSI 目录包含 `checkpoint.pt` 和 `config.json`。

3. **Semantic-Conditioned Alignment**
   训练模型把 SSI 语义安全信号和拒答/遵循行为关联起来。本仓库保留 KTO 风格训练实现。

推理时，如果 SSI 判断语义有害，会使用论文中的条件生成提示：

```text
Harmful query detected. I should refuse this request and provide a safe response in the user's language.
```

## 环境安装

```bash
git clone <this-repository-url>
cd lasa-multilingual-safety
pip install -r requirements.txt
```

可选环境配置文件：

```bash
cp scripts/env.example scripts/env.local
# 编辑 scripts/env.local，填入你的模型和可选集群参数。
source scripts/env.local
```

所有脚本都基于 `git clone` 后的仓库根目录解析相对路径。本 release 使用的 safety、UltraFeedback、SSI 训练数据和 Latent KTO 数据已经放在 `data/` 下；模型 checkpoint 通过环境变量传入，例如 `MODEL_PATH`。

如果你想使用本地 TRL 源码而不是 pip 安装的 `trl`：

```bash
export TRL_SOURCE_DIR=/path/to/trl
```

## Checkpoints

发布的 LASA 模型和 SSI 分类器可从 Hugging Face 下载：

```text
https://huggingface.co/yangjunxiao2021/LASA_Models
```

SSI 分类器目录格式：

```text
ssi_classifier/
  checkpoint.pt
  config.json        # 包含 layer_idx、threshold、hidden_dim
```

## 随仓库数据

本仓库已经包含语义瓶颈分析、SSI 训练和 Latent KTO 训练所需的数据：

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

- `ssi/unsafe_train.json`：11,000 条 harmful SSI 训练样本，由多语言 translated safety 数据和 HarmBench training 数据合并得到。
- `ssi/safety_train_translated.json`：用于构建 `unsafe_train.json` 的 translated safety 源数据。
- `ssi/harmbench_training.json`：用于构建 `unsafe_train.json` 的 HarmBench training 源数据。
- `training/mixed_dpo_ultra_safety_converted.json`：随仓库提供的 Latent KTO 训练数据。
- `safety_train_translated.json`：为语义瓶颈分析兼容保留的多语言 translated safety 数据。
- `ultrafeedback/monolingual_first1000_<lang>.json`：每种语言 1,000 条 benign UltraFeedback prompts。
- 语言：`en`、`zh`、`ko`、`th`、`sw`、`bn`、`ar`、`it`、`jw`、`vi`。

运行脚本默认使用这些仓库内路径。Stage 1 使用 `data/safety_train_translated.json` 和 `data/ultrafeedback/`；Stage 2 使用 `data/ssi/unsafe_train.json`；Stage 3 使用 `data/training/mixed_dpo_ultra_safety_converted.json`。只有当你想使用另一份数据时，才需要覆盖对应环境变量。

## Stage 1：识别语义瓶颈层

```bash
MODEL_PATH=/path/to/base_model \
LANGUAGES="en zh ko th sw bn ar it jw vi" \
LAYER_INDICES="0 4 8 12 16 20 24 28 32" \
OUTPUT_DIR=outputs/layer_clustering \
bash scripts/run_layer_analysis.sh
```

可选 SLURM 提交：

```bash
PARTITION=<partition> GPUS=1 CPUS=4 bash scripts/submit_layer_analysis.sh
```

如果你的集群不需要 partition，可以不设置 `PARTITION`。通用 wrapper 默认使用 `--gres=gpu:${GPUS}` 申请 GPU；如果集群语法不同，可用 `GPU_REQUEST` 覆盖。

论文中的瓶颈层示例：

| 模型 | 瓶颈层 | 相对深度 |
| --- | ---: | ---: |
| Llama-3.1-8B-Instruct | 14 / 32 | 43.8% |
| Qwen2.5-7B-Instruct | 19 / 28 | 67.9% |
| Qwen2.5-14B-Instruct | 29 / 48 | 60.4% |
| Qwen2.5-32B-Instruct | 29 / 64 | 45.3% |
| Qwen3-8B | 21 / 36 | 58.3% |
| Qwen3-14B | 25 / 40 | 62.5% |
| Qwen3-32B | 42 / 64 | 65.6% |

## Stage 2：训练 SSI

该阶段默认使用 benign UltraFeedback prompts 和 `data/ssi/unsafe_train.json` 中合并后的 harmful SSI 数据。`MAX_UNSAFE=0` 表示使用全部 harmful 样本，也是脚本默认值。

```bash
MODEL_PATH=/path/to/base_model \
LAYER_IDX=14 \
LANGUAGES="en zh ko" \
OUTPUT_DIR=outputs/ssi_llama31_8b_layer14 \
bash scripts/run_train_safety_classifier.sh
```

可选 SLURM 提交：

```bash
PARTITION=<partition> GPUS=1 CPUS=4 bash scripts/submit_train_safety_classifier.sh
```

论文训练语言为 English、Chinese、Korean；测试覆盖十种语言：`en`、`zh`、`ko`、`th`、`sw`、`bn`、`ar`、`it`、Javanese、`vi`；本代码中的 Javanese 文件名使用 `jw`。

## Stage 3：Semantic-Conditioned KTO

默认 Latent KTO 数据已经放在 `data/training/mixed_dpo_ultra_safety_converted.json`，使用 paired preference 格式：

- `prompt`：输入 prompt 或 chat-format messages
- `chosen`：desirable response
- `rejected`：undesirable response
- `source`：通常为 `safety` 或 `ultrafeedback`

Trainer 也支持已经拆成 unpaired KTO 的 `prompt`、`completion`、`label`、`source` 格式。如果要换成其他数据集，可以设置 `DATASET_PATH`。

```bash
MODEL_PATH=/path/to/base_model \
OUTPUT_DIR=outputs/latent_kto/run1 \
SAFETY_UNSAFE_RATIO=0.5 \
ULTRAFEEDBACK_UNSAFE_RATIO=0 \
bash scripts/run_latent_kto.sh
```

可选 SLURM 提交：

```bash
PARTITION=<partition> GPUS=4 CPUS=8 bash scripts/submit_latent_kto.sh
```

## 推理

输入格式：

```json
[
  {"prompt": "Your question here"}
]
```

普通生成：

```bash
MODEL_PATH=/path/to/aligned_model \
INPUT_FILE=/path/to/input.json \
OUTPUT_FILE=results/output.json \
bash scripts/run_generate.sh
```

启用 SSI 的 LASA 推理：

```bash
MODEL_PATH=/path/to/aligned_model \
CLASSIFIER_DIR=/path/to/ssi_classifier \
INPUT_FILE=/path/to/input.json \
OUTPUT_FILE=results/output.json \
bash scripts/run_generate.sh
```

输出 JSON 包含 `answer`；启用 SSI 时还会包含 `safety_prefix_used` 和 `safety_score`。

## Smoke Test

本地检查主要入口：

```bash
bash scripts/slurm_smoke_test.sh
```

通过 SLURM 检查：

```bash
PARTITION=<partition> bash scripts/submit_smoke_test.sh
```

日志会写到 `logs/smoke-<jobid>.log` 和 `logs/smoke-<jobid>.error`。

## 引用

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

## 使用边界

本仓库用于防御性多语言安全研究。安全评测数据包含 harmful prompts；请谨慎处理数据和生成结果，不要将本代码用于优化或传播 jailbreak 攻击。
