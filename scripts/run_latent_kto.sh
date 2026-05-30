#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"
ACCELERATE_BIN="${ACCELERATE_BIN:-accelerate}"

DATASET_PATH="${DATASET_PATH:-${REPO_ROOT}/data/training/mixed_dpo_ultra_safety_converted.json}"
MODEL_PATH="${MODEL_PATH:-}"
REF_MODEL_PATH="${REF_MODEL_PATH:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
TRL_SOURCE_DIR="${TRL_SOURCE_DIR:-}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-${REPO_ROOT}/scripts/deepspeed_zero3.json}"

NUM_PROCESSES="${NUM_PROCESSES:-4}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29522}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-3}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
SAFETY_UNSAFE_RATIO="${SAFETY_UNSAFE_RATIO:-0.5}"
ULTRAFEEDBACK_UNSAFE_RATIO="${ULTRAFEEDBACK_UNSAFE_RATIO:-0}"
BETA="${BETA:-0.1}"
MAX_LENGTH="${MAX_LENGTH:-512}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-384}"
REPORT_TO="${REPORT_TO:-none}"

if [[ -z "${MODEL_PATH}" ]]; then
  echo "MODEL_PATH is required. Example: MODEL_PATH=/path/to/model bash scripts/run_latent_kto.sh" >&2
  exit 1
fi

if [[ ! -f "${DATASET_PATH}" ]]; then
  echo "DATASET_PATH not found: ${DATASET_PATH}" >&2
  exit 1
fi

if [[ -z "${OUTPUT_DIR}" ]]; then
  OUTPUT_DIR="${REPO_ROOT}/outputs/latent_kto/$(basename "${MODEL_PATH}")"
fi

if [[ -n "${TRL_SOURCE_DIR}" ]]; then
  export TRL_SOURCE_DIR
fi
export CUDA_VISIBLE_DEVICES

mkdir -p "${OUTPUT_DIR}" "${REPO_ROOT}/logs"

args=(
  "${REPO_ROOT}/training/latent_kto.py"
  --dataset_path "${DATASET_PATH}"
  --model_path "${MODEL_PATH}"
  --output_dir "${OUTPUT_DIR}"
  --overwrite_output_dir true
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}"
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
  --num_train_epochs "${NUM_TRAIN_EPOCHS}"
  --learning_rate "${LEARNING_RATE}"
  --lr_scheduler_type cosine
  --warmup_steps 100
  --logging_steps 10
  --bf16 true
  --gradient_checkpointing true
  --max_grad_norm 1.0
  --save_only_model true
  --save_strategy epoch
  --save_total_limit 2
  --seed 42
  --beta "${BETA}"
  --max_length "${MAX_LENGTH}"
  --max_prompt_length "${MAX_PROMPT_LENGTH}"
  --remove_unused_columns false
  --safety_unsafe_ratio "${SAFETY_UNSAFE_RATIO}"
  --ultrafeedback_unsafe_ratio "${ULTRAFEEDBACK_UNSAFE_RATIO}"
  --report_to "${REPORT_TO}"
)

if [[ -n "${REF_MODEL_PATH}" ]]; then
  args+=(--ref_model_path "${REF_MODEL_PATH}")
fi

if [[ -f "${DEEPSPEED_CONFIG}" ]]; then
  args+=(--deepspeed "${DEEPSPEED_CONFIG}")
fi

echo "Running LASA semantic-conditioned KTO training"
echo "  dataset: ${DATASET_PATH}"
echo "  model: ${MODEL_PATH}"
echo "  output: ${OUTPUT_DIR}"
echo "  TRL_SOURCE_DIR: ${TRL_SOURCE_DIR}"

"${ACCELERATE_BIN}" launch --num_processes "${NUM_PROCESSES}" --main_process_port "${MAIN_PROCESS_PORT}" "${args[@]}"
