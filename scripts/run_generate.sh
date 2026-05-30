#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"
PYTHON_BIN="${PYTHON_BIN:-python3}"

MODEL_PATH="${MODEL_PATH:-}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${MODEL_PATH}}"
CLASSIFIER_DIR="${CLASSIFIER_DIR:-}"
INPUT_FILE="${INPUT_FILE:-${REPO_ROOT}/examples/multijail_1k.json}"
OUTPUT_FILE="${OUTPUT_FILE:-${REPO_ROOT}/results/generation/multijail_1k.json}"

MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-0.8}"
FREQUENCY_PENALTY="${FREQUENCY_PENALTY:-0.0}"
GPU_MEMORY_UTIL="${GPU_MEMORY_UTIL:-0.9}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
LATENT_BATCH_SIZE="${LATENT_BATCH_SIZE:-1}"
LIMIT="${LIMIT:-0}"

if [[ -z "${MODEL_PATH}" ]]; then
  echo "MODEL_PATH is required. Example: MODEL_PATH=/path/to/model bash scripts/run_generate.sh" >&2
  exit 1
fi

mkdir -p "$(dirname "${OUTPUT_FILE}")"

args=(
  --base_model "${MODEL_PATH}"
  --tokenizer_path "${TOKENIZER_PATH}"
  --input_file "${INPUT_FILE}"
  --output_file "${OUTPUT_FILE}"
  --max_new_tokens "${MAX_NEW_TOKENS}"
  --temperature "${TEMPERATURE}"
  --top_p "${TOP_P}"
  --frequency_penalty "${FREQUENCY_PENALTY}"
  --gpu_memory_utilization "${GPU_MEMORY_UTIL}"
  --tensor_parallel_size "${TENSOR_PARALLEL_SIZE}"
)

if [[ "${LIMIT}" != "0" ]]; then
  args+=(--limit "${LIMIT}")
fi

if [[ -n "${CLASSIFIER_DIR}" ]]; then
  args+=(--use_safety_prefix --classifier_dir "${CLASSIFIER_DIR}" --latent_batch_size "${LATENT_BATCH_SIZE}")
fi

echo "Running generation"
echo "  model: ${MODEL_PATH}"
echo "  input: ${INPUT_FILE}"
echo "  output: ${OUTPUT_FILE}"
echo "  classifier: ${CLASSIFIER_DIR:-none}"

"${PYTHON_BIN}" "${REPO_ROOT}/generation/generate.py" "${args[@]}"
