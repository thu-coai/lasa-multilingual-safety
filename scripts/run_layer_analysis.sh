#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"
PYTHON_BIN="${PYTHON_BIN:-python3}"

MODEL_PATH="${MODEL_PATH:-}"
ULTRAFEEDBACK_DIR="${ULTRAFEEDBACK_DIR:-${REPO_ROOT}/data/ultrafeedback}"
SAFETY_DATA_PATH="${SAFETY_DATA_PATH:-${REPO_ROOT}/data/safety_train_translated.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/layer_clustering}"
DATA_TYPE="${DATA_TYPE:-both}"
LANGUAGES="${LANGUAGES:-en zh ar sw}"
LAYER_INDICES="${LAYER_INDICES:-0 8 16 24 32}"
MAX_SAMPLES_PER_LANG="${MAX_SAMPLES_PER_LANG:-100}"
BATCH_SIZE="${BATCH_SIZE:-8}"
SKIP_VISUALIZATION="${SKIP_VISUALIZATION:-0}"

if [[ -z "${MODEL_PATH}" ]]; then
  echo "MODEL_PATH is required." >&2
  exit 1
fi

if [[ "${DATA_TYPE}" == "ultrafeedback" || "${DATA_TYPE}" == "both" ]] && [[ ! -d "${ULTRAFEEDBACK_DIR}" ]]; then
  echo "ULTRAFEEDBACK_DIR not found when DATA_TYPE=${DATA_TYPE}: ${ULTRAFEEDBACK_DIR}" >&2
  exit 1
fi

if [[ "${DATA_TYPE}" == "safety" || "${DATA_TYPE}" == "both" ]] && [[ ! -f "${SAFETY_DATA_PATH}" ]]; then
  echo "SAFETY_DATA_PATH not found when DATA_TYPE=${DATA_TYPE}: ${SAFETY_DATA_PATH}" >&2
  exit 1
fi

args=(
  --model_name_or_path "${MODEL_PATH}"
  --data_type "${DATA_TYPE}"
  --ultrafeedback_dir "${ULTRAFEEDBACK_DIR}"
  --safety_data_path "${SAFETY_DATA_PATH}"
  --languages ${LANGUAGES}
  --layer_indices ${LAYER_INDICES}
  --max_samples_per_lang "${MAX_SAMPLES_PER_LANG}"
  --batch_size "${BATCH_SIZE}"
  --output_dir "${OUTPUT_DIR}"
)

if [[ "${SKIP_VISUALIZATION}" == "1" ]]; then
  args+=(--skip_visualization)
fi

echo "Running silhouette/t-SNE layer analysis"
echo "  model: ${MODEL_PATH}"
echo "  output: ${OUTPUT_DIR}"

"${PYTHON_BIN}" "${REPO_ROOT}/analysis/analyze_layer_clustering.py" "${args[@]}"
