#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"
PYTHON_BIN="${PYTHON_BIN:-python3}"

MODEL_PATH="${MODEL_PATH:-}"
ULTRAFEEDBACK_DIR="${ULTRAFEEDBACK_DIR:-${REPO_ROOT}/data/ultrafeedback}"
SAFETY_DATA_PATH="${SAFETY_DATA_PATH:-${REPO_ROOT}/data/ssi/unsafe_train.json}"
OOD_DATA_PATH="${OOD_DATA_PATH:-}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/safety_classifier}"

LAYER_IDX="${LAYER_IDX:-20}"
LANGUAGES="${LANGUAGES:-en zh ar sw}"
MAX_SAFE_PER_LANG="${MAX_SAFE_PER_LANG:-200}"
MAX_UNSAFE="${MAX_UNSAFE:-0}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-8}"
CLASSIFIER_HIDDEN_DIM="${CLASSIFIER_HIDDEN_DIM:-256}"

if [[ -z "${MODEL_PATH}" ]]; then
  echo "MODEL_PATH is required. Example: MODEL_PATH=/path/to/model bash scripts/run_train_safety_classifier.sh" >&2
  exit 1
fi

if [[ ! -d "${ULTRAFEEDBACK_DIR}" || ! -f "${SAFETY_DATA_PATH}" ]]; then
  echo "Expected data not found. Set ULTRAFEEDBACK_DIR and SAFETY_DATA_PATH or use the bundled data/ directory." >&2
  echo "  ULTRAFEEDBACK_DIR=${ULTRAFEEDBACK_DIR}" >&2
  echo "  SAFETY_DATA_PATH=${SAFETY_DATA_PATH}" >&2
  exit 1
fi

args=(
  --model_path "${MODEL_PATH}"
  --layer_idx "${LAYER_IDX}"
  --ultrafeedback_dir "${ULTRAFEEDBACK_DIR}"
  --safety_data_path "${SAFETY_DATA_PATH}"
  --languages ${LANGUAGES}
  --max_safe_per_lang "${MAX_SAFE_PER_LANG}"
  --max_unsafe "${MAX_UNSAFE}"
  --classifier_hidden_dim "${CLASSIFIER_HIDDEN_DIM}"
  --train_epochs "${TRAIN_EPOCHS}"
  --batch_size "${BATCH_SIZE}"
  --output_dir "${OUTPUT_DIR}"
)

if [[ -n "${OOD_DATA_PATH}" ]]; then
  args+=(--ood_data_path "${OOD_DATA_PATH}")
fi

echo "Training safety classifier"
echo "  model: ${MODEL_PATH}"
echo "  layer: ${LAYER_IDX}"
echo "  output: ${OUTPUT_DIR}"

"${PYTHON_BIN}" "${REPO_ROOT}/analysis/train_safety_classifier.py" "${args[@]}"
