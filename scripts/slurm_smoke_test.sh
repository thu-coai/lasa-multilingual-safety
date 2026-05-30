#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"
GENERAL_PYTHON_BIN="${GENERAL_PYTHON_BIN:-${PYTHON_BIN:-python3}}"
KTO_PYTHON_BIN="${KTO_PYTHON_BIN:-${PYTHON_BIN:-python3}}"

echo "repo: ${REPO_ROOT}"
echo "general python: ${GENERAL_PYTHON_BIN}"
"${GENERAL_PYTHON_BIN}" --version
echo "kto python: ${KTO_PYTHON_BIN}"
"${KTO_PYTHON_BIN}" --version

echo "[1/5] generation help"
"${GENERAL_PYTHON_BIN}" "${REPO_ROOT}/generation/generate.py" --help >/tmp/lasa_generate_help.txt

echo "[2/5] safety classifier help"
"${GENERAL_PYTHON_BIN}" "${REPO_ROOT}/analysis/train_safety_classifier.py" --help >/tmp/lasa_classifier_help.txt

echo "[3/5] layer analysis help"
"${GENERAL_PYTHON_BIN}" "${REPO_ROOT}/analysis/analyze_layer_clustering.py" --help >/tmp/lasa_layer_help.txt

echo "[4/5] safety clustering analysis help"
"${GENERAL_PYTHON_BIN}" "${REPO_ROOT}/analysis/analyze_safety_clustering.py" --help >/tmp/lasa_safety_help.txt

echo "[5/5] latent kto help"
"${KTO_PYTHON_BIN}" "${REPO_ROOT}/training/latent_kto.py" --help >/tmp/lasa_latent_kto_help.txt

echo "All smoke checks passed."
