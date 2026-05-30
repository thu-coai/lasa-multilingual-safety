#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"
PARTITION="${PARTITION:-}"
ACCOUNT="${ACCOUNT:-}"
TIME="${TIME:-}"
CPUS="${CPUS:-2}"
SBATCH_BIN="${SBATCH_BIN:-$(command -v sbatch || true)}"
SBATCH_ARGS="${SBATCH_ARGS:-}"
if [[ -z "${SBATCH_BIN}" ]]; then
  echo "sbatch not found. Set SBATCH_BIN=/path/to/sbatch." >&2
  exit 1
fi

mkdir -p "${REPO_ROOT}/logs"

sbatch_args=(-N 1 -c "${CPUS}" --output "${REPO_ROOT}/logs/smoke-%j.log" --error "${REPO_ROOT}/logs/smoke-%j.error")
if [[ -n "${PARTITION}" ]]; then
  sbatch_args+=(-p "${PARTITION}")
fi
if [[ -n "${ACCOUNT}" ]]; then
  sbatch_args+=(-A "${ACCOUNT}")
fi
if [[ -n "${TIME}" ]]; then
  sbatch_args+=(-t "${TIME}")
fi
if [[ -n "${SBATCH_ARGS}" ]]; then
  read -r -a extra_args <<< "${SBATCH_ARGS}"
  sbatch_args+=("${extra_args[@]}")
fi

"${SBATCH_BIN}" "${sbatch_args[@]}" --wrap "cd '${REPO_ROOT}' && bash scripts/slurm_smoke_test.sh"
