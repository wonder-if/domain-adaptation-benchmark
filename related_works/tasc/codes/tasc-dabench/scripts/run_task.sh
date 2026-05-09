#!/bin/bash

set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "usage: $0 <gpu_id> <config_relpath> [extra opts...]"
  exit 1
fi

GPU_ID="$1"
CFG_REL="$2"
shift 2

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/data/wyh/envs/miniforge3/envs/vlms-research/bin/python}"
CFG_PATH="${PROJECT_DIR}/${CFG_REL}"

if [ ! -f "${CFG_PATH}" ]; then
  echo "config not found: ${CFG_PATH}"
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

TASK_DESC="$("${PYTHON_BIN}" - <<PY
import yaml
from pathlib import Path

cfg = yaml.safe_load(Path("${CFG_PATH}").read_text(encoding="utf-8"))
dataset = cfg["dataset"]["name"]
task = cfg["dataset"]["task"]
shared = cfg["dataset"]["shared"]
source_private = cfg["dataset"]["source_private"]
target_private = cfg["dataset"]["target_private"]
print(f"{dataset} {task} {shared}/{source_private}/{target_private}")
PY
)"

echo "[start] gpu=${GPU_ID} cfg=${CFG_REL} task=${TASK_DESC}"
"${PYTHON_BIN}" train_sapphire.py --cfg "${CFG_REL}" --gpu "${GPU_ID}" "$@"
echo "[done] gpu=${GPU_ID} cfg=${CFG_REL}"
