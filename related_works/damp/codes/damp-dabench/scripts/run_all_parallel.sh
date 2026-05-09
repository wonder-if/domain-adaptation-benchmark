#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/data/wyh/envs/miniforge3/envs/vlms-research/bin/python}"
PATHS_TEMPLATE="${PATHS_TEMPLATE:-/data/wyh/codes/domain-adaptation-benchmark/src/dabench/config/paths.json}"
PATHS_RUNTIME="${PATHS_RUNTIME:-/tmp/damp_dabench_paths_runtime.json}"
LOG_DIR="${LOG_DIR:-${PROJECT_DIR}/logs}"
SEED="${SEED:-1}"
mkdir -p "${LOG_DIR}"

SPLIT_DIR=""
if [ -d "/data/wyh/datasets/domainnet/splits_mini" ]; then
  SPLIT_DIR="/data/wyh/datasets/domainnet/splits_mini"
elif [ -d "/tmp/dabench_minidomainnet_splits" ]; then
  SPLIT_DIR="/tmp/dabench_minidomainnet_splits"
else
  echo "miniDomainNet split_dir not found. expected /data/wyh/datasets/domainnet/splits_mini or /tmp/dabench_minidomainnet_splits"
  exit 1
fi

"${PYTHON_BIN}" - <<PY
import json
from pathlib import Path

src = Path("${PATHS_TEMPLATE}")
dst = Path("${PATHS_RUNTIME}")
payload = json.loads(src.read_text(encoding="utf-8"))
payload.setdefault("datasets", {})
payload.setdefault("datasets", {}).setdefault("minidomainnet", {})
payload["datasets"]["minidomainnet"]["split_dir"] = "${SPLIT_DIR}"
dst.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(dst)
PY

TASK_FILE_GPU0="$(mktemp /tmp/damp_gpu0_tasks.XXXXXX)"
TASK_FILE_GPU1="$(mktemp /tmp/damp_gpu1_tasks.XXXXXX)"

cat > "${TASK_FILE_GPU0}" <<EOF
office_home configs/datasets/office_home.yaml art clipart 0.6 1.0 ${SEED}
office_home configs/datasets/office_home.yaml art product 0.6 1.0 ${SEED}
office_home configs/datasets/office_home.yaml art real_world 0.6 1.0 ${SEED}
office_home configs/datasets/office_home.yaml clipart art 0.6 1.0 ${SEED}
office_home configs/datasets/office_home.yaml clipart product 0.6 1.0 ${SEED}
office_home configs/datasets/office_home.yaml clipart real_world 0.6 1.0 ${SEED}
visda17 configs/datasets/visda17.yaml synthetic real 0.5 2.0 ${SEED}
miniDomainNet configs/datasets/miniDomainNet.yaml clipart painting 0.5 1.0 ${SEED}
miniDomainNet configs/datasets/miniDomainNet.yaml clipart real 0.5 1.0 ${SEED}
miniDomainNet configs/datasets/miniDomainNet.yaml clipart sketch 0.5 1.0 ${SEED}
miniDomainNet configs/datasets/miniDomainNet.yaml painting clipart 0.5 1.0 ${SEED}
miniDomainNet configs/datasets/miniDomainNet.yaml painting real 0.5 1.0 ${SEED}
miniDomainNet configs/datasets/miniDomainNet.yaml painting sketch 0.5 1.0 ${SEED}
EOF

cat > "${TASK_FILE_GPU1}" <<EOF
office_home configs/datasets/office_home.yaml product art 0.6 1.0 ${SEED}
office_home configs/datasets/office_home.yaml product clipart 0.6 1.0 ${SEED}
office_home configs/datasets/office_home.yaml product real_world 0.6 1.0 ${SEED}
office_home configs/datasets/office_home.yaml real_world art 0.6 1.0 ${SEED}
office_home configs/datasets/office_home.yaml real_world clipart 0.6 1.0 ${SEED}
office_home configs/datasets/office_home.yaml real_world product 0.6 1.0 ${SEED}
miniDomainNet configs/datasets/miniDomainNet.yaml real clipart 0.5 1.0 ${SEED}
miniDomainNet configs/datasets/miniDomainNet.yaml real painting 0.5 1.0 ${SEED}
miniDomainNet configs/datasets/miniDomainNet.yaml real sketch 0.5 1.0 ${SEED}
miniDomainNet configs/datasets/miniDomainNet.yaml sketch clipart 0.5 1.0 ${SEED}
miniDomainNet configs/datasets/miniDomainNet.yaml sketch painting 0.5 1.0 ${SEED}
miniDomainNet configs/datasets/miniDomainNet.yaml sketch real 0.5 1.0 ${SEED}
EOF

run_queue() {
  local gpu_id="$1"
  local task_file="$2"
  local log_file="$3"

  while read -r dataset_key dataset_cfg source target tau u seed; do
    ./scripts/run_task.sh "${gpu_id}" "${dataset_key}" "${dataset_cfg}" "${source}" "${target}" "${tau}" "${u}" "${seed}" 2>&1 | tee -a "${log_file}"
  done < "${task_file}"
}

export DABENCH_PATHS_FILE="${PATHS_RUNTIME}"
export PYTHON_BIN

: > "${LOG_DIR}/gpu0.log"
: > "${LOG_DIR}/gpu1.log"

run_queue 0 "${TASK_FILE_GPU0}" "${LOG_DIR}/gpu0.log" &
PID0=$!
run_queue 1 "${TASK_FILE_GPU1}" "${LOG_DIR}/gpu1.log" &
PID1=$!

echo "paths_file=${PATHS_RUNTIME}"
echo "gpu0_tasks=${TASK_FILE_GPU0}"
echo "gpu1_tasks=${TASK_FILE_GPU1}"
echo "logs=${LOG_DIR}/gpu0.log ${LOG_DIR}/gpu1.log"
echo "pids=${PID0} ${PID1}"

wait "${PID0}"
wait "${PID1}"
