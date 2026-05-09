#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/data/wyh/envs/miniforge3/envs/vlms-research/bin/python}"
LOG_DIR="${LOG_DIR:-${PROJECT_DIR}/logs}"
mkdir -p "${LOG_DIR}"

TASK_FILE_GPU0="$(mktemp /tmp/tasc_gpu0_tasks.XXXXXX)"
TASK_FILE_GPU1="$(mktemp /tmp/tasc_gpu1_tasks.XXXXXX)"

cat > "${TASK_FILE_GPU0}" <<'EOF'
configs-reproduce/exp-re/office/CDA/office_ad_31-0-0.yaml
configs-reproduce/exp-re/office/CDA/office_da_31-0-0.yaml
configs-reproduce/exp-re/office/CDA/office_wa_31-0-0.yaml
configs-reproduce/exp-re/office/ODA/office_ad_10-0-11.yaml
configs-reproduce/exp-re/office/ODA/office_da_10-0-11.yaml
configs-reproduce/exp-re/office/ODA/office_wa_10-0-11.yaml
configs-reproduce/exp-re/office/OPDA/office_ad_10-10-11.yaml
configs-reproduce/exp-re/office/OPDA/office_da_10-10-11.yaml
configs-reproduce/exp-re/office/OPDA/office_wa_10-10-11.yaml
configs-reproduce/exp-re/office/PDA/office_ad_10-21-0.yaml
configs-reproduce/exp-re/office/PDA/office_da_10-21-0.yaml
configs-reproduce/exp-re/office/PDA/office_wa_10-21-0.yaml
configs-reproduce/exp-re/officehome/CDA/officehome_AC_65-0-0.yaml
configs-reproduce/exp-re/officehome/CDA/officehome_AR_65-0-0.yaml
configs-reproduce/exp-re/officehome/CDA/officehome_CP_65-0-0.yaml
configs-reproduce/exp-re/officehome/CDA/officehome_PA_65-0-0.yaml
configs-reproduce/exp-re/officehome/CDA/officehome_PR_65-0-0.yaml
configs-reproduce/exp-re/officehome/CDA/officehome_RC_65-0-0.yaml
configs-reproduce/exp-re/officehome/ODA/officehome_AC_25-0-40.yaml
configs-reproduce/exp-re/officehome/ODA/officehome_AR_25-0-40.yaml
configs-reproduce/exp-re/officehome/ODA/officehome_CP_25-0-40.yaml
configs-reproduce/exp-re/officehome/ODA/officehome_PA_25-0-40.yaml
configs-reproduce/exp-re/officehome/ODA/officehome_PR_25-0-40.yaml
configs-reproduce/exp-re/officehome/ODA/officehome_RC_25-0-40.yaml
configs-reproduce/exp-re/officehome/OPDA/officehome_AC_10-5-50.yaml
configs-reproduce/exp-re/officehome/OPDA/officehome_AR_10-5-50.yaml
configs-reproduce/exp-re/officehome/OPDA/officehome_CP_10-5-50.yaml
configs-reproduce/exp-re/officehome/OPDA/officehome_PA_10-5-50.yaml
configs-reproduce/exp-re/officehome/OPDA/officehome_PR_10-5-50.yaml
configs-reproduce/exp-re/officehome/OPDA/officehome_RC_10-5-50.yaml
configs-reproduce/exp-re/officehome/PDA/officehome_AC_25-40-0.yaml
configs-reproduce/exp-re/officehome/PDA/officehome_AR_25-40-0.yaml
configs-reproduce/exp-re/officehome/PDA/officehome_CP_25-40-0.yaml
configs-reproduce/exp-re/officehome/PDA/officehome_PA_25-40-0.yaml
configs-reproduce/exp-re/officehome/PDA/officehome_PR_25-40-0.yaml
configs-reproduce/exp-re/officehome/PDA/officehome_RC_25-40-0.yaml
configs-reproduce/exp-re/domainnet/OPDA/domainnet_pr_150-50-145.yaml
configs-reproduce/exp-re/domainnet/OPDA/domainnet_rp_150-50-145.yaml
configs-reproduce/exp-re/domainnet/OPDA/domainnet_sp_150-50-145.yaml
configs-reproduce/exp-re/visda/OPDA/visda_SR_6-3-3.yaml
EOF

cat > "${TASK_FILE_GPU1}" <<'EOF'
configs-reproduce/exp-re/office/CDA/office_aw_31-0-0.yaml
configs-reproduce/exp-re/office/CDA/office_dw_31-0-0.yaml
configs-reproduce/exp-re/office/CDA/office_wd_31-0-0.yaml
configs-reproduce/exp-re/office/ODA/office_aw_10-0-11.yaml
configs-reproduce/exp-re/office/ODA/office_dw_10-0-11.yaml
configs-reproduce/exp-re/office/ODA/office_wd_10-0-11.yaml
configs-reproduce/exp-re/office/OPDA/office_aw_10-10-11.yaml
configs-reproduce/exp-re/office/OPDA/office_dw_10-10-11.yaml
configs-reproduce/exp-re/office/OPDA/office_wd_10-10-11.yaml
configs-reproduce/exp-re/office/PDA/office_aw_10-21-0.yaml
configs-reproduce/exp-re/office/PDA/office_dw_10-21-0.yaml
configs-reproduce/exp-re/office/PDA/office_wd_10-21-0.yaml
configs-reproduce/exp-re/officehome/CDA/officehome_AP_65-0-0.yaml
configs-reproduce/exp-re/officehome/CDA/officehome_CA_65-0-0.yaml
configs-reproduce/exp-re/officehome/CDA/officehome_CR_65-0-0.yaml
configs-reproduce/exp-re/officehome/CDA/officehome_PC_65-0-0.yaml
configs-reproduce/exp-re/officehome/CDA/officehome_RA_65-0-0.yaml
configs-reproduce/exp-re/officehome/CDA/officehome_RP_65-0-0.yaml
configs-reproduce/exp-re/officehome/ODA/officehome_AP_25-0-40.yaml
configs-reproduce/exp-re/officehome/ODA/officehome_CA_25-0-40.yaml
configs-reproduce/exp-re/officehome/ODA/officehome_CR_25-0-40.yaml
configs-reproduce/exp-re/officehome/ODA/officehome_PC_25-0-40.yaml
configs-reproduce/exp-re/officehome/ODA/officehome_RA_25-0-40.yaml
configs-reproduce/exp-re/officehome/ODA/officehome_RP_25-0-40.yaml
configs-reproduce/exp-re/officehome/OPDA/officehome_AP_10-5-50.yaml
configs-reproduce/exp-re/officehome/OPDA/officehome_CA_10-5-50.yaml
configs-reproduce/exp-re/officehome/OPDA/officehome_CR_10-5-50.yaml
configs-reproduce/exp-re/officehome/OPDA/officehome_PC_10-5-50.yaml
configs-reproduce/exp-re/officehome/OPDA/officehome_RA_10-5-50.yaml
configs-reproduce/exp-re/officehome/OPDA/officehome_RP_10-5-50.yaml
configs-reproduce/exp-re/officehome/PDA/officehome_AP_25-40-0.yaml
configs-reproduce/exp-re/officehome/PDA/officehome_CA_25-40-0.yaml
configs-reproduce/exp-re/officehome/PDA/officehome_CR_25-40-0.yaml
configs-reproduce/exp-re/officehome/PDA/officehome_PC_25-40-0.yaml
configs-reproduce/exp-re/officehome/PDA/officehome_RA_25-40-0.yaml
configs-reproduce/exp-re/officehome/PDA/officehome_RP_25-40-0.yaml
configs-reproduce/exp-re/domainnet/OPDA/domainnet_ps_150-50-145.yaml
configs-reproduce/exp-re/domainnet/OPDA/domainnet_rs_150-50-145.yaml
configs-reproduce/exp-re/domainnet/OPDA/domainnet_sr_150-50-145.yaml
EOF

run_queue() {
  local gpu_id="$1"
  local task_file="$2"
  local log_file="$3"

  while read -r cfg_rel; do
    [ -n "${cfg_rel}" ] || continue
    ./scripts/run_task.sh "${gpu_id}" "${cfg_rel}" 2>&1 | tee -a "${log_file}"
  done < "${task_file}"
}

export PYTHON_BIN

: > "${LOG_DIR}/gpu0.log"
: > "${LOG_DIR}/gpu1.log"

run_queue 0 "${TASK_FILE_GPU0}" "${LOG_DIR}/gpu0.log" &
PID0=$!
run_queue 1 "${TASK_FILE_GPU1}" "${LOG_DIR}/gpu1.log" &
PID1=$!

echo "gpu0_tasks=${TASK_FILE_GPU0}"
echo "gpu1_tasks=${TASK_FILE_GPU1}"
echo "logs=${LOG_DIR}/gpu0.log ${LOG_DIR}/gpu1.log"
echo "pids=${PID0} ${PID1}"

wait "${PID0}"
wait "${PID1}"
