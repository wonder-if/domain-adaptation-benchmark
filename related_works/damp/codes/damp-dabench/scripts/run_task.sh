#!/bin/bash

set -euo pipefail

if [ "$#" -lt 8 ]; then
  echo "usage: $0 <gpu_id> <dataset_key> <dataset_cfg> <source> <target> <tau> <u> <seed> [extra opts...]"
  exit 1
fi

GPU_ID="$1"
DATASET_KEY="$2"
DATASET_CFG="$3"
SOURCE="$4"
TARGET="$5"
TAU="$6"
U="$7"
SEED="$8"
shift 8

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/data/wyh/envs/miniforge3/envs/vlms-research/bin/python}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

case "${DATASET_KEY}" in
  office_home)
    DATASET_NAME="office_home"
    BACKBONE_LOCAL_PATH="${PROJECT_DIR}/assets/RN50.pt"
    ;;
  visda17)
    DATASET_NAME="visda17"
    BACKBONE_LOCAL_PATH="${PROJECT_DIR}/assets/ViT-B-16.pt"
    ;;
  miniDomainNet)
    DATASET_NAME="miniDomainNet"
    BACKBONE_LOCAL_PATH="${PROJECT_DIR}/assets/RN50.pt"
    ;;
  *)
    echo "unsupported dataset key: ${DATASET_KEY}"
    exit 1
    ;;
esac

RUN_TAG="${TAU}_${U}"
NAME="${SOURCE}_to_${TARGET}"
OUTPUT_DIR="${PROJECT_DIR}/output/${DATASET_NAME}/DAMP/damp/${RUN_TAG}_${NAME}/seed_${SEED}"
mkdir -p "${OUTPUT_DIR}"

echo "[start] gpu=${GPU_ID} dataset=${DATASET_KEY} source=${SOURCE} target=${TARGET} output=${OUTPUT_DIR}"

"${PYTHON_BIN}" train.py \
  --root /unused/with-dabench \
  --trainer DAMP \
  --dataset-config-file "${DATASET_CFG}" \
  --config-file "configs/trainers/DAMP/damp.yaml" \
  --output-dir "${OUTPUT_DIR}" \
  --source-domains "${SOURCE}" \
  --target-domains "${TARGET}" \
  --seed "${SEED}" \
  TRAINER.DAMP.TAU "${TAU}" \
  TRAINER.DAMP.U "${U}" \
  MODEL.BACKBONE.LOCAL_PATH "${BACKBONE_LOCAL_PATH}" \
  "$@"

echo "[done] gpu=${GPU_ID} dataset=${DATASET_KEY} source=${SOURCE} target=${TARGET}"
