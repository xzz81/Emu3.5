#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

RUN_ID="entropy_emu35image_100diverse_20260529"
OUT="outputs/emu3p5-image/t2i/ume_trace_runs/${RUN_ID}"
mkdir -p "${OUT}/logs"

COMMON_ENV=(
  HOME="${HOME}"
  HF_HOME="${PROJECT_ROOT}/.cache/huggingface"
  HF_HUB_OFFLINE=1
  PYTHONPATH="${PROJECT_ROOT}"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
)

PY="${PY:-${PROJECT_ROOT}/.venv-transformers/bin/python}"
if [[ ! -x "${PY}" ]]; then
  PY="python"
fi
CFG="configs/ume_real_t2i_100diverse.py"

(
  env "${COMMON_ENV[@]}" CUDA_VISIBLE_DEVICES=0,1 \
    "${PY}" scripts/run_entropy_trace.py \
      --cfg "${CFG}" \
      --run-id "${RUN_ID}" \
      --num-workers 2 \
      --worker-id 0
) > "${OUT}/logs/worker0.log" 2>&1 &
echo $! > "${OUT}/logs/worker0.pid"

(
  env "${COMMON_ENV[@]}" CUDA_VISIBLE_DEVICES=2,3 \
    "${PY}" scripts/run_entropy_trace.py \
      --cfg "${CFG}" \
      --run-id "${RUN_ID}" \
      --num-workers 2 \
      --worker-id 1
) > "${OUT}/logs/worker1.log" 2>&1 &
echo $! > "${OUT}/logs/worker1.pid"

echo "run_id=${RUN_ID}"
echo "out=${OUT}"
echo "worker0_pid=$(cat "${OUT}/logs/worker0.pid")"
echo "worker1_pid=$(cat "${OUT}/logs/worker1.pid")"
