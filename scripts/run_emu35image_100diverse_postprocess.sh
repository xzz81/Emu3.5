#!/usr/bin/env bash
set -euo pipefail

cd "/workspace/home/AAAI 2027/Emu3.5"

RUN_ID="entropy_emu35image_100diverse_20260529"
RUN_DIR="outputs/emu3p5-image/t2i/ume_trace_runs/${RUN_ID}"
LOG_DIR="${RUN_DIR}/logs"
POST_LOG="${LOG_DIR}/postprocess.log"

mkdir -p "${LOG_DIR}"

{
  echo "[INFO] postprocess watcher started at $(date -Is)"
  echo "[INFO] run_dir=${RUN_DIR}"

  for worker in worker0 worker1; do
    pid_file="${LOG_DIR}/${worker}.pid"
    if [[ ! -f "${pid_file}" ]]; then
      echo "[ERROR] missing ${pid_file}"
      exit 1
    fi
  done

  for worker in worker0 worker1; do
    pid="$(cat "${LOG_DIR}/${worker}.pid")"
    echo "[INFO] waiting for ${worker} pid=${pid}"
    while kill -0 "${pid}" 2>/dev/null; do
      decoded_count="$(find "${RUN_DIR}/decoded" -name '*.png' 2>/dev/null | wc -l | tr -d ' ')"
      trace_count="$(find "${RUN_DIR}/entropy_traces" -name '*.jsonl' 2>/dev/null | wc -l | tr -d ' ')"
      echo "[INFO] $(date -Is) decoded=${decoded_count} traces=${trace_count}"
      sleep 300
    done
  done

  decoded_count="$(find "${RUN_DIR}/decoded" -name '*.png' 2>/dev/null | wc -l | tr -d ' ')"
  trace_count="$(find "${RUN_DIR}/entropy_traces" -name '*.jsonl' 2>/dev/null | wc -l | tr -d ' ')"
  echo "[INFO] generation finished decoded=${decoded_count} traces=${trace_count}"
  if [[ "${decoded_count}" -lt 100 || "${trace_count}" -lt 100 ]]; then
    echo "[ERROR] expected 100 decoded images and traces"
    exit 1
  fi

  HOME=/workspace/home \
  HF_HOME=/workspace/home/.cache/huggingface \
  HF_HUB_OFFLINE=1 \
  PYTHONPATH=. \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  /workspace/home/conda_envs/emu35-transformers/bin/python \
    scripts/analyze_entropy_attention_relation.py \
      --run-dir "${RUN_DIR}" \
      --model-path /workspace/data/models/Emu3.5/Emu3.5-Image \
      --vq-path /workspace/data/models/Emu3.5/Emu3.5-VisionTokenizer \
      --model-device auto \
      --vq-device cuda:0 \
      --image-area 262144 \
      --describe-max-new-tokens 64

  /workspace/home/conda_envs/emu35-transformers/bin/python \
    scripts/build_entropy_attention_deliverables.py \
      --run-dir "${RUN_DIR}" \
      --entropy-field u_tok_full \
      --top-frac 0.2

  echo "[INFO] postprocess finished at $(date -Is)"
} >> "${POST_LOG}" 2>&1
