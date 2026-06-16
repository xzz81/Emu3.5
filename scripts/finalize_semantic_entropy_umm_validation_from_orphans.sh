#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${RUN_ID:?RUN_ID is required}"
GEN_WORKER0_PID="${GEN_WORKER0_PID:?GEN_WORKER0_PID is required}"
GEN_WORKER1_PID="${GEN_WORKER1_PID:?GEN_WORKER1_PID is required}"

BASE="${BASE:-outputs/semantic_entropy_umm/${RUN_ID}}"
PILOT_DIR="${PILOT_DIR:-${BASE}/pilot}"
STRICT_DIR="${STRICT_DIR:-${BASE}/strict}"
LOG_DIR="${LOG_DIR:-${BASE}/logs}"
PY="${PY:-./.venv-transformers/bin/python}"

NUM_CONCEPTS="${NUM_CONCEPTS:-3}"
T2I_SAMPLES="${T2I_SAMPLES:-3}"
DEVICES_A="${DEVICES_A:-0,7}"
DEVICES_B="${DEVICES_B:-3,5}"
POLL_SECONDS="${POLL_SECONDS:-60}"

mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/orphan_finalize.log") 2>&1

export PYTHONPATH="$PWD"
export HF_HOME="$PWD/.cache/huggingface"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

echo "[RUN] $(date -Is)"
echo "[RUN] run_id=$RUN_ID"
echo "[RUN] pilot_dir=$PILOT_DIR"
echo "[RUN] strict_dir=$STRICT_DIR"
echo "[RUN] worker0_pid=$GEN_WORKER0_PID worker1_pid=$GEN_WORKER1_PID"

wait_for_external_pid_exit() {
  local pid="$1"
  local label="$2"
  while true; do
    local stat
    stat="$(ps -p "$pid" -o stat= 2>/dev/null | awk '{print $1}')"
    if [[ -z "$stat" || "$stat" == Z* ]]; then
      break
    fi
    echo "[WAIT] $label pid=$pid still running stat=$stat at $(date -Is)"
    sleep "$POLL_SECONDS"
  done
  echo "[WAIT] $label pid=$pid exited at $(date -Is)"
}

common_gen_args=(
  --mode worker
  --out-dir "$PILOT_DIR"
  --routes t2i
  --num-workers 3
  --samples-per-concept 0
  --t2i-samples-per-concept "$T2I_SAMPLES"
  --model-path model/Emu3.5
  --image-area 1048576
  --target-height 64
  --target-width 64
  --t2i-max-new-tokens 5120
  --classifier-free-guidance 2.0
  --image-top-k 5120
  --image-temperature 1.0
  --skip-t2i-readback
  --skip-existing
)

wait_for_external_pid_exit "$GEN_WORKER0_PID" "t2i-worker0"

echo "[LAUNCH] t2i worker=2 devices=$DEVICES_A"
env CUDA_VISIBLE_DEVICES="$DEVICES_A" "$PY" scripts/run_semantic_entropy_umm_pilot.py \
  "${common_gen_args[@]}" --worker-id 2 \
  > "$LOG_DIR/t2i_worker2.log" 2>&1 &
GEN_WORKER2_PID=$!

wait_for_external_pid_exit "$GEN_WORKER1_PID" "t2i-worker1"
wait "$GEN_WORKER2_PID"
echo "[WAIT] t2i-worker2 pid=$GEN_WORKER2_PID exited at $(date -Is)"

"$PY" scripts/run_semantic_entropy_umm_pilot.py \
  --mode aggregate \
  --out-dir "$PILOT_DIR"

common_strict_args=(
  --mode worker
  --pilot-dir "$PILOT_DIR"
  --strict-dir "$STRICT_DIR"
  --model-path model/Emu3.5
  --samples-per-concept "$T2I_SAMPLES"
  --num-workers 2
)

echo "[LAUNCH] strict worker=0 devices=$DEVICES_A"
env CUDA_VISIBLE_DEVICES="$DEVICES_A" "$PY" scripts/run_semantic_entropy_strict_compare.py \
  "${common_strict_args[@]}" --worker-id 0 \
  > "$LOG_DIR/strict_worker0.log" 2>&1 &
STRICT0_PID=$!

echo "[LAUNCH] strict worker=1 devices=$DEVICES_B"
env CUDA_VISIBLE_DEVICES="$DEVICES_B" "$PY" scripts/run_semantic_entropy_strict_compare.py \
  "${common_strict_args[@]}" --worker-id 1 \
  > "$LOG_DIR/strict_worker1.log" 2>&1 &
STRICT1_PID=$!

wait "$STRICT0_PID"
wait "$STRICT1_PID"

"$PY" scripts/run_semantic_entropy_strict_compare.py \
  --mode aggregate \
  --pilot-dir "$PILOT_DIR" \
  --strict-dir "$STRICT_DIR" \
  --samples-per-concept "$T2I_SAMPLES"

"$PY" scripts/run_semantic_entropy_strict_compare.py \
  --mode audit \
  --pilot-dir "$PILOT_DIR" \
  --strict-dir "$STRICT_DIR" \
  --samples-per-concept "$T2I_SAMPLES"

echo "[DONE] $(date -Is)"
