#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${RUN_ID:?RUN_ID is required, e.g. t2i_emu35_fixed_validation_20260605_224142}"
BASE="${BASE:-outputs/semantic_entropy_umm/${RUN_ID}}"
PILOT_DIR="${PILOT_DIR:-${BASE}/pilot}"
STRICT_DIR="${STRICT_DIR:-${BASE}/strict}"
LOG_DIR="${LOG_DIR:-${BASE}/logs}"
PY="${PY:-./.venv-transformers/bin/python}"

NUM_CONCEPTS="${NUM_CONCEPTS:-3}"
T2I_SAMPLES="${T2I_SAMPLES:-3}"
DEVICES_A="${DEVICES_A:-0,7}"
DEVICES_B="${DEVICES_B:-3,5}"

mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/parallel_resume.log") 2>&1

export PYTHONPATH="$PWD"
export HF_HOME="$PWD/.cache/huggingface"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

echo "[RUN] $(date -Is)"
echo "[RUN] run_id=$RUN_ID"
echo "[RUN] pilot_dir=$PILOT_DIR"
echo "[RUN] strict_dir=$STRICT_DIR"
echo "[RUN] num_concepts=$NUM_CONCEPTS t2i_samples=$T2I_SAMPLES"
echo "[RUN] devices_a=$DEVICES_A devices_b=$DEVICES_B"

if [[ ! -s "$PILOT_DIR/concepts.jsonl" ]]; then
  "$PY" scripts/run_semantic_entropy_umm_pilot.py \
    --mode prepare \
    --out-dir "$PILOT_DIR" \
    --num-concepts "$NUM_CONCEPTS"
fi

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

launch_gen_worker() {
  local worker_id="$1"
  local devices="$2"
  local log="$LOG_DIR/t2i_worker${worker_id}.log"
  echo "[LAUNCH] t2i worker=$worker_id devices=$devices log=$log" >&2
  env CUDA_VISIBLE_DEVICES="$devices" "$PY" scripts/run_semantic_entropy_umm_pilot.py \
    "${common_gen_args[@]}" --worker-id "$worker_id" \
    > "$log" 2>&1 &
  GEN_PID=$!
}

launch_gen_worker 0 "$DEVICES_A"
pid0="$GEN_PID"
launch_gen_worker 1 "$DEVICES_B"
pid1="$GEN_PID"
wait "$pid0"
launch_gen_worker 2 "$DEVICES_A"
pid2="$GEN_PID"
wait "$pid1"
wait "$pid2"

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

launch_strict_worker() {
  local worker_id="$1"
  local devices="$2"
  local log="$LOG_DIR/strict_worker${worker_id}.log"
  echo "[LAUNCH] strict worker=$worker_id devices=$devices log=$log" >&2
  env CUDA_VISIBLE_DEVICES="$devices" "$PY" scripts/run_semantic_entropy_strict_compare.py \
    "${common_strict_args[@]}" --worker-id "$worker_id" \
    > "$log" 2>&1 &
  STRICT_PID=$!
}

launch_strict_worker 0 "$DEVICES_A"
spid0="$STRICT_PID"
launch_strict_worker 1 "$DEVICES_B"
spid1="$STRICT_PID"
wait "$spid0"
wait "$spid1"

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
