#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${RUN_ID:-semantic_uncertainty_complex_t2i_emu35_$(date +%Y%m%d_%H%M%S)}"
BASE="${BASE:-outputs/semantic_entropy_umm/${RUN_ID}}"
PILOT_DIR="${PILOT_DIR:-${BASE}/pilot}"
STRICT_DIR="${STRICT_DIR:-${BASE}/strict_compare}"
QA_DIR="${QA_DIR:-${BASE}/semantic_uncertainty_qa}"
LOG_DIR="${LOG_DIR:-${BASE}/logs}"
PY="${PY:-./.venv-transformers/bin/python}"
CONCEPTS_FILE="${CONCEPTS_FILE:-configs/semantic_entropy_t2i_complex_scene_concepts.jsonl}"

NUM_CONCEPTS="${NUM_CONCEPTS:-30}"
T2I_SAMPLES="${T2I_SAMPLES:-10}"
GEN_WORKERS="${GEN_WORKERS:-2}"
STRICT_WORKERS="${STRICT_WORKERS:-2}"
QA_WORKERS="${QA_WORKERS:-2}"
RUN_STRICT_QA="${RUN_STRICT_QA:-1}"
DEVICE_GROUPS="${DEVICE_GROUPS:-0,3;5,7}"
IFS=';' read -r -a DEVICE_GROUPS_ARRAY <<< "$DEVICE_GROUPS"
if [ "${#DEVICE_GROUPS_ARRAY[@]}" -eq 0 ]; then
  echo "[ERROR] DEVICE_GROUPS resolved to no device groups" >&2
  exit 2
fi

mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/driver.log") 2>&1

export PYTHONPATH="$PWD"
export HF_HOME="$PWD/.cache/huggingface"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

echo "[RUN] $(date -Is)"
echo "[RUN] run_id=$RUN_ID"
echo "[RUN] pilot_dir=$PILOT_DIR"
echo "[RUN] strict_dir=$STRICT_DIR"
echo "[RUN] qa_dir=$QA_DIR"
echo "[RUN] concepts_file=$CONCEPTS_FILE"
echo "[RUN] num_concepts=$NUM_CONCEPTS t2i_samples=$T2I_SAMPLES"
echo "[RUN] device_groups=$DEVICE_GROUPS"
echo "[RUN] gen_workers=$GEN_WORKERS strict_workers=$STRICT_WORKERS qa_workers=$QA_WORKERS run_strict_qa=$RUN_STRICT_QA"
echo "[RUN] GPU snapshot before launch:"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader || true

"$PY" scripts/run_semantic_entropy_umm_pilot.py \
  --mode prepare \
  --out-dir "$PILOT_DIR" \
  --concepts-file "$CONCEPTS_FILE" \
  --num-concepts "$NUM_CONCEPTS"

common_gen_args=(
  --mode worker
  --out-dir "$PILOT_DIR"
  --concepts-file "$CONCEPTS_FILE"
  --routes t2i
  --num-workers "$GEN_WORKERS"
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

launch_worker() {
  local label="$1"
  local worker_id="$2"
  local devices="$3"
  shift 3
  local log="$LOG_DIR/${label}_worker${worker_id}.log"
  echo "[LAUNCH] ${label} worker=$worker_id devices=$devices log=$log"
  env CUDA_VISIBLE_DEVICES="$devices" "$PY" "$@" --worker-id "$worker_id" > "$log" 2>&1 &
  LAUNCHED_PID=$!
}

launch_worker_group() {
  local label="$1"
  local worker_count="$2"
  shift 2
  local pids=()
  local worker_id devices idx
  for ((worker_id = 0; worker_id < worker_count; worker_id++)); do
    idx=$((worker_id % ${#DEVICE_GROUPS_ARRAY[@]}))
    devices="${DEVICE_GROUPS_ARRAY[$idx]}"
    launch_worker "$label" "$worker_id" "$devices" "$@"
    pids+=("$LAUNCHED_PID")
  done
  for pid in "${pids[@]}"; do
    wait "$pid"
  done
}

echo "[STAGE] Complex T2I generation"
launch_worker_group t2i "$GEN_WORKERS" scripts/run_semantic_entropy_umm_pilot.py "${common_gen_args[@]}"

"$PY" scripts/run_semantic_entropy_umm_pilot.py \
  --mode aggregate \
  --out-dir "$PILOT_DIR" \
  --concepts-file "$CONCEPTS_FILE"

if [ "$RUN_STRICT_QA" != "1" ]; then
  echo "[DONE] generation-only complex T2I run $(date -Is)"
  exit 0
fi

echo "[STAGE] Strict forced-choice compare"
common_strict_args=(
  --mode worker
  --pilot-dir "$PILOT_DIR"
  --strict-dir "$STRICT_DIR"
  --model-path model/Emu3.5
  --samples-per-concept "$T2I_SAMPLES"
  --num-workers "$STRICT_WORKERS"
  --skip-existing
)
launch_worker_group strict "$STRICT_WORKERS" scripts/run_semantic_entropy_strict_compare.py "${common_strict_args[@]}"

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

echo "[STAGE] Semantic-uncertainty-style QA compare"
common_qa_args=(
  --mode worker
  --pilot-dir "$PILOT_DIR"
  --qa-dir "$QA_DIR"
  --model-path model/Emu3.5
  --samples-per-concept "$T2I_SAMPLES"
  --num-workers "$QA_WORKERS"
  --answer-temperature 1.0
  --answer-top-k 1024
  --answer-top-p 0.95
  --skip-existing
)
launch_worker_group qa "$QA_WORKERS" scripts/run_semantic_uncertainty_qa_compare.py "${common_qa_args[@]}"

"$PY" scripts/run_semantic_uncertainty_qa_compare.py \
  --mode aggregate \
  --pilot-dir "$PILOT_DIR" \
  --qa-dir "$QA_DIR" \
  --samples-per-concept "$T2I_SAMPLES"

"$PY" scripts/run_semantic_uncertainty_qa_compare.py \
  --mode audit \
  --pilot-dir "$PILOT_DIR" \
  --qa-dir "$QA_DIR" \
  --samples-per-concept "$T2I_SAMPLES"

echo "[DONE] $(date -Is)"
