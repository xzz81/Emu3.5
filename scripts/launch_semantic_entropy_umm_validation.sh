#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${RUN_ID:-t2i_emu35_fixed_validation_$(date +%Y%m%d_%H%M%S)}"
BASE="${BASE:-outputs/semantic_entropy_umm/${RUN_ID}}"
PILOT_DIR="${PILOT_DIR:-${BASE}/pilot}"
STRICT_DIR="${STRICT_DIR:-${BASE}/strict}"
LOG_DIR="${LOG_DIR:-${BASE}/logs}"
PY="${PY:-./.venv-transformers/bin/python}"

NUM_CONCEPTS="${NUM_CONCEPTS:-3}"
T2I_SAMPLES="${T2I_SAMPLES:-3}"
GEN_DEVICES="${GEN_DEVICES:-0,7}"
STRICT_DEVICES="${STRICT_DEVICES:-0,7}"

mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/driver.log") 2>&1

export PYTHONPATH="$PWD"
export HF_HOME="$PWD/.cache/huggingface"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

echo "[RUN] $(date -Is)"
echo "[RUN] run_id=$RUN_ID"
echo "[RUN] pilot_dir=$PILOT_DIR"
echo "[RUN] strict_dir=$STRICT_DIR"
echo "[RUN] num_concepts=$NUM_CONCEPTS t2i_samples=$T2I_SAMPLES"
echo "[RUN] gen_devices=$GEN_DEVICES strict_devices=$STRICT_DEVICES"

"$PY" scripts/run_semantic_entropy_umm_pilot.py \
  --mode prepare \
  --out-dir "$PILOT_DIR" \
  --num-concepts "$NUM_CONCEPTS"

env CUDA_VISIBLE_DEVICES="$GEN_DEVICES" "$PY" scripts/run_semantic_entropy_umm_pilot.py \
  --mode worker \
  --out-dir "$PILOT_DIR" \
  --routes t2i \
  --num-workers 1 \
  --worker-id 0 \
  --samples-per-concept 0 \
  --t2i-samples-per-concept "$T2I_SAMPLES" \
  --model-path model/Emu3.5 \
  --image-area 1048576 \
  --target-height 64 \
  --target-width 64 \
  --t2i-max-new-tokens 5120 \
  --classifier-free-guidance 2.0 \
  --image-top-k 5120 \
  --image-temperature 1.0 \
  --skip-t2i-readback

"$PY" scripts/run_semantic_entropy_umm_pilot.py \
  --mode aggregate \
  --out-dir "$PILOT_DIR"

env CUDA_VISIBLE_DEVICES="$STRICT_DEVICES" "$PY" scripts/run_semantic_entropy_strict_compare.py \
  --mode worker \
  --pilot-dir "$PILOT_DIR" \
  --strict-dir "$STRICT_DIR" \
  --model-path model/Emu3.5 \
  --samples-per-concept "$T2I_SAMPLES" \
  --num-workers 1 \
  --worker-id 0

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
