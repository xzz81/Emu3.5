#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${RUN_ID:-semantic_uncertainty_formal_emu35_20260606_095911}"
BASE="${BASE:-outputs/semantic_entropy_umm/${RUN_ID}}"
PILOT_DIR="${PILOT_DIR:-${BASE}/pilot}"
STRICT_DIR="${STRICT_DIR:-${BASE}/strict_compare}"
QA_DIR="${QA_DIR:-${BASE}/semantic_uncertainty_qa}"
LOG_DIR="${LOG_DIR:-${BASE}/logs}"
PY="${PY:-./.venv-transformers/bin/python}"
GEN_WORKERS="${GEN_WORKERS:-2}"
STRICT_WORKERS="${STRICT_WORKERS:-2}"
QA_WORKERS="${QA_WORKERS:-2}"
DEVICE_GROUPS="${DEVICE_GROUPS:-0,7;3,5}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-600}"
MAX_RESTARTS="${MAX_RESTARTS:-3}"

mkdir -p "$LOG_DIR"
WATCH_LOG="$LOG_DIR/watchdog.log"
LOCK_DIR="$BASE/.watchdog.lock"

if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "[WATCH] $(date -Is) another watchdog lock exists: $LOCK_DIR" | tee -a "$WATCH_LOG"
  exit 0
fi
trap 'rmdir "$LOCK_DIR" 2>/dev/null || true' EXIT

has_done() {
  grep -q "\\[DONE\\]" "$LOG_DIR/driver.log" 2>/dev/null
}

has_active_run_process() {
  ps -u "$USER" -o args= \
    | grep -E "tee -a ${LOG_DIR}/driver\\.log|run_semantic_entropy_umm_pilot\\.py.*${RUN_ID}|run_semantic_entropy_strict_compare\\.py.*${RUN_ID}|run_semantic_uncertainty_qa_compare\\.py.*${RUN_ID}" \
    | grep -v grep \
    | grep -v watch_semantic_uncertainty_formal_emu35 >/dev/null
}

t2i_count() {
  find "$PILOT_DIR/generated_images" -type f -name "*.png" 2>/dev/null | wc -l
}

restart_count=0
last_count=""
last_change_ts="$(date +%s)"
echo "[WATCH] $(date -Is) start run_id=$RUN_ID interval=${INTERVAL_SECONDS}s max_restarts=$MAX_RESTARTS" | tee -a "$WATCH_LOG"

while true; do
  if has_done; then
    echo "[WATCH] $(date -Is) driver is done; exiting" | tee -a "$WATCH_LOG"
    exit 0
  fi

  count="$(t2i_count | tr -d ' ')"
  now_ts="$(date +%s)"
  if [ "$count" != "$last_count" ]; then
    last_change_ts="$now_ts"
    last_count="$count"
    stale_seconds=0
  else
    stale_seconds=$((now_ts - last_change_ts))
  fi

  if has_active_run_process; then
    echo "[WATCH] $(date -Is) active count=$count stale_seconds=$stale_seconds" >> "$WATCH_LOG"
    sleep "$INTERVAL_SECONDS"
    continue
  fi

  if [ "$restart_count" -ge "$MAX_RESTARTS" ]; then
    echo "[WATCH] $(date -Is) no active process, restart limit reached count=$count" | tee -a "$WATCH_LOG"
    exit 1
  fi

  restart_count=$((restart_count + 1))
  echo "[WATCH] $(date -Is) no active process; restarting attempt=$restart_count count=$count" | tee -a "$WATCH_LOG"
  RUN_ID="$RUN_ID" \
    BASE="$BASE" \
    PILOT_DIR="$PILOT_DIR" \
    STRICT_DIR="$STRICT_DIR" \
    QA_DIR="$QA_DIR" \
    LOG_DIR="$LOG_DIR" \
    PY="$PY" \
    GEN_WORKERS="$GEN_WORKERS" \
    STRICT_WORKERS="$STRICT_WORKERS" \
    QA_WORKERS="$QA_WORKERS" \
    DEVICE_GROUPS="$DEVICE_GROUPS" \
    bash scripts/launch_semantic_uncertainty_formal_emu35.sh || true
  echo "[WATCH] $(date -Is) launcher returned attempt=$restart_count count=$(t2i_count | tr -d ' ')" | tee -a "$WATCH_LOG"
  sleep "$INTERVAL_SECONDS"
done
