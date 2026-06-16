#!/usr/bin/env bash
set -euo pipefail

# Run after outputs/semantic_entropy_umm/extractor_validation/manual_annotations.csv
# has been completed. By default this validates manual annotations, scores the
# extractor validation, and refreshes paper/status summaries. Full robustness
# runs are opt-in through RUN_FULL_ROBUSTNESS.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PY="${PY:-./.venv-transformers/bin/python}"
STRICT_DIR="${STRICT_DIR:-outputs/semantic_entropy_umm/strict_compare}"
EXTRACTOR_VALIDATION_DIR="${EXTRACTOR_VALIDATION_DIR:-outputs/semantic_entropy_umm/extractor_validation}"
MANUAL_PROGRESS_DIR="${MANUAL_PROGRESS_DIR:-outputs/semantic_entropy_umm/manual_annotation_progress}"
MANUAL_PACKAGE_DIR="${MANUAL_PACKAGE_DIR:-outputs/semantic_entropy_umm/manual_annotation_package}"
MANUAL_PACKAGE_QA_DIR="${MANUAL_PACKAGE_QA_DIR:-outputs/semantic_entropy_umm/manual_annotation_package_qa}"
MANUAL_IMPORT_DIR="${MANUAL_IMPORT_DIR:-outputs/semantic_entropy_umm/manual_annotation_import}"
POST_MANUAL_SMOKE_DIR="${POST_MANUAL_SMOKE_DIR:-outputs/semantic_entropy_umm/post_manual_smoke}"
PILOT_DIR="${PILOT_DIR:-outputs/semantic_entropy_umm/pilot}"
QUADRANT_DIR="${QUADRANT_DIR:-outputs/semantic_entropy_umm/quadrant_analysis}"
BOOTSTRAP_DIR="${BOOTSTRAP_DIR:-outputs/semantic_entropy_umm/bootstrap_stability}"
SAMPLE_SIZE_DIR="${SAMPLE_SIZE_DIR:-outputs/semantic_entropy_umm/sample_size_sensitivity}"
OPTION_ORDER_SMOKE_DIR="${OPTION_ORDER_SMOKE_DIR:-outputs/semantic_entropy_umm/robustness_option_order_smoke}"
OPTION_ORDER_FULL_DIR="${OPTION_ORDER_FULL_DIR:-outputs/semantic_entropy_umm/robustness_option_order}"
PROMPT_TEMPLATE_SMOKE_DIR="${PROMPT_TEMPLATE_SMOKE_DIR:-outputs/semantic_entropy_umm/robustness_prompt_template_smoke}"
PROMPT_TEMPLATE_FULL_DIR="${PROMPT_TEMPLATE_FULL_DIR:-outputs/semantic_entropy_umm/robustness_prompt_template}"
ROBUSTNESS_PREFLIGHT_DIR="${ROBUSTNESS_PREFLIGHT_DIR:-outputs/semantic_entropy_umm/robustness_preflight}"
ALL_GPU_RUNBOOK_DIR="${ALL_GPU_RUNBOOK_DIR:-outputs/semantic_entropy_umm/all_gpu_execution_runbook}"
OBJECT_BINDING_DIR="${OBJECT_BINDING_DIR:-outputs/semantic_entropy_umm/object_binding_sanity}"
IMAGE_DUPLICATE_DIR="${IMAGE_DUPLICATE_DIR:-outputs/semantic_entropy_umm/image_duplicate_mode_collapse}"
CONCEPT_DIFFICULTY_DIR="${CONCEPT_DIFFICULTY_DIR:-outputs/semantic_entropy_umm/concept_difficulty_split}"
RANDOM_CONTROL_DIR="${RANDOM_CONTROL_DIR:-outputs/semantic_entropy_umm/random_concept_control}"
SLOT_ABLATION_DIR="${SLOT_ABLATION_DIR:-outputs/semantic_entropy_umm/slot_ablation}"
CLAIM_BOUNDARY_DIR="${CLAIM_BOUNDARY_DIR:-outputs/semantic_entropy_umm/claim_boundary_audit}"
STAGE_REPORT_AUDIT_DIR="${STAGE_REPORT_AUDIT_DIR:-outputs/semantic_entropy_umm/stage_report_audit}"
REQUIREMENT_AUDIT_DIR="${REQUIREMENT_AUDIT_DIR:-outputs/semantic_entropy_umm/requirement_audit}"
EXECUTION_STATUS_DIR="${EXECUTION_STATUS_DIR:-outputs/semantic_entropy_umm/execution_status}"
PAPER_DIR="${PAPER_DIR:-outputs/semantic_entropy_umm/paper_tables_preliminary}"
RUN_FULL_ROBUSTNESS="${RUN_FULL_ROBUSTNESS:-0}"
WAIT_FOR_GPUS_SECONDS="${WAIT_FOR_GPUS_SECONDS:-0}"
WAIT_FOR_GPUS_INTERVAL_SECONDS="${WAIT_FOR_GPUS_INTERVAL_SECONDS:-60}"
DRY_RUN_FULL_ROBUSTNESS="${DRY_RUN_FULL_ROBUSTNESS:-0}"
FULL_ROBUSTNESS_DRY_RUN_DIR="${FULL_ROBUSTNESS_DRY_RUN_DIR:-outputs/semantic_entropy_umm/full_robustness_dry_run}"

export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export PYTHONPATH="${PYTHONPATH:-$PWD}"

prepare_full_robustness() {
  echo "[INFO] regenerating all-GPU full robustness launchers"
  "$PY" scripts/run_semantic_entropy_option_order_robustness.py \
    --mode write-launcher \
    --out-dir "$OPTION_ORDER_FULL_DIR"
  "$PY" scripts/run_semantic_entropy_prompt_template_robustness.py \
    --mode write-launcher \
    --out-dir "$PROMPT_TEMPLATE_FULL_DIR"

  echo "[INFO] checking full robustness GPU coverage and availability"
  local start_ts
  start_ts="$(date +%s)"
  while true; do
    "$PY" scripts/build_semantic_entropy_robustness_preflight.py \
      --validation-report "$EXTRACTOR_VALIDATION_DIR/manual_annotation_validation_report.json" \
      --option-launcher-dir "$OPTION_ORDER_FULL_DIR/launchers" \
      --prompt-launcher-dir "$PROMPT_TEMPLATE_FULL_DIR/launchers" \
      --out-dir "$ROBUSTNESS_PREFLIGHT_DIR"
    "$PY" scripts/build_semantic_entropy_all_gpu_runbook.py \
      --validation-report "$EXTRACTOR_VALIDATION_DIR/manual_annotation_validation_report.json" \
      --robustness-preflight-report "$ROBUSTNESS_PREFLIGHT_DIR/robustness_preflight_report.json" \
      --option-launcher-dir "$OPTION_ORDER_FULL_DIR/launchers" \
      --prompt-launcher-dir "$PROMPT_TEMPLATE_FULL_DIR/launchers" \
      --out-dir "$ALL_GPU_RUNBOOK_DIR"

    if "$PY" - <<PY
import json
from pathlib import Path

path = Path("$ROBUSTNESS_PREFLIGHT_DIR") / "robustness_preflight_report.json"
data = json.loads(path.read_text(encoding="utf-8"))
if data.get("manual_validation_status") != "PASS":
    raise SystemExit(f"manual validation is not PASS: {data.get('manual_validation_status')}")
if not data.get("launcher_gpu_coverage_ok"):
    raise SystemExit(f"launcher GPU coverage is not OK: {path}")
if not data.get("gpu_availability_ok_for_launch"):
    raise SystemExit(f"GPU availability is not OK; busy GPUs: {data.get('busy_gpu_indices')}")
print("[INFO] full robustness preflight PASS; all detected GPUs are covered and available")
PY
    then
      break
    fi

    local now_ts
    now_ts="$(date +%s)"
    local elapsed=$((now_ts - start_ts))
    if [ "$WAIT_FOR_GPUS_SECONDS" -le 0 ] || [ "$elapsed" -ge "$WAIT_FOR_GPUS_SECONDS" ]; then
      echo "[ERROR] full robustness preflight did not pass within WAIT_FOR_GPUS_SECONDS=$WAIT_FOR_GPUS_SECONDS" >&2
      return 1
    fi
    echo "[INFO] GPUs not ready yet; sleeping ${WAIT_FOR_GPUS_INTERVAL_SECONDS}s before retry ($elapsed/${WAIT_FOR_GPUS_SECONDS}s elapsed)"
    sleep "$WAIT_FOR_GPUS_INTERVAL_SECONDS"
  done
}

run_full_robustness_launcher() {
  local family="$1"
  local launcher="$2"
  if [ "$DRY_RUN_FULL_ROBUSTNESS" = "1" ]; then
    echo "[INFO] dry-run full robustness launcher: $family -> $launcher"
    mkdir -p "$FULL_ROBUSTNESS_DRY_RUN_DIR"
    "$PY" - <<PY
import json
from pathlib import Path

out_dir = Path("$FULL_ROBUSTNESS_DRY_RUN_DIR")
payload = {
    "status": "DRY_RUN_READY_TO_LAUNCH",
    "family": "$family",
    "launcher": "$launcher",
    "run_full_robustness": "$RUN_FULL_ROBUSTNESS",
    "dry_run_full_robustness": True,
    "claim_boundary": "dry-run only; no model worker launched",
}
(out_dir / "full_robustness_dry_run_report.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
(out_dir / "full_robustness_dry_run_report.md").write_text(
    "\n".join([
        "# Full Robustness Dry Run",
        "",
        "Status: DRY_RUN_READY_TO_LAUNCH",
        "",
        "This report records the launcher that would run after manual validation and all-GPU preflight pass. It did not start model inference.",
        "",
        f"- Family: `{payload['family']}`",
        f"- Launcher: `{payload['launcher']}`",
        f"- RUN_FULL_ROBUSTNESS: `{payload['run_full_robustness']}`",
    ]) + "\n",
    encoding="utf-8",
)
print(json.dumps({"status": payload["status"], "family": payload["family"], "launcher": payload["launcher"]}, indent=2))
PY
  else
    bash "$launcher"
  fi
}

echo "[INFO] refreshing manual annotation progress"
"$PY" scripts/build_semantic_entropy_manual_progress.py \
  --extractor-validation-dir "$EXTRACTOR_VALIDATION_DIR" \
  --out-dir "$MANUAL_PROGRESS_DIR"

echo "[INFO] validating manual annotations"
"$PY" scripts/build_semantic_entropy_extractor_validation.py \
  --mode validate-manual \
  --out-dir "$EXTRACTOR_VALIDATION_DIR"

echo "[INFO] scoring extractor validation"
"$PY" scripts/build_semantic_entropy_extractor_validation.py \
  --mode score \
  --strict-dir "$STRICT_DIR" \
  --out-dir "$EXTRACTOR_VALIDATION_DIR"

case "$RUN_FULL_ROBUSTNESS" in
  0|none|"")
    echo "[INFO] full robustness not requested"
    ;;
  option-order)
    prepare_full_robustness
    echo "[INFO] running full option-order robustness"
    run_full_robustness_launcher "option-order" "$OPTION_ORDER_FULL_DIR/launchers/launch_all_option_order_seeds.sh"
    ;;
  prompt-template)
    prepare_full_robustness
    echo "[INFO] running full prompt-template robustness"
    run_full_robustness_launcher "prompt-template" "$PROMPT_TEMPLATE_FULL_DIR/launchers/launch_all_prompt_templates.sh"
    ;;
  both)
    prepare_full_robustness
    echo "[INFO] running full option-order robustness"
    run_full_robustness_launcher "option-order" "$OPTION_ORDER_FULL_DIR/launchers/launch_all_option_order_seeds.sh"
    echo "[INFO] running full prompt-template robustness"
    run_full_robustness_launcher "prompt-template" "$PROMPT_TEMPLATE_FULL_DIR/launchers/launch_all_prompt_templates.sh"
    ;;
  *)
    echo "[ERROR] RUN_FULL_ROBUSTNESS must be 0, none, option-order, prompt-template, or both" >&2
    exit 2
    ;;
esac

echo "[INFO] refreshing claim-boundary audit"
"$PY" scripts/build_semantic_entropy_claim_boundary_audit.py \
  --docs-dir docs \
  --outputs-dir outputs/semantic_entropy_umm \
  --out-dir "$CLAIM_BOUNDARY_DIR"

echo "[INFO] refreshing preliminary paper tables"
"$PY" scripts/build_semantic_entropy_paper_tables.py \
  --strict-dir "$STRICT_DIR" \
  --extractor-validation-dir "$EXTRACTOR_VALIDATION_DIR" \
  --manual-progress-dir "$MANUAL_PROGRESS_DIR" \
  --manual-package-dir "$MANUAL_PACKAGE_DIR" \
  --manual-package-qa-dir "$MANUAL_PACKAGE_QA_DIR" \
  --manual-import-dir "$MANUAL_IMPORT_DIR" \
  --post-manual-smoke-dir "$POST_MANUAL_SMOKE_DIR" \
  --quadrant-dir "$QUADRANT_DIR" \
  --bootstrap-dir "$BOOTSTRAP_DIR" \
  --sample-size-dir "$SAMPLE_SIZE_DIR" \
  --option-order-smoke-dir "$OPTION_ORDER_SMOKE_DIR" \
  --option-order-full-dir "$OPTION_ORDER_FULL_DIR" \
  --prompt-template-smoke-dir "$PROMPT_TEMPLATE_SMOKE_DIR" \
  --prompt-template-full-dir "$PROMPT_TEMPLATE_FULL_DIR" \
  --robustness-preflight-dir "$ROBUSTNESS_PREFLIGHT_DIR" \
  --object-binding-dir "$OBJECT_BINDING_DIR" \
  --image-duplicate-dir "$IMAGE_DUPLICATE_DIR" \
  --concept-difficulty-dir "$CONCEPT_DIFFICULTY_DIR" \
  --random-concept-control-dir "$RANDOM_CONTROL_DIR" \
  --slot-ablation-dir "$SLOT_ABLATION_DIR" \
  --claim-boundary-dir "$CLAIM_BOUNDARY_DIR" \
  --stage-report-audit-dir "$STAGE_REPORT_AUDIT_DIR" \
  --execution-status-dir "$EXECUTION_STATUS_DIR" \
  --out-dir "$PAPER_DIR"

echo "[INFO] refreshing requirement audit"
"$PY" scripts/build_semantic_entropy_requirement_audit.py \
  --strict-dir "$STRICT_DIR" \
  --extractor-validation-dir "$EXTRACTOR_VALIDATION_DIR" \
  --manual-progress-dir "$MANUAL_PROGRESS_DIR" \
  --manual-package-dir "$MANUAL_PACKAGE_DIR" \
  --manual-package-qa-dir "$MANUAL_PACKAGE_QA_DIR" \
  --manual-import-dir "$MANUAL_IMPORT_DIR" \
  --post-manual-smoke-dir "$POST_MANUAL_SMOKE_DIR" \
  --quadrant-dir "$QUADRANT_DIR" \
  --bootstrap-dir "$BOOTSTRAP_DIR" \
  --sample-size-dir "$SAMPLE_SIZE_DIR" \
  --option-order-full-dir "$OPTION_ORDER_FULL_DIR" \
  --prompt-template-full-dir "$PROMPT_TEMPLATE_FULL_DIR" \
  --robustness-preflight-dir "$ROBUSTNESS_PREFLIGHT_DIR" \
  --object-binding-dir "$OBJECT_BINDING_DIR" \
  --image-duplicate-dir "$IMAGE_DUPLICATE_DIR" \
  --claim-boundary-dir "$CLAIM_BOUNDARY_DIR" \
  --stage-report-audit-dir "$STAGE_REPORT_AUDIT_DIR" \
  --paper-dir "$PAPER_DIR" \
  --out-dir "$REQUIREMENT_AUDIT_DIR"

echo "[INFO] refreshing execution status"
"$PY" scripts/build_semantic_entropy_execution_status.py \
  --strict-dir "$STRICT_DIR" \
  --extractor-validation-dir "$EXTRACTOR_VALIDATION_DIR" \
  --manual-progress-dir "$MANUAL_PROGRESS_DIR" \
  --manual-package-dir "$MANUAL_PACKAGE_DIR" \
  --manual-package-qa-dir "$MANUAL_PACKAGE_QA_DIR" \
  --manual-import-dir "$MANUAL_IMPORT_DIR" \
  --post-manual-smoke-dir "$POST_MANUAL_SMOKE_DIR" \
  --quadrant-dir "$QUADRANT_DIR" \
  --bootstrap-dir "$BOOTSTRAP_DIR" \
  --sample-size-dir "$SAMPLE_SIZE_DIR" \
  --option-order-smoke-dir "$OPTION_ORDER_SMOKE_DIR" \
  --option-order-full-dir "$OPTION_ORDER_FULL_DIR" \
  --prompt-template-smoke-dir "$PROMPT_TEMPLATE_SMOKE_DIR" \
  --prompt-template-full-dir "$PROMPT_TEMPLATE_FULL_DIR" \
  --robustness-preflight-dir "$ROBUSTNESS_PREFLIGHT_DIR" \
  --object-binding-dir "$OBJECT_BINDING_DIR" \
  --image-duplicate-dir "$IMAGE_DUPLICATE_DIR" \
  --concept-difficulty-dir "$CONCEPT_DIFFICULTY_DIR" \
  --random-concept-control-dir "$RANDOM_CONTROL_DIR" \
  --slot-ablation-dir "$SLOT_ABLATION_DIR" \
  --claim-boundary-dir "$CLAIM_BOUNDARY_DIR" \
  --stage-report-audit-dir "$STAGE_REPORT_AUDIT_DIR" \
  --requirement-audit-dir "$REQUIREMENT_AUDIT_DIR" \
  --paper-dir "$PAPER_DIR" \
  --out-dir "$EXECUTION_STATUS_DIR"

echo "[INFO] post-manual pipeline complete"
