#!/usr/bin/env python3
# Copyright 2025 BAAI. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""Build a reproducibility manifest for real UME research artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


KEY_ARTIFACTS = [
    "phase1_findings_report.md",
    "ume_phase1_log.md",
    "real_ume_run_index.csv",
    "real_ume_sample_index.csv",
    "manual_sample_judgments.jsonl",
    "manual_sample_judgment_summary.csv",
    "corpus_audit/task_coverage.csv",
    "phase1_coverage/phase1_coverage_plan.md",
    "phase1_coverage/phase1_coverage_progress.csv",
    "region_labeled_analysis/tables/hallucination_diagnostics.csv",
    "auc_uncertainty/auc_uncertainty.csv",
    "auc_robustness/auc_robustness_summary.csv",
    "prompt_cluster_effects/prompt_cluster_auc_summary.csv",
    "token_extremes/token_extreme_thresholds.csv",
    "boundary_entropy/boundary_sample_auc.csv",
    "error_type_strata/error_type_strata_summary.csv",
    "phase1_figures/region_auc_summary.svg",
    "phase1_figures/sample_auc_summary.svg",
    "phase1_figures/task_error_ume_bars.svg",
    "phase1_figures/calibration_bins.svg",
]

MODEL_CACHE_PATTERNS = [
    "models--BAAI--Emu3.5-Image",
    "models--BAAI--Emu3.5-VisionTokenizer",
    "models--BAAI--Emu3.5",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--log-dir", default="../research_logs")
    parser.add_argument("--out-dir", default="../research_logs/reproducibility")
    parser.add_argument("--hf-cache", default=str(Path.home() / ".cache" / "huggingface" / "hub"))
    return parser.parse_args()


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_rows(log_dir: Path, rel_paths: Sequence[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for rel_path in rel_paths:
        path = log_dir / rel_path
        rows.append(
            {
                "rel_path": rel_path,
                "exists": path.exists(),
                "bytes": path.stat().st_size if path.exists() else "",
                "sha256": sha256_file(path) if path.exists() and path.is_file() else "",
            }
        )
    return rows


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"true", "1", "yes"}
    return bool(value)


def corpus_snapshot(log_dir: Path) -> Dict[str, Any]:
    sample_rows = read_csv(log_dir / "real_ume_sample_index.csv")
    labels = read_jsonl(log_dir / "manual_sample_judgments.jsonl")
    task_coverage = read_csv(log_dir / "corpus_audit" / "task_coverage.csv")
    region_dir = log_dir / "region_labeled_entropy_traces"
    region_files = sorted(region_dir.glob("*_entropy.jsonl"))
    trace_records = 0
    visual_tokens = 0
    error_tokens = 0
    for path in region_files:
        for row in read_jsonl(path):
            trace_records += 1
            if row.get("token_type") == "visual":
                visual_tokens += 1
                if as_bool(row.get("is_error", False)):
                    error_tokens += 1
    binary_labels = [row for row in labels if row.get("include_in_analysis") is True and row.get("is_error") != ""]
    decoded = sum(1 for row in sample_rows if as_bool(row.get("decoded", "")))
    complete = sum(1 for row in sample_rows if as_bool(row.get("image_complete", "")))
    return {
        "sample_rows": len(sample_rows),
        "decoded_samples": decoded,
        "complete_images": complete,
        "manual_label_rows": len(labels),
        "binary_manual_labels": len(binary_labels),
        "manual_error_labels": sum(1 for row in binary_labels if as_bool(row.get("is_error", False))),
        "region_labeled_trace_files": len(region_files),
        "region_trace_records": trace_records,
        "region_visual_tokens": visual_tokens,
        "region_error_tokens": error_tokens,
        "task_coverage": task_coverage,
    }


def run_command(args: Sequence[str], cwd: Path) -> str:
    try:
        return subprocess.check_output(args, cwd=str(cwd), text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"ERROR: {exc}"


def environment_snapshot(repo_root: Path, hf_cache: Path) -> Dict[str, Any]:
    env: Dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "git_status_short": run_command(["git", "status", "--short"], repo_root),
        "git_head": run_command(["git", "rev-parse", "HEAD"], repo_root),
    }
    try:
        import torch

        env.update(
            {
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
                "cuda_available": torch.cuda.is_available(),
                "gpu_count": torch.cuda.device_count(),
                "gpus": [torch.cuda.get_device_name(idx) for idx in range(torch.cuda.device_count())],
            }
        )
    except Exception as exc:
        env["torch_error"] = repr(exc)
    try:
        import transformers

        env["transformers_version"] = transformers.__version__
    except Exception as exc:
        env["transformers_error"] = repr(exc)
    try:
        import flash_attn

        env["flash_attn_version"] = getattr(flash_attn, "__version__", "unknown")
    except Exception as exc:
        env["flash_attn_error"] = repr(exc)
    env["hf_cache"] = str(hf_cache)
    env["hf_model_cache"] = {
        pattern: (hf_cache / pattern).exists()
        for pattern in MODEL_CACHE_PATTERNS
    }
    return env


def validation_rows(log_dir: Path, corpus: Mapping[str, Any], artifacts: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    missing_artifacts = [row["rel_path"] for row in artifacts if not row["exists"]]
    checks = [
        ("has_real_sample_rows", corpus["sample_rows"] > 0, corpus["sample_rows"]),
        ("has_binary_manual_labels", corpus["binary_manual_labels"] > 0, corpus["binary_manual_labels"]),
        ("has_region_labeled_traces", corpus["region_labeled_trace_files"] > 0, corpus["region_labeled_trace_files"]),
        ("has_region_visual_tokens", corpus["region_visual_tokens"] > 0, corpus["region_visual_tokens"]),
        ("no_missing_key_artifacts", not missing_artifacts, ";".join(missing_artifacts)),
        ("synthetic_not_counted_in_phase1_report", "Synthetic fixtures are used only for code regression tests" in (log_dir / "phase1_findings_report.md").read_text(encoding="utf-8"), ""),
    ]
    return [{"check": name, "passed": bool(passed), "detail": detail} for name, passed, detail in checks]


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    headers: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in headers:
                headers.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def fmt_bool(value: Any) -> str:
    return "yes" if bool(value) else "no"


def build_report(
    manifest: Mapping[str, Any],
    artifact_rows_: Sequence[Mapping[str, Any]],
    validation_rows_: Sequence[Mapping[str, Any]],
) -> str:
    corpus = manifest["corpus"]
    env = manifest["environment"]
    lines = [
        "# Reproducibility Manifest",
        "",
        "This manifest records the local environment and real research artifacts used by the current UME phase-1 reports.",
        "Synthetic fixtures are only code tests and are not counted as research evidence.",
        "",
        "## Environment",
        "",
        f"- Generated UTC: `{env.get('generated_at_utc', '')}`",
        f"- Python: `{env.get('python_executable', '')}` (`{env.get('python_version', '')}`)",
        f"- Torch/CUDA: `{env.get('torch_version', '')}` / `{env.get('cuda_version', '')}`",
        f"- GPUs: `{env.get('gpu_count', '')}` " + ", ".join(f"`{gpu}`" for gpu in env.get("gpus", [])),
        f"- Transformers: `{env.get('transformers_version', '')}`",
        f"- flash_attn: `{env.get('flash_attn_version', env.get('flash_attn_error', ''))}`",
        "",
        "## Corpus Snapshot",
        "",
        f"- Sample rows: `{corpus['sample_rows']}`",
        f"- Decoded samples: `{corpus['decoded_samples']}`",
        f"- Complete images: `{corpus['complete_images']}`",
        f"- Manual label rows: `{corpus['manual_label_rows']}`",
        f"- Binary manual labels: `{corpus['binary_manual_labels']}`",
        f"- Manual error labels: `{corpus['manual_error_labels']}`",
        f"- Region-labeled trace files: `{corpus['region_labeled_trace_files']}`",
        f"- Region visual tokens: `{corpus['region_visual_tokens']}`",
        f"- Region error tokens: `{corpus['region_error_tokens']}`",
        "",
        "## Validation",
        "",
        "| check | passed | detail |",
        "| --- | --- | --- |",
    ]
    for row in validation_rows_:
        lines.append(f"| {row['check']} | {fmt_bool(row['passed'])} | {row['detail']} |")
    lines.extend(["", "## Artifact Hashes", "", "| artifact | exists | bytes | sha256 |", "| --- | --- | ---: | --- |"])
    for row in artifact_rows_:
        lines.append(f"| `{row['rel_path']}` | {fmt_bool(row['exists'])} | {row['bytes']} | `{row['sha256']}` |")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    log_dir = Path(args.log_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    hf_cache = Path(args.hf_cache).expanduser()

    artifacts = artifact_rows(log_dir, KEY_ARTIFACTS)
    corpus = corpus_snapshot(log_dir)
    environment = environment_snapshot(repo_root, hf_cache)
    validations = validation_rows(log_dir, corpus, artifacts)
    manifest = {
        "environment": environment,
        "corpus": corpus,
        "artifact_hashes": artifacts,
        "validations": validations,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "reproducibility_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_csv(out_dir / "artifact_hashes.csv", artifacts)
    write_csv(out_dir / "validation_checks.csv", validations)
    (out_dir / "reproducibility_manifest.md").write_text(build_report(manifest, artifacts, validations), encoding="utf-8")
    failed = [row for row in validations if not row["passed"]]
    print(f"[INFO] wrote reproducibility manifest to {out_dir}")
    print(f"[INFO] validation checks: {len(validations) - len(failed)}/{len(validations)} passed")


if __name__ == "__main__":
    main()
