#!/usr/bin/env python3
"""Prepare an alternative-extractor feasibility audit and fixed run contract.

This script does not call external services, download models, or run inference.
It records what local extractor backends appear available, freezes the image
set to the extractor-validation manifest, and writes output templates for a
future alternative extractor run.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path


SLOTS = ["object_1", "color_1", "object_2", "color_2", "relation", "background"]
MODULES = [
    "torch",
    "transformers",
    "accelerate",
    "qwen_vl_utils",
    "decord",
    "sentencepiece",
    "protobuf",
    "PIL",
    "openai",
    "anthropic",
    "google.genai",
]
ENV_KEYS = ["OPENAI_API_KEY", "DASHSCOPE_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY"]
MODEL_PATTERNS = ["qwen", "internvl", "llava", "blip", "minicpm", "idefics", "paligemma"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extractor-validation-dir", default="outputs/semantic_entropy_umm/extractor_validation")
    parser.add_argument("--out-dir", default="outputs/semantic_entropy_umm/alternative_extractor_feasibility")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or (list(rows[0].keys()) if rows else [])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def module_status() -> dict[str, bool]:
    status = {}
    for module in MODULES:
        try:
            status[module] = importlib.util.find_spec(module) is not None
        except ModuleNotFoundError:
            status[module] = False
    return status


def env_status() -> dict[str, bool]:
    return {key: bool(os.environ.get(key)) for key in ENV_KEYS}


def model_candidates() -> list[dict]:
    roots = [Path(".cache/huggingface"), Path("model")]
    candidates = []
    seen = set()
    for root in roots:
        if not root.exists():
            continue
        for path in root.glob("**/*"):
            if not path.is_dir():
                continue
            name = path.name.lower()
            if not any(pattern in name for pattern in MODEL_PATTERNS):
                continue
            if str(path) in seen:
                continue
            seen.add(str(path))
            snapshots = list((path / "snapshots").glob("*")) if (path / "snapshots").exists() else [path]
            complete_snapshots = []
            for snapshot in snapshots:
                has_config = (snapshot / "config.json").exists()
                has_weights = bool(list(snapshot.glob("*.safetensors"))) or bool(list(snapshot.glob("pytorch_model*.bin")))
                has_processor = any(
                    (snapshot / filename).exists()
                    for filename in ["preprocessor_config.json", "processor_config.json", "tokenizer_config.json"]
                )
                if has_config and has_weights and has_processor:
                    complete_snapshots.append(str(snapshot))
            candidates.append(
                {
                    "model_hint": path.name,
                    "path": str(path),
                    "complete_snapshots": len(complete_snapshots),
                    "is_complete": bool(complete_snapshots),
                    "complete_snapshot_paths": ";".join(complete_snapshots[:3]),
                }
            )
    return candidates


def build_input_manifest(rows: list[dict]) -> list[dict]:
    out = []
    for row in rows:
        out.append(
            {
                "annotation_id": row["annotation_id"],
                "concept_id": row["concept_id"],
                "route": row["route"],
                "sample_id": row["sample_id"],
                "image_path": row["image_path"],
                "canonical_prompt": row["canonical_prompt"],
                "question_object_1": "Which shape is object_1, the first object named in the target description?",
                "question_color_1": f"Which color is object_1 ({row['gold_object_1']})?",
                "question_object_2": "Which shape is object_2, the second object named in the target description?",
                "question_color_2": f"Which color is object_2 ({row['gold_object_2']})?",
                "question_relation": "What is the spatial relation from object_1 to object_2?",
                "question_background": "What is the image background color category?",
                "allowed_object_values": "cube;sphere;cone;unknown",
                "allowed_color_values": "red;blue;green;yellow;unknown",
                "allowed_relation_values": "object_1_left_of_object_2;object_1_right_of_object_2;object_1_above_object_2;object_1_below_object_2;unknown",
                "allowed_background_values": "white;other;unknown",
            }
        )
    return out


def build_output_template(rows: list[dict]) -> list[dict]:
    out = []
    for row in rows:
        base = {
            "annotation_id": row["annotation_id"],
            "extractor_backend": "",
            "model_id_or_endpoint": "",
        }
        for slot in SLOTS:
            base[slot] = ""
            base[f"raw_{slot}"] = ""
        base["notes"] = ""
        out.append(base)
    return out


def write_report(out_dir: Path, report: dict, candidates: list[dict], args: argparse.Namespace) -> None:
    status = report["status"]
    lines = [
        "# Alternative Extractor Feasibility",
        "",
        f"Status: {status}",
        "",
        "This audit checks whether an alternative local or already configured extractor is available. It does not download models, call external APIs, or run inference.",
        "",
        "## Input Files",
        "",
        f"- Extractor validation manifest: `{Path(args.extractor_validation_dir) / 'annotation_manifest.csv'}`",
        "",
        "## Commands",
        "",
        "```bash",
        "./.venv-transformers/bin/python scripts/build_semantic_entropy_alternative_extractor_feasibility.py \\",
        f"  --extractor-validation-dir {args.extractor_validation_dir} \\",
        f"  --out-dir {args.out_dir}",
        "```",
        "",
        "## Output Files",
        "",
        f"- `{out_dir / 'alternative_extractor_feasibility.json'}`",
        f"- `{out_dir / 'alternative_extractor_candidates.csv'}`",
        f"- `{out_dir / 'alternative_extractor_input_manifest.csv'}`",
        f"- `{out_dir / 'alternative_extractor_outputs_template.csv'}`",
        f"- `{out_dir / 'alternative_extractor_score_report.md'}` after scoring",
        f"- `{out_dir / 'alternative_extractor_runbook.md'}`",
        "",
        "## Sample Counts",
        "",
        f"- Manifest rows: `{report['manifest_rows']}`",
        f"- Candidate rows: `{len(candidates)}`",
        f"- Complete local alternative extractor candidates: `{report['complete_local_candidates']}`",
        f"- External API credentials configured: `{report['external_api_credentials_configured']}`",
        "",
        "## Pass/Fail Checks",
        "",
        f"- Feasibility status: `{status}`",
        f"- No side effects: `{report['side_effects']}`",
        f"- Complete local candidate available: `{report['complete_local_candidates'] > 0}`",
        f"- External service requires approval: `{report['external_api_credentials_configured']}`",
        "",
        "## Summary",
        "",
        f"- Manifest rows: `{report['manifest_rows']}`",
        f"- Complete local alternative extractor candidates: `{report['complete_local_candidates']}`",
        f"- External API credentials configured: `{report['external_api_credentials_configured']}`",
        f"- Missing critical local modules: `{report['missing_critical_modules']}`",
        "",
        "## Candidate Models",
        "",
        "| model_hint | path | is_complete | complete_snapshots |",
        "| --- | --- | --- | ---: |",
    ]
    for row in candidates:
        lines.append(
            f"| {row['model_hint']} | {row['path']} | {row['is_complete']} | {row['complete_snapshots']} |"
        )
    if not candidates:
        lines.append("| NA | NA | False | 0 |")
    lines.extend(
        [
            "",
            "## Run Contract",
            "",
            "Only run an alternative extractor after selecting a complete local model or explicitly approving an external API, cost, credentials, and output format. Use `alternative_extractor_input_manifest.csv` as the fixed 60-image input set, and write outputs to `alternative_extractor_outputs_template.csv` fields.",
            "",
            "Required output fields per row:",
            "",
            "`annotation_id, extractor_backend, model_id_or_endpoint, object_1, color_1, object_2, color_2, relation, background, raw_object_1, raw_color_1, raw_object_2, raw_color_2, raw_relation, raw_background, notes`",
            "",
            "Scoring must use the same fixed vocabulary and compare against the same annotation manifest or manual labels after manual validation passes.",
            "",
            "Validate a blank or filled output file with:",
            "",
            "```bash",
            "PY=./.venv-transformers/bin/python",
            "$PY scripts/score_semantic_entropy_alternative_extractor.py \\",
            f"  --manifest-csv {Path(args.extractor_validation_dir) / 'annotation_manifest.csv'} \\",
            f"  --outputs-csv {out_dir / 'alternative_extractor_outputs_template.csv'} \\",
            f"  --out-dir {out_dir} \\",
            "  --allow-incomplete",
            "```",
            "",
            "The scorer only writes accuracy metrics when all 60 rows and 360 slot labels are present and legal.",
            "",
            "## Claim Allowed After This Step",
            "",
            "Allowed now: alternative-extractor feasibility and a fixed future run contract are documented.",
            "",
            "## Claim Still Not Allowed",
            "",
            "Do not claim alternative-extractor robustness. No complete local alternative model or configured external service was used in this audit.",
        ]
    )
    (out_dir / "alternative_extractor_runbook.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    validation_dir = Path(args.extractor_validation_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = read_csv(validation_dir / "annotation_manifest.csv")
    modules = module_status()
    env = env_status()
    candidates = model_candidates()
    complete_candidates = [row for row in candidates if row["is_complete"]]
    missing_critical = [
        module
        for module in ["qwen_vl_utils", "decord", "sentencepiece", "protobuf"]
        if not modules.get(module)
    ]
    external_configured = any(env.values())
    status = (
        "READY_LOCAL_ALTERNATIVE_AVAILABLE"
        if complete_candidates
        else "BLOCKED_NO_LOCAL_ALTERNATIVE_EXTRACTOR"
    )
    if external_configured and not complete_candidates:
        status = "BLOCKED_EXTERNAL_EXTRACTOR_REQUIRES_APPROVAL"
    report = {
        "status": status,
        "manifest_rows": len(manifest),
        "module_status": modules,
        "env_credentials_present": env,
        "candidate_count": len(candidates),
        "complete_local_candidates": len(complete_candidates),
        "external_api_credentials_configured": external_configured,
        "missing_critical_modules": missing_critical,
        "side_effects": "none; no model download, no external API call, no inference",
    }
    input_manifest = build_input_manifest(manifest)
    output_template = build_output_template(manifest)
    (out_dir / "alternative_extractor_feasibility.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_csv(out_dir / "alternative_extractor_candidates.csv", candidates)
    write_csv(out_dir / "alternative_extractor_input_manifest.csv", input_manifest)
    write_csv(out_dir / "alternative_extractor_outputs_template.csv", output_template)
    write_report(out_dir, report, candidates, args)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
