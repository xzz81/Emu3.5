#!/usr/bin/env python3
"""Validate and score alternative extractor outputs for semantic entropy."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


SLOTS = ["object_1", "color_1", "object_2", "color_2", "relation", "background"]
VOCAB = {
    "object_1": ["cube", "sphere", "cone", "unknown"],
    "object_2": ["cube", "sphere", "cone", "unknown"],
    "color_1": ["red", "blue", "green", "yellow", "unknown"],
    "color_2": ["red", "blue", "green", "yellow", "unknown"],
    "relation": [
        "object_1_left_of_object_2",
        "object_1_right_of_object_2",
        "object_1_above_object_2",
        "object_1_below_object_2",
        "unknown",
    ],
    "background": ["white", "other", "unknown"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-csv", default="outputs/semantic_entropy_umm/extractor_validation/annotation_manifest.csv")
    parser.add_argument("--outputs-csv", default="outputs/semantic_entropy_umm/alternative_extractor_feasibility/alternative_extractor_outputs_template.csv")
    parser.add_argument("--out-dir", default="outputs/semantic_entropy_umm/alternative_extractor_feasibility")
    parser.add_argument("--allow-incomplete", action="store_true")
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


def normalize(text: str) -> str:
    value = text.lower().strip()
    value = re.sub(r"[^a-z0-9_ ]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value.replace(" ", "_")


def normalize_slot(slot: str, value: str, raw_value: str) -> tuple[str, str]:
    response = value.strip() or raw_value.strip()
    if not response:
        return "", "empty_answer"
    normalized = normalize(response)
    allowed = {item: item for item in VOCAB[slot]}
    allowed.update({normalize(item): item for item in VOCAB[slot]})
    if normalized in allowed:
        return allowed[normalized], ""
    if slot.startswith("object"):
        synonyms = {
            "cube": ["cube", "box", "block", "square"],
            "sphere": ["sphere", "circle", "ball", "round"],
            "cone": ["cone", "triangle", "pyramid"],
        }
        hits = [label for label, words in synonyms.items() if any(word in normalized for word in words)]
        if len(hits) == 1:
            return hits[0], ""
    if slot.startswith("color"):
        hits = [label for label in VOCAB[slot] if label != "unknown" and label in normalized]
        if len(hits) == 1:
            return hits[0], ""
    if slot == "relation":
        relation_phrases = {
            "left": "object_1_left_of_object_2",
            "right": "object_1_right_of_object_2",
            "above": "object_1_above_object_2",
            "below": "object_1_below_object_2",
        }
        hits = [label for word, label in relation_phrases.items() if word in normalized]
        if len(hits) == 1:
            return hits[0], ""
    if slot == "background" and "white" in normalized:
        return "white", ""
    if "unknown" in normalized and "unknown" in VOCAB[slot]:
        return "unknown", ""
    return "", "illegal_value"


def index_by_id(rows: list[dict]) -> tuple[dict[str, dict], list[str]]:
    indexed = {}
    duplicates = []
    for row in rows:
        annotation_id = row.get("annotation_id", "")
        if annotation_id in indexed:
            duplicates.append(annotation_id)
        indexed[annotation_id] = row
    return indexed, duplicates


def macro_f1(labels: list[str], gold: list[str], pred: list[str]) -> float:
    scores = []
    for label in labels:
        tp = sum(g == label and p == label for g, p in zip(gold, pred))
        fp = sum(g != label and p == label for g, p in zip(gold, pred))
        fn = sum(g == label and p != label for g, p in zip(gold, pred))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append((2 * precision * recall / (precision + recall)) if precision + recall else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def metric_rows(joined: list[dict]) -> list[dict]:
    rows = []
    for slot in SLOTS:
        gold = [row[f"gold_{slot}"] for row in joined]
        pred = [row[f"alt_{slot}"] for row in joined]
        rows.extend(
            [
                {
                    "metric": "slot_accuracy",
                    "route": "ALL",
                    "slot": slot,
                    "value": sum(g == p for g, p in zip(gold, pred)) / len(gold),
                    "n": len(gold),
                },
                {
                    "metric": "macro_f1",
                    "route": "ALL",
                    "slot": slot,
                    "value": macro_f1(VOCAB[slot], gold, pred),
                    "n": len(gold),
                },
                {
                    "metric": "unknown_agreement",
                    "route": "ALL",
                    "slot": slot,
                    "value": sum((g == "unknown") == (p == "unknown") for g, p in zip(gold, pred)) / len(gold),
                    "n": len(gold),
                },
            ]
        )
        for route in ["I2T", "T2I"]:
            route_rows = [row for row in joined if row["route"] == route]
            if route_rows:
                rows.append(
                    {
                        "metric": "route_slot_accuracy",
                        "route": route,
                        "slot": slot,
                        "value": sum(row[f"{slot}_match"] == "True" for row in route_rows) / len(route_rows),
                        "n": len(route_rows),
                    }
                )
    for route in ["ALL", "I2T", "T2I"]:
        route_rows = joined if route == "ALL" else [row for row in joined if row["route"] == route]
        if not route_rows:
            continue
        rows.extend(
            [
                {
                    "metric": "relation_accuracy",
                    "route": route,
                    "slot": "relation",
                    "value": sum(row["relation_match"] == "True" for row in route_rows) / len(route_rows),
                    "n": len(route_rows),
                },
                {
                    "metric": "object_binding_accuracy",
                    "route": route,
                    "slot": "object_1+object_2",
                    "value": sum(row["object_1_match"] == "True" and row["object_2_match"] == "True" for row in route_rows)
                    / len(route_rows),
                    "n": len(route_rows),
                },
                {
                    "metric": "color_object_binding_accuracy",
                    "route": route,
                    "slot": "color_1+color_2",
                    "value": sum(row["color_1_match"] == "True" and row["color_2_match"] == "True" for row in route_rows)
                    / len(route_rows),
                    "n": len(route_rows),
                },
            ]
        )
    return rows


def md_table(rows: list[dict]) -> list[str]:
    if not rows:
        return ["(no rows)"]
    fields = list(rows[0].keys())
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    return lines


def write_report(out_dir: Path, report: dict, metrics: list[dict], issues: list[dict], args: argparse.Namespace) -> None:
    lines = [
        "# Alternative Extractor Score Report",
        "",
        f"Status: {report['status']}",
        "",
        "This report validates and scores a filled alternative-extractor output CSV against the fixed extractor-validation manifest. It does not run the extractor itself.",
        "",
        "## Input Files",
        "",
        f"- Manifest: `{args.manifest_csv}`",
        f"- Outputs: `{args.outputs_csv}`",
        "",
        "## Summary",
        "",
        f"- Manifest rows: `{report['manifest_rows']}`",
        f"- Output rows: `{report['output_rows']}`",
        f"- Completed rows: `{report['completed_rows']}` / `{report['manifest_rows']}`",
        f"- Completed slot labels: `{report['completed_slot_labels']}` / `{report['required_slot_labels']}`",
        f"- Missing output IDs: `{report['missing_output_ids']}`",
        f"- Unknown output IDs: `{report['unknown_output_ids']}`",
        f"- Duplicate output IDs: `{report['duplicate_output_ids']}`",
        f"- Illegal values: `{report['illegal_values']}`",
        "",
        "## Metrics",
        "",
        *md_table(metrics[:40]),
        "",
        "## Issues Preview",
        "",
        *md_table(issues[:40]),
        "",
        "## Claim Boundary",
        "",
        "Allowed now: validation/scoring status for a fixed alternative-extractor output file.",
        "",
        "Still not allowed: alternative-extractor robustness unless this report is PASS from a real approved extractor run.",
    ]
    (out_dir / "alternative_extractor_score_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    manifest_rows = read_csv(Path(args.manifest_csv))
    output_rows = read_csv(Path(args.outputs_csv))
    manifest, duplicate_manifest = index_by_id(manifest_rows)
    outputs, duplicate_outputs = index_by_id(output_rows)
    expected_ids = [row["annotation_id"] for row in manifest_rows]
    missing_ids = [annotation_id for annotation_id in expected_ids if annotation_id not in outputs]
    unknown_ids = sorted(set(outputs) - set(expected_ids))

    joined = []
    issues = []
    completed_rows = 0
    completed_slots = 0
    illegal_values = 0
    for annotation_id in expected_ids:
        meta = manifest[annotation_id]
        output = outputs.get(annotation_id, {})
        row_complete = True
        joined_row = {
            "annotation_id": annotation_id,
            "concept_id": meta["concept_id"],
            "route": meta["route"],
            "sample_id": meta["sample_id"],
            "image_path": meta["image_path"],
            "extractor_backend": output.get("extractor_backend", ""),
            "model_id_or_endpoint": output.get("model_id_or_endpoint", ""),
        }
        for slot in SLOTS:
            gold = meta[f"gold_{slot}"]
            alt_value, issue = normalize_slot(slot, output.get(slot, ""), output.get(f"raw_{slot}", ""))
            if issue:
                row_complete = False
                issues.append({"annotation_id": annotation_id, "slot": slot, "issue": issue, "value": output.get(slot, ""), "raw_value": output.get(f"raw_{slot}", "")})
                if issue == "illegal_value":
                    illegal_values += 1
            else:
                completed_slots += 1
            joined_row[f"gold_{slot}"] = gold
            joined_row[f"alt_{slot}"] = alt_value
            joined_row[f"raw_{slot}"] = output.get(f"raw_{slot}", "")
            joined_row[f"{slot}_match"] = str(alt_value == gold and not issue)
        if row_complete:
            completed_rows += 1
        joined.append(joined_row)
    for annotation_id in missing_ids:
        issues.append({"annotation_id": annotation_id, "slot": "", "issue": "missing_output_row", "value": "", "raw_value": ""})
    for annotation_id in unknown_ids:
        issues.append({"annotation_id": annotation_id, "slot": "", "issue": "unknown_output_id", "value": "", "raw_value": ""})
    for annotation_id in duplicate_outputs:
        issues.append({"annotation_id": annotation_id, "slot": "", "issue": "duplicate_output_id", "value": "", "raw_value": ""})
    for annotation_id in duplicate_manifest:
        issues.append({"annotation_id": annotation_id, "slot": "", "issue": "duplicate_manifest_id", "value": "", "raw_value": ""})

    complete = (
        completed_slots == len(manifest_rows) * len(SLOTS)
        and completed_rows == len(manifest_rows)
        and not missing_ids
        and not unknown_ids
        and not duplicate_outputs
        and not duplicate_manifest
        and illegal_values == 0
    )
    metrics = metric_rows(joined) if complete else []
    status = "PASS" if complete else "FAIL_INCOMPLETE"
    report = {
        "status": status,
        "manifest_rows": len(manifest_rows),
        "output_rows": len(output_rows),
        "completed_rows": completed_rows,
        "completed_slot_labels": completed_slots,
        "required_slot_labels": len(manifest_rows) * len(SLOTS),
        "missing_output_ids": len(missing_ids),
        "unknown_output_ids": len(unknown_ids),
        "duplicate_output_ids": len(duplicate_outputs),
        "duplicate_manifest_ids": len(duplicate_manifest),
        "illegal_values": illegal_values,
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    joined_fields = [
        "annotation_id",
        "concept_id",
        "route",
        "sample_id",
        "image_path",
        "extractor_backend",
        "model_id_or_endpoint",
        *[f"gold_{slot}" for slot in SLOTS],
        *[f"alt_{slot}" for slot in SLOTS],
        *[f"raw_{slot}" for slot in SLOTS],
        *[f"{slot}_match" for slot in SLOTS],
    ]
    write_csv(out_dir / "alternative_extractor_joined_outputs.csv", joined, joined_fields)
    write_csv(out_dir / "alternative_extractor_metrics.csv", metrics, ["metric", "route", "slot", "value", "n"])
    (out_dir / "alternative_extractor_score_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_report(out_dir, report, metrics, issues, args)
    print(json.dumps(report, indent=2))
    if status != "PASS" and not args.allow_incomplete:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
