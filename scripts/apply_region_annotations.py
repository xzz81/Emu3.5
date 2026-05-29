#!/usr/bin/env python3
# Copyright 2025 BAAI. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""Apply manual visual-token region annotations to UME trace JSONL files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, help="UME run directory containing entropy_traces/ and manual_annotations/.")
    parser.add_argument("--annotation-dir", default=None)
    parser.add_argument("--out-dir", default=None)
    return parser.parse_args()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: List[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_regions(annotation: Mapping[str, Any]) -> List[Dict[str, Any]]:
    regions = []
    if "regions" in annotation:
        for idx, region in enumerate(annotation["regions"]):
            bbox = region["token_bbox"]
            regions.append(
                {
                    "region_id": region.get("region_id", f"region_{idx}"),
                    "region_label": region.get("region_label", region.get("label", "")),
                    "is_error": bool(region.get("is_error", False)),
                    "error_type": region.get("error_type", annotation.get("failure_type", "")),
                    "bbox": bbox,
                }
            )
    if "error_region_token_bbox" in annotation:
        regions.append(
            {
                "region_id": "error_region",
                "region_label": annotation.get("failure_type", "manual_error_region"),
                "is_error": True,
                "error_type": annotation.get("failure_type", "manual_error"),
                "bbox": annotation["error_region_token_bbox"],
            }
        )
    return regions


def add_visual_coordinates(rows: List[Dict[str, Any]], grid_width: int) -> None:
    visual_index = 0
    for row in rows:
        if row.get("token_type") == "visual":
            row["visual_row"] = visual_index // grid_width
            row["visual_col"] = visual_index % grid_width
            visual_index += 1
        elif row.get("segment") == "visual":
            row["visual_row"] = visual_index // grid_width if grid_width else None
            row["visual_col"] = None


def in_bbox(row: Mapping[str, Any], bbox: Mapping[str, int]) -> bool:
    if row.get("token_type") != "visual":
        return False
    visual_row = row.get("visual_row")
    visual_col = row.get("visual_col")
    if visual_row is None or visual_col is None:
        return False
    return (
        int(bbox["row_min"]) <= int(visual_row) <= int(bbox["row_max"])
        and int(bbox["col_min"]) <= int(visual_col) <= int(bbox["col_max"])
    )


def apply_annotation(run_dir: Path, annotation_path: Path, out_dir: Path) -> int:
    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    sample_id = annotation["sample_id"]
    trace_path = run_dir / "entropy_traces" / f"{sample_id}_entropy.jsonl"
    if not trace_path.exists():
        raise FileNotFoundError(f"missing trace for annotation {annotation_path}: {trace_path}")
    rows = read_jsonl(trace_path)
    token_grid = annotation["token_grid"]
    add_visual_coordinates(rows, int(token_grid["width"]))
    regions = normalize_regions(annotation)
    source = str(annotation_path.relative_to(run_dir)) if annotation_path.is_relative_to(run_dir) else str(annotation_path)

    for row in rows:
        row["is_error"] = False
        row["error_type"] = ""
        row["manual_region_label"] = ""
        row["manual_region_id"] = ""
        row["manual_label_source"] = source
        region_hits = []
        for region in regions:
            if in_bbox(row, region["bbox"]):
                region_hits.append(region)
        if not region_hits:
            continue
        row["manual_region_id"] = ";".join(str(region["region_id"]) for region in region_hits)
        row["manual_region_label"] = ";".join(str(region["region_label"]) for region in region_hits)
        if any(region["is_error"] for region in region_hits):
            row["is_error"] = True
            row["error_type"] = ";".join(
                sorted({str(region["error_type"]) for region in region_hits if region["is_error"] and region["error_type"]})
            )

    write_jsonl(out_dir / f"{sample_id}_entropy.jsonl", rows)
    return len(rows)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    annotation_dir = Path(args.annotation_dir) if args.annotation_dir else run_dir / "manual_annotations"
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "entropy_traces_labeled_regions"
    total = 0
    count = 0
    for annotation_path in sorted(annotation_dir.glob("*_manual_annotation.json")):
        total += apply_annotation(run_dir, annotation_path, out_dir)
        count += 1
    print(f"[INFO] applied {count} annotations and wrote {total} records to {out_dir}")


if __name__ == "__main__":
    main()
