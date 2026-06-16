#!/usr/bin/env python3
# Copyright 2025 BAAI. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""Render real image overlays for token-extreme UME case catalogs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from PIL import Image, ImageDraw


CATEGORIES = [
    ("high_ume", "high_ume_tokens.csv", "High default UME tokens", (255, 196, 0)),
    ("low_ume_error", "low_ume_error_tokens.csv", "Low-UME error tokens", (255, 68, 68)),
    ("high_cfg_error", "high_cfg_error_tokens.csv", "High-CFG error tokens", (0, 200, 255)),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-index", default="data/real_ume/real_ume_sample_index.csv")
    parser.add_argument("--catalog-dir", default="outputs/research_logs/token_extremes")
    parser.add_argument("--out-dir", default="outputs/research_logs/token_case_sheets")
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--samples-per-category", type=int, default=6)
    parser.add_argument("--tokens-per-sample", type=int, default=24)
    return parser.parse_args()


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"true", "1", "yes"}


def as_int(value: Any, default: int = 0) -> int:
    if value == "" or value is None:
        return default
    return int(float(value))


def as_float(value: Any, default: float = 0.0) -> float:
    if value == "" or value is None:
        return default
    return float(value)


def resolve_path(path_text: str, repo_root: Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else repo_root / path


def decoded_image_lookup(sample_index: Path, repo_root: Path) -> Dict[tuple[str, str], Path]:
    lookup: Dict[tuple[str, str], Path] = {}
    for row in read_csv(sample_index):
        if not as_bool(row.get("decoded", "")):
            continue
        decoded_image = row.get("decoded_image", "")
        if not decoded_image:
            continue
        path = resolve_path(decoded_image, repo_root)
        if not path.exists():
            continue
        key = (str(row.get("task", "")), str(row.get("sample_id", "")))
        lookup.setdefault(key, path)
    return lookup


def select_case_rows(rows: Sequence[Mapping[str, str]], samples_per_category: int) -> Dict[tuple[str, str], List[Mapping[str, str]]]:
    grouped: Dict[tuple[str, str], List[Mapping[str, str]]] = {}
    for row in rows:
        key = (str(row.get("task", "")), str(row.get("sample_id", "")))
        grouped.setdefault(key, []).append(row)
    ordered_keys = sorted(grouped, key=lambda key: min(as_int(row.get("rank", 999999)) for row in grouped[key]))
    return {key: grouped[key] for key in ordered_keys[:samples_per_category]}


def grid_shape(rows: Sequence[Mapping[str, Any]], image_width: int, image_height: int) -> tuple[int, int]:
    if image_width % 16 == 0 and image_height % 16 == 0:
        return image_height // 16, image_width // 16
    max_row = max(as_int(row.get("visual_row", 0)) for row in rows)
    max_col = max(as_int(row.get("visual_col", 0)) for row in rows)
    return max_row + 1, max_col + 1


def draw_overlay(
    image_path: Path,
    rows: Sequence[Mapping[str, str]],
    out_path: Path,
    *,
    color: tuple[int, int, int],
    max_tokens: int,
) -> Dict[str, Any]:
    with Image.open(image_path) as image:
        base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    selected = list(rows[:max_tokens])
    grid_h, grid_w = grid_shape(selected, base.width, base.height)
    cell_w = base.width / grid_w
    cell_h = base.height / grid_h
    for row in selected:
        r = as_int(row.get("visual_row", 0))
        c = as_int(row.get("visual_col", 0))
        x0 = c * cell_w
        y0 = r * cell_h
        x1 = (c + 1) * cell_w
        y1 = (r + 1) * cell_h
        draw.rectangle([x0, y0, x1, y1], outline=(*color, 255), width=3)
        draw.rectangle([x0, y0, x1, y1], fill=(*color, 45))
    composited = Image.alpha_composite(base, overlay).convert("RGB")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    composited.save(out_path)
    return {
        "overlay_path": str(out_path),
        "source_image": str(image_path),
        "image_width": base.width,
        "image_height": base.height,
        "grid_rows": grid_h,
        "grid_cols": grid_w,
        "highlighted_tokens": len(selected),
        "top_rank": min(as_int(row.get("rank", 999999)) for row in selected),
        "max_ume": max(as_float(row.get("ume", 0.0)) for row in selected),
        "max_u_cfg": max(as_float(row.get("u_cfg", 0.0)) for row in selected),
        "min_ume": min(as_float(row.get("ume", 0.0)) for row in selected),
    }


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


def fmt(value: Any, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def report_lines(rows: Sequence[Mapping[str, Any]]) -> List[str]:
    lines = [
        "# Token Extreme Case Sheets",
        "",
        "These overlays are rendered from real decoded Emu3.5 images and token-extreme rows. Highlighted cells mark visual token positions selected from the catalog CSVs.",
        "",
        "| category | task | sample | tokens | grid | min UME | max UME | max u_cfg | overlay |",
        "| --- | --- | --- | ---: | --- | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        rel = Path(str(row["overlay_path"])).name
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["category"]),
                    str(row["task"]),
                    str(row["sample_id"]),
                    str(row["highlighted_tokens"]),
                    f"{row['grid_rows']}x{row['grid_cols']}",
                    fmt(row["min_ume"]),
                    fmt(row["max_ume"]),
                    fmt(row["max_u_cfg"]),
                    f"[{rel}]({rel})",
                ]
            )
            + " |"
        )
    return lines


def build_case_sheets(
    *,
    sample_index: Path,
    catalog_dir: Path,
    out_dir: Path,
    repo_root: Path,
    samples_per_category: int,
    tokens_per_sample: int,
) -> List[Dict[str, Any]]:
    lookup = decoded_image_lookup(sample_index, repo_root)
    summary_rows: List[Dict[str, Any]] = []
    for category, file_name, title, color in CATEGORIES:
        rows = read_csv(catalog_dir / file_name)
        selected = select_case_rows(rows, samples_per_category)
        for (task, sample_id), sample_rows in selected.items():
            image_path = lookup.get((task, sample_id))
            if image_path is None:
                continue
            out_path = out_dir / f"{category}__{task}__{sample_id}.png"
            overlay_info = draw_overlay(image_path, sample_rows, out_path, color=color, max_tokens=tokens_per_sample)
            summary_rows.append(
                {
                    "category": category,
                    "category_title": title,
                    "task": task,
                    "sample_id": sample_id,
                    **overlay_info,
                }
            )
    write_csv(out_dir / "token_case_sheet_summary.csv", summary_rows)
    (out_dir / "token_case_sheets_report.md").write_text("\n".join(report_lines(summary_rows)) + "\n", encoding="utf-8")
    return summary_rows


def main() -> None:
    args = parse_args()
    rows = build_case_sheets(
        sample_index=Path(args.sample_index),
        catalog_dir=Path(args.catalog_dir),
        out_dir=Path(args.out_dir),
        repo_root=Path(args.repo_root),
        samples_per_category=args.samples_per_category,
        tokens_per_sample=args.tokens_per_sample,
    )
    print(f"[INFO] wrote {len(rows)} token case-sheet overlays to {args.out_dir}")


if __name__ == "__main__":
    main()
