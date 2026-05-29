#!/usr/bin/env python3
# Copyright 2025 BAAI. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""Render raw and position-residual visual-token entropy heatmaps."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from PIL import Image, ImageDraw, ImageFont


BOI_TEXT = "<|image start|>"
IMG_TEXT = "<|image token|>"
EOI_TEXT = "<|image end|>"
EOL_TEXT = "<|extra_200|>"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", default="../research_logs/region_labeled_entropy_traces")
    parser.add_argument("--sample-index", default="../research_logs/real_ume_sample_index.csv")
    parser.add_argument("--out-dir", default="../research_logs/entropy_heatmaps")
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--metric", default="ume", choices=["ume", "u_tok", "u_intra", "u_cfg"])
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


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


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"true", "1", "yes"}


def as_float(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
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


def parse_hw(header_tokens: Sequence[str]) -> tuple[int | None, int | None]:
    header = "".join(header_tokens)
    match = re.search(r"(\d+)\*(\d+)", header)
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))


def iter_visual_positions(records: Sequence[Mapping[str, Any]], metric: str) -> Iterable[Dict[str, Any]]:
    """Yield visual-token rows with image-local grid coordinates.

    Some real X2I generations contain a complete decoded image followed by an
    incomplete second image attempt. Existing region traces keep cumulative
    visual_row/visual_col values, so this parser reconstructs image-local rows
    from structure tokens and keeps image_index explicit.
    """
    sample_id = str(records[0].get("sample_id", "")) if records else ""
    task = str(records[0].get("task", "unknown")) if records else "unknown"
    image_index = -1
    header_tokens: List[str] = []
    grid_h: int | None = None
    grid_w: int | None = None
    row = 0
    col = 0
    in_header = False
    in_visual = False

    for record in records:
        token_text = str(record.get("token_text", ""))
        if token_text == BOI_TEXT:
            image_index += 1
            header_tokens = []
            grid_h = None
            grid_w = None
            row = 0
            col = 0
            in_header = True
            in_visual = False
            continue

        if in_header:
            if token_text == IMG_TEXT:
                grid_h, grid_w = parse_hw(header_tokens)
                row = 0
                col = 0
                in_header = False
                in_visual = True
            elif token_text not in {EOI_TEXT, EOL_TEXT}:
                header_tokens.append(token_text)
            continue

        if not in_visual:
            continue

        if token_text == EOL_TEXT:
            row += 1
            col = 0
            continue
        if token_text == EOI_TEXT:
            in_visual = False
            continue

        if record.get("token_type") != "visual":
            continue
        yield {
            "task": task,
            "sample_id": sample_id,
            "image_index": image_index,
            "grid_h": grid_h,
            "grid_w": grid_w,
            "row": row,
            "col": col,
            "step": record.get("step", ""),
            "token_id": record.get("token_id", ""),
            "token_text": token_text,
            "metric": metric,
            "value": as_float(record.get(metric, 0.0)),
            "ume": as_float(record.get("ume", 0.0)),
            "u_tok": as_float(record.get("u_tok", 0.0)),
            "u_intra": as_float(record.get("u_intra", 0.0)),
            "u_cfg": as_float(record.get("u_cfg", 0.0)),
            "is_error": as_bool(record.get("is_error", False)),
            "manual_region_label": record.get("manual_region_label", ""),
        }
        col += 1


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def quantile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = round((len(ordered) - 1) * fraction)
    return ordered[max(0, min(len(ordered) - 1, idx))]


def collect_token_rows(trace_dir: Path, metric: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(trace_dir.glob("*_entropy.jsonl")):
        for row in iter_visual_positions(read_jsonl(path), metric):
            row["trace_file"] = str(path)
            rows.append(row)
    return rows


def add_position_residuals(rows: Sequence[Dict[str, Any]]) -> None:
    grouped: Dict[tuple[str, int, int, int, int], List[float]] = {}
    for row in rows:
        grid_h = row.get("grid_h")
        grid_w = row.get("grid_w")
        if grid_h is None or grid_w is None:
            continue
        key = (str(row["task"]), int(grid_h), int(grid_w), int(row["row"]), int(row["col"]))
        grouped.setdefault(key, []).append(float(row["value"]))

    baselines = {key: mean(values) for key, values in grouped.items()}
    for row in rows:
        grid_h = row.get("grid_h")
        grid_w = row.get("grid_w")
        if grid_h is None or grid_w is None:
            row["position_baseline"] = ""
            row["residual"] = ""
            continue
        key = (str(row["task"]), int(grid_h), int(grid_w), int(row["row"]), int(row["col"]))
        baseline = baselines.get(key, float(row["value"]))
        row["position_baseline"] = baseline
        row["residual"] = float(row["value"]) - baseline


def raw_color(value: float) -> tuple[int, int, int, int]:
    value = max(0.0, min(1.0, value))
    # Dark transparent -> amber -> red.
    if value < 0.5:
        t = value / 0.5
        r = int(255 * t)
        g = int(196 * t)
        b = 0
    else:
        t = (value - 0.5) / 0.5
        r = 255
        g = int(196 * (1.0 - t))
        b = 0
    alpha = int(35 + 145 * value)
    return r, g, b, alpha


def residual_color(value: float, scale: float) -> tuple[int, int, int, int]:
    if scale <= 1e-12:
        return 0, 0, 0, 0
    t = max(-1.0, min(1.0, value / scale))
    alpha = int(35 + 125 * abs(t))
    if t >= 0:
        return 255, int(80 * (1.0 - t)), 0, alpha
    return 0, int(128 * (1.0 + t)), 255, alpha


def draw_heat_overlay(
    image: Image.Image,
    rows: Sequence[Mapping[str, Any]],
    *,
    mode: str,
    residual_scale: float,
) -> Image.Image:
    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    grid_h = int(rows[0]["grid_h"])
    grid_w = int(rows[0]["grid_w"])
    cell_w = base.width / grid_w
    cell_h = base.height / grid_h
    for row in rows:
        r = int(row["row"])
        c = int(row["col"])
        if r < 0 or c < 0 or r >= grid_h or c >= grid_w:
            continue
        value = float(row["value"]) if mode == "raw" else float(row["residual"])
        color = raw_color(value) if mode == "raw" else residual_color(value, residual_scale)
        x0 = c * cell_w
        y0 = r * cell_h
        x1 = (c + 1) * cell_w
        y1 = (r + 1) * cell_h
        draw.rectangle([x0, y0, x1, y1], fill=color)
    return Image.alpha_composite(base, overlay).convert("RGB")


def label_panel(image: Image.Image, label: str) -> Image.Image:
    banner_h = 34
    out = Image.new("RGB", (image.width, image.height + banner_h), (250, 250, 250))
    out.paste(image, (0, banner_h))
    draw = ImageDraw.Draw(out)
    draw.rectangle([0, 0, image.width, banner_h], fill=(245, 245, 245))
    draw.text((10, 9), label, fill=(20, 20, 20), font=ImageFont.load_default())
    return out


def render_sample_heatmap(
    *,
    image_path: Path,
    rows: Sequence[Mapping[str, Any]],
    out_path: Path,
    metric: str,
) -> Dict[str, Any]:
    with Image.open(image_path) as image:
        base = image.convert("RGB")
    residual_values = [abs(float(row["residual"])) for row in rows if row.get("residual") != ""]
    residual_scale = quantile(residual_values, 0.95) or max(residual_values or [1.0])
    raw = draw_heat_overlay(base, rows, mode="raw", residual_scale=residual_scale)
    residual = draw_heat_overlay(base, rows, mode="residual", residual_scale=residual_scale)
    panels = [
        label_panel(base, "decoded image"),
        label_panel(raw, f"raw/process {metric}"),
        label_panel(residual, f"position-residual {metric}"),
    ]
    gap = 10
    width = sum(panel.width for panel in panels) + gap * (len(panels) - 1)
    height = max(panel.height for panel in panels)
    sheet = Image.new("RGB", (width, height), (255, 255, 255))
    x = 0
    for panel in panels:
        sheet.paste(panel, (x, 0))
        x += panel.width + gap
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    values = [float(row["value"]) for row in rows]
    residuals = [float(row["residual"]) for row in rows if row.get("residual") != ""]
    return {
        "heatmap_path": str(out_path),
        "source_image": str(image_path),
        "image_width": base.width,
        "image_height": base.height,
        "grid_rows": int(rows[0]["grid_h"]),
        "grid_cols": int(rows[0]["grid_w"]),
        "tokens": len(rows),
        "mean_raw": mean(values),
        "p90_raw": quantile(values, 0.9),
        "max_raw": max(values) if values else "",
        "mean_residual": mean(residuals),
        "p90_abs_residual": quantile([abs(v) for v in residuals], 0.9),
        "max_abs_residual": max([abs(v) for v in residuals]) if residuals else "",
    }


def build_report(rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# Entropy Heatmaps",
        "",
        "Each sheet contains the decoded image, raw/process visual-token entropy, and position-residual entropy.",
        "",
        "| task | sample | grid | tokens | mean raw | p90 raw | p90 abs residual | heatmap |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        rel = Path(str(row["heatmap_path"])).name
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["task"]),
                    str(row["sample_id"]),
                    f"{row['grid_rows']}x{row['grid_cols']}",
                    str(row["tokens"]),
                    f"{float(row['mean_raw']):.4f}",
                    f"{float(row['p90_raw']):.4f}",
                    f"{float(row['p90_abs_residual']):.4f}",
                    f"[{rel}]({rel})",
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def render_heatmaps(
    *,
    token_rows: Sequence[Dict[str, Any]],
    image_lookup: Mapping[tuple[str, str], Path],
    out_dir: Path,
    metric: str,
    max_samples: int | None,
) -> List[Dict[str, Any]]:
    by_sample: Dict[tuple[str, str], List[Dict[str, Any]]] = {}
    for row in token_rows:
        if int(row.get("image_index", -1)) != 0:
            continue
        if row.get("grid_h") is None or row.get("grid_w") is None:
            continue
        key = (str(row["task"]), str(row["sample_id"]))
        by_sample.setdefault(key, []).append(row)

    summary_rows: List[Dict[str, Any]] = []
    for idx, ((task, sample_id), rows) in enumerate(sorted(by_sample.items())):
        if max_samples is not None and idx >= max_samples:
            break
        image_path = image_lookup.get((task, sample_id))
        if image_path is None:
            continue
        grid_h = int(rows[0]["grid_h"])
        grid_w = int(rows[0]["grid_w"])
        complete_rows = [
            row
            for row in rows
            if 0 <= int(row["row"]) < grid_h and 0 <= int(row["col"]) < grid_w
        ]
        if len(complete_rows) < grid_h * grid_w:
            continue
        out_path = out_dir / f"{metric}__{task}__{sample_id}.png"
        info = render_sample_heatmap(image_path=image_path, rows=complete_rows, out_path=out_path, metric=metric)
        summary_rows.append({"task": task, "sample_id": sample_id, **info})
    return summary_rows


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root)
    trace_dir = Path(args.trace_dir)
    sample_index = Path(args.sample_index)
    out_dir = Path(args.out_dir)
    token_rows = collect_token_rows(trace_dir, args.metric)
    add_position_residuals(token_rows)
    write_csv(out_dir / f"{args.metric}_visual_token_entropy_residuals.csv", token_rows)
    summary_rows = render_heatmaps(
        token_rows=token_rows,
        image_lookup=decoded_image_lookup(sample_index, repo_root),
        out_dir=out_dir,
        metric=args.metric,
        max_samples=args.max_samples,
    )
    write_csv(out_dir / f"{args.metric}_entropy_heatmap_summary.csv", summary_rows)
    (out_dir / f"{args.metric}_entropy_heatmaps_report.md").write_text(build_report(summary_rows), encoding="utf-8")
    print(
        f"[INFO] wrote {len(summary_rows)} heatmaps and {len(token_rows)} token rows to {out_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
