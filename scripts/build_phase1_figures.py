#!/usr/bin/env python3
# Copyright 2025 BAAI. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""Build dependency-free SVG figures from real phase-1 UME result tables."""

from __future__ import annotations

import argparse
import csv
import html
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


REGION_AUC_KEYS = [
    ("region_token", "all", "ume_default"),
    ("region_token", "all", "inverse_ume_default"),
    ("region_token", "all", "u_cfg"),
    ("region_token", "all", "inverse_u_tok"),
    ("region_token", "t2i", "inverse_best_t2i_mix"),
    ("region_token", "x2i", "u_cfg"),
]

SAMPLE_AUC_KEYS = [
    ("sample", "all", "inverse_p90_best_all_mix"),
    ("sample", "t2i", "inverse_p90_u_cfg"),
    ("sample", "x2i", "inverse_mean_ume_default"),
]

FIGURE_SPECS = [
    ("region_auc_summary.svg", "Region-token AUC summary with bootstrap CIs"),
    ("sample_auc_summary.svg", "Sample-level AUC summary with bootstrap CIs"),
    ("task_error_ume_bars.svg", "Mean UME in manually labeled error vs correct regions"),
    ("calibration_bins.svg", "Default UME calibration bins"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", default="outputs/research_logs")
    parser.add_argument("--out-dir", default="outputs/research_logs/phase1_figures")
    return parser.parse_args()


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def as_float(value: Any, default: float = 0.0) -> float:
    if value == "" or value is None:
        return default
    return float(value)


def fmt(value: Any, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def by_key(rows: Iterable[Mapping[str, str]], *keys: str) -> Dict[tuple[str, ...], Mapping[str, str]]:
    return {tuple(str(row[key]) for key in keys): row for row in rows}


def select_rows(rows: Sequence[Mapping[str, str]], keys: Sequence[tuple[str, str, str]]) -> List[Mapping[str, str]]:
    lookup = by_key(rows, "level", "group", "metric")
    return [lookup[key] for key in keys if key in lookup]


def metric_label(row: Mapping[str, str]) -> str:
    metric = row["metric"]
    replacements = {
        "ume_default": "default UME",
        "inverse_ume_default": "inverse default UME",
        "inverse_u_tok": "inverse u_tok",
        "inverse_best_t2i_mix": "inverse best T2I mix",
        "inverse_p90_best_all_mix": "inverse p90 best mix",
        "inverse_p90_u_cfg": "inverse p90 u_cfg",
        "inverse_mean_ume_default": "inverse mean UME",
    }
    return f"{row['group']} / {replacements.get(metric, metric)}"


def svg_document(width: int, height: int, body: Sequence[str]) -> str:
    style = """
<style>
  text { font-family: Arial, Helvetica, sans-serif; fill: #172026; }
  .title { font-size: 20px; font-weight: 700; }
  .subtitle { font-size: 12px; fill: #5b6770; }
  .axis { stroke: #69737b; stroke-width: 1; }
  .grid { stroke: #d8dee3; stroke-width: 1; }
  .tick { font-size: 11px; fill: #5b6770; }
  .label { font-size: 12px; }
  .small { font-size: 10px; fill: #5b6770; }
  .bar-a { fill: #2f7d79; }
  .bar-b { fill: #c36b3f; }
  .bar-c { fill: #4f6fad; }
  .ci { stroke: #25313a; stroke-width: 2; }
  .ref { stroke: #8b949e; stroke-width: 1.5; stroke-dasharray: 5 5; }
</style>""".strip()
    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">',
            style,
            *body,
            "</svg>",
            "",
        ]
    )


def x_scale(value: float, left: float, width: float, lo: float = 0.0, hi: float = 1.0) -> float:
    value = max(lo, min(hi, value))
    return left + (value - lo) / (hi - lo) * width


def auc_bar_svg(rows: Sequence[Mapping[str, str]], title: str, subtitle: str, out_path: Path) -> None:
    left = 245
    top = 82
    bar_h = 20
    gap = 24
    chart_w = 460
    row_h = bar_h + gap
    height = top + len(rows) * row_h + 78
    width = 780
    body: List[str] = [
        f'<text class="title" x="24" y="34">{esc(title)}</text>',
        f'<text class="subtitle" x="24" y="55">{esc(subtitle)}</text>',
    ]

    for tick in [0.0, 0.25, 0.5, 0.75, 1.0]:
        x = x_scale(tick, left, chart_w)
        cls = "ref" if tick == 0.5 else "grid"
        body.append(f'<line class="{cls}" x1="{x:.1f}" y1="{top - 18}" x2="{x:.1f}" y2="{height - 62}"/>')
        body.append(f'<text class="tick" x="{x - 10:.1f}" y="{height - 40}">{tick:.2f}</text>')

    body.append(f'<line class="axis" x1="{left}" y1="{height - 62}" x2="{left + chart_w}" y2="{height - 62}"/>')
    body.append(f'<text class="small" x="{left + chart_w - 70}" y="{height - 17}">AUC vs error</text>')

    for i, row in enumerate(rows):
        y = top + i * row_h
        auc = as_float(row["auc"])
        lo = as_float(row.get("bootstrap_ci_low", auc), auc)
        hi = as_float(row.get("bootstrap_ci_high", auc), auc)
        bar_w = x_scale(auc, left, chart_w) - left
        ci_x1 = x_scale(lo, left, chart_w)
        ci_x2 = x_scale(hi, left, chart_w)
        ci_y = y + bar_h / 2
        bar_cls = "bar-a" if auc >= 0.5 else "bar-b"
        body.extend(
            [
                f'<text class="label" x="24" y="{y + 14}">{esc(metric_label(row))}</text>',
                f'<rect class="{bar_cls}" x="{left}" y="{y}" width="{bar_w:.1f}" height="{bar_h}" rx="2"/>',
                f'<line class="ci" x1="{ci_x1:.1f}" y1="{ci_y:.1f}" x2="{ci_x2:.1f}" y2="{ci_y:.1f}"/>',
                f'<line class="ci" x1="{ci_x1:.1f}" y1="{ci_y - 5:.1f}" x2="{ci_x1:.1f}" y2="{ci_y + 5:.1f}"/>',
                f'<line class="ci" x1="{ci_x2:.1f}" y1="{ci_y - 5:.1f}" x2="{ci_x2:.1f}" y2="{ci_y + 5:.1f}"/>',
                f'<text class="small" x="{left + chart_w + 12}" y="{y + 14}">{fmt(auc)} [{fmt(lo)}, {fmt(hi)}]</text>',
            ]
        )

    out_path.write_text(svg_document(width, height, body), encoding="utf-8")


def task_error_ume_svg(rows: Sequence[Mapping[str, str]], out_path: Path) -> None:
    width = 720
    height = 380
    left = 88
    top = 74
    chart_w = 500
    chart_h = 230
    max_v = max([as_float(row["mean_ume_error"]) for row in rows] + [as_float(row["mean_ume_correct"]) for row in rows] + [0.35])
    max_v = min(1.0, max_v * 1.25)

    def y_scale(value: float) -> float:
        return top + chart_h - value / max_v * chart_h

    body: List[str] = [
        '<text class="title" x="24" y="34">Mean UME by manual region label</text>',
        '<text class="subtitle" x="24" y="55">Real region-labeled tokens only; lower error UME indicates false-confidence behavior.</text>',
    ]
    for tick in [0.0, max_v / 2, max_v]:
        y = y_scale(tick)
        body.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{left + chart_w}" y2="{y:.1f}"/>')
        body.append(f'<text class="tick" x="42" y="{y + 4:.1f}">{fmt(tick)}</text>')
    body.append(f'<line class="axis" x1="{left}" y1="{top + chart_h}" x2="{left + chart_w}" y2="{top + chart_h}"/>')
    body.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_h}"/>')

    group_w = chart_w / max(1, len(rows))
    bar_w = 58
    for i, row in enumerate(rows):
        cx = left + group_w * i + group_w / 2
        err = as_float(row["mean_ume_error"])
        ok = as_float(row["mean_ume_correct"])
        for x, value, cls, label in [(cx - bar_w - 5, err, "bar-b", "error"), (cx + 5, ok, "bar-a", "correct")]:
            y = y_scale(value)
            body.append(f'<rect class="{cls}" x="{x:.1f}" y="{y:.1f}" width="{bar_w}" height="{top + chart_h - y:.1f}" rx="2"/>')
            body.append(f'<text class="small" x="{x + 8:.1f}" y="{y - 6:.1f}">{fmt(value)}</text>')
            body.append(f'<text class="small" x="{x + 8:.1f}" y="{top + chart_h + 36:.1f}">{label}</text>')
        body.append(f'<text class="label" text-anchor="middle" x="{cx:.1f}" y="{top + chart_h + 18:.1f}">{esc(row["task"])}</text>')

    body.extend(
        [
            f'<rect class="bar-b" x="{left + chart_w + 26}" y="{top + 8}" width="14" height="14"/>',
            f'<text class="small" x="{left + chart_w + 48}" y="{top + 20}">error region</text>',
            f'<rect class="bar-a" x="{left + chart_w + 26}" y="{top + 32}" width="14" height="14"/>',
            f'<text class="small" x="{left + chart_w + 48}" y="{top + 44}">correct region</text>',
        ]
    )
    out_path.write_text(svg_document(width, height, body), encoding="utf-8")


def calibration_svg(rows: Sequence[Mapping[str, str]], out_path: Path) -> None:
    width = 760
    height = 400
    left = 74
    top = 74
    chart_w = 560
    chart_h = 235
    max_v = max([as_float(row["error_rate"]) for row in rows] + [as_float(row["mean_ume"]) for row in rows] + [0.55])

    def y_scale(value: float) -> float:
        return top + chart_h - value / max_v * chart_h

    body: List[str] = [
        '<text class="title" x="24" y="34">Calibration of default UME bins</text>',
        '<text class="subtitle" x="24" y="55">Bars show observed error rate; line shows mean UME in each bin.</text>',
    ]
    for tick in [0.0, max_v / 2, max_v]:
        y = y_scale(tick)
        body.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{left + chart_w}" y2="{y:.1f}"/>')
        body.append(f'<text class="tick" x="28" y="{y + 4:.1f}">{fmt(tick)}</text>')
    body.append(f'<line class="axis" x1="{left}" y1="{top + chart_h}" x2="{left + chart_w}" y2="{top + chart_h}"/>')
    body.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_h}"/>')

    bin_w = chart_w / max(1, len(rows))
    points: List[tuple[float, float]] = []
    for i, row in enumerate(rows):
        x = left + i * bin_w + 14
        bar_w = max(18, bin_w - 28)
        error_rate = as_float(row["error_rate"])
        mean_ume = as_float(row["mean_ume"])
        y = y_scale(error_rate)
        body.append(f'<rect class="bar-c" x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{top + chart_h - y:.1f}" rx="2"/>')
        body.append(f'<text class="small" text-anchor="middle" x="{x + bar_w / 2:.1f}" y="{top + chart_h + 18}">{esc(row["ume_lower"])}-{esc(row["ume_upper"])}</text>')
        body.append(f'<text class="small" text-anchor="middle" x="{x + bar_w / 2:.1f}" y="{y - 6:.1f}">{fmt(error_rate)}</text>')
        points.append((x + bar_w / 2, y_scale(mean_ume)))

    for (x1, y1), (x2, y2) in zip(points, points[1:]):
        body.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="#c36b3f" stroke-width="3"/>')
    for x, y in points:
        body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="#c36b3f"/>')

    body.extend(
        [
            f'<rect class="bar-c" x="{left + chart_w + 24}" y="{top + 8}" width="14" height="14"/>',
            f'<text class="small" x="{left + chart_w + 45}" y="{top + 20}">error rate</text>',
            f'<line x1="{left + chart_w + 24}" y1="{top + 40}" x2="{left + chart_w + 38}" y2="{top + 40}" stroke="#c36b3f" stroke-width="3"/>',
            f'<text class="small" x="{left + chart_w + 45}" y="{top + 44}">mean UME</text>',
        ]
    )
    out_path.write_text(svg_document(width, height, body), encoding="utf-8")


def build_figures(log_dir: Path, out_dir: Path) -> List[Path]:
    uncertainty = read_csv(log_dir / "auc_uncertainty" / "auc_uncertainty.csv")
    hallucination = read_csv(log_dir / "region_labeled_analysis" / "tables" / "hallucination_diagnostics.csv")
    calibration = read_csv(log_dir / "region_labeled_analysis" / "tables" / "calibration_bins.csv")
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs = [
        out_dir / "region_auc_summary.svg",
        out_dir / "sample_auc_summary.svg",
        out_dir / "task_error_ume_bars.svg",
        out_dir / "calibration_bins.svg",
    ]
    auc_bar_svg(
        select_rows(uncertainty, REGION_AUC_KEYS),
        "Region-token AUC summary",
        "Bootstrap CIs from real manually region-labeled Emu3.5 traces; dashed line is random AUC=0.5.",
        outputs[0],
    )
    auc_bar_svg(
        select_rows(uncertainty, SAMPLE_AUC_KEYS),
        "Sample-level AUC summary",
        "Exploratory sample-level metrics from real manual labels; X2I remains small-N.",
        outputs[1],
    )
    task_error_ume_svg(hallucination, outputs[2])
    calibration_svg(calibration, outputs[3])
    return outputs


def main() -> None:
    args = parse_args()
    outputs = build_figures(Path(args.log_dir), Path(args.out_dir))
    for path in outputs:
        print(f"[INFO] wrote {path}")


if __name__ == "__main__":
    main()
