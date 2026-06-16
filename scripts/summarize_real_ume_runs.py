#!/usr/bin/env python3
# Copyright 2025 BAAI. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""Summarize real UME trace runs from local experiment outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from PIL import Image


MARKERS = (
    "<|image start|>",
    "<|image token|>",
    "<|extra_200|>",
    "<|image end|>",
    "<|extra_101|>",
    "<|extra_204|>",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="outputs/emu3p5-image")
    parser.add_argument("--out-dir", default="outputs/research_logs")
    parser.add_argument("--run-id-contains", default=None)
    return parser.parse_args()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def quantile(values: Sequence[float], fraction: float) -> float | str:
    if not values:
        return ""
    ordered = sorted(values)
    idx = round((len(ordered) - 1) * fraction)
    return ordered[max(0, min(len(ordered) - 1, idx))]


def mean(values: Sequence[float]) -> float | str:
    if not values:
        return ""
    return sum(values) / len(values)


def count_token_text(records: Iterable[Mapping[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in records:
        text = str(row.get("token_text", ""))
        counts[text] = counts.get(text, 0) + 1
    return counts


def boundary_window_stats(records: Sequence[Mapping[str, Any]], window: int = 3) -> Dict[str, Any]:
    if not records:
        return {
            "boundary_window": window,
            "boundary_events": 0,
            "boundary_window_tokens": 0,
            "boundary_mean_u_mod": "",
            "boundary_max_u_mod": "",
        }
    boundary_steps = [
        int(row.get("step", idx))
        for idx, row in enumerate(records)
        if bool(row.get("is_boundary")) or str(row.get("token_type")) == "structure"
    ]
    selected = set()
    max_idx = len(records) - 1
    for step in boundary_steps:
        for idx in range(max(0, step - window), min(max_idx, step + window) + 1):
            selected.add(idx)
    values = [float(records[idx].get("u_mod", 0.0)) for idx in sorted(selected)]
    return {
        "boundary_window": window,
        "boundary_events": len(boundary_steps),
        "boundary_window_tokens": len(values),
        "boundary_mean_u_mod": mean(values),
        "boundary_max_u_mod": max(values) if values else "",
    }


def image_metadata(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"decoded_image": "", "decoded_width": "", "decoded_height": "", "decoded_mode": "", "decoded_bytes": ""}
    with Image.open(path) as image:
        width, height = image.size
        mode = image.mode
    return {
        "decoded_image": str(path),
        "decoded_width": width,
        "decoded_height": height,
        "decoded_mode": mode,
        "decoded_bytes": path.stat().st_size,
    }


def find_run_dirs(root: Path) -> List[Path]:
    return sorted(path.parent for path in root.glob("*/ume_trace_runs/*/entropy_traces") if path.is_dir())


def sample_row(run_dir: Path, trace_path: Path, root: Path) -> Dict[str, Any]:
    records = read_jsonl(trace_path)
    sample_id = trace_path.name.replace("_entropy.jsonl", "")
    task = str(records[0].get("task", run_dir.parent.parent.name if records else "unknown"))
    token_counts: Dict[str, int] = {}
    segment_counts: Dict[str, int] = {}
    for record in records:
        token_type = str(record.get("token_type", "unknown"))
        segment = str(record.get("segment", "unknown"))
        token_counts[token_type] = token_counts.get(token_type, 0) + 1
        segment_counts[segment] = segment_counts.get(segment, 0) + 1

    visual_umes = [
        float(record["ume"])
        for record in records
        if record.get("token_type") == "visual" and record.get("ume") is not None
    ]
    visual_cfgs = [
        float(record.get("u_cfg", 0.0))
        for record in records
        if record.get("token_type") == "visual"
    ]
    all_umes = [float(record["ume"]) for record in records if record.get("ume") is not None]
    marker_counts = count_token_text(records)
    terminal = records[-1] if records else {}
    boundary_stats = boundary_window_stats(records)
    terminal_text = str(terminal.get("token_text", ""))
    if marker_counts.get("<|extra_204|>", 0) >= 1:
        terminal_state = "eos"
    elif terminal_text == "<|image end|>":
        terminal_state = "stopped_after_image_end"
    else:
        terminal_state = f"ended_in_{terminal.get('segment', 'unknown')}"

    decoded_path = run_dir / "decoded" / f"{sample_id}_image_00.png"
    raw_path = run_dir / "raw_generations" / f"{sample_id}.txt"
    raw_counts = {marker: "" for marker in MARKERS}
    if raw_path.exists():
        raw_text = raw_path.read_text(encoding="utf-8", errors="replace")
        raw_counts = {marker: raw_text.count(marker) for marker in MARKERS}

    image_meta = image_metadata(decoded_path)
    return {
        "task": task,
        "run_id": run_dir.name,
        "sample_id": sample_id,
        "trace_path": str(trace_path),
        "decoded": bool(decoded_path.exists()),
        "first_image_complete": marker_counts.get("<|image end|>", 0) >= 1,
        "image_complete": marker_counts.get("<|image end|>", 0) >= 1,
        "completed_images": marker_counts.get("<|image end|>", 0),
        "sequence_eos": marker_counts.get("<|extra_204|>", 0) >= 1,
        "eos": marker_counts.get("<|extra_204|>", 0) >= 1,
        "terminal_step": terminal.get("step", ""),
        "terminal_token_text": terminal.get("token_text", ""),
        "terminal_token_type": terminal.get("token_type", ""),
        "terminal_segment": terminal.get("segment", ""),
        "terminal_state": terminal_state,
        "stop_after_completed_images": records[0].get("stop_after_completed_images", "") if records else "",
        "stop_after_eoi_extra_tokens": records[0].get("stop_after_eoi_extra_tokens", "") if records else "",
        "tokens": len(records),
        "visual_tokens": token_counts.get("visual", 0),
        "structure_tokens": token_counts.get("structure", 0),
        "text_tokens": token_counts.get("text", 0),
        "visual_segment_tokens": segment_counts.get("visual", 0),
        "mean_ume": mean(all_umes),
        "max_ume": max(all_umes) if all_umes else "",
        "visual_mean_ume": mean(visual_umes),
        "visual_p50_ume": quantile(visual_umes, 0.5),
        "visual_p90_ume": quantile(visual_umes, 0.9),
        "visual_max_ume": max(visual_umes) if visual_umes else "",
        "visual_mean_u_cfg": mean(visual_cfgs),
        "visual_max_u_cfg": max(visual_cfgs) if visual_cfgs else "",
        "image_start_count": marker_counts.get("<|image start|>", 0),
        "image_token_count": marker_counts.get("<|image token|>", 0),
        "eol_count": marker_counts.get("<|extra_200|>", 0),
        "image_end_count": marker_counts.get("<|image end|>", 0),
        "ess_count": marker_counts.get("<|extra_101|>", 0),
        "eos_count": marker_counts.get("<|extra_204|>", 0),
        "raw_image_start_count": raw_counts["<|image start|>"],
        "raw_image_end_count": raw_counts["<|image end|>"],
        "raw_eol_count": raw_counts["<|extra_200|>"],
        "raw_eos_count": raw_counts["<|extra_204|>"],
        **boundary_stats,
        **image_meta,
    }


def run_row(run_dir: Path, sample_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    visual_means = [float(row["visual_mean_ume"]) for row in sample_rows if row["visual_mean_ume"] != ""]
    visual_p90s = [float(row["visual_p90_ume"]) for row in sample_rows if row["visual_p90_ume"] != ""]
    decoded_count = sum(1 for row in sample_rows if row["decoded"])
    first_image_complete_count = sum(1 for row in sample_rows if row["first_image_complete"])
    eos_count = sum(1 for row in sample_rows if row["sequence_eos"])
    manual_notes = run_dir / "manual_visual_notes.md"
    labeled_analysis = run_dir / "labeled_analysis" / "tables" / "hallucination_diagnostics.csv"
    return {
        "task": sample_rows[0]["task"] if sample_rows else run_dir.parent.parent.name,
        "run_id": run_dir.name,
        "samples": len(sample_rows),
        "decoded_samples": decoded_count,
        "first_image_complete_samples": first_image_complete_count,
        "image_complete_samples": first_image_complete_count,
        "eos_samples": eos_count,
        "tokens": sum(int(row["tokens"]) for row in sample_rows),
        "visual_tokens": sum(int(row["visual_tokens"]) for row in sample_rows),
        "mean_sample_visual_mean_ume": mean(visual_means),
        "mean_sample_visual_p90_ume": mean(visual_p90s),
        "max_visual_mean_ume": max(visual_means) if visual_means else "",
        "min_visual_mean_ume": min(visual_means) if visual_means else "",
        "manual_notes": str(manual_notes) if manual_notes.exists() else "",
        "has_labeled_analysis": labeled_analysis.exists() and labeled_analysis.stat().st_size > 0,
    }


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    headers = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_markdown(
    path: Path,
    run_rows: Sequence[Mapping[str, Any]],
    sample_rows: Sequence[Mapping[str, Any]],
    root: Path,
) -> None:
    lines = [
        "# Real UME Run Index",
        "",
        f"This report is generated only from local real run trace files under `{root}/*/ume_trace_runs/`.",
        "Synthetic smoke tests are intentionally excluded unless they are present in that real-output tree.",
        "",
        "## Aggregate",
        "",
    ]
    by_task: Dict[str, List[Mapping[str, Any]]] = {}
    for row in sample_rows:
        by_task.setdefault(str(row["task"]), []).append(row)
    lines.extend(["| task | runs | samples | decoded | complete images | visual tokens | mean sample visual UME | mean sample visual p90 |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for task, rows in sorted(by_task.items()):
        run_count = len({row["run_id"] for row in rows})
        visual_means = [float(row["visual_mean_ume"]) for row in rows if row["visual_mean_ume"] != ""]
        visual_p90s = [float(row["visual_p90_ume"]) for row in rows if row["visual_p90_ume"] != ""]
        lines.append(
            "| "
            + " | ".join(
                [
                    task,
                    str(run_count),
                    str(len(rows)),
                    str(sum(1 for row in rows if row["decoded"])),
                    str(sum(1 for row in rows if row["first_image_complete"])),
                    str(sum(int(row["visual_tokens"]) for row in rows)),
                    fmt(mean(visual_means)),
                    fmt(mean(visual_p90s)),
                ]
            )
            + " |"
        )

    lines.extend(["", "## Runs", ""])
    lines.extend(["| task | run_id | samples | decoded | first image complete | sequence eos | visual tokens | mean visual UME | mean visual p90 | notes | labeled |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |"])
    for row in run_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["task"]),
                    str(row["run_id"]),
                    str(row["samples"]),
                    str(row["decoded_samples"]),
                    str(row["first_image_complete_samples"]),
                    str(row["eos_samples"]),
                    str(row["visual_tokens"]),
                    fmt(row["mean_sample_visual_mean_ume"]),
                    fmt(row["mean_sample_visual_p90_ume"]),
                    "yes" if row["manual_notes"] else "",
                    "yes" if row["has_labeled_analysis"] else "",
                ]
            )
            + " |"
        )

    lines.extend(["", "## Samples", ""])
    lines.extend(["| task | run_id | sample_id | decoded | first image | eos | visual tokens | visual mean UME | visual p90 | max UME | terminal state | boundary mean U_mod | decoded size |", "| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | --- |"])
    for row in sample_rows:
        size = ""
        if row["decoded_width"] != "":
            size = f"{row['decoded_width']}x{row['decoded_height']}"
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["task"]),
                    str(row["run_id"]),
                    str(row["sample_id"]),
                    "yes" if row["decoded"] else "",
                    "yes" if row["first_image_complete"] else "",
                    "yes" if row["sequence_eos"] else "",
                    str(row["visual_tokens"]),
                    fmt(row["visual_mean_ume"]),
                    fmt(row["visual_p90_ume"]),
                    fmt(row["visual_max_ume"]),
                    str(row["terminal_state"]),
                    fmt(row["boundary_mean_u_mod"]),
                    size,
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    out_dir = Path(args.out_dir)
    run_rows: List[Dict[str, Any]] = []
    sample_rows: List[Dict[str, Any]] = []

    for run_dir in find_run_dirs(root):
        if args.run_id_contains and args.run_id_contains not in run_dir.name:
            continue
        rows = [sample_row(run_dir, trace_path, root) for trace_path in sorted((run_dir / "entropy_traces").glob("*_entropy.jsonl"))]
        if not rows:
            continue
        sample_rows.extend(rows)
        run_rows.append(run_row(run_dir, rows))

    write_csv(out_dir / "real_ume_run_index.csv", run_rows)
    write_csv(out_dir / "real_ume_sample_index.csv", sample_rows)
    write_markdown(out_dir / "real_ume_run_index.md", run_rows, sample_rows, root)
    print(f"[INFO] wrote {len(run_rows)} run rows and {len(sample_rows)} sample rows to {out_dir}")


if __name__ == "__main__":
    main()
