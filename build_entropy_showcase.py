#!/usr/bin/env python3
"""Build entropy visualizations and token tables for synthetic concept benchmarks."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from datasets import load_from_disk
from PIL import Image


BENCH_DIR = Path("outputs/bench/modal_memory_lora_entropy_full")
TASK_DIR = Path("outputs/bench/task_entropy_compare_full")
DATASET_DIR = Path("data/synthetic_images_384")
OUT_DIR = BENCH_DIR / "entropy_showcase"
GRID_SIZE = 24
IMAGE_SIZE = 384
TOP_FRAC = 0.20


def load_jsonl(paths) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line:
                rows.append(json.loads(line))
    return rows


def entropy_grid(path: str | Path) -> np.ndarray:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    grid = np.full((GRID_SIZE, GRID_SIZE), np.nan, dtype=np.float32)
    for step in payload["steps"]:
        if step.get("phase") != "visual":
            continue
        idx = step.get("visual_index")
        if idx is None:
            continue
        idx = int(idx)
        if 0 <= idx < GRID_SIZE * GRID_SIZE:
            grid[idx // GRID_SIZE, idx % GRID_SIZE] = float(step["entropy"])
    return grid


def top_mask(grid: np.ndarray) -> np.ndarray:
    values = grid[np.isfinite(grid)]
    threshold = np.quantile(values, 1.0 - TOP_FRAC)
    return np.isfinite(grid) & (grid >= threshold)


def gt_attrs(row: dict[str, Any]) -> str:
    return f"{row['color']} {row['pattern']} {row['position']} {row['shape']}"


def detected_attrs(row: dict[str, Any]) -> str:
    return " ".join(str(row.get(f"grading_detected_{k}")) for k in ("color", "pattern", "position", "shape"))


def save_entropy_source_map(mean_grid: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(3.1, 3.1), dpi=220)
    ax.imshow(mean_grid, cmap="viridis")
    ax.text(0.02, 0.04, "Entropy", transform=ax.transAxes, color="white", fontsize=11, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout(pad=0.0)
    fig.savefig(path)
    plt.close(fig)


def save_sample_panel(row: dict[str, Any], gt_image: Image.Image, out_path: Path) -> None:
    generated = Image.open(row["image_path"]).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    gt_image = gt_image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    grid = entropy_grid(row["entropy_path"])
    top = top_mask(grid)
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.8), dpi=170)
    axes[0].imshow(gt_image)
    axes[0].set_title("GT image")
    axes[1].imshow(generated)
    axes[1].set_title("Generated")
    im = axes[2].imshow(grid, cmap="viridis")
    axes[2].text(0.02, 0.04, "Entropy", transform=axes[2].transAxes, color="white", fontsize=11, fontweight="bold")
    axes[2].set_title("Entropy grid")
    axes[3].imshow(generated)
    axes[3].imshow(top, cmap="Reds", alpha=0.48, extent=(0, IMAGE_SIZE, IMAGE_SIZE, 0), vmin=0, vmax=1)
    axes[3].set_title("Top 20% entropy")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    title = (
        f"{row['sample_id']} | prompt: {row['prompt']}\n"
        f"GT: {gt_attrs(row)} | detected: {detected_attrs(row)} | "
        f"correct: C={row['grading_correct_color']} P={row['grading_correct_pattern']} "
        f"Pos={row['grading_correct_position']} S={row['grading_correct_shape']}"
    )
    fig.suptitle(title, fontsize=9)
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def is_special_token(token: str) -> bool:
    return token.startswith("<|") and token.endswith("|>")


def add_text_entropy_rows(
    rows_out: list[dict[str, Any]],
    source: str,
    row: dict[str, Any],
    gt_text: str,
) -> None:
    entropy_path = row.get("entropy_path")
    if not entropy_path:
        return
    payload = json.loads(Path(entropy_path).read_text(encoding="utf-8"))
    word_index = 0
    for step in payload["steps"]:
        token = step.get("selected_token", "")
        if is_special_token(token):
            word_or_token = token
            is_special = True
        else:
            word_or_token = token.strip() or token
            is_special = False
            word_index += 1
        rows_out.append(
            {
                "source": source,
                "sample_id": row["sample_id"],
                "word_index": "" if is_special else word_index,
                "step": step["step"],
                "token": token,
                "word_or_token": word_or_token,
                "is_special_token": is_special,
                "entropy": step["entropy"],
                "normalized_entropy": step.get("normalized_entropy"),
                "finite_token_count": step.get("finite_token_count"),
                "gt": gt_text,
                "generated_completion": row.get("inference_completion", ""),
                "correct": row.get("grading_correct"),
                "expected_key": row.get("expected_key"),
                "answer_key": row.get("grading_answer_key"),
                "attribute": row.get("attribute", ""),
                "expected_value": row.get("expected_value", ""),
            }
        )


def save_text_entropy_html(rows: list[dict[str, Any]], path: Path, max_rows: int = 220) -> None:
    lines = [
        "<html><head><meta charset='utf-8'><style>",
        "body{font-family:Arial,sans-serif;margin:24px} table{border-collapse:collapse;font-size:12px}",
        "td,th{border:1px solid #ddd;padding:5px 7px} th{background:#f3f3f3} .bar{height:10px;background:#1f9fca}",
        "</style></head><body>",
        "<h2>Generated Text Token Entropy</h2>",
        "<p>Each row is one generated token. Special stop tokens are kept so the full generated sequence is visible.</p>",
        "<table><tr><th>source</th><th>sample</th><th>token</th><th>entropy</th><th>bar</th><th>GT</th><th>generated</th><th>correct</th></tr>",
    ]
    max_entropy = max(float(r["entropy"]) for r in rows[:max_rows]) if rows else 1.0
    for row in rows[:max_rows]:
        width = 120 * float(row["entropy"]) / max_entropy if max_entropy else 0
        lines.append(
            "<tr>"
            f"<td>{row['source']}</td><td>{row['sample_id']}</td><td><code>{row['word_or_token']}</code></td>"
            f"<td>{float(row['entropy']):.4f}</td><td><div class='bar' style='width:{width:.1f}px'></div></td>"
            f"<td>{row['gt']}</td><td><code>{row['generated_completion']}</code></td><td>{row['correct']}</td>"
            "</tr>"
        )
    lines.extend(["</table></body></html>"])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    panel_dir = OUT_DIR / "image_panels"
    gt_dir = OUT_DIR / "gt_images"
    panel_dir.mkdir(exist_ok=True)
    gt_dir.mkdir(exist_ok=True)

    dataset = load_from_disk(str(DATASET_DIR))
    image_rows = load_jsonl(sorted(BENCH_DIR.glob("image_memory_worker*.jsonl")))
    text_rows = load_jsonl(sorted(BENCH_DIR.glob("text_memory_worker*.jsonl")))
    understanding_rows = load_jsonl(sorted(TASK_DIR.glob("image_understanding_worker*.jsonl")))

    grids = [entropy_grid(row["entropy_path"]) for row in image_rows if row.get("entropy_path")]
    mean_grid = np.nanmean(np.stack(grids), axis=0)
    save_entropy_source_map(mean_grid, OUT_DIR / "entropy_source_map_mean.png")

    ranked = sorted(
        image_rows,
        key=lambda r: (float((r.get("entropy_summary") or {}).get("mean_visual_entropy") or 0.0), not bool(r.get("grading_all_correct"))),
        reverse=True,
    )
    selected = []
    seen = set()
    for row in ranked:
        key = row["sample_id"]
        if key in seen:
            continue
        selected.append(row)
        seen.add(key)
        if len(selected) >= 12:
            break
    for row in selected:
        gt_image = dataset[row["split"]][int(row["dataset_index"])]["image"]
        gt_path = gt_dir / f"{row['sample_id']}_gt.png"
        gt_image.save(gt_path)
        save_sample_panel(row, gt_image, panel_dir / f"{row['sample_id']}_entropy_panel.png")

    text_token_rows: list[dict[str, Any]] = []
    for row in text_rows:
        gt = f"expected option {row['expected_key']} ({row['concept_type']}: {row['concept_value']} / {row['concept_value_synthetic']})"
        add_text_entropy_rows(text_token_rows, "text_QA", row, gt)
    for row in understanding_rows:
        gt = f"{row['attribute']} = {row['expected_value']} | expected option {row['expected_key']}"
        add_text_entropy_rows(text_token_rows, "image_understanding", row, gt)

    fieldnames = list(text_token_rows[0].keys())
    with (OUT_DIR / "generated_text_token_entropy.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(text_token_rows)
    save_text_entropy_html(text_token_rows, OUT_DIR / "generated_text_token_entropy.html")

    manifest = {
        "output_dir": str(OUT_DIR),
        "entropy_source_map": str(OUT_DIR / "entropy_source_map_mean.png"),
        "image_panels": [str(panel_dir / f"{row['sample_id']}_entropy_panel.png") for row in selected],
        "gt_images": [str(gt_dir / f"{row['sample_id']}_gt.png") for row in selected],
        "generated_text_token_entropy_csv": str(OUT_DIR / "generated_text_token_entropy.csv"),
        "generated_text_token_entropy_html": str(OUT_DIR / "generated_text_token_entropy.html"),
        "num_text_token_rows": len(text_token_rows),
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
