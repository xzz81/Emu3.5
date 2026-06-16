#!/usr/bin/env python3
"""Build contact sheets for Codex visual GT first-pass annotation."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


RUNS = {
    "formal30": Path("outputs/semantic_entropy_umm/semantic_uncertainty_formal_emu35_20260606_095911"),
    "bench20": Path("outputs/semantic_entropy_umm/t2i_bench20_emu35_20260608"),
}
OUT = Path("outputs/semantic_entropy_umm/codex_visual_gt_first_pass")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def wrap_text(text: str, width: int = 90) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    for word in words:
        if current and sum(map(len, current)) + len(current) + len(word) > width:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))
    return lines


def main() -> None:
    sheet_root = OUT / "contact_sheets"
    sheet_root.mkdir(parents=True, exist_ok=True)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
        small = ImageFont.truetype("DejaVuSans.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
        small = font

    index_rows: list[dict] = []
    manifest_rows: list[dict] = []

    for run_name, run_dir in RUNS.items():
        concepts = read_jsonl(run_dir / "pilot/concepts.jsonl")
        samples = read_jsonl(run_dir / "pilot/t2i_samples.jsonl")
        by_concept: dict[str, list[dict]] = {}
        for sample in samples:
            if sample.get("image_path"):
                by_concept.setdefault(str(sample["concept_id"]), []).append(sample)

        for concept in concepts:
            concept_id = str(concept["concept_id"])
            rows = sorted(by_concept.get(concept_id, []), key=lambda row: int(row["sample_id"]))
            if not rows:
                continue

            cols = 5
            thumb = 210
            label_h = 54
            header_h = 100
            row_count = math.ceil(len(rows) / cols)
            canvas = Image.new("RGB", (cols * thumb, header_h + row_count * (thumb + label_h)), "white")
            draw = ImageDraw.Draw(canvas)
            prompt = concept.get("prompt", "")
            target = concept.get("target_semantics", {})
            header = f"{run_name} | {concept_id} | target={target}"
            draw.text((8, 6), header[:150], fill="black", font=small)
            y = 30
            for line in wrap_text(prompt)[:3]:
                draw.text((8, y), line, fill="black", font=font)
                y += 22

            for idx, sample in enumerate(rows):
                image = Image.open(sample["image_path"]).convert("RGB")
                image.thumbnail((thumb, thumb - 10))
                x = (idx % cols) * thumb + (thumb - image.width) // 2
                y = header_h + (idx // cols) * (thumb + label_h)
                canvas.paste(image, (x, y))
                tx = (idx % cols) * thumb + 6
                ty = y + thumb - 4
                draw.text((tx, ty), f"sample {int(sample['sample_id']):03d}", fill="black", font=small)
                draw.text((tx, ty + 18), Path(sample["image_path"]).name, fill="gray", font=small)
                manifest_rows.append(
                    {
                        "run": run_name,
                        "concept_id": concept_id,
                        "sample_id": int(sample["sample_id"]),
                        "prompt": prompt,
                        "target_semantics": json.dumps(target, ensure_ascii=False),
                        "image_path": sample["image_path"],
                    }
                )

            sheet_path = sheet_root / run_name / f"{concept_id}.png"
            sheet_path.parent.mkdir(parents=True, exist_ok=True)
            canvas.save(sheet_path)
            index_rows.append(
                {
                    "run": run_name,
                    "concept_id": concept_id,
                    "prompt": prompt,
                    "sheet_path": str(sheet_path),
                    "samples": len(rows),
                }
            )

    with (OUT / "codex_visual_gt_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["run", "concept_id", "sample_id", "prompt", "target_semantics", "image_path"],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    with (OUT / "codex_visual_gt_sheet_index.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["run", "concept_id", "prompt", "sheet_path", "samples"])
        writer.writeheader()
        writer.writerows(index_rows)

    print(f"sheets={len(index_rows)} samples={len(manifest_rows)} out={OUT}")


if __name__ == "__main__":
    main()
