#!/usr/bin/env python3
"""Build atom-level visual-check annotation files for complex T2I runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import textwrap
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        default="outputs/semantic_entropy_umm/semantic_uncertainty_complex_t2i_emu35_20260616",
    )
    parser.add_argument("--samples-per-concept", type=int, default=10)
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Defaults to RUN_DIR/atom_check_annotation.",
    )
    parser.add_argument("--sheet-cols", type=int, default=5)
    parser.add_argument("--allow-empty-images", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        if not rows:
            return
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_font(size: int):
    for name in ["DejaVuSans.ttf", "Arial.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def image_path_for(run_dir: Path, concept_id: str, sample_id: int) -> Path:
    return run_dir / "pilot" / "generated_images" / concept_id / f"{sample_id:03d}.png"


def build_template_rows(concepts: list[dict[str, Any]], run_id: str, samples_per_concept: int) -> list[dict[str, Any]]:
    rows = []
    for concept in concepts:
        checks = concept.get("visual_checks") or []
        for sample_id in range(samples_per_concept):
            for check_idx, check in enumerate(checks):
                rows.append(
                    {
                        "run_id": run_id,
                        "concept_id": concept["concept_id"],
                        "sample_id": sample_id,
                        "check_id": f"check_{check_idx:02d}",
                        "check_text": check,
                        "pass_fail": "",
                        "failure_type": "",
                        "confidence": "",
                        "image_path": str(image_path_for(Path(""), concept["concept_id"], sample_id)).lstrip("/"),
                        "notes": "",
                    }
                )
    return rows


def build_concept_summary_rows(concepts: list[dict[str, Any]], samples_per_concept: int) -> list[dict[str, Any]]:
    rows = []
    for concept in concepts:
        checks = concept.get("visual_checks") or []
        rows.append(
            {
                "concept_id": concept["concept_id"],
                "samples_per_concept": samples_per_concept,
                "atom_check_count": len(checks),
                "annotation_rows": samples_per_concept * len(checks),
                "complexity_tags": ";".join(concept.get("complexity_tags") or []),
                "visual_checks": " || ".join(checks),
                "prompt": concept["prompt"],
            }
        )
    return rows


def make_contact_sheet(run_dir: Path, concept: dict[str, Any], samples_per_concept: int, out_path: Path, cols: int) -> bool:
    image_paths = [image_path_for(run_dir, concept["concept_id"], idx) for idx in range(samples_per_concept)]
    existing = [path for path in image_paths if path.exists()]
    if not existing:
        return False

    cell_w, cell_h = 260, 310
    header_h = 210
    rows = math.ceil(samples_per_concept / cols)
    sheet = Image.new("RGB", (cols * cell_w, header_h + rows * cell_h), "white")
    draw = ImageDraw.Draw(sheet)
    title_font = load_font(16)
    small_font = load_font(11)
    tiny_font = load_font(10)

    draw.text((12, 10), concept["concept_id"], fill="black", font=title_font)
    y = 36
    for line in textwrap.wrap(concept["prompt"], width=120)[:4]:
        draw.text((12, y), line, fill="black", font=small_font)
        y += 16
    y += 4
    draw.text((12, y), "Atom visual checks:", fill="black", font=small_font)
    y += 16
    for idx, check in enumerate(concept.get("visual_checks") or []):
        for line_i, line in enumerate(textwrap.wrap(f"{idx}. {check}", width=110)[:2]):
            draw.text((20, y), line, fill="black", font=tiny_font)
            y += 13
            if line_i >= 1:
                break

    for sample_id, image_path in enumerate(image_paths):
        x = (sample_id % cols) * cell_w
        y0 = header_h + (sample_id // cols) * cell_h
        if image_path.exists():
            img = Image.open(image_path).convert("RGB")
            img.thumbnail((224, 224))
        else:
            img = Image.new("RGB", (224, 224), "#dddddd")
            d = ImageDraw.Draw(img)
            d.text((20, 100), "missing", fill="black", font=small_font)
        sheet.paste(img, (x + 18, y0 + 12))
        draw.text((x + 18, y0 + 244), f"sample {sample_id:03d}", fill="black", font=small_font)
        draw.text((x + 18, y0 + 263), str(image_path), fill="#333333", font=tiny_font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    return True


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "atom_check_annotation"
    concepts = read_jsonl(run_dir / "pilot" / "concepts.jsonl")
    if not concepts:
        raise SystemExit(f"missing concepts: {run_dir / 'pilot' / 'concepts.jsonl'}")

    run_id = run_dir.name
    template_rows = build_template_rows(concepts, run_id, args.samples_per_concept)
    summary_rows = build_concept_summary_rows(concepts, args.samples_per_concept)
    write_csv(out_dir / "complex_t2i_atom_check_gt_template.csv", template_rows)
    write_csv(out_dir / "complex_t2i_atom_check_concept_summary.csv", summary_rows)

    sheet_rows = []
    for concept in concepts:
        sheet_path = out_dir / "contact_sheets" / f"{concept['concept_id']}.png"
        made = make_contact_sheet(run_dir, concept, args.samples_per_concept, sheet_path, args.sheet_cols)
        if made:
            sheet_rows.append(
                {
                    "concept_id": concept["concept_id"],
                    "sheet_path": str(sheet_path),
                    "atom_check_count": len(concept.get("visual_checks") or []),
                    "samples_per_concept": args.samples_per_concept,
                }
            )
    write_csv(out_dir / "complex_t2i_atom_check_sheet_index.csv", sheet_rows)

    if not sheet_rows and not args.allow_empty_images:
        raise SystemExit(
            "no contact sheets were created because no generated images were found; "
            "rerun with --allow-empty-images to create only CSV templates"
        )
    print(f"[INFO] concepts={len(concepts)} template_rows={len(template_rows)} sheets={len(sheet_rows)} out={out_dir}")


if __name__ == "__main__":
    main()
