#!/usr/bin/env python3
"""Duplicate-image and mode-collapse checks for semantic entropy T2I outputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-dir", default="outputs/semantic_entropy_umm/pilot")
    parser.add_argument("--strict-dir", default="outputs/semantic_entropy_umm/strict_compare")
    parser.add_argument("--out-dir", default="outputs/semantic_entropy_umm/image_duplicate_mode_collapse")
    parser.add_argument("--hash-size", type=int, default=8)
    parser.add_argument("--perceptual-threshold", type=int, default=4)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None and rows:
        fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or [])
        writer.writeheader()
        writer.writerows(rows)


def exact_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def average_hash(path: Path, hash_size: int) -> str:
    image = Image.open(path).convert("L").resize((hash_size, hash_size), Image.Resampling.LANCZOS)
    values = list(image.getdata())
    mean = sum(values) / len(values)
    return "".join("1" if value >= mean else "0" for value in values)


def hamming(left: str, right: str) -> int:
    if len(left) != len(right):
        raise ValueError("hash strings must have the same length")
    return sum(a != b for a, b in zip(left, right))


def collect_generated_images(pilot_dir: Path) -> dict[str, list[dict]]:
    rows = read_jsonl(pilot_dir / "t2i_samples.jsonl")
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        path = Path(row["image_path"])
        if not path.is_absolute():
            path = pilot_dir.parents[2] / path if str(path).startswith("outputs/") else path
        grouped[row["concept_id"]].append(
            {
                "concept_id": row["concept_id"],
                "sample_id": int(row["sample_id"]),
                "image_path": str(path),
            }
        )
    for concept_id in grouped:
        grouped[concept_id].sort(key=lambda item: item["sample_id"])
    return grouped


def collect_reference_images(pilot_dir: Path) -> list[dict]:
    concepts = read_jsonl(pilot_dir / "concepts.jsonl")
    rows = []
    for concept in concepts:
        path = pilot_dir / "reference_images" / f"{concept['concept_id']}.png"
        rows.append({"concept_id": concept["concept_id"], "sample_id": 0, "image_path": str(path)})
    return rows


def metric_index(strict_dir: Path) -> dict[tuple[str, str, str], float]:
    out: dict[tuple[str, str, str], float] = {}
    for row in read_csv(strict_dir / "strict_route_entropy.csv"):
        if row.get("route") == "T2I":
            out[(row["concept_id"], row["route"], f"{row['slot']}_entropy")] = float(row["entropy"])
    for row in read_csv(strict_dir / "strict_route_error.csv"):
        if row.get("route") == "T2I":
            out[(row["concept_id"], row["route"], f"{row['slot']}_error")] = float(row["error_rate"])
    return out


def add_hashes(rows: list[dict], hash_size: int) -> list[dict]:
    out = []
    for row in rows:
        path = Path(row["image_path"])
        out.append(
            {
                **row,
                "exists": path.exists(),
                "exact_hash": exact_hash(path) if path.exists() else "",
                "average_hash": average_hash(path, hash_size) if path.exists() else "",
            }
        )
    return out


def summarize_generated(
    grouped: dict[str, list[dict]],
    strict_metrics: dict[tuple[str, str, str], float],
    perceptual_threshold: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    summary = []
    pairwise = []
    high_similarity = []
    for concept_id, images in sorted(grouped.items()):
        existing = [row for row in images if row["exists"]]
        pairs = list(combinations(existing, 2))
        distances = []
        exact_duplicate_pairs = 0
        perceptual_duplicate_pairs = 0
        for left, right in pairs:
            dist = hamming(left["average_hash"], right["average_hash"])
            distances.append(dist)
            exact_dup = left["exact_hash"] == right["exact_hash"]
            perceptual_dup = dist <= perceptual_threshold
            exact_duplicate_pairs += int(exact_dup)
            perceptual_duplicate_pairs += int(perceptual_dup)
            row = {
                "concept_id": concept_id,
                "left_sample_id": left["sample_id"],
                "right_sample_id": right["sample_id"],
                "left_image_path": left["image_path"],
                "right_image_path": right["image_path"],
                "exact_duplicate": exact_dup,
                "average_hash_hamming": dist,
                "average_hash_similarity": 1.0 - (dist / len(left["average_hash"])),
                "perceptual_duplicate_threshold": perceptual_threshold,
                "perceptual_duplicate": perceptual_dup,
            }
            pairwise.append(row)
            if perceptual_dup or exact_dup:
                high_similarity.append(row)
        pair_count = len(pairs)
        unique_exact = len({row["exact_hash"] for row in existing})
        summary.append(
            {
                "concept_id": concept_id,
                "generated_images": len(images),
                "existing_images": len(existing),
                "exact_unique_images": unique_exact,
                "exact_duplicate_image_rate": 1.0 - (unique_exact / len(existing)) if existing else "",
                "pair_count": pair_count,
                "exact_duplicate_pairs": exact_duplicate_pairs,
                "perceptual_duplicate_pairs": perceptual_duplicate_pairs,
                "perceptual_duplicate_pair_rate": perceptual_duplicate_pairs / pair_count if pair_count else "",
                "min_average_hash_hamming": min(distances) if distances else "",
                "mean_average_hash_hamming": sum(distances) / len(distances) if distances else "",
                "max_average_hash_similarity": 1.0 - (min(distances) / (len(existing[0]["average_hash"]))) if distances else "",
                "t2i_joint_entropy": strict_metrics.get((concept_id, "T2I", "joint_entropy"), ""),
                "t2i_object_1_entropy": strict_metrics.get((concept_id, "T2I", "object_1_entropy"), ""),
                "t2i_object_1_error": strict_metrics.get((concept_id, "T2I", "object_1_error"), ""),
                "mode_collapse_flag": bool(exact_duplicate_pairs or perceptual_duplicate_pairs / pair_count >= 0.5) if pair_count else False,
            }
        )
    high_similarity.sort(key=lambda row: (not row["exact_duplicate"], row["average_hash_hamming"], row["concept_id"]))
    return summary, pairwise, high_similarity


def summarize_reference(rows: list[dict], perceptual_threshold: int) -> tuple[list[dict], list[dict]]:
    existing = [row for row in rows if row["exists"]]
    pairwise = []
    for left, right in combinations(existing, 2):
        dist = hamming(left["average_hash"], right["average_hash"])
        pairwise.append(
            {
                "left_concept_id": left["concept_id"],
                "right_concept_id": right["concept_id"],
                "left_image_path": left["image_path"],
                "right_image_path": right["image_path"],
                "exact_duplicate": left["exact_hash"] == right["exact_hash"],
                "average_hash_hamming": dist,
                "average_hash_similarity": 1.0 - (dist / len(left["average_hash"])),
                "perceptual_duplicate_threshold": perceptual_threshold,
                "perceptual_duplicate": dist <= perceptual_threshold,
            }
        )
    exact_unique = len({row["exact_hash"] for row in existing})
    exact_pairs = sum(row["exact_duplicate"] for row in pairwise)
    perceptual_pairs = sum(row["perceptual_duplicate"] for row in pairwise)
    summary = [
        {
            "reference_images": len(rows),
            "existing_images": len(existing),
            "exact_unique_images": exact_unique,
            "exact_duplicate_image_rate": 1.0 - (exact_unique / len(existing)) if existing else "",
            "pair_count": len(pairwise),
            "exact_duplicate_pairs": exact_pairs,
            "perceptual_duplicate_pairs": perceptual_pairs,
            "perceptual_duplicate_pair_rate": perceptual_pairs / len(pairwise) if pairwise else "",
            "min_average_hash_hamming": min([row["average_hash_hamming"] for row in pairwise]) if pairwise else "",
            "mean_average_hash_hamming": sum(row["average_hash_hamming"] for row in pairwise) / len(pairwise) if pairwise else "",
        }
    ]
    return summary, pairwise


def md_table(rows: list[dict], fields: list[str] | None = None) -> list[str]:
    if not rows:
        return ["(no rows)"]
    fields = fields or list(rows[0].keys())
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    return lines


def write_report(
    out_dir: Path,
    generated_summary: list[dict],
    high_similarity: list[dict],
    reference_summary: list[dict],
    args: argparse.Namespace,
) -> None:
    total_generated = sum(int(row["existing_images"]) for row in generated_summary)
    total_exact_duplicates = sum(int(row["generated_images"]) - int(row["exact_unique_images"]) for row in generated_summary)
    total_pairs = sum(int(row["pair_count"]) for row in generated_summary)
    perceptual_pairs = sum(int(row["perceptual_duplicate_pairs"]) for row in generated_summary)
    flagged = [row for row in generated_summary if row["mode_collapse_flag"]]
    top_fields = [
        "concept_id",
        "generated_images",
        "exact_duplicate_image_rate",
        "perceptual_duplicate_pair_rate",
        "min_average_hash_hamming",
        "t2i_joint_entropy",
        "t2i_object_1_error",
        "mode_collapse_flag",
    ]
    pair_fields = [
        "concept_id",
        "left_sample_id",
        "right_sample_id",
        "exact_duplicate",
        "average_hash_hamming",
        "average_hash_similarity",
        "left_image_path",
        "right_image_path",
    ]
    lines = [
        "# Duplicate Image / Mode Collapse Check",
        "",
        "Status: RETROSPECTIVE_PIXEL_HASH_ANALYSIS",
        "",
        "This report checks exact file duplicates and 8x8 average-hash similarity for existing pilot images. It does not use CLIP/image embeddings and does not prove semantic diversity by itself.",
        "",
        "## Input Files",
        "",
        f"- `{args.pilot_dir}/t2i_samples.jsonl`",
        f"- `{args.pilot_dir}/generated_images/`",
        f"- `{args.pilot_dir}/reference_images/`",
        f"- `{args.strict_dir}/strict_route_entropy.csv`",
        f"- `{args.strict_dir}/strict_route_error.csv`",
        "",
        "## Commands",
        "",
        "```bash",
        "./.venv-transformers/bin/python scripts/analyze_semantic_entropy_image_duplicates.py \\",
        f"  --pilot-dir {args.pilot_dir} \\",
        f"  --strict-dir {args.strict_dir} \\",
        f"  --out-dir {args.out_dir} \\",
        f"  --hash-size {args.hash_size} \\",
        f"  --perceptual-threshold {args.perceptual_threshold}",
        "```",
        "",
        "## Output Files",
        "",
        f"- `{out_dir / 'generated_image_duplicate_summary.csv'}`",
        f"- `{out_dir / 'generated_image_pairwise_similarity.csv'}`",
        f"- `{out_dir / 'high_similarity_cases.csv'}`",
        f"- `{out_dir / 'reference_image_duplicate_summary.csv'}`",
        f"- `{out_dir / 'reference_image_pairwise_similarity.csv'}`",
        f"- `{out_dir / 'image_duplicate_mode_collapse_report.md'}`",
        "",
        "## Sample Counts",
        "",
        f"- Existing generated images: `{total_generated}`",
        f"- Generated concept groups: `{len(generated_summary)}`",
        f"- Pairwise generated-image comparisons: `{total_pairs}`",
        f"- Reference summary rows: `{len(reference_summary)}`",
        f"- High-similarity generated pairs listed: `{len(high_similarity)}`",
        "",
        "## Pass/Fail Checks",
        "",
        f"- Generated images present: `{total_generated > 0}`",
        f"- Generated summary written: `{bool(generated_summary)}`",
        f"- Reference summary written: `{bool(reference_summary)}`",
        "- Pixel/hash-only caveat retained: `True`",
        "",
        "## Overall Generated-Image Summary",
        "",
        f"- Existing generated images: {total_generated}",
        f"- Exact duplicate images within concept groups: {total_exact_duplicates}",
        f"- Pairwise generated-image comparisons within concept groups: {total_pairs}",
        f"- Perceptual duplicate pairs at average-hash threshold <= {args.perceptual_threshold}: {perceptual_pairs}",
        f"- Concepts flagged by this heuristic: {len(flagged)}",
        "",
        "## Reference-Image Summary",
        "",
        *md_table(reference_summary),
        "",
        "## Concept Summary",
        "",
        *md_table(generated_summary, top_fields),
        "",
        "## High-Similarity Generated Pairs",
        "",
        *md_table(high_similarity[:30], pair_fields),
        "",
        "## Claim Allowed After This Step",
        "",
        "Use this as a pixel/perceptual-hash duplicate check for whether low T2I semantic entropy is trivially explained by exact or near-duplicate generated images.",
        "",
        "## Claim Still Not Allowed",
        "",
        "Do not claim full mode-collapse robustness without stronger image embeddings or human/semantic diversity checks.",
    ]
    (out_dir / "image_duplicate_mode_collapse_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    pilot_dir = Path(args.pilot_dir)
    strict_dir = Path(args.strict_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    strict_metrics = metric_index(strict_dir)
    generated = {
        concept_id: add_hashes(rows, args.hash_size)
        for concept_id, rows in collect_generated_images(pilot_dir).items()
    }
    generated_summary, pairwise, high_similarity = summarize_generated(
        generated,
        strict_metrics,
        args.perceptual_threshold,
    )
    reference_rows = add_hashes(collect_reference_images(pilot_dir), args.hash_size)
    reference_summary, reference_pairwise = summarize_reference(reference_rows, args.perceptual_threshold)
    write_csv(out_dir / "generated_image_duplicate_summary.csv", generated_summary)
    write_csv(out_dir / "generated_image_pairwise_similarity.csv", pairwise)
    write_csv(out_dir / "high_similarity_cases.csv", high_similarity)
    write_csv(out_dir / "reference_image_duplicate_summary.csv", reference_summary)
    write_csv(out_dir / "reference_image_pairwise_similarity.csv", reference_pairwise)
    write_report(out_dir, generated_summary, high_similarity, reference_summary, args)
    print(
        f"[INFO] wrote duplicate/mode-collapse analysis for {len(generated_summary)} concepts under {out_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
