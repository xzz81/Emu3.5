#!/usr/bin/env python3
"""Offline image-embedding duplicate checks for semantic entropy images.

This script first records whether common semantic image embedding packages or
cached models are available. It then runs a no-download fallback based on
deterministic pixel-stat embeddings. The fallback is useful for catching
near-identical generated images, but it is not CLIP semantic evidence.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
from collections import defaultdict
from itertools import combinations
from pathlib import Path

from PIL import Image, ImageFilter, ImageStat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-dir", default="outputs/semantic_entropy_umm/pilot")
    parser.add_argument("--strict-dir", default="outputs/semantic_entropy_umm/strict_compare")
    parser.add_argument("--out-dir", default="outputs/semantic_entropy_umm/image_embedding_duplicate_check")
    parser.add_argument("--cosine-threshold", type=float, default=0.995)
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
    fields = fields or (list(rows[0].keys()) if rows else [])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else path


def preflight() -> dict:
    modules = {}
    for module in ["torch", "torchvision", "transformers", "open_clip", "timm", "PIL", "numpy", "sklearn"]:
        modules[module] = importlib.util.find_spec(module) is not None
    cache_candidates = []
    complete_cache_candidates = []
    incomplete_cache_candidates = []
    for root_text in [".cache/huggingface", "weights"]:
        root = Path(root_text)
        if not root.exists():
            continue
        for path in root.glob("**/*"):
            if path.is_dir() and any(token in path.name.lower() for token in ["clip", "siglip", "dinov", "vit"]):
                cache_candidates.append(str(path))
                snapshots = list((path / "snapshots").glob("*")) if (path / "snapshots").exists() else [path]
                complete = any(
                    (snapshot / "config.json").exists()
                    and ((snapshot / "pytorch_model.bin").exists() or (snapshot / "model.safetensors").exists())
                    and ((snapshot / "preprocessor_config.json").exists() or (snapshot / "processor_config.json").exists())
                    for snapshot in snapshots
                )
                if complete:
                    complete_cache_candidates.append(str(path))
                else:
                    incomplete_cache_candidates.append(str(path))
    semantic_model_available = bool(modules.get("open_clip") or complete_cache_candidates)
    return {
        "embedding_backend": "pixel_stat_fallback",
        "semantic_embedding_model_available": semantic_model_available,
        "modules": modules,
        "cache_candidates": cache_candidates[:50],
        "complete_cache_candidates": complete_cache_candidates[:50],
        "incomplete_cache_candidates": incomplete_cache_candidates[:50],
        "claim_boundary": "fallback is not CLIP/SigLIP/DINO semantic embedding evidence",
    }


def collect_generated_images(pilot_dir: Path) -> dict[str, list[dict]]:
    rows = read_jsonl(pilot_dir / "t2i_samples.jsonl")
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["concept_id"]].append(
            {
                "concept_id": row["concept_id"],
                "sample_id": int(row["sample_id"]),
                "image_path": str(resolve_path(row["image_path"])),
                "image_group": "generated",
            }
        )
    for concept_id in grouped:
        grouped[concept_id].sort(key=lambda item: item["sample_id"])
    return grouped


def collect_reference_images(pilot_dir: Path) -> list[dict]:
    concepts = read_jsonl(pilot_dir / "concepts.jsonl")
    rows = []
    for concept in concepts:
        rows.append(
            {
                "concept_id": concept["concept_id"],
                "sample_id": 0,
                "image_path": str(pilot_dir / "reference_images" / f"{concept['concept_id']}.png"),
                "image_group": "reference",
            }
        )
    return rows


def metric_index(strict_dir: Path) -> dict[tuple[str, str, str], float]:
    out: dict[tuple[str, str, str], float] = {}
    for row in read_csv(strict_dir / "strict_route_entropy.csv"):
        if row.get("route") == "T2I":
            out[(row["concept_id"], "T2I", f"{row['slot']}_entropy")] = float(row["entropy"])
    for row in read_csv(strict_dir / "strict_route_error.csv"):
        if row.get("route") == "T2I":
            out[(row["concept_id"], "T2I", f"{row['slot']}_error")] = float(row["error_rate"])
    return out


def normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if not norm:
        return vector
    return [value / norm for value in vector]


def image_embedding(path: Path) -> list[float]:
    image = Image.open(path).convert("RGB").resize((64, 64), Image.Resampling.BICUBIC)
    gray = image.convert("L")
    channels = image.split()
    features: list[float] = []

    for channel in channels:
        hist = channel.histogram()
        total = sum(hist) or 1
        for start in range(0, 256, 32):
            features.append(sum(hist[start : start + 32]) / total)

    small = gray.resize((16, 16), Image.Resampling.BICUBIC)
    pixels = [value / 255.0 for value in small.getdata()]
    mean = sum(pixels) / len(pixels)
    centered = [value - mean for value in pixels]
    features.extend(centered)

    edges = gray.filter(ImageFilter.FIND_EDGES)
    stat = ImageStat.Stat(edges)
    features.append(stat.mean[0] / 255.0)
    features.append(stat.stddev[0] / 255.0)
    return normalize(features)


def cosine(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def add_embeddings(rows: list[dict]) -> list[dict]:
    out = []
    for row in rows:
        path = Path(row["image_path"])
        exists = path.exists()
        out.append({**row, "exists": exists, "embedding": image_embedding(path) if exists else []})
    return out


def generated_pairwise(
    grouped: dict[str, list[dict]],
    strict_metrics: dict[tuple[str, str, str], float],
    threshold: float,
) -> tuple[list[dict], list[dict], list[dict]]:
    summaries = []
    pairwise = []
    high_similarity = []
    for concept_id, images in sorted(grouped.items()):
        existing = [row for row in images if row["exists"]]
        similarities = []
        high_pairs = 0
        for left, right in combinations(existing, 2):
            sim = cosine(left["embedding"], right["embedding"])
            similarities.append(sim)
            high = sim >= threshold
            high_pairs += int(high)
            pair = {
                "concept_id": concept_id,
                "left_sample_id": left["sample_id"],
                "right_sample_id": right["sample_id"],
                "left_image_path": left["image_path"],
                "right_image_path": right["image_path"],
                "pixel_stat_cosine": sim,
                "high_similarity_threshold": threshold,
                "high_similarity": high,
            }
            pairwise.append(pair)
            if high:
                high_similarity.append(pair)
        pair_count = len(similarities)
        summaries.append(
            {
                "concept_id": concept_id,
                "generated_images": len(images),
                "existing_images": len(existing),
                "pair_count": pair_count,
                "high_similarity_pairs": high_pairs,
                "high_similarity_pair_rate": high_pairs / pair_count if pair_count else "",
                "min_pixel_stat_cosine": min(similarities) if similarities else "",
                "mean_pixel_stat_cosine": sum(similarities) / pair_count if pair_count else "",
                "max_pixel_stat_cosine": max(similarities) if similarities else "",
                "t2i_joint_entropy": strict_metrics.get((concept_id, "T2I", "joint_entropy"), ""),
                "t2i_object_1_entropy": strict_metrics.get((concept_id, "T2I", "object_1_entropy"), ""),
                "t2i_object_1_error": strict_metrics.get((concept_id, "T2I", "object_1_error"), ""),
                "fallback_mode_collapse_flag": bool(high_pairs / pair_count >= 0.5) if pair_count else False,
            }
        )
    high_similarity.sort(key=lambda row: (-float(row["pixel_stat_cosine"]), row["concept_id"]))
    return summaries, pairwise, high_similarity


def reference_pairwise(rows: list[dict], threshold: float) -> tuple[list[dict], list[dict]]:
    existing = [row for row in rows if row["exists"]]
    pairwise = []
    for left, right in combinations(existing, 2):
        sim = cosine(left["embedding"], right["embedding"])
        pairwise.append(
            {
                "left_concept_id": left["concept_id"],
                "right_concept_id": right["concept_id"],
                "left_image_path": left["image_path"],
                "right_image_path": right["image_path"],
                "pixel_stat_cosine": sim,
                "high_similarity_threshold": threshold,
                "high_similarity": sim >= threshold,
            }
        )
    similarities = [row["pixel_stat_cosine"] for row in pairwise]
    high_pairs = sum(row["high_similarity"] for row in pairwise)
    summary = [
        {
            "reference_images": len(rows),
            "existing_images": len(existing),
            "pair_count": len(pairwise),
            "high_similarity_pairs": high_pairs,
            "high_similarity_pair_rate": high_pairs / len(pairwise) if pairwise else "",
            "min_pixel_stat_cosine": min(similarities) if similarities else "",
            "mean_pixel_stat_cosine": sum(similarities) / len(similarities) if similarities else "",
            "max_pixel_stat_cosine": max(similarities) if similarities else "",
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
    preflight_report: dict,
    generated_summary: list[dict],
    generated_high: list[dict],
    reference_summary: list[dict],
    args: argparse.Namespace,
) -> None:
    total_generated = sum(int(row["existing_images"]) for row in generated_summary)
    total_pairs = sum(int(row["pair_count"]) for row in generated_summary)
    high_pairs = sum(int(row["high_similarity_pairs"]) for row in generated_summary)
    flagged = [row for row in generated_summary if row["fallback_mode_collapse_flag"]]
    concept_fields = [
        "concept_id",
        "generated_images",
        "high_similarity_pair_rate",
        "mean_pixel_stat_cosine",
        "max_pixel_stat_cosine",
        "t2i_joint_entropy",
        "t2i_object_1_error",
        "fallback_mode_collapse_flag",
    ]
    pair_fields = [
        "concept_id",
        "left_sample_id",
        "right_sample_id",
        "pixel_stat_cosine",
        "left_image_path",
        "right_image_path",
    ]
    lines = [
        "# Image-Embedding Duplicate Check",
        "",
        "Status: FALLBACK_PIXEL_STAT_EMBEDDING",
        "",
        "This report records that no local CLIP/SigLIP/DINO embedding backend was available and uses a deterministic pixel-stat embedding fallback. It can catch near-identical images, but it is not semantic CLIP evidence.",
        "",
        "## Preflight",
        "",
        f"- Embedding backend: `{preflight_report['embedding_backend']}`",
        f"- Semantic embedding model available: `{preflight_report['semantic_embedding_model_available']}`",
        f"- Cache candidates: `{len(preflight_report['cache_candidates'])}`",
        f"- Complete cache candidates: `{len(preflight_report['complete_cache_candidates'])}`",
        f"- Incomplete cache candidates: `{len(preflight_report['incomplete_cache_candidates'])}`",
        f"- Modules: `{preflight_report['modules']}`",
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
        "./.venv-transformers/bin/python scripts/analyze_semantic_entropy_image_embedding_duplicates.py \\",
        f"  --pilot-dir {args.pilot_dir} \\",
        f"  --strict-dir {args.strict_dir} \\",
        f"  --out-dir {args.out_dir} \\",
        f"  --cosine-threshold {args.cosine_threshold}",
        "```",
        "",
        "## Output Files",
        "",
        f"- `{out_dir / 'embedding_preflight.json'}`",
        f"- `{out_dir / 'generated_embedding_similarity_summary.csv'}`",
        f"- `{out_dir / 'generated_embedding_pairwise_similarity.csv'}`",
        f"- `{out_dir / 'generated_embedding_high_similarity_cases.csv'}`",
        f"- `{out_dir / 'reference_embedding_similarity_summary.csv'}`",
        f"- `{out_dir / 'reference_embedding_pairwise_similarity.csv'}`",
        f"- `{out_dir / 'image_embedding_duplicate_report.md'}`",
        "",
        "## Sample Counts",
        "",
        f"- Existing generated images: `{total_generated}`",
        f"- Generated concept groups: `{len(generated_summary)}`",
        f"- Pairwise generated-image comparisons: `{total_pairs}`",
        f"- Reference summary rows: `{len(reference_summary)}`",
        f"- High-similarity generated pairs listed: `{len(generated_high)}`",
        "",
        "## Pass/Fail Checks",
        "",
        f"- Fallback backend recorded: `{preflight_report['embedding_backend'] == 'pixel_stat_fallback'}`",
        f"- Generated images present: `{total_generated > 0}`",
        f"- Generated summary written: `{bool(generated_summary)}`",
        "- Semantic-embedding caveat retained: `True`",
        "",
        "## Overall Generated-Image Summary",
        "",
        f"- Existing generated images: {total_generated}",
        f"- Pairwise generated-image comparisons within concept groups: {total_pairs}",
        f"- High-similarity pairs at cosine >= {args.cosine_threshold}: {high_pairs}",
        f"- Concepts flagged by fallback heuristic: {len(flagged)}",
        "",
        "## Reference-Image Summary",
        "",
        *md_table(reference_summary),
        "",
        "## Concept Summary",
        "",
        *md_table(generated_summary, concept_fields),
        "",
        "## Top High-Similarity Generated Pairs",
        "",
        *md_table(generated_high[:30], pair_fields),
        "",
        "## Claim Allowed After This Step",
        "",
        "Use this only as an offline pixel-stat embedding duplicate check for near-identical generated images.",
        "",
        "## Claim Still Not Allowed",
        "",
        "Do not claim CLIP/image-embedding semantic diversity or full mode-collapse robustness from this fallback. A real cached/local CLIP, SigLIP, DINO, or equivalent semantic embedding model is still needed for that stronger check.",
    ]
    (out_dir / "image_embedding_duplicate_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    pilot_dir = Path(args.pilot_dir)
    strict_dir = Path(args.strict_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    preflight_report = preflight()
    strict_metrics = metric_index(strict_dir)
    generated = {concept: add_embeddings(rows) for concept, rows in collect_generated_images(pilot_dir).items()}
    reference = add_embeddings(collect_reference_images(pilot_dir))
    generated_summary, generated_pairwise_rows, generated_high = generated_pairwise(
        generated, strict_metrics, args.cosine_threshold
    )
    reference_summary, reference_pairwise_rows = reference_pairwise(reference, args.cosine_threshold)

    (out_dir / "embedding_preflight.json").write_text(json.dumps(preflight_report, indent=2), encoding="utf-8")
    write_csv(out_dir / "generated_embedding_similarity_summary.csv", generated_summary)
    write_csv(out_dir / "generated_embedding_pairwise_similarity.csv", generated_pairwise_rows)
    write_csv(out_dir / "generated_embedding_high_similarity_cases.csv", generated_high)
    write_csv(out_dir / "reference_embedding_similarity_summary.csv", reference_summary)
    write_csv(out_dir / "reference_embedding_pairwise_similarity.csv", reference_pairwise_rows)
    write_report(out_dir, preflight_report, generated_summary, generated_high, reference_summary, args)
    print(
        json.dumps(
            {
                "status": "FALLBACK_PIXEL_STAT_EMBEDDING",
                "generated_concepts": len(generated_summary),
                "generated_pairwise_rows": len(generated_pairwise_rows),
                "high_similarity_pairs": len(generated_high),
                "semantic_embedding_model_available": preflight_report["semantic_embedding_model_available"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
