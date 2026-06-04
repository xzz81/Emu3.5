#!/usr/bin/env python3
# Copyright 2026
# SPDX-License-Identifier: Apache-2.0

"""Build residual-patch target caches from saved raw generation text files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_generation_target_hue_cache import build_tokenizer  # noqa: E402
from scripts.run_generation_residual_patch_recovery import (  # noqa: E402
    cache_path_for,
    load_cfg,
    normalize_prompt_rows,
    target_cache_metadata,
    trim_to_first_image,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--sample-validation-csv", required=True)
    parser.add_argument("--out-cache-dir", required=True)
    parser.add_argument(
        "--only-hue-match",
        action="store_true",
        help="Only cache rows with hue_match=1.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def prompt_row_by_sample(cfg) -> dict[str, dict]:
    rows = {}
    for _pair_id, a_row, b_row in normalize_prompt_rows(cfg, 0):
        rows[a_row["sample_id"]] = a_row
        rows[b_row["sample_id"]] = b_row
    return rows


def main() -> None:
    args = parse_args()
    cfg = load_cfg(args.cfg)
    tokenizer = build_tokenizer(cfg.tokenizer_path)
    cfg.special_token_ids = {key: tokenizer.encode(value)[0] for key, value in cfg.special_tokens.items()}
    rows_by_sample = prompt_row_by_sample(cfg)
    out_cache_dir = Path(args.out_cache_dir)
    out_cache_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    for row in read_csv(Path(args.sample_validation_csv)):
        if args.only_hue_match and int(row.get("hue_match") or 0) != 1:
            continue
        sample_id = row["sample_id"]
        clean_row = rows_by_sample.get(sample_id)
        raw_path = Path(row.get("raw_generation_path", ""))
        if clean_row is None or not raw_path.exists():
            continue
        raw_text = raw_path.read_text(encoding="utf-8").strip()
        target_ids = tokenizer.encode(raw_text, add_special_tokens=False)
        target_ids = trim_to_first_image(target_ids, cfg.special_token_ids["EOI"])
        sample_seed = int(cfg.seed) + sum(ord(ch) for ch in sample_id)
        cache_path = cache_path_for(out_cache_dir, sample_id)
        payload = {
            "metadata": target_cache_metadata(cfg, clean_row, sample_seed),
            "target_ids": [int(x) for x in target_ids],
            "source": {
                "raw_generation_path": str(raw_path),
                "image_path": row.get("image_path", ""),
                "entropy_trace_path": row.get("entropy_trace_path", ""),
                "retrospective_hue_match": row.get("hue_match", ""),
                "retrospective_predicted_color": row.get("predicted_color", ""),
            },
        }
        cache_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        manifest_rows.append(
            {
                "sample_id": sample_id,
                "pair_id": row.get("pair_id", ""),
                "expected_color": row.get("expected_color", ""),
                "predicted_color": row.get("predicted_color", ""),
                "target_tokens": len(target_ids),
                "cache_path": str(cache_path),
                "raw_generation_path": str(raw_path),
            }
        )

    manifest_path = out_cache_dir / "raw_target_cache_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "sample_id",
            "pair_id",
            "expected_color",
            "predicted_color",
            "target_tokens",
            "cache_path",
            "raw_generation_path",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"[INFO] wrote {len(manifest_rows)} raw target caches to {out_cache_dir}")


if __name__ == "__main__":
    main()
