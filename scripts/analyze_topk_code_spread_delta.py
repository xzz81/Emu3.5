#!/usr/bin/env python3
"""Compare traced top-k visual code embedding spread with mask DeltaNLL maps."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import numpy as np
import torch


VISUAL_RE = re.compile(r"<\|visual token (\d+)\|>")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--mask-dir-name", default="mask_delta_analysis")
    parser.add_argument("--out-dir-name", default=None)
    parser.add_argument(
        "--vq-ckpt",
        default="model/Emu3.5-VisionTokenizer/model.safetensors",
    )
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def write_csv(rows, path: Path):
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def rankdata(x):
    x = np.asarray(x)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    return ranks


def corr(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.std() < 1e-12 or y.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def compare_signal(signal, delta):
    x = np.asarray(signal, dtype=np.float64).reshape(-1)
    y = np.asarray(delta, dtype=np.float64).reshape(-1)
    k = max(1, int(round(0.2 * len(x))))
    top_x = set(np.argsort(x)[-k:].tolist())
    top_y = set(np.argsort(y)[-k:].tolist())
    rest_idx = [i for i in range(len(y)) if i not in top_x]
    top_vals = y[list(top_x)]
    rest_vals = y[rest_idx]
    return {
        "pearson": corr(x, y),
        "spearman": corr(rankdata(x), rankdata(y)),
        "top20_overlap": len(top_x & top_y) / k,
        "top20_minus_rest": float(top_vals.mean() - np.asarray(rest_vals).mean()),
    }


def load_codebook(path: Path, device: str):
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        tensors = load_file(str(path), device="cpu")
    else:
        ckpt = torch.load(path, map_location="cpu")
        tensors = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    emb = tensors["quantize.embedding.weight"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return emb.to(device=device, dtype=torch.float32)


def block_mean(arr, grid: int):
    arr = np.asarray(arr, dtype=np.float32)
    h, w = arr.shape
    out = np.zeros((grid, grid), dtype=np.float32)
    for row in range(grid):
        y0, y1 = row * h // grid, (row + 1) * h // grid
        for col in range(grid):
            x0, x1 = col * w // grid, (col + 1) * w // grid
            out[row, col] = float(arr[y0:y1, x0:x1].mean())
    return out


def load_cell_maps(mask_dir: Path):
    with (mask_dir / "mask_cell_metrics.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    by_sample = {}
    for row in rows:
        by_sample.setdefault(row["sample_id"], []).append(row)
    maps = {}
    for sample_id, sample_rows in by_sample.items():
        grid = max(max(int(r["row"]), int(r["col"])) for r in sample_rows) + 1
        sample_maps = {
            "coarse_ume": np.zeros((grid, grid), dtype=np.float32),
            "delta_nll": np.zeros((grid, grid), dtype=np.float32),
            "delta_entropy": np.zeros((grid, grid), dtype=np.float32),
        }
        for r in sample_rows:
            y = int(r["row"])
            x = int(r["col"])
            for field in sample_maps:
                sample_maps[field][y, x] = float(r[field])
        maps[sample_id] = sample_maps
    return maps


def visual_records(trace_path: Path):
    rows = []
    with trace_path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("token_type") == "visual":
                rows.append(row)
    side = int(math.sqrt(len(rows)))
    if side * side != len(rows):
        raise ValueError(f"expected square visual grid in {trace_path}, got {len(rows)} visual tokens")
    return rows, side


def topk_feature_maps(records, side: int, codebook: torch.Tensor):
    device = codebook.device
    emb_norm = torch.nn.functional.normalize(codebook, dim=-1)
    fields = {
        "topk_embedding_spread": [],
        "topk_entropy": [],
        "topk_prob_mass": [],
        "top1_prob": [],
        "sampled_code_topk_prob": [],
    }
    for row in records:
        code_ids = row.get("visual_topk_code_ids")
        probs = row.get("visual_topk_probs")
        if not code_ids or not probs:
            raise ValueError("trace is missing visual_topk_code_ids/probs; rerun with trace_topk_visual > 0")
        ids = torch.as_tensor(code_ids, dtype=torch.long, device=device)
        p = torch.as_tensor(probs, dtype=torch.float32, device=device)
        mass = p.sum().clamp_min(1e-12)
        q = p / mass
        e = emb_norm[ids]
        mean = (q[:, None] * e).sum(dim=0, keepdim=True)
        spread = (q * ((e - mean) ** 2).sum(dim=-1)).sum()
        entropy = -(q * torch.log(q.clamp_min(1e-12))).sum() / math.log(max(2, int(q.numel())))

        sampled_match = VISUAL_RE.fullmatch(row.get("token_text", ""))
        sampled_code = int(sampled_match.group(1)) if sampled_match else -1
        sampled_prob = 0.0
        for code_id, prob in zip(code_ids, probs):
            if int(code_id) == sampled_code:
                sampled_prob = float(prob)
                break

        fields["topk_embedding_spread"].append(float(spread.detach().cpu()))
        fields["topk_entropy"].append(float(entropy.detach().cpu()))
        fields["topk_prob_mass"].append(float(mass.detach().cpu()))
        fields["top1_prob"].append(float(torch.max(p).detach().cpu()))
        fields["sampled_code_topk_prob"].append(sampled_prob)

    return {
        name: np.asarray(values, dtype=np.float32).reshape(side, side)
        for name, values in fields.items()
    }


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    mask_dir = run_dir / args.mask_dir_name
    out_dir = run_dir / (args.out_dir_name or f"{args.mask_dir_name}_topk_code_spread")
    codebook = load_codebook(Path(args.vq_ckpt), args.device)
    print(f"[INFO] loaded codebook {tuple(codebook.shape)} on {codebook.device}")

    summary_rows = []
    cell_rows = []
    for sample_id, maps in sorted(load_cell_maps(mask_dir).items()):
        records, side = visual_records(run_dir / "entropy_traces" / f"{sample_id}_entropy.jsonl")
        grid = maps["delta_nll"].shape[0]
        features = {
            name: block_mean(values, grid)
            for name, values in topk_feature_maps(records, side, codebook).items()
        }
        summary = {"sample_id": sample_id, "grid": grid, "visual_grid": side}
        for signal_name, signal_map in {"ume": maps["coarse_ume"], **features}.items():
            for delta_name in ("delta_nll", "delta_entropy"):
                stats = compare_signal(signal_map, maps[delta_name])
                prefix = f"{signal_name}_{delta_name}"
                for key, value in stats.items():
                    summary[f"{prefix}_{key}"] = value
        summary_rows.append(summary)

        for row in range(grid):
            for col in range(grid):
                out = {
                    "sample_id": sample_id,
                    "row": row,
                    "col": col,
                    "coarse_ume": float(maps["coarse_ume"][row, col]),
                    "delta_nll": float(maps["delta_nll"][row, col]),
                    "delta_entropy": float(maps["delta_entropy"][row, col]),
                }
                for name in sorted(features):
                    out[name] = float(features[name][row, col])
                cell_rows.append(out)

    write_csv(summary_rows, out_dir / "summary_topk_code_spread.csv")
    write_csv(cell_rows, out_dir / "topk_code_spread_cell_metrics.csv")
    print(f"[INFO] wrote {len(summary_rows)} sample rows and {len(cell_rows)} cell rows to {out_dir}")


if __name__ == "__main__":
    main()
