#!/usr/bin/env python3
"""Compare VQ codebook embedding maps with mask DeltaNLL maps.

This is a post-hoc metric over the sampled visual codes. It does not estimate
distributional embedding spread over candidate next tokens because the entropy
traces do not store candidate probabilities.
"""

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
SIGNALS = (
    "ume",
    "code_embedding_edge_cos",
    "code_embedding_edge_l2",
    "code_embedding_cell_variance",
    "code_embedding_global_distance",
    "code_embedding_norm",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="outputs/emu3p5-main")
    parser.add_argument(
        "--vq-ckpt",
        default="model/Emu3.5-VisionTokenizer/model.safetensors",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--run-id-contains", default="first_image_stop")
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

    for key in ("quantize.embedding.weight", "model.quantize.embedding.weight"):
        if key in tensors:
            emb = tensors[key]
            break
    else:
        candidates = [key for key in tensors if key.endswith("quantize.embedding.weight")]
        if not candidates:
            raise KeyError(f"cannot find quantize.embedding.weight in {path}")
        emb = tensors[candidates[0]]

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    emb = emb.to(device=device, dtype=torch.float32)
    return emb


def load_visual_code_grid(trace_path: Path):
    codes = []
    with trace_path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("token_type") != "visual":
                continue
            match = VISUAL_RE.fullmatch(row.get("token_text", ""))
            if match is None:
                raise ValueError(f"cannot parse visual token from {trace_path}: {row.get('token_text')}")
            codes.append(int(match.group(1)))
    side = int(math.sqrt(len(codes)))
    if side * side != len(codes):
        raise ValueError(f"expected square visual code grid in {trace_path}, got {len(codes)} codes")
    return np.asarray(codes, dtype=np.int64).reshape(side, side)


def load_cell_maps(mask_dir: Path):
    with (mask_dir / "mask_cell_metrics.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    by_sample = {}
    for row in rows:
        by_sample.setdefault(row["sample_id"], []).append(row)
    maps = {}
    for sample_id, sample_rows in by_sample.items():
        grid = max(max(int(r["row"]), int(r["col"])) for r in sample_rows) + 1
        fields = ("coarse_ume", "delta_nll", "delta_entropy")
        sample_maps = {field: np.zeros((grid, grid), dtype=np.float32) for field in fields}
        for r in sample_rows:
            y = int(r["row"])
            x = int(r["col"])
            for field in fields:
                sample_maps[field][y, x] = float(r[field])
        maps[sample_id] = sample_maps
    return maps


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


def code_embedding_feature_maps(code_grid: np.ndarray, codebook: torch.Tensor):
    device = codebook.device
    ids = torch.as_tensor(code_grid, dtype=torch.long, device=device)
    emb = codebook[ids]
    emb_norm = torch.nn.functional.normalize(emb, dim=-1)

    edge_cos_terms = []
    edge_l2_terms = []
    counts = torch.zeros(ids.shape, dtype=torch.float32, device=device)
    edge_cos = torch.zeros_like(counts)
    edge_l2 = torch.zeros_like(counts)
    for dy, dx in ((1, 0), (0, 1)):
        a = emb_norm[:-dy or None, :-dx or None]
        b = emb_norm[dy:, dx:]
        dist_cos = 1.0 - (a * b).sum(dim=-1)
        dist_l2 = torch.linalg.vector_norm(emb[:-dy or None, :-dx or None] - emb[dy:, dx:], dim=-1)

        target_a = (slice(None, -dy or None), slice(None, -dx or None))
        target_b = (slice(dy, None), slice(dx, None))
        edge_cos[target_a] += dist_cos
        edge_cos[target_b] += dist_cos
        edge_l2[target_a] += dist_l2
        edge_l2[target_b] += dist_l2
        counts[target_a] += 1.0
        counts[target_b] += 1.0
        edge_cos_terms.append(dist_cos)
        edge_l2_terms.append(dist_l2)

    edge_cos = edge_cos / counts.clamp_min(1.0)
    edge_l2 = edge_l2 / counts.clamp_min(1.0)
    image_mean = emb_norm.reshape(-1, emb_norm.shape[-1]).mean(dim=0)
    global_distance = 1.0 - (emb_norm * torch.nn.functional.normalize(image_mean, dim=0)).sum(dim=-1)
    code_norm = torch.linalg.vector_norm(emb, dim=-1)
    return {
        "code_embedding_edge_cos": edge_cos.detach().cpu().numpy(),
        "code_embedding_edge_l2": edge_l2.detach().cpu().numpy(),
        "code_embedding_global_distance": global_distance.detach().cpu().numpy(),
        "code_embedding_norm": code_norm.detach().cpu().numpy(),
    }


def cell_variance_map(code_grid: np.ndarray, codebook: torch.Tensor, grid: int):
    device = codebook.device
    ids = torch.as_tensor(code_grid, dtype=torch.long, device=device)
    emb = torch.nn.functional.normalize(codebook[ids], dim=-1)
    h, w = ids.shape
    out = np.zeros((grid, grid), dtype=np.float32)
    for row in range(grid):
        y0, y1 = row * h // grid, (row + 1) * h // grid
        for col in range(grid):
            x0, x1 = col * w // grid, (col + 1) * w // grid
            vals = emb[y0:y1, x0:x1].reshape(-1, emb.shape[-1])
            center = vals.mean(dim=0, keepdim=True)
            out[row, col] = float(((vals - center) ** 2).sum(dim=-1).mean().detach().cpu())
    return out


def run_dirs(root: Path, run_id_contains: str):
    for mask_path in sorted(root.glob("*/ume_trace_runs/*/mask_delta_analysis*/mask_cell_metrics.csv")):
        mask_dir = mask_path.parent
        if mask_dir.name not in {"mask_delta_analysis", "mask_delta_analysis_8x8"}:
            continue
        run_dir = mask_dir.parent
        if run_id_contains and run_id_contains not in run_dir.name:
            continue
        task = run_dir.parent.parent.name
        run_id = run_dir.name
        batch = "expand4" if "expand4" in run_id else "batch2" if "batch2" in run_id else "other"
        grid = 8 if mask_dir.name.endswith("8x8") else 4
        yield task, run_id, batch, grid, run_dir, mask_dir


def main():
    args = parse_args()
    root = Path(args.root)
    out_dir = Path(args.out_dir)
    codebook = load_codebook(Path(args.vq_ckpt), args.device)
    print(f"[INFO] loaded codebook {tuple(codebook.shape)} on {codebook.device}")

    sample_rows = []
    cell_rows = []
    for task, run_id, batch, grid, run_dir, mask_dir in run_dirs(root, args.run_id_contains):
        cell_maps = load_cell_maps(mask_dir)
        for sample_id, maps in sorted(cell_maps.items()):
            trace_path = run_dir / "entropy_traces" / f"{sample_id}_entropy.jsonl"
            code_grid = load_visual_code_grid(trace_path)
            token_features = code_embedding_feature_maps(code_grid, codebook)
            features = {
                name: block_mean(values, grid)
                for name, values in token_features.items()
            }
            features["code_embedding_cell_variance"] = cell_variance_map(code_grid, codebook, grid)

            summary = {
                "task": task,
                "run_id": run_id,
                "batch": batch,
                "grid": grid,
                "sample_id": sample_id,
                "visual_grid": code_grid.shape[0],
            }
            for signal_name, signal_map in {"ume": maps["coarse_ume"], **features}.items():
                for delta_name in ("delta_nll", "delta_entropy"):
                    stats = compare_signal(signal_map, maps[delta_name])
                    prefix = f"{signal_name}_{delta_name}"
                    for key, value in stats.items():
                        summary[f"{prefix}_{key}"] = value
            sample_rows.append(summary)

            for row in range(grid):
                for col in range(grid):
                    out = {
                        "task": task,
                        "run_id": run_id,
                        "batch": batch,
                        "grid": grid,
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

    write_csv(sample_rows, out_dir / "summary_code_embedding_delta.csv")
    write_csv(cell_rows, out_dir / "code_embedding_cell_metrics.csv")
    print(f"[INFO] wrote {len(sample_rows)} sample rows and {len(cell_rows)} cell rows to {out_dir}")


if __name__ == "__main__":
    main()
