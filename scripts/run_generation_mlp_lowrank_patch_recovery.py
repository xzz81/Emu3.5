#!/usr/bin/env python3
# Copyright 2026
# SPDX-License-Identifier: Apache-2.0

"""Low-rank MLP-feature patching for Emu3.5 generation raw targets."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

try:
    _TORCHVISION_SCHEMA_LIB = torch.library.Library("torchvision", "DEF")
    _TORCHVISION_SCHEMA_LIB.define("nms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor")
except Exception:
    pass

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_generation_mlp_neuron_patch_recovery import (  # noqa: E402
    FullMLPIntermediateCollector,
    direction_context,
    selected_positions,
)
from scripts.run_generation_residual_patch_recovery import (  # noqa: E402
    build_patch_mask,
    filter_pairs,
    load_cfg,
    normalize_prompt_rows,
    target_nll,
)
from src.utils.activation_probe import write_csv_rows  # noqa: E402
from src.utils.model_utils import build_emu3p5  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--layer", type=int, default=61)
    parser.add_argument("--ranks", default="1,5,10,20,50")
    parser.add_argument("--random-seed", type=int, default=17)
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--pair-id", action="append", default=[])
    parser.add_argument(
        "--patch-scope",
        choices=["prompt", "color-token", "object-token", "visual-score-positions", "all-score-positions"],
        default="visual-score-positions",
    )
    parser.add_argument(
        "--score-scope",
        choices=["visual", "visual-and-structure", "all-target"],
        default="visual",
    )
    parser.add_argument("--target-cache-dir", required=True)
    parser.add_argument("--max-target-tokens", type=int, default=0)
    parser.add_argument("--pca-extra-rank", type=int, default=8)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def parse_ranks(text: str) -> List[int]:
    ranks = []
    for item in text.split(","):
        item = item.strip()
        if item:
            ranks.append(int(item))
    ranks = sorted(set(rank for rank in ranks if rank > 0))
    if not ranks:
        raise ValueError("provide at least one rank")
    return ranks


def collect_intermediate(model, input_ids: torch.Tensor, layer_idx: int) -> torch.Tensor:
    collector = FullMLPIntermediateCollector(model, layer_idx)
    collector.install()
    try:
        model.model(input_ids=input_ids, use_cache=False, return_dict=True)
    finally:
        collector.remove()
    if collector.intermediate is None:
        raise RuntimeError(f"missing MLP intermediate for layer {layer_idx}")
    return collector.intermediate


def collect_delta_rows(
    model,
    cfg,
    tokenizer,
    pairs,
    layer_idx: int,
    patch_scope: str,
    max_target_tokens: int,
    target_cache_dir: Path,
) -> Tuple[torch.Tensor, List[dict]]:
    deltas = []
    contexts = []
    for _pair_id, a_row, b_row in tqdm(pairs, desc="collect-deltas"):
        for clean_row, corrupt_row in ((a_row, b_row), (b_row, a_row)):
            ctx = direction_context(model, cfg, tokenizer, clean_row, corrupt_row, max_target_tokens, target_cache_dir)
            patch_mask, patched_positions, patch_note, source_position_for_target = build_patch_mask(
                tokenizer,
                ctx["clean_full"],
                ctx["corrupt_full"],
                ctx["clean_prompt"].shape[1],
                ctx["corrupt_prompt"].shape[1],
                ctx["target_ids"],
                patch_scope,
                clean_row["color"],
                corrupt_row["color"],
                clean_row["shape"],
                corrupt_row["shape"],
            )
            pos_pairs = selected_positions(patch_mask, source_position_for_target)
            clean_intermediate = collect_intermediate(model, ctx["clean_full"], layer_idx)
            torch.cuda.empty_cache()
            corrupt_intermediate = collect_intermediate(model, ctx["corrupt_full"], layer_idx)
            torch.cuda.empty_cache()
            for source_pos, target_pos in pos_pairs:
                if source_pos < clean_intermediate.shape[1] and target_pos < corrupt_intermediate.shape[1]:
                    delta = clean_intermediate[:, source_pos, :] - corrupt_intermediate[:, target_pos, :]
                    deltas.append(delta.detach().float().cpu())
            contexts.append(
                {
                    "pair_id": clean_row["pair_id"],
                    "clean_sample_id": clean_row["sample_id"],
                    "corrupt_sample_id": corrupt_row["sample_id"],
                    "patched_positions": patched_positions,
                    "patch_note": patch_note,
                }
            )
            del clean_intermediate, corrupt_intermediate
            torch.cuda.empty_cache()
    if not deltas:
        raise RuntimeError("no MLP delta rows collected")
    return torch.cat(deltas, dim=0).contiguous(), contexts


def compute_bases(deltas: torch.Tensor, ranks: Sequence[int], random_seed: int, extra_rank: int) -> Tuple[torch.Tensor, torch.Tensor, List[dict]]:
    max_rank = max(ranks)
    q = min(deltas.shape[0], deltas.shape[1], max_rank + max(0, extra_rank))
    centered = deltas - deltas.mean(dim=0, keepdim=True)
    _u, singular_values, v = torch.pca_lowrank(centered, q=q, center=False, niter=2)
    pca_basis = v[:, :max_rank].contiguous()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(random_seed))
    random_matrix = torch.randn(deltas.shape[1], max_rank, generator=generator, dtype=torch.float32)
    random_basis, _r = torch.linalg.qr(random_matrix, mode="reduced")
    energy = singular_values.float().pow(2)
    total_energy = float(centered.float().pow(2).sum().item())
    variance_rows = []
    for idx in range(min(max_rank, singular_values.numel())):
        variance_rows.append(
            {
                "rank": idx + 1,
                "singular_value": float(singular_values[idx].item()),
                "explained_energy": float(energy[idx].item()),
                "explained_energy_fraction_of_total": float(energy[idx].item() / total_energy) if total_energy else float("nan"),
            }
        )
    return pca_basis, random_basis.contiguous(), variance_rows


class LowRankMLPPatcher:
    def __init__(
        self,
        model,
        layer_idx: int,
        basis: torch.Tensor,
        clean_intermediate: torch.Tensor,
        position_pairs: Sequence[Tuple[int, int]],
    ):
        self.model = model
        self.layer_idx = int(layer_idx)
        self.basis = basis
        self.clean_intermediate = clean_intermediate
        self.position_pairs = [(int(source), int(target)) for source, target in position_pairs]
        self.handles = []
        self._gate = None
        self._corrupt_intermediate = None

    def install(self):
        mlp = self.model.model.layers[self.layer_idx].mlp
        self.handles.append(mlp.gate_proj.register_forward_hook(self._gate_hook(mlp.act_fn)))
        self.handles.append(mlp.up_proj.register_forward_hook(self._up_hook))
        self.handles.append(mlp.register_forward_hook(self._mlp_hook))

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self._gate = None
        self._corrupt_intermediate = None

    def _gate_hook(self, act_fn):
        def hook(_module, _inputs, output):
            self._gate = act_fn(output.detach())

        return hook

    def _up_hook(self, _module, _inputs, output):
        if self._gate is None:
            return
        self._corrupt_intermediate = (self._gate * output.detach()).detach()
        self._gate = None

    def _mlp_hook(self, module, _inputs, output):
        if self._corrupt_intermediate is None:
            return output
        valid_pairs = [
            (source_pos, target_pos)
            for source_pos, target_pos in self.position_pairs
            if source_pos < self.clean_intermediate.shape[1] and target_pos < self._corrupt_intermediate.shape[1]
        ]
        if not valid_pairs:
            return output
        source_idx = torch.tensor([source for source, _target in valid_pairs], device=output.device, dtype=torch.long)
        target_idx = torch.tensor([target for _source, target in valid_pairs], device=output.device, dtype=torch.long)
        basis = self.basis.to(device=output.device, dtype=module.down_proj.weight.dtype)
        clean = self.clean_intermediate.to(device=output.device, dtype=module.down_proj.weight.dtype).index_select(1, source_idx)
        corrupt = self._corrupt_intermediate.to(device=output.device, dtype=module.down_proj.weight.dtype).index_select(1, target_idx)
        delta = clean - corrupt
        coeff = torch.matmul(delta, basis)
        projected_delta = torch.matmul(coeff, basis.transpose(0, 1))
        contribution_delta = F.linear(projected_delta, module.down_proj.weight).to(output.dtype)
        patched = output.clone()
        patched[:, target_idx, :] = patched[:, target_idx, :] + contribution_delta
        self._corrupt_intermediate = None
        return patched


def run_patch_group(
    model,
    cfg,
    tokenizer,
    pairs,
    layer_idx: int,
    group_label: str,
    basis: torch.Tensor,
    patch_scope: str,
    score_scope: str,
    max_target_tokens: int,
    target_cache_dir: Path,
) -> List[dict]:
    rows = []
    for _pair_id, a_row, b_row in tqdm(pairs, desc=group_label):
        for clean_row, corrupt_row in ((a_row, b_row), (b_row, a_row)):
            ctx = direction_context(model, cfg, tokenizer, clean_row, corrupt_row, max_target_tokens, target_cache_dir)
            patch_mask, patched_positions, patch_note, source_position_for_target = build_patch_mask(
                tokenizer,
                ctx["clean_full"],
                ctx["corrupt_full"],
                ctx["clean_prompt"].shape[1],
                ctx["corrupt_prompt"].shape[1],
                ctx["target_ids"],
                patch_scope,
                clean_row["color"],
                corrupt_row["color"],
                clean_row["shape"],
                corrupt_row["shape"],
            )
            pos_pairs = selected_positions(patch_mask, source_position_for_target)
            clean_intermediate = collect_intermediate(model, ctx["clean_full"], layer_idx)
            torch.cuda.empty_cache()
            patcher = LowRankMLPPatcher(model, layer_idx, basis, clean_intermediate, pos_pairs)
            patcher.install()
            try:
                patched = target_nll(model, ctx["corrupt_prompt"], ctx["target_ids"], score_scope)
            finally:
                patcher.remove()
            torch.cuda.empty_cache()
            clean = ctx["clean"]
            corrupt = ctx["corrupt"]
            denom = corrupt["nll_sum"] - clean["nll_sum"]
            valid_gap = denom > 1e-9
            recovery = (corrupt["nll_sum"] - patched["nll_sum"]) / denom if valid_gap else float("nan")
            rows.append(
                {
                    "layer": layer_idx,
                    "group": group_label,
                    "rank": int(basis.shape[1]),
                    "pair_id": clean_row["pair_id"],
                    "clean_sample_id": clean_row["sample_id"],
                    "corrupt_sample_id": corrupt_row["sample_id"],
                    "shape": clean_row["shape"],
                    "clean_color": clean_row["color"],
                    "corrupt_color": corrupt_row["color"],
                    "target_tokens": clean["target_tokens"],
                    "visual_tokens": clean["visual_tokens"],
                    "scored_tokens": clean["scored_tokens"],
                    "clean_nll_sum": clean["nll_sum"],
                    "corrupt_nll_sum": corrupt["nll_sum"],
                    "patched_nll_sum": patched["nll_sum"],
                    "corrupt_minus_clean_nll_sum": denom,
                    "patched_delta_vs_corrupt_nll_sum": patched["nll_sum"] - corrupt["nll_sum"],
                    "recovery_fraction": recovery,
                    "valid_recovery_gap": int(valid_gap),
                    "patch_scope": patch_scope,
                    "score_scope": score_scope,
                    "patched_positions": patched_positions,
                    "patch_note": patch_note,
                }
            )
            del clean_intermediate
    return rows


def summarize(rows: List[dict]) -> List[dict]:
    grouped: Dict[str, List[dict]] = {}
    all_grouped: Dict[str, List[dict]] = {}
    for row in rows:
        group = row["group"]
        all_grouped.setdefault(group, []).append(row)
        if math.isfinite(float(row["recovery_fraction"])):
            grouped.setdefault(group, []).append(row)
    summary = []
    for group, all_rows in sorted(all_grouped.items()):
        valid = grouped.get(group, [])
        if not valid:
            continue
        recoveries = [float(row["recovery_fraction"]) for row in valid]
        deltas = [float(row["patched_delta_vs_corrupt_nll_sum"]) for row in valid]
        gaps = [float(row["corrupt_minus_clean_nll_sum"]) for row in valid]
        summary.append(
            {
                "layer": int(valid[0]["layer"]),
                "group": group,
                "rank": int(valid[0]["rank"]),
                "n_directions": len(valid),
                "invalid_gap_directions": len(all_rows) - len(valid),
                "helped_directions": sum(1 for delta in deltas if delta < 0),
                "mean_recovery_fraction": sum(recoveries) / len(recoveries),
                "median_recovery_fraction": statistics.median(recoveries),
                "mean_patch_delta_vs_corrupt_nll_sum": sum(deltas) / len(deltas),
                "sum_corrupt_minus_clean_nll_sum": sum(gaps),
                "sum_patch_delta_vs_corrupt_nll_sum": sum(deltas),
            }
        )
    return summary


def write_report(out_dir: Path, args: argparse.Namespace, summary_rows: List[dict]) -> None:
    lines = [
        "# Generation MLP Low-Rank Patch Recovery",
        "",
        f"- Config: `{args.cfg}`",
        f"- Layer: `{args.layer}`",
        f"- Ranks: `{args.ranks}`",
        f"- Patch scope: `{args.patch_scope}`",
        f"- Score scope: `{args.score_scope}`",
        f"- Target cache dir: `{args.target_cache_dir}`",
        "",
        "| group | rank | directions | helped | mean recovery | median recovery | sum delta NLL vs corrupt |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {group} | {rank} | {n_directions} | {helped_directions} | "
            "{mean_recovery_fraction:+.6f} | {median_recovery_fraction:+.6f} | "
            "{sum_patch_delta_vs_corrupt_nll_sum:+.6f} |".format(**row)
        )
    (out_dir / "generation_mlp_lowrank_patch_recovery_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    ranks = parse_ranks(args.ranks)
    cfg = load_cfg(args.cfg)
    model, tokenizer, vq_model = build_emu3p5(
        cfg.model_path,
        cfg.tokenizer_path,
        cfg.vq_path,
        vq_type=cfg.vq_type,
        model_device=cfg.hf_device,
        vq_device=cfg.vq_device,
        **getattr(cfg, "diffusion_decoder_kwargs", {}),
    )
    del vq_model
    model.eval()
    cfg.special_token_ids = {key: tokenizer.encode(value)[0] for key, value in cfg.special_tokens.items()}
    pairs = filter_pairs(normalize_prompt_rows(cfg, args.max_pairs), args.pair_id)
    if not pairs:
        raise SystemExit("No generation pairs matched the requested filters.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    deltas, contexts = collect_delta_rows(
        model,
        cfg,
        tokenizer,
        pairs,
        args.layer,
        args.patch_scope,
        args.max_target_tokens,
        Path(args.target_cache_dir),
    )
    (out_dir / "delta_contexts.json").write_text(json.dumps(contexts, indent=2, ensure_ascii=False), encoding="utf-8")
    pca_basis, random_basis, variance_rows = compute_bases(deltas, ranks, args.random_seed, args.pca_extra_rank)
    write_csv_rows(variance_rows, out_dir / "pca_variance.csv")
    torch.save(
        {
            "pca_basis": pca_basis,
            "random_basis": random_basis,
            "ranks": ranks,
            "delta_shape": list(deltas.shape),
        },
        out_dir / "lowrank_bases.pt",
    )
    del deltas

    rows = []
    for rank in ranks:
        rows.extend(
            run_patch_group(
                model,
                cfg,
                tokenizer,
                pairs,
                args.layer,
                f"pca{rank}",
                pca_basis[:, :rank],
                args.patch_scope,
                args.score_scope,
                args.max_target_tokens,
                Path(args.target_cache_dir),
            )
        )
        rows.extend(
            run_patch_group(
                model,
                cfg,
                tokenizer,
                pairs,
                args.layer,
                f"random{rank}",
                random_basis[:, :rank],
                args.patch_scope,
                args.score_scope,
                args.max_target_tokens,
                Path(args.target_cache_dir),
            )
        )

    write_csv_rows(rows, out_dir / "generation_mlp_lowrank_patch_recovery.csv")
    summary_rows = summarize(rows)
    write_csv_rows(summary_rows, out_dir / "summary.csv")
    write_report(out_dir, args, summary_rows)
    metadata = {
        "cfg": args.cfg,
        "layer": args.layer,
        "ranks": args.ranks,
        "patch_scope": args.patch_scope,
        "score_scope": args.score_scope,
        "pair_id": args.pair_id,
        "max_pairs": args.max_pairs,
        "target_cache_dir": args.target_cache_dir,
        "random_seed": args.random_seed,
        "pca_extra_rank": args.pca_extra_rank,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[INFO] generation MLP low-rank patch recovery saved to {out_dir}")


if __name__ == "__main__":
    main()
