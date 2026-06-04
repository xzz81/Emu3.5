#!/usr/bin/env python3
# Copyright 2026
# SPDX-License-Identifier: Apache-2.0

"""Clean/corrupt MLP-neuron contribution patching for Emu3.5 generation raw targets."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Dict, Iterable, List, Sequence, Tuple

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

from scripts.run_generation_residual_patch_recovery import (  # noqa: E402
    build_patch_mask,
    build_prompt_ids,
    filter_pairs,
    get_target_ids,
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
    parser.add_argument("--top-n-values", default="10,50,200", help="Comma-separated top-k groups to patch.")
    parser.add_argument("--bottom-n-values", default="10,50,200", help="Comma-separated bottom-k groups to patch.")
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
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def parse_int_list(text: str) -> List[int]:
    values = []
    for item in text.split(","):
        item = item.strip()
        if item:
            values.append(int(item))
    return sorted(set(value for value in values if value > 0))


def selected_positions(mask: torch.Tensor, source_position_for_target: Dict[int, int]) -> List[Tuple[int, int]]:
    out = []
    for target_pos in mask.nonzero(as_tuple=False).flatten().detach().cpu().tolist():
        target_pos = int(target_pos)
        source_pos = int(source_position_for_target.get(target_pos, target_pos))
        out.append((source_pos, target_pos))
    return out


class FullMLPIntermediateCollector:
    def __init__(self, model, layer_idx: int):
        self.model = model
        self.layer_idx = int(layer_idx)
        self.handles = []
        self._gate = None
        self.intermediate = None

    def install(self):
        mlp = self.model.model.layers[self.layer_idx].mlp
        self.handles.append(mlp.gate_proj.register_forward_hook(self._gate_hook(mlp.act_fn)))
        self.handles.append(mlp.up_proj.register_forward_hook(self._up_hook))

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self._gate = None

    def _gate_hook(self, act_fn):
        def hook(_module, _inputs, output):
            self._gate = act_fn(output.detach())

        return hook

    def _up_hook(self, _module, _inputs, output):
        if self._gate is None:
            return
        self.intermediate = (self._gate * output.detach()).detach()
        self._gate = None


class CleanSelectedMLPCollector:
    def __init__(self, model, layer_idx: int, neurons: Sequence[int]):
        self.model = model
        self.layer_idx = int(layer_idx)
        self.neurons = [int(neuron) for neuron in neurons]
        self.handles = []
        self._gate = None
        self.clean_selected = None

    def install(self):
        mlp = self.model.model.layers[self.layer_idx].mlp
        self.handles.append(mlp.gate_proj.register_forward_hook(self._gate_hook(mlp.act_fn)))
        self.handles.append(mlp.up_proj.register_forward_hook(self._up_hook))

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self._gate = None

    def _gate_hook(self, act_fn):
        def hook(_module, _inputs, output):
            self._gate = act_fn(output.detach())

        return hook

    def _up_hook(self, _module, _inputs, output):
        if self._gate is None:
            return
        neuron_ids = torch.tensor(self.neurons, device=output.device, dtype=torch.long)
        self.clean_selected = (self._gate * output.detach()).index_select(dim=-1, index=neuron_ids).detach()
        self._gate = None


class MLPNeuronContributionPatcher:
    def __init__(
        self,
        model,
        layer_idx: int,
        neurons: Sequence[int],
        clean_selected: torch.Tensor,
        position_pairs: Sequence[Tuple[int, int]],
    ):
        self.model = model
        self.layer_idx = int(layer_idx)
        self.neurons = [int(neuron) for neuron in neurons]
        self.clean_selected = clean_selected
        self.position_pairs = [(int(source), int(target)) for source, target in position_pairs]
        self.handles = []
        self._gate = None
        self._corrupt_selected = None

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
        self._corrupt_selected = None

    def _gate_hook(self, act_fn):
        def hook(_module, _inputs, output):
            self._gate = act_fn(output.detach())

        return hook

    def _up_hook(self, _module, _inputs, output):
        if self._gate is None:
            return
        neuron_ids = torch.tensor(self.neurons, device=output.device, dtype=torch.long)
        self._corrupt_selected = (self._gate * output.detach()).index_select(dim=-1, index=neuron_ids).detach()
        self._gate = None

    def _mlp_hook(self, module, _inputs, output):
        corrupt_selected = self._corrupt_selected
        if self.clean_selected is None or corrupt_selected is None:
            return output
        clean_selected = self.clean_selected.to(device=output.device, dtype=module.down_proj.weight.dtype)
        corrupt_selected = corrupt_selected.to(device=output.device, dtype=module.down_proj.weight.dtype)
        neuron_ids = torch.tensor(self.neurons, device=output.device, dtype=torch.long)
        weight = module.down_proj.weight.index_select(dim=1, index=neuron_ids)
        patched = output.clone()
        valid_pairs = [
            (source_pos, target_pos)
            for source_pos, target_pos in self.position_pairs
            if 0 <= source_pos < clean_selected.shape[1] and 0 <= target_pos < corrupt_selected.shape[1]
        ]
        if valid_pairs:
            source_idx = torch.tensor([source for source, _target in valid_pairs], device=output.device, dtype=torch.long)
            target_idx = torch.tensor([target for _source, target in valid_pairs], device=output.device, dtype=torch.long)
            clean_values = clean_selected.index_select(dim=1, index=source_idx)
            corrupt_values = corrupt_selected.index_select(dim=1, index=target_idx)
            contribution_delta = F.linear(clean_values - corrupt_values, weight).to(output.dtype)
            patched[:, target_idx, :] = patched[:, target_idx, :] + contribution_delta
        self._corrupt_selected = None
        return patched


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


def direction_context(model, cfg, tokenizer, clean_row, corrupt_row, max_target_tokens: int, target_cache_dir: Path):
    clean_prompt, unconditional_ids, full_unc_ids = build_prompt_ids(cfg, tokenizer, clean_row["prompt"], model.device)
    corrupt_prompt, _corrupt_unc, _corrupt_full_unc = build_prompt_ids(cfg, tokenizer, corrupt_row["prompt"], model.device)
    seed_offset = sum(ord(ch) for ch in clean_row["sample_id"])
    target_ids, target_source, target_cache_path = get_target_ids(
        cfg,
        model,
        tokenizer,
        clean_row,
        clean_prompt,
        unconditional_ids,
        full_unc_ids,
        int(cfg.seed) + seed_offset,
        target_cache_dir,
        False,
    )
    if max_target_tokens and target_ids.shape[1] > max_target_tokens:
        target_ids = target_ids[:, :max_target_tokens]
    clean = target_nll(model, clean_prompt, target_ids, "visual")
    torch.cuda.empty_cache()
    corrupt = target_nll(model, corrupt_prompt, target_ids, "visual")
    torch.cuda.empty_cache()
    clean_full = clean["full_ids"]
    corrupt_full = torch.cat([corrupt_prompt, target_ids], dim=1)
    return {
        "clean_prompt": clean_prompt,
        "corrupt_prompt": corrupt_prompt,
        "target_ids": target_ids,
        "target_source": target_source,
        "target_cache_path": target_cache_path,
        "clean": clean,
        "corrupt": corrupt,
        "clean_full": clean_full,
        "corrupt_full": corrupt_full,
    }


def collect_candidate_scores(
    model,
    cfg,
    tokenizer,
    pairs,
    layer_idx: int,
    patch_scope: str,
    max_target_tokens: int,
    target_cache_dir: Path,
) -> Tuple[List[dict], List[dict]]:
    score_sum = None
    count = 0
    contexts = []
    for _pair_id, a_row, b_row in tqdm(pairs, desc="select-neurons"):
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
            local_sum = None
            local_count = 0
            for source_pos, target_pos in pos_pairs:
                if source_pos >= clean_intermediate.shape[1] or target_pos >= corrupt_intermediate.shape[1]:
                    continue
                diff = (clean_intermediate[:, source_pos, :] - corrupt_intermediate[:, target_pos, :]).abs().float()
                local_sum = diff.sum(dim=0) if local_sum is None else local_sum + diff.sum(dim=0)
                local_count += int(diff.shape[0])
            if local_sum is None:
                raise RuntimeError("no valid positions for neuron selection")
            score_sum = local_sum.detach().cpu() if score_sum is None else score_sum + local_sum.detach().cpu()
            count += local_count
            contexts.append(
                {
                    "clean_row": clean_row,
                    "corrupt_row": corrupt_row,
                    "patched_positions": patched_positions,
                    "patch_note": patch_note,
                }
            )
            del clean_intermediate, corrupt_intermediate, local_sum
            torch.cuda.empty_cache()
    if score_sum is None or count == 0:
        raise RuntimeError("no candidate scores collected")
    scores = (score_sum / float(count)).tolist()
    rows = [{"layer": layer_idx, "neuron": idx, "selection_score": score} for idx, score in enumerate(scores)]
    rows.sort(key=lambda row: float(row["selection_score"]), reverse=True)
    return rows, contexts


def run_patch_group(
    model,
    cfg,
    tokenizer,
    pairs,
    layer_idx: int,
    group_label: str,
    neurons: Sequence[int],
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
            collector = CleanSelectedMLPCollector(model, layer_idx, neurons)
            collector.install()
            try:
                model.model(input_ids=ctx["clean_full"], use_cache=False, return_dict=True)
            finally:
                collector.remove()
            if collector.clean_selected is None:
                raise RuntimeError(f"missing clean selected MLP activations for layer {layer_idx}")
            torch.cuda.empty_cache()
            patcher = MLPNeuronContributionPatcher(
                model,
                layer_idx,
                neurons,
                collector.clean_selected,
                pos_pairs,
            )
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
                    "n_neurons": len(neurons),
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
                    "neurons": ";".join(str(neuron) for neuron in neurons),
                }
            )
    return rows


def summarize(rows: List[dict]) -> List[dict]:
    by_group: Dict[str, List[dict]] = {}
    all_group: Dict[str, List[dict]] = {}
    for row in rows:
        group = row["group"]
        all_group.setdefault(group, []).append(row)
        if math.isfinite(float(row["recovery_fraction"])):
            by_group.setdefault(group, []).append(row)
    summary = []
    for group, all_rows in sorted(all_group.items()):
        valid = by_group.get(group, [])
        if not valid:
            continue
        recoveries = [float(row["recovery_fraction"]) for row in valid]
        deltas = [float(row["patched_delta_vs_corrupt_nll_sum"]) for row in valid]
        gaps = [float(row["corrupt_minus_clean_nll_sum"]) for row in valid]
        summary.append(
            {
                "layer": int(valid[0]["layer"]),
                "group": group,
                "n_neurons": int(valid[0]["n_neurons"]),
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
        "# Generation MLP Neuron Patch Recovery",
        "",
        f"- Config: `{args.cfg}`",
        f"- Layer: `{args.layer}`",
        f"- Patch scope: `{args.patch_scope}`",
        f"- Score scope: `{args.score_scope}`",
        f"- Target cache dir: `{args.target_cache_dir}`",
        "",
        "| group | neurons | directions | helped | mean recovery | median recovery | sum delta NLL vs corrupt |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {group} | {n_neurons} | {n_directions} | {helped_directions} | "
            "{mean_recovery_fraction:+.6f} | {median_recovery_fraction:+.6f} | "
            "{sum_patch_delta_vs_corrupt_nll_sum:+.6f} |".format(**row)
        )
    (out_dir / "generation_mlp_neuron_patch_recovery_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
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

    candidate_rows, _contexts = collect_candidate_scores(
        model,
        cfg,
        tokenizer,
        pairs,
        args.layer,
        args.patch_scope,
        args.max_target_tokens,
        Path(args.target_cache_dir),
    )
    write_csv_rows(candidate_rows, out_dir / "candidate_scores.csv")

    groups: List[Tuple[str, List[int]]] = []
    for n in parse_int_list(args.top_n_values):
        groups.append((f"top{n}", [int(row["neuron"]) for row in candidate_rows[:n]]))
    ascending = list(reversed(candidate_rows))
    for n in parse_int_list(args.bottom_n_values):
        groups.append((f"bottom{n}", [int(row["neuron"]) for row in ascending[:n]]))

    rows = []
    for group_label, neurons in groups:
        if not neurons:
            continue
        rows.extend(
            run_patch_group(
                model,
                cfg,
                tokenizer,
                pairs,
                args.layer,
                group_label,
                neurons,
                args.patch_scope,
                args.score_scope,
                args.max_target_tokens,
                Path(args.target_cache_dir),
            )
        )

    write_csv_rows(rows, out_dir / "generation_mlp_neuron_patch_recovery.csv")
    summary_rows = summarize(rows)
    write_csv_rows(summary_rows, out_dir / "summary.csv")
    write_report(out_dir, args, summary_rows)
    metadata = {
        "cfg": args.cfg,
        "layer": args.layer,
        "top_n_values": args.top_n_values,
        "bottom_n_values": args.bottom_n_values,
        "patch_scope": args.patch_scope,
        "score_scope": args.score_scope,
        "pair_id": args.pair_id,
        "max_pairs": args.max_pairs,
        "target_cache_dir": args.target_cache_dir,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[INFO] generation MLP neuron patch recovery saved to {out_dir}")


if __name__ == "__main__":
    main()
