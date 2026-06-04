#!/usr/bin/env python3
# Copyright 2026
# SPDX-License-Identifier: Apache-2.0

"""Patch MLP low-rank feature directions in Emu3.5 hue understanding runs."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Dict, List, Sequence, Tuple

from PIL import Image
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

from src.utils.activation_probe import write_csv_rows  # noqa: E402
from src.utils.input_utils import build_image  # noqa: E402
from src.utils.logits_processor import BOI, BOV, EOI, IMG  # noqa: E402
from src.utils.model_utils import build_emu3p5  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="configs/ume_main_image_read_hue_control_gt_greedy_seed71.py")
    parser.add_argument("--manifest", default="/workspace/home/AAAI 2027/research_logs/hue_control_gt_images_seed70/manifest.json")
    parser.add_argument("--basis-path", required=True)
    parser.add_argument("--layer", type=int, default=61)
    parser.add_argument("--ranks", default="50,200")
    parser.add_argument("--basis-types", default="pca,random", help="Comma-separated basis types: pca,random.")
    parser.add_argument("--max-pairs", type=int, default=4)
    parser.add_argument("--question-template", default="What color is the {shape}? Answer with one word.")
    parser.add_argument(
        "--patch-scope",
        choices=[
            "prompt",
            "answer",
            "all",
            "visual-prompt",
            "eoi-prompt",
            "post-image-text-prompt",
            "question-text-no-score",
            "score-positions",
        ],
        default="score-positions",
    )
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def parse_int_list(text: str) -> List[int]:
    values = []
    for item in text.split(","):
        item = item.strip()
        if item:
            values.append(int(item))
    values = sorted(set(value for value in values if value > 0))
    if not values:
        raise ValueError("provide at least one positive rank")
    return values


def parse_basis_types(text: str) -> List[str]:
    out = []
    for item in text.split(","):
        item = item.strip().lower()
        if not item:
            continue
        if item not in ("pca", "random"):
            raise ValueError(f"unknown basis type: {item}")
        out.append(item)
    return sorted(set(out))


def load_cfg(path: str):
    cfg_path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location(cfg_path.stem, cfg_path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"could not load config: {cfg_path}")
    sys.path.insert(0, str(REPO_ROOT))
    spec.loader.exec_module(module)
    return module


def grouped_manifest(path: str, max_pairs: int):
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    by_pair: Dict[str, List[dict]] = {}
    for row in rows:
        by_pair.setdefault(row["pair_id"], []).append(row)
    pairs = []
    for pair_id, pair_rows in sorted(by_pair.items()):
        if len(pair_rows) != 2:
            continue
        pair_rows = sorted(pair_rows, key=lambda item: item["variant"])
        pairs.append((pair_id, pair_rows[0], pair_rows[1]))
        if max_pairs and len(pairs) >= max_pairs:
            break
    return pairs


def build_prompt_ids(cfg, tokenizer, vq_model, image_path: str, question: str, device):
    _unc_prompt, template = cfg.build_unc_and_template(getattr(cfg, "task_type", "unknown"), with_image=True)
    image_str = build_image(Image.open(image_path).convert("RGB"), cfg, tokenizer, vq_model)
    prompt = template.format(question=question).replace("<|IMAGE|>", image_str)
    input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    if input_ids[0, 0] != cfg.special_token_ids["BOS"]:
        bos = torch.tensor([[cfg.special_token_ids["BOS"]]], device=device, dtype=input_ids.dtype)
        input_ids = torch.cat([bos, input_ids], dim=1)
    return input_ids


def encode_answer(tokenizer, answer: str, device):
    variants = [answer, " " + answer, answer.capitalize(), " " + answer.capitalize()]
    encoded = []
    for text in variants:
        ids = tokenizer.encode(text, return_tensors="pt", add_special_tokens=False).to(device)
        if ids.numel() > 0:
            encoded.append((text, ids))
    encoded.sort(key=lambda item: (item[1].shape[1], len(item[0])))
    return encoded[0]


@torch.no_grad()
def answer_nll(model, prompt_ids, answer_ids):
    full_ids = torch.cat([prompt_ids, answer_ids], dim=1)
    answer_start = prompt_ids.shape[1]
    outputs = model.model(input_ids=full_ids, use_cache=False, return_dict=True)
    hidden_states = outputs.last_hidden_state[:, answer_start - 1 : -1, :]
    logits = model.lm_head(hidden_states).float()
    labels = answer_ids.to(logits.device)
    losses = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        reduction="none",
    )
    return {
        "full_ids": full_ids,
        "nll_mean": float(losses.mean().item()),
        "nll_sum": float(losses.sum().item()),
        "answer_tokens": int(labels.numel()),
    }


def build_patch_mask(corrupt_full, corrupt_prompt_len: int, answer_len: int, patch_scope: str):
    mask = torch.zeros(corrupt_full.shape[1], dtype=torch.bool, device=corrupt_full.device)
    if patch_scope == "prompt":
        mask[:corrupt_prompt_len] = True
    elif patch_scope == "answer":
        mask[corrupt_prompt_len:] = True
    elif patch_scope == "all":
        mask[:] = True
    elif patch_scope == "visual-prompt":
        prompt_ids = corrupt_full[0, :corrupt_prompt_len]
        mask[:corrupt_prompt_len] = prompt_ids >= BOV
    elif patch_scope == "eoi-prompt":
        prompt_ids = corrupt_full[0, :corrupt_prompt_len]
        mask[:corrupt_prompt_len] = prompt_ids == EOI
    elif patch_scope in ("post-image-text-prompt", "question-text-no-score"):
        prompt_ids = corrupt_full[0, :corrupt_prompt_len]
        eoi_positions = (prompt_ids == EOI).nonzero(as_tuple=False).flatten()
        start = int(eoi_positions[-1].item()) + 1 if int(eoi_positions.numel()) > 0 else 0
        post_image = torch.zeros_like(prompt_ids, dtype=torch.bool)
        post_image[start:corrupt_prompt_len] = True
        text_like = prompt_ids < BOV
        special_image = (prompt_ids == BOI) | (prompt_ids == IMG) | (prompt_ids == EOI)
        mask[:corrupt_prompt_len] = post_image & text_like & ~special_image
        if patch_scope == "question-text-no-score":
            score_start = max(0, corrupt_prompt_len - 1)
            mask[score_start:corrupt_prompt_len] = False
    elif patch_scope == "score-positions":
        start = max(0, corrupt_prompt_len - 1)
        mask[start : start + answer_len] = True
    else:
        raise ValueError(f"unknown patch_scope: {patch_scope}")
    return mask


def selected_positions(mask: torch.Tensor) -> List[int]:
    return [int(pos) for pos in mask.nonzero(as_tuple=False).flatten().detach().cpu().tolist()]


class MLPIntermediateCollector:
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


def collect_intermediate(model, input_ids: torch.Tensor, layer_idx: int) -> torch.Tensor:
    collector = MLPIntermediateCollector(model, layer_idx)
    collector.install()
    try:
        model.model(input_ids=input_ids, use_cache=False, return_dict=True)
    finally:
        collector.remove()
    if collector.intermediate is None:
        raise RuntimeError(f"missing intermediate for layer {layer_idx}")
    return collector.intermediate


class LowRankMLPPatcher:
    def __init__(self, model, layer_idx: int, basis: torch.Tensor, clean_intermediate: torch.Tensor, positions: Sequence[int]):
        self.model = model
        self.layer_idx = int(layer_idx)
        self.basis = basis
        self.clean_intermediate = clean_intermediate
        self.positions = [int(pos) for pos in positions]
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
        valid_positions = [
            pos
            for pos in self.positions
            if pos < self.clean_intermediate.shape[1] and pos < self._corrupt_intermediate.shape[1]
        ]
        if not valid_positions:
            return output
        pos_idx = torch.tensor(valid_positions, device=output.device, dtype=torch.long)
        basis = self.basis.to(device=output.device, dtype=module.down_proj.weight.dtype)
        clean = self.clean_intermediate.to(device=output.device, dtype=module.down_proj.weight.dtype).index_select(1, pos_idx)
        corrupt = self._corrupt_intermediate.to(device=output.device, dtype=module.down_proj.weight.dtype).index_select(1, pos_idx)
        coeff = torch.matmul(clean - corrupt, basis)
        projected_delta = torch.matmul(coeff, basis.transpose(0, 1))
        contribution_delta = F.linear(projected_delta, module.down_proj.weight).to(output.dtype)
        patched = output.clone()
        patched[:, pos_idx, :] = patched[:, pos_idx, :] + contribution_delta
        self._corrupt_intermediate = None
        return patched


def run_direction(
    model,
    cfg,
    tokenizer,
    vq_model,
    clean_row,
    corrupt_row,
    layer_idx: int,
    group: str,
    basis: torch.Tensor,
    question_template: str,
    patch_scope: str,
) -> dict:
    question = question_template.format(shape=clean_row["shape"])
    answer_text, answer_ids = encode_answer(tokenizer, clean_row["fill_color"], model.device)
    clean_prompt = build_prompt_ids(cfg, tokenizer, vq_model, clean_row["local_path"], question, model.device)
    corrupt_prompt = build_prompt_ids(cfg, tokenizer, vq_model, corrupt_row["local_path"], question, model.device)
    if clean_prompt.shape != corrupt_prompt.shape:
        raise RuntimeError(f"prompt shape mismatch: {clean_prompt.shape} vs {corrupt_prompt.shape}")

    clean = answer_nll(model, clean_prompt, answer_ids)
    torch.cuda.empty_cache()
    corrupt = answer_nll(model, corrupt_prompt, answer_ids)
    torch.cuda.empty_cache()
    clean_full = clean["full_ids"]
    corrupt_full = torch.cat([corrupt_prompt, answer_ids], dim=1)
    patch_mask = build_patch_mask(corrupt_full, corrupt_prompt.shape[1], answer_ids.shape[1], patch_scope)
    positions = selected_positions(patch_mask)
    clean_intermediate = collect_intermediate(model, clean_full, layer_idx)
    torch.cuda.empty_cache()
    patcher = LowRankMLPPatcher(model, layer_idx, basis, clean_intermediate, positions)
    patcher.install()
    try:
        patched = answer_nll(model, corrupt_prompt, answer_ids)
    finally:
        patcher.remove()
    torch.cuda.empty_cache()

    denom = corrupt["nll_sum"] - clean["nll_sum"]
    valid_gap = denom > 1e-9
    recovery = (corrupt["nll_sum"] - patched["nll_sum"]) / denom if valid_gap else float("nan")
    return {
        "layer": layer_idx,
        "group": group,
        "rank": int(basis.shape[1]),
        "pair_id": clean_row["pair_id"],
        "clean_image_id": clean_row["image_id"],
        "corrupt_image_id": corrupt_row["image_id"],
        "shape": clean_row["shape"],
        "clean_color": clean_row["fill_color"],
        "corrupt_color": corrupt_row["fill_color"],
        "answer_text": answer_text,
        "answer_tokens": clean["answer_tokens"],
        "clean_nll_sum": clean["nll_sum"],
        "corrupt_nll_sum": corrupt["nll_sum"],
        "patched_nll_sum": patched["nll_sum"],
        "corrupt_minus_clean_nll_sum": denom,
        "patched_delta_vs_corrupt_nll_sum": patched["nll_sum"] - corrupt["nll_sum"],
        "recovery_fraction": recovery,
        "valid_recovery_gap": int(valid_gap),
        "patch_scope": patch_scope,
        "patched_positions": int(patch_mask.sum().item()),
    }


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
        "# Understanding MLP Low-Rank Patch Recovery",
        "",
        f"- Config: `{args.cfg}`",
        f"- Manifest: `{args.manifest}`",
        f"- Basis path: `{args.basis_path}`",
        f"- Layer: `{args.layer}`",
        f"- Ranks: `{args.ranks}`",
        f"- Basis types: `{args.basis_types}`",
        f"- Patch scope: `{args.patch_scope}`",
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
    (out_dir / "understanding_mlp_lowrank_patch_recovery_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    ranks = parse_int_list(args.ranks)
    basis_types = parse_basis_types(args.basis_types)
    basis_payload = torch.load(args.basis_path, map_location="cpu")
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
    model.eval()
    cfg.special_token_ids = {k: tokenizer.encode(v)[0] for k, v in cfg.special_tokens.items()}

    groups: List[Tuple[str, torch.Tensor]] = []
    for basis_type in basis_types:
        key = "pca_basis" if basis_type == "pca" else "random_basis"
        basis = basis_payload[key]
        for rank in ranks:
            if rank > basis.shape[1]:
                raise ValueError(f"rank {rank} exceeds {key} width {basis.shape[1]}")
            groups.append((f"{basis_type}{rank}", basis[:, :rank].contiguous()))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for _pair_id, a_row, b_row in tqdm(grouped_manifest(args.manifest, args.max_pairs), desc="pairs"):
        for clean_row, corrupt_row in ((a_row, b_row), (b_row, a_row)):
            for group, basis in groups:
                rows.append(
                    run_direction(
                        model,
                        cfg,
                        tokenizer,
                        vq_model,
                        clean_row,
                        corrupt_row,
                        args.layer,
                        group,
                        basis,
                        args.question_template,
                        args.patch_scope,
                    )
                )

    write_csv_rows(rows, out_dir / "understanding_mlp_lowrank_patch_recovery.csv")
    summary_rows = summarize(rows)
    write_csv_rows(summary_rows, out_dir / "summary.csv")
    write_report(out_dir, args, summary_rows)
    metadata = {
        "cfg": args.cfg,
        "manifest": args.manifest,
        "basis_path": args.basis_path,
        "layer": args.layer,
        "ranks": args.ranks,
        "basis_types": args.basis_types,
        "max_pairs": args.max_pairs,
        "patch_scope": args.patch_scope,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[INFO] understanding MLP low-rank patch recovery saved to {out_dir}")


if __name__ == "__main__":
    main()
