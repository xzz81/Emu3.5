#!/usr/bin/env python3
# Copyright 2026
# SPDX-License-Identifier: Apache-2.0

"""Clean/corrupt residual-stream patching for Emu3.5 hue generation."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
from pathlib import Path
import random
import statistics
import sys
from typing import Dict, List, Sequence, Tuple
import zlib

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
from src.utils.generation_utils import generate  # noqa: E402
from src.utils.logits_processor import BOI, BOV, EOI, EOL, IMG  # noqa: E402
from src.utils.model_utils import build_emu3p5  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="configs/ume_main_t2i_counterfactual_pairs_hueonly_seed69.py")
    parser.add_argument("--layers", default="60,61,62", help="Comma-separated decoder layer ids, or 'all'.")
    parser.add_argument(
        "--component",
        choices=["full-residual", "input-embedding"],
        default="full-residual",
        help="Patch decoder layer outputs or input token embeddings.",
    )
    parser.add_argument("--max-pairs", type=int, default=1)
    parser.add_argument(
        "--pair-id",
        action="append",
        default=[],
        help="Only run matching pair id(s). Can be passed multiple times.",
    )
    parser.add_argument(
        "--patch-scope",
        choices=["prompt", "color-token", "object-token", "visual-score-positions", "all-score-positions"],
        default="prompt",
        help="Sequence positions where clean residual states are patched into corrupt-prompt scoring.",
    )
    parser.add_argument(
        "--score-scope",
        choices=["visual", "visual-and-structure", "all-target"],
        default="visual",
        help="Target generated-token subset used for NLL/recovery.",
    )
    parser.add_argument("--max-target-tokens", type=int, default=0, help="Optional trim of generated target ids.")
    parser.add_argument("--target-height", type=int, default=None, help="Override cfg.target_height.")
    parser.add_argument("--target-width", type=int, default=None, help="Override cfg.target_width.")
    parser.add_argument("--image-area", type=int, default=None, help="Override cfg.image_area.")
    parser.add_argument("--generation-max-new-tokens", type=int, default=None, help="Override generation max_new_tokens.")
    parser.add_argument("--classifier-free-guidance", type=float, default=None, help="Override CFG scale.")
    parser.add_argument(
        "--target-cache-dir",
        default=None,
        help="Directory for cached clean generated target token ids. Defaults to OUT_DIR/target_cache.",
    )
    parser.add_argument("--disable-target-cache", action="store_true", help="Always regenerate clean target tokens.")
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def load_cfg(path: str):
    cfg_path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location(cfg_path.stem, cfg_path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"could not load config: {cfg_path}")
    sys.path.insert(0, str(REPO_ROOT))
    spec.loader.exec_module(module)
    return module


def parse_layers(text: str, num_layers: int) -> List[int]:
    if text.strip().lower() == "all":
        return list(range(num_layers))
    layers = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        layer_idx = int(item)
        if layer_idx < 0 or layer_idx >= num_layers:
            raise ValueError(f"layer out of range: {layer_idx} for num_layers={num_layers}")
        layers.append(layer_idx)
    if not layers:
        raise ValueError("provide at least one layer")
    return sorted(set(layers))


def normalize_prompt_rows(cfg, max_pairs: int):
    prompts = cfg.prompts
    if not isinstance(prompts, dict):
        raise ValueError("generation residual patching expects cfg.prompts to be a dict with pair_metadata")
    metadata = getattr(cfg, "pair_metadata", {})
    by_pair: Dict[str, List[dict]] = {}
    for sample_id, prompt in prompts.items():
        meta = dict(metadata.get(sample_id, {}))
        pair_id = meta.get("pair_id")
        if not pair_id:
            continue
        value = str(meta.get("counterfactual_value", ""))
        parts = value.split()
        color = parts[0] if parts else ""
        shape = parts[1] if len(parts) > 1 else ""
        by_pair.setdefault(pair_id, []).append(
            {
                "sample_id": sample_id,
                "prompt": prompt,
                "pair_id": pair_id,
                "variant": meta.get("variant", sample_id),
                "counterfactual_value": value,
                "color": color,
                "shape": shape,
            }
        )
    pairs = []
    for pair_id, rows in sorted(by_pair.items()):
        if len(rows) != 2:
            continue
        rows = sorted(rows, key=lambda row: str(row["variant"]))
        pairs.append((pair_id, rows[0], rows[1]))
        if max_pairs and len(pairs) >= max_pairs:
            break
    return pairs


def filter_pairs(pairs, pair_ids: Sequence[str]):
    selected = set(pair_ids)
    if not selected:
        return pairs
    return [pair for pair in pairs if pair[0] in selected]


def build_prompt_ids(cfg, tokenizer, prompt_text: str, device):
    if hasattr(cfg, "build_unc_and_template"):
        unc_prompt, template = cfg.build_unc_and_template(getattr(cfg, "task_type", "unknown"), with_image=False)
    else:
        unc_prompt, template = cfg.unc_prompt, cfg.template
    prompt = template.format(question=prompt_text)
    input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    if input_ids[0, 0] != cfg.special_token_ids["BOS"]:
        bos = torch.tensor([[cfg.special_token_ids["BOS"]]], device=device, dtype=input_ids.dtype)
        input_ids = torch.cat([bos, input_ids], dim=1)
    unconditional_ids = tokenizer.encode(unc_prompt, return_tensors="pt", add_special_tokens=False).to(device)
    full_unc_ids = None
    if hasattr(cfg, "img_unc_prompt"):
        full_unc_ids = tokenizer.encode(cfg.img_unc_prompt, return_tensors="pt", add_special_tokens=False).to(device)
    return input_ids, unconditional_ids, full_unc_ids


def trim_to_first_image(token_ids: Sequence[int], eoi_token_id: int) -> List[int]:
    out = [int(token_id) for token_id in token_ids]
    for idx, token_id in enumerate(out):
        if int(token_id) == int(eoi_token_id):
            return out[: idx + 1]
    return out


@torch.no_grad()
def generate_target_ids(cfg, model, tokenizer, clean_prompt_ids, unconditional_ids, full_unc_ids, sample_seed: int):
    random.seed(sample_seed)
    torch.manual_seed(sample_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(sample_seed)
    generated = list(
        generate(
            cfg,
            model,
            tokenizer,
            clean_prompt_ids,
            unconditional_ids,
            full_unc_ids,
            force_same_image_size=True,
        )
    )
    if not generated:
        raise RuntimeError("generation returned no token ids")
    target_ids = trim_to_first_image(generated[0].tolist() if hasattr(generated[0], "tolist") else generated[0], cfg.special_token_ids["EOI"])
    return torch.tensor([target_ids], dtype=clean_prompt_ids.dtype, device=clean_prompt_ids.device)


def stable_crc(text: str) -> int:
    return zlib.crc32(text.encode("utf-8")) & 0xFFFFFFFF


def target_cache_metadata(cfg, clean_row: dict, sample_seed: int) -> dict:
    return {
        "cache_version": 1,
        "sample_id": clean_row["sample_id"],
        "prompt_crc32": stable_crc(clean_row["prompt"]),
        "task_type": getattr(cfg, "task_type", "unknown"),
        "target_height": getattr(cfg, "target_height", None),
        "target_width": getattr(cfg, "target_width", None),
        "image_area": getattr(cfg, "image_area", None),
        "max_new_tokens": cfg.sampling_params.get("max_new_tokens"),
        "classifier_free_guidance": getattr(cfg, "classifier_free_guidance", None),
        "sample_seed": int(sample_seed),
    }


def cache_path_for(cache_dir: Path, sample_id: str) -> Path:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in sample_id)
    return cache_dir / f"{safe}.json"


@torch.no_grad()
def get_target_ids(
    cfg,
    model,
    tokenizer,
    clean_row: dict,
    clean_prompt_ids,
    unconditional_ids,
    full_unc_ids,
    sample_seed: int,
    target_cache_dir: Path | None,
    disable_target_cache: bool,
):
    metadata = target_cache_metadata(cfg, clean_row, sample_seed)
    cache_path = cache_path_for(target_cache_dir, clean_row["sample_id"]) if target_cache_dir is not None else None
    if cache_path is not None and not disable_target_cache and cache_path.exists():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        if payload.get("metadata") == metadata and payload.get("target_ids"):
            ids = torch.tensor(
                [payload["target_ids"]],
                dtype=clean_prompt_ids.dtype,
                device=clean_prompt_ids.device,
            )
            return ids, "cache", str(cache_path)

    ids = generate_target_ids(cfg, model, tokenizer, clean_prompt_ids, unconditional_ids, full_unc_ids, sample_seed)
    if cache_path is not None and not disable_target_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "metadata": metadata,
            "target_ids": [int(x) for x in ids[0].detach().cpu().tolist()],
        }
        cache_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return ids, "generated", str(cache_path) if cache_path is not None else ""


def first_tensor(output):
    if isinstance(output, tuple):
        return output[0]
    return output


def replace_first_tensor(output, hidden_states):
    if isinstance(output, tuple):
        return (hidden_states,) + output[1:]
    return hidden_states


class CleanResidualCollector:
    def __init__(self, model, layers: Sequence[int]):
        self.model = model
        self.layers = list(layers)
        self.handles = []
        self.clean_hidden: Dict[int, torch.Tensor] = {}

    def install(self):
        for layer_idx in self.layers:
            layer = self.model.model.layers[layer_idx]
            self.handles.append(layer.register_forward_hook(self._hook(layer_idx)))

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _hook(self, layer_idx: int):
        def hook(_module, _inputs, output):
            self.clean_hidden[layer_idx] = first_tensor(output).detach()

        return hook


class CleanEmbeddingCollector:
    def __init__(self, model):
        self.model = model
        self.handle = None
        self.clean_hidden: torch.Tensor | None = None

    def install(self):
        self.handle = self.model.model.embed_tokens.register_forward_hook(self._hook)

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

    def _hook(self, _module, _inputs, output):
        self.clean_hidden = first_tensor(output).detach()


class ResidualPatcher:
    def __init__(
        self,
        model,
        layer_idx: int,
        clean_hidden: torch.Tensor,
        position_mask: torch.Tensor,
        source_position_for_target: Dict[int, int] | None = None,
    ):
        self.model = model
        self.layer_idx = layer_idx
        self.clean_hidden = clean_hidden
        self.position_mask = position_mask
        self.source_position_for_target = source_position_for_target or {}
        self.handle = None

    def install(self):
        layer = self.model.model.layers[self.layer_idx]
        self.handle = layer.register_forward_hook(self._hook)

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

    def _hook(self, _module, _inputs, output):
        corrupt_hidden = first_tensor(output)
        seq_len = min(corrupt_hidden.shape[1], self.clean_hidden.shape[1], self.position_mask.numel())
        mask = self.position_mask[:seq_len].to(corrupt_hidden.device).to(torch.bool).view(1, seq_len, 1)
        clean = self.clean_hidden[:, :seq_len, :].to(device=corrupt_hidden.device, dtype=corrupt_hidden.dtype)
        patched = corrupt_hidden.clone()
        patched[:, :seq_len, :] = torch.where(mask, clean, patched[:, :seq_len, :])
        for target_pos, source_pos in self.source_position_for_target.items():
            if 0 <= target_pos < corrupt_hidden.shape[1] and 0 <= source_pos < self.clean_hidden.shape[1]:
                patched[:, target_pos, :] = self.clean_hidden[:, source_pos, :].to(
                    device=corrupt_hidden.device,
                    dtype=corrupt_hidden.dtype,
                )
        return replace_first_tensor(output, patched)


class EmbeddingPatcher:
    def __init__(
        self,
        model,
        clean_hidden: torch.Tensor,
        position_mask: torch.Tensor,
        source_position_for_target: Dict[int, int] | None = None,
    ):
        self.model = model
        self.clean_hidden = clean_hidden
        self.position_mask = position_mask
        self.source_position_for_target = source_position_for_target or {}
        self.handle = None

    def install(self):
        self.handle = self.model.model.embed_tokens.register_forward_hook(self._hook)

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

    def _hook(self, _module, _inputs, output):
        corrupt_hidden = first_tensor(output)
        seq_len = min(corrupt_hidden.shape[1], self.clean_hidden.shape[1], self.position_mask.numel())
        mask = self.position_mask[:seq_len].to(corrupt_hidden.device).to(torch.bool).view(1, seq_len, 1)
        clean = self.clean_hidden[:, :seq_len, :].to(device=corrupt_hidden.device, dtype=corrupt_hidden.dtype)
        patched = corrupt_hidden.clone()
        patched[:, :seq_len, :] = torch.where(mask, clean, patched[:, :seq_len, :])
        for target_pos, source_pos in self.source_position_for_target.items():
            if 0 <= target_pos < corrupt_hidden.shape[1] and 0 <= source_pos < self.clean_hidden.shape[1]:
                patched[:, target_pos, :] = self.clean_hidden[:, source_pos, :].to(
                    device=corrupt_hidden.device,
                    dtype=corrupt_hidden.dtype,
                )
        return replace_first_tensor(output, patched)


@torch.no_grad()
def collect_input_embeddings(model, input_ids: torch.Tensor) -> torch.Tensor:
    collector = CleanEmbeddingCollector(model)
    collector.install()
    try:
        model.model(input_ids=input_ids, use_cache=False, return_dict=True)
    finally:
        collector.remove()
    if collector.clean_hidden is None:
        raise RuntimeError("missing input embedding states")
    return collector.clean_hidden


def mean_abs_patch_delta(
    clean_hidden: torch.Tensor,
    corrupt_hidden: torch.Tensor,
    position_mask: torch.Tensor,
    source_position_for_target: Dict[int, int] | None = None,
) -> float:
    source_position_for_target = source_position_for_target or {}
    seq_len = min(clean_hidden.shape[1], corrupt_hidden.shape[1], position_mask.numel())
    pairs = []
    mask_values = position_mask[:seq_len].detach().cpu().to(torch.bool).tolist()
    for pos, should_patch in enumerate(mask_values):
        if should_patch:
            pairs.append((int(source_position_for_target.get(pos, pos)), pos))
    if not pairs:
        return float("nan")
    source_idx = torch.tensor([source for source, _target in pairs], device=clean_hidden.device, dtype=torch.long)
    target_idx = torch.tensor([target for _source, target in pairs], device=corrupt_hidden.device, dtype=torch.long)
    clean = clean_hidden.index_select(1, source_idx).to(device=corrupt_hidden.device, dtype=corrupt_hidden.dtype)
    corrupt = corrupt_hidden.index_select(1, target_idx)
    return float((clean - corrupt).abs().mean().item())


def build_score_mask(target_ids: torch.Tensor, score_scope: str) -> torch.Tensor:
    target = target_ids[0]
    if score_scope == "visual":
        return target >= BOV
    if score_scope == "visual-and-structure":
        return (target >= BOV) | (target == BOI) | (target == IMG) | (target == EOI) | (target == EOL)
    if score_scope == "all-target":
        return torch.ones_like(target, dtype=torch.bool)
    raise ValueError(f"unknown score_scope: {score_scope}")


@torch.no_grad()
def target_nll(model, prompt_ids, target_ids, score_scope: str):
    full_ids = torch.cat([prompt_ids, target_ids], dim=1)
    prompt_len = prompt_ids.shape[1]
    outputs = model.model(input_ids=full_ids, use_cache=False, return_dict=True)
    hidden_states = outputs.last_hidden_state[:, prompt_len - 1 : -1, :]
    logits = model.lm_head(hidden_states).float()
    labels = target_ids.to(logits.device)
    losses = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        reduction="none",
    ).view(labels.shape)
    score_mask = build_score_mask(labels, score_scope).to(losses.device)
    selected = losses[:, score_mask]
    if selected.numel() == 0:
        raise RuntimeError(f"no target tokens selected by score_scope={score_scope}")
    return {
        "full_ids": full_ids,
        "nll_mean": float(selected.mean().item()),
        "nll_sum": float(selected.sum().item()),
        "target_tokens": int(labels.numel()),
        "scored_tokens": int(selected.numel()),
        "visual_tokens": int((labels[0] >= BOV).sum().item()),
    }


def find_subsequence(haystack: List[int], needle: List[int]) -> List[int]:
    if not needle:
        return []
    matches = []
    n = len(needle)
    for idx in range(0, len(haystack) - n + 1):
        if haystack[idx : idx + n] == needle:
            matches.extend(range(idx, idx + n))
    return matches


def text_token_positions(tokenizer, prompt_ids: torch.Tensor, text_value: str) -> List[int]:
    prompt = prompt_ids[0].detach().cpu().tolist()
    variants = [text_value, " " + text_value]
    for text in variants:
        ids = tokenizer.encode(text, add_special_tokens=False)
        positions = find_subsequence(prompt, [int(x) for x in ids])
        if positions:
            return positions
    return []


def color_token_positions(tokenizer, prompt_ids: torch.Tensor, color: str) -> List[int]:
    return text_token_positions(tokenizer, prompt_ids, color)


def object_token_positions(tokenizer, prompt_ids: torch.Tensor, shape: str) -> List[int]:
    return text_token_positions(tokenizer, prompt_ids, shape)


def map_source_span_to_target_span(source_positions: List[int], target_positions: List[int]) -> Dict[int, int]:
    if not source_positions or not target_positions:
        return {}
    if len(source_positions) == len(target_positions):
        return {int(target): int(source) for source, target in zip(source_positions, target_positions)}
    if len(target_positions) == 1:
        return {int(target_positions[0]): int(source_positions[0])}
    if len(source_positions) == 1:
        return {int(target): int(source_positions[0]) for target in target_positions}
    mapping: Dict[int, int] = {}
    for target_idx, target in enumerate(target_positions):
        source_idx = round(target_idx * (len(source_positions) - 1) / (len(target_positions) - 1))
        mapping[int(target)] = int(source_positions[source_idx])
    return mapping


def build_patch_mask(
    tokenizer,
    clean_full: torch.Tensor,
    corrupt_full: torch.Tensor,
    clean_prompt_len: int,
    corrupt_prompt_len: int,
    target_ids: torch.Tensor,
    patch_scope: str,
    clean_color: str,
    corrupt_color: str,
    clean_shape: str,
    corrupt_shape: str,
) -> Tuple[torch.Tensor, int, str, Dict[int, int]]:
    mask = torch.zeros(corrupt_full.shape[1], dtype=torch.bool, device=corrupt_full.device)
    note = ""
    source_position_for_target: Dict[int, int] = {}
    if patch_scope == "prompt":
        mask[:corrupt_prompt_len] = True
    elif patch_scope in ("color-token", "object-token"):
        if patch_scope == "color-token":
            clean_positions = color_token_positions(tokenizer, clean_full[:, :clean_prompt_len], clean_color)
            corrupt_positions = color_token_positions(tokenizer, corrupt_full[:, :corrupt_prompt_len], corrupt_color)
        else:
            if not clean_shape or not corrupt_shape:
                raise RuntimeError("object-token patch requires clean/corrupt shape values")
            clean_positions = object_token_positions(tokenizer, clean_full[:, :clean_prompt_len], clean_shape)
            corrupt_positions = object_token_positions(tokenizer, corrupt_full[:, :corrupt_prompt_len], corrupt_shape)
        if not clean_positions or not corrupt_positions:
            raise RuntimeError(
                f"{patch_scope} patch requires matched token spans, got clean={clean_positions}, corrupt={corrupt_positions}"
            )
        if clean_positions != corrupt_positions:
            note = f"clean_{patch_scope}_positions={clean_positions};corrupt_{patch_scope}_positions={corrupt_positions}"
        if len(clean_positions) != len(corrupt_positions):
            note = (note + ";" if note else "") + (
                f"{patch_scope}_span_len_mismatch_clean={len(clean_positions)}_corrupt={len(corrupt_positions)}"
            )
        source_position_for_target = map_source_span_to_target_span(clean_positions, corrupt_positions)
        for corrupt_pos in corrupt_positions:
            mask[corrupt_pos] = True
    elif patch_scope in ("visual-score-positions", "all-score-positions"):
        target = target_ids[0]
        if patch_scope == "visual-score-positions":
            target_mask = target >= BOV
        else:
            target_mask = torch.ones_like(target, dtype=torch.bool)
        for target_idx, should_patch in enumerate(target_mask.detach().cpu().tolist()):
            if should_patch:
                target_score_pos = corrupt_prompt_len + target_idx - 1
                source_score_pos = clean_prompt_len + target_idx - 1
                if target_score_pos >= 0:
                    mask[target_score_pos] = True
                    if source_score_pos >= 0 and source_score_pos != target_score_pos:
                        source_position_for_target[int(target_score_pos)] = int(source_score_pos)
        if clean_prompt_len != corrupt_prompt_len:
            note = f"score_position_source_offset={clean_prompt_len - corrupt_prompt_len}"
    else:
        raise ValueError(f"unknown patch_scope: {patch_scope}")

    if clean_full.shape[1] != corrupt_full.shape[1]:
        note = (note + ";" if note else "") + f"seq_len_mismatch_clean={clean_full.shape[1]}_corrupt={corrupt_full.shape[1]}"
    return mask, int(mask.sum().item()), note, source_position_for_target


def run_direction(
    model,
    cfg,
    tokenizer,
    clean_row,
    corrupt_row,
    layers,
    component,
    patch_scope,
    score_scope,
    max_target_tokens,
    target_cache_dir: Path | None,
    disable_target_cache: bool,
):
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
        disable_target_cache,
    )
    if max_target_tokens and target_ids.shape[1] > max_target_tokens:
        target_ids = target_ids[:, :max_target_tokens]

    clean = target_nll(model, clean_prompt, target_ids, score_scope)
    torch.cuda.empty_cache()
    corrupt = target_nll(model, corrupt_prompt, target_ids, score_scope)
    torch.cuda.empty_cache()
    clean_full = clean["full_ids"]
    corrupt_full = torch.cat([corrupt_prompt, target_ids], dim=1)

    patch_mask, patched_positions, patch_note, source_position_for_target = build_patch_mask(
        tokenizer,
        clean_full,
        corrupt_full,
        clean_prompt.shape[1],
        corrupt_prompt.shape[1],
        target_ids,
        patch_scope,
        clean_row["color"],
        corrupt_row["color"],
        clean_row["shape"],
        corrupt_row["shape"],
    )
    rows = []

    def make_row(layer_idx, patched, embedding_delta_mean_abs):
        denom = corrupt["nll_sum"] - clean["nll_sum"]
        valid_recovery_gap = denom > 1e-9
        recovery = (corrupt["nll_sum"] - patched["nll_sum"]) / denom if valid_recovery_gap else float("nan")
        return {
            "layer": layer_idx,
            "component": component,
            "pair_id": clean_row["pair_id"],
            "clean_sample_id": clean_row["sample_id"],
            "corrupt_sample_id": corrupt_row["sample_id"],
            "shape": clean_row["shape"],
            "clean_color": clean_row["color"],
            "corrupt_color": corrupt_row["color"],
            "target_tokens": clean["target_tokens"],
            "visual_tokens": clean["visual_tokens"],
            "scored_tokens": clean["scored_tokens"],
            "target_source": target_source,
            "target_cache_path": target_cache_path,
            "clean_nll_sum": clean["nll_sum"],
            "corrupt_nll_sum": corrupt["nll_sum"],
            "patched_nll_sum": patched["nll_sum"],
            "clean_nll_mean": clean["nll_mean"],
            "corrupt_nll_mean": corrupt["nll_mean"],
            "patched_nll_mean": patched["nll_mean"],
            "corrupt_minus_clean_nll_sum": denom,
            "patched_delta_vs_corrupt_nll_sum": patched["nll_sum"] - corrupt["nll_sum"],
            "recovery_fraction": recovery,
            "valid_recovery_gap": int(valid_recovery_gap),
            "patch_scope": patch_scope,
            "score_scope": score_scope,
            "patched_positions": patched_positions,
            "patch_note": patch_note,
            "embedding_clean_corrupt_patch_delta_mean_abs": embedding_delta_mean_abs,
        }

    if component == "input-embedding":
        clean_embedding = collect_input_embeddings(model, clean_full)
        torch.cuda.empty_cache()
        corrupt_embedding = collect_input_embeddings(model, corrupt_full)
        torch.cuda.empty_cache()
        embedding_delta_mean_abs = mean_abs_patch_delta(
            clean_embedding,
            corrupt_embedding,
            patch_mask,
            source_position_for_target=source_position_for_target,
        )
        patcher = EmbeddingPatcher(
            model,
            clean_embedding,
            patch_mask,
            source_position_for_target=source_position_for_target,
        )
        patcher.install()
        try:
            patched = target_nll(model, corrupt_prompt, target_ids, score_scope)
        finally:
            patcher.remove()
        torch.cuda.empty_cache()
        rows.append(make_row(-1, patched, embedding_delta_mean_abs))
        return rows

    collector = CleanResidualCollector(model, layers)
    collector.install()
    try:
        model.model(input_ids=clean_full, use_cache=False, return_dict=True)
    finally:
        collector.remove()
    torch.cuda.empty_cache()

    for layer_idx in layers:
        if layer_idx not in collector.clean_hidden:
            raise RuntimeError(f"missing clean hidden for layer {layer_idx}")
        patcher = ResidualPatcher(
            model,
            layer_idx,
            collector.clean_hidden[layer_idx],
            patch_mask,
            source_position_for_target=source_position_for_target,
        )
        patcher.install()
        try:
            patched = target_nll(model, corrupt_prompt, target_ids, score_scope)
        finally:
            patcher.remove()
        torch.cuda.empty_cache()
        rows.append(make_row(layer_idx, patched, float("nan")))
    return rows


def summarize(rows: List[dict]) -> List[dict]:
    by_layer: Dict[int, List[dict]] = {}
    all_by_layer: Dict[int, List[dict]] = {}
    for row in rows:
        all_by_layer.setdefault(int(row["layer"]), []).append(row)
        if math.isfinite(float(row["recovery_fraction"])):
            by_layer.setdefault(int(row["layer"]), []).append(row)
    summary = []
    for layer_idx, all_rows in sorted(all_by_layer.items()):
        layer_rows = by_layer.get(layer_idx, [])
        if not layer_rows:
            summary.append(
                {
                    "layer": layer_idx,
                    "n_directions": 0,
                    "invalid_gap_directions": len(all_rows),
                    "helped_directions": 0,
                    "mean_recovery_fraction": float("nan"),
                    "median_recovery_fraction": float("nan"),
                    "mean_patch_delta_vs_corrupt_nll_sum": float("nan"),
                    "sum_corrupt_minus_clean_nll_sum": sum(float(row["corrupt_minus_clean_nll_sum"]) for row in all_rows),
                    "sum_patch_delta_vs_corrupt_nll_sum": sum(float(row["patched_delta_vs_corrupt_nll_sum"]) for row in all_rows),
                }
            )
            continue
        recoveries = [float(row["recovery_fraction"]) for row in layer_rows]
        deltas = [float(row["patched_delta_vs_corrupt_nll_sum"]) for row in layer_rows]
        gaps = [float(row["corrupt_minus_clean_nll_sum"]) for row in layer_rows]
        summary.append(
            {
                "layer": layer_idx,
                "n_directions": len(layer_rows),
                "invalid_gap_directions": len(all_rows) - len(layer_rows),
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
        "# Stage 8 Report: Generation Residual Stream Patch Recovery",
        "",
        "## Purpose",
        "",
        "This experiment tests whether clean-prompt residual stream states can recover the likelihood of clean generated visual tokens under a corrupt-prompt hue run.",
        "",
        "## Setup",
        "",
        f"- Config: `{args.cfg}`",
        f"- Component: `{args.component}`",
        f"- Layers: `{args.layers}`",
        f"- Patch scope: `{args.patch_scope}`",
        f"- Score scope: `{args.score_scope}`",
        f"- Max pairs: `{args.max_pairs}`",
        f"- Output CSV: `generation_residual_patch_recovery.csv`",
        "",
        "Metric:",
        "",
        "```text",
        "recovery = (corrupt_nll - patched_nll) / (corrupt_nll - clean_nll)",
        "```",
        "",
        "Positive recovery means the patch moved corrupt-prompt scoring toward clean-prompt scoring for the clean generated visual-token target.",
        "",
        "## Aggregate Results",
        "",
        "| layer | valid directions | invalid gaps | helped | mean recovery | median recovery | sum delta NLL vs corrupt |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {layer} | {n_directions} | {invalid_gap_directions} | {helped_directions} | {mean_recovery_fraction:+.6f} | "
            "{median_recovery_fraction:+.6f} | {sum_patch_delta_vs_corrupt_nll_sum:+.6f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Interpretation Guide",
            "",
            "- Strong positive recovery supports a prompt-to-visual-token generation path at the patched layer/scope.",
            "- Directions where `corrupt_nll <= clean_nll` are marked as invalid gaps because recovery is not well-defined.",
            "- Prompt-scope patching is a path existence test, not a neuron-level localization.",
            "- Input-embedding patching tests the token-embedding boundary rather than a Transformer layer.",
            "- Near-zero recovery means the tested residual states are not sufficient for teacher-forced visual-token recovery.",
            "- Negative recovery means the patch makes the clean generated visual tokens less likely than the corrupt-prompt baseline.",
            "",
        ]
    )
    (out_dir / "generation_residual_patch_recovery_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    cfg = load_cfg(args.cfg)
    if args.target_height is not None:
        cfg.target_height = args.target_height
    if args.target_width is not None:
        cfg.target_width = args.target_width
    if args.image_area is not None:
        cfg.image_area = args.image_area
    if args.generation_max_new_tokens is not None:
        cfg.max_new_tokens = args.generation_max_new_tokens
        cfg.sampling_params["max_new_tokens"] = args.generation_max_new_tokens
    if args.classifier_free_guidance is not None:
        cfg.classifier_free_guidance = args.classifier_free_guidance
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
    cfg.special_token_ids = {k: tokenizer.encode(v)[0] for k, v in cfg.special_tokens.items()}
    layers = [-1] if args.component == "input-embedding" else parse_layers(args.layers, len(model.model.layers))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target_cache_dir = None
    if not args.disable_target_cache:
        target_cache_dir = Path(args.target_cache_dir) if args.target_cache_dir else out_dir / "target_cache"
        target_cache_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    pairs = filter_pairs(normalize_prompt_rows(cfg, args.max_pairs), args.pair_id)
    if not pairs:
        raise SystemExit("No generation pairs matched the requested filters.")
    for _pair_id, a_row, b_row in tqdm(pairs):
        for clean_row, corrupt_row in ((a_row, b_row), (b_row, a_row)):
            rows.extend(
                run_direction(
                    model,
                    cfg,
                    tokenizer,
                    clean_row,
                    corrupt_row,
                    layers,
                    args.component,
                    args.patch_scope,
                    args.score_scope,
                    args.max_target_tokens,
                    target_cache_dir,
                    args.disable_target_cache,
                )
            )

    write_csv_rows(rows, out_dir / "generation_residual_patch_recovery.csv")
    summary_rows = summarize(rows)
    if summary_rows:
        write_csv_rows(summary_rows, out_dir / "summary.csv")
    write_report(out_dir, args, summary_rows)
    metadata = {
        "cfg": args.cfg,
        "component": args.component,
        "layers": args.layers,
        "patch_scope": args.patch_scope,
        "score_scope": args.score_scope,
        "max_pairs": args.max_pairs,
        "pair_id": args.pair_id,
        "max_target_tokens": args.max_target_tokens,
        "target_cache_dir": str(target_cache_dir) if target_cache_dir is not None else "",
        "disable_target_cache": bool(args.disable_target_cache),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[INFO] generation residual patch recovery saved to {out_dir}")


if __name__ == "__main__":
    main()
