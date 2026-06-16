#!/usr/bin/env python3
# Copyright 2026
# SPDX-License-Identifier: Apache-2.0

"""Decoded-image validation for generation target-side residual patching.

This is a Stage 24 bridge from teacher-forced visual-token likelihood to
free decoded images. It uses a validated clean raw-target cache as the source
trajectory, collects clean residual states on clean_prompt + clean_target, and
patches the corresponding target-side score positions during corrupt decoding.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import random
import sys
from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F
from transformers import GenerationConfig
from transformers.generation import LogitsProcessorList, StoppingCriteriaList
from tqdm import tqdm

try:
    import torchvision  # noqa: F401
except RuntimeError as exc:
    if "operator torchvision::nms does not exist" in str(exc):
        try:
            _TORCHVISION_SCHEMA_LIB = torch.library.Library("torchvision", "DEF")
            _TORCHVISION_SCHEMA_LIB.define("nms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor")
        except Exception:
            pass
except Exception:
    pass

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_generation_patched_decode_hue import (  # noqa: E402
    NoCFGImageFormatLogitsProcessor,
    StopAfterCompletedImagesCriteria,
    classify_decoded,
    decode_image,
)
from scripts.run_generation_residual_patch_recovery import (  # noqa: E402
    CleanResidualCollector,
    build_prompt_ids,
    cache_path_for,
    color_token_positions,
    filter_pairs,
    first_tensor,
    load_cfg,
    map_source_span_to_target_span,
    normalize_prompt_rows,
    object_token_positions,
    replace_first_tensor,
    trim_to_first_image,
)
from src.utils.logits_processor import BOV  # noqa: E402
from src.utils.model_utils import build_emu3p5  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--target-cache-dir", required=True)
    parser.add_argument("--layer", type=int, default=61)
    parser.add_argument(
        "--component",
        choices=[
            "full-residual",
            "input-embedding",
            "self-attn",
            "mlp",
            "mlp-lowrank",
            "attention-heads",
            "attention-heads-mlp",
            "self-attn-mlp",
        ],
        default="full-residual",
    )
    parser.add_argument(
        "--heads",
        default="",
        help="Comma-separated attention head ids for --component attention-heads or attention-heads-mlp.",
    )
    parser.add_argument("--lowrank-basis-path", default="")
    parser.add_argument("--lowrank-basis-kind", choices=["pca", "random"], default="pca")
    parser.add_argument("--lowrank-rank", type=int, default=0)
    parser.add_argument(
        "--patch-scope",
        choices=["prompt", "color-token", "object-token", "visual-score-positions", "all-score-positions"],
        default="visual-score-positions",
    )
    parser.add_argument("--max-pairs", type=int, default=1)
    parser.add_argument("--max-directions", type=int, default=0)
    parser.add_argument("--pair-id", action="append", default=[])
    parser.add_argument("--target-height", type=int, default=0)
    parser.add_argument("--target-width", type=int, default=0)
    parser.add_argument("--image-area", type=int, default=0)
    parser.add_argument("--generation-max-new-tokens", type=int, default=0)
    parser.add_argument("--use-cache", action="store_true")
    parser.add_argument("--include-wrong-target-control", action="store_true")
    parser.add_argument("--include-random-target-control", action="store_true")
    parser.add_argument("--seed-offset", type=int, default=93000)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_target_ids(cache_dir: Path, sample_id: str, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    path = cache_path_for(cache_dir, sample_id)
    if not path.exists():
        raise FileNotFoundError(f"missing target cache for {sample_id}: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    target_ids = payload.get("target_ids")
    if not target_ids:
        raise RuntimeError(f"target cache has no target_ids: {path}")
    return torch.tensor([[int(token_id) for token_id in target_ids]], dtype=dtype, device=device)


def parse_heads(text: str, num_heads: int) -> List[int]:
    heads = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        head_idx = int(item)
        if head_idx < 0 or head_idx >= num_heads:
            raise ValueError(f"head out of range: {head_idx} for num_heads={num_heads}")
        heads.append(head_idx)
    if not heads:
        raise ValueError("provide at least one head with --heads when using attention-head components")
    return sorted(set(heads))


class TargetSideResidualDecodePatcher:
    def __init__(
        self,
        model,
        layer_idx: int,
        clean_hidden: torch.Tensor,
        clean_prompt_len: int,
        corrupt_prompt_len: int,
        clean_target_ids: torch.Tensor,
        patch_scope: str,
        component: str,
    ):
        self.model = model
        self.layer_idx = int(layer_idx)
        self.clean_hidden = clean_hidden
        self.clean_prompt_len = int(clean_prompt_len)
        self.corrupt_prompt_len = int(corrupt_prompt_len)
        self.clean_target_ids = clean_target_ids.detach().clone()
        self.patch_scope = patch_scope
        self.component = component
        self.handle = None
        self.pre_handle = None
        self.post_handle = None
        self._abs_positions: List[int] | None = None
        self._next_cache_abs_pos: int | None = None
        self.forward_calls = 0
        self.patched_forward_calls = 0
        self.patched_positions_total = 0
        self.max_seq_len_seen = 0

    def install(self) -> None:
        self.pre_handle = self.model.register_forward_pre_hook(self._model_pre_hook, with_kwargs=True)
        self.post_handle = self.model.register_forward_hook(self._model_post_hook, with_kwargs=True)
        self._next_cache_abs_pos = None
        if self.component == "full-residual":
            module = self.model.model.layers[self.layer_idx]
        elif self.component == "self-attn":
            module = self.model.model.layers[self.layer_idx].self_attn
        elif self.component == "mlp":
            module = self.model.model.layers[self.layer_idx].mlp
        else:
            raise ValueError(f"unknown component: {self.component}")
        self.handle = module.register_forward_hook(self._hook)

    def remove(self) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
        if self.pre_handle is not None:
            self.pre_handle.remove()
            self.pre_handle = None
        if self.post_handle is not None:
            self.post_handle.remove()
            self.post_handle = None
        self._abs_positions = None
        self._next_cache_abs_pos = None

    def _extract_input_ids(self, args, kwargs):
        if kwargs and kwargs.get("input_ids") is not None:
            return kwargs["input_ids"]
        if args:
            return args[0]
        return None

    def _model_pre_hook(self, _module, args, kwargs):
        input_ids = self._extract_input_ids(args, kwargs)
        cache_position = kwargs.get("cache_position") if kwargs else None
        if cache_position is not None:
            self._abs_positions = [int(pos) for pos in cache_position.detach().cpu().flatten().tolist()]
        elif input_ids is not None and input_ids.ndim == 2:
            seq_len = int(input_ids.shape[1])
            if seq_len > 1:
                self._abs_positions = list(range(seq_len))
                self._next_cache_abs_pos = seq_len
            elif self._next_cache_abs_pos is not None:
                self._abs_positions = [int(self._next_cache_abs_pos)]
                self._next_cache_abs_pos += 1
            else:
                self._abs_positions = [0]
                self._next_cache_abs_pos = 1
        else:
            self._abs_positions = None
        return None

    def _model_post_hook(self, _module, args, kwargs, _output):
        self._abs_positions = None
        return None

    def _should_patch_target_idx(self, target_idx: int) -> bool:
        if self.patch_scope == "all-score-positions":
            return True
        token_id = int(self.clean_target_ids[0, target_idx].item())
        return token_id >= BOV

    def _hook(self, _module, _inputs, output):
        hidden = first_tensor(output)
        self.forward_calls += 1
        self.max_seq_len_seen = max(self.max_seq_len_seen, int(hidden.shape[1]))
        if hidden.shape[1] == 0:
            return output

        patched = hidden.clone()
        patched_count = 0
        target_len = int(self.clean_target_ids.shape[1])
        abs_positions = self._abs_positions
        if abs_positions is None or len(abs_positions) != hidden.shape[1]:
            abs_positions = list(range(int(hidden.shape[1])))
        for rel_pos, abs_pos in enumerate(abs_positions):
            target_idx = int(abs_pos) - self.corrupt_prompt_len + 1
            if not (0 <= target_idx < target_len):
                continue
            if not self._should_patch_target_idx(target_idx):
                continue
            source_score_pos = self.clean_prompt_len + target_idx - 1
            if 0 <= source_score_pos < self.clean_hidden.shape[1]:
                patched[:, rel_pos, :] = self.clean_hidden[:, source_score_pos, :].to(
                    device=patched.device,
                    dtype=patched.dtype,
                )
                patched_count += 1
        if patched_count:
            self.patched_forward_calls += 1
            self.patched_positions_total += patched_count
            return replace_first_tensor(output, patched)
        return output


class TargetSideEmbeddingDecodePatcher:
    def __init__(
        self,
        model,
        clean_hidden: torch.Tensor,
        clean_prompt_len: int,
        corrupt_prompt_len: int,
        clean_target_ids: torch.Tensor,
        patch_scope: str,
    ):
        self.model = model
        self.clean_hidden = clean_hidden
        self.clean_prompt_len = int(clean_prompt_len)
        self.corrupt_prompt_len = int(corrupt_prompt_len)
        self.clean_target_ids = clean_target_ids.detach().clone()
        self.patch_scope = patch_scope
        self.handle = None
        self.pre_handle = None
        self.post_handle = None
        self._abs_positions: List[int] | None = None
        self._next_cache_abs_pos: int | None = None
        self.forward_calls = 0
        self.patched_forward_calls = 0
        self.patched_positions_total = 0
        self.max_seq_len_seen = 0

    def install(self) -> None:
        self.pre_handle = self.model.register_forward_pre_hook(self._model_pre_hook, with_kwargs=True)
        self.post_handle = self.model.register_forward_hook(self._model_post_hook, with_kwargs=True)
        self._next_cache_abs_pos = None
        self.handle = self.model.model.embed_tokens.register_forward_hook(self._hook)

    def remove(self) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
        if self.pre_handle is not None:
            self.pre_handle.remove()
            self.pre_handle = None
        if self.post_handle is not None:
            self.post_handle.remove()
            self.post_handle = None
        self._abs_positions = None
        self._next_cache_abs_pos = None

    def _extract_input_ids(self, args, kwargs):
        if kwargs and kwargs.get("input_ids") is not None:
            return kwargs["input_ids"]
        if args:
            return args[0]
        return None

    def _model_pre_hook(self, _module, args, kwargs):
        input_ids = self._extract_input_ids(args, kwargs)
        cache_position = kwargs.get("cache_position") if kwargs else None
        if cache_position is not None:
            self._abs_positions = [int(pos) for pos in cache_position.detach().cpu().flatten().tolist()]
        elif input_ids is not None and input_ids.ndim == 2:
            seq_len = int(input_ids.shape[1])
            if seq_len > 1:
                self._abs_positions = list(range(seq_len))
                self._next_cache_abs_pos = seq_len
            elif self._next_cache_abs_pos is not None:
                self._abs_positions = [int(self._next_cache_abs_pos)]
                self._next_cache_abs_pos += 1
            else:
                self._abs_positions = [0]
                self._next_cache_abs_pos = 1
        else:
            self._abs_positions = None
        return None

    def _model_post_hook(self, _module, args, kwargs, _output):
        self._abs_positions = None
        return None

    def _should_patch_target_idx(self, target_idx: int) -> bool:
        if self.patch_scope == "all-score-positions":
            return True
        token_id = int(self.clean_target_ids[0, target_idx].item())
        return token_id >= BOV

    def _hook(self, _module, _inputs, output):
        hidden = first_tensor(output)
        self.forward_calls += 1
        self.max_seq_len_seen = max(self.max_seq_len_seen, int(hidden.shape[1]))
        if hidden.shape[1] == 0:
            return output

        patched = hidden.clone()
        patched_count = 0
        target_len = int(self.clean_target_ids.shape[1])
        abs_positions = self._abs_positions
        if abs_positions is None or len(abs_positions) != hidden.shape[1]:
            abs_positions = list(range(int(hidden.shape[1])))
        for rel_pos, abs_pos in enumerate(abs_positions):
            target_idx = int(abs_pos) - self.corrupt_prompt_len + 1
            if not (0 <= target_idx < target_len):
                continue
            if not self._should_patch_target_idx(target_idx):
                continue
            source_score_pos = self.clean_prompt_len + target_idx - 1
            if 0 <= source_score_pos < self.clean_hidden.shape[1]:
                patched[:, rel_pos, :] = self.clean_hidden[:, source_score_pos, :].to(
                    device=patched.device,
                    dtype=patched.dtype,
                )
                patched_count += 1
        if patched_count:
            self.patched_forward_calls += 1
            self.patched_positions_total += patched_count
            return replace_first_tensor(output, patched)
        return output


class InputEmbeddingPositionDecodePatcher:
    def __init__(
        self,
        model,
        clean_hidden: torch.Tensor,
        patch_pairs: Sequence[tuple[int, int]],
    ):
        self.model = model
        self.clean_hidden = clean_hidden
        self.patch_pairs = [(int(source), int(target)) for source, target in patch_pairs]
        self.source_for_target = {target: source for source, target in self.patch_pairs}
        self.handle = None
        self.pre_handle = None
        self.post_handle = None
        self._abs_positions: List[int] | None = None
        self._next_cache_abs_pos: int | None = None
        self.forward_calls = 0
        self.patched_forward_calls = 0
        self.patched_positions_total = 0
        self.max_seq_len_seen = 0

    def install(self) -> None:
        self.pre_handle = self.model.register_forward_pre_hook(self._model_pre_hook, with_kwargs=True)
        self.post_handle = self.model.register_forward_hook(self._model_post_hook, with_kwargs=True)
        self._next_cache_abs_pos = None
        self.handle = self.model.model.embed_tokens.register_forward_hook(self._hook)

    def remove(self) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
        if self.pre_handle is not None:
            self.pre_handle.remove()
            self.pre_handle = None
        if self.post_handle is not None:
            self.post_handle.remove()
            self.post_handle = None
        self._abs_positions = None
        self._next_cache_abs_pos = None

    def _extract_input_ids(self, args, kwargs):
        if kwargs and kwargs.get("input_ids") is not None:
            return kwargs["input_ids"]
        if args:
            return args[0]
        return None

    def _model_pre_hook(self, _module, args, kwargs):
        input_ids = self._extract_input_ids(args, kwargs)
        cache_position = kwargs.get("cache_position") if kwargs else None
        if cache_position is not None:
            self._abs_positions = [int(pos) for pos in cache_position.detach().cpu().flatten().tolist()]
        elif input_ids is not None and input_ids.ndim == 2:
            seq_len = int(input_ids.shape[1])
            if seq_len > 1:
                self._abs_positions = list(range(seq_len))
                self._next_cache_abs_pos = seq_len
            elif self._next_cache_abs_pos is not None:
                self._abs_positions = [int(self._next_cache_abs_pos)]
                self._next_cache_abs_pos += 1
            else:
                self._abs_positions = [0]
                self._next_cache_abs_pos = 1
        else:
            self._abs_positions = None
        return None

    def _model_post_hook(self, _module, args, kwargs, _output):
        self._abs_positions = None
        return None

    def _hook(self, _module, _inputs, output):
        hidden = first_tensor(output)
        self.forward_calls += 1
        self.max_seq_len_seen = max(self.max_seq_len_seen, int(hidden.shape[1]))
        if hidden.shape[1] == 0 or not self.source_for_target:
            return output

        patched = hidden.clone()
        patched_count = 0
        abs_positions = self._abs_positions
        if abs_positions is None or len(abs_positions) != hidden.shape[1]:
            abs_positions = list(range(int(hidden.shape[1])))
        for rel_pos, abs_pos in enumerate(abs_positions):
            source_pos = self.source_for_target.get(int(abs_pos))
            if source_pos is None:
                continue
            if 0 <= source_pos < self.clean_hidden.shape[1]:
                patched[:, rel_pos, :] = self.clean_hidden[:, source_pos, :].to(
                    device=patched.device,
                    dtype=patched.dtype,
                )
                patched_count += 1
        if patched_count:
            self.patched_forward_calls += 1
            self.patched_positions_total += patched_count
            return replace_first_tensor(output, patched)
        return output


class TargetSideLowRankMLPDecodePatcher:
    def __init__(
        self,
        model,
        layer_idx: int,
        basis: torch.Tensor,
        clean_intermediate: torch.Tensor,
        clean_prompt_len: int,
        corrupt_prompt_len: int,
        clean_target_ids: torch.Tensor,
        patch_scope: str,
    ):
        self.model = model
        self.layer_idx = int(layer_idx)
        self.basis = basis.detach().float().cpu().contiguous()
        self.clean_intermediate = clean_intermediate.detach()
        self.clean_prompt_len = int(clean_prompt_len)
        self.corrupt_prompt_len = int(corrupt_prompt_len)
        self.clean_target_ids = clean_target_ids.detach().clone()
        self.patch_scope = patch_scope
        self.handles = []
        self.pre_handle = None
        self.post_handle = None
        self._gate = None
        self._corrupt_intermediate = None
        self._abs_positions: List[int] | None = None
        self._next_cache_abs_pos: int | None = None
        self.forward_calls = 0
        self.patched_forward_calls = 0
        self.patched_positions_total = 0
        self.max_seq_len_seen = 0

    def install(self) -> None:
        self.pre_handle = self.model.register_forward_pre_hook(self._model_pre_hook, with_kwargs=True)
        self.post_handle = self.model.register_forward_hook(self._model_post_hook, with_kwargs=True)
        self._next_cache_abs_pos = None
        mlp = self.model.model.layers[self.layer_idx].mlp
        self.handles.append(mlp.gate_proj.register_forward_hook(self._gate_hook(mlp.act_fn)))
        self.handles.append(mlp.up_proj.register_forward_hook(self._up_hook))
        self.handles.append(mlp.register_forward_hook(self._mlp_hook))

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        if self.pre_handle is not None:
            self.pre_handle.remove()
            self.pre_handle = None
        if self.post_handle is not None:
            self.post_handle.remove()
            self.post_handle = None
        self._gate = None
        self._corrupt_intermediate = None
        self._abs_positions = None
        self._next_cache_abs_pos = None

    def _extract_input_ids(self, args, kwargs):
        if kwargs and kwargs.get("input_ids") is not None:
            return kwargs["input_ids"]
        if args:
            return args[0]
        return None

    def _model_pre_hook(self, _module, args, kwargs):
        input_ids = self._extract_input_ids(args, kwargs)
        cache_position = kwargs.get("cache_position") if kwargs else None
        if cache_position is not None:
            self._abs_positions = [int(pos) for pos in cache_position.detach().cpu().flatten().tolist()]
        elif input_ids is not None and input_ids.ndim == 2:
            seq_len = int(input_ids.shape[1])
            if seq_len > 1:
                self._abs_positions = list(range(seq_len))
                self._next_cache_abs_pos = seq_len
            elif self._next_cache_abs_pos is not None:
                self._abs_positions = [int(self._next_cache_abs_pos)]
                self._next_cache_abs_pos += 1
            else:
                self._abs_positions = [0]
                self._next_cache_abs_pos = 1
        else:
            self._abs_positions = None
        return None

    def _model_post_hook(self, _module, args, kwargs, _output):
        self._abs_positions = None
        return None

    def _should_patch_target_idx(self, target_idx: int) -> bool:
        if self.patch_scope == "all-score-positions":
            return True
        token_id = int(self.clean_target_ids[0, target_idx].item())
        return token_id >= BOV

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
        corrupt_intermediate = self._corrupt_intermediate
        self._corrupt_intermediate = None
        self.forward_calls += 1
        self.max_seq_len_seen = max(self.max_seq_len_seen, int(output.shape[1]))
        if corrupt_intermediate is None or output.shape[1] == 0:
            return output

        target_len = int(self.clean_target_ids.shape[1])
        abs_positions = self._abs_positions
        if abs_positions is None or len(abs_positions) != output.shape[1]:
            abs_positions = list(range(int(output.shape[1])))

        pos_pairs = []
        for rel_pos, abs_pos in enumerate(abs_positions):
            target_idx = int(abs_pos) - self.corrupt_prompt_len + 1
            if not (0 <= target_idx < target_len):
                continue
            if not self._should_patch_target_idx(target_idx):
                continue
            source_score_pos = self.clean_prompt_len + target_idx - 1
            if 0 <= source_score_pos < self.clean_intermediate.shape[1]:
                pos_pairs.append((source_score_pos, rel_pos))
        if not pos_pairs:
            return output

        source_idx = torch.tensor([source for source, _rel in pos_pairs], device=output.device, dtype=torch.long)
        rel_idx = torch.tensor([rel for _source, rel in pos_pairs], device=output.device, dtype=torch.long)
        dtype = module.down_proj.weight.dtype
        basis = self.basis.to(device=output.device, dtype=dtype)
        clean = self.clean_intermediate.to(device=output.device, dtype=dtype).index_select(1, source_idx)
        corrupt = corrupt_intermediate.to(device=output.device, dtype=dtype).index_select(1, rel_idx)
        delta = clean - corrupt
        coeff = torch.matmul(delta, basis)
        projected_delta = torch.matmul(coeff, basis.transpose(0, 1))
        contribution_delta = F.linear(projected_delta, module.down_proj.weight).to(output.dtype)
        patched = output.clone()
        patched[:, rel_idx, :] = patched[:, rel_idx, :] + contribution_delta
        self.patched_forward_calls += 1
        self.patched_positions_total += len(pos_pairs)
        return patched


class TargetSideAttentionHeadDecodePatcher:
    def __init__(
        self,
        model,
        layer_idx: int,
        heads: Sequence[int],
        clean_o_proj_input: torch.Tensor,
        clean_prompt_len: int,
        corrupt_prompt_len: int,
        clean_target_ids: torch.Tensor,
        patch_scope: str,
    ):
        self.model = model
        self.layer_idx = int(layer_idx)
        self.heads = [int(head) for head in heads]
        self.clean_o_proj_input = clean_o_proj_input.detach()
        self.clean_prompt_len = int(clean_prompt_len)
        self.corrupt_prompt_len = int(corrupt_prompt_len)
        self.clean_target_ids = clean_target_ids.detach().clone()
        self.patch_scope = patch_scope
        attn = self.model.model.layers[self.layer_idx].self_attn
        self.num_heads = int(attn.num_heads)
        self.head_dim = int(attn.head_dim)
        self.handle = None
        self.pre_handle = None
        self.post_handle = None
        self._abs_positions: List[int] | None = None
        self._next_cache_abs_pos: int | None = None
        self.forward_calls = 0
        self.patched_forward_calls = 0
        self.patched_positions_total = 0
        self.max_seq_len_seen = 0

    def install(self) -> None:
        self.pre_handle = self.model.register_forward_pre_hook(self._model_pre_hook, with_kwargs=True)
        self.post_handle = self.model.register_forward_hook(self._model_post_hook, with_kwargs=True)
        self._next_cache_abs_pos = None
        o_proj = self.model.model.layers[self.layer_idx].self_attn.o_proj
        self.handle = o_proj.register_forward_pre_hook(self._hook)

    def remove(self) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
        if self.pre_handle is not None:
            self.pre_handle.remove()
            self.pre_handle = None
        if self.post_handle is not None:
            self.post_handle.remove()
            self.post_handle = None
        self._abs_positions = None
        self._next_cache_abs_pos = None

    def _extract_input_ids(self, args, kwargs):
        if kwargs and kwargs.get("input_ids") is not None:
            return kwargs["input_ids"]
        if args:
            return args[0]
        return None

    def _model_pre_hook(self, _module, args, kwargs):
        input_ids = self._extract_input_ids(args, kwargs)
        cache_position = kwargs.get("cache_position") if kwargs else None
        if cache_position is not None:
            self._abs_positions = [int(pos) for pos in cache_position.detach().cpu().flatten().tolist()]
        elif input_ids is not None and input_ids.ndim == 2:
            seq_len = int(input_ids.shape[1])
            if seq_len > 1:
                self._abs_positions = list(range(seq_len))
                self._next_cache_abs_pos = seq_len
            elif self._next_cache_abs_pos is not None:
                self._abs_positions = [int(self._next_cache_abs_pos)]
                self._next_cache_abs_pos += 1
            else:
                self._abs_positions = [0]
                self._next_cache_abs_pos = 1
        else:
            self._abs_positions = None
        return None

    def _model_post_hook(self, _module, args, kwargs, _output):
        self._abs_positions = None
        return None

    def _should_patch_target_idx(self, target_idx: int) -> bool:
        if self.patch_scope == "all-score-positions":
            return True
        token_id = int(self.clean_target_ids[0, target_idx].item())
        return token_id >= BOV

    def _hook(self, _module, inputs):
        corrupt = inputs[0]
        self.forward_calls += 1
        self.max_seq_len_seen = max(self.max_seq_len_seen, int(corrupt.shape[1]))
        if corrupt.shape[1] == 0:
            return inputs

        target_len = int(self.clean_target_ids.shape[1])
        abs_positions = self._abs_positions
        if abs_positions is None or len(abs_positions) != corrupt.shape[1]:
            abs_positions = list(range(int(corrupt.shape[1])))

        pos_pairs = []
        for rel_pos, abs_pos in enumerate(abs_positions):
            target_idx = int(abs_pos) - self.corrupt_prompt_len + 1
            if not (0 <= target_idx < target_len):
                continue
            if not self._should_patch_target_idx(target_idx):
                continue
            source_score_pos = self.clean_prompt_len + target_idx - 1
            if 0 <= source_score_pos < self.clean_o_proj_input.shape[1]:
                pos_pairs.append((source_score_pos, rel_pos))
        if not pos_pairs:
            return inputs

        source_idx = torch.tensor([source for source, _rel in pos_pairs], device=corrupt.device, dtype=torch.long)
        rel_idx = torch.tensor([rel for _source, rel in pos_pairs], device=corrupt.device, dtype=torch.long)
        clean = self.clean_o_proj_input.to(device=corrupt.device, dtype=corrupt.dtype)
        patched = corrupt.clone()
        for head_idx in self.heads:
            start = int(head_idx) * self.head_dim
            end = start + self.head_dim
            clean_slice = clean.index_select(1, source_idx)[:, :, start:end]
            patched[:, rel_idx, start:end] = clean_slice
        self.patched_forward_calls += 1
        self.patched_positions_total += len(pos_pairs)
        return (patched,) + tuple(inputs[1:])


class CompositeDecodePatcher:
    def __init__(self, patchers: Sequence[object]):
        self.patchers = list(patchers)

    def install(self) -> None:
        for patcher in self.patchers:
            patcher.install()

    def remove(self) -> None:
        for patcher in reversed(self.patchers):
            patcher.remove()

    @property
    def forward_calls(self) -> int:
        return max((int(getattr(patcher, "forward_calls", 0)) for patcher in self.patchers), default=0)

    @property
    def patched_forward_calls(self) -> int:
        return max((int(getattr(patcher, "patched_forward_calls", 0)) for patcher in self.patchers), default=0)

    @property
    def patched_positions_total(self) -> int:
        return sum(int(getattr(patcher, "patched_positions_total", 0)) for patcher in self.patchers)

    @property
    def max_seq_len_seen(self) -> int:
        return max((int(getattr(patcher, "max_seq_len_seen", 0)) for patcher in self.patchers), default=0)


@torch.no_grad()
def generate_ids_nocfg(
    cfg,
    model,
    tokenizer,
    prompt_ids: torch.Tensor,
    seed: int,
    use_cache: bool,
) -> List[int]:
    set_seed(seed)
    processor = NoCFGImageFormatLogitsProcessor(tokenizer, cfg.target_height, cfg.target_width)
    stopping_criteria = StoppingCriteriaList(
        [StopAfterCompletedImagesCriteria(prompt_ids.shape[1], cfg.special_token_ids["EOI"])]
    )
    extra_criteria = getattr(cfg, "extra_stopping_criteria", None)
    if extra_criteria is not None:
        if isinstance(extra_criteria, (list, tuple, StoppingCriteriaList)):
            for item in extra_criteria:
                stopping_criteria.append(item)
        else:
            stopping_criteria.append(extra_criteria)
    sampling_params = dict(cfg.sampling_params)
    sampling_params["use_cache"] = bool(use_cache)
    config = GenerationConfig(
        **sampling_params,
        pad_token_id=cfg.special_token_ids["PAD"],
        eos_token_id=cfg.special_token_ids["EOS"],
    )
    outputs = model.generate(
        prompt_ids,
        config,
        logits_processor=LogitsProcessorList([processor]),
        stopping_criteria=stopping_criteria,
        use_cache=bool(use_cache),
    )
    generated = outputs[:, prompt_ids.shape[1] :][0].detach().cpu().tolist()
    return trim_to_first_image(generated, cfg.special_token_ids["EOI"])


def collect_clean_source_states(model, input_ids: torch.Tensor, layer_idx: int, component: str) -> torch.Tensor:
    if component == "input-embedding":
        captured: Dict[str, torch.Tensor] = {}

        def hook(_module, _inputs, output):
            captured["states"] = first_tensor(output).detach()

        handle = model.model.embed_tokens.register_forward_hook(hook)
        try:
            model.model(input_ids=input_ids, use_cache=False, return_dict=True)
        finally:
            handle.remove()
        if "states" not in captured:
            raise RuntimeError("missing clean input embedding states")
        return captured["states"]

    if component == "full-residual":
        collector = CleanResidualCollector(model, [layer_idx])
        collector.install()
        try:
            model.model(input_ids=input_ids, use_cache=False, return_dict=True)
        finally:
            collector.remove()
        if layer_idx not in collector.clean_hidden:
            raise RuntimeError(f"missing clean hidden for layer {layer_idx}")
        return collector.clean_hidden[layer_idx]

    if component == "self-attn":
        captured: Dict[str, torch.Tensor] = {}

        def hook(_module, _inputs, output):
            captured["states"] = first_tensor(output).detach()

        handle = model.model.layers[layer_idx].self_attn.register_forward_hook(hook)
        try:
            model.model(input_ids=input_ids, use_cache=False, return_dict=True)
        finally:
            handle.remove()
        if "states" not in captured:
            raise RuntimeError(f"missing clean self-attn output for layer {layer_idx}")
        return captured["states"]

    if component == "mlp":
        captured: Dict[str, torch.Tensor] = {}

        def hook(_module, _inputs, output):
            captured["states"] = first_tensor(output).detach()

        handle = model.model.layers[layer_idx].mlp.register_forward_hook(hook)
        try:
            model.model(input_ids=input_ids, use_cache=False, return_dict=True)
        finally:
            handle.remove()
        if "states" not in captured:
            raise RuntimeError(f"missing clean MLP output for layer {layer_idx}")
        return captured["states"]

    if component == "mlp-lowrank":
        captured: Dict[str, torch.Tensor] = {}
        mlp = model.model.layers[layer_idx].mlp
        gate = {"states": None}

        def gate_hook(_module, _inputs, output):
            gate["states"] = mlp.act_fn(output.detach())

        def up_hook(_module, _inputs, output):
            if gate["states"] is None:
                return
            captured["states"] = (gate["states"] * output.detach()).detach()
            gate["states"] = None

        handles = [
            mlp.gate_proj.register_forward_hook(gate_hook),
            mlp.up_proj.register_forward_hook(up_hook),
        ]
        try:
            model.model(input_ids=input_ids, use_cache=False, return_dict=True)
        finally:
            for handle in handles:
                handle.remove()
        if "states" not in captured:
            raise RuntimeError(f"missing clean MLP intermediate for layer {layer_idx}")
        return captured["states"]

    if component == "attention-heads":
        captured: Dict[str, torch.Tensor] = {}
        o_proj = model.model.layers[layer_idx].self_attn.o_proj

        def hook(_module, inputs):
            captured["states"] = inputs[0].detach()
            return inputs

        handle = o_proj.register_forward_pre_hook(hook)
        try:
            model.model(input_ids=input_ids, use_cache=False, return_dict=True)
        finally:
            handle.remove()
        if "states" not in captured:
            raise RuntimeError(f"missing clean attention o_proj input for layer {layer_idx}")
        return captured["states"]

    if component == "attention-heads-mlp":
        return {
            "attention-heads": collect_clean_source_states(model, input_ids, layer_idx, "attention-heads"),
            "mlp": collect_clean_source_states(model, input_ids, layer_idx, "mlp"),
        }

    if component == "self-attn-mlp":
        return {
            "self-attn": collect_clean_source_states(model, input_ids, layer_idx, "self-attn"),
            "mlp": collect_clean_source_states(model, input_ids, layer_idx, "mlp"),
        }

    raise ValueError(f"unknown component: {component}")


def build_input_embedding_patch_pairs(
    tokenizer,
    clean_prompt: torch.Tensor,
    corrupt_prompt: torch.Tensor,
    clean_target_ids: torch.Tensor,
    patch_scope: str,
    clean_row: dict,
    corrupt_row: dict,
) -> List[tuple[int, int]]:
    clean_prompt_len = int(clean_prompt.shape[1])
    corrupt_prompt_len = int(corrupt_prompt.shape[1])
    if patch_scope == "prompt":
        return [(pos, pos) for pos in range(min(clean_prompt_len, corrupt_prompt_len))]

    if patch_scope == "color-token":
        clean_positions = color_token_positions(tokenizer, clean_prompt, clean_row["color"])
        corrupt_positions = color_token_positions(tokenizer, corrupt_prompt, corrupt_row["color"])
        mapping = map_source_span_to_target_span(clean_positions, corrupt_positions)
        return [(source, target) for target, source in sorted(mapping.items())]

    if patch_scope == "object-token":
        if not clean_row.get("shape") or not corrupt_row.get("shape"):
            raise RuntimeError("object-token patch requires clean/corrupt shape values")
        clean_positions = object_token_positions(tokenizer, clean_prompt, clean_row["shape"])
        corrupt_positions = object_token_positions(tokenizer, corrupt_prompt, corrupt_row["shape"])
        mapping = map_source_span_to_target_span(clean_positions, corrupt_positions)
        return [(source, target) for target, source in sorted(mapping.items())]

    if patch_scope in ("visual-score-positions", "all-score-positions"):
        pairs = []
        target = clean_target_ids[0]
        for target_idx, token_id in enumerate(target.detach().cpu().tolist()):
            if patch_scope == "visual-score-positions" and int(token_id) < BOV:
                continue
            source_score_pos = clean_prompt_len + target_idx - 1
            target_score_pos = corrupt_prompt_len + target_idx - 1
            if target_score_pos >= 0 and source_score_pos >= 0:
                pairs.append((source_score_pos, target_score_pos))
        return pairs

    raise ValueError(f"unknown patch_scope: {patch_scope}")


def load_lowrank_basis(args: argparse.Namespace) -> torch.Tensor | None:
    if args.component != "mlp-lowrank":
        return None
    if not args.lowrank_basis_path:
        raise ValueError("--lowrank-basis-path is required when --component mlp-lowrank")
    if args.lowrank_rank <= 0:
        raise ValueError("--lowrank-rank must be positive when --component mlp-lowrank")
    payload = torch.load(args.lowrank_basis_path, map_location="cpu")
    key = f"{args.lowrank_basis_kind}_basis"
    if key not in payload:
        raise KeyError(f"{args.lowrank_basis_path} has no key {key!r}")
    basis = payload[key]
    if basis.ndim != 2:
        raise ValueError(f"{key} must be a matrix, got shape {tuple(basis.shape)}")
    if args.lowrank_rank > basis.shape[1]:
        raise ValueError(f"requested rank {args.lowrank_rank}, but {key} has only {basis.shape[1]} columns")
    return basis[:, : args.lowrank_rank].contiguous()


def summarize(rows: List[dict]) -> List[dict]:
    variants = sorted(set(row["variant"] for row in rows))
    summary = []
    for variant in variants:
        subset = [row for row in rows if row["variant"] == variant]
        if not subset:
            continue
        summary.append(
            {
                "variant": variant,
                "n": len(subset),
                "decoded": sum(int(row["decoded"]) for row in subset),
                "hue_matches": sum(int(row["hue_match"]) for row in subset),
                "mean_colorful_pixel_fraction": sum(float(row["colorful_pixel_fraction"] or 0.0) for row in subset)
                / len(subset),
            }
        )
    return summary


def write_report(out_dir: Path, args: argparse.Namespace, summary_rows: List[dict]) -> None:
    lines = [
        "# Stage 24 Report: Target-Side Residual Patched Decoded Generation",
        "",
        "This experiment tests whether teacher-forced target-side residual evidence transfers to free decoded images.",
        "",
        f"- Config: `{args.cfg}`",
        f"- Target cache dir: `{args.target_cache_dir}`",
        f"- Layer: `{args.layer}`",
        f"- Component: `{args.component}`",
        f"- Heads: `{args.heads}`",
        f"- Low-rank basis path: `{args.lowrank_basis_path}`",
        f"- Low-rank basis kind: `{args.lowrank_basis_kind}`",
        f"- Low-rank rank: `{args.lowrank_rank}`",
        f"- Patch scope: `{args.patch_scope}`",
        f"- Include wrong-target control: `{int(args.include_wrong_target_control)}`",
        f"- Include random-target control: `{int(args.include_random_target_control)}`",
        "",
        "| variant | n | decoded | hue matches | mean colorful fraction |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {variant} | {n} | {decoded} | {hue_matches} | {mean_colorful_pixel_fraction:.6f} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "Interpretation boundary:",
            "",
            "```text",
            "A positive result here would be decoded-image evidence for target-side residual patching.",
            "A weak or failed result does not negate the teacher-forced likelihood pathway;",
            "it means the decoded-image robust generation-circuit claim remains unsupported.",
            "```",
        ]
    )
    (out_dir / "stage24_target_residual_patched_decode_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def flush_outputs(out_dir: Path, rows: List[dict], args: argparse.Namespace) -> None:
    write_csv(out_dir / "stage24_target_residual_patched_decode.csv", rows)
    summary_rows = summarize(rows)
    write_csv(out_dir / "summary.csv", summary_rows)
    write_report(out_dir, args, summary_rows)


def flatten_rows(pairs) -> List[dict]:
    rows = []
    seen = set()
    for _pair_id, a_row, b_row in pairs:
        for row in (a_row, b_row):
            sample_id = row["sample_id"]
            if sample_id not in seen:
                rows.append(row)
                seen.add(sample_id)
    return sorted(rows, key=lambda item: item["sample_id"])


def choose_random_control_row(control_rows: List[dict], clean_row: dict, corrupt_row: dict) -> dict | None:
    candidates = [
        row
        for row in control_rows
        if row["sample_id"] not in {clean_row["sample_id"], corrupt_row["sample_id"]}
        and row.get("pair_id") != clean_row.get("pair_id")
    ]
    if not candidates:
        candidates = [
            row
            for row in control_rows
            if row["sample_id"] not in {clean_row["sample_id"], corrupt_row["sample_id"]}
        ]
    if not candidates:
        return None
    index = sum(ord(ch) for ch in clean_row["sample_id"]) % len(candidates)
    return candidates[index]


def main() -> None:
    args = parse_args()
    cfg = load_cfg(args.cfg)
    if args.target_height:
        cfg.target_height = args.target_height
    if args.target_width:
        cfg.target_width = args.target_width
    if args.image_area:
        cfg.image_area = args.image_area
    if args.generation_max_new_tokens:
        cfg.max_new_tokens = args.generation_max_new_tokens
        cfg.sampling_params["max_new_tokens"] = args.generation_max_new_tokens

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
    cfg.special_token_ids = {key: tokenizer.encode(value)[0] for key, value in cfg.special_tokens.items()}
    lowrank_basis = load_lowrank_basis(args)
    head_ids = None
    if args.component in ("attention-heads", "attention-heads-mlp"):
        num_heads = int(model.model.layers[int(args.layer)].self_attn.num_heads)
        head_ids = parse_heads(args.heads, num_heads)

    out_dir = Path(args.out_dir)
    image_dir = out_dir / "decoded"
    out_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    pairs = filter_pairs(normalize_prompt_rows(cfg, args.max_pairs), args.pair_id)
    control_rows = flatten_rows(pairs)
    rows = []
    directions_seen = 0
    for _pair_id, a_row, b_row in tqdm(pairs, desc="pairs"):
        for clean_row, corrupt_row in ((a_row, b_row), (b_row, a_row)):
            if args.max_directions and directions_seen >= args.max_directions:
                break
            directions_seen += 1

            clean_prompt, _clean_unc, _clean_full_unc = build_prompt_ids(cfg, tokenizer, clean_row["prompt"], model.device)
            corrupt_prompt, _corrupt_unc, _corrupt_full_unc = build_prompt_ids(cfg, tokenizer, corrupt_row["prompt"], model.device)
            clean_target_ids = load_target_ids(
                Path(args.target_cache_dir),
                clean_row["sample_id"],
                clean_prompt.dtype,
                model.device,
            )
            corrupt_target_ids = None
            corrupt_hidden = None
            random_control_row = None
            random_control_target_ids = None
            random_control_hidden = None
            if args.include_wrong_target_control:
                corrupt_target_ids = load_target_ids(
                    Path(args.target_cache_dir),
                    corrupt_row["sample_id"],
                    corrupt_prompt.dtype,
                    model.device,
                )
            clean_full = torch.cat([clean_prompt, clean_target_ids], dim=1)
            clean_hidden = collect_clean_source_states(model, clean_full, args.layer, args.component)
            torch.cuda.empty_cache()
            if args.include_wrong_target_control and corrupt_target_ids is not None:
                corrupt_full = torch.cat([corrupt_prompt, corrupt_target_ids], dim=1)
                corrupt_hidden = collect_clean_source_states(model, corrupt_full, args.layer, args.component)
                torch.cuda.empty_cache()
            if args.include_random_target_control:
                random_control_row = choose_random_control_row(control_rows, clean_row, corrupt_row)
                if random_control_row is None:
                    raise RuntimeError("random-target control requested but no control row is available")
                random_prompt, _random_unc, _random_full_unc = build_prompt_ids(
                    cfg,
                    tokenizer,
                    random_control_row["prompt"],
                    model.device,
                )
                random_control_target_ids = load_target_ids(
                    Path(args.target_cache_dir),
                    random_control_row["sample_id"],
                    random_prompt.dtype,
                    model.device,
                )
                random_full = torch.cat([random_prompt, random_control_target_ids], dim=1)
                random_control_hidden = collect_clean_source_states(model, random_full, args.layer, args.component)
                torch.cuda.empty_cache()

            base_seed = int(cfg.seed) + args.seed_offset + sum(ord(ch) for ch in clean_row["sample_id"])
            variant_specs = [
                ("clean_cached_target", clean_row, clean_target_ids[0].detach().cpu().tolist(), clean_row["color"], None),
                ("corrupt_decode", corrupt_row, None, clean_row["color"], None),
            ]

            if args.component == "input-embedding":
                patcher = InputEmbeddingPositionDecodePatcher(
                    model,
                    clean_hidden,
                    build_input_embedding_patch_pairs(
                        tokenizer,
                        clean_prompt,
                        corrupt_prompt,
                        clean_target_ids,
                        args.patch_scope,
                        clean_row,
                        corrupt_row,
                    ),
                )
            elif args.component == "mlp-lowrank":
                patcher = TargetSideLowRankMLPDecodePatcher(
                    model,
                    args.layer,
                    lowrank_basis,
                    clean_hidden,
                    clean_prompt.shape[1],
                    corrupt_prompt.shape[1],
                    clean_target_ids,
                    args.patch_scope,
                )
            elif args.component == "attention-heads":
                patcher = TargetSideAttentionHeadDecodePatcher(
                    model,
                    args.layer,
                    head_ids,
                    clean_hidden,
                    clean_prompt.shape[1],
                    corrupt_prompt.shape[1],
                    clean_target_ids,
                    args.patch_scope,
                )
            elif args.component == "attention-heads-mlp":
                patcher = CompositeDecodePatcher(
                    [
                        TargetSideAttentionHeadDecodePatcher(
                            model,
                            args.layer,
                            head_ids,
                            clean_hidden["attention-heads"],
                            clean_prompt.shape[1],
                            corrupt_prompt.shape[1],
                            clean_target_ids,
                            args.patch_scope,
                        ),
                        TargetSideResidualDecodePatcher(
                            model,
                            args.layer,
                            clean_hidden["mlp"],
                            clean_prompt.shape[1],
                            corrupt_prompt.shape[1],
                            clean_target_ids,
                            args.patch_scope,
                            "mlp",
                        ),
                    ]
                )
            elif args.component == "self-attn-mlp":
                patcher = CompositeDecodePatcher(
                    [
                        TargetSideResidualDecodePatcher(
                            model,
                            args.layer,
                            clean_hidden["self-attn"],
                            clean_prompt.shape[1],
                            corrupt_prompt.shape[1],
                            clean_target_ids,
                            args.patch_scope,
                            "self-attn",
                        ),
                        TargetSideResidualDecodePatcher(
                            model,
                            args.layer,
                            clean_hidden["mlp"],
                            clean_prompt.shape[1],
                            corrupt_prompt.shape[1],
                            clean_target_ids,
                            args.patch_scope,
                            "mlp",
                        ),
                    ]
                )
            else:
                patcher = TargetSideResidualDecodePatcher(
                    model,
                    args.layer,
                    clean_hidden,
                    clean_prompt.shape[1],
                    corrupt_prompt.shape[1],
                    clean_target_ids,
                    args.patch_scope,
                    args.component,
                )
            variant_specs.append(("patched_corrupt_decode", corrupt_row, None, clean_row["color"], patcher))
            if args.include_wrong_target_control and corrupt_target_ids is not None and corrupt_hidden is not None:
                if args.component == "input-embedding":
                    wrong_target_patcher = InputEmbeddingPositionDecodePatcher(
                        model,
                        corrupt_hidden,
                        build_input_embedding_patch_pairs(
                            tokenizer,
                            corrupt_prompt,
                            corrupt_prompt,
                            corrupt_target_ids,
                            args.patch_scope,
                            corrupt_row,
                            corrupt_row,
                        ),
                    )
                elif args.component == "mlp-lowrank":
                    wrong_target_patcher = TargetSideLowRankMLPDecodePatcher(
                        model,
                        args.layer,
                        lowrank_basis,
                        corrupt_hidden,
                        corrupt_prompt.shape[1],
                        corrupt_prompt.shape[1],
                        corrupt_target_ids,
                        args.patch_scope,
                    )
                elif args.component == "attention-heads":
                    wrong_target_patcher = TargetSideAttentionHeadDecodePatcher(
                        model,
                        args.layer,
                        head_ids,
                        corrupt_hidden,
                        corrupt_prompt.shape[1],
                        corrupt_prompt.shape[1],
                        corrupt_target_ids,
                        args.patch_scope,
                    )
                elif args.component == "attention-heads-mlp":
                    wrong_target_patcher = CompositeDecodePatcher(
                        [
                            TargetSideAttentionHeadDecodePatcher(
                                model,
                                args.layer,
                                head_ids,
                                corrupt_hidden["attention-heads"],
                                corrupt_prompt.shape[1],
                                corrupt_prompt.shape[1],
                                corrupt_target_ids,
                                args.patch_scope,
                            ),
                            TargetSideResidualDecodePatcher(
                                model,
                                args.layer,
                                corrupt_hidden["mlp"],
                                corrupt_prompt.shape[1],
                                corrupt_prompt.shape[1],
                                corrupt_target_ids,
                                args.patch_scope,
                                "mlp",
                            ),
                        ]
                    )
                elif args.component == "self-attn-mlp":
                    wrong_target_patcher = CompositeDecodePatcher(
                        [
                            TargetSideResidualDecodePatcher(
                                model,
                                args.layer,
                                corrupt_hidden["self-attn"],
                                corrupt_prompt.shape[1],
                                corrupt_prompt.shape[1],
                                corrupt_target_ids,
                                args.patch_scope,
                                "self-attn",
                            ),
                            TargetSideResidualDecodePatcher(
                                model,
                                args.layer,
                                corrupt_hidden["mlp"],
                                corrupt_prompt.shape[1],
                                corrupt_prompt.shape[1],
                                corrupt_target_ids,
                                args.patch_scope,
                                "mlp",
                            ),
                        ]
                    )
                else:
                    wrong_target_patcher = TargetSideResidualDecodePatcher(
                        model,
                        args.layer,
                        corrupt_hidden,
                        corrupt_prompt.shape[1],
                        corrupt_prompt.shape[1],
                        corrupt_target_ids,
                        args.patch_scope,
                        args.component,
                    )
                variant_specs.append(
                    ("wrong_target_patched_corrupt_decode", corrupt_row, None, clean_row["color"], wrong_target_patcher)
                )
            if (
                args.include_random_target_control
                and random_control_row is not None
                and random_control_target_ids is not None
                and random_control_hidden is not None
            ):
                random_prompt, _random_unc, _random_full_unc = build_prompt_ids(
                    cfg,
                    tokenizer,
                    random_control_row["prompt"],
                    model.device,
                )
                if args.component == "input-embedding":
                    random_target_patcher = InputEmbeddingPositionDecodePatcher(
                        model,
                        random_control_hidden,
                        build_input_embedding_patch_pairs(
                            tokenizer,
                            random_prompt,
                            corrupt_prompt,
                            random_control_target_ids,
                            args.patch_scope,
                            random_control_row,
                            corrupt_row,
                        ),
                    )
                elif args.component == "mlp-lowrank":
                    random_target_patcher = TargetSideLowRankMLPDecodePatcher(
                        model,
                        args.layer,
                        lowrank_basis,
                        random_control_hidden,
                        random_prompt.shape[1],
                        corrupt_prompt.shape[1],
                        random_control_target_ids,
                        args.patch_scope,
                    )
                elif args.component == "attention-heads":
                    random_target_patcher = TargetSideAttentionHeadDecodePatcher(
                        model,
                        args.layer,
                        head_ids,
                        random_control_hidden,
                        random_prompt.shape[1],
                        corrupt_prompt.shape[1],
                        random_control_target_ids,
                        args.patch_scope,
                    )
                elif args.component == "attention-heads-mlp":
                    random_target_patcher = CompositeDecodePatcher(
                        [
                            TargetSideAttentionHeadDecodePatcher(
                                model,
                                args.layer,
                                head_ids,
                                random_control_hidden["attention-heads"],
                                random_prompt.shape[1],
                                corrupt_prompt.shape[1],
                                random_control_target_ids,
                                args.patch_scope,
                            ),
                            TargetSideResidualDecodePatcher(
                                model,
                                args.layer,
                                random_control_hidden["mlp"],
                                random_prompt.shape[1],
                                corrupt_prompt.shape[1],
                                random_control_target_ids,
                                args.patch_scope,
                                "mlp",
                            ),
                        ]
                    )
                elif args.component == "self-attn-mlp":
                    random_target_patcher = CompositeDecodePatcher(
                        [
                            TargetSideResidualDecodePatcher(
                                model,
                                args.layer,
                                random_control_hidden["self-attn"],
                                random_prompt.shape[1],
                                corrupt_prompt.shape[1],
                                random_control_target_ids,
                                args.patch_scope,
                                "self-attn",
                            ),
                            TargetSideResidualDecodePatcher(
                                model,
                                args.layer,
                                random_control_hidden["mlp"],
                                random_prompt.shape[1],
                                corrupt_prompt.shape[1],
                                random_control_target_ids,
                                args.patch_scope,
                                "mlp",
                            ),
                        ]
                    )
                else:
                    random_target_patcher = TargetSideResidualDecodePatcher(
                        model,
                        args.layer,
                        random_control_hidden,
                        random_prompt.shape[1],
                        corrupt_prompt.shape[1],
                        random_control_target_ids,
                        args.patch_scope,
                        args.component,
                    )
                variant_specs.append(
                    ("random_target_patched_corrupt_decode", corrupt_row, None, clean_row["color"], random_target_patcher)
                )

            for variant_name, prompt_row, preset_token_ids, expected_color, maybe_patcher in variant_specs:
                if preset_token_ids is None:
                    if maybe_patcher is not None:
                        maybe_patcher.install()
                    try:
                        token_ids = generate_ids_nocfg(
                            cfg,
                            model,
                            tokenizer,
                            corrupt_prompt,
                            base_seed,
                            args.use_cache,
                        )
                    finally:
                        if maybe_patcher is not None:
                            maybe_patcher.remove()
                else:
                    token_ids = [int(token_id) for token_id in preset_token_ids]

                image = decode_image(tokenizer, vq_model, token_ids)
                image_path = ""
                if image is not None:
                    image_path = str(
                        image_dir
                        / f"{clean_row['sample_id']}__corrupt_{corrupt_row['sample_id']}__{variant_name}.png"
                    )
                    image.save(image_path)
                hue_row = classify_decoded(image, expected_color)
                rows.append(
                    {
                        "pair_id": clean_row["pair_id"],
                        "direction": f"{clean_row['color']}->{corrupt_row['color']}",
                        "variant": variant_name,
                        "prompt_sample_id": prompt_row["sample_id"],
                        "clean_sample_id": clean_row["sample_id"],
                        "corrupt_sample_id": corrupt_row["sample_id"],
                        "expected_color": expected_color,
                        "prompt_color": prompt_row["color"],
                        "shape": clean_row["shape"],
                        "component": args.component,
                        "patch_heads": "+".join(str(head) for head in head_ids) if head_ids else "",
                        "patch_head_count": len(head_ids) if head_ids else "",
                        "control_source_sample_id": (
                            random_control_row["sample_id"]
                            if variant_name == "random_target_patched_corrupt_decode" and random_control_row is not None
                            else corrupt_row["sample_id"]
                            if variant_name == "wrong_target_patched_corrupt_decode"
                            else clean_row["sample_id"]
                            if variant_name == "patched_corrupt_decode"
                            else ""
                        ),
                        "layer": args.layer,
                        "patch_scope": args.patch_scope,
                        "generation_mode": f"no-cfg-use-cache-{int(args.use_cache)}",
                        "generated_tokens": len(token_ids),
                        "image_path": image_path,
                        "patch_forward_calls": maybe_patcher.forward_calls if maybe_patcher is not None else "",
                        "patch_patched_forward_calls": maybe_patcher.patched_forward_calls if maybe_patcher is not None else "",
                        "patch_positions_total": maybe_patcher.patched_positions_total if maybe_patcher is not None else "",
                        "patch_max_seq_len_seen": maybe_patcher.max_seq_len_seen if maybe_patcher is not None else "",
                        **hue_row,
                    }
                )
                torch.cuda.empty_cache()
            del clean_hidden
            if corrupt_hidden is not None:
                del corrupt_hidden
            if random_control_hidden is not None:
                del random_control_hidden
            torch.cuda.empty_cache()
            flush_outputs(out_dir, rows, args)
        if args.max_directions and directions_seen >= args.max_directions:
            break

    flush_outputs(out_dir, rows, args)
    print(f"[INFO] Stage 24 target residual patched decode saved to {out_dir}")


if __name__ == "__main__":
    main()
