#!/usr/bin/env python3
# Copyright 2026
# SPDX-License-Identifier: Apache-2.0

"""Generate clean/corrupt/patched images and classify decoded hue."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
from pathlib import Path
import random
import sys
from typing import Dict, List, Sequence, Tuple

import torch
from transformers import GenerationConfig
from transformers.generation import LogitsProcessor, LogitsProcessorList, StoppingCriteria, StoppingCriteriaList
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

from scripts.analyze_generation_target_hue_cache import classify_hue, hue_distance, PALETTE_HUES  # noqa: E402
from scripts.run_generation_residual_patch_recovery import (  # noqa: E402
    CleanResidualCollector,
    build_prompt_ids,
    color_token_positions,
    first_tensor,
    load_cfg,
    map_source_span_to_target_span,
    normalize_prompt_rows,
    parse_layers,
    replace_first_tensor,
    trim_to_first_image,
)
from src.utils.generation_utils import generate, multimodal_decode  # noqa: E402
from src.utils.logits_processor import BOI, BOV, EOI, EOL, IMG  # noqa: E402
from src.utils.model_utils import build_emu3p5  # noqa: E402


class StopAfterCompletedImagesCriteria(StoppingCriteria):
    def __init__(self, prompt_len: int, eoi_token_id: int, completed_images: int = 1):
        self.prompt_len = int(prompt_len)
        self.eoi_token_id = int(eoi_token_id)
        self.completed_images = int(completed_images)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        generated = input_ids[0, self.prompt_len :]
        eoi_positions = (generated == self.eoi_token_id).nonzero().flatten()
        return int(eoi_positions.numel()) >= self.completed_images


class NoCFGImageFormatLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer, target_height: int, target_width: int):
        self.tokenizer = tokenizer
        self.target_height = int(target_height)
        self.target_width = int(target_width)
        self.in_image = False
        self.in_visual = False

    def __call__(self, input_ids, scores):
        last = int(input_ids[0, -1].item())
        prev = int(input_ids[0, -2].item()) if input_ids.shape[1] >= 2 else None
        if last == BOI:
            self.in_image = True
            self.in_visual = False
        if last == EOI:
            self.in_image = False
            self.in_visual = False

        if self.in_image:
            return self._image_scores(input_ids, scores)
        if prev == EOI:
            mask = torch.full_like(scores, -math.inf)
            mask[:, :BOV] = 0
            scores = scores + mask
            return scores
        mask = torch.full_like(scores, -math.inf)
        mask[:, :BOV] = 0
        return scores + mask

    def _image_scores(self, input_ids, scores):
        if self.in_visual:
            img_idx = self._last_index(input_ids[0], IMG)
            vis_idx = input_ids.shape[1] - img_idx
            if vis_idx == self.target_height * (self.target_width + 1):
                mask = torch.full_like(scores, -math.inf)
                mask[:, EOI] = 0
                return scores + mask
            if vis_idx % (self.target_width + 1) == 0:
                mask = torch.full_like(scores, -math.inf)
                mask[:, EOL] = 0
                return scores + mask
            mask = torch.full_like(scores, -math.inf)
            mask[:, BOV:] = 0
            return scores + mask

        if int(input_ids[0, -1].item()) == IMG:
            self.in_visual = True
            mask = torch.full_like(scores, -math.inf)
            mask[:, BOV:] = 0
            return scores + mask

        boi_idx = self._last_index(input_ids[0], BOI)
        hw_idx = input_ids.shape[1] - boi_idx
        hw_tokens = self.tokenizer.encode(f"{self.target_height}*{self.target_width}", add_special_tokens=False)
        allowed = [hw_tokens[hw_idx - 1]] if hw_idx <= len(hw_tokens) else [IMG]
        mask = torch.full_like(scores, -math.inf)
        for token_id in allowed:
            mask[:, token_id] = 0
        return scores + mask

    @staticmethod
    def _last_index(seq, token_id: int) -> int:
        found = (seq == token_id).nonzero().flatten()
        return int(found[-1].item()) if int(found.numel()) else -1


class GenerationPrefillPatcher:
    def __init__(
        self,
        model,
        layer_idx: int,
        clean_hidden: torch.Tensor,
        target_positions: Sequence[int],
        source_position_for_target: Dict[int, int],
        corrupt_prompt_len: int,
        target_prefix_ids: torch.Tensor | None = None,
    ):
        self.model = model
        self.layer_idx = int(layer_idx)
        self.clean_hidden = clean_hidden
        self.target_positions = [int(pos) for pos in target_positions]
        self.source_position_for_target = {int(k): int(v) for k, v in source_position_for_target.items()}
        self.corrupt_prompt_len = int(corrupt_prompt_len)
        self.target_prefix_ids = target_prefix_ids.detach().clone() if target_prefix_ids is not None else None
        self.handle = None
        self.pre_handle = None
        self.post_handle = None
        self.active = self.target_prefix_ids is None
        self.forward_calls = 0
        self.active_forward_calls = 0
        self.patched_layer_calls = 0

    def install(self):
        if self.target_prefix_ids is not None:
            self.pre_handle = self.model.register_forward_pre_hook(self._model_pre_hook, with_kwargs=True)
            self.post_handle = self.model.register_forward_hook(self._model_post_hook, with_kwargs=True)
        layer = self.model.model.layers[self.layer_idx]
        self.handle = layer.register_forward_hook(self._hook)

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
        if self.pre_handle is not None:
            self.pre_handle.remove()
            self.pre_handle = None
        if self.post_handle is not None:
            self.post_handle.remove()
            self.post_handle = None
        self.active = self.target_prefix_ids is None

    def _extract_input_ids(self, args, kwargs):
        if kwargs and kwargs.get("input_ids") is not None:
            return kwargs["input_ids"]
        if args:
            return args[0]
        return None

    def _model_pre_hook(self, _module, args, kwargs):
        self.forward_calls += 1
        input_ids = self._extract_input_ids(args, kwargs)
        self.active = self._matches_target_prefix(input_ids)
        if self.active:
            self.active_forward_calls += 1
        return None

    def _model_post_hook(self, _module, args, kwargs, _output):
        self.active = False
        return None

    def _matches_target_prefix(self, input_ids) -> bool:
        if input_ids is None or self.target_prefix_ids is None:
            return False
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            return False
        prefix = self.target_prefix_ids.to(input_ids.device)
        if input_ids.shape[1] < prefix.shape[1]:
            return False
        return bool(torch.equal(input_ids[0, : prefix.shape[1]], prefix[0]))

    def _hook(self, _module, _inputs, output):
        hidden = first_tensor(output)
        if not self.active:
            return output
        if hidden.shape[1] < self.corrupt_prompt_len:
            return output
        self.patched_layer_calls += 1
        patched = hidden.clone()
        for target_pos in self.target_positions:
            source_pos = self.source_position_for_target.get(target_pos, target_pos)
            if 0 <= target_pos < patched.shape[1] and 0 <= source_pos < self.clean_hidden.shape[1]:
                patched[:, target_pos, :] = self.clean_hidden[:, source_pos, :].to(
                    device=patched.device,
                    dtype=patched.dtype,
                )
        return replace_first_tensor(output, patched)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="configs/ume_main_t2i_counterfactual_pairs_hueonly_seed69.py")
    parser.add_argument("--layer", type=int, default=62)
    parser.add_argument("--max-pairs", type=int, default=1)
    parser.add_argument("--patch-scope", choices=["color-token"], default="color-token")
    parser.add_argument("--target-height", type=int, default=16)
    parser.add_argument("--target-width", type=int, default=16)
    parser.add_argument("--image-area", type=int, default=65536)
    parser.add_argument("--generation-max-new-tokens", type=int, default=700)
    parser.add_argument("--generation-mode", choices=["no-cfg", "cfg"], default="no-cfg")
    parser.add_argument("--classifier-free-guidance", type=float, default=None)
    parser.add_argument("--seed-offset", type=int, default=91000)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def generate_ids_nocfg(cfg, model, tokenizer, prompt_ids, seed: int):
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
    config = GenerationConfig(
        **cfg.sampling_params,
        pad_token_id=cfg.special_token_ids["PAD"],
        eos_token_id=cfg.special_token_ids["EOS"],
    )
    outputs = model.generate(
        prompt_ids,
        config,
        logits_processor=LogitsProcessorList([processor]),
        stopping_criteria=stopping_criteria,
    )
    generated = outputs[:, prompt_ids.shape[1] :][0].detach().cpu().tolist()
    return trim_to_first_image(generated, cfg.special_token_ids["EOI"])


@torch.no_grad()
def generate_ids_cfg(cfg, model, tokenizer, prompt_ids, unconditional_ids, full_unc_ids, seed: int):
    set_seed(seed)
    generated = list(
        generate(
            cfg,
            model,
            tokenizer,
            prompt_ids,
            unconditional_ids,
            full_unc_ids,
            force_same_image_size=True,
        )
    )
    if not generated:
        raise RuntimeError("generation returned no token ids")
    values = generated[0].tolist() if hasattr(generated[0], "tolist") else generated[0]
    return trim_to_first_image(values, cfg.special_token_ids["EOI"])


def generate_ids(cfg, model, tokenizer, prompt_ids, unconditional_ids, full_unc_ids, seed: int, generation_mode: str):
    if generation_mode == "cfg":
        return generate_ids_cfg(cfg, model, tokenizer, prompt_ids, unconditional_ids, full_unc_ids, seed)
    return generate_ids_nocfg(cfg, model, tokenizer, prompt_ids, seed)


def decode_image(tokenizer, vq_model, token_ids: Sequence[int]):
    text = tokenizer.decode([int(x) for x in token_ids], skip_special_tokens=False)
    for kind, payload in multimodal_decode(text, tokenizer, vq_model):
        if kind == "image":
            return payload
    return None


def classify_decoded(image, expected_color: str) -> dict:
    if image is None:
        return {
            "decoded": 0,
            "predicted_color": "",
            "hue_match": 0,
            "mean_hue_degrees": "",
            "expected_hue_degrees": PALETTE_HUES.get(expected_color, ""),
            "hue_distance_degrees": "",
            "colorful_pixel_fraction": "",
        }
    hue = classify_hue(image, sat_threshold=0.25, value_min=0.15, value_max=0.98)
    expected_hue = PALETTE_HUES.get(expected_color)
    mean_hue = hue["mean_hue_degrees"]
    distance = (
        hue_distance(float(mean_hue), expected_hue)
        if expected_hue is not None and isinstance(mean_hue, float) and math.isfinite(mean_hue)
        else float("nan")
    )
    return {
        "decoded": 1,
        "predicted_color": hue["predicted_color"],
        "hue_match": int(hue["predicted_color"] == expected_color),
        "mean_hue_degrees": mean_hue,
        "expected_hue_degrees": expected_hue if expected_hue is not None else "",
        "hue_distance_degrees": distance,
        "colorful_pixel_fraction": hue["colorful_pixel_fraction"],
    }


def color_patch_mapping(tokenizer, clean_prompt, corrupt_prompt, clean_color: str, corrupt_color: str):
    clean_positions = color_token_positions(tokenizer, clean_prompt, clean_color)
    corrupt_positions = color_token_positions(tokenizer, corrupt_prompt, corrupt_color)
    if not clean_positions or not corrupt_positions:
        raise RuntimeError(f"missing color span: clean={clean_positions}, corrupt={corrupt_positions}")
    mapping = map_source_span_to_target_span(clean_positions, corrupt_positions)
    note = ""
    if clean_positions != corrupt_positions:
        note = f"clean_color_positions={clean_positions};corrupt_color_positions={corrupt_positions}"
    if len(clean_positions) != len(corrupt_positions):
        note = (note + ";" if note else "") + (
            f"color_span_len_mismatch_clean={len(clean_positions)}_corrupt={len(corrupt_positions)}"
        )
    return corrupt_positions, mapping, note


def write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    cfg = load_cfg(args.cfg)
    cfg.target_height = args.target_height
    cfg.target_width = args.target_width
    cfg.image_area = args.image_area
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
    model.eval()
    cfg.special_token_ids = {k: tokenizer.encode(v)[0] for k, v in cfg.special_tokens.items()}
    parse_layers(str(args.layer), len(model.model.layers))

    out_dir = Path(args.out_dir)
    image_dir = out_dir / "decoded"
    out_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for _pair_id, a_row, b_row in tqdm(normalize_prompt_rows(cfg, args.max_pairs)):
        for clean_row, corrupt_row in ((a_row, b_row), (b_row, a_row)):
            clean_prompt, clean_unc, clean_full_unc = build_prompt_ids(
                cfg, tokenizer, clean_row["prompt"], model.device
            )
            corrupt_prompt, corrupt_unc, corrupt_full_unc = build_prompt_ids(
                cfg, tokenizer, corrupt_row["prompt"], model.device
            )
            patch_positions, source_map, patch_note = color_patch_mapping(
                tokenizer,
                clean_prompt,
                corrupt_prompt,
                clean_row["color"],
                corrupt_row["color"],
            )
            collector = CleanResidualCollector(model, [args.layer])
            collector.install()
            try:
                model.model(input_ids=clean_prompt, use_cache=False, return_dict=True)
            finally:
                collector.remove()
            clean_hidden = collector.clean_hidden[args.layer]

            base_seed = int(cfg.seed) + args.seed_offset + sum(ord(ch) for ch in clean_row["sample_id"])
            variants = []
            variants.append(("clean", clean_row, clean_prompt, clean_unc, clean_full_unc, clean_row["color"], None))
            variants.append(("corrupt", corrupt_row, corrupt_prompt, corrupt_unc, corrupt_full_unc, clean_row["color"], None))
            patcher = GenerationPrefillPatcher(
                model,
                args.layer,
                clean_hidden,
                patch_positions,
                source_map,
                corrupt_prompt.shape[1],
                target_prefix_ids=corrupt_prompt if args.generation_mode == "cfg" else None,
            )
            variants.append(
                (
                    "patched_corrupt",
                    corrupt_row,
                    corrupt_prompt,
                    corrupt_unc,
                    corrupt_full_unc,
                    clean_row["color"],
                    patcher,
                )
            )

            for variant_name, prompt_row, prompt_ids, unconditional_ids, full_unc_ids, expected_color, maybe_patcher in variants:
                if maybe_patcher is not None:
                    maybe_patcher.install()
                try:
                    token_ids = generate_ids(
                        cfg,
                        model,
                        tokenizer,
                        prompt_ids,
                        unconditional_ids,
                        full_unc_ids,
                        base_seed,
                        args.generation_mode,
                    )
                finally:
                    if maybe_patcher is not None:
                        maybe_patcher.remove()
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
                        "layer": args.layer,
                        "patch_scope": args.patch_scope,
                        "patch_positions": ";".join(str(pos) for pos in patch_positions),
                        "patch_note": patch_note,
                        "patch_forward_calls": maybe_patcher.forward_calls if maybe_patcher is not None else "",
                        "patch_active_forward_calls": maybe_patcher.active_forward_calls if maybe_patcher is not None else "",
                        "patch_layer_calls": maybe_patcher.patched_layer_calls if maybe_patcher is not None else "",
                        "generation_mode": args.generation_mode,
                        "classifier_free_guidance": getattr(cfg, "classifier_free_guidance", ""),
                        "generated_tokens": len(token_ids),
                        "image_path": image_path,
                        **hue_row,
                    }
                )
                torch.cuda.empty_cache()

    write_csv(out_dir / "patched_decode_hue.csv", rows)
    summary = []
    for variant in sorted(set(row["variant"] for row in rows)):
        variant_rows = [row for row in rows if row["variant"] == variant]
        summary.append(
            {
                "variant": variant,
                "n": len(variant_rows),
                "decoded": sum(int(row["decoded"]) for row in variant_rows),
                "hue_matches": sum(int(row["hue_match"]) for row in variant_rows),
                "mean_colorful_pixel_fraction": sum(
                    float(row["colorful_pixel_fraction"] or 0.0) for row in variant_rows
                )
                / len(variant_rows),
            }
        )
    write_csv(out_dir / "summary.csv", summary)
    report = [
        "# Stage 9 Report: Patched Decoded Generation Hue Probe",
        "",
        "This experiment generates clean, corrupt, and patched-corrupt images, then applies the same HSV hue classifier to decoded images.",
        "",
        f"- Config: `{args.cfg}`",
        f"- Layer: {args.layer}",
        f"- Patch scope: `{args.patch_scope}`",
        f"- Target grid: {args.target_height}x{args.target_width}",
        f"- Generation mode: `{args.generation_mode}`",
        f"- Classifier-free guidance: {getattr(cfg, 'classifier_free_guidance', '')}",
        "",
        "| variant | n | decoded | hue matches | mean colorful fraction |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary:
        report.append(
            "| {variant} | {n} | {decoded} | {hue_matches} | {mean_colorful_pixel_fraction:.6f} |".format(**row)
        )
    (out_dir / "patched_decode_hue_report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"[INFO] patched decoded generation hue probe saved to {out_dir}")


if __name__ == "__main__":
    main()
