# Copyright 2025 BAAI. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""Utilities for token-level Unified Multimodal Entropy tracing."""

from __future__ import annotations

import json
import math
from html import escape
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import torch


DEFAULT_SPECIAL_TOKEN_IDS = {
    "BOS": 151849,
    "EOS": 151850,
    "IMG": 151851,
    "BOI": 151852,
    "EOI": 151853,
    "EOL": 151846,
    "EOF": 151847,
    "BOV": 151854,
    "BOG": None,
    "EOG": None,
    "BOC": None,
    "EOC": None,
    "BSS": None,
    "ESS": None,
    "PAD": None,
}

DEFAULT_WEIGHTS = {
    "alpha": 0.35,
    "beta": 0.20,
    "gamma": 0.25,
    "delta": 0.20,
}

MODALITIES = ("text", "visual", "thinking", "structure")


@dataclass
class SegmentState:
    segment: str = "text"
    in_image: bool = False
    in_visual: bool = False


def merge_special_token_ids(
    special_token_ids: Optional[Mapping[str, Optional[int]]] = None,
) -> Dict[str, Optional[int]]:
    merged = dict(DEFAULT_SPECIAL_TOKEN_IDS)
    if special_token_ids:
        merged.update(special_token_ids)
    return merged


def safe_token_text(tokenizer: Any, token_id: int) -> str:
    if tokenizer is None:
        return str(token_id)
    try:
        return tokenizer.decode([int(token_id)], skip_special_tokens=False)
    except TypeError:
        return tokenizer.decode([int(token_id)])
    except Exception:
        return str(token_id)


def normalized_entropy_from_scores(scores: torch.Tensor) -> float:
    """Compute H(p) / log(|A|) over finite logits/scores."""
    finite_scores = scores[torch.isfinite(scores)].float()
    candidate_count = int(finite_scores.numel())
    if candidate_count <= 1:
        return 0.0
    probs = torch.softmax(finite_scores, dim=-1)
    entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum()
    return float((entropy / math.log(candidate_count)).clamp(0.0, 1.0).item())


def modality_masks(
    scores: torch.Tensor,
    segment: str,
    special_token_ids: Mapping[str, Optional[int]],
    visual_token_start: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    vocab_size = scores.shape[-1]
    device = scores.device
    ids = torch.arange(vocab_size, device=device)
    finite = torch.isfinite(scores)

    visual_start = visual_token_start
    if visual_start is None:
        visual_start = special_token_ids.get("BOV")
    if visual_start is None:
        visual = torch.zeros(vocab_size, dtype=torch.bool, device=device)
    else:
        visual = ids >= int(visual_start)

    structure_ids = {
        value
        for key, value in special_token_ids.items()
        if value is not None
        and key
        in {
            "BOS",
            "EOS",
            "PAD",
            "IMG",
            "BOI",
            "EOI",
            "EOL",
            "EOF",
            "BSS",
            "ESS",
            "BOG",
            "EOG",
            "BOC",
            "EOC",
        }
    }
    structure = torch.zeros(vocab_size, dtype=torch.bool, device=device)
    for token_id in structure_ids:
        if 0 <= int(token_id) < vocab_size:
            structure[int(token_id)] = True

    non_visual_non_structure = ~(visual | structure)
    thinking_segment = segment in {"global_cot", "image_cot"}
    thinking = non_visual_non_structure if thinking_segment else torch.zeros_like(structure)
    text = non_visual_non_structure & ~thinking

    return {
        "text": text & finite,
        "visual": visual & finite,
        "thinking": thinking & finite,
        "structure": structure & finite,
    }


def generated_token_type(
    token_id: int,
    segment: str,
    special_token_ids: Mapping[str, Optional[int]],
    visual_token_start: Optional[int] = None,
) -> str:
    structure_values = {
        value
        for key, value in special_token_ids.items()
        if value is not None
        and key
        in {
            "BOS",
            "EOS",
            "PAD",
            "IMG",
            "BOI",
            "EOI",
            "EOL",
            "EOF",
            "BSS",
            "ESS",
            "BOG",
            "EOG",
            "BOC",
            "EOC",
        }
    }
    if token_id in structure_values:
        return "structure"
    visual_start = visual_token_start
    if visual_start is None:
        visual_start = special_token_ids.get("BOV")
    if visual_start is not None and token_id >= int(visual_start):
        return "visual"
    if segment in {"global_cot", "image_cot"}:
        return "thinking"
    return "text"


def update_segment_state(
    state: SegmentState,
    token_id: int,
    special_token_ids: Mapping[str, Optional[int]],
) -> None:
    if token_id == special_token_ids.get("BOG"):
        state.segment = "global_cot"
        return
    if token_id == special_token_ids.get("EOG"):
        state.segment = "text"
        return
    if token_id == special_token_ids.get("BOC"):
        state.segment = "image_cot"
        return
    if token_id == special_token_ids.get("EOC"):
        state.segment = "text"
        return
    if token_id == special_token_ids.get("BOI"):
        state.segment = "image_header"
        state.in_image = True
        state.in_visual = False
        return
    if token_id == special_token_ids.get("IMG"):
        state.segment = "visual"
        state.in_visual = True
        return
    if token_id == special_token_ids.get("EOI"):
        state.segment = "text"
        state.in_image = False
        state.in_visual = False


def modality_entropy_and_masses(
    scores: torch.Tensor,
    masks: Mapping[str, torch.Tensor],
) -> tuple[float, Dict[str, float]]:
    probs = torch.softmax(scores.float(), dim=-1)
    masses: Dict[str, float] = {}
    mass_values = []
    for modality in MODALITIES:
        mass = probs[masks[modality]].sum() if masks[modality].any() else probs.new_tensor(0.0)
        masses[modality] = float(mass.item())
        if mass.item() > 1e-12:
            mass_values.append(mass)

    if len(mass_values) <= 1:
        return 0.0, masses

    mass_tensor = torch.stack(mass_values)
    entropy = -(mass_tensor * torch.log(mass_tensor.clamp_min(1e-12))).sum()
    return float((entropy / math.log(len(mass_values))).clamp(0.0, 1.0).item()), masses


def intra_modality_entropy(scores: torch.Tensor, mask: torch.Tensor) -> float:
    if int(mask.sum().item()) <= 1:
        return 0.0
    return normalized_entropy_from_scores(torch.where(mask, scores, scores.new_full(scores.shape, -math.inf)))


def normalize_cfg_js(value: Optional[float]) -> tuple[float, bool]:
    if value is None:
        return 0.0, False
    return float(max(0.0, min(1.0, value))), True


def build_trace_records(
    scores: Sequence[torch.Tensor],
    generated_token_ids: Sequence[int] | torch.Tensor,
    *,
    sample_id: str,
    tokenizer: Any = None,
    special_token_ids: Optional[Mapping[str, Optional[int]]] = None,
    cfg_trace: Optional[Sequence[Mapping[str, Any]]] = None,
    weights: Optional[Mapping[str, float]] = None,
    visual_token_start: Optional[int] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> List[Dict[str, Any]]:
    ids = merge_special_token_ids(special_token_ids)
    ume_weights = dict(DEFAULT_WEIGHTS)
    if weights:
        ume_weights.update(weights)

    if isinstance(generated_token_ids, torch.Tensor):
        generated = [int(x) for x in generated_token_ids.detach().cpu().flatten().tolist()]
    else:
        generated = [int(x) for x in generated_token_ids]

    if len(scores) != len(generated):
        raise ValueError(f"scores/token length mismatch: {len(scores)} scores vs {len(generated)} tokens")

    records: List[Dict[str, Any]] = []
    state = SegmentState()
    for step, (step_scores, token_id) in enumerate(zip(scores, generated)):
        score_vec = step_scores.detach()
        if score_vec.dim() == 2:
            score_vec = score_vec[0]
        score_vec = score_vec.float().cpu()

        segment = state.segment
        token_type = generated_token_type(token_id, segment, ids, visual_token_start)
        masks = modality_masks(score_vec, segment, ids, visual_token_start)
        u_tok = normalized_entropy_from_scores(score_vec)
        u_mod, masses = modality_entropy_and_masses(score_vec, masks)
        u_intra = intra_modality_entropy(score_vec, masks[token_type])
        sample_candidate_count = int(torch.isfinite(score_vec).sum().item())

        trace_entry = cfg_trace[step] if cfg_trace is not None and step < len(cfg_trace) else {}
        cfg_value = trace_entry.get("u_cfg", trace_entry.get("cfg_js")) if trace_entry else None
        u_cfg, cfg_available = normalize_cfg_js(cfg_value)

        full_candidate_count = trace_entry.get("visual_full_candidate_count") if trace_entry else None
        visual_full_value = trace_entry.get("u_visual_full") if trace_entry else None
        if token_type == "visual" and visual_full_value is not None:
            u_tok_full = float(visual_full_value)
            u_intra_full = float(visual_full_value)
        else:
            u_tok_full = u_tok
            u_intra_full = u_intra
            full_candidate_count = sample_candidate_count if full_candidate_count is None else full_candidate_count

        ume = (
            ume_weights["alpha"] * u_tok
            + ume_weights["beta"] * u_mod
            + ume_weights["gamma"] * u_intra
            + ume_weights["delta"] * u_cfg
        )
        ume_full = (
            ume_weights["alpha"] * u_tok_full
            + ume_weights["beta"] * u_mod
            + ume_weights["gamma"] * u_intra_full
            + ume_weights["delta"] * u_cfg
        )

        record = {
            "sample_id": sample_id,
            "step": step,
            "token_id": token_id,
            "token_text": safe_token_text(tokenizer, token_id),
            "token_type": token_type,
            "segment": segment,
            "is_boundary": token_type == "structure",
            "u_tok": u_tok,
            "u_mod": u_mod,
            "u_intra": u_intra,
            "u_cfg": u_cfg,
            "u_cfg_available": cfg_available,
            "u_sample": u_tok,
            "u_tok_sample": u_tok,
            "u_intra_sample": u_intra,
            "u_visual_full": u_tok_full if token_type == "visual" else None,
            "u_tok_full": u_tok_full,
            "u_intra_full": u_intra_full,
            "ume": float(max(0.0, min(1.0, ume))),
            "ume_sample": float(max(0.0, min(1.0, ume))),
            "ume_full": float(max(0.0, min(1.0, ume_full))),
            "modality_mass": masses,
            "candidate_count": sample_candidate_count,
            "candidate_count_sample": sample_candidate_count,
            "candidate_count_full": full_candidate_count,
        }
        if metadata:
            for key, value in metadata.items():
                if key not in record:
                    record[key] = value
        records.append(record)
        update_segment_state(state, token_id, ids)

    return records


def write_jsonl(records: Iterable[Mapping[str, Any]], path: str | Path) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def summarize_records(records: Sequence[Mapping[str, Any]], group_key: str) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[float]] = {}
    for record in records:
        grouped.setdefault(str(record.get(group_key, "unknown")), []).append(float(record["ume"]))

    rows = []
    for key, values in sorted(grouped.items()):
        tensor = torch.tensor(values, dtype=torch.float32)
        rows.append(
            {
                group_key: key,
                "count": int(tensor.numel()),
                "mean_ume": float(tensor.mean().item()),
                "std_ume": float(tensor.std(unbiased=False).item()) if tensor.numel() > 1 else 0.0,
                "min_ume": float(tensor.min().item()),
                "p50_ume": float(tensor.quantile(0.5).item()),
                "p90_ume": float(tensor.quantile(0.9).item()),
                "max_ume": float(tensor.max().item()),
            }
        )
    return rows


def component_correlation_rows(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    keys = ["u_tok", "u_mod", "u_intra", "u_cfg", "ume"]
    if not records:
        return []

    matrix = torch.tensor(
        [[float(record.get(key, 0.0)) for key in keys] for record in records],
        dtype=torch.float32,
    )
    centered = matrix - matrix.mean(dim=0, keepdim=True)
    denom = torch.sqrt((centered.square().sum(dim=0, keepdim=True).T @ centered.square().sum(dim=0, keepdim=True)).clamp_min(1e-12))
    corr = (centered.T @ centered) / denom
    corr = torch.nan_to_num(corr, nan=0.0).clamp(-1.0, 1.0)

    rows: List[Dict[str, Any]] = []
    for i, key in enumerate(keys):
        row = {"component": key}
        for j, other in enumerate(keys):
            row[other] = float(corr[i, j].item())
        rows.append(row)
    return rows


def _as_error_label(record: Mapping[str, Any]) -> Optional[int]:
    for key in ("is_error", "hallucination", "has_error"):
        if key in record:
            return int(bool(record[key]))
    return None


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    dx = [x - mean_x for x in xs]
    dy = [y - mean_y for y in ys]
    denom_x = math.sqrt(sum(x * x for x in dx))
    denom_y = math.sqrt(sum(y * y for y in dy))
    if denom_x <= 1e-12 or denom_y <= 1e-12:
        return None
    return sum(x * y for x, y in zip(dx, dy)) / (denom_x * denom_y)


def _binary_auc(scores: Sequence[float], labels: Sequence[int]) -> Optional[float]:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    ranked = sorted(enumerate(scores), key=lambda item: item[1])
    rank_sum_pos = 0.0
    rank = 1
    idx = 0
    while idx < len(ranked):
        j = idx + 1
        while j < len(ranked) and ranked[j][1] == ranked[idx][1]:
            j += 1
        avg_rank = (rank + rank + (j - idx) - 1) / 2
        for k in range(idx, j):
            original_idx = ranked[k][0]
            if labels[original_idx] == 1:
                rank_sum_pos += avg_rank
        rank += j - idx
        idx = j
    return (rank_sum_pos - positives * (positives + 1) / 2) / (positives * negatives)


def hallucination_diagnostic_rows(
    records: Sequence[Mapping[str, Any]],
    *,
    group_key: str = "task",
    low_ume_threshold: float = 0.20,
    high_ume_threshold: float = 0.70,
    high_cfg_threshold: float = 0.50,
) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Mapping[str, Any]]] = {}
    for record in records:
        if _as_error_label(record) is not None:
            grouped.setdefault(str(record.get(group_key, "unknown")), []).append(record)

    rows: List[Dict[str, Any]] = []
    for group, group_records in sorted(grouped.items()):
        labels = [_as_error_label(record) for record in group_records]
        label_values = [int(label) for label in labels if label is not None]
        umes = [float(record["ume"]) for record in group_records]
        cfgs = [float(record.get("u_cfg", 0.0)) for record in group_records]
        error_count = sum(label_values)
        correct_count = len(label_values) - error_count
        low_indices = [idx for idx, ume in enumerate(umes) if ume < low_ume_threshold]
        high_indices = [idx for idx, ume in enumerate(umes) if ume > high_ume_threshold]
        low_cfg_indices = [
            idx
            for idx, (ume, cfg) in enumerate(zip(umes, cfgs))
            if ume < low_ume_threshold and cfg > high_cfg_threshold
        ]

        def rate(indices: Sequence[int]) -> Optional[float]:
            if not indices:
                return None
            return sum(label_values[idx] for idx in indices) / len(indices)

        error_umes = [ume for ume, label in zip(umes, label_values) if label == 1]
        correct_umes = [ume for ume, label in zip(umes, label_values) if label == 0]
        rows.append(
            {
                group_key: group,
                "labeled_tokens": len(label_values),
                "error_tokens": error_count,
                "error_rate": error_count / len(label_values) if label_values else 0.0,
                "mean_ume_error": sum(error_umes) / len(error_umes) if error_umes else "",
                "mean_ume_correct": sum(correct_umes) / len(correct_umes) if correct_umes else "",
                "corr_ume_error": _pearson(umes, label_values) if error_count and correct_count else "",
                "auc_ume_error": _binary_auc(umes, label_values) if error_count and correct_count else "",
                "false_conf_tokens": sum(label_values[idx] for idx in low_indices),
                "false_conf_rate": rate(low_indices) if low_indices else "",
                "unresolved_high_ume_tokens": sum(label_values[idx] for idx in high_indices),
                "unresolved_high_ume_rate": rate(high_indices) if high_indices else "",
                "low_ume_high_cfg_tokens": len(low_cfg_indices),
                "low_ume_high_cfg_error_rate": rate(low_cfg_indices) if low_cfg_indices else "",
            }
        )
    return rows


def calibration_bin_rows(
    records: Sequence[Mapping[str, Any]],
    *,
    bins: int = 10,
) -> List[Dict[str, Any]]:
    labeled = [(float(record["ume"]), _as_error_label(record)) for record in records]
    labeled = [(ume, int(label)) for ume, label in labeled if label is not None]
    if not labeled:
        return []
    rows: List[Dict[str, Any]] = []
    total = len(labeled)
    weighted_abs_gap = 0.0
    for idx in range(bins):
        lower = idx / bins
        upper = (idx + 1) / bins
        in_bin = [
            (ume, label)
            for ume, label in labeled
            if (lower <= ume < upper) or (idx == bins - 1 and lower <= ume <= upper)
        ]
        if not in_bin:
            continue
        mean_ume = sum(ume for ume, _ in in_bin) / len(in_bin)
        error_rate = sum(label for _, label in in_bin) / len(in_bin)
        abs_gap = abs(mean_ume - error_rate)
        weighted_abs_gap += len(in_bin) / total * abs_gap
        rows.append(
            {
                "bin": idx,
                "ume_lower": lower,
                "ume_upper": upper,
                "count": len(in_bin),
                "mean_ume": mean_ume,
                "error_rate": error_rate,
                "abs_gap": abs_gap,
                "running_ece": weighted_abs_gap,
            }
        )
    return rows


def write_csv(rows: Sequence[Mapping[str, Any]], path: str | Path) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        out_path.write_text("", encoding="utf-8")
        return
    headers = list(rows[0].keys())
    lines = [",".join(headers)]
    for row in rows:
        lines.append(",".join(str(row.get(header, "")) for header in headers))
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def thinking_transition_rows(
    records: Sequence[Mapping[str, Any]],
    after_segments: Sequence[str] = ("visual",),
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    grouped: Dict[str, List[Mapping[str, Any]]] = {}
    for record in records:
        grouped.setdefault(str(record.get("sample_id", "unknown")), []).append(record)

    for sample_id, sample_records in sorted(grouped.items()):
        ordered = sorted(sample_records, key=lambda row: int(row.get("step", 0)))
        thinking_values = [
            float(row["ume"])
            for row in ordered
            if str(row.get("segment")) in {"global_cot", "image_cot"}
            and str(row.get("token_type")) != "structure"
        ]
        after_values = [
            float(row["ume"])
            for row in ordered
            if str(row.get("segment")) in after_segments
            and str(row.get("token_type")) != "structure"
        ]
        if not thinking_values or not after_values:
            continue
        thinking_mean = sum(thinking_values) / len(thinking_values)
        after_mean = sum(after_values) / len(after_values)
        rows.append(
            {
                "sample_id": sample_id,
                "task": ordered[0].get("task", "unknown"),
                "thinking_tokens": len(thinking_values),
                "after_tokens": len(after_values),
                "mean_thinking_ume": thinking_mean,
                "mean_after_ume": after_mean,
                "delta_thinking_minus_after": thinking_mean - after_mean,
            }
        )
    return rows


def _svg_header(width: int, height: int) -> List[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text{font-family:Arial,sans-serif;font-size:12px;fill:#17202a}.label{font-size:11px;fill:#566573}.title{font-size:15px;font-weight:700}.axis{stroke:#85929e;stroke-width:1}.grid{stroke:#d5dbdb;stroke-width:1}.bar{fill:#2e86ab}.box{fill:#f6c85f;stroke:#7d6608;stroke-width:1.2}.line{fill:none;stroke:#2e86ab;stroke-width:2}.point{fill:#d1495b}.heat-pos{fill:#2e86ab}.heat-neg{fill:#d1495b}</style>',
    ]


def _write_svg(path: Path, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = q * (len(ordered) - 1)
    low = int(math.floor(pos))
    high = int(math.ceil(pos))
    if low == high:
        return ordered[low]
    weight = pos - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def write_histogram_svg(records: Sequence[Mapping[str, Any]], path: str | Path) -> Optional[str]:
    values = [float(row["ume"]) for row in records]
    if not values:
        return None
    width, height = 760, 360
    margin_l, margin_b, margin_t, margin_r = 56, 44, 42, 22
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b
    bins = 20
    counts = [0] * bins
    for value in values:
        idx = min(bins - 1, max(0, int(value * bins)))
        counts[idx] += 1
    max_count = max(counts) or 1
    lines = _svg_header(width, height)
    lines.append('<text class="title" x="56" y="25">UME distribution</text>')
    lines.append(f'<line class="axis" x1="{margin_l}" y1="{height-margin_b}" x2="{width-margin_r}" y2="{height-margin_b}"/>')
    lines.append(f'<line class="axis" x1="{margin_l}" y1="{margin_t}" x2="{margin_l}" y2="{height-margin_b}"/>')
    for i, count in enumerate(counts):
        x = margin_l + i * plot_w / bins + 2
        bar_w = plot_w / bins - 4
        bar_h = plot_h * count / max_count
        y = height - margin_b - bar_h
        lines.append(f'<rect class="bar" x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}"/>')
    for tick in range(6):
        value = tick / 5
        x = margin_l + value * plot_w
        lines.append(f'<text class="label" x="{x-8:.1f}" y="{height-18}">{value:.1f}</text>')
    lines.append('<text class="label" x="12" y="32">count</text>')
    lines.append("</svg>")
    _write_svg(Path(path), lines)
    return str(path)


def write_boxplot_svg(
    records: Sequence[Mapping[str, Any]],
    group_key: str,
    path: str | Path,
    title: Optional[str] = None,
) -> Optional[str]:
    grouped: Dict[str, List[float]] = {}
    for row in records:
        grouped.setdefault(str(row.get(group_key, "unknown")), []).append(float(row["ume"]))
    grouped = {key: values for key, values in grouped.items() if values}
    if not grouped:
        return None

    width = max(760, 110 * len(grouped) + 120)
    height = 380
    margin_l, margin_b, margin_t, margin_r = 56, 82, 44, 24
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b
    lines = _svg_header(width, height)
    lines.append(f'<text class="title" x="56" y="26">{escape(title or f"UME by {group_key}")}</text>')
    for tick in range(6):
        value = tick / 5
        y = margin_t + (1 - value) * plot_h
        lines.append(f'<line class="grid" x1="{margin_l}" y1="{y:.1f}" x2="{width-margin_r}" y2="{y:.1f}"/>')
        lines.append(f'<text class="label" x="22" y="{y+4:.1f}">{value:.1f}</text>')
    lines.append(f'<line class="axis" x1="{margin_l}" y1="{height-margin_b}" x2="{width-margin_r}" y2="{height-margin_b}"/>')
    lines.append(f'<line class="axis" x1="{margin_l}" y1="{margin_t}" x2="{margin_l}" y2="{height-margin_b}"/>')

    keys = sorted(grouped)
    step = plot_w / len(keys)
    for idx, key in enumerate(keys):
        values = grouped[key]
        q1 = _quantile(values, 0.25)
        q2 = _quantile(values, 0.50)
        q3 = _quantile(values, 0.75)
        low = min(values)
        high = max(values)
        x = margin_l + step * idx + step / 2
        box_w = min(54, step * 0.55)

        def y_of(value: float) -> float:
            return margin_t + (1 - max(0.0, min(1.0, value))) * plot_h

        lines.append(f'<line class="axis" x1="{x:.1f}" y1="{y_of(low):.1f}" x2="{x:.1f}" y2="{y_of(high):.1f}"/>')
        lines.append(f'<line class="axis" x1="{x-box_w/3:.1f}" y1="{y_of(low):.1f}" x2="{x+box_w/3:.1f}" y2="{y_of(low):.1f}"/>')
        lines.append(f'<line class="axis" x1="{x-box_w/3:.1f}" y1="{y_of(high):.1f}" x2="{x+box_w/3:.1f}" y2="{y_of(high):.1f}"/>')
        lines.append(f'<rect class="box" x="{x-box_w/2:.1f}" y="{y_of(q3):.1f}" width="{box_w:.1f}" height="{max(1.0, y_of(q1)-y_of(q3)):.1f}"/>')
        lines.append(f'<line class="axis" x1="{x-box_w/2:.1f}" y1="{y_of(q2):.1f}" x2="{x+box_w/2:.1f}" y2="{y_of(q2):.1f}"/>')
        label = escape(key[:18])
        lines.append(f'<text class="label" x="{x-32:.1f}" y="{height-54}" transform="rotate(35 {x-32:.1f},{height-54})">{label}</text>')
        lines.append(f'<text class="label" x="{x-8:.1f}" y="{height-16}">n={len(values)}</text>')
    lines.append("</svg>")
    _write_svg(Path(path), lines)
    return str(path)


def write_correlation_heatmap_svg(records: Sequence[Mapping[str, Any]], path: str | Path) -> Optional[str]:
    rows = component_correlation_rows(records)
    if not rows:
        return None
    keys = ["u_tok", "u_mod", "u_intra", "u_cfg", "ume"]
    cell = 72
    width = 520
    height = 480
    start_x = 126
    start_y = 76
    lines = _svg_header(width, height)
    lines.append('<text class="title" x="56" y="28">UME component correlations</text>')
    for i, key in enumerate(keys):
        lines.append(f'<text class="label" x="{start_x+i*cell+14}" y="58">{key}</text>')
        lines.append(f'<text class="label" x="45" y="{start_y+i*cell+42}">{key}</text>')
    for i, row in enumerate(rows):
        for j, key in enumerate(keys):
            value = float(row[key])
            color = "#2e86ab" if value >= 0 else "#d1495b"
            opacity = 0.12 + 0.78 * abs(value)
            x = start_x + j * cell
            y = start_y + i * cell
            lines.append(f'<rect x="{x}" y="{y}" width="{cell-3}" height="{cell-3}" fill="{color}" opacity="{opacity:.3f}"/>')
            lines.append(f'<text x="{x+16}" y="{y+42}">{value:.2f}</text>')
    lines.append("</svg>")
    _write_svg(Path(path), lines)
    return str(path)


def write_trace_svg(records: Sequence[Mapping[str, Any]], path: str | Path) -> Optional[str]:
    ordered = sorted(records, key=lambda row: int(row.get("step", 0)))
    if not ordered:
        return None
    width, height = 820, 320
    margin_l, margin_b, margin_t, margin_r = 56, 52, 46, 24
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b
    max_step = max(1, len(ordered) - 1)
    lines = _svg_header(width, height)
    sample_id = escape(str(ordered[0].get("sample_id", "sample")))
    lines.append(f'<text class="title" x="56" y="28">Trace: {sample_id}</text>')
    for tick in range(6):
        value = tick / 5
        y = margin_t + (1 - value) * plot_h
        lines.append(f'<line class="grid" x1="{margin_l}" y1="{y:.1f}" x2="{width-margin_r}" y2="{y:.1f}"/>')
        lines.append(f'<text class="label" x="22" y="{y+4:.1f}">{value:.1f}</text>')
    points = []
    colors = {"text": "#2e86ab", "visual": "#6a994e", "thinking": "#7d5fff", "structure": "#d1495b"}
    for idx, row in enumerate(ordered):
        x = margin_l + idx * plot_w / max_step
        y = margin_t + (1 - float(row["ume"])) * plot_h
        points.append(f"{x:.1f},{y:.1f}")
    lines.append(f'<polyline class="line" points="{" ".join(points)}"/>')
    for idx, row in enumerate(ordered):
        x = margin_l + idx * plot_w / max_step
        y = margin_t + (1 - float(row["ume"])) * plot_h
        color = colors.get(str(row.get("token_type")), "#34495e")
        lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.2" fill="{color}"/>')
    legend_x = width - 310
    for idx, (name, color) in enumerate(colors.items()):
        x = legend_x + idx * 78
        lines.append(f'<circle cx="{x}" cy="27" r="4" fill="{color}"/>')
        lines.append(f'<text class="label" x="{x+8}" y="31">{name}</text>')
    lines.append("</svg>")
    _write_svg(Path(path), lines)
    return str(path)


def write_visual_artifacts(records: Sequence[Mapping[str, Any]], out_dir: str | Path, max_traces: int = 10) -> List[str]:
    out_path = Path(out_dir)
    figures_dir = out_path / "figures"
    written: List[str] = []
    figure_specs = [
        write_histogram_svg(records, figures_dir / "ume_histogram.svg"),
        write_boxplot_svg(records, "token_type", figures_dir / "ume_by_token_type_boxplot.svg", "UME by token type"),
        write_boxplot_svg(records, "segment", figures_dir / "ume_by_segment_boxplot.svg", "UME by segment"),
        write_boxplot_svg(records, "task", figures_dir / "ume_by_task_boxplot.svg", "UME by task"),
        write_correlation_heatmap_svg(records, figures_dir / "ume_component_correlation_heatmap.svg"),
    ]
    written.extend(str(path) for path in figure_specs if path)

    grouped: Dict[str, List[Mapping[str, Any]]] = {}
    for row in records:
        grouped.setdefault(str(row.get("sample_id", "unknown")), []).append(row)
    ranked_samples = sorted(
        grouped.items(),
        key=lambda item: max(float(row["ume"]) for row in item[1]),
        reverse=True,
    )
    for sample_id, sample_records in ranked_samples[:max_traces]:
        safe_name = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in sample_id)
        path = write_trace_svg(sample_records, figures_dir / f"trace_{safe_name}.svg")
        if path:
            written.append(str(path))
    return written


def write_html_report(
    records: Sequence[Mapping[str, Any]],
    out_dir: str | Path,
    title: str,
    figure_paths: Optional[Sequence[str]] = None,
) -> Path:
    out_path = Path(out_dir)
    figure_paths = list(figure_paths or [])
    token_rows = summarize_records(records, "token_type")
    segment_rows = summarize_records(records, "segment")
    task_rows = summarize_records(records, "task")
    transition = thinking_transition_rows(records)
    diagnostic_rows = hallucination_diagnostic_rows(records)
    calibration_rows = calibration_bin_rows(records)

    def table_html(rows: Sequence[Mapping[str, Any]]) -> str:
        if not rows:
            return "<p>No rows.</p>"
        headers = list(rows[0].keys())
        head = "".join(f"<th>{escape(str(header))}</th>" for header in headers)
        body_lines = []
        for row in rows:
            cells = []
            for header in headers:
                value = row.get(header, "")
                if isinstance(value, float):
                    value = f"{value:.4f}"
                cells.append(f"<td>{escape(str(value))}</td>")
            body_lines.append("<tr>" + "".join(cells) + "</tr>")
        return "<table><thead><tr>" + head + "</tr></thead><tbody>" + "".join(body_lines) + "</tbody></table>"

    figure_html = []
    for figure in figure_paths:
        figure_path = Path(figure)
        rel = figure_path.relative_to(out_path) if figure_path.is_relative_to(out_path) else figure_path
        label = figure_path.stem
        figure_html.append(f'<h3>{escape(label)}</h3><img src="{escape(str(rel))}" alt="{escape(label)}">')

    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>{escape(title)}</title>
<style>
body{{font-family:Arial,sans-serif;margin:32px;line-height:1.45;color:#17202a}}
table{{border-collapse:collapse;margin:12px 0 28px 0;font-size:13px}}
th,td{{border:1px solid #d5dbdb;padding:6px 9px;text-align:right}}
th:first-child,td:first-child{{text-align:left}}
img{{max-width:100%;border:1px solid #d5dbdb;margin:4px 0 24px 0}}
h1,h2,h3{{margin-top:24px}}
</style>
</head>
<body>
<h1>{escape(title)}</h1>
<p>Total traced tokens: {len(records)}</p>
<h2>UME by token_type</h2>
{table_html(token_rows)}
<h2>UME by segment</h2>
{table_html(segment_rows)}
<h2>UME by task</h2>
{table_html(task_rows)}
<h2>Thinking transition</h2>
{table_html(transition)}
<h2>Hallucination diagnostics</h2>
{table_html(diagnostic_rows)}
<h2>Calibration bins</h2>
{table_html(calibration_rows)}
<h2>Figures</h2>
{''.join(figure_html)}
</body>
</html>
"""
    html_path = out_path / "entropy_distribution_report.html"
    html_path.write_text(html, encoding="utf-8")
    return html_path


def markdown_table(rows: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        return "_No rows._"
    headers = list(rows[0].keys())
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        values = []
        for header in headers:
            value = row.get(header, "")
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def render_summary_report(records: Sequence[Mapping[str, Any]], title: str) -> str:
    token_rows = summarize_records(records, "token_type")
    segment_rows = summarize_records(records, "segment")
    task_rows = summarize_records(records, "task")
    corr_rows = component_correlation_rows(records)
    transition_rows = thinking_transition_rows(records)
    diagnostic_rows = hallucination_diagnostic_rows(records)
    calibration_rows = calibration_bin_rows(records)
    high = sorted(records, key=lambda row: float(row["ume"]), reverse=True)[:20]
    low = sorted(records, key=lambda row: float(row["ume"]))[:20]

    def compact_token_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
        return [
            {
                "step": row["step"],
                "token_type": row["token_type"],
                "segment": row["segment"],
                "ume": row["ume"],
                "u_tok": row["u_tok"],
                "u_mod": row["u_mod"],
                "u_intra": row["u_intra"],
                "u_cfg": row["u_cfg"],
                "token_text": str(row["token_text"]).replace("\n", "\\n")[:60],
            }
            for row in rows
        ]

    return "\n\n".join(
        [
            f"# {title}",
            f"Total traced tokens: {len(records)}",
            "## UME by token_type",
            markdown_table(token_rows),
            "## UME by segment",
            markdown_table(segment_rows),
            "## UME by task",
            markdown_table(task_rows),
            "## Thinking transition",
            markdown_table(transition_rows),
            "## Hallucination diagnostics",
            markdown_table(diagnostic_rows),
            "## Calibration bins",
            markdown_table(calibration_rows),
            "## Component correlations",
            markdown_table(corr_rows),
            "## Highest UME tokens",
            markdown_table(compact_token_rows(high)),
            "## Lowest UME tokens",
            markdown_table(compact_token_rows(low)),
        ]
    )
