# Copyright 2025 BAAI. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""Entropy helpers for synthetic-concept text/image experiments.

This module keeps two notions separate:

* token entropy: entropy of the next-token distribution during generation.
* semantic entropy: entropy over a fixed attribute/value space after readout.

The first is useful for within-channel analysis, such as locating high-entropy
visual tokens. The second is useful for concept-level comparisons after all
channels have been mapped to the same semantic variable.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch


def entropy_from_logits(logits: torch.Tensor) -> tuple[float, float | None, int]:
    """Return raw entropy, normalized entropy, and finite candidate count."""
    finite = torch.isfinite(logits)
    finite_count = int(finite.sum().item())
    if finite_count == 0:
        return float("nan"), None, 0
    values = logits[finite].float()
    probs = torch.softmax(values, dim=-1)
    entropy = -(probs * torch.log(probs.clamp_min(1e-20))).sum().item()
    normalized = entropy / float(math.log(finite_count)) if finite_count > 1 else 0.0
    return float(entropy), float(normalized), finite_count


def attach_selected_tokens(
    trace: list[dict[str, Any]],
    completion_ids: torch.Tensor | list[int],
    tokenizer: Any,
) -> list[dict[str, Any]]:
    """Add selected token ids/text to an existing per-step entropy trace."""
    ids = completion_ids.tolist() if isinstance(completion_ids, torch.Tensor) else list(completion_ids)
    enriched = []
    for item, token_id in zip(trace, ids):
        row = dict(item)
        row["selected_token_id"] = int(token_id)
        row["selected_token"] = tokenizer.decode([int(token_id)], skip_special_tokens=False)
        enriched.append(row)
    return enriched


def text_token_entropy_trace(
    scores: tuple[torch.Tensor, ...],
    completion_ids: torch.Tensor,
    tokenizer: Any,
) -> list[dict[str, Any]]:
    """Build a per-token entropy trace from text generation scores."""
    trace = []
    token_ids = completion_ids.tolist()
    for step, (score, token_id) in enumerate(zip(scores, token_ids)):
        row_scores = score[0]
        entropy, normalized, finite_count = entropy_from_logits(row_scores)
        trace.append(
            {
                "step": step,
                "phase": "text",
                "entropy": entropy,
                "normalized_entropy": normalized,
                "finite_token_count": finite_count,
                "visual_index": None,
                "selected_token_id": int(token_id),
                "selected_token": tokenizer.decode([int(token_id)], skip_special_tokens=False),
            }
        )
    return trace


def summarize_token_entropy_trace(trace: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize a token-level entropy trace without replacing the trace."""
    visual = [row for row in trace if row.get("phase") == "visual"]
    text = [row for row in trace if row.get("phase") == "text"]
    entropies = [row["entropy"] for row in trace if row.get("entropy") is not None]
    return {
        "num_steps": len(trace),
        "num_visual_steps": len(visual),
        "num_text_steps": len(text),
        "mean_entropy": float(np.mean(entropies)) if entropies else None,
        "mean_visual_entropy": float(np.mean([row["entropy"] for row in visual])) if visual else None,
        "mean_text_entropy": float(np.mean([row["entropy"] for row in text])) if text else None,
        "first_token_entropy": float(trace[0]["entropy"]) if trace else None,
        "max_entropy": float(np.max(entropies)) if entropies else None,
    }


def write_entropy_trace(
    out_dir: Path,
    sample_id: str,
    trace: list[dict[str, Any]],
    *,
    subdir: str = "entropy",
) -> tuple[str, dict[str, Any]]:
    """Write a per-token trace JSON and return path plus compact summary."""
    entropy_dir = out_dir / subdir
    entropy_dir.mkdir(parents=True, exist_ok=True)
    path = entropy_dir / f"{sample_id}.json"
    summary = summarize_token_entropy_trace(trace)
    payload = {"sample_id": sample_id, "summary": summary, "steps": trace}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path), summary


def semantic_entropy_from_distribution(
    distribution: dict[str, float],
    options: list[str],
) -> dict[str, Any]:
    """Compute entropy over a fixed semantic attribute space."""
    probs = np.asarray([float(distribution.get(option, 0.0)) for option in options], dtype=float)
    positive = probs[probs > 0]
    entropy = float(-(positive * np.log(positive)).sum()) if positive.size else 0.0
    denom = math.log(len(options)) if len(options) > 1 else 1.0
    predicted = max(options, key=lambda option: distribution.get(option, 0.0))
    return {
        "entropy": entropy,
        "normalized_entropy": entropy / denom if denom else 0.0,
        "effective_candidates": math.exp(entropy),
        "predicted_value": predicted,
        "distribution": {option: float(distribution.get(option, 0.0)) for option in options},
    }
