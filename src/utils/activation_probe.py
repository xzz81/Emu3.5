# Copyright 2026
# SPDX-License-Identifier: Apache-2.0

"""Utilities for lightweight MLP-neuron activation probes."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import torch


STRUCTURE_TOKEN_NAMES = {
    "BOS",
    "EOS",
    "PAD",
    "EOL",
    "EOF",
    "TMS",
    "IMG",
    "BOI",
    "EOI",
    "BSS",
    "ESS",
    "BOG",
    "EOG",
    "BOC",
    "EOC",
}


def token_type_for_id(
    token_id: int,
    special_token_ids: Mapping[str, int],
    visual_token_start: Optional[int] = None,
) -> str:
    special_ids = {int(v): k for k, v in special_token_ids.items() if k in STRUCTURE_TOKEN_NAMES}
    if int(token_id) in special_ids:
        return "structure"
    if visual_token_start is None:
        visual_token_start = int(special_token_ids.get("BOV", 151854))
    if int(token_id) >= int(visual_token_start):
        return "visual"
    return "text"


def token_group_labels(
    input_ids: torch.Tensor,
    task_label: str,
    special_token_ids: Mapping[str, int],
    visual_token_start: Optional[int] = None,
) -> List[Tuple[str, ...]]:
    ids = input_ids[0].detach().cpu().tolist()
    labels = []
    for token_id in ids:
        token_type = token_type_for_id(token_id, special_token_ids, visual_token_start)
        labels.append((f"task={task_label}", f"token_type={token_type}", f"task={task_label}|token_type={token_type}"))
    return labels


class ActivationAccumulator:
    """Accumulate per-neuron activation moments without retaining activations."""

    def __init__(self) -> None:
        self._stats: Dict[Tuple[int, str], Dict[str, object]] = {}

    def update(
        self,
        layer_idx: int,
        group_labels: Sequence[Union[str, Sequence[str]]],
        activation: torch.Tensor,
    ) -> None:
        if activation.dim() != 3 or activation.shape[0] != 1:
            raise ValueError(f"expected activation shape [1, seq, neurons], got {tuple(activation.shape)}")
        if activation.shape[1] != len(group_labels):
            raise ValueError(
                f"activation seq len {activation.shape[1]} does not match {len(group_labels)} group labels"
            )

        flat_labels = []
        for item in group_labels:
            if isinstance(item, str):
                flat_labels.append((item,))
            else:
                flat_labels.append(tuple(item))
        all_labels = sorted({label for labels in flat_labels for label in labels})
        values = activation[0].detach().float()

        for label in all_labels:
            positions = [idx for idx, labels in enumerate(flat_labels) if label in labels]
            if not positions:
                continue
            selected = values[positions, :]
            key = (int(layer_idx), label)
            if key not in self._stats:
                neurons = selected.shape[-1]
                self._stats[key] = {
                    "count": 0,
                    "sum": torch.zeros(neurons, dtype=torch.float64),
                    "abs_sum": torch.zeros(neurons, dtype=torch.float64),
                    "sq_sum": torch.zeros(neurons, dtype=torch.float64),
                    "max_abs": torch.zeros(neurons, dtype=torch.float64),
                }
            stat = self._stats[key]
            selected_cpu = selected.detach().cpu().to(torch.float64)
            stat["count"] = int(stat["count"]) + int(selected_cpu.shape[0])
            stat["sum"] += selected_cpu.sum(dim=0)
            stat["abs_sum"] += selected_cpu.abs().sum(dim=0)
            stat["sq_sum"] += selected_cpu.square().sum(dim=0)
            stat["max_abs"] = torch.maximum(stat["max_abs"], selected_cpu.abs().max(dim=0).values)

    def group_rows(self) -> List[Dict[str, object]]:
        rows = []
        for (layer_idx, label), stat in sorted(self._stats.items()):
            count = int(stat["count"])
            if count <= 0:
                continue
            mean = stat["sum"] / count
            mean_abs = stat["abs_sum"] / count
            second = stat["sq_sum"] / count
            variance = torch.clamp(second - mean.square(), min=0.0)
            rows.append(
                {
                    "layer": layer_idx,
                    "group": label,
                    "count": count,
                    "mean_abs_avg": float(mean_abs.mean().item()),
                    "mean_abs_max": float(mean_abs.max().item()),
                    "mean_avg": float(mean.mean().item()),
                    "std_avg": float(torch.sqrt(variance).mean().item()),
                    "active_neurons_mean_abs_gt_1": int((mean_abs > 1.0).sum().item()),
                    "active_neurons_mean_abs_gt_3": int((mean_abs > 3.0).sum().item()),
                }
            )
        return rows

    def top_active_rows(self, top_k: int = 50) -> List[Dict[str, object]]:
        rows = []
        for (layer_idx, label), stat in sorted(self._stats.items()):
            count = int(stat["count"])
            if count <= 0:
                continue
            mean = stat["sum"] / count
            mean_abs = stat["abs_sum"] / count
            max_abs = stat["max_abs"]
            k = min(int(top_k), int(mean_abs.numel()))
            values, indices = torch.topk(mean_abs, k=k)
            for rank, (value, neuron_idx) in enumerate(zip(values.tolist(), indices.tolist()), start=1):
                rows.append(
                    {
                        "layer": layer_idx,
                        "group": label,
                        "rank": rank,
                        "neuron": int(neuron_idx),
                        "mean_abs": float(value),
                        "mean": float(mean[neuron_idx].item()),
                        "max_abs": float(max_abs[neuron_idx].item()),
                        "count": count,
                    }
                )
        return rows

    def top_contrast_rows(self, positive_group: str, negative_group: str, top_k: int = 100) -> List[Dict[str, object]]:
        rows = []
        layers = sorted({layer for layer, group in self._stats if group in {positive_group, negative_group}})
        for layer_idx in layers:
            pos = self._stats.get((layer_idx, positive_group))
            neg = self._stats.get((layer_idx, negative_group))
            if pos is None or neg is None:
                continue
            pos_count = int(pos["count"])
            neg_count = int(neg["count"])
            if pos_count <= 0 or neg_count <= 0:
                continue
            pos_mean_abs = pos["abs_sum"] / pos_count
            neg_mean_abs = neg["abs_sum"] / neg_count
            delta = pos_mean_abs - neg_mean_abs
            ratio = (pos_mean_abs + 1e-9) / (neg_mean_abs + 1e-9)
            k = min(int(top_k), int(delta.numel()))
            values, indices = torch.topk(delta, k=k)
            for rank, (value, neuron_idx) in enumerate(zip(values.tolist(), indices.tolist()), start=1):
                rows.append(
                    {
                        "layer": layer_idx,
                        "rank": rank,
                        "neuron": int(neuron_idx),
                        "positive_group": positive_group,
                        "negative_group": negative_group,
                        "delta_mean_abs": float(value),
                        "ratio_mean_abs": float(ratio[neuron_idx].item()),
                        "positive_mean_abs": float(pos_mean_abs[neuron_idx].item()),
                        "negative_mean_abs": float(neg_mean_abs[neuron_idx].item()),
                        "positive_count": pos_count,
                        "negative_count": neg_count,
                    }
                )
        rows.sort(key=lambda row: row["delta_mean_abs"], reverse=True)
        return rows[:top_k]


class MLPIntermediateActivationProbe:
    """Hook Emu3 SwiGLU intermediate activations: act(gate_proj(x)) * up_proj(x)."""

    def __init__(
        self,
        model,
        accumulator: ActivationAccumulator,
        layer_indices: Optional[Iterable[int]] = None,
    ) -> None:
        self.model = model
        self.accumulator = accumulator
        self.layer_indices = self._resolve_layers(layer_indices)
        self.current_labels: Optional[List[Union[str, Sequence[str]]]] = None
        self._handles = []
        self._gate_outputs: Dict[int, torch.Tensor] = {}

    def _resolve_layers(self, layer_indices: Optional[Iterable[int]]) -> List[int]:
        num_layers = len(self.model.model.layers)
        if layer_indices is None:
            return list(range(num_layers))
        resolved = []
        for idx in layer_indices:
            idx = int(idx)
            if idx < 0:
                idx += num_layers
            if idx < 0 or idx >= num_layers:
                raise ValueError(f"layer index {idx} outside [0, {num_layers})")
            resolved.append(idx)
        return sorted(set(resolved))

    def install(self) -> None:
        for layer_idx in self.layer_indices:
            mlp = self.model.model.layers[layer_idx].mlp
            self._handles.append(
                mlp.gate_proj.register_forward_hook(self._make_gate_hook(layer_idx, mlp.act_fn))
            )
            self._handles.append(mlp.up_proj.register_forward_hook(self._make_up_hook(layer_idx)))

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._gate_outputs.clear()

    def set_context(self, labels: Sequence[Union[str, Sequence[str]]]) -> None:
        self.current_labels = list(labels)

    def clear_context(self) -> None:
        self.current_labels = None
        self._gate_outputs.clear()

    def _make_gate_hook(self, layer_idx: int, act_fn):
        def hook(_module, _inputs, output):
            if self.current_labels is None:
                return
            self._gate_outputs[layer_idx] = act_fn(output.detach())

        return hook

    def _make_up_hook(self, layer_idx: int):
        def hook(_module, _inputs, output):
            labels = self.current_labels
            gate = self._gate_outputs.pop(layer_idx, None)
            if labels is None or gate is None:
                return
            if output.shape[:2] != gate.shape[:2] or output.shape[1] != len(labels):
                return
            intermediate = gate * output.detach()
            self.accumulator.update(layer_idx, labels, intermediate)

        return hook


def write_csv_rows(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl_rows(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
