#!/usr/bin/env python3
"""Evaluate T2I semantic uncertainty as a hallucination/error detector.

The analysis mirrors the jlko/semantic_uncertainty evaluation shape: each
datapoint has an uncertainty score and a false-answer label, then AUROC and
selective-accuracy metrics are reported. For T2I, a datapoint is a
concept-route-slot group with 10 generated images/readbacks.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


SLOTS = ["object_1", "color_1", "object_2", "color_2", "relation", "background"]
ALL_SLOTS = ["joint", *SLOTS]
ROUTES = ["I2T", "T2I"]
VOCAB_SIZE = {
    "object_1": 4,
    "color_1": 5,
    "object_2": 4,
    "color_2": 5,
    "relation": 5,
    "background": 3,
}
VOCAB_SIZE["joint"] = math.prod(VOCAB_SIZE[slot] for slot in SLOTS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        default="outputs/semantic_entropy_umm/semantic_uncertainty_formal_emu35_20260606_095911",
    )
    parser.add_argument("--compare-dir-name", default="semantic_uncertainty_qa")
    parser.add_argument("--route", default="T2I", choices=ROUTES)
    parser.add_argument("--out-dir", default="outputs/semantic_entropy_umm/t2i_hallucination_uncertainty")
    parser.add_argument("--case-limit", type=int, default=36)
    parser.add_argument("--contact-sheet-cols", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20270608)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None and rows:
        fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or [])
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any, digits: int = 4) -> str:
    if value in ("", None):
        return "NA"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def normalized_entropy(entropy: float, slot: str, num_samples: int) -> float:
    denom = math.log(max(1, min(VOCAB_SIZE[slot], num_samples)))
    return entropy / denom if denom > 0 else 0.0


def state_value(state: dict[str, Any], slot: str) -> str:
    slots = state["slots"]
    if slot == "joint":
        return "|".join(str(slots.get(name, "unknown")) for name in SLOTS)
    return str(slots.get(slot, "unknown"))


def target_value(state: dict[str, Any], slot: str) -> str:
    target = state["target_semantics"]
    if slot == "joint":
        return "|".join(str(target.get(name, "unknown")) for name in SLOTS)
    return str(target.get(slot, "unknown"))


def is_error(state: dict[str, Any], slot: str) -> bool:
    if slot == "joint":
        return any(state["slots"].get(name, "unknown") != state["target_semantics"].get(name) for name in SLOTS)
    return state["slots"].get(slot, "unknown") != state["target_semantics"].get(slot)


def has_unknown(state: dict[str, Any], slot: str) -> bool:
    if slot == "joint":
        return any(state["slots"].get(name, "unknown") == "unknown" for name in SLOTS)
    return state["slots"].get(slot, "unknown") == "unknown"


def auroc(labels: list[int], scores: list[float]) -> float | None:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    pairs = sorted(zip(scores, labels), key=lambda item: item[0])
    ranks = [0.0] * len(pairs)
    i = 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[k] = avg_rank
        i = j
    pos_rank_sum = sum(rank for rank, (_, label) in zip(ranks, pairs) if label == 1)
    return (pos_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = (len(values) - 1) * q
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - idx) + values[hi] * (idx - lo)


def accuracy_at_quantile(accuracies: list[int], uncertainties: list[float], q: float) -> float:
    cutoff = quantile(uncertainties, q)
    selected = [acc for acc, unc in zip(accuracies, uncertainties) if unc <= cutoff]
    return sum(selected) / len(selected) if selected else 0.0


def area_under_thresholded_accuracy(accuracies: list[int], uncertainties: list[float]) -> float:
    qs = [0.1 + idx * (0.9 / 19) for idx in range(20)]
    dx = qs[1] - qs[0]
    return sum(accuracy_at_quantile(accuracies, uncertainties, q) * dx for q in qs)


def build_indexes(run_dir: Path, compare_dir_name: str) -> tuple[dict, dict, dict]:
    compare_dir = run_dir / compare_dir_name
    states = read_jsonl(compare_dir / "qa_semantic_states.jsonl")
    entropy_rows = read_csv(compare_dir / "qa_route_entropy.csv")
    error_rows = read_csv(compare_dir / "qa_route_error.csv")
    states_by_cr: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in states:
        row["sample_id"] = int(row["sample_id"])
        states_by_cr[(row["concept_id"], row["route"])].append(row)
    for rows in states_by_cr.values():
        rows.sort(key=lambda item: item["sample_id"])
    entropy_by_crs = {(row["concept_id"], row["route"], row["slot"]): row for row in entropy_rows}
    error_by_crs = {(row["concept_id"], row["route"], row["slot"]): row for row in error_rows}
    return states_by_cr, entropy_by_crs, error_by_crs


def build_datapoints(run_dir: Path, compare_dir_name: str, route: str) -> list[dict[str, Any]]:
    states_by_cr, entropy_by_crs, error_by_crs = build_indexes(run_dir, compare_dir_name)
    rows = []
    for (concept_id, row_route), states in sorted(states_by_cr.items()):
        if row_route != route:
            continue
        for slot in ALL_SLOTS:
            entropy_row = entropy_by_crs[(concept_id, route, slot)]
            entropy = float(entropy_row["cluster_assignment_entropy"])
            num_samples = int(entropy_row["num_samples"])
            values = [state_value(state, slot) for state in states]
            targets = [target_value(state, slot) for state in states]
            target = targets[0]
            counts = Counter(values)
            mode_value, mode_count = counts.most_common(1)[0]
            sample_errors = [is_error(state, slot) for state in states]
            sample_unknowns = [has_unknown(state, slot) for state in states]
            error_rate = sum(sample_errors) / len(sample_errors)
            unknown_rate = sum(sample_unknowns) / len(sample_unknowns)
            if slot == "joint":
                error_table_rate = error_rate
            else:
                error_table_rate = float(error_by_crs[(concept_id, route, slot)]["error_rate"])
            first_state = states[0]
            rows.append(
                {
                    "concept_id": concept_id,
                    "route": route,
                    "slot": slot,
                    "cluster_assignment_entropy": entropy,
                    "normalized_entropy": normalized_entropy(entropy, slot, num_samples),
                    "effective_semantic_clusters": math.exp(entropy),
                    "num_samples": num_samples,
                    "target_value": target,
                    "mode_value": mode_value,
                    "mode_fraction": mode_count / len(values),
                    "error_rate": error_rate,
                    "error_table_rate": error_table_rate,
                    "unknown_rate": unknown_rate,
                    "any_error_label": int(error_rate > 0),
                    "majority_error_label": int(error_rate >= 0.5),
                    "mode_error_label": int(mode_value != target),
                    "first_sample_error_label": int(sample_errors[0]),
                    "first_sample_image_path": first_state["image_path"],
                    "first_sample_value": values[0],
                    "distribution": entropy_row["distribution"],
                }
            )
    return rows


def build_metrics(datapoints: list[dict[str, Any]]) -> list[dict[str, Any]]:
    score_specs = [
        ("semantic_entropy", "cluster_assignment_entropy"),
        ("normalized_entropy", "normalized_entropy"),
        ("effective_semantic_clusters", "effective_semantic_clusters"),
    ]
    label_specs = [
        ("any_error", "any_error_label"),
        ("majority_error", "majority_error_label"),
        ("mode_error", "mode_error_label"),
        ("first_sample_error", "first_sample_error_label"),
    ]
    rows = []
    groups = [("ALL", datapoints)]
    for slot in ALL_SLOTS:
        groups.append((slot, [row for row in datapoints if row["slot"] == slot]))
    for group_name, group_rows in groups:
        if not group_rows:
            continue
        for score_name, score_field in score_specs:
            scores = [float(row[score_field]) for row in group_rows]
            for label_name, label_field in label_specs:
                labels = [int(row[label_field]) for row in group_rows]
                accuracies = [1 - label for label in labels]
                auc = auroc(labels, scores)
                rows.append(
                    {
                        "group": group_name,
                        "uncertainty_measure": score_name,
                        "hallucination_label": label_name,
                        "n": len(group_rows),
                        "positives": sum(labels),
                        "positive_rate": sum(labels) / len(labels),
                        "mean_uncertainty": sum(scores) / len(scores),
                        "AUROC": "" if auc is None else auc,
                        "area_under_thresholded_accuracy": area_under_thresholded_accuracy(accuracies, scores),
                        "accuracy_at_0.8_answer_fraction": accuracy_at_quantile(accuracies, scores, 0.8),
                        "accuracy_at_0.9_answer_fraction": accuracy_at_quantile(accuracies, scores, 0.9),
                        "accuracy_at_0.95_answer_fraction": accuracy_at_quantile(accuracies, scores, 0.95),
                        "accuracy_at_1.0_answer_fraction": accuracy_at_quantile(accuracies, scores, 1.0),
                    }
                )
    return rows


def build_cases(datapoints: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    candidates = [row for row in datapoints if row["slot"] in {"joint", "object_1", "object_2", "color_1", "color_2", "relation"}]
    low_bad = sorted(
        [row for row in candidates if row["mode_error_label"] or row["majority_error_label"]],
        key=lambda row: (float(row["normalized_entropy"]), -float(row["error_rate"]), row["concept_id"], row["slot"]),
    )
    high_bad = sorted(
        [row for row in candidates if row["mode_error_label"] or row["majority_error_label"]],
        key=lambda row: (-float(row["normalized_entropy"]), -float(row["error_rate"]), row["concept_id"], row["slot"]),
    )
    high_uncertain_ok = sorted(
        [row for row in candidates if not row["mode_error_label"] and not row["majority_error_label"]],
        key=lambda row: (-float(row["normalized_entropy"]), row["concept_id"], row["slot"]),
    )
    selected = []
    seen = set()
    for bucket_name, bucket in [
        ("low_entropy_hallucination", low_bad),
        ("high_entropy_hallucination", high_bad),
        ("high_entropy_non_hallucination", high_uncertain_ok),
    ]:
        per_bucket = max(1, limit // 3)
        count = 0
        for row in bucket:
            key = (row["concept_id"], row["slot"], row["first_sample_image_path"], bucket_name)
            if key in seen:
                continue
            seen.add(key)
            selected.append(
                {
                    "case_bucket": bucket_name,
                    "concept_id": row["concept_id"],
                    "route": row["route"],
                    "slot": row["slot"],
                    "target_value": row["target_value"],
                    "mode_value": row["mode_value"],
                    "first_sample_value": row["first_sample_value"],
                    "error_rate": row["error_rate"],
                    "cluster_assignment_entropy": row["cluster_assignment_entropy"],
                    "normalized_entropy": row["normalized_entropy"],
                    "mode_fraction": row["mode_fraction"],
                    "image_path": row["first_sample_image_path"],
                    "distribution": row["distribution"],
                }
            )
            count += 1
            if count >= per_bucket:
                break
    return selected[:limit]


def make_contact_sheet(run_dir: Path, cases: list[dict[str, Any]], path: Path, cols: int) -> None:
    if not cases:
        return
    cell_w, cell_h = 320, 400
    img_size = 224
    rows = math.ceil(len(cases) / cols)
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 12)
        small = ImageFont.truetype("DejaVuSans.ttf", 10)
    except OSError:
        font = ImageFont.load_default()
        small = ImageFont.load_default()
    for idx, case in enumerate(cases):
        x = (idx % cols) * cell_w
        y = (idx // cols) * cell_h
        image_path = run_dir.parent.parent / case["image_path"]
        if not image_path.exists():
            image_path = Path(case["image_path"])
        try:
            img = Image.open(image_path).convert("RGB")
            img.thumbnail((img_size, img_size))
        except OSError:
            img = Image.new("RGB", (img_size, img_size), "#dddddd")
        sheet.paste(img, (x + 10, y + 10))
        text_lines = [
            f"#{idx} {case['case_bucket']}",
            f"{case['concept_id']}",
            f"slot={case['slot']}",
            f"target={case['target_value']}",
            f"mode={case['mode_value']} err={fmt(case['error_rate'])}",
            f"Hn={fmt(case['normalized_entropy'])} H={fmt(case['cluster_assignment_entropy'])}",
        ]
        ty = y + img_size + 18
        for line in text_lines:
            draw.text((x + 10, ty), line[:48], fill="black", font=font if ty == y + img_size + 18 else small)
            ty += 18
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def sample_image_path(run_dir: Path, concept_id: str, sample_id: int) -> Path:
    return run_dir / "pilot" / "generated_images" / concept_id / f"{sample_id:03d}.png"


def make_concept_sample_sheets(run_dir: Path, cases: list[dict[str, Any]], out_dir: Path) -> list[dict[str, Any]]:
    concepts = sorted({case["concept_id"] for case in cases})
    sheet_dir = out_dir / "concept_sample_sheets"
    sheet_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    try:
        title_font = ImageFont.truetype("DejaVuSans.ttf", 18)
        small = ImageFont.truetype("DejaVuSans.ttf", 12)
    except OSError:
        title_font = ImageFont.load_default()
        small = ImageFont.load_default()
    for concept_id in concepts:
        cell = 180
        header = 90
        cols = 5
        rows_n = 2
        sheet = Image.new("RGB", (cols * cell, header + rows_n * (cell + 30)), "white")
        draw = ImageDraw.Draw(sheet)
        case_slots = sorted({case["slot"] for case in cases if case["concept_id"] == concept_id})
        draw.text((10, 10), concept_id, fill="black", font=title_font)
        draw.text((10, 40), f"audit slots: {', '.join(case_slots)}", fill="black", font=small)
        for sample_id in range(10):
            x = (sample_id % cols) * cell
            y = header + (sample_id // cols) * (cell + 30)
            img_path = sample_image_path(run_dir, concept_id, sample_id)
            try:
                img = Image.open(img_path).convert("RGB")
                img.thumbnail((160, 160))
            except OSError:
                img = Image.new("RGB", (160, 160), "#dddddd")
            sheet.paste(img, (x + 10, y + 5))
            draw.text((x + 10, y + 168), f"sample {sample_id:03d}", fill="black", font=small)
        out_path = sheet_dir / f"{concept_id}.png"
        sheet.save(out_path)
        rows.append(
            {
                "concept_id": concept_id,
                "sheet_path": str(out_path),
                "audit_slots": ";".join(case_slots),
                "sample_count": 10,
            }
        )
    return rows


def build_report(
    out_dir: Path,
    run_dir: Path,
    datapoints: list[dict[str, Any]],
    metrics: list[dict[str, Any]],
    cases: list[dict[str, Any]],
    route: str,
) -> str:
    def table(rows: list[dict[str, Any]], fields: list[str]) -> list[str]:
        if not rows:
            return ["(none)"]
        lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
        for row in rows:
            lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
        return lines

    primary = [
        row
        for row in metrics
        if row["group"] in {"ALL", "joint", "object_1", "relation"}
        and row["uncertainty_measure"] == "semantic_entropy"
        and row["hallucination_label"] in {"mode_error", "majority_error", "any_error"}
    ]
    primary = sorted(primary, key=lambda row: (row["group"], row["hallucination_label"]))
    low_bad = [row for row in cases if row["case_bucket"] == "low_entropy_hallucination"]
    lines = [
        "# T2I Semantic Uncertainty vs Hallucination",
        "",
        f"Status: ANALYSIS_COMPLETE",
        "",
        "This report adapts the jlko/semantic_uncertainty evaluation pattern to T2I. Each concept-slot group has one semantic-uncertainty score computed over 10 generated images/readbacks, then the score is evaluated as a detector for hallucination/error labels.",
        "",
        "## Inputs",
        "",
        f"- Run dir: `{run_dir}`",
        f"- Route: `{route}`",
        f"- Semantic states: `{run_dir / 'semantic_uncertainty_qa' / 'qa_semantic_states.jsonl'}`",
        f"- Entropy: `{run_dir / 'semantic_uncertainty_qa' / 'qa_route_entropy.csv'}`",
        f"- Error: `{run_dir / 'semantic_uncertainty_qa' / 'qa_route_error.csv'}`",
        "",
        "## Output Files",
        "",
        f"- `{out_dir / 't2i_uncertainty_datapoints.csv'}`",
        f"- `{out_dir / 't2i_uncertainty_detector_metrics.csv'}`",
        f"- `{out_dir / 't2i_hallucination_visual_audit_manifest.csv'}`",
        f"- `{out_dir / 't2i_hallucination_contact_sheet.png'}`",
        f"- `{out_dir / 'concept_sample_sheets'}`",
        f"- `{out_dir / 't2i_hallucination_uncertainty_report.md'}`",
        "",
        "## Method Mapping From jlko/semantic_uncertainty",
        "",
        "- jlko: multiple text answers per question -> semantic equivalence classes -> semantic entropy -> `validation_is_false` detector metrics.",
        "- T2I adaptation: multiple generated images/readbacks per concept-slot -> finite semantic clusters -> cluster-assignment semantic entropy -> hallucination/error detector metrics.",
        "- Reported detector metrics follow the same shape: AUROC, area under thresholded accuracy, and accuracy at retained low-uncertainty fractions.",
        "",
        "## Label Definitions",
        "",
        "- `any_error`: at least one of 10 generated T2I samples mismatches the target slot/state.",
        "- `majority_error`: at least 50% of generated T2I samples mismatch the target.",
        "- `mode_error`: the most frequent semantic cluster mismatches the target; this is closest to judging the representative answer in jlko.",
        "- `first_sample_error`: the first generated sample mismatches the target; included as a single-sample diagnostic.",
        "",
        "## Sample Counts",
        "",
        f"- Datapoints: `{len(datapoints)}` concept-slot groups",
        f"- Concepts: `{len({row['concept_id'] for row in datapoints})}`",
        f"- Slots including joint: `{len(ALL_SLOTS)}`",
        f"- Visual-audit candidate cases: `{len(cases)}`",
        "",
        "## Primary Detector Metrics",
        "",
        *table(
            [
                {
                    **row,
                    "mean_uncertainty": fmt(row["mean_uncertainty"]),
                    "AUROC": fmt(row["AUROC"]) if row["AUROC"] != "" else "",
                    "area_under_thresholded_accuracy": fmt(row["area_under_thresholded_accuracy"]),
                    "accuracy_at_0.8_answer_fraction": fmt(row["accuracy_at_0.8_answer_fraction"]),
                    "accuracy_at_0.9_answer_fraction": fmt(row["accuracy_at_0.9_answer_fraction"]),
                }
                for row in primary
            ],
            [
                "group",
                "hallucination_label",
                "n",
                "positives",
                "positive_rate",
                "mean_uncertainty",
                "AUROC",
                "area_under_thresholded_accuracy",
                "accuracy_at_0.8_answer_fraction",
                "accuracy_at_0.9_answer_fraction",
            ],
        ),
        "",
        "## Low-Entropy Hallucination Cases",
        "",
        *table(
            [
                {
                    "concept_id": row["concept_id"],
                    "slot": row["slot"],
                    "target": row["target_value"],
                    "mode": row["mode_value"],
                    "error_rate": fmt(row["error_rate"]),
                    "normalized_entropy": fmt(row["normalized_entropy"]),
                    "image_path": row["image_path"],
                }
                for row in low_bad[:12]
            ],
            ["concept_id", "slot", "target", "mode", "error_rate", "normalized_entropy", "image_path"],
        ),
        "",
        "## Initial Interpretation",
        "",
        "- High semantic entropy can be evaluated as a hallucination detector, but low semantic entropy must not be read as correctness: the low-entropy hallucination cases are stable wrong generations/readbacks.",
        "- The T2I setting therefore needs both uncertainty metrics and error labels, matching the original semantic_uncertainty distinction between uncertainty and false-answer validation.",
        "- The contact sheet is intended for manual visual audit of whether automatic slot errors are actual image hallucinations or extractor/readback artifacts.",
        "",
        "## Broader Unified Entropy Questions",
        "",
        "- I2T/T2I unified entropy: compare whether entropy predicts hallucination symmetrically across image understanding and image generation after projection into the same semantic state space.",
        "- Route asymmetry: test whether low-entropy high-hallucination appears more often in T2I than I2T, indicating stable generation failure rather than answer uncertainty.",
        "- Binding-specific entropy: separate object/color/relation entropy to determine whether hallucination is driven by attribute binding, spatial relation, or object identity.",
        "- X2X unified entropy: extend the state variable to interleaved text/image outputs, scoring consistency between generated captions, generated images, and cross-modal references over the same latent scene graph.",
        "- Intervention hooks: use route-specific entropy/hallucination cases to select candidates for attention/head/residual patching rather than treating entropy as only an output metric.",
        "",
        "## Claim Boundary",
        "",
        "This analysis evaluates semantic-state uncertainty as a detector for controlled T2I hallucination/error labels. It does not claim raw text/image entropy comparability or a unified internal entropy space.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    datapoints = build_datapoints(run_dir, args.compare_dir_name, args.route)
    metrics = build_metrics(datapoints)
    cases = build_cases(datapoints, args.case_limit)
    write_csv(out_dir / "t2i_uncertainty_datapoints.csv", datapoints)
    write_csv(out_dir / "t2i_uncertainty_detector_metrics.csv", metrics)
    write_csv(out_dir / "t2i_hallucination_visual_audit_manifest.csv", cases)
    make_contact_sheet(run_dir, cases, out_dir / "t2i_hallucination_contact_sheet.png", args.contact_sheet_cols)
    sheet_rows = make_concept_sample_sheets(run_dir, cases, out_dir)
    write_csv(out_dir / "t2i_hallucination_concept_sheet_index.csv", sheet_rows)
    report = build_report(out_dir, run_dir, datapoints, metrics, cases, args.route)
    (out_dir / "t2i_hallucination_uncertainty_report.md").write_text(report, encoding="utf-8")
    print(f"[INFO] wrote {len(datapoints)} datapoints, {len(metrics)} metrics, {len(cases)} visual-audit cases to {out_dir}")


if __name__ == "__main__":
    main()
