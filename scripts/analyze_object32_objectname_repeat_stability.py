#!/usr/bin/env python3
"""Aggregate object-name repeat stability for generated object32 image reads."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def words(text: str) -> list[str]:
    text = re.sub(r"<\|[^>]+?\|>", " ", text.lower())
    return re.findall(r"[a-z0-9]+", text)


def normalize_answer(text: str) -> str:
    return " ".join(words(text))


def mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        yield from csv.DictReader(f)


def write_csv(rows, path: Path) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rows = list(read_csv(Path(args.sample_csv)))
    groups = {}
    for row in rows:
        if row.get("mode") != "objectname":
            continue
        groups.setdefault(row["base_id"], []).append(row)

    summary = []
    for base_id, group in sorted(groups.items()):
        answers = [normalize_answer(r["raw_text"]) for r in group]
        unique_answers = sorted(set(answers))
        counts = {answer: answers.count(answer) for answer in unique_answers}
        loose_hits = sum(int(r["target_loose_hit"]) for r in group)
        exact_hits = sum(int(r["target_exact_phrase_hit"]) for r in group)
        stable_answer = len(unique_answers) == 1
        all_loose_hit = loose_hits == len(group)
        any_loose_miss = loose_hits < len(group)
        majority_answer = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0] if counts else ""
        summary.append(
            {
                "base_id": base_id,
                "target_object": group[0]["target_object"],
                "sample_count": len(group),
                "unique_answer_count": len(unique_answers),
                "stable_answer": int(stable_answer),
                "all_loose_hit": int(all_loose_hit),
                "any_loose_miss": int(any_loose_miss),
                "loose_hits": loose_hits,
                "exact_hits": exact_hits,
                "majority_answer": majority_answer,
                "answers": " | ".join(f"{answer}:{counts[answer]}" for answer in unique_answers),
                "mean_answer_ume": mean(float(r["mean_answer_ume"]) for r in group),
                "mean_target_object_ume": mean(float(r["target_object_mean_ume"]) for r in group),
                "max_answer_ume": max(float(r["max_answer_ume"]) for r in group),
                "uas_8x8_ume_delta_nll_pearson": group[0].get("uas_8x8_ume_delta_nll_pearson", ""),
            }
        )

    overall = [
        {
            "group_count": len(summary),
            "total_samples": len(rows),
            "stable_groups": sum(int(r["stable_answer"]) for r in summary),
            "all_loose_hit_groups": sum(int(r["all_loose_hit"]) for r in summary),
            "any_loose_miss_groups": sum(int(r["any_loose_miss"]) for r in summary),
            "mean_unique_answer_count": mean(float(r["unique_answer_count"]) for r in summary),
            "mean_group_answer_ume": mean(float(r["mean_answer_ume"]) for r in summary),
            "mean_group_target_object_ume": mean(float(r["mean_target_object_ume"]) for r in summary),
        }
    ]

    out_dir = Path(args.out_dir)
    write_csv(summary, out_dir / "objectname_repeat_stability_by_image.csv")
    write_csv(overall, out_dir / "objectname_repeat_stability_overall.csv")

    lines = [
        "# Object32 Object-name Repeat Stability Report",
        "",
        "## Overall",
        "",
    ]
    if overall:
        row = overall[0]
        lines.extend(
            [
                f"- Groups: {row['group_count']}",
                f"- Total samples: {row['total_samples']}",
                f"- Stable answer groups: {row['stable_groups']}/{row['group_count']}",
                f"- All-loose-hit groups: {row['all_loose_hit_groups']}/{row['group_count']}",
                f"- Any-loose-miss groups: {row['any_loose_miss_groups']}/{row['group_count']}",
                f"- Mean unique answers per image: {float(row['mean_unique_answer_count']):.4f}",
                f"- Mean group answer UME: {float(row['mean_group_answer_ume']):.4f}",
                f"- Mean group target-object UME: {float(row['mean_group_target_object_ume']):.4f}",
            ]
        )
    lines.extend(["", "## Unstable Or Missed Groups", ""])
    for row in summary:
        if int(row["stable_answer"]) and not int(row["any_loose_miss"]):
            continue
        lines.append(
            f"- `{row['base_id']}` target={row['target_object']}: "
            f"stable={row['stable_answer']}, loose_hits={row['loose_hits']}/{row['sample_count']}, "
            f"answers={row['answers']}, mean UME={float(row['mean_answer_ume']):.4f}"
        )
    (out_dir / "objectname_repeat_stability_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[INFO] wrote repeat stability for {len(summary)} image groups to {out_dir}")


if __name__ == "__main__":
    main()
