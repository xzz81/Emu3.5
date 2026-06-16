#!/usr/bin/env python3
"""Build a readable text-token entropy report for synthetic concept benchmarks."""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Any

import numpy as np


BENCH_DIR = Path("outputs/bench/modal_memory_lora_entropy_full")
TASK_DIR = Path("outputs/bench/task_entropy_compare_full")
OUT_DIR = BENCH_DIR / "entropy_showcase"


SPECIAL_LABELS = {
    "<|extra_101|>": "[assistant-end]",
    "<|extra_204|>": "[eos]",
    "<|endoftext|>": "[pad/eos]",
}


def read_jsonl(paths: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line:
                rows.append(json.loads(line))
    return rows


def load_steps(path: str | None) -> list[dict[str, Any]]:
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        return []
    return json.loads(p.read_text(encoding="utf-8")).get("steps", [])


def visible_token(token: str) -> str:
    return SPECIAL_LABELS.get(token, token.replace("\n", "\\n"))


def entropy_color(entropy: float, vmax: float = 2.0) -> str:
    # Light yellow -> orange -> red. Keep text readable.
    t = min(max(entropy / vmax, 0.0), 1.0)
    r = int(255)
    g = int(244 - 150 * t)
    b = int(204 - 170 * t)
    return f"rgb({r},{g},{b})"


def token_chips(steps: list[dict[str, Any]]) -> str:
    chips = []
    for step in steps:
        token = visible_token(str(step.get("selected_token", "")))
        entropy = float(step.get("entropy", 0.0))
        label = html.escape(token if token.strip() else repr(token))
        chips.append(
            "<span class='tok' "
            f"style='background:{entropy_color(entropy)}' "
            f"title='step {step.get('step')} | entropy {entropy:.4f} | normalized {float(step.get('normalized_entropy', 0.0)):.4f}'>"
            f"{label}<small>{entropy:.2f}</small></span>"
        )
    return " ".join(chips)


def trace_string(steps: list[dict[str, Any]]) -> str:
    return " ".join(f"{visible_token(str(s.get('selected_token', '')))}({float(s.get('entropy', 0.0)):.3f})" for s in steps)


def text_qa_case(row: dict[str, Any]) -> dict[str, Any]:
    steps = load_steps(row.get("entropy_path"))
    first_entropy = float(steps[0]["entropy"]) if steps else None
    gt = f"{row['concept_type']} | expected {row['expected_key']} | real={row['concept_value']} | fake={row['concept_value_synthetic']}"
    return {
        "source": "text_QA",
        "sample_id": row["sample_id"],
        "attribute": row["concept_type"],
        "gt": gt,
        "prompt": row["prompt"],
        "expected_key": row["expected_key"],
        "answer_key": row.get("grading_answer_key"),
        "generated": row.get("inference_completion", ""),
        "correct": bool(row.get("grading_correct")),
        "first_entropy": first_entropy,
        "mean_entropy": float(np.mean([float(s["entropy"]) for s in steps])) if steps else None,
        "trace": trace_string(steps),
        "chips": token_chips(steps),
    }


def image_understanding_case(row: dict[str, Any]) -> dict[str, Any]:
    steps = load_steps(row.get("entropy_path"))
    first_entropy = float(steps[0]["entropy"]) if steps else None
    gt = f"{row['attribute']} = {row['expected_value']} | expected {row['expected_key']}"
    return {
        "source": "image_understanding",
        "sample_id": row["sample_id"],
        "attribute": row["attribute"],
        "gt": gt,
        "prompt": row.get("options_text", ""),
        "expected_key": row["expected_key"],
        "answer_key": row.get("grading_answer_key"),
        "generated": row.get("inference_completion", ""),
        "correct": bool(row.get("grading_correct")),
        "first_entropy": first_entropy,
        "mean_entropy": float(np.mean([float(s["entropy"]) for s in steps])) if steps else None,
        "trace": trace_string(steps),
        "chips": token_chips(steps),
    }


def write_case_csv(cases: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "source",
        "sample_id",
        "attribute",
        "gt",
        "expected_key",
        "answer_key",
        "generated",
        "correct",
        "first_entropy",
        "mean_entropy",
        "trace",
        "prompt",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for case in cases:
            writer.writerow({field: case.get(field) for field in fields})


def stats(cases: list[dict[str, Any]]) -> dict[str, float | int | None]:
    vals = [case["first_entropy"] for case in cases if case["first_entropy"] is not None]
    arr = np.asarray(vals, dtype=float)
    if arr.size == 0:
        return {"n": 0, "mean": None, "median": None, "p90": None}
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
    }


def render_rows(cases: list[dict[str, Any]], limit: int | None = None) -> str:
    rows = []
    subset = cases if limit is None else cases[:limit]
    for case in subset:
        ok = "ok" if case["correct"] else "bad"
        first = "" if case["first_entropy"] is None else f"{case['first_entropy']:.3f}"
        mean = "" if case["mean_entropy"] is None else f"{case['mean_entropy']:.3f}"
        rows.append(
            "<tr>"
            f"<td><span class='{ok}'>{'✓' if case['correct'] else '✗'}</span></td>"
            f"<td>{html.escape(case['source'])}</td>"
            f"<td>{html.escape(case['sample_id'])}</td>"
            f"<td>{html.escape(case['attribute'])}</td>"
            f"<td>{html.escape(case['gt'])}</td>"
            f"<td>{html.escape(str(case['answer_key']))} / {html.escape(str(case['expected_key']))}</td>"
            f"<td><code>{html.escape(case['generated'])}</code></td>"
            f"<td class='num'>{first}</td>"
            f"<td class='num'>{mean}</td>"
            f"<td class='tokens'>{case['chips']}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def render_html(text_cases: list[dict[str, Any]], read_cases: list[dict[str, Any]], path: Path) -> None:
    text_sorted = sorted(text_cases, key=lambda c: (not c["correct"], -(c["first_entropy"] or -1)))
    read_wrong = sorted([c for c in read_cases if not c["correct"]], key=lambda c: -(c["first_entropy"] or -1))
    read_high = sorted(read_cases, key=lambda c: -(c["first_entropy"] or -1))[:240]
    all_compact = text_sorted + read_wrong + read_high
    by_attr = {}
    for attr in ("color", "pattern", "position", "shape"):
        by_attr[attr] = stats([c for c in read_cases if c["attribute"] == attr])

    summary_cards = [
        ("Text QA", stats(text_cases)),
        ("Image Understanding", stats(read_cases)),
        ("Image Understanding Wrong", stats(read_wrong)),
    ]
    cards = []
    for title, item in summary_cards:
        cards.append(
            f"<div class='card'><h3>{title}</h3>"
            f"<p><b>n</b> {item['n']}</p><p><b>mean</b> {item['mean']:.3f}</p>"
            f"<p><b>median</b> {item['median']:.3f}</p><p><b>p90</b> {item['p90']:.3f}</p></div>"
            if item["n"]
            else f"<div class='card'><h3>{title}</h3><p>n 0</p></div>"
        )
    attr_rows = "\n".join(
        f"<tr><td>{attr}</td><td>{item['n']}</td><td>{item['mean']:.3f}</td><td>{item['median']:.3f}</td><td>{item['p90']:.3f}</td></tr>"
        for attr, item in by_attr.items()
    )
    html_text = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Readable Text Entropy</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #20242a; }}
    h1, h2 {{ margin-bottom: 8px; }}
    .cards {{ display: flex; gap: 12px; flex-wrap: wrap; margin: 14px 0 20px; }}
    .card {{ border: 1px solid #ddd; border-radius: 8px; padding: 10px 14px; min-width: 190px; background: #fafafa; }}
    .card h3 {{ margin: 0 0 8px; font-size: 15px; }}
    .card p {{ margin: 3px 0; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 12px; margin: 12px 0 28px; }}
    th, td {{ border: 1px solid #ddd; padding: 6px 7px; vertical-align: top; }}
    th {{ background: #f1f3f5; position: sticky; top: 0; }}
    .num {{ text-align: right; font-variant-numeric: tabular-nums; }}
    .tok {{ display: inline-flex; gap: 4px; align-items: baseline; border: 1px solid rgba(0,0,0,.13); border-radius: 6px; padding: 3px 5px; margin: 1px 2px 2px 0; }}
    .tok small {{ font-size: 10px; opacity: .78; }}
    .tokens {{ min-width: 230px; }}
    .ok {{ color: #16833a; font-weight: 700; }}
    .bad {{ color: #b00020; font-weight: 700; }}
    code {{ white-space: pre-wrap; }}
    .note {{ color: #555; max-width: 980px; line-height: 1.45; }}
  </style>
</head>
<body>
  <h1>Readable Text Entropy Report</h1>
  <p class="note">每个彩色块是一个生成 token；块内数字是该 token 的 entropy。颜色越红，模型生成该 token 时越不确定。Special tokens 被保留为 [assistant-end] / [eos]，方便看到完整停止过程。</p>
  <div class="cards">{''.join(cards)}</div>

  <h2>Image Understanding By Attribute</h2>
  <table>
    <tr><th>attribute</th><th>n</th><th>mean first-token H</th><th>median</th><th>p90</th></tr>
    {attr_rows}
  </table>

  <h2>Text QA: All 44 Cases</h2>
  <table>
    <tr><th>ok</th><th>source</th><th>sample</th><th>attr</th><th>GT</th><th>answer/expected</th><th>generated</th><th>first H</th><th>mean H</th><th>token entropy</th></tr>
    {render_rows(text_sorted)}
  </table>

  <h2>Image Understanding: Wrong Cases First</h2>
  <table>
    <tr><th>ok</th><th>source</th><th>sample</th><th>attr</th><th>GT</th><th>answer/expected</th><th>generated</th><th>first H</th><th>mean H</th><th>token entropy</th></tr>
    {render_rows(read_wrong)}
  </table>

  <h2>Image Understanding: Highest-Entropy Examples</h2>
  <table>
    <tr><th>ok</th><th>source</th><th>sample</th><th>attr</th><th>GT</th><th>answer/expected</th><th>generated</th><th>first H</th><th>mean H</th><th>token entropy</th></tr>
    {render_rows(read_high)}
  </table>

  <h2>Compact Mixed View</h2>
  <table>
    <tr><th>ok</th><th>source</th><th>sample</th><th>attr</th><th>GT</th><th>answer/expected</th><th>generated</th><th>first H</th><th>mean H</th><th>token entropy</th></tr>
    {render_rows(all_compact, limit=360)}
  </table>
</body>
</html>
"""
    path.write_text(html_text, encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    text_rows = read_jsonl(sorted(BENCH_DIR.glob("text_memory_worker*.jsonl")))
    read_rows = read_jsonl(sorted(TASK_DIR.glob("image_understanding_worker*.jsonl")))
    text_cases = [text_qa_case(row) for row in text_rows]
    read_cases = [image_understanding_case(row) for row in read_rows]

    all_cases = text_cases + read_cases
    write_case_csv(all_cases, OUT_DIR / "readable_text_entropy_cases.csv")
    render_html(text_cases, read_cases, OUT_DIR / "readable_text_entropy_report.html")
    manifest = {
        "html": str(OUT_DIR / "readable_text_entropy_report.html"),
        "csv": str(OUT_DIR / "readable_text_entropy_cases.csv"),
        "num_text_qa_cases": len(text_cases),
        "num_image_understanding_cases": len(read_cases),
    }
    (OUT_DIR / "readable_text_entropy_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
