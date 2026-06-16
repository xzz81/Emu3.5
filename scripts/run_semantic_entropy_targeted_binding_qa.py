#!/usr/bin/env python3
"""Run Emu3.5 visual QA over the targeted object-binding question package."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_semantic_uncertainty_qa_compare import (  # noqa: E402
    encode_image_qa,
    generate_answer,
    load_runtime,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--questions-csv",
        default="outputs/semantic_entropy_umm/object_binding_sanity/targeted_questions/targeted_binding_questions.csv",
    )
    parser.add_argument(
        "--outputs-csv",
        default="outputs/semantic_entropy_umm/object_binding_sanity/targeted_questions/targeted_binding_outputs_template.csv",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/semantic_entropy_umm/object_binding_sanity/targeted_questions",
    )
    parser.add_argument("--model-path", default="model/Emu3.5")
    parser.add_argument("--vq-path", default="model/Emu3.5-VisionTokenizer")
    parser.add_argument("--tokenizer-path", default="./src/tokenizer_emu3_ibq")
    parser.add_argument("--vq-type", default="ibq")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--answer-temperature", type=float, default=0.7)
    parser.add_argument("--answer-top-k", type=int, default=1024)
    parser.add_argument("--answer-top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20270608)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--mode", choices=("worker", "aggregate"), default="worker")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def append_csv(path: Path, row: dict, fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()


def normalize_text(text: str) -> str:
    value = text.lower().strip()
    value = re.sub(r"[^a-z0-9_ ]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value.replace(" ", "_")


def normalize_to_allowed(raw: str, allowed_values: list[str]) -> str:
    normalized = normalize_text(raw)
    allowed = {value: value for value in allowed_values}
    allowed.update({normalize_text(value): value for value in allowed_values})
    if normalized in allowed:
        return allowed[normalized]
    text = f" {normalized} "
    hits = [value for value in allowed_values if f" {normalize_text(value)} " in text]
    if len(hits) == 1:
        return hits[0]
    relation_aliases = {
        "left": "object_1_left_of_object_2",
        "right": "object_1_right_of_object_2",
        "above": "object_1_above_object_2",
        "over": "object_1_above_object_2",
        "below": "object_1_below_object_2",
        "under": "object_1_below_object_2",
    }
    relation_hits = [value for key, value in relation_aliases.items() if key in normalized and value in allowed_values]
    if len(relation_hits) == 1:
        return relation_hits[0]
    if "unknown" in normalized and "unknown" in allowed_values:
        return "unknown"
    return "unknown" if "unknown" in allowed_values else ""


def worker_output_path(out_dir: Path, worker_id: int) -> Path:
    return out_dir / "workers" / f"targeted_binding_outputs_worker{worker_id}.csv"


def existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {row["question_id"] for row in read_csv(path) if row.get("answer") or row.get("raw_output")}


def run_worker(args: argparse.Namespace) -> None:
    questions = read_csv(Path(args.questions_csv))
    out_dir = Path(args.out_dir)
    output_path = worker_output_path(out_dir, args.worker_id)
    done = existing_ids(output_path) if args.skip_existing else set()
    jobs = questions[args.worker_id :: args.num_workers]
    fields = ["question_id", "model_or_annotator", "answer", "raw_output", "notes"]
    cfg, model, tokenizer, vq_model = load_runtime(args)
    torch.manual_seed(args.seed + args.worker_id)
    completed = 0
    for row in jobs:
        question_id = row["question_id"]
        if question_id in done:
            continue
        torch.cuda.empty_cache()
        allowed_values = row["allowed_values"].split(";")
        input_ids, _ = encode_image_qa(
            cfg,
            tokenizer,
            vq_model,
            Path(row["image_path"]),
            row["target_description"],
            row["question"],
            model.device,
        )
        raw = generate_answer(cfg, model, tokenizer, input_ids, args)
        answer = normalize_to_allowed(raw, allowed_values)
        append_csv(
            output_path,
            {
                "question_id": question_id,
                "model_or_annotator": "Emu3.5_targeted_binding_qa",
                "answer": answer,
                "raw_output": raw,
                "notes": f"worker_id={args.worker_id};question_type={row['question_type']}",
            },
            fields,
        )
        completed += 1
        print(f"[INFO] worker {args.worker_id} wrote {question_id} -> {answer} raw={raw!r}", flush=True)
    print(f"[INFO] targeted binding worker {args.worker_id} completed {completed}/{len(jobs)} jobs", flush=True)


def aggregate(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    fields = ["question_id", "model_or_annotator", "answer", "raw_output", "notes"]
    rows = []
    seen = set()
    for path in sorted((out_dir / "workers").glob("targeted_binding_outputs_worker*.csv")):
        for row in read_csv(path):
            qid = row["question_id"]
            if qid in seen:
                continue
            seen.add(qid)
            rows.append(row)
    question_order = {row["question_id"]: idx for idx, row in enumerate(read_csv(Path(args.questions_csv)))}
    rows.sort(key=lambda row: question_order.get(row["question_id"], 10**9))
    write_csv(Path(args.outputs_csv), rows, fields)
    print(f"[INFO] aggregated {len(rows)} targeted binding answers into {args.outputs_csv}", flush=True)


def main() -> None:
    args = parse_args()
    if args.mode == "worker":
        run_worker(args)
    else:
        aggregate(args)


if __name__ == "__main__":
    main()
