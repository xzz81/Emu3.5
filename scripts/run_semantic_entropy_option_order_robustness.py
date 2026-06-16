#!/usr/bin/env python3
"""Option-order robustness for strict semantic entropy extraction.

This script reuses the strict_compare images and parser, but shuffles the
allowed value order in the forced-choice prompt. It writes to a separate output
directory and never overwrites the baseline strict_compare results.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_semantic_entropy_strict_compare import (  # noqa: E402
    SLOTS,
    VOCAB,
    append_jsonl,
    build_jobs,
    encode_image_question,
    entropy,
    existing_state_keys,
    generate_answer,
    load_runtime,
    normalize_answer,
    read_jsonl,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("worker", "aggregate", "audit", "write-launcher"), required=True)
    parser.add_argument("--pilot-dir", default="outputs/semantic_entropy_umm/pilot")
    parser.add_argument("--baseline-strict-dir", default="outputs/semantic_entropy_umm/strict_compare")
    parser.add_argument("--out-dir", default="outputs/semantic_entropy_umm/robustness_option_order")
    parser.add_argument("--model-path", default="model/Emu3.5")
    parser.add_argument("--vq-path", default="model/Emu3.5-VisionTokenizer")
    parser.add_argument("--tokenizer-path", default="./src/tokenizer_emu3_ibq")
    parser.add_argument("--vq-type", default="ibq")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--samples-per-concept", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--max-jobs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20270605)
    parser.add_argument("--option-order-seed", type=int, default=20270605)
    parser.add_argument("--option-order-seeds", default="20270605,20270606,20270607")
    parser.add_argument("--gpu-pairs", default="auto")
    parser.add_argument("--validation-report", default="outputs/semantic_entropy_umm/extractor_validation/manual_annotation_validation_report.json")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def shuffled_values(slot: str, option_order_seed: int) -> list[str]:
    values = list(VOCAB[slot])
    rng = random.Random(option_order_seed + sum((idx + 1) * ord(ch) for idx, ch in enumerate(slot)))
    rng.shuffle(values)
    return values


def forced_choice_question(slot: str, concept: dict, option_order_seed: int) -> tuple[str, list[str]]:
    target = concept["target_semantics"]
    prompt = concept["prompt"]
    values = shuffled_values(slot, option_order_seed)
    choices = ", ".join(values)
    if slot == "object_1":
        ask = "Which shape is object_1, the first object named in the target description?"
    elif slot == "color_1":
        ask = f"Which color is object_1 ({target['object_1']})?"
    elif slot == "object_2":
        ask = "Which shape is object_2, the second object named in the target description?"
    elif slot == "color_2":
        ask = f"Which color is object_2 ({target['object_2']})?"
    elif slot == "relation":
        ask = "What is the spatial relation from object_1 to object_2?"
    else:
        ask = "What is the image background color category?"
    question = (
        f"Target description: {prompt}\n"
        f"{ask}\n"
        f"Choose exactly one label from: {choices}.\n"
        "Answer with the label only."
    )
    return question, values


def run_worker(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    worker_state_path = out_dir / "workers" / f"option_order_states_worker{args.worker_id}.jsonl"
    worker_answer_path = out_dir / "workers" / f"option_order_slot_answers_worker{args.worker_id}.jsonl"
    done = existing_state_keys(worker_state_path) if args.skip_existing else set()
    jobs = build_jobs(args)[args.worker_id :: args.num_workers]
    if args.max_jobs is not None:
        jobs = jobs[: args.max_jobs]
    cfg, model, tokenizer, vq_model = load_runtime(args)
    torch.manual_seed(args.seed + args.worker_id)
    for job in jobs:
        concept = job["concept"]
        concept_id = concept["concept_id"]
        route = job["route"]
        sample_id = int(job["sample_id"])
        if (concept_id, route, sample_id) in done:
            continue
        image_path = Path(job["image_path"])
        slots = {}
        evidence = {}
        option_orders = {}
        for slot in SLOTS:
            torch.cuda.empty_cache()
            question, values = forced_choice_question(slot, concept, args.option_order_seed)
            input_ids = encode_image_question(cfg, tokenizer, vq_model, image_path, question, model.device)
            raw = generate_answer(cfg, model, tokenizer, input_ids, args.max_new_tokens)
            value = normalize_answer(slot, raw)
            slots[slot] = value
            evidence[slot] = raw
            option_orders[slot] = values
            append_jsonl(
                worker_answer_path,
                {
                    "concept_id": concept_id,
                    "route": route,
                    "sample_id": sample_id,
                    "slot": slot,
                    "question": question,
                    "raw_output": raw,
                    "value": value,
                    "allowed_values": values,
                    "canonical_allowed_values": VOCAB[slot],
                    "image_path": str(image_path),
                    "option_order_seed": args.option_order_seed,
                },
            )
        append_jsonl(
            worker_state_path,
            {
                "concept_id": concept_id,
                "route": route,
                "sample_id": sample_id,
                "image_path": str(image_path),
                "slots": slots,
                "evidence": evidence,
                "target_semantics": concept["target_semantics"],
                "parser": f"strict_forced_choice_vqa_option_order_seed_{args.option_order_seed}",
                "slot_schema": SLOTS,
                "option_orders": option_orders,
            },
        )
    print(f"[INFO] option-order worker {args.worker_id} finished {len(jobs)} state jobs", flush=True)


def aggregate(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    states = []
    answers = []
    for path in sorted((out_dir / "workers").glob("option_order_states_worker*.jsonl")):
        states.extend(read_jsonl(path))
    for path in sorted((out_dir / "workers").glob("option_order_slot_answers_worker*.jsonl")):
        answers.extend(read_jsonl(path))
    with (out_dir / "option_order_semantic_states.jsonl").open("w", encoding="utf-8") as handle:
        for row in states:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (out_dir / "option_order_slot_answers.jsonl").open("w", encoding="utf-8") as handle:
        for row in answers:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    entropy_rows = []
    error_rows = []
    concept_ids = sorted({row["concept_id"] for row in states})
    for concept_id in concept_ids:
        for route in ["I2T", "T2I"]:
            rows = [row for row in states if row["concept_id"] == concept_id and row["route"] == route]
            if not rows:
                continue
            joint_counts = Counter(tuple(row["slots"].get(slot, "unknown") for slot in SLOTS) for row in rows)
            entropy_rows.append(
                {
                    "concept_id": concept_id,
                    "route": route,
                    "slot": "joint",
                    "entropy": entropy(joint_counts),
                    "num_samples": len(rows),
                    "distribution": json.dumps({"|".join(k): v for k, v in joint_counts.items()}, ensure_ascii=False),
                }
            )
            target = rows[0]["target_semantics"]
            for slot in SLOTS:
                counts = Counter(row["slots"].get(slot, "unknown") for row in rows)
                misses = [row["slots"].get(slot, "unknown") != target.get(slot) for row in rows]
                entropy_rows.append(
                    {
                        "concept_id": concept_id,
                        "route": route,
                        "slot": slot,
                        "entropy": entropy(counts),
                        "num_samples": len(rows),
                        "distribution": json.dumps(dict(counts), ensure_ascii=False),
                    }
                )
                error_rows.append(
                    {
                        "concept_id": concept_id,
                        "route": route,
                        "slot": slot,
                        "error_rate": sum(misses) / len(misses),
                        "num_samples": len(misses),
                        "target_value": target.get(slot),
                    }
                )
    write_csv(out_dir / "option_order_route_entropy.csv", entropy_rows)
    write_csv(out_dir / "option_order_route_error.csv", error_rows)
    write_option_order_audit(out_dir, Path(args.baseline_strict_dir), states, answers, entropy_rows, error_rows, args)
    print(f"[INFO] option-order aggregate wrote {out_dir}", flush=True)


def load_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def mean(rows: list[dict], field: str) -> float | None:
    vals = [float(row[field]) for row in rows]
    return sum(vals) / len(vals) if vals else None


def value_distribution(rows: list[dict]) -> dict[str, int]:
    return dict(Counter(row.get("value", "unknown") for row in rows))


def write_option_order_audit(
    out_dir: Path,
    baseline_dir: Path,
    states: list[dict],
    answers: list[dict],
    entropy_rows: list[dict],
    error_rows: list[dict],
    args: argparse.Namespace,
) -> None:
    baseline_answers = read_jsonl(baseline_dir / "strict_slot_answers.jsonl")
    baseline_entropy = load_csv(baseline_dir / "strict_route_entropy.csv")
    baseline_error = load_csv(baseline_dir / "strict_route_error.csv")
    keyset = {(row["concept_id"], row["route"], int(row["sample_id"]), row["slot"]) for row in answers}
    baseline_subset = [
        row
        for row in baseline_answers
        if (row["concept_id"], row["route"], int(row["sample_id"]), row["slot"]) in keyset
    ]
    state_keys = {(row["concept_id"], row["route"], int(row["sample_id"])) for row in states}
    baseline_state_count = len({(concept_id, route, sample_id) for concept_id, route, sample_id, _ in keyset})
    answers_by_route_slot = Counter((row["route"], row["slot"]) for row in answers)
    unknown_by_route_slot = Counter((row["route"], row["slot"]) for row in answers if row["value"] == "unknown")
    changed = 0
    baseline_by_key = {
        (row["concept_id"], row["route"], int(row["sample_id"]), row["slot"]): row.get("value")
        for row in baseline_subset
    }
    for row in answers:
        key = (row["concept_id"], row["route"], int(row["sample_id"]), row["slot"])
        changed += baseline_by_key.get(key) != row.get("value")

    route_slot_rows = []
    for route in ["I2T", "T2I"]:
        for slot in SLOTS:
            subset = [row for row in answers if row["route"] == route and row["slot"] == slot]
            base_subset = [row for row in baseline_subset if row["route"] == route and row["slot"] == slot]
            option_entropy = [row for row in entropy_rows if row["route"] == route and row["slot"] == slot]
            option_error = [row for row in error_rows if row["route"] == route and row["slot"] == slot]
            base_entropy = [row for row in baseline_entropy if row["route"] == route and row["slot"] == slot]
            base_error = [row for row in baseline_error if row["route"] == route and row["slot"] == slot]
            route_slot_rows.append(
                {
                    "route": route,
                    "slot": slot,
                    "answers": len(subset),
                    "value_distribution": json.dumps(value_distribution(subset), ensure_ascii=False),
                    "baseline_value_distribution_subset": json.dumps(value_distribution(base_subset), ensure_ascii=False),
                    "unknown_rate": (unknown_by_route_slot[(route, slot)] / answers_by_route_slot[(route, slot)])
                    if answers_by_route_slot[(route, slot)]
                    else 0.0,
                    "entropy_mean": mean(option_entropy, "entropy"),
                    "baseline_entropy_mean_full": mean(base_entropy, "entropy"),
                    "error_mean": mean(option_error, "error_rate"),
                    "baseline_error_mean_full": mean(base_error, "error_rate"),
                }
            )
    write_csv(out_dir / "option_order_route_slot_summary.csv", route_slot_rows)

    audit = {
        "status": "PASS" if states and answers else "FAIL",
        "option_order_seed": args.option_order_seed,
        "state_rows": len(states),
        "slot_answer_rows": len(answers),
        "baseline_state_overlap": baseline_state_count,
        "changed_slot_values_vs_baseline_subset": changed,
        "changed_slot_value_rate_vs_baseline_subset": changed / len(answers) if answers else None,
        "schema_ok": all(row.get("slot_schema") == SLOTS for row in states),
        "vocab_ok": all(row.get("value") in VOCAB[row.get("slot")] for row in answers),
        "output_dir": str(out_dir),
        "baseline_dir": str(baseline_dir),
        "caveat": "Smoke runs with max-jobs are interface checks, not full robustness evidence.",
    }
    (out_dir / "option_order_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Option-Order Robustness Audit",
        "",
        f"Status: **{audit['status']}**",
        "",
        "## Input Files",
        "",
        f"- `{baseline_dir / 'strict_slot_answers.jsonl'}`",
        f"- `{baseline_dir / 'strict_route_entropy.csv'}`",
        f"- `{baseline_dir / 'strict_route_error.csv'}`",
        f"- `{out_dir / 'workers/option_order_states_worker*.jsonl'}`",
        f"- `{out_dir / 'workers/option_order_slot_answers_worker*.jsonl'}`",
        "",
        "## Commands",
        "",
        "```bash",
        "./.venv-transformers/bin/python scripts/run_semantic_entropy_option_order_robustness.py \\",
        "  --mode aggregate \\",
        f"  --pilot-dir {args.pilot_dir} \\",
        f"  --baseline-strict-dir {args.baseline_strict_dir} \\",
        f"  --out-dir {args.out_dir} \\",
        f"  --samples-per-concept {args.samples_per_concept} \\",
        f"  --option-order-seed {args.option_order_seed}",
        "./.venv-transformers/bin/python scripts/run_semantic_entropy_option_order_robustness.py \\",
        "  --mode audit \\",
        f"  --out-dir {args.out_dir}",
        "```",
        "",
        "## Output Files",
        "",
        f"- `{out_dir / 'option_order_semantic_states.jsonl'}`",
        f"- `{out_dir / 'option_order_slot_answers.jsonl'}`",
        f"- `{out_dir / 'option_order_route_entropy.csv'}`",
        f"- `{out_dir / 'option_order_route_error.csv'}`",
        f"- `{out_dir / 'option_order_route_slot_summary.csv'}`",
        f"- `{out_dir / 'option_order_audit.json'}`",
        f"- `{out_dir / 'option_order_audit.md'}`",
        "",
        "## Scope",
        "",
        "- Same images, concepts, slots, extractor, parser, and normalization as strict compare.",
        f"- Allowed value order shuffled with seed `{args.option_order_seed}`.",
        f"- Output directory: `{out_dir}`.",
        "- Baseline strict_compare outputs were read for comparison only.",
        "",
        "## Counts",
        "",
        f"- State rows: `{len(states)}`",
        f"- Slot answer rows: `{len(answers)}`",
        f"- Baseline state overlap: `{baseline_state_count}`",
        f"- Schema OK: `{audit['schema_ok']}`",
        f"- Vocabulary OK: `{audit['vocab_ok']}`",
        f"- Changed slot values vs baseline subset: `{changed}` / `{len(answers)}`",
        "",
        "## Sample Counts",
        "",
        f"- State rows: `{len(states)}`",
        f"- Slot answer rows: `{len(answers)}`",
        f"- Baseline state overlap: `{baseline_state_count}`",
        f"- Route-slot summary rows: `{len(route_slot_rows)}`",
        "",
        "## Pass/Fail Checks",
        "",
        f"- Status PASS: `{audit['status'] == 'PASS'}`",
        f"- Schema OK: `{audit['schema_ok']}`",
        f"- Vocabulary OK: `{audit['vocab_ok']}`",
        "- Smoke-only caveat retained: `True`",
        "",
        "## Route-Slot Summary",
        "",
        "| Route | Slot | Answers | Unknown Rate | Entropy | Baseline Entropy | Error | Baseline Error | Values | Baseline Values |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in route_slot_rows:
        lines.append(
            f"| {row['route']} | {row['slot']} | {row['answers']} | {row['unknown_rate']:.4f} | "
            f"{float(row['entropy_mean'] or 0):.4f} | {float(row['baseline_entropy_mean_full'] or 0):.4f} | "
            f"{float(row['error_mean'] or 0):.4f} | {float(row['baseline_error_mean_full'] or 0):.4f} | "
            f"`{row['value_distribution']}` | `{row['baseline_value_distribution_subset']}` |"
        )
    lines.extend(
        [
            "",
            "## Claim Allowed After This Step",
            "",
            "A smoke run only proves the option-order robustness pipeline is runnable and produces comparable files.",
            "",
            "## Claim Still Not Allowed",
            "",
            "Do not claim option-order robustness until the full planned seeds are run after extractor validation.",
        ]
    )
    (out_dir / "option_order_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def audit(args: argparse.Namespace) -> None:
    audit_path = Path(args.out_dir) / "option_order_audit.json"
    data = json.loads(audit_path.read_text(encoding="utf-8"))
    print(json.dumps(data, indent=2, ensure_ascii=False))
    if data.get("status") != "PASS":
        raise SystemExit(1)


def parse_seed_list(text: str) -> list[int]:
    seeds = []
    for item in text.split(","):
        item = item.strip()
        if item:
            seeds.append(int(item))
    if not seeds:
        raise ValueError("at least one option-order seed is required")
    return seeds


def detect_gpu_indices() -> list[int]:
    nvidia_smi = os.environ.get("NVIDIA_SMI", "nvidia-smi")
    try:
        completed = subprocess.run(
            [nvidia_smi, "--query-gpu=index", "--format=csv,noheader"],
            text=True,
            capture_output=True,
            check=False,
        )
    except FileNotFoundError:
        completed = None
    if completed and completed.returncode == 0:
        indices = []
        for line in completed.stdout.splitlines():
            line = line.strip()
            if line:
                indices.append(int(line.split(",")[0].strip()))
        if indices:
            return sorted(indices)
    return [0, 1, 2, 3]


def pair_gpu_indices(indices: list[int]) -> list[str]:
    pairs = []
    for offset in range(0, len(indices), 2):
        group = indices[offset : offset + 2]
        pairs.append(",".join(str(index) for index in group))
    return pairs


def parse_gpu_pairs(text: str) -> list[str]:
    if text.strip().lower() == "auto":
        return pair_gpu_indices(detect_gpu_indices())
    pairs = [item.strip() for item in text.split(";") if item.strip()]
    if not pairs:
        raise ValueError("at least one GPU pair is required")
    return pairs


def write_launcher(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    launcher_dir = out_dir / "launchers"
    log_dir = out_dir / "logs"
    launcher_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    seeds = parse_seed_list(args.option_order_seeds)
    gpu_pairs = parse_gpu_pairs(args.gpu_pairs)
    seed_launchers = []
    for seed in seeds:
        seed_dir = out_dir / f"seed_{seed}"
        lines = [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            'cd "$(dirname "${BASH_SOURCE[0]}")/../../../.."',
            "",
            f'VALIDATION_REPORT="{args.validation_report}"',
            'if [ "${FORCE_ROBUSTNESS:-0}" != "1" ]; then',
            '  ./.venv-transformers/bin/python - <<PY',
            "import json",
            "from pathlib import Path",
            'path = Path("' + args.validation_report + '")',
            "if not path.exists():",
            "    raise SystemExit(f'missing manual validation report: {path}')",
            "data = json.loads(path.read_text(encoding='utf-8'))",
            "if data.get('status') != 'PASS':",
            "    raise SystemExit(f'manual annotation validation is not PASS: {data.get(\"status\")}')",
            "print('[INFO] manual annotation validation PASS')",
            "PY",
            "else",
            '  echo "[WARN] FORCE_ROBUSTNESS=1 bypassed manual validation guard"',
            "fi",
            "",
            'export HF_HOME="$PWD/.cache/huggingface"',
            "export TRANSFORMERS_OFFLINE=1",
            "export HF_HUB_OFFLINE=1",
            'export PYTHONPATH="$PWD"',
            "",
            f"OUT_DIR=\"{seed_dir}\"",
            'mkdir -p "$OUT_DIR/logs"',
            "",
        ]
        for worker_id, gpu_pair in enumerate(gpu_pairs):
            lines.extend(
                [
                    f"CUDA_VISIBLE_DEVICES={gpu_pair} ./.venv-transformers/bin/python scripts/run_semantic_entropy_option_order_robustness.py \\",
                    "  --mode worker \\",
                    f"  --pilot-dir {args.pilot_dir} \\",
                    f"  --baseline-strict-dir {args.baseline_strict_dir} \\",
                    '  --out-dir "$OUT_DIR" \\',
                    f"  --num-workers {len(gpu_pairs)} \\",
                    f"  --worker-id {worker_id} \\",
                    f"  --samples-per-concept {args.samples_per_concept} \\",
                    f"  --max-new-tokens {args.max_new_tokens} \\",
                    f"  --seed {args.seed} \\",
                    f"  --option-order-seed {seed} \\",
                    f"  --skip-existing > \"$OUT_DIR/logs/worker{worker_id}.out\" 2>&1 &",
                    f"pids+=(\"$!\")",
                    "",
                ]
            )
        lines.insert(lines.index(""), "pids=()")
        lines.extend(
            [
                'for pid in "${pids[@]}"; do',
                '  wait "$pid"',
                "done",
                "",
                "./.venv-transformers/bin/python scripts/run_semantic_entropy_option_order_robustness.py \\",
                "  --mode aggregate \\",
                f"  --pilot-dir {args.pilot_dir} \\",
                f"  --baseline-strict-dir {args.baseline_strict_dir} \\",
                '  --out-dir "$OUT_DIR" \\',
                f"  --samples-per-concept {args.samples_per_concept} \\",
                f"  --option-order-seed {seed}",
                "./.venv-transformers/bin/python scripts/run_semantic_entropy_option_order_robustness.py --mode audit --out-dir \"$OUT_DIR\"",
                f'echo "[INFO] seed {seed} finished: $OUT_DIR"',
            ]
        )
        path = launcher_dir / f"launch_option_order_seed_{seed}.sh"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        path.chmod(0o755)
        seed_launchers.append(path)

    full_lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        'cd "$(dirname "${BASH_SOURCE[0]}")"',
        "",
    ]
    for path in seed_launchers:
        full_lines.append(f'./{path.name}')
    full_launcher = launcher_dir / "launch_all_option_order_seeds.sh"
    full_launcher.write_text("\n".join(full_lines) + "\n", encoding="utf-8")
    full_launcher.chmod(0o755)

    runbook = [
        "# Option-Order Robustness Full Launcher",
        "",
        "Status: READY_BUT_GUARDED",
        "",
        "This launcher is prepared for the full option-order robustness run. It does not bypass the manual annotation gate.",
        "",
        "## Guard",
        "",
        f"- Requires `{args.validation_report}` with `status == PASS`.",
        "- Set `FORCE_ROBUSTNESS=1` only for an explicit emergency bypass.",
        "",
        "## Seeds",
        "",
        *[f"- `{seed}` -> `{out_dir / f'seed_{seed}'}`" for seed in seeds],
        "",
        "## GPU Layout",
        "",
        f"- GPU pairs: `{';'.join(gpu_pairs)}`",
        f"- Workers per seed: `{len(gpu_pairs)}`",
        "",
        "## Commands",
        "",
        "```bash",
        f"bash {full_launcher}",
        "```",
        "",
        "## Claim Boundary",
        "",
        "Running these launchers can provide option-order robustness evidence only after extractor validation is complete and the per-seed audits PASS.",
    ]
    runbook_path = launcher_dir / "option_order_full_runbook.md"
    runbook_path.write_text("\n".join(runbook) + "\n", encoding="utf-8")
    manifest = {
        "status": "READY_BUT_GUARDED",
        "out_dir": str(out_dir),
        "launcher_dir": str(launcher_dir),
        "full_launcher": str(full_launcher),
        "seed_launchers": [str(path) for path in seed_launchers],
        "seeds": seeds,
        "gpu_pairs": gpu_pairs,
        "validation_report": args.validation_report,
        "samples_per_concept": args.samples_per_concept,
        "claim_boundary": "not robustness evidence until run after manual validation PASS",
    }
    (launcher_dir / "option_order_full_launcher_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


def main() -> None:
    args = parse_args()
    if args.mode == "worker":
        run_worker(args)
    elif args.mode == "aggregate":
        aggregate(args)
    elif args.mode == "audit":
        audit(args)
    else:
        write_launcher(args)


if __name__ == "__main__":
    main()
