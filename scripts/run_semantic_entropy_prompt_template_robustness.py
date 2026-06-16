#!/usr/bin/env python3
"""Prompt-template robustness for strict semantic entropy extraction.

This script reuses strict_compare images and normalization, but changes the
forced-choice slot-question template. It writes to a separate output directory
and never overwrites the baseline strict_compare results.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
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


TEMPLATE_IDS = ["template_v2_short_direct", "template_v3_question_first"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("worker", "aggregate", "audit", "write-launcher"), required=True)
    parser.add_argument("--pilot-dir", default="outputs/semantic_entropy_umm/pilot")
    parser.add_argument("--baseline-strict-dir", default="outputs/semantic_entropy_umm/strict_compare")
    parser.add_argument("--out-dir", default="outputs/semantic_entropy_umm/robustness_prompt_template")
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
    parser.add_argument("--template-id", choices=TEMPLATE_IDS, default="template_v2_short_direct")
    parser.add_argument("--template-ids", default="template_v2_short_direct,template_v3_question_first")
    parser.add_argument("--gpu-pairs", default="auto")
    parser.add_argument("--validation-report", default="outputs/semantic_entropy_umm/extractor_validation/manual_annotation_validation_report.json")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def slot_description(slot: str, concept: dict) -> str:
    target = concept["target_semantics"]
    if slot == "object_1":
        return "shape of object_1, the first object in the target description"
    if slot == "color_1":
        return f"color of object_1, whose target shape is {target['object_1']}"
    if slot == "object_2":
        return "shape of object_2, the second object in the target description"
    if slot == "color_2":
        return f"color of object_2, whose target shape is {target['object_2']}"
    if slot == "relation":
        return "spatial relation from object_1 to object_2"
    return "background color category"


def forced_choice_question(slot: str, concept: dict, template_id: str) -> str:
    prompt = concept["prompt"]
    choices = ", ".join(VOCAB[slot])
    desc = slot_description(slot, concept)
    if template_id == "template_v2_short_direct":
        return (
            f"Target: {prompt}\n"
            f"Label the {desc}.\n"
            f"Allowed labels: {choices}.\n"
            "Return one allowed label only."
        )
    if template_id == "template_v3_question_first":
        return (
            f"What is the {desc}?\n"
            f"Options: {choices}.\n"
            f"Use the target object binding from this description: {prompt}.\n"
            "Answer with exactly one option."
        )
    raise ValueError(f"unknown template id: {template_id}")


def run_worker(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    worker_state_path = out_dir / "workers" / f"prompt_template_states_worker{args.worker_id}.jsonl"
    worker_answer_path = out_dir / "workers" / f"prompt_template_slot_answers_worker{args.worker_id}.jsonl"
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
        questions = {}
        for slot in SLOTS:
            torch.cuda.empty_cache()
            question = forced_choice_question(slot, concept, args.template_id)
            input_ids = encode_image_question(cfg, tokenizer, vq_model, image_path, question, model.device)
            raw = generate_answer(cfg, model, tokenizer, input_ids, args.max_new_tokens)
            value = normalize_answer(slot, raw)
            slots[slot] = value
            evidence[slot] = raw
            questions[slot] = question
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
                    "allowed_values": VOCAB[slot],
                    "image_path": str(image_path),
                    "template_id": args.template_id,
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
                "parser": f"strict_forced_choice_vqa_{args.template_id}",
                "slot_schema": SLOTS,
                "template_id": args.template_id,
                "questions": questions,
            },
        )
    print(f"[INFO] prompt-template worker {args.worker_id} finished {len(jobs)} state jobs", flush=True)


def aggregate(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    states = []
    answers = []
    for path in sorted((out_dir / "workers").glob("prompt_template_states_worker*.jsonl")):
        states.extend(read_jsonl(path))
    for path in sorted((out_dir / "workers").glob("prompt_template_slot_answers_worker*.jsonl")):
        answers.extend(read_jsonl(path))
    with (out_dir / "prompt_template_semantic_states.jsonl").open("w", encoding="utf-8") as handle:
        for row in states:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (out_dir / "prompt_template_slot_answers.jsonl").open("w", encoding="utf-8") as handle:
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
    write_csv(out_dir / "prompt_template_route_entropy.csv", entropy_rows)
    write_csv(out_dir / "prompt_template_route_error.csv", error_rows)
    write_prompt_template_audit(out_dir, Path(args.baseline_strict_dir), states, answers, entropy_rows, error_rows, args)
    print(f"[INFO] prompt-template aggregate wrote {out_dir}", flush=True)


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


def write_prompt_template_audit(
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
    baseline_state_overlap = len({(concept_id, route, sample_id) for concept_id, route, sample_id, _ in keyset})
    answers_by_route_slot = Counter((row["route"], row["slot"]) for row in answers)
    unknown_by_route_slot = Counter((row["route"], row["slot"]) for row in answers if row["value"] == "unknown")
    baseline_by_key = {
        (row["concept_id"], row["route"], int(row["sample_id"]), row["slot"]): row.get("value")
        for row in baseline_subset
    }
    changed = 0
    for row in answers:
        key = (row["concept_id"], row["route"], int(row["sample_id"]), row["slot"])
        changed += baseline_by_key.get(key) != row.get("value")

    route_slot_rows = []
    for route in ["I2T", "T2I"]:
        for slot in SLOTS:
            subset = [row for row in answers if row["route"] == route and row["slot"] == slot]
            base_subset = [row for row in baseline_subset if row["route"] == route and row["slot"] == slot]
            prompt_entropy = [row for row in entropy_rows if row["route"] == route and row["slot"] == slot]
            prompt_error = [row for row in error_rows if row["route"] == route and row["slot"] == slot]
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
                    "entropy_mean": mean(prompt_entropy, "entropy"),
                    "baseline_entropy_mean_full": mean(base_entropy, "entropy"),
                    "error_mean": mean(prompt_error, "error_rate"),
                    "baseline_error_mean_full": mean(base_error, "error_rate"),
                }
            )
    write_csv(out_dir / "prompt_template_route_slot_summary.csv", route_slot_rows)
    audit = {
        "status": "PASS" if states and answers else "FAIL",
        "template_id": args.template_id,
        "state_rows": len(states),
        "slot_answer_rows": len(answers),
        "baseline_state_overlap": baseline_state_overlap,
        "changed_slot_values_vs_baseline_subset": changed,
        "changed_slot_value_rate_vs_baseline_subset": changed / len(answers) if answers else None,
        "schema_ok": all(row.get("slot_schema") == SLOTS for row in states),
        "vocab_ok": all(row.get("value") in VOCAB[row.get("slot")] for row in answers),
        "output_dir": str(out_dir),
        "baseline_dir": str(baseline_dir),
        "caveat": "Smoke runs with max-jobs are interface checks, not full robustness evidence.",
    }
    (out_dir / "prompt_template_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Prompt-Template Robustness Audit",
        "",
        f"Status: **{audit['status']}**",
        "",
        "## Input Files",
        "",
        f"- `{baseline_dir / 'strict_slot_answers.jsonl'}`",
        f"- `{baseline_dir / 'strict_route_entropy.csv'}`",
        f"- `{baseline_dir / 'strict_route_error.csv'}`",
        f"- `{out_dir / 'workers/prompt_template_states_worker*.jsonl'}`",
        f"- `{out_dir / 'workers/prompt_template_slot_answers_worker*.jsonl'}`",
        "",
        "## Commands",
        "",
        "```bash",
        "./.venv-transformers/bin/python scripts/run_semantic_entropy_prompt_template_robustness.py \\",
        "  --mode aggregate \\",
        f"  --pilot-dir {args.pilot_dir} \\",
        f"  --baseline-strict-dir {args.baseline_strict_dir} \\",
        f"  --out-dir {args.out_dir} \\",
        f"  --samples-per-concept {args.samples_per_concept} \\",
        f"  --template-id {args.template_id}",
        "./.venv-transformers/bin/python scripts/run_semantic_entropy_prompt_template_robustness.py \\",
        "  --mode audit \\",
        f"  --out-dir {args.out_dir}",
        "```",
        "",
        "## Output Files",
        "",
        f"- `{out_dir / 'prompt_template_semantic_states.jsonl'}`",
        f"- `{out_dir / 'prompt_template_slot_answers.jsonl'}`",
        f"- `{out_dir / 'prompt_template_route_entropy.csv'}`",
        f"- `{out_dir / 'prompt_template_route_error.csv'}`",
        f"- `{out_dir / 'prompt_template_route_slot_summary.csv'}`",
        f"- `{out_dir / 'prompt_template_audit.json'}`",
        f"- `{out_dir / 'prompt_template_audit.md'}`",
        "",
        "## Scope",
        "",
        "- Same images, concepts, slots, extractor, parser, and normalization as strict compare.",
        f"- Prompt template: `{args.template_id}`.",
        f"- Output directory: `{out_dir}`.",
        "- Baseline strict_compare outputs were read for comparison only.",
        "",
        "## Counts",
        "",
        f"- State rows: `{len(states)}`",
        f"- Slot answer rows: `{len(answers)}`",
        f"- Baseline state overlap: `{baseline_state_overlap}`",
        f"- Schema OK: `{audit['schema_ok']}`",
        f"- Vocabulary OK: `{audit['vocab_ok']}`",
        f"- Changed slot values vs baseline subset: `{changed}` / `{len(answers)}`",
        "",
        "## Sample Counts",
        "",
        f"- State rows: `{len(states)}`",
        f"- Slot answer rows: `{len(answers)}`",
        f"- Baseline state overlap: `{baseline_state_overlap}`",
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
            "A smoke run only proves the prompt-template robustness pipeline is runnable and produces comparable files.",
            "",
            "## Claim Still Not Allowed",
            "",
            "Do not claim prompt-template robustness until the full planned templates are run after extractor validation.",
        ]
    )
    (out_dir / "prompt_template_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def audit(args: argparse.Namespace) -> None:
    audit_path = Path(args.out_dir) / "prompt_template_audit.json"
    data = json.loads(audit_path.read_text(encoding="utf-8"))
    print(json.dumps(data, indent=2, ensure_ascii=False))
    if data.get("status") != "PASS":
        raise SystemExit(1)


def parse_template_ids(text: str) -> list[str]:
    values = []
    for item in text.split(","):
        item = item.strip()
        if item:
            if item not in TEMPLATE_IDS:
                raise ValueError(f"unknown template id: {item}")
            values.append(item)
    if not values:
        raise ValueError("at least one template id is required")
    return values


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
    launcher_dir.mkdir(parents=True, exist_ok=True)
    templates = parse_template_ids(args.template_ids)
    gpu_pairs = parse_gpu_pairs(args.gpu_pairs)
    template_launchers = []
    for template_id in templates:
        template_dir = out_dir / template_id
        lines = [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            'cd "$(dirname "${BASH_SOURCE[0]}")/../../../.."',
            "pids=()",
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
            f"OUT_DIR=\"{template_dir}\"",
            'mkdir -p "$OUT_DIR/logs"',
            "",
        ]
        for worker_id, gpu_pair in enumerate(gpu_pairs):
            lines.extend(
                [
                    f"CUDA_VISIBLE_DEVICES={gpu_pair} ./.venv-transformers/bin/python scripts/run_semantic_entropy_prompt_template_robustness.py \\",
                    "  --mode worker \\",
                    f"  --pilot-dir {args.pilot_dir} \\",
                    f"  --baseline-strict-dir {args.baseline_strict_dir} \\",
                    '  --out-dir "$OUT_DIR" \\',
                    f"  --num-workers {len(gpu_pairs)} \\",
                    f"  --worker-id {worker_id} \\",
                    f"  --samples-per-concept {args.samples_per_concept} \\",
                    f"  --max-new-tokens {args.max_new_tokens} \\",
                    f"  --seed {args.seed} \\",
                    f"  --template-id {template_id} \\",
                    f"  --skip-existing > \"$OUT_DIR/logs/worker{worker_id}.out\" 2>&1 &",
                    f"pids+=(\"$!\")",
                    "",
                ]
            )
        lines.extend(
            [
                'for pid in "${pids[@]}"; do',
                '  wait "$pid"',
                "done",
                "",
                "./.venv-transformers/bin/python scripts/run_semantic_entropy_prompt_template_robustness.py \\",
                "  --mode aggregate \\",
                f"  --pilot-dir {args.pilot_dir} \\",
                f"  --baseline-strict-dir {args.baseline_strict_dir} \\",
                '  --out-dir "$OUT_DIR" \\',
                f"  --samples-per-concept {args.samples_per_concept} \\",
                f"  --template-id {template_id}",
                "./.venv-transformers/bin/python scripts/run_semantic_entropy_prompt_template_robustness.py --mode audit --out-dir \"$OUT_DIR\"",
                f'echo "[INFO] template {template_id} finished: $OUT_DIR"',
            ]
        )
        path = launcher_dir / f"launch_prompt_template_{template_id}.sh"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        path.chmod(0o755)
        template_launchers.append(path)

    full_lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        'cd "$(dirname "${BASH_SOURCE[0]}")"',
        "",
    ]
    for path in template_launchers:
        full_lines.append(f'./{path.name}')
    full_launcher = launcher_dir / "launch_all_prompt_templates.sh"
    full_launcher.write_text("\n".join(full_lines) + "\n", encoding="utf-8")
    full_launcher.chmod(0o755)

    runbook = [
        "# Prompt-Template Robustness Full Launcher",
        "",
        "Status: READY_BUT_GUARDED",
        "",
        "This launcher is prepared for the full prompt-template robustness run. It does not bypass the manual annotation gate.",
        "",
        "## Guard",
        "",
        f"- Requires `{args.validation_report}` with `status == PASS`.",
        "- Set `FORCE_ROBUSTNESS=1` only for an explicit emergency bypass.",
        "",
        "## Templates",
        "",
        *[f"- `{template_id}` -> `{out_dir / template_id}`" for template_id in templates],
        "",
        "## GPU Layout",
        "",
        f"- GPU pairs: `{';'.join(gpu_pairs)}`",
        f"- Workers per template: `{len(gpu_pairs)}`",
        "",
        "## Commands",
        "",
        "```bash",
        f"bash {full_launcher}",
        "```",
        "",
        "## Claim Boundary",
        "",
        "Running these launchers can provide prompt-template robustness evidence only after extractor validation is complete and the per-template audits PASS.",
    ]
    runbook_path = launcher_dir / "prompt_template_full_runbook.md"
    runbook_path.write_text("\n".join(runbook) + "\n", encoding="utf-8")
    manifest = {
        "status": "READY_BUT_GUARDED",
        "out_dir": str(out_dir),
        "launcher_dir": str(launcher_dir),
        "full_launcher": str(full_launcher),
        "template_launchers": [str(path) for path in template_launchers],
        "templates": templates,
        "gpu_pairs": gpu_pairs,
        "validation_report": args.validation_report,
        "samples_per_concept": args.samples_per_concept,
        "claim_boundary": "not robustness evidence until run after manual validation PASS",
    }
    (launcher_dir / "prompt_template_full_launcher_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
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
