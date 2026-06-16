#!/usr/bin/env python3
"""Semantic-uncertainty-style QA comparison for Emu3.5 I2T/T2I.

The jlko/semantic_uncertainty protocol samples multiple short answers for a QA
item, maps answers into semantic equivalence classes, and computes entropy over
those semantic classes.  For multimodal I2T/T2I, this script keeps the same
brief Context/Question/Answer surface form, but uses a shared finite visual
slot vocabulary instead of an NLI entailment model.  This keeps the two routes
comparable in one semantic state space.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import torch
from transformers import GenerationConfig
from transformers.generation import LogitsProcessor, LogitsProcessorList, StoppingCriteria, StoppingCriteriaList

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

from src.utils.input_utils import build_image  # noqa: E402
from src.utils.logits_processor import BOI, BOV, EOI, EOF, EOL, IMG  # noqa: E402
from src.utils.model_utils import build_emu3p5  # noqa: E402


BRIEF_PROMPT = "Answer the following question as briefly as possible.\n"
SLOTS = ["object_1", "color_1", "object_2", "color_2", "relation", "background"]
VOCAB = {
    "object_1": ["cube", "sphere", "cone", "unknown"],
    "object_2": ["cube", "sphere", "cone", "unknown"],
    "color_1": ["red", "blue", "green", "yellow", "unknown"],
    "color_2": ["red", "blue", "green", "yellow", "unknown"],
    "relation": [
        "object_1_left_of_object_2",
        "object_1_right_of_object_2",
        "object_1_above_object_2",
        "object_1_below_object_2",
        "unknown",
    ],
    "background": ["white", "other", "unknown"],
}
SPECIAL_TOKENS = dict(
    BOS="<|extra_203|>",
    EOS="<|extra_204|>",
    PAD="<|endoftext|>",
    EOL="<|extra_200|>",
    EOF="<|extra_201|>",
    TMS="<|extra_202|>",
    IMG="<|image token|>",
    BOI="<|image start|>",
    EOI="<|image end|>",
    BSS="<|extra_100|>",
    ESS="<|extra_101|>",
    BOG="<|extra_60|>",
    EOG="<|extra_61|>",
    BOC="<|extra_50|>",
    EOC="<|extra_51|>",
)


class TextOnlyLogitsProcessor(LogitsProcessor):
    def __init__(self, top_k: int = 1024, top_p: float = 0.95, temperature: float = 1.0):
        self.top_k = int(top_k)
        self.top_p = float(top_p)
        self.temperature = float(temperature)
        self.forbidden_ids = {BOI, EOI, IMG, EOL, EOF}

    def __call__(self, input_ids, scores):
        scores = scores.clone()
        scores[:, BOV:] = -math.inf
        for token_id in self.forbidden_ids:
            if 0 <= int(token_id) < scores.shape[-1]:
                scores[:, int(token_id)] = -math.inf
        if self.temperature != 1.0:
            scores = scores / self.temperature
        if 0 < self.top_k < scores.shape[-1]:
            threshold = torch.topk(scores, self.top_k, dim=-1).values[:, -1, None]
            scores = scores.masked_fill(scores < threshold, -math.inf)
        if self.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(scores, descending=True, dim=-1)
            cumulative = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_remove = cumulative > self.top_p
            sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
            sorted_remove[..., 0] = False
            remove = torch.zeros_like(sorted_remove).scatter(1, sorted_indices, sorted_remove)
            scores = scores.masked_fill(remove, -math.inf)
        return scores


class StopOnGeneratedTokenCriteria(StoppingCriteria):
    def __init__(self, prompt_len: int, stop_token_ids: list[int]):
        self.prompt_len = int(prompt_len)
        self.stop_token_ids = {int(token_id) for token_id in stop_token_ids}

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        generated = input_ids[0, self.prompt_len :]
        return bool(generated.numel() and int(generated[-1].item()) in self.stop_token_ids)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("worker", "aggregate", "audit"), required=True)
    parser.add_argument("--pilot-dir", default="outputs/semantic_entropy_umm/pilot")
    parser.add_argument("--qa-dir", default="outputs/semantic_entropy_umm/semantic_uncertainty_qa")
    parser.add_argument("--model-path", default="model/Emu3.5")
    parser.add_argument("--vq-path", default="model/Emu3.5-VisionTokenizer")
    parser.add_argument("--tokenizer-path", default="./src/tokenizer_emu3_ibq")
    parser.add_argument("--vq-type", default="ibq")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--samples-per-concept", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--answer-temperature", type=float, default=1.0)
    parser.add_argument("--answer-top-k", type=int, default=1024)
    parser.add_argument("--answer-top-p", type=float, default=0.95)
    parser.add_argument("--max-jobs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20270606)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def cfg_namespace(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        vq_path=args.vq_path,
        vq_type=args.vq_type,
        hf_device="auto",
        vq_device="cuda:0",
        special_tokens=SPECIAL_TOKENS,
        special_token_ids={},
        image_area=65536,
    )


def load_runtime(args: argparse.Namespace):
    cfg = cfg_namespace(args)
    model, tokenizer, vq_model = build_emu3p5(
        cfg.model_path,
        cfg.tokenizer_path,
        cfg.vq_path,
        vq_type=cfg.vq_type,
        model_device=cfg.hf_device,
        vq_device=cfg.vq_device,
        attn_implementation="sdpa",
    )
    cfg.special_token_ids = {k: tokenizer.encode(v)[0] for k, v in cfg.special_tokens.items()}
    return cfg, model, tokenizer, vq_model


def semantic_uncertainty_question(slot: str) -> str:
    if slot == "object_1":
        return "What shape is object_1, the first object named in the context?"
    if slot == "color_1":
        return "What color is object_1, the first object named in the context?"
    if slot == "object_2":
        return "What shape is object_2, the second object named in the context?"
    if slot == "color_2":
        return "What color is object_2, the second object named in the context?"
    if slot == "relation":
        return "What is the spatial relation from object_1 to object_2?"
    if slot == "background":
        return "What is the image background color?"
    raise ValueError(f"unknown slot {slot}")


def encode_image_qa(cfg, tokenizer, vq_model, image_path: Path, context: str, question: str, model_device):
    image = Image.open(image_path).convert("RGB")
    image_str = build_image(image, cfg, tokenizer, vq_model)
    qa_text = f"{BRIEF_PROMPT}Context: {context}\nQuestion: {question}\n"
    prompt = (
        "<|extra_203|>You are a concise visual question answering model. "
        "USER: <|IMAGE|>{qa_text}ASSISTANT: <|extra_100|>Answer:"
    ).format(qa_text=qa_text).replace("<|IMAGE|>", image_str)
    input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False).to(model_device)
    if input_ids[0, 0] != cfg.special_token_ids["BOS"]:
        bos = torch.tensor([[cfg.special_token_ids["BOS"]]], device=input_ids.device, dtype=input_ids.dtype)
        input_ids = torch.cat([bos, input_ids], dim=1)
    return input_ids, qa_text


@torch.no_grad()
def generate_answer(cfg, model, tokenizer, input_ids, args: argparse.Namespace) -> str:
    input_len = input_ids.shape[1]
    config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        num_beams=1,
        use_cache=True,
        pad_token_id=cfg.special_token_ids["PAD"],
        eos_token_id=cfg.special_token_ids["EOS"],
    )
    stop_ids = [cfg.special_token_ids["EOS"], cfg.special_token_ids["ESS"]]
    processor = TextOnlyLogitsProcessor(
        top_k=args.answer_top_k,
        top_p=args.answer_top_p,
        temperature=args.answer_temperature,
    )
    outputs = model.generate(
        input_ids,
        generation_config=config,
        logits_processor=LogitsProcessorList([processor]),
        stopping_criteria=StoppingCriteriaList([StopOnGeneratedTokenCriteria(input_len, stop_ids)]),
    )
    generated = outputs[:, input_len:][0].detach().cpu().tolist()
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def normalize_answer(slot: str, raw: str) -> str:
    text = raw.lower().strip()
    text = re.sub(r"[^a-z0-9_ ]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    if slot.startswith("object"):
        synonyms = {
            "cube": ["cube", "box", "block", "square"],
            "sphere": ["sphere", "circle", "ball", "round"],
            "cone": ["cone", "triangle", "pyramid"],
        }
        for label, words in synonyms.items():
            if any(re.search(rf"\b{re.escape(word)}\b", text) for word in words):
                return label
    elif slot.startswith("color"):
        for color in ["red", "blue", "green", "yellow"]:
            if re.search(rf"\b{color}\b", text):
                return color
    elif slot == "relation":
        compact = text.replace(" ", "_")
        for label in VOCAB["relation"]:
            if label != "unknown" and label in compact:
                return label
        if "left" in text:
            return "object_1_left_of_object_2"
        if "right" in text:
            return "object_1_right_of_object_2"
        if "above" in text or "over" in text or "top" in text:
            return "object_1_above_object_2"
        if "below" in text or "under" in text or "beneath" in text or "bottom" in text:
            return "object_1_below_object_2"
    elif slot == "background":
        if "white" in text:
            return "white"
        if "other" in text or "black" in text or "gray" in text or "grey" in text:
            return "other"
    return "unknown"


def build_jobs(args: argparse.Namespace) -> list[dict]:
    pilot_dir = Path(args.pilot_dir)
    concepts = read_jsonl(pilot_dir / "concepts.jsonl")
    t2i_rows = read_jsonl(pilot_dir / "t2i_samples.jsonl")
    t2i_map = {(r["concept_id"], int(r["sample_id"])): r for r in t2i_rows}
    jobs = []
    for concept in concepts:
        concept_id = concept["concept_id"]
        ref_image = pilot_dir / "reference_images" / f"{concept_id}.png"
        for sample_id in range(args.samples_per_concept):
            jobs.append({"concept": concept, "route": "I2T", "sample_id": sample_id, "image_path": str(ref_image)})
            t2i = t2i_map.get((concept_id, sample_id))
            if t2i is not None and t2i.get("image_path"):
                jobs.append({"concept": concept, "route": "T2I", "sample_id": sample_id, "image_path": t2i["image_path"]})
    return jobs


def existing_state_keys(path: Path) -> set[tuple[str, str, int]]:
    return {
        (row["concept_id"], row["route"], int(row["sample_id"]))
        for row in read_jsonl(path)
    }


def run_worker(args: argparse.Namespace) -> None:
    qa_dir = Path(args.qa_dir)
    worker_state_path = qa_dir / "workers" / f"qa_semantic_states_worker{args.worker_id}.jsonl"
    worker_answer_path = qa_dir / "workers" / f"qa_slot_answers_worker{args.worker_id}.jsonl"
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
        for slot in SLOTS:
            torch.cuda.empty_cache()
            question = semantic_uncertainty_question(slot)
            input_ids, qa_text = encode_image_qa(cfg, tokenizer, vq_model, image_path, concept["prompt"], question, model.device)
            raw = generate_answer(cfg, model, tokenizer, input_ids, args)
            value = normalize_answer(slot, raw)
            slots[slot] = value
            evidence[slot] = raw
            append_jsonl(
                worker_answer_path,
                {
                    "concept_id": concept_id,
                    "route": route,
                    "sample_id": sample_id,
                    "slot": slot,
                    "context": concept["prompt"],
                    "question": question,
                    "prompt_text": qa_text,
                    "raw_output": raw,
                    "value": value,
                    "allowed_values": VOCAB[slot],
                    "image_path": str(image_path),
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
                "parser": "semantic_uncertainty_qa_finite_slot_v1",
                "slot_schema": SLOTS,
            },
        )
    print(f"[INFO] qa worker {args.worker_id} finished {len(jobs)} state jobs", flush=True)


def entropy(counter: Counter) -> float:
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    return -sum((count / total) * math.log(count / total) for count in counter.values() if count > 0)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        if not rows:
            return
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(args: argparse.Namespace) -> None:
    qa_dir = Path(args.qa_dir)
    states = []
    answers = []
    for path in sorted((qa_dir / "workers").glob("qa_semantic_states_worker*.jsonl")):
        states.extend(read_jsonl(path))
    for path in sorted((qa_dir / "workers").glob("qa_slot_answers_worker*.jsonl")):
        answers.extend(read_jsonl(path))
    with (qa_dir / "qa_semantic_states.jsonl").open("w", encoding="utf-8") as handle:
        for row in states:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (qa_dir / "qa_slot_answers.jsonl").open("w", encoding="utf-8") as handle:
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
                    "cluster_assignment_entropy": entropy(joint_counts),
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
                        "cluster_assignment_entropy": entropy(counts),
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
    write_csv(qa_dir / "qa_route_entropy.csv", entropy_rows)
    write_csv(qa_dir / "qa_route_error.csv", error_rows)
    write_audit(qa_dir, states, answers, entropy_rows, error_rows, args.samples_per_concept)
    print(f"[INFO] qa aggregate wrote {qa_dir}", flush=True)


def means_by_slot(rows: list[dict], value_field: str) -> dict:
    out = {}
    for slot in ["joint", *SLOTS] if value_field == "cluster_assignment_entropy" else SLOTS:
        out[slot] = {}
        for route in ["I2T", "T2I"]:
            vals = [float(row[value_field]) for row in rows if row["slot"] == slot and row["route"] == route]
            out[slot][route] = sum(vals) / len(vals) if vals else None
        if out[slot]["I2T"] is not None and out[slot]["T2I"] is not None:
            out[slot]["delta_t2i_minus_i2t"] = out[slot]["T2I"] - out[slot]["I2T"]
    return out


def write_audit(
    qa_dir: Path,
    states: list[dict],
    answers: list[dict],
    entropy_rows: list[dict],
    error_rows: list[dict],
    expected_samples_per_concept: int,
) -> None:
    concept_ids = sorted({row["concept_id"] for row in states})
    per_route = Counter(row["route"] for row in states)
    per_concept_route = Counter((row["concept_id"], row["route"]) for row in states)
    per_slot_route = Counter((row["route"], row["slot"]) for row in answers)
    schema_ok = all(row.get("slot_schema") == SLOTS for row in states)
    vocab_ok = all(row.get("value") in VOCAB[row.get("slot")] for row in answers)
    parity_bad = {
        f"{concept_id}:{route}": per_concept_route[(concept_id, route)]
        for concept_id in concept_ids
        for route in ["I2T", "T2I"]
        if per_concept_route[(concept_id, route)] != expected_samples_per_concept
    }
    slot_bad = {
        f"{route}:{slot}": per_slot_route[(route, slot)]
        for route in ["I2T", "T2I"]
        for slot in SLOTS
        if per_slot_route[(route, slot)] != len(concept_ids) * expected_samples_per_concept
    }
    entropy_means = means_by_slot(entropy_rows, "cluster_assignment_entropy")
    error_means = means_by_slot(error_rows, "error_rate")
    audit = {
        "claim": "Semantic-uncertainty-style QA answers are compared after finite-slot semantic clustering.",
        "semantic_uncertainty_repo": "third_party/semantic_uncertainty",
        "source_prompt_style": {
            "brief_prompt": BRIEF_PROMPT,
            "context_question_answer_format": "Context: {context}\\nQuestion: {question}\\nAnswer:",
            "source_files": [
                "third_party/semantic_uncertainty/semantic_uncertainty/uncertainty/utils/utils.py",
                "third_party/semantic_uncertainty/semantic_uncertainty/uncertainty/uncertainty_measures/semantic_entropy.py",
            ],
        },
        "adaptation": "NLI equivalence classes are replaced by a shared finite visual slot vocabulary so I2T and T2I share the same semantic state space.",
        "state_schema": SLOTS,
        "semantic_state_space": {slot: VOCAB[slot] for slot in SLOTS},
        "expected_samples_per_concept": expected_samples_per_concept,
        "sample_count_by_route": dict(per_route),
        "slot_answer_count_by_route_slot": {f"{k[0]}:{k[1]}": v for k, v in per_slot_route.items()},
        "schema_ok": schema_ok,
        "vocab_ok": vocab_ok,
        "per_concept_route_sample_parity_failures": parity_bad,
        "slot_answer_parity_failures": slot_bad,
        "entropy_means": entropy_means,
        "error_means": error_means,
        "verdict": "PASS" if schema_ok and vocab_ok and not parity_bad and not slot_bad else "FAIL",
        "important_caveat": "This is cluster-assignment semantic entropy in a controlled visual QA state space, not raw token entropy and not NLI over free-form long answers.",
    }
    (qa_dir / "qa_comparability_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# Semantic-Uncertainty-Style QA I2T/T2I Audit",
        "",
        f"Verdict: **{audit['verdict']}**",
        "",
        "## Protocol Source",
        "",
        "- Cloned source: `third_party/semantic_uncertainty`.",
        "- Prompt style: `Answer the following question as briefly as possible.` plus `Context`, `Question`, `Answer` fields.",
        "- Entropy style: cluster-assignment entropy over semantic classes.",
        "",
        "## Multimodal Adaptation",
        "",
        audit["adaptation"],
        "",
        "## Sample Counts",
        "",
        f"- State rows by route: `{dict(per_route)}`",
        f"- Total state rows: `{len(states)}`",
        f"- Total slot answer rows: `{len(answers)}`",
        f"- Concept count: `{len(concept_ids)}`",
        f"- Expected samples per concept per route: `{expected_samples_per_concept}`",
        f"- Schema OK: `{schema_ok}`",
        f"- Vocabulary OK: `{vocab_ok}`",
        f"- Per-concept route parity failures: `{len(parity_bad)}`",
        f"- Slot-answer parity failures: `{len(slot_bad)}`",
        "",
        "## Mean Cluster-Assignment Entropy",
        "",
        "| Slot | I2T | T2I | Delta T2I-I2T |",
        "|---|---:|---:|---:|",
    ]
    for slot, row in entropy_means.items():
        lines.append(f"| {slot} | {row['I2T']:.4f} | {row['T2I']:.4f} | {row['delta_t2i_minus_i2t']:.4f} |")
    lines.extend(["", "## Mean Error", "", "| Slot | I2T | T2I | Delta T2I-I2T |", "|---|---:|---:|---:|"])
    for slot, row in error_means.items():
        lines.append(f"| {slot} | {row['I2T']:.4f} | {row['T2I']:.4f} | {row['delta_t2i_minus_i2t']:.4f} |")
    lines.extend(["", "## Caveat", "", audit["important_caveat"]])
    (qa_dir / "qa_comparability_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def audit(args: argparse.Namespace) -> None:
    qa_dir = Path(args.qa_dir)
    data = json.loads((qa_dir / "qa_comparability_audit.json").read_text(encoding="utf-8"))
    print(json.dumps(data, indent=2, ensure_ascii=False))
    if data["verdict"] != "PASS":
        raise SystemExit(1)


def main() -> None:
    args = parse_args()
    if args.mode == "worker":
        run_worker(args)
    elif args.mode == "aggregate":
        aggregate(args)
    elif args.mode == "audit":
        audit(args)


if __name__ == "__main__":
    main()
