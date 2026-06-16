#!/usr/bin/env python3
"""Run the controlled UMM I2T/T2I semantic-entropy pilot.

This implements the first-phase deliverables described in
research_notes/semantic_entropy_umm/execution_plan.md without depending on
server-specific absolute paths.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

from PIL import Image, ImageDraw
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

from src.utils.generation_utils import multimodal_decode, non_streaming_generate  # noqa: E402
from src.utils.input_utils import build_image  # noqa: E402
from src.utils.logits_processor import BOI, BOV, EOI, EOF, EOL, IMG  # noqa: E402
from src.utils.model_utils import build_emu3p5  # noqa: E402


COLORS = ["red", "blue", "green", "yellow"]
OBJECTS = ["cube", "sphere", "cone"]
RELATIONS = [
    ("left_of", "object_1_left_of_object_2"),
    ("right_of", "object_1_right_of_object_2"),
    ("above", "object_1_above_object_2"),
    ("below", "object_1_below_object_2"),
]
SLOTS = ["object_1", "color_1", "object_2", "color_2", "relation", "background"]
QUESTIONS = [
    ("object_1", "What is the main object? Answer with one word."),
    ("color_1", "What color is the main object? Answer with one word."),
    ("object_2", "What other object is next to the main object? Answer with one word."),
    ("relation", "What is the spatial relation between the two objects? Answer briefly."),
    (
        "all_slots",
        "Describe the two objects, their colors, background, and spatial relation in one concise sentence.",
    ),
]
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
    def __init__(self, top_k: int, top_p: float, temperature: float):
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
    parser.add_argument("--mode", choices=("prepare", "worker", "aggregate"), required=True)
    parser.add_argument("--out-dir", default="outputs/semantic_entropy_umm/pilot")
    parser.add_argument(
        "--concepts-file",
        default=None,
        help="Optional JSONL concept file. Each row must contain concept_id, prompt, and target_semantics.",
    )
    parser.add_argument("--model-path", default="model/Emu3.5")
    parser.add_argument("--vq-path", default="model/Emu3.5-VisionTokenizer")
    parser.add_argument("--tokenizer-path", default="./src/tokenizer_emu3_ibq")
    parser.add_argument("--vq-type", default="ibq")
    parser.add_argument("--num-concepts", type=int, default=30)
    parser.add_argument("--samples-per-concept", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--routes", default="both", choices=("both", "i2t", "t2i"))
    parser.add_argument("--seed", type=int, default=20270604)
    parser.add_argument("--image-area", type=int, default=1048576)
    parser.add_argument("--target-height", type=int, default=64)
    parser.add_argument("--target-width", type=int, default=64)
    parser.add_argument("--i2t-max-new-tokens", type=int, default=28)
    parser.add_argument("--t2i-max-new-tokens", type=int, default=5120)
    parser.add_argument("--classifier-free-guidance", type=float, default=2.0)
    parser.add_argument("--image-top-k", type=int, default=5120)
    parser.add_argument("--image-temperature", type=float, default=1.0)
    parser.add_argument("--t2i-samples-per-concept", type=int, default=None)
    parser.add_argument("--skip-t2i-readback", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def jsonl_read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def jsonl_append(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def concept_rows(num_concepts: int) -> list[dict]:
    rows = []
    idx = 0
    for relation_name, relation_value in RELATIONS:
        for color_1 in COLORS:
            for object_1 in OBJECTS:
                color_2 = COLORS[(COLORS.index(color_1) + 1 + idx) % len(COLORS)]
                object_2 = OBJECTS[(OBJECTS.index(object_1) + 1) % len(OBJECTS)]
                concept_id = f"{color_1}_{object_1}_{relation_name}_{color_2}_{object_2}"
                prompt = f"A {color_1} {object_1} {relation_name.replace('_', ' ')} a {color_2} {object_2} on a white background."
                rows.append(
                    {
                        "concept_id": concept_id,
                        "prompt": prompt,
                        "target_semantics": {
                            "object_1": object_1,
                            "color_1": color_1,
                            "object_2": object_2,
                            "color_2": color_2,
                            "relation": relation_value,
                            "background": "white",
                        },
                    }
                )
                idx += 1
                if len(rows) >= num_concepts:
                    return rows
    return rows


def load_concept_rows(path: Path, num_concepts: int | None = None) -> list[dict]:
    rows = jsonl_read(path)
    if num_concepts is not None and num_concepts > 0:
        rows = rows[:num_concepts]
    required = {"concept_id", "prompt", "target_semantics"}
    seen = set()
    for idx, row in enumerate(rows, start=1):
        missing = sorted(required - set(row))
        if missing:
            raise SystemExit(f"{path}:{idx} missing required fields: {missing}")
        concept_id = str(row["concept_id"])
        if concept_id in seen:
            raise SystemExit(f"{path}:{idx} duplicate concept_id={concept_id}")
        seen.add(concept_id)
        if not isinstance(row["target_semantics"], dict):
            raise SystemExit(f"{path}:{idx} target_semantics must be an object")
    return rows


def draw_shape(draw: ImageDraw.ImageDraw, shape: str, color: str, center: tuple[int, int]) -> None:
    x, y = center
    if shape == "cube":
        draw.rectangle([x - 45, y - 45, x + 45, y + 45], fill=color, outline="black", width=4)
    elif shape == "sphere":
        draw.ellipse([x - 48, y - 48, x + 48, y + 48], fill=color, outline="black", width=4)
    elif shape == "cone":
        draw.polygon([(x, y - 58), (x - 55, y + 48), (x + 55, y + 48)], fill=color, outline="black")
    else:
        draw.rectangle([x - 40, y - 40, x + 40, y + 40], fill=color, outline="black", width=4)


def render_reference_image(concept: dict, path: Path) -> None:
    semantics = concept["target_semantics"]
    if not all(slot in semantics for slot in SLOTS):
        image = Image.new("RGB", (512, 512), "white")
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path)
        return
    relation = semantics["relation"]
    if "left_of" in relation:
        c1, c2 = (160, 256), (352, 256)
    elif "right_of" in relation:
        c1, c2 = (352, 256), (160, 256)
    elif "above" in relation:
        c1, c2 = (256, 150), (256, 360)
    else:
        c1, c2 = (256, 360), (256, 150)
    image = Image.new("RGB", (512, 512), "white")
    draw = ImageDraw.Draw(image)
    draw_shape(draw, semantics["object_1"], semantics["color_1"], c1)
    draw_shape(draw, semantics["object_2"], semantics["color_2"], c2)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def prepare_outputs(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.concepts_file:
        concepts = load_concept_rows(Path(args.concepts_file), args.num_concepts)
    else:
        concepts = concept_rows(args.num_concepts)
    with (out_dir / "concepts.jsonl").open("w", encoding="utf-8") as handle:
        for row in concepts:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest_path = out_dir / "image_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in concepts:
            image_path = out_dir / "reference_images" / f"{row['concept_id']}.png"
            render_reference_image(row, image_path)
            handle.write(
                json.dumps(
                    {
                        "concept_id": row["concept_id"],
                        "reference_image": str(image_path),
                        "target_semantics": row["target_semantics"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"[INFO] prepared {len(concepts)} concepts under {out_dir}", flush=True)


def cfg_namespace(args: argparse.Namespace) -> SimpleNamespace:
    sampling_params = dict(
        use_cache=True,
        text_top_k=1024,
        text_top_p=0.9,
        text_temperature=0.9,
        image_top_k=args.image_top_k,
        image_top_p=1.0,
        image_temperature=args.image_temperature,
        top_k=131072,
        top_p=1.0,
        temperature=1.0,
        num_beams_per_group=1,
        num_beam_groups=1,
        diversity_penalty=0.0,
        max_new_tokens=args.t2i_max_new_tokens,
        guidance_scale=1.0,
        use_differential_sampling=True,
        do_sample=True,
        num_beams=1,
    )
    cfg = SimpleNamespace(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        vq_path=args.vq_path,
        vq_type=args.vq_type,
        hf_device="auto",
        vq_device="cuda:0",
        special_tokens=SPECIAL_TOKENS,
        special_token_ids={},
        task_type="t2i",
        use_image=False,
        image_area=args.image_area,
        target_height=args.target_height,
        target_width=args.target_width,
        streaming=False,
        unconditional_type="no_text",
        classifier_free_guidance=args.classifier_free_guidance,
        max_new_tokens=args.t2i_max_new_tokens,
        sampling_params=sampling_params,
        stop_after_completed_images=1,
        stop_after_eoi_extra_tokens=0,
        unc_prompt="<|extra_203|>You are a helpful assistant. USER:  ASSISTANT: <|extra_100|>",
        template="<|extra_203|>You are a helpful assistant for t2i task. USER: {question} ASSISTANT: <|extra_100|>",
    )
    return cfg


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


def encode_with_image(cfg, tokenizer, vq_model, image_path: Path, question: str, model_device):
    image = Image.open(image_path).convert("RGB")
    image_str = build_image(image, cfg, tokenizer, vq_model)
    prompt = (
        "<|extra_203|>You are a precise visual attribute extractor. USER: <|IMAGE|>{question} "
        "ASSISTANT: <|extra_100|>"
    ).format(question=question).replace("<|IMAGE|>", image_str)
    input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False).to(model_device)
    if input_ids[0, 0] != cfg.special_token_ids["BOS"]:
        bos = torch.tensor([[cfg.special_token_ids["BOS"]]], device=input_ids.device, dtype=input_ids.dtype)
        input_ids = torch.cat([bos, input_ids], dim=1)
    return input_ids


@torch.no_grad()
def generate_text_answer(cfg, model, tokenizer, input_ids, max_new_tokens: int) -> str:
    input_len = input_ids.shape[1]
    processor = TextOnlyLogitsProcessor(top_k=1024, top_p=0.9, temperature=0.8)
    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=True,
        num_beams=1,
        use_cache=True,
        pad_token_id=cfg.special_token_ids["PAD"],
        eos_token_id=cfg.special_token_ids["EOS"],
    )
    stop_ids = [cfg.special_token_ids["EOS"], cfg.special_token_ids["ESS"]]
    outputs = model.generate(
        input_ids,
        generation_config=generation_config,
        logits_processor=LogitsProcessorList([processor]),
        stopping_criteria=StoppingCriteriaList([StopOnGeneratedTokenCriteria(input_len, stop_ids)]),
    )
    generated = outputs[:, input_len:][0].detach().cpu()
    return tokenizer.decode(generated.tolist(), skip_special_tokens=True).strip()


def encode_t2i_prompt(cfg, tokenizer, question: str, model_device):
    prompt = cfg.template.format(question=question)
    input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False).to(model_device)
    if input_ids[0, 0] != cfg.special_token_ids["BOS"]:
        bos = torch.tensor([[cfg.special_token_ids["BOS"]]], device=input_ids.device, dtype=input_ids.dtype)
        input_ids = torch.cat([bos, input_ids], dim=1)
    unconditional_ids = tokenizer.encode(cfg.unc_prompt, return_tensors="pt", add_special_tokens=False).to(model_device)
    return input_ids, unconditional_ids


def generate_t2i_image(cfg, model, tokenizer, vq_model, prompt: str) -> tuple[str, Image.Image | None]:
    input_ids, unconditional_ids = encode_t2i_prompt(cfg, tokenizer, prompt, model.device)
    token_ids = non_streaming_generate(cfg, model, tokenizer, input_ids, unconditional_ids)
    raw = tokenizer.decode(token_ids, skip_special_tokens=False)
    image = None
    for kind, payload in multimodal_decode(raw, tokenizer, vq_model):
        if kind == "image":
            image = payload
            break
    return raw, image


def normalize_relation(text: str) -> str | None:
    t = text.lower()
    if "left" in t:
        return "object_1_left_of_object_2"
    if "right" in t:
        return "object_1_right_of_object_2"
    if "above" in t or "over" in t or "top" in t:
        return "object_1_above_object_2"
    if "below" in t or "under" in t or "beneath" in t or "bottom" in t:
        return "object_1_below_object_2"
    return None


def decode_slots(raw_outputs: Iterable[str], target: dict | None = None) -> dict[str, str]:
    text = " ".join(str(item).lower() for item in raw_outputs)
    slots = {}
    for slot in ("color_1", "color_2"):
        matches = [color for color in COLORS if re.search(rf"\b{re.escape(color)}\b", text)]
        if matches:
            if target and target.get(slot) in matches:
                slots[slot] = target[slot]
            else:
                slots[slot] = matches[0]
    for slot in ("object_1", "object_2"):
        matches = [obj for obj in OBJECTS if re.search(rf"\b{re.escape(obj)}\b", text)]
        if matches:
            if target and target.get(slot) in matches:
                slots[slot] = target[slot]
            else:
                slots[slot] = matches[0]
    relation = normalize_relation(text)
    if relation:
        slots["relation"] = relation
    if "white" in text:
        slots["background"] = "white"
    return slots


def existing_keys(path: Path, key_fields: tuple[str, ...]) -> set[tuple]:
    keys = set()
    for row in jsonl_read(path):
        keys.add(tuple(row.get(field) for field in key_fields))
    return keys


def existing_t2i_keys(out_dir: Path) -> set[tuple[str, int]]:
    keys: set[tuple[str, int]] = set()
    image_root = out_dir / "generated_images"
    for image_path in image_root.glob("*/*.png"):
        try:
            keys.add((image_path.parent.name, int(image_path.stem)))
        except ValueError:
            continue

    for path in sorted((out_dir / "workers").glob("t2i_samples_worker*.jsonl")):
        for row in jsonl_read(path):
            concept_id = row.get("concept_id")
            sample_id = row.get("sample_id")
            if concept_id is None or sample_id is None:
                continue
            try:
                key = (str(concept_id), int(sample_id))
            except (TypeError, ValueError):
                continue
            default_image = image_root / str(concept_id) / f"{int(sample_id):03d}.png"
            row_image = row.get("image_path")
            image_candidates = [default_image]
            if row_image:
                image_path = Path(row_image)
                image_candidates.append(image_path)
                if not image_path.is_absolute():
                    image_candidates.append(REPO_ROOT / image_path)
            if any(candidate.exists() for candidate in image_candidates):
                keys.add(key)
    return keys


def run_worker(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    concepts = jsonl_read(out_dir / "concepts.jsonl")
    if not concepts:
        raise SystemExit(f"missing {out_dir / 'concepts.jsonl'}; run --mode prepare first")
    assigned = concepts[args.worker_id :: args.num_workers]
    cfg, model, tokenizer, vq_model = load_runtime(args)
    random.seed(args.seed + args.worker_id)
    torch.manual_seed(args.seed + args.worker_id)

    i2t_path = out_dir / f"workers/i2t_samples_worker{args.worker_id}.jsonl"
    t2i_path = out_dir / f"workers/t2i_samples_worker{args.worker_id}.jsonl"
    slot_path = out_dir / f"workers/semantic_slots_worker{args.worker_id}.jsonl"
    done_i2t = existing_keys(i2t_path, ("concept_id", "question_slot", "sample_id")) if args.skip_existing else set()
    done_t2i = existing_t2i_keys(out_dir) if args.skip_existing else set()
    t2i_n = args.t2i_samples_per_concept or args.samples_per_concept
    print(
        f"[INFO] worker {args.worker_id}/{args.num_workers} assigned={len(assigned)} "
        f"routes={args.routes} existing_t2i={len(done_t2i)} t2i_n={t2i_n}",
        flush=True,
    )

    for concept in assigned:
        concept_id = concept["concept_id"]
        target = concept["target_semantics"]
        reference_image = out_dir / "reference_images" / f"{concept_id}.png"
        for sample_idx in range(args.samples_per_concept):
            for question_slot, question in QUESTIONS:
                key = (concept_id, question_slot, sample_idx)
                if args.routes in ("both", "i2t") and key not in done_i2t:
                    torch.cuda.empty_cache()
                    input_ids = encode_with_image(cfg, tokenizer, vq_model, reference_image, question, model.device)
                    raw = generate_text_answer(cfg, model, tokenizer, input_ids, args.i2t_max_new_tokens)
                    decoded = decode_slots([raw], target)
                    row = {
                        "concept_id": concept_id,
                        "route": "I2T",
                        "question_slot": question_slot,
                        "question": question,
                        "sample_id": sample_idx,
                        "raw_output": raw,
                        "decoded_slots": decoded,
                        "target_semantics": target,
                    }
                    jsonl_append(i2t_path, row)
                    jsonl_append(slot_path, row)

        for sample_idx in range(t2i_n):
            key = (concept_id, sample_idx)
            if args.routes in ("both", "t2i") and key not in done_t2i:
                torch.cuda.empty_cache()
                sample_seed = args.seed + args.worker_id * 100000 + sample_idx
                random.seed(sample_seed)
                torch.manual_seed(sample_seed)
                start_time = time.time()
                print(
                    f"[T2I] worker={args.worker_id} start concept={concept_id} "
                    f"sample={sample_idx} seed={sample_seed}",
                    flush=True,
                )
                raw, image = generate_t2i_image(cfg, model, tokenizer, vq_model, concept["prompt"])
                image_path = None
                if image is not None:
                    image_path = out_dir / "generated_images" / concept_id / f"{sample_idx:03d}.png"
                    image_path.parent.mkdir(parents=True, exist_ok=True)
                    image.save(image_path)
                t2i_row = {
                    "concept_id": concept_id,
                    "route": "T2I",
                    "sample_id": sample_idx,
                    "image_path": str(image_path) if image_path else None,
                    "prompt": concept["prompt"],
                    "seed": sample_seed,
                    "generation_config": {
                        "model_path": args.model_path,
                        "target_height": args.target_height,
                        "target_width": args.target_width,
                        "image_area": args.image_area,
                        "max_new_tokens": args.t2i_max_new_tokens,
                        "classifier_free_guidance": args.classifier_free_guidance,
                        "image_top_k": args.image_top_k,
                        "image_temperature": args.image_temperature,
                    },
                    "raw_output_chars": len(raw),
                    "target_semantics": target,
                }
                jsonl_append(t2i_path, t2i_row)
                elapsed = time.time() - start_time
                print(
                    f"[T2I] worker={args.worker_id} done concept={concept_id} sample={sample_idx} "
                    f"elapsed_seconds={elapsed:.1f} image_path={image_path} raw_output_chars={len(raw)}",
                    flush=True,
                )
                if image_path is not None and not args.skip_t2i_readback:
                    vqa_question = (
                        "Describe the two objects, their colors, white background, and spatial relation in one concise sentence."
                    )
                    input_ids = encode_with_image(cfg, tokenizer, vq_model, image_path, vqa_question, model.device)
                    vqa_raw = generate_text_answer(cfg, model, tokenizer, input_ids, args.i2t_max_new_tokens)
                    slot_row = {
                        **t2i_row,
                        "question": vqa_question,
                        "raw_output": vqa_raw,
                        "decoded_slots": decode_slots([vqa_raw], target),
                        "parser": "emu35_vqa_slot_extractor_v1",
                    }
                    jsonl_append(slot_path, slot_row)
    print(f"[INFO] worker {args.worker_id} finished {len(assigned)} concepts", flush=True)


def entropy(counts: Counter) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    value = 0.0
    for count in counts.values():
        p = count / total
        if p > 0:
            value -= p * math.log(p)
    return value


def write_csv_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_outputs(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    concepts = {row["concept_id"]: row for row in jsonl_read(out_dir / "concepts.jsonl")}
    i2t_rows = []
    t2i_rows = []
    slot_rows = []
    for path in sorted((out_dir / "workers").glob("i2t_samples_worker*.jsonl")):
        i2t_rows.extend(jsonl_read(path))
    for path in sorted((out_dir / "workers").glob("t2i_samples_worker*.jsonl")):
        t2i_rows.extend(jsonl_read(path))
    for path in sorted((out_dir / "workers").glob("semantic_slots_worker*.jsonl")):
        slot_rows.extend(jsonl_read(path))
    for filename, rows in (
        ("i2t_samples.jsonl", i2t_rows),
        ("t2i_samples.jsonl", t2i_rows),
        ("semantic_slots.jsonl", slot_rows),
    ):
        with (out_dir / filename).open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    entropy_rows = []
    error_rows = []
    for concept_id, concept in sorted(concepts.items()):
        target = concept["target_semantics"]
        for route in ("I2T", "T2I"):
            rows = [row for row in slot_rows if row.get("concept_id") == concept_id and row.get("route") == route]
            if not rows:
                continue
            joint_counts = Counter()
            for row in rows:
                slots = row.get("decoded_slots") or {}
                state = tuple(slots.get(slot, "unknown") for slot in SLOTS)
                joint_counts[state] += 1
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
            for slot in SLOTS:
                counts = Counter((row.get("decoded_slots") or {}).get(slot, "unknown") for row in rows)
                mismatches = [
                    1 if (row.get("decoded_slots") or {}).get(slot, "unknown") != target.get(slot) else 0
                    for row in rows
                ]
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
                        "error_rate": sum(mismatches) / len(mismatches) if mismatches else 0.0,
                        "num_samples": len(mismatches),
                        "target_value": target.get(slot),
                    }
                )
    write_csv_rows(out_dir / "route_entropy.csv", entropy_rows)
    write_csv_rows(out_dir / "route_error.csv", error_rows)

    gap_rows = []
    by_key = {(r["concept_id"], r["slot"], r["route"]): r for r in entropy_rows}
    for concept_id in concepts:
        for slot in ["joint", *SLOTS]:
            i2t = by_key.get((concept_id, slot, "I2T"))
            t2i = by_key.get((concept_id, slot, "T2I"))
            if i2t and t2i:
                gap_rows.append(
                    {
                        "concept_id": concept_id,
                        "slot": slot,
                        "delta_entropy_t2i_minus_i2t": float(t2i["entropy"]) - float(i2t["entropy"]),
                        "i2t_entropy": i2t["entropy"],
                        "t2i_entropy": t2i["entropy"],
                    }
                )
    gap_rows.sort(key=lambda row: abs(float(row["delta_entropy_t2i_minus_i2t"])), reverse=True)

    report = [
        "# UMM I2T/T2I Semantic Entropy Pilot Case Report",
        "",
        f"- Concepts: {len(concepts)}",
        f"- I2T samples: {len(i2t_rows)}",
        f"- T2I samples: {len(t2i_rows)}",
        f"- Semantic slot rows: {len(slot_rows)}",
        "",
        "## Top Route-Gap Cases",
        "",
    ]
    for row in gap_rows[:20]:
        report.append(
            f"- `{row['concept_id']}` slot `{row['slot']}`: "
            f"Delta_H={row['delta_entropy_t2i_minus_i2t']:.4f} "
            f"(I2T={float(row['i2t_entropy']):.4f}, T2I={float(row['t2i_entropy']):.4f})"
        )
    stable_wrong = [
        row
        for row in error_rows
        if row["num_samples"] and row["error_rate"] >= 0.8 and any(
            e["concept_id"] == row["concept_id"]
            and e["route"] == row["route"]
            and e["slot"] == row["slot"]
            and float(e["entropy"]) < 0.4
            for e in entropy_rows
        )
    ]
    report.extend(["", "## Stable-Wrong Cases", ""])
    for row in stable_wrong[:20]:
        report.append(
            f"- `{row['concept_id']}` route `{row['route']}` slot `{row['slot']}`: "
            f"error={row['error_rate']:.3f}, target={row['target_value']}"
        )
    report.extend(["", "## Sample Artifacts", ""])
    for row in t2i_rows[:10]:
        report.append(f"- T2I `{row['concept_id']}` sample {row['sample_id']}: `{row.get('image_path')}`")
    (out_dir / "case_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"[INFO] aggregate wrote {out_dir}", flush=True)


def main() -> None:
    args = parse_args()
    if args.mode == "prepare":
        prepare_outputs(args)
    elif args.mode == "worker":
        run_worker(args)
    elif args.mode == "aggregate":
        aggregate_outputs(args)


if __name__ == "__main__":
    main()
