#!/usr/bin/env python3
"""Prepare and score human validation for strict semantic-entropy extraction.

The prepare step fixes one I2T reference image and one T2I generated image per
concept, then writes a manual annotation template. It intentionally does not
run Emu3.5 or regenerate any model output.
"""

from __future__ import annotations

import argparse
import base64
import csv
import html
import json
from collections import Counter, defaultdict
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


SLOTS = ["object_1", "color_1", "object_2", "color_2", "relation", "background"]
ROUTES = ["I2T", "T2I"]
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
MANUAL_FIELDS = ["annotation_id", "annotator_id", *SLOTS, "notes"]
MANIFEST_FIELDS = [
    "annotation_id",
    "concept_id",
    "route",
    "sample_id",
    "image_path",
    "canonical_prompt",
    *[f"gold_{slot}" for slot in SLOTS],
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("prepare", "validate-manual", "score"), required=True)
    parser.add_argument("--pilot-dir", default="outputs/semantic_entropy_umm/pilot")
    parser.add_argument("--strict-dir", default="outputs/semantic_entropy_umm/strict_compare")
    parser.add_argument("--out-dir", default="outputs/semantic_entropy_umm/extractor_validation")
    parser.add_argument("--preferred-t2i-sample-id", type=int, default=0)
    parser.add_argument("--contact-sheet-concepts-per-page", type=int, default=5)
    parser.add_argument("--overwrite-manual-template", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def manual_template_has_labels(path: Path) -> bool:
    if not path.exists():
        return False
    for row in read_csv(path):
        if any(row.get(slot, "").strip() for slot in SLOTS):
            return True
    return False


def concepts_in_order(pilot_dir: Path) -> list[dict]:
    concepts = read_jsonl(pilot_dir / "concepts.jsonl")
    seen = set()
    ordered = []
    for concept in concepts:
        concept_id = concept["concept_id"]
        if concept_id in seen:
            raise ValueError(f"duplicate concept_id in concepts.jsonl: {concept_id}")
        seen.add(concept_id)
        ordered.append(concept)
    return ordered


def strict_state_index(strict_dir: Path) -> dict[tuple[str, str], list[dict]]:
    rows = read_jsonl(strict_dir / "strict_semantic_states.jsonl")
    index: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        row["sample_id"] = int(row["sample_id"])
        index[(row["concept_id"], row["route"])].append(row)
    for values in index.values():
        values.sort(key=lambda row: row["sample_id"])
    return index


def choose_state(rows: list[dict], route: str, preferred_t2i_sample_id: int) -> dict:
    if not rows:
        raise ValueError(f"no strict state rows for route {route}")
    if route == "T2I":
        for row in rows:
            if row["sample_id"] == preferred_t2i_sample_id and Path(row["image_path"]).exists():
                return row
        for row in rows:
            if Path(row["image_path"]).exists():
                return row
    for row in rows:
        if row["sample_id"] == 0 and Path(row["image_path"]).exists():
            return row
    for row in rows:
        if Path(row["image_path"]).exists():
            return row
    # Keep deterministic output even if an image is missing; validation will
    # report the missing path instead of silently swapping to an unknown file.
    return rows[0]


def build_manifest(args: argparse.Namespace) -> list[dict]:
    pilot_dir = Path(args.pilot_dir)
    strict_dir = Path(args.strict_dir)
    concepts = concepts_in_order(pilot_dir)
    state_index = strict_state_index(strict_dir)
    manifest = []
    missing = []
    for concept_pos, concept in enumerate(concepts):
        concept_id = concept["concept_id"]
        target = concept["target_semantics"]
        for route in ROUTES:
            rows = state_index.get((concept_id, route), [])
            state = choose_state(rows, route, args.preferred_t2i_sample_id)
            image_path = state["image_path"]
            if not Path(image_path).exists():
                missing.append(image_path)
            row = {
                "annotation_id": f"semval_{concept_pos:03d}_{route.lower()}",
                "concept_id": concept_id,
                "route": route,
                "sample_id": state["sample_id"],
                "image_path": image_path,
                "canonical_prompt": concept["prompt"],
            }
            for slot in SLOTS:
                row[f"gold_{slot}"] = target[slot]
            manifest.append(row)
    if len(manifest) != len(concepts) * len(ROUTES):
        raise ValueError(f"expected {len(concepts) * len(ROUTES)} manifest rows, got {len(manifest)}")
    if missing:
        raise FileNotFoundError("missing selected image paths:\n" + "\n".join(missing[:20]))
    return manifest


def fit_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)] + "..."


def draw_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str) -> None:
    draw.text(xy, text, fill=(20, 20, 20), font=ImageFont.load_default())


def build_contact_sheets(out_dir: Path, manifest: list[dict], concepts_per_page: int) -> list[Path]:
    by_concept: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in manifest:
        by_concept[row["concept_id"]][row["route"]] = row
    concept_ids = sorted(by_concept)
    concepts_per_page = max(1, int(concepts_per_page))
    sheet_dir = out_dir / "contact_sheets"
    sheet_dir.mkdir(parents=True, exist_ok=True)
    for old_sheet in sheet_dir.glob("*.png"):
        old_sheet.unlink()

    sheets = []
    thumb_w, thumb_h = 256, 256
    label_h = 76
    margin = 24
    gutter = 24
    row_h = thumb_h + label_h + 18
    page_w = margin * 2 + thumb_w * 2 + gutter
    page_h = margin * 2 + row_h * concepts_per_page + 28
    for page_idx in range(0, len(concept_ids), concepts_per_page):
        page_concepts = concept_ids[page_idx : page_idx + concepts_per_page]
        page = Image.new("RGB", (page_w, page_h), "white")
        draw = ImageDraw.Draw(page)
        draw_label(draw, (margin, 8), f"Extractor validation contact sheet {page_idx // concepts_per_page + 1}")
        for row_idx, concept_id in enumerate(page_concepts):
            y = margin + 28 + row_idx * row_h
            for col_idx, route in enumerate(ROUTES):
                row = by_concept[concept_id][route]
                x = margin + col_idx * (thumb_w + gutter)
                image = Image.open(row["image_path"]).convert("RGB")
                image.thumbnail((thumb_w, thumb_h))
                tile = Image.new("RGB", (thumb_w, thumb_h), (245, 245, 245))
                paste_x = (thumb_w - image.width) // 2
                paste_y = (thumb_h - image.height) // 2
                tile.paste(image, (paste_x, paste_y))
                page.paste(tile, (x, y))
                draw.rectangle([x, y, x + thumb_w - 1, y + thumb_h - 1], outline=(120, 120, 120), width=1)
                label_y = y + thumb_h + 4
                draw_label(draw, (x, label_y), row["annotation_id"])
                draw_label(draw, (x, label_y + 14), f"{route} sample={row['sample_id']}")
                draw_label(draw, (x, label_y + 28), "blind visual annotation")
                draw_label(draw, (x, label_y + 42), "gold labels hidden")
        sheet_path = sheet_dir / f"contact_sheet_{page_idx // concepts_per_page + 1:02d}.png"
        page.save(sheet_path)
        sheets.append(sheet_path)
    return sheets


def image_data_url(path: str, max_size: int = 384) -> str:
    image = Image.open(path).convert("RGB")
    image.thumbnail((max_size, max_size))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def manual_rows_by_id(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    return {row["annotation_id"]: row for row in read_csv(path)}


def build_annotation_form(out_dir: Path, manifest: list[dict], manual_path: Path, *, include_context: bool, filename: str) -> Path:
    existing = manual_rows_by_id(manual_path)
    rows = []
    for row in manifest:
        manual = existing.get(row["annotation_id"], {})
        entry = {
            "annotation_id": row["annotation_id"],
            "route": row["route"],
            "sample_id": row["sample_id"],
            "image_data_url": image_data_url(row["image_path"]),
            "annotator_id": manual.get("annotator_id", ""),
            "notes": manual.get("notes", ""),
        }
        if include_context:
            entry["canonical_prompt"] = row["canonical_prompt"]
        for slot in SLOTS:
            entry[slot] = manual.get(slot, "")
        rows.append(entry)
    payload = {
        "include_context": include_context,
        "fields": MANUAL_FIELDS,
        "slots": SLOTS,
        "vocab": VOCAB,
        "rows": rows,
    }
    payload_json = json.dumps(payload, ensure_ascii=False)
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Semantic Entropy Extractor Validation</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; background: #f6f7f8; color: #1f2328; }}
    header {{ position: sticky; top: 0; z-index: 2; background: #ffffff; border-bottom: 1px solid #d0d7de; padding: 12px 18px; display: flex; gap: 16px; align-items: center; }}
    header h1 {{ font-size: 18px; margin: 0; }}
    button {{ border: 1px solid #8c959f; background: #ffffff; border-radius: 6px; padding: 7px 10px; cursor: pointer; }}
    button.primary {{ background: #1f6feb; color: #ffffff; border-color: #1f6feb; }}
    main {{ padding: 18px; }}
    .rules {{ background: #ffffff; border: 1px solid #d0d7de; border-radius: 8px; padding: 12px; margin-bottom: 16px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(420px, 1fr)); gap: 14px; }}
    .card {{ background: #ffffff; border: 1px solid #d0d7de; border-radius: 8px; padding: 12px; }}
    .meta {{ font-size: 12px; color: #57606a; line-height: 1.45; margin-bottom: 8px; }}
    .aid {{ font-weight: 700; color: #1f2328; }}
    img {{ display: block; width: 100%; max-height: 320px; object-fit: contain; background: #f0f2f4; border: 1px solid #d0d7de; border-radius: 6px; }}
    .fields {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; margin-top: 10px; }}
    label {{ font-size: 12px; color: #57606a; display: grid; gap: 3px; }}
    select, input, textarea {{ font: inherit; font-size: 13px; border: 1px solid #8c959f; border-radius: 6px; padding: 6px; background: #ffffff; min-width: 0; }}
    textarea {{ grid-column: 1 / -1; min-height: 54px; resize: vertical; }}
    .complete {{ border-color: #2da44e; }}
    .warn {{ border-color: #bf8700; }}
  </style>
</head>
<body>
  <header>
    <h1>Semantic Entropy Extractor Validation</h1>
    <button class="primary" id="download">Export manual_annotations.csv</button>
    <button id="copy">Copy CSV</button>
    <span id="status"></span>
  </header>
  <main>
    <section class="rules">
      Use only the fixed vocabulary. Use <code>unknown</code> when unclear. This form intentionally hides gold labels; scoring uses the separate manifest after export. Exported CSV fields exactly match <code>manual_annotations.csv</code>.
    </section>
    <section class="grid" id="grid"></section>
  </main>
  <script>
    const payload = {payload_json};
    const grid = document.getElementById('grid');
    const statusEl = document.getElementById('status');

    function escapeCsv(value) {{
      const text = String(value ?? '');
      if (/[",\\n\\r]/.test(text)) return '"' + text.replaceAll('"', '""') + '"';
      return text;
    }}

    function currentRows() {{
      return payload.rows.map((row) => {{
        const out = {{ annotation_id: row.annotation_id }};
        out.annotator_id = document.querySelector(`[data-id="${{row.annotation_id}}"][data-field="annotator_id"]`).value.trim();
        for (const slot of payload.slots) {{
          out[slot] = document.querySelector(`[data-id="${{row.annotation_id}}"][data-field="${{slot}}"]`).value;
        }}
        out.notes = document.querySelector(`[data-id="${{row.annotation_id}}"][data-field="notes"]`).value.trim();
        return out;
      }});
    }}

    function toCsv(rows) {{
      const lines = [payload.fields.join(',')];
      for (const row of rows) lines.push(payload.fields.map((field) => escapeCsv(row[field])).join(','));
      return lines.join('\\n') + '\\n';
    }}

    function updateStatus() {{
      let complete = 0;
      for (const row of currentRows()) {{
        if (payload.slots.every((slot) => row[slot])) complete += 1;
      }}
      statusEl.textContent = `${{complete}} / ${{payload.rows.length}} rows complete`;
    }}

    function render() {{
      for (const row of payload.rows) {{
        const card = document.createElement('article');
        card.className = 'card';
        const fields = payload.slots.map((slot) => {{
          const options = [''].concat(payload.vocab[slot]).map((value) => {{
            const selected = value === (row[slot] || '') ? ' selected' : '';
            const label = value || 'select';
            return `<option value="${{value}}"${{selected}}>${{label}}</option>`;
          }}).join('');
          return `<label>${{slot}}<select data-id="${{row.annotation_id}}" data-field="${{slot}}">${{options}}</select></label>`;
        }}).join('');
        card.innerHTML = `
          <div class="meta">
            <div class="aid">${{row.annotation_id}}</div>
            <div>${{row.route}} sample=${{row.sample_id}}</div>
            ${{payload.include_context ? `<div>target prompt: ${{row.canonical_prompt}}</div>` : '<div>blind visual annotation; prompt and gold labels hidden</div>'}}
          </div>
          <img src="${{row.image_data_url}}" alt="${{row.annotation_id}}">
          <div class="fields">
            <label>annotator_id<input data-id="${{row.annotation_id}}" data-field="annotator_id" value="${{row.annotator_id || ''}}"></label>
            ${{fields}}
            <label>notes<textarea data-id="${{row.annotation_id}}" data-field="notes">${{row.notes || ''}}</textarea></label>
          </div>`;
        grid.appendChild(card);
      }}
      grid.addEventListener('change', updateStatus);
      grid.addEventListener('input', updateStatus);
      updateStatus();
    }}

    document.getElementById('download').addEventListener('click', () => {{
      const blob = new Blob([toCsv(currentRows())], {{ type: 'text/csv;charset=utf-8' }});
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'manual_annotations.csv';
      a.click();
      URL.revokeObjectURL(url);
    }});
    document.getElementById('copy').addEventListener('click', async () => {{
      await navigator.clipboard.writeText(toCsv(currentRows()));
      statusEl.textContent = 'CSV copied';
      setTimeout(updateStatus, 1200);
    }});
    render();
  </script>
</body>
</html>
"""
    path = out_dir / filename
    path.write_text(html_text, encoding="utf-8")
    return path


def write_prepare_report(
    out_dir: Path,
    manifest: list[dict],
    contact_sheets: list[Path],
    annotation_form: Path,
    prompt_context_form: Path,
    manual_preserved: bool,
    args: argparse.Namespace,
) -> None:
    route_counts = Counter(row["route"] for row in manifest)
    lines = [
        "# Extractor Validation Prepare Report",
        "",
        "Status: READY_FOR_MANUAL_ANNOTATION",
        "",
        "## Input Files",
        "",
        f"- `{args.pilot_dir}/concepts.jsonl`",
        f"- `{args.strict_dir}/strict_semantic_states.jsonl`",
        f"- `{args.strict_dir}/strict_slot_answers.jsonl`",
        "",
        "## Commands",
        "",
        "```bash",
        "PY=./.venv-transformers/bin/python",
        "$PY scripts/build_semantic_entropy_extractor_validation.py --mode prepare",
        "```",
        "",
        "## Output Files",
        "",
        f"- `{out_dir / 'annotation_manifest.csv'}`",
        f"- `{out_dir / 'manual_annotations.csv'}`",
        f"- `{annotation_form}`",
        f"- `{prompt_context_form}`",
        f"- `{out_dir / 'contact_sheets/'}`",
        f"- `{out_dir / 'extractor_validation_report.md'}`",
        "",
        "## Sample Counts",
        "",
        f"- Concepts: `{len({row['concept_id'] for row in manifest})}`",
        f"- Manifest rows: `{len(manifest)}`",
        f"- Route counts: `{dict(route_counts)}`",
        f"- Manual slot labels required: `{len(manifest) * len(SLOTS)}`",
        "",
        "## Pass/Fail Checks",
        "",
        "- One I2T row per concept: PASS",
        "- One T2I row per concept: PASS",
        "- Selected image paths exist: PASS",
        "- T2I sampling rule: prefer sample_id=0, otherwise minimum existing sample_id.",
        f"- Existing manual annotations preserved: {'YES' if manual_preserved else 'NO'}",
        "- Browser annotation form generated with embedded thumbnails: PASS",
        "- Annotation UI hides concept ids, prompts, and gold labels: PASS",
        "- Prompt-context form generated separately for target-binding annotations: PASS",
        "",
        "## Contact Sheets",
        "",
        *[f"- `{path}`" for path in contact_sheets],
        "",
        "## Annotation Form",
        "",
        f"- `{annotation_form}`: blind image-only form; lower leakage risk, but object_1/object_2 binding can be ambiguous.",
        f"- `{prompt_context_form}`: target-prompt context form; use when object_1/object_2 binding must follow the strict slot schema.",
        "",
        "## Claim Allowed After This Step",
        "",
        "The human validation sample has been fixed and is traceable to the strict compare outputs.",
        "",
        "## Claim Still Not Allowed",
        "",
        "Do not claim extractor validity until `manual_annotations.csv` is completed and `--mode score` has produced metrics.",
    ]
    (out_dir / "extractor_validation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manual_handoff(out_dir: Path, args: argparse.Namespace, validation_report: dict | None = None) -> None:
    manifest_path = out_dir / "annotation_manifest.csv"
    manual_path = out_dir / "manual_annotations.csv"
    manifest_rows = read_csv(manifest_path) if manifest_path.exists() else []
    manual_rows = read_csv(manual_path) if manual_path.exists() else []
    completed_rows = 0
    completed_slots = 0
    if manual_rows:
        for row in manual_rows:
            complete = True
            for slot in SLOTS:
                if row.get(slot, "").strip() in VOCAB[slot]:
                    completed_slots += 1
                else:
                    complete = False
            if complete:
                completed_rows += 1
    if validation_report is None and (out_dir / "manual_annotation_validation_report.json").exists():
        validation_report = json.loads((out_dir / "manual_annotation_validation_report.json").read_text(encoding="utf-8"))
    status = validation_report.get("status", "NOT_VALIDATED") if validation_report else "NOT_VALIDATED"
    required_slots = len(manifest_rows) * len(SLOTS)
    lines = [
        "# Manual Annotation Handoff",
        "",
        f"Status: {status}",
        "",
        "This file is the handoff for completing the human extractor validation gate. It does not contain gold labels in the annotation UI; scoring joins against the manifest after validation passes.",
        "",
        "## Current Progress",
        "",
        f"- Manifest rows: `{len(manifest_rows)}`",
        f"- Manual rows: `{len(manual_rows)}`",
        f"- Completed rows: `{completed_rows}` / `{len(manifest_rows)}`",
        f"- Completed slot labels: `{completed_slots}` / `{required_slots}`",
        "",
        "## Files To Use",
        "",
        f"- Manifest: `{manifest_path}`",
        f"- Manual CSV to fill: `{manual_path}`",
        f"- Blind image-only form: `{out_dir / 'annotation_form.html'}`",
        f"- Prompt-context form: `{out_dir / 'annotation_form_prompt_context.html'}`",
        f"- Contact sheets: `{out_dir / 'contact_sheets'}`",
        f"- Optional small batches: `{out_dir / 'annotation_batches'}`",
        f"- Validator report: `{out_dir / 'manual_annotation_validation_report.md'}`",
        "",
        "Use the prompt-context form when object_1/object_2 binding is ambiguous; use the blind form when minimizing prompt leakage matters more.",
        "",
        "## Optional Batch Workflow",
        "",
        "To split the 60 fixed rows into six 10-row packets aligned with the contact sheets:",
        "",
        "```bash",
        'cd "$(git rev-parse --show-toplevel)"',
        "PY=./.venv-transformers/bin/python",
        "$PY scripts/build_semantic_entropy_annotation_batches.py --mode prepare",
        "```",
        "",
        "After filling the per-batch CSVs, merge and inspect the merged file before overwriting the main manual CSV:",
        "",
        "```bash",
        "$PY scripts/build_semantic_entropy_annotation_batches.py --mode merge",
        "$PY scripts/build_semantic_entropy_annotation_batches.py --mode merge --write-main",
        "```",
        "",
        "## Required CSV Fields",
        "",
        "`annotation_id, annotator_id, object_1, color_1, object_2, color_2, relation, background, notes`",
        "",
        "## Fixed Vocabulary",
        "",
        "- `object_1`, `object_2`: `cube`, `sphere`, `cone`, `unknown`",
        "- `color_1`, `color_2`: `red`, `blue`, `green`, `yellow`, `unknown`",
        "- `relation`: `object_1_left_of_object_2`, `object_1_right_of_object_2`, `object_1_above_object_2`, `object_1_below_object_2`, `unknown`",
        "- `background`: `white`, `other`, `unknown`",
        "",
        "## Annotation Rules",
        "",
        "- Fill all six slot columns for every annotation row.",
        "- Use only the fixed vocabulary; leave no slot blank.",
        "- Use `unknown` when the image is unclear; do not guess.",
        "- For relation, judge the relation from object_1 to object_2 under the target binding.",
        "- Put uncertainty comments in `notes`; do not paste gold labels, concept ids, or target prompts into notes.",
        "",
        "## Validate",
        "",
        "If the annotator returns a separate CSV, dry-run the safe importer first:",
        "",
        "```bash",
        'cd "$(git rev-parse --show-toplevel)"',
        "PY=./.venv-transformers/bin/python",
        "$PY scripts/import_semantic_entropy_manual_annotations.py --input data/manual_annotation_returns/returned_manual_annotations.csv",
        "```",
        "",
        "If the import report says `READY_TO_IMPORT`, write it with an automatic backup:",
        "",
        "```bash",
        "$PY scripts/import_semantic_entropy_manual_annotations.py --input data/manual_annotation_returns/returned_manual_annotations.csv --write",
        "```",
        "",
        "Then validate the main `manual_annotations.csv`:",
        "",
        "```bash",
        'cd "$(git rev-parse --show-toplevel)"',
        "PY=./.venv-transformers/bin/python",
        "$PY scripts/build_semantic_entropy_extractor_validation.py \\",
        "  --mode validate-manual \\",
        f"  --out-dir {out_dir}",
        "```",
        "",
        "The validation report must show `Status: PASS` before scoring or full robustness launchers are allowed.",
        "",
        "## Score After PASS",
        "",
        "```bash",
        'cd "$(git rev-parse --show-toplevel)"',
        "PY=./.venv-transformers/bin/python",
        "$PY scripts/build_semantic_entropy_extractor_validation.py \\",
        "  --mode score \\",
        f"  --strict-dir {args.strict_dir} \\",
        f"  --out-dir {out_dir}",
        "```",
        "",
        "Expected score outputs:",
        "",
        f"- `{out_dir / 'extractor_joined_annotations.csv'}`",
        f"- `{out_dir / 'extractor_validation_metrics.csv'}`",
        f"- `{out_dir / 'extractor_validation_report.md'}`",
        "",
        "## One-Command Post-Manual Pipeline",
        "",
        "After `manual_annotations.csv` is complete, this guarded command validates, scores, refreshes claim-boundary audit, rebuilds paper tables, and refreshes execution status:",
        "",
        "```bash",
        'cd "$(git rev-parse --show-toplevel)"',
        "bash scripts/run_semantic_entropy_post_manual_pipeline.sh",
        "```",
        "",
        "To also run full robustness after validation PASS, explicitly opt in:",
        "",
        "```bash",
        "RUN_FULL_ROBUSTNESS=option-order bash scripts/run_semantic_entropy_post_manual_pipeline.sh",
        "RUN_FULL_ROBUSTNESS=prompt-template bash scripts/run_semantic_entropy_post_manual_pipeline.sh",
        "```",
        "",
        "## Claim Boundary",
        "",
        "Before validator PASS and score outputs exist, all semantic entropy conclusions remain extractor-limited.",
    ]
    (out_dir / "manual_annotation_handoff.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    manifest = build_manifest(args)
    write_csv(out_dir / "annotation_manifest.csv", MANIFEST_FIELDS, manifest)
    manual_path = out_dir / "manual_annotations.csv"
    manual_preserved = manual_template_has_labels(manual_path) and not args.overwrite_manual_template
    if manual_preserved:
        print(f"[INFO] preserving existing labeled manual template: {manual_path}")
    else:
        manual_rows = [{"annotation_id": row["annotation_id"], "annotator_id": ""} for row in manifest]
        for row in manual_rows:
            for slot in SLOTS:
                row[slot] = ""
            row["notes"] = ""
        write_csv(manual_path, MANUAL_FIELDS, manual_rows)
    contact_sheets = build_contact_sheets(out_dir, manifest, args.contact_sheet_concepts_per_page)
    annotation_form = build_annotation_form(
        out_dir,
        manifest,
        manual_path,
        include_context=False,
        filename="annotation_form.html",
    )
    prompt_context_form = build_annotation_form(
        out_dir,
        manifest,
        manual_path,
        include_context=True,
        filename="annotation_form_prompt_context.html",
    )
    write_prepare_report(out_dir, manifest, contact_sheets, annotation_form, prompt_context_form, manual_preserved, args)
    write_manual_handoff(out_dir, args)
    print(f"[INFO] wrote {len(manifest)} validation rows under {out_dir}")


def validate_manual(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    manifest_path = out_dir / "annotation_manifest.csv"
    manual_path = out_dir / "manual_annotations.csv"
    manifest = read_csv(manifest_path)
    manual = read_csv(manual_path)
    manifest_ids = [row["annotation_id"] for row in manifest]
    manifest_id_set = set(manifest_ids)
    manual_ids = [row.get("annotation_id", "") for row in manual]
    duplicate_ids = sorted({annotation_id for annotation_id in manual_ids if manual_ids.count(annotation_id) > 1})
    unknown_ids = sorted(set(manual_ids) - manifest_id_set)
    missing_ids = sorted(manifest_id_set - set(manual_ids))
    missing_cells = []
    illegal_values = []
    leakage_notes = []
    leakage_terms = [
        "gold_",
        "canonical_prompt",
        "concept_id",
        "target_semantics",
    ]
    for row in manifest:
        leakage_terms.append(row.get("concept_id", "").lower())
        leakage_terms.append(row.get("canonical_prompt", "").lower())
    for row in manual:
        annotation_id = row.get("annotation_id", "")
        for field in MANUAL_FIELDS:
            if field not in row:
                missing_cells.append({"annotation_id": annotation_id, "field": field, "issue": "missing_field"})
        for slot in SLOTS:
            value = row.get(slot, "").strip()
            if not value:
                missing_cells.append({"annotation_id": annotation_id, "field": slot, "issue": "empty"})
            elif value not in VOCAB[slot]:
                illegal_values.append({"annotation_id": annotation_id, "slot": slot, "value": value})
        notes = row.get("notes", "")
        lowered = notes.lower()
        if any(term in lowered for term in leakage_terms):
            leakage_notes.append({"annotation_id": annotation_id, "notes": notes})

    completed_rows = 0
    for row in manual:
        if all(row.get(slot, "").strip() in VOCAB[slot] for slot in SLOTS):
            completed_rows += 1
    report = {
        "status": "PASS"
        if len(manual) == len(manifest)
        and not duplicate_ids
        and not unknown_ids
        and not missing_ids
        and not missing_cells
        and not illegal_values
        and not leakage_notes
        else "FAIL",
        "manifest_rows": len(manifest),
        "manual_rows": len(manual),
        "completed_rows": completed_rows,
        "required_slot_labels": len(manifest) * len(SLOTS),
        "completed_slot_labels": sum(bool(row.get(slot, "").strip()) for row in manual for slot in SLOTS),
        "duplicate_annotation_ids": duplicate_ids,
        "unknown_annotation_ids": unknown_ids,
        "missing_annotation_ids": missing_ids,
        "missing_cells": missing_cells,
        "illegal_values": illegal_values,
        "leakage_notes": leakage_notes,
        "manifest_path": str(manifest_path),
        "manual_path": str(manual_path),
    }
    (out_dir / "manual_annotation_validation_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    lines = [
        "# Manual Annotation Validation Report",
        "",
        f"Status: **{report['status']}**",
        "",
        "## Counts",
        "",
        f"- Manifest rows: `{report['manifest_rows']}`",
        f"- Manual rows: `{report['manual_rows']}`",
        f"- Completed rows: `{report['completed_rows']}`",
        f"- Completed slot labels: `{report['completed_slot_labels']}` / `{report['required_slot_labels']}`",
        "",
        "## Checks",
        "",
        f"- Duplicate annotation ids: `{len(duplicate_ids)}`",
        f"- Unknown annotation ids: `{len(unknown_ids)}`",
        f"- Missing annotation ids: `{len(missing_ids)}`",
        f"- Missing/empty required cells: `{len(missing_cells)}`",
        f"- Illegal vocabulary values: `{len(illegal_values)}`",
        f"- Possible leakage notes: `{len(leakage_notes)}`",
        "",
        "## Next Step",
        "",
        "Run `--mode score` only when this report is PASS.",
    ]
    if missing_cells:
        lines.extend(["", "## Missing Cells", "", "| Annotation | Field | Issue |", "|---|---|---|"])
        for row in missing_cells[:80]:
            lines.append(f"| {row['annotation_id']} | {row['field']} | {row['issue']} |")
    if illegal_values:
        lines.extend(["", "## Illegal Values", "", "| Annotation | Slot | Value |", "|---|---|---|"])
        for row in illegal_values[:80]:
            lines.append(f"| {row['annotation_id']} | {row['slot']} | `{row['value']}` |")
    (out_dir / "manual_annotation_validation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_manual_handoff(out_dir, args, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "manifest_rows": report["manifest_rows"],
                "manual_rows": report["manual_rows"],
                "completed_rows": report["completed_rows"],
                "completed_slot_labels": report["completed_slot_labels"],
                "required_slot_labels": report["required_slot_labels"],
                "duplicate_annotation_ids": len(report["duplicate_annotation_ids"]),
                "unknown_annotation_ids": len(report["unknown_annotation_ids"]),
                "missing_annotation_ids": len(report["missing_annotation_ids"]),
                "missing_cells": len(report["missing_cells"]),
                "illegal_values": len(report["illegal_values"]),
                "leakage_notes": len(report["leakage_notes"]),
                "report_json": str(out_dir / "manual_annotation_validation_report.json"),
                "report_md": str(out_dir / "manual_annotation_validation_report.md"),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    if report["status"] != "PASS":
        raise SystemExit(1)


def macro_f1(labels: list[str], gold: list[str], pred: list[str]) -> float:
    scores = []
    for label in labels:
        tp = sum(g == label and p == label for g, p in zip(gold, pred))
        fp = sum(g != label and p == label for g, p in zip(gold, pred))
        fn = sum(g == label and p != label for g, p in zip(gold, pred))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append((2 * precision * recall / (precision + recall)) if precision + recall else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def score(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    manifest = {row["annotation_id"]: row for row in read_csv(out_dir / "annotation_manifest.csv")}
    manual = read_csv(out_dir / "manual_annotations.csv")
    states = strict_state_index(Path(args.strict_dir))
    joined = []
    skipped = []
    for manual_row in manual:
        annotation_id = manual_row["annotation_id"]
        if annotation_id not in manifest:
            raise ValueError(f"manual annotation has unknown annotation_id: {annotation_id}")
        if any(not manual_row.get(slot, "").strip() for slot in SLOTS):
            skipped.append(annotation_id)
            continue
        meta = manifest[annotation_id]
        for slot in SLOTS:
            value = manual_row[slot].strip()
            if value not in VOCAB[slot]:
                raise ValueError(f"{annotation_id} has illegal value for {slot}: {value}")
        rows = states[(meta["concept_id"], meta["route"])]
        state = next(row for row in rows if int(row["sample_id"]) == int(meta["sample_id"]))
        joined_row = dict(meta)
        joined_row["annotator_id"] = manual_row.get("annotator_id", "")
        joined_row["notes"] = manual_row.get("notes", "")
        for slot in SLOTS:
            joined_row[f"manual_{slot}"] = manual_row[slot].strip()
            joined_row[f"extractor_{slot}"] = state["slots"].get(slot, "unknown")
            joined_row[f"{slot}_match"] = joined_row[f"manual_{slot}"] == joined_row[f"extractor_{slot}"]
        joined.append(joined_row)
    joined_fields = [
        *MANIFEST_FIELDS,
        "annotator_id",
        *[f"manual_{slot}" for slot in SLOTS],
        *[f"extractor_{slot}" for slot in SLOTS],
        *[f"{slot}_match" for slot in SLOTS],
        "notes",
    ]
    write_csv(out_dir / "extractor_joined_annotations.csv", joined_fields, joined)

    metric_rows = []
    for slot in SLOTS:
        gold = [row[f"manual_{slot}"] for row in joined]
        pred = [row[f"extractor_{slot}"] for row in joined]
        if not gold:
            continue
        metric_rows.append(
            {
                "metric": "slot_accuracy",
                "route": "ALL",
                "slot": slot,
                "value": sum(g == p for g, p in zip(gold, pred)) / len(gold),
                "n": len(gold),
            }
        )
        metric_rows.append(
            {
                "metric": "macro_f1",
                "route": "ALL",
                "slot": slot,
                "value": macro_f1(VOCAB[slot], gold, pred),
                "n": len(gold),
            }
        )
        metric_rows.append(
            {
                "metric": "unknown_agreement",
                "route": "ALL",
                "slot": slot,
                "value": sum((g == "unknown") == (p == "unknown") for g, p in zip(gold, pred)) / len(gold),
                "n": len(gold),
            }
        )
        for route in ROUTES:
            route_rows = [row for row in joined if row["route"] == route]
            if route_rows:
                metric_rows.append(
                    {
                        "metric": "route_slot_accuracy",
                        "route": route,
                        "slot": slot,
                        "value": sum(row[f"{slot}_match"] for row in route_rows) / len(route_rows),
                        "n": len(route_rows),
                    }
                )
    for route in ["ALL", *ROUTES]:
        rows = joined if route == "ALL" else [row for row in joined if row["route"] == route]
        if not rows:
            continue
        metric_rows.extend(
            [
                {
                    "metric": "relation_accuracy",
                    "route": route,
                    "slot": "relation",
                    "value": sum(row["relation_match"] for row in rows) / len(rows),
                    "n": len(rows),
                },
                {
                    "metric": "object_binding_accuracy",
                    "route": route,
                    "slot": "object_1+object_2",
                    "value": sum(row["object_1_match"] and row["object_2_match"] for row in rows) / len(rows),
                    "n": len(rows),
                },
                {
                    "metric": "color_object_binding_accuracy",
                    "route": route,
                    "slot": "color_1+color_2",
                    "value": sum(row["color_1_match"] and row["color_2_match"] for row in rows) / len(rows),
                    "n": len(rows),
                },
            ]
        )
    write_csv(out_dir / "extractor_validation_metrics.csv", ["metric", "route", "slot", "value", "n"], metric_rows)
    write_score_report(out_dir, joined, metric_rows, skipped)
    write_manual_handoff(out_dir, args)
    print(f"[INFO] scored {len(joined)} completed annotations under {out_dir}")


def write_score_report(out_dir: Path, joined: list[dict], metric_rows: list[dict], skipped: list[str]) -> None:
    lines = [
        "# Extractor Validation Report",
        "",
        "Status: SCORED" if joined else "Status: NO_COMPLETED_ANNOTATIONS",
        "",
        "## Input Files",
        "",
        f"- `{out_dir / 'annotation_manifest.csv'}`",
        f"- `{out_dir / 'manual_annotations.csv'}`",
        "",
        "## Output Files",
        "",
        f"- `{out_dir / 'extractor_joined_annotations.csv'}`",
        f"- `{out_dir / 'extractor_validation_metrics.csv'}`",
        f"- `{out_dir / 'extractor_validation_report.md'}`",
        "",
        "## Sample Counts",
        "",
        f"- Completed annotation rows: `{len(joined)}`",
        f"- Skipped incomplete annotation rows: `{len(skipped)}`",
        f"- Completed manual slot labels: `{len(joined) * len(SLOTS)}`",
        "",
        "## Metrics",
        "",
        "| Metric | Route | Slot | Value | N |",
        "|---|---|---|---:|---:|",
    ]
    for row in metric_rows:
        lines.append(f"| {row['metric']} | {row['route']} | {row['slot']} | {float(row['value']):.4f} | {row['n']} |")
    disagreement_rows = []
    for row in joined:
        bad = [slot for slot in SLOTS if not row[f"{slot}_match"]]
        if bad:
            disagreement_rows.append((row, bad))
    lines.extend(["", "## Disagreements", ""])
    if disagreement_rows:
        lines.extend(["| Annotation | Concept | Route | Sample | Slots | Image |", "|---|---|---|---:|---|---|"])
        for row, bad in disagreement_rows:
            lines.append(
                f"| {row['annotation_id']} | {row['concept_id']} | {row['route']} | {row['sample_id']} | {', '.join(bad)} | `{row['image_path']}` |"
            )
    else:
        lines.append("No disagreements among completed annotations.")
    lines.extend(
        [
            "",
            "## Claim Allowed After This Step",
            "",
            "Extractor validity can only be interpreted for the completed manual annotation rows and reported slot metrics.",
            "",
            "## Claim Still Not Allowed",
            "",
            "Do not treat entropy as correctness; report entropy together with error, unknown rate, and invalid parse rate.",
        ]
    )
    (out_dir / "extractor_validation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.mode == "prepare":
        prepare(args)
    elif args.mode == "validate-manual":
        validate_manual(args)
    else:
        score(args)


if __name__ == "__main__":
    main()
