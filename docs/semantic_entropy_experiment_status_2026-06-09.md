# Semantic Entropy Experiment Status, 2026-06-09

This note summarizes the current Emu3.5 I2T/T2I semantic entropy experiments on `lb-gzs`.
Remote root: `/home/wentao/project/Emu3.5`.

## Current Bottom Line

The controlled semantic-state protocol is implemented and has enough evidence for a paper-draft-level, extractor-limited claim:

- I2T and T2I are projected into the same finite semantic state space: `object_1`, `color_1`, `object_2`, `color_2`, `relation`, `background`.
- The main 30-concept strict comparison passes count/schema parity: 600 states, 3600 slot answers, 30 concepts, 10 samples per concept-route.
- The strongest current finding is not raw cross-modal entropy comparability. It is route-level semantic-state comparability under a shared extractor/parser.
- For T2I hallucination, semantic entropy strongly predicts automatic QA/readback errors, but manual visual audit shows those automatic labels mostly do not equal true image hallucination.

## Completed Experiment Families

| family | status | key evidence | main result |
| --- | --- | --- | --- |
| Strict I2T/T2I semantic comparability | PASS | `outputs/semantic_entropy_umm/strict_compare/comparability_audit.md` | Main baseline has I2T/T2I parity over the same semantic schema. T2I has higher object_1 error in the baseline run: 0.3200 vs 0.1833. |
| Formal Emu3.5 run | PASS | `outputs/semantic_entropy_umm/semantic_uncertainty_formal_emu35_20260606_095911` | Formal run has 30 concepts x 10 samples. Strict audit passes; strict entropy/error direction differs from earlier baseline, so robustness caveats matter. |
| T2I semantic entropy vs hallucination/readback | COMPLETE_WITH_MANUAL_VISUAL_AUDIT | `outputs/semantic_entropy_umm/t2i_hallucination_uncertainty/t2i_hallucination_uncertainty_detailed_report.md` | Automatic labels: ALL mode-error AUROC 0.9520, joint mode-error AUROC 0.9080. Manual visual audit: 0/24 automatic hallucination candidates supported as true image hallucination; 12/12 non-hallucination candidates supported. |
| T2I 20-benchmark extension | COMPLETE_AUTOMATIC_ANALYSIS | `outputs/semantic_entropy_umm/t2i_bench20_emu35_20260608` | 20 new concepts x 5 T2I samples. Strict and QA audits pass. Automatic detector: ALL mode-error AUROC 0.9848, joint mode-error AUROC 1.0000, object_1 mode-error AUROC 0.9722. |
| Extractor validation | SCORED | `outputs/semantic_entropy_umm/extractor_validation/extractor_validation_report.md` | 60 completed annotation rows, 360 slot labels. I2T route is strong, but T2I route validation is poor under current object_1/object_2 semantics, indicating readback/slot-definition artifacts. |
| Bootstrap stability | PASS | `outputs/semantic_entropy_umm/bootstrap_stability/bootstrap_report.md` | Concept-level bootstrap complete. Joint entropy delta CI crosses zero; joint/object_1 error delta is positive in the baseline. |
| Sample-size sensitivity | PASS | `outputs/semantic_entropy_umm/sample_size_sensitivity/sample_size_sensitivity_report.md` | n=5 and n=10 resampling complete; supports current automated-extractor pilot only. |
| Quadrant analysis | PASS | `outputs/semantic_entropy_umm/quadrant_analysis/quadrant_report.md` | Low-entropy high-error cases exist, especially object_1. This matters because entropy alone is not correctness. |
| Slot ablation | PASS | `outputs/semantic_entropy_umm/slot_ablation/slot_ablation_report.md` | Positive entropy/error deltas are localized to the object family, specifically object_1. |
| Easy/hard concept split | PASS | `outputs/semantic_entropy_umm/concept_difficulty_split/concept_difficulty_report.md` | Hard concepts concentrate T2I joint entropy and object_1 errors; easy concepts are near zero under the automated extractor. |
| Random concept control | PASS | `outputs/semantic_entropy_umm/random_concept_control/random_concept_control_report.md` | Mismatched-target control raises error strongly, so the metric is sensitive to semantic target mismatch. |
| Object-binding targeted questions | PASS | `outputs/semantic_entropy_umm/object_binding_sanity/targeted_questions/targeted_binding_score_report.md` | 90/90 targeted outputs complete; all accuracy 0.9222, relation accuracy 1.0000, T2I accuracy 0.9067. |
| Full robustness families | PASS | `outputs/semantic_entropy_umm/robustness_option_order_idle_gpus`, `outputs/semantic_entropy_umm/robustness_prompt_template_idle_gpus` | Option-order seeds and prompt-template variants completed; some settings change many slot values, so final wording must keep robustness caveats. |
| Duplicate/mode-collapse checks | PASS with caveat | `outputs/semantic_entropy_umm/image_duplicate_mode_collapse`, `outputs/semantic_entropy_umm/image_embedding_duplicate_check` | Exact/perceptual hash checks complete. Embedding check is only pixel-stat fallback, not CLIP/SigLIP/DINO semantic diversity. |
| Claim-boundary audit | PASS | `outputs/semantic_entropy_umm/claim_boundary_audit/claim_boundary_audit_report.md` | No detected overclaim wording in scanned Markdown. |
| Paper table package | PASS | `outputs/semantic_entropy_umm/paper_tables_preliminary/paper_tables_preliminary.md` | Preliminary evidence tables and failure appendix exist. |

## New 20-Benchmark Extension

Remote run:

`outputs/semantic_entropy_umm/t2i_bench20_emu35_20260608`

Status:

- 20 additional T2I benchmark concepts were created.
- 5 formal samples per concept were generated: 100/100 PNGs present.
- Formal Emu3.5 T2I parameters were used: `model/Emu3.5`, `target_height=64`, `target_width=64`, `image_area=1048576`, `t2i_max_new_tokens=5120`, CFG 2.0, top-k 5120, temperature 1.0.
- Initial postprocess failed because `pilot/reference_images/*.png` was missing.
- This was fixed by generating 20 reference images and `image_manifest.jsonl`.
- Postprocess completed via `outputs/semantic_entropy_umm/t2i_bench20_emu35_20260608/resume_postprocess.sh`.

Completed postprocess:

1. strict compare: PASS, 200 states, 1200 slot answers.
2. semantic uncertainty QA: PASS, 200 states, 1200 slot answers.
3. T2I hallucination uncertainty analysis: complete, 140 datapoints, 96 detector metrics, 28 visual-audit candidates.

This run is complete for automatic analysis. It still needs manual visual audit before making true image hallucination claims.

Current strict result:

- `strict_compare/comparability_audit.md` exists and passes.
- Expected samples per concept-route: 5.
- Sample count by route: I2T 100, T2I 100.
- Joint entropy: I2T 0.0837, T2I 0.1260, delta +0.0423.
- Object_1 entropy: I2T 0.0587, T2I 0.1010, delta +0.0423.
- Object_1 error: I2T 0.1800, T2I 0.1700, delta -0.0100.
- Relation error: I2T 0.0100, T2I 0.0100, delta 0.0000.

Current QA and detector result:

- `semantic_uncertainty_qa/qa_comparability_audit.md` exists and passes.
- QA joint entropy: I2T 0.4984, T2I 0.5036, delta +0.0052.
- QA object_2 entropy has the largest positive slot delta: +0.1001.
- QA object_1 error: I2T 0.1200, T2I 0.0900, delta -0.0300.
- Detector datapoints: 140 concept-slot groups.
- ALL mode-error AUROC: 0.9848.
- Joint mode-error AUROC: 1.0000.
- Object_1 mode-error AUROC: 0.9722.
- Visual-audit candidates: 28.

## Remaining Gaps

| gap | status | why it matters | action |
| --- | --- | --- | --- |
| Bench20 postprocessing | COMPLETE_AUTOMATIC_ANALYSIS | The 100 new images, strict compare, QA compare, and automatic hallucination analysis are complete. | Manual visual audit remains if this run is used for true image hallucination claims. |
| True T2I hallucination labels for all generated images | MISSING | Current automatic hallucination labels are mostly readback/extractor errors under manual audit. | Build or fill manual visual GT over generated images, then rerun AUROC against true visual hallucination labels. |
| Position-grounded slot schema | MISSING | `object_1/object_2` is a major source of readback artifacts, especially when object_1 is on the right. | Rerun a left/right or top/bottom object schema to separate image failure from slot-index failure. |
| Second independent extractor | BLOCKED | One VQA/readback model cannot prove true image hallucination. | Need a complete local extractor or approved external API/service. Current feasibility audit found no complete local alternative. |
| CLIP/SigLIP/DINO semantic embedding duplicate check | BLOCKED | Pixel/hash checks do not prove semantic diversity. | Need complete local embedding model cache or approved external fetch. |
| X2X unified entropy | NOT_STARTED | Extends beyond I2T/T2I into interleaved text+image consistency. | Future stage after the I2T/T2I label problem is resolved. |
| Internal/path-level entropy claim | NOT_STARTED | Needed for claims about unified internal entropy space. | Requires hidden-state, causal intervention, residual/head/path experiments; not supported by current output-level protocol. |

## Current Claim Boundary

Allowed:

- Controlled route-level semantic entropy comparison after projecting I2T and T2I into the same finite semantic state schema.
- Semantic entropy as a detector of automatic T2I QA/readback errors.
- The observation that current automatic hallucination labels can be readback/slot-binding artifacts rather than true image hallucination.

Not allowed:

- Raw text entropy and image entropy are directly comparable.
- Emu3.5 has a unified internal entropy space.
- Current semantic entropy already detects true T2I image hallucination without stronger visual labels.
- Current T2I automatic readback errors are necessarily generation hallucinations.

## Recommended Next Order

1. Let the bench20 postprocess finish and add its metrics to the report package.
2. Run a position-grounded schema (`left_object/right_object` or `top_object/bottom_object`) on the formal 30-concept run and the new 20-benchmark run.
3. Build manual visual GT for a bounded T2I set: at minimum all high/low entropy candidates plus a random negative set; ideally all formal 300 images plus bench20 100 images.
4. Rerun the jlko-style AUROC/selective-accuracy analysis against manual visual hallucination labels.
5. Only after that, consider a second extractor or CLIP/SigLIP/DINO semantic embedding check if the claim needs stronger defense.
