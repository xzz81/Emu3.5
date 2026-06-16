# T2I Complex Scene Upgrade Report

Status: READY_TO_RUN_COMPLEX_BENCHMARK

## Why the current scene is too simple

- Current visual GT covers `50` concepts and `400` images, but the task schema is always two primitive objects, two colors, one binary spatial relation, and a background slot.
- True visual errors are sparse: strict bad images `10/400` (0.0250), core bad images `6/400` (0.0150).
- At concept level, strict-any positives are `8/50` and core-any positives are `5/50`; both majority labels have `0` / `0` positives.
- This positive rate is too low for a stable hallucination detector evaluation: AUROC is defined, but the confidence boundary is weak because only a few concepts contribute positives.

## Existing detector signal under true visual GT

| label | positives | AUROC | AUPRC | mean_positive_entropy | mean_negative_entropy |
| --- | --- | --- | --- | --- | --- |
| strict_any_visual_hallucination | 8 | 0.5223 | 0.2437 | 0.6107 | 0.5648 |
| core_any_visual_hallucination | 5 | 0.5333 | 0.1383 | 0.6007 | 0.5690 |
| background_any_visual_issue | 4 | 0.4647 | 0.1890 | 0.5518 | 0.5739 |

## Simplicity evidence from sample diversity

- Formal generated-image duplicate audit covers `30` concepts.
- Mean high-similarity pair rate is `0.5519` and max is `1.0000`.
- Fallback/mode-collapse flag appears in `15` concepts.

## Proposed complex benchmark

- Concept manifest: `configs/semantic_entropy_t2i_complex_scene_concepts.jsonl`
- Run launcher: `scripts/launch_semantic_uncertainty_complex_t2i_emu35.sh`
- It keeps the current formal Emu3.5 T2I baseline: `model/Emu3.5`, `target_height=64`, `target_width=64`, `image_area=1048576`, `t2i_max_new_tokens=5120`, `classifier_free_guidance=2.0`, `image_top_k=5120`, `image_temperature=1.0`.
- The current finite-slot automatic QA remains available for anchor slots, but the richer hallucination target should be the visual GT over `visual_checks` because current automatic slots cannot score count, occlusion, text, material, containment, and negative constraints.

## Suite-level complexity comparison

| suite | concepts | mean_prompt_tokens | mean_visual_check_count | mean_complexity_tag_count | unique_complexity_tags | top_complexity_tags |
| --- | --- | --- | --- | --- | --- | --- |
| complex_proposed | 30 | 43.7333 | 5.0000 | 3.4000 | 62 | count_exact:11;negative_constraint:9;third_object:6;factual_diagram:6;three_objects:3;relation_binding:3;four_objects:2;occlusion:2;depth_order:2;relative_order:2;attribute_binding:2;count_control:2 |
| current | 50 | 11.6800 | 6.0000 | 1.0000 | 1 | two_object_single_relation:50 |

## Complex axes covered

- `count_exact`: 11
- `negative_constraint`: 9
- `third_object`: 6
- `factual_diagram`: 6
- `three_objects`: 3
- `relation_binding`: 3
- `four_objects`: 2
- `occlusion`: 2
- `depth_order`: 2
- `relative_order`: 2
- `attribute_binding`: 2
- `count_control`: 2
- `relative_size`: 2
- `ordered_sequence`: 2
- `relation_chain`: 1
- `vertical_order`: 1
- `transparency`: 1
- `distractor_absence`: 1
- `containment`: 1
- `nested_layout`: 1
- `multi_relation`: 1
- `reflection`: 1
- `asymmetric_property`: 1
- `text_binding`: 1
- `asymmetric_label`: 1
- `distractor_object`: 1
- `material_binding`: 1
- `swap_risk`: 1
- `partial_occlusion`: 1
- `attribute_visibility`: 1
- `grid_position`: 1
- `empty_cells`: 1
- `spatial_precision`: 1
- `same_color_binding`: 1
- `shape_discrimination`: 1
- `same_shape_binding`: 1
- `different_color`: 1
- `lighting_direction`: 1
- `compound_relation`: 1
- `container`: 1
- `inside_outside`: 1
- `negative_background`: 1
- `two_relations`: 1
- `shared_anchor`: 1
- `l_shape`: 1
- `similar_shape_distractor`: 1
- `local_attribute`: 1
- `color_contamination_risk`: 1
- `object_part`: 1
- `asymmetry`: 1
- `negative_relation`: 1
- `depth_scale`: 1
- `perspective`: 1
- `traffic_light`: 1
- `chemistry`: 1
- `molecule_geometry`: 1
- `astronomy`: 1
- `circuit`: 1
- `connectivity`: 1
- `chart_reading`: 1
- `map_legend`: 1
- `symbol_binding`: 1

## Recommended next run

```bash
RUN_ID=semantic_uncertainty_complex_t2i_emu35_20260616 \
NUM_CONCEPTS=30 T2I_SAMPLES=10 GEN_WORKERS=2 STRICT_WORKERS=2 QA_WORKERS=2 \
DEVICE_GROUPS='0,3;5,7' \
bash scripts/launch_semantic_uncertainty_complex_t2i_emu35.sh
```

After generation, build contact sheets and annotate all complex `visual_checks`; then rerun true-visual-GT detector metrics with the complex GT. The expected research value is a higher positive rate and richer error axes, which makes the semantic entropy vs hallucination relationship testable beyond the current toy two-object setting.
