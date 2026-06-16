# Emu3.5 Semantic-Uncertainty-Style I2T/T2I Formal Protocol

## Source Protocol

- Source repository: `third_party/semantic_uncertainty` cloned from `https://github.com/jlko/semantic_uncertainty`.
- Source files inspected:
  - `third_party/semantic_uncertainty/semantic_uncertainty/generate_answers.py`
  - `third_party/semantic_uncertainty/semantic_uncertainty/compute_uncertainty_measures.py`
  - `third_party/semantic_uncertainty/semantic_uncertainty/uncertainty/utils/utils.py`
  - `third_party/semantic_uncertainty/semantic_uncertainty/uncertainty/uncertainty_measures/semantic_entropy.py`
- The source protocol samples multiple answers for each QA item and computes uncertainty over semantic equivalence classes.
- The source default short-answer prompt is:

```text
Answer the following question as briefly as possible.
Context: {context}
Question: {question}
Answer:
```

## Multimodal Adaptation

Raw free-form answer entailment is not used for I2T/T2I route comparison. Instead, both routes are projected into the same finite visual semantic state space:

```text
object_1, color_1, object_2, color_2, relation, background
```

For each slot, Emu3.5 is asked a semantic-uncertainty-style visual QA question with the same `brief + Context + Question + Answer` structure. Free-form answers are normalized into the finite slot vocabulary, then cluster-assignment entropy is computed from answer/state counts.

## Convergence Decision

The current baseline keeps `model/Emu3.5` rather than switching the main experiment to `Emu3.5-Image`. `Emu3.5-Image` is still the stronger specialized T2I/X2I model, but the earlier nearly blank or striped 30-image batch should be treated as a misconfigured generation run, not as evidence that `Emu3.5` cannot produce usable T2I outputs.

The key correction is to stop using smoke-test generation defaults for formal T2I. The old `16x16` visual grid with `max_new_tokens=360` is too small for this experiment. Formal runs use the `64x64` grid and `max_new_tokens=5120` setting below, so semantic-entropy analysis is less likely to confuse generation failure with semantic uncertainty.

Operationally, `Emu3.5` remains the default baseline and `Emu3.5-Image` is reserved for an explicit model-family comparison. The old 30-image failure batch is excluded from capability claims because it was generated under smoke-test settings.

## Formal Run

Remote run id:

```text
semantic_uncertainty_formal_emu35_20260606_095911
```

Remote path:

```text
/home/wentao/project/Emu3.5/outputs/semantic_entropy_umm/semantic_uncertainty_formal_emu35_20260606_095911
```

Scale:

```text
30 concepts x 10 T2I samples = 300 generated images
I2T: 30 x 10 repeated QA state samples
T2I: 30 x 10 generated-image QA state samples
```

T2I settings:

```text
model_path=model/Emu3.5
target_height=64
target_width=64
image_area=1048576
max_new_tokens=5120
classifier_free_guidance=2.0
image_top_k=5120
image_temperature=1.0
```

Outputs:

```text
pilot/t2i_samples.jsonl
pilot/generated_images/
strict_compare/strict_route_entropy.csv
strict_compare/strict_route_error.csv
strict_compare/comparability_audit.json
semantic_uncertainty_qa/qa_route_entropy.csv
semantic_uncertainty_qa/qa_route_error.csv
semantic_uncertainty_qa/qa_comparability_audit.json
```

## Caveat

This is semantic entropy over controlled visual QA state clusters. It is not raw token entropy and should not be compared to text/image token entropy. The QA-style run is the semantic_uncertainty-inspired primary readout; the strict forced-choice run is retained as a comparability audit.
