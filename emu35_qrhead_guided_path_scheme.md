# Emu3.5 QRHead-Guided Generation / Understanding Path Scheme

## 0. One-Sentence Goal

Use the literature chain:

```text
annotated evidence -> attention/evidence score -> sparse candidate heads
-> causal patch / ablation -> cross-task controls -> path claim
```

to refine the current Emu3.5 UMM pathway work.

The important change is not to replace the existing patch-recovery pipeline. The change is to add a QRHead-style evidence ranking stage before patching, so the tested heads are selected by "what evidence they route" rather than only by downstream recovery.

## 1. Current Local Evidence Boundary

The existing reports already support a conservative state:

```text
understanding:
  strong late residual route
  image information is mediated through post-image question/text states
  L62 heads 36/38/46 are the cleanest head-level route so far

generation:
  strongest route is target-side visual-score residual / feature-subspace evidence
  prompt-side L60 attention heads are weak and shared
  L61 visual-score heads and MLP feature directions are generation-target-side specific
  color-token input embedding patch has compact decoded support
```

Therefore the next question should be:

```text
Can evidence-ranked heads and edges explain a meaningful fraction of these already-known residual/module effects?
```

not:

```text
Can attention heatmaps alone prove a generation/understanding path?
```

## 2. Claim Ladder

Every component or path must pass this ladder before being named a path.

| stage | evidence | allowed claim |
|---|---|---|
| A | attention / activation contrast | candidate route |
| B | clean ablation or knockout hurts target score | necessary component under this benchmark |
| C | clean-to-corrupt patch recovers target score | sufficient causal component under this benchmark |
| D | wrong/random/bottom/object/cross-task controls fail | task- or attribute-selective component |
| E | path explains a fraction of full residual/module effect | partial mechanistic path |

Do not claim a path from A alone.

## 3. Shared Scoring Definitions

For any layer-head `h`, use post-softmax attention:

```text
A_h[q, s] = attention from query/destination token q to source/evidence token s
```

For an evidence set `E` and query set `Q`:

```text
EvidenceMass(h; Q, E) =
  mean_{q in Q} sum_{s in E} A_h[q, s]
```

For visual grounding, also compute:

```text
ImageMass(h)      = mean_q sum_{s in all_visual_tokens} A_h[q, s]
RegionRatio(h)    = RegionMass(h) / (ImageMass(h) + eps)
SpatialEntropy(h) = normalized entropy over the visual-token grid
Concentration(h)  = 1 - SpatialEntropy(h)
```

Recommended ranking score:

```text
RankScore(h) =
  z(EvidenceMass)
  + z(RegionRatio)
  + z(Concentration)
  + z(SelectionFrequency)
```

For image-level evidence without region labels, drop `RegionRatio` and mark the result as image-level, not grounding-level.

## 4. Understanding Path

### 4.1 Path Hypothesis

Current best hypothesis:

```text
image visual tokens / EOI
-> post-image question/text residual states
-> answer score positions
-> answer-token logits
```

Direct answer-score to image attention is not expected to be the whole route.

### 4.2 Detection Data

Use the current hue-control image-read benchmark first:

```text
cfg:      configs/ume_main_image_read_hue_control_gt_seed70.py
manifest: research_logs or configured hue-control GT manifest
task:     What color is the {shape}? Answer with one word.
```

Add region labels if available. If unavailable, run a two-tier detection:

```text
U0 image-level: source = all visual tokens + EOI
U1 region-level: source = known object-color patch tokens / bbox tokens
```

### 4.3 Understanding Head Ranking

Use QRHead-style query-to-evidence attention:

```text
Q_U_primary   = post-image question/text tokens, excluding answer score tokens
Q_U_secondary = answer score positions
E_U_image     = visual input tokens + EOI
E_U_region    = GT object/attribute region tokens
```

Scores:

```text
U_image(h)  = EvidenceMass(h; Q_U_primary, E_U_image)
U_region(h) = EvidenceMass(h; Q_U_primary, E_U_region)
U_score(h)  = RankScore over image/region mass and spatial concentration
```

Expected sanity check:

```text
L62 heads 36/38/46 should rank high or be recovered by causal tests.
If they do not rank high by raw attention, that is useful evidence that their value/result vector matters more than their raw attention mass.
```

### 4.4 Understanding Causal Tests

Use existing scripts:

```text
scripts/run_understanding_attention_head_patch_recovery.py
scripts/run_understanding_attention_knockout.py
scripts/run_understanding_residual_patch_recovery.py
scripts/run_understanding_module_patch_recovery.py
```

Required tests:

```text
1. patch top U heads into corrupt image runs
2. ablate or zero top U heads in clean image runs
3. reverse patch corrupt heads into clean runs
4. compare against same-layer random heads and bottom heads
5. run generation cross-task control for top U heads
```

Primary metrics:

```text
answer NLL recovery
answer logit margin recovery
answer flip rate
VQA exact match
region mass drop after masking
```

A strong understanding claim needs:

```text
S_under(top_U) > random/bottom controls
S_gen(top_U) approx 0 on generation controls
effect stable across multiple color/shape pairs
```

## 5. Generation Path

Generation must be split into two different things.

### 5.1 Prompt-Conditioning Path

This is the compact path we actually want:

```text
prompt color token
-> prompt / early residual states
-> visual-score destination positions
-> clean visual-token logits
-> decoded hue
```

This path is supported at the input boundary by Stage 45/46:

```text
color-token input embedding patch: 2/4 exact, 4/4 hue-distance improved
object-token input embedding control: 0/4 exact, 0/4 improved
```

The next target is to locate internal edges/components downstream of that boundary.

### 5.2 Target-Trajectory Path

This is the current upper bound:

```text
clean target-side residual / input trajectory
-> visual-token logits / decoded hue
```

It is real causal evidence, but it is partly a trajectory overwrite. Treat it as an upper bound and completeness denominator, not as a compact generation circuit.

### 5.3 Generation Head Ranking

Use destination visual-score positions as queries, because in a decoder-only UMM the visual target positions attend backward to prompt tokens:

```text
Q_G = visual-score positions, preferably GT object/attribute region positions
E_G_color  = prompt color-token positions
E_G_object = prompt object-token positions
E_G_prompt = all prompt semantic tokens
```

Attribute-focused score:

```text
G_color_attn(h) =
  mean_{q in Q_G} sum_{s in E_G_color} A_h[q, s]
```

Color specificity score:

```text
G_color_specific(h) =
  G_color_attn(h)
  - G_object_attn(h)
  - G_null_attn(h)
```

where `G_null_attn` can be the same score under a null or attribute-shuffled prompt.

For multi-object generation, define `Q_G` as region-specific target visual tokens. For current highcontrast single-object cases, `Q_G` can start as all visual target tokens, but the claim should remain image-level.

### 5.4 Add Write/Logit Attribution

Generation heads should not be ranked by attention alone.

For each candidate head result vector `r_h(q)` at visual-score position `q`, add a write score:

```text
WriteScore(h) =
  mean_q dot(r_h(q), clean_visual_logit_direction(q))
```

Practical approximations:

```text
1. direct contribution to clean target visual-token logit
2. contribution to clean-vs-corrupt target-token logit margin
3. patch recovery from the existing head-output patch script
```

Final candidate score:

```text
G_candidate(h) =
  z(G_color_specific)
  + z(WriteScore)
  + z(clean-to-corrupt head patch recovery on visual-score positions)
```

This protects against high-attention heads that only point at the right token but write the wrong thing.

### 5.5 Generation Causal Tests

Use existing scripts:

```text
scripts/run_generation_attention_head_patch_recovery.py
scripts/run_generation_residual_patch_recovery.py
scripts/run_generation_module_patch_recovery.py
scripts/run_generation_mlp_lowrank_patch_recovery.py
scripts/run_generation_target_residual_patched_decode_hue.py
```

Required teacher-forced tests:

```text
1. clean-to-corrupt patch top G heads at visual-score positions
2. clean ablation / mean-replace top G heads
3. reverse corrupt-to-clean patch
4. compare top heads to bottom heads, random heads, and attention-matched low-write heads
5. object-token and wrong-color controls
6. understanding cross-task controls
```

Required decoded tests before a strong generation claim:

```text
1. HSV/VQA improvement on free corrupt-prompt decode
2. wrong-target source control does not exact-restore
3. object-token or random-color-token control does not improve
4. image quality / format does not collapse
```

Expected interpretation:

```text
If top G heads recover only ~10% while full residual recovers ~100%,
write: "head subset is a partial target-side contributor".

If MLP PCA/SAE directions recover more than heads and pass controls,
write: "generation path is feature-subspace structured rather than sparse-head structured".
```

## 6. Minimal Next Experiment Set

### Experiment U1: QR-Style Understanding Head Census

Implement or notebook a small head-ranking pass:

```text
task = understanding
layers = 56-63 first, then all layers
query scopes = post-image-text-prompt, score-positions
source scopes = visual tokens, EOI, optional GT region tokens
output = head_scores_understanding.csv
```

Success condition:

```text
top-ranked heads overlap with L62 36/38/46 or explain why attention ranking differs from recovery ranking.
```

### Experiment U2: Necessity For L62 Heads

Run clean ablation/knockout for:

```text
L62 heads 36/38/46
L62 top7 from Stage 6
same-layer random3/random7
bottom/negative heads 35/32
```

Success condition:

```text
clean answer NLL/logit margin degrades more for top heads than controls.
```

### Experiment G1: Attribute-Focused Generation Head Census

Implement or notebook a generation ranking pass:

```text
task = generation
cfg = configs/ume_main_t2i_counterfactual_pairs_highcontrast_seed68.py
target cache = outputs/generation_raw_target_cache_highcontrast_seed68
query = visual-score positions
source = color token, object token, all prompt tokens
layers = 0,8,16,32,40,48,52,56-63 first
output = head_scores_generation.csv
```

Success condition:

```text
heads with high color-specific attention also show positive write score or patch recovery.
```

Failure is still useful:

```text
If attention-ranked heads do not patch well, generation is not QRHead-like at head level.
Then focus on color-token boundary plus MLP feature subspace.
```

### Experiment G2: Generation Edge Knockout

Adapt `run_understanding_attention_knockout.py` to generation:

```text
query = visual-score positions
source = color-token positions
heads = G1 top heads and controls
intervention = add -inf attention mask for selected query-source edges
metric = clean target visual-token NLL / logit margin
```

Success condition:

```text
masking color-token -> visual-score edges hurts clean score more than object-token/random edges.
```

### Experiment G3: Decode Only After Teacher-Forced Success

Only run free decoding for components that pass G1/G2.

Use:

```text
scripts/run_generation_target_residual_patched_decode_hue.py
```

Success condition:

```text
top component improves hue distance and exact HSV over corrupt decode,
while wrong/object/random controls fail.
```

## 7. Recommended New Utility Script

A single ranking script would reduce repeated notebook work:

```text
scripts/run_attention_evidence_head_ranking.py
```

Suggested arguments:

```text
--task understanding|generation
--cfg PATH
--manifest PATH
--target-cache-dir PATH
--layers all|60,61,62
--query-scope post-image-text-prompt|score-positions|visual-score-positions
--source-scope visual|gt-region|eoi|color-token|object-token|prompt
--attn-implementation eager
--out-dir PATH
```

Output:

```text
head_scores.csv
head_scores_topk.json
attention_maps/
ranking_report.md
```

Columns:

```text
task
layer
head
query_scope
source_scope
evidence_mass
image_mass
region_ratio
spatial_entropy
concentration
selection_frequency
write_score
patch_recovery_if_available
rank_score
```

## 8. Final Working Claims To Test

### Understanding Claim

Likely testable and strong:

```text
Emu3.5 hue understanding uses a late-layer image-conditioned text-side route.
L62 heads 36/38/46 are a causal, understanding-selective head subset under current clean/corrupt controls.
```

Need next evidence:

```text
QR-style evidence ranking plus clean ablation/knockout necessity.
```

### Generation Claim

Likely testable but should remain narrower:

```text
Emu3.5 hue generation has compact prompt color-token boundary control and target-side visual-token likelihood causal components.
The internal generation route is currently better described as residual / feature-subspace structured than sparse-head or sparse-neuron structured.
```

Need next evidence:

```text
attribute-focused visual-score -> color-token attention ranking,
edge knockout,
write/logit attribution,
and decoded controls for any candidate internal component.
```

## 9. Stop Conditions

Use these to avoid overclaiming.

```text
If attention-ranked heads fail patch/ablation:
  attention is only a candidate signal, not a path.

If top-k heads work only in teacher-forced NLL but fail decoded controls:
  claim target-likelihood component, not decoded generation circuit.

If top-k neurons remain weak while PCA/SAE directions work:
  claim distributed feature subspace, not sparse neurons.

If generation heads also help understanding:
  claim shared prompt-side influence, not generation-specific heads.

If full residual works everywhere from input embedding to late layers:
  treat it as trajectory overwrite upper bound, not local circuit evidence.
```
