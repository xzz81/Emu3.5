# Emu3.5 Generation / Understanding Pathway Final Report

## 1. Research Question

本研究从原始问题：

```text
Emu3.5 在执行图像生成或图像理解任务时，是否会激活特定神经元？
```

收敛为更严格的因果问题：

```text
Emu3.5 在 generation 与 understanding 中出现的任务差异性激活，
是否能被推进为 clean/corrupt recovery 与 cross-task specificity 证据？
```

由于 Emu3.5 是 decoder-only unified multimodal model，本研究关注同一 Transformer 主干内的：

```text
residual stream
self-attention heads
MLP modules
MLP intermediate neurons / feature directions
token-to-token information flow
```

## 2. Evidence Ladder

所有 claim 按四级证据推进：

| stage | evidence | allowed claim |
|---|---|---|
| A | activation contrast | 有任务差异性候选 |
| B | ablation / likelihood effect | 组件影响输出 likelihood |
| C | clean/corrupt patch recovery | 组件参与 causal path |
| D | cross-task control | 组件呈现任务选择性 |

本报告只把通过 clean/corrupt recovery 或 cross-task control 的结果作为主要证据。

## 3. Understanding Pathway

### 3.1 Strong Residual Route

Understanding 侧最强证据来自 answer score-position residual patch：

```text
L61/L62/L63 answer score-position full residual patch ~= full recovery
```

解释：

```text
image-to-answer 信息确实集中到 late residual stream 的 answer score positions。
```

### 3.2 Mediated Route

注意力 knockout 显示：

```text
answer score -> image/EOI direct attention edge weak or negative
```

mediated residual patch 显示：

```text
post-image question/text residual states recover much of the gap
```

解释：

```text
understanding 不是单一 direct image attention gate，
而是 image information 经 post-image text/question residual states 中介到 answer score position。
```

### 3.3 Head-Level Specificity

Understanding 侧最清晰的局部组件是：

```text
L62 heads 36/38/46
```

关键 cross-task 结果：

| component | S_under | S_gen | reading |
|---|---:|---:|---|
| L62 heads 36/38/46 | +0.152832 | +0.001438 | understanding-specific head-level route |

解释：

```text
L62 heads 36/38/46 对 understanding answer likelihood 有明显恢复，
但对 generation prompt-side raw-target likelihood 近零。
```

### 3.4 MLP Neuron Boundary

Understanding 侧 MLP modules 有贡献，但当前 top-k neuron patch 没有稳定恢复到 module/residual 级别。

因此当前不能写：

```text
Emu3.5 存在 sparse understanding neurons。
```

更稳妥写法：

```text
Understanding route is late-residual and partly head-localized;
MLP contribution appears distributed or feature-level rather than sparse-neuron-level.
```

## 4. Generation Pathway

### 4.1 Benchmark Boundary

在线 decoded generation benchmark 成本高且不稳定：

```text
CFG seed sweep slow;
red/blue blue-side generation remained unreliable;
decoded-image robust benchmark not yet established across pairs.
```

因此 generation 侧主证据来自：

```text
retrospective highcontrast raw-target benchmark
teacher-forced visual-token likelihood
HSV-validated old decoded images
single-direction decoded validation for full L61 target-side residual patching
4-direction decoded replication for full L61 target-side residual patching
wrong-target decoded control for full L61 target-side residual patching
random/other-target decoded control for full L61 target-side residual patching
decoded L61 MLP-module patching
decoded L61 PCA200 patching
decoded L61 top10 attention-head patching
decoded L61 bottom10 attention-head control
decoded L61 top10 attention-head + MLP patching
decoded L61 self-attn + MLP patching
decoded L60-L62 full residual band patching
decoded L60-L62 self-attn + MLP band control
decoded L59-L63 full residual boundary patching
decoded L56-L63 full residual wider-boundary patching
decoded L48/L52 sampled full residual earlier-boundary patching
decoded L32/L40 sampled full residual mid-layer boundary patching
decoded L8/L16 sampled full residual early-layer boundary patching
decoded L0 sampled full residual boundary patching
decoded L0 wrong/random target controls
decoded input-embedding boundary wrong/random target controls
teacher-forced input-embedding decomposition
decoded color-token input-embedding patching
decoded object-token input-embedding control
```

边界：

```text
当前 generation evidence 主要仍是 raw-target visual-token likelihood evidence；
Stage 24/25 增加了 full L61 target-side residual 的 decoded positive / replication，
Stage 26/27 增加了 wrong-target 与 random/other-target negative controls，
Stage 28 显示 L61 MLP module 不能复现 full residual decoded restoration，
Stage 29 显示 L61 PCA200 也不能复现 full residual decoded restoration，
Stage 30 显示 L61 top10 attention heads 有部分 decoded effect，但仍不能复现 full residual decoded restoration，
Stage 31 bottom10 attention-head control 支持弱 top-vs-bottom decoded separation，
Stage 32 显示 top10 attention heads + MLP module 组合也不能复现 full residual decoded restoration，
Stage 33 显示 full self-attn + MLP module output 组合有更强连续 hue 改善但 exact restoration 仍不足，
Stage 34 显示 L60/L61/L62 full residual patching 均可复现 decoded restoration，
Stage 35 显示 L60/L61/L62 self-attn+MLP module-output patching 不形成同样 band，
Stage 36 显示 full residual band 至少延伸到 L59-L63，
Stage 37 显示 full residual band 进一步延伸到 L56-L63，
Stage 38 显示 sampled L48/L52 也有同样 full-residual restoration，
Stage 39 显示 sampled L32/L40 仍有同样 full-residual restoration，
Stage 40 显示 sampled L8/L16 也有同样 full-residual restoration，
Stage 41 显示 sampled L0 已经有同样 full-residual restoration，
Stage 42 显示 L0 wrong/random controls 不复现 exact restoration，
Stage 43 显示 input-embedding target-side patch 已经复现同级 clean-source-specific restoration，
Stage 44 显示 teacher-forced target visual-score input embeddings 无恢复，而 prompt color-token input embeddings 几乎完整恢复 raw-target likelihood，
Stage 45 显示 free decoded color-token input-embedding patching 有 2/4 exact restoration 与 4/4 hue-distance improvement，
Stage 46 显示 free decoded object-token input-embedding patching 为 0/4 exact、0/4 improvement，
但还不是 compact decoded-image prompt-to-image circuit evidence。
```

### 4.2 Target-Side Residual Upper Bound

Generation 侧最强证据来自 target-side visual-score residual：

| component | recovery |
|---|---:|
| L60 visual-score full residual | +0.985674 |
| L61 visual-score full residual | +0.999021 |
| L62 visual-score full residual | +0.998530 |

解释：

```text
target-side visual score residual 几乎完整携带 clean/corrupt visual-token likelihood gap。
```

### 4.2.1 Stage 24 Decoded Validation

Stage 24/25 tested whether the strongest target-side residual intervention transfers to free decoded images.

Artifact:

```text
outputs/stage24_decoded_generation_patch_validation.md
outputs/stage25_decoded_generation_patch_replication.md
outputs/stage26_decoded_generation_wrong_target_control.md
outputs/stage27_decoded_generation_random_target_control.md
outputs/stage28_decoded_generation_mlp_module_patch.md
outputs/stage29_decoded_generation_pca200_patch.md
outputs/stage30_decoded_generation_attention_top10_patch.md
outputs/stage31_decoded_generation_attention_bottom10_control.md
outputs/stage32_decoded_generation_attention_top10_plus_mlp_patch.md
outputs/stage33_decoded_generation_self_attn_plus_mlp_patch.md
outputs/stage34_decoded_generation_late_residual_band.md
outputs/stage35_decoded_generation_self_attn_mlp_band_control.md
outputs/stage36_decoded_generation_residual_band_boundary.md
outputs/stage37_decoded_generation_residual_band_wider_boundary.md
outputs/stage38_decoded_generation_earlier_residual_boundary_sampling.md
outputs/stage39_decoded_generation_mid_residual_boundary_sampling.md
outputs/stage40_decoded_generation_early_residual_boundary_sampling.md
outputs/stage41_decoded_generation_l0_residual_boundary_sampling.md
outputs/stage42_decoded_generation_l0_wrong_random_control.md
outputs/stage43_decoded_generation_input_embedding_boundary.md
outputs/stage44_generation_input_embedding_teacherforced_decomposition.md
outputs/stage45_decoded_generation_color_token_input_embedding_patch.md
outputs/stage46_decoded_generation_object_token_input_embedding_control.md
```

Single-direction result:

| variant | decoded color | hue match |
|---|---|---:|
| clean cached target | cyan | 1 |
| corrupt free decode | green | 0 |
| patched corrupt free decode | cyan | 1 |

Patch:

```text
layer = 61
patch_scope = visual-score-positions
patched visual-score positions = 1024
```

Interpretation:

```text
This is preliminary decoded-image support for the full L61 target-side residual route.
It does not prove sparse neurons or compact subspace decoded control.
```

Stage 25 replication:

| metric | result |
|---|---:|
| validated directions | 4 |
| clean cached target exact HSV matches | 4/4 |
| corrupt decode exact HSV matches | 0/4 |
| patched corrupt exact HSV matches | 3/4 |
| patched corrupt hue-distance improvement | 4/4 |

Stage 26 wrong-target control:

| metric | result |
|---|---:|
| wrong-target patched exact HSV matches | 0/4 |
| wrong-target hue-distance improvement | 0/4 |

Stage 27 random/other-target control:

| metric | result |
|---|---:|
| random-target patched exact HSV matches | 0/4 |

Stage 28 MLP-module patch:

| metric | result |
|---|---:|
| L61 MLP-module patched exact HSV matches | 0/4 |
| L61 MLP-module hue-distance improvement | 1/4 |

Stage 29 PCA200 patch:

| metric | result |
|---|---:|
| L61 PCA200 patched exact HSV matches | 0/4 |
| L61 PCA200 hue-distance improvement | 0/4 |

Stage 30 top10 attention-head patch:

| metric | result |
|---|---:|
| L61 top10-head patched exact HSV matches | 1/4 |
| L61 top10-head hue-distance improvement | 2/4 |

Stage 31 bottom10 attention-head control:

| metric | result |
|---|---:|
| L61 bottom10-head patched exact HSV matches | 0/4 |
| L61 bottom10-head hue-distance improvement | 1/4 |

Stage 32 top10 attention-head + MLP patch:

| metric | result |
|---|---:|
| L61 top10-head + MLP patched exact HSV matches | 0/4 |
| L61 top10-head + MLP hue-distance improvement | 1/4 |

Stage 33 self-attn + MLP patch:

| metric | result |
|---|---:|
| L61 self-attn + MLP patched exact HSV matches | 1/4 |
| L61 self-attn + MLP hue-distance improvement | 3/4 |

Stage 34 late residual band:

| layer | exact HSV matches | hue-distance improvement |
|---:|---:|---:|
| L60 full residual | 3/4 | 4/4 |
| L61 full residual | 3/4 | 4/4 |
| L62 full residual | 3/4 | 4/4 |

Stage 35 self-attn + MLP band control:

| layer | exact HSV matches | hue-distance improvement |
|---:|---:|---:|
| L60 self-attn + MLP | 0/4 | 1/4 |
| L61 self-attn + MLP | 1/4 | 3/4 |
| L62 self-attn + MLP | 0/4 | 1/4 |

Stage 36 residual-band boundary:

| layer | exact HSV matches | hue-distance improvement |
|---:|---:|---:|
| L59 full residual | 3/4 | 4/4 |
| L60 full residual | 3/4 | 4/4 |
| L61 full residual | 3/4 | 4/4 |
| L62 full residual | 3/4 | 4/4 |
| L63 full residual | 3/4 | 4/4 |

Stage 37 wider residual-band boundary:

| layer | exact HSV matches | hue-distance improvement |
|---:|---:|---:|
| L56 full residual | 3/4 | 4/4 |
| L57 full residual | 3/4 | 4/4 |
| L58 full residual | 3/4 | 4/4 |
| L59 full residual | 3/4 | 4/4 |
| L60 full residual | 3/4 | 4/4 |
| L61 full residual | 3/4 | 4/4 |
| L62 full residual | 3/4 | 4/4 |
| L63 full residual | 3/4 | 4/4 |

Stage 38 earlier residual-boundary sampling:

| layer | tested status | exact HSV matches | hue-distance improvement |
|---:|---|---:|---:|
| L48 full residual | sampled | 3/4 | 4/4 |
| L52 full residual | sampled | 3/4 | 4/4 |
| L56-L63 full residual | continuous tested band | each 3/4 | each 4/4 |

Stage 39 mid-layer residual-boundary sampling:

| layer | tested status | exact HSV matches | hue-distance improvement |
|---:|---|---:|---:|
| L32 full residual | sampled | 3/4 | 4/4 |
| L40 full residual | sampled | 3/4 | 4/4 |
| L48/L52 full residual | sampled | each 3/4 | each 4/4 |
| L56-L63 full residual | continuous tested band | each 3/4 | each 4/4 |

Stage 40 early residual-boundary sampling:

| layer | tested status | exact HSV matches | hue-distance improvement |
|---:|---|---:|---:|
| L8 full residual | sampled | 3/4 | 4/4 |
| L16 full residual | sampled | 3/4 | 4/4 |
| L32/L40/L48/L52 full residual | sampled | each 3/4 | each 4/4 |
| L56-L63 full residual | continuous tested band | each 3/4 | each 4/4 |

Stage 41 L0 residual-boundary sampling:

| layer | tested status | exact HSV matches | hue-distance improvement |
|---:|---|---:|---:|
| L0 full residual | sampled | 3/4 | 4/4 |
| L8/L16/L32/L40/L48/L52 full residual | sampled | each 3/4 | each 4/4 |
| L56-L63 full residual | continuous tested band | each 3/4 | each 4/4 |

Stage 42 L0 wrong/random controls:

| variant | exact HSV matches | hue-distance improvement |
|---|---:|---:|
| L0 clean-target full residual | 3/4 | 4/4 |
| L0 wrong-target full residual | 0/4 | 0/4 |
| L0 random-target full residual | 0/4 | 2/4 |

Stage 43 input-embedding boundary controls:

| variant | exact HSV matches | hue-distance improvement |
|---|---:|---:|
| clean-target input embedding | 3/4 | 4/4 |
| wrong-target input embedding | 0/4 | 0/4 |
| random-target input embedding | 0/4 | 2/4 |

Stage 44 teacher-forced input-embedding decomposition:

| patch scope | directions | mean recovery | helped directions | mean patched embedding delta |
|---|---:|---:|---:|---:|
| visual-score positions | 4 | +0.000000 | 0/4 | 0.000000 |
| color-token | 4 | +1.012833 | 4/4 | 0.011765 |
| prompt | 4 | +0.949991 | 4/4 | 0.002468 |

Stage 45 decoded color-token input-embedding patching:

| variant | exact HSV matches | hue-distance improvement | patched positions |
|---|---:|---:|---:|
| clean cached target | 4/4 | n/a | n/a |
| corrupt decode | 0/4 | n/a | n/a |
| color-token input embedding | 2/4 | 4/4 | 1-2 prompt tokens |

Stage 46 decoded object-token input-embedding control:

| variant | exact HSV matches | hue-distance improvement | patched positions |
|---|---:|---:|---:|
| object-token input embedding | 0/4 | 0/4 | 1 prompt token |

Updated interpretation:

```text
Full L61 target-side residual patching has replicated decoded restoration over a small highcontrast benchmark:
exact HSV restoration is 3/4 and continuous hue-distance improvement is 4/4.
Wrong-target residual patching does not restore clean hue/object, giving 0/4 exact restoration and 0/4 hue-distance improvement.
Random/other-target residual patching also gives 0/4 exact restoration, though continuous hue-distance can move closer when the random source color is nearby on the hue wheel.
L61 MLP-module patching gives 0/4 exact restoration, so the decoded restoration does not yet localize to the strongest teacher-forced single module.
L61 PCA200 patching gives 0/4 exact restoration and 0/4 hue-distance improvement, so the decoded restoration also does not localize to the strongest compact teacher-forced MLP subspace.
L61 top10 attention-head patching gives 1/4 exact restoration and 2/4 hue-distance improvement, which is a partial decoded-active component but still far below full residual restoration.
L61 bottom10 attention-head patching gives 0/4 exact restoration and 1/4 hue-distance improvement, supporting a weak top-vs-bottom decoded separation.
L61 top10 attention-head + MLP patching gives 0/4 exact restoration and 1/4 hue-distance improvement, so simple component addition does not recover the decoded trajectory.
L61 self-attn + MLP patching gives 1/4 exact restoration and 3/4 hue-distance improvement, making it the strongest tested component combination but still below full residual restoration.
L60/L61/L62 full residual patching gives the same 3/4 exact restoration and 4/4 hue-distance improvement, so the intervention is best described as late residual-band target-trajectory restoration, not compact circuit localization.
L60/L61/L62 self-attn+MLP patching does not show the same band behavior, so explicit same-layer module outputs do not explain the residual-band route.
L59/L60/L61/L62/L63 full residual patching all show the same restoration pattern, so the current residual-band boundary has not been found.
L56/L57/L58/L59/L60/L61/L62/L63 full residual patching all show the same restoration pattern, so the route is now best described as a broad late target-side residual trajectory rather than a narrow final-layer band.
Sampled L48 and L52 full residual patching also show the same restoration pattern, so the current tested evidence extends beyond the continuous L56-L63 band; however, not every layer from L48 to L63 has been tested.
Sampled L32 and L40 full residual patching also show the same restoration pattern, so the decoded effect is already present at sampled middle layers; however, not every layer from L32 to L63 has been tested.
Sampled L8 and L16 full residual patching also show the same restoration pattern, so the decoded effect is already present at sampled early layers; however, not every layer from L8 to L63 has been tested.
Sampled L0 full residual patching also shows the same restoration pattern, so the decoded full-state result is best treated as a target trajectory overwrite upper bound rather than a localized layer route.
L0 wrong-target and random-target controls do not reproduce exact restoration, so the L0 overwrite result is clean-source-specific rather than arbitrary target-state injection.
Input-embedding target-side patching also shows 3/4 exact restoration and 4/4 hue-distance improvement with wrong-target 0/4 exact and 0/4 improvement; therefore the decoded restoration is already present at the target-side input embedding boundary.
Because the input-embedding boundary is positive, the decoded full-state generation result should be framed as clean-source-specific target trajectory overwrite evidence, not as a localized residual-layer circuit.
Teacher-forced target visual-score input-embedding patching has zero recovery and zero clean/corrupt embedding delta because the raw-target scoring run uses the same clean target token sequence.
Teacher-forced color-token input-embedding patching recovers nearly the full clean/corrupt visual-token likelihood gap, so color conditioning is already available at the prompt color-token input boundary under raw-target scoring.
This separates decoded target-trajectory overwrite from teacher-forced prompt color-token conditioning.
Decoded color-token input-embedding patching restores exact hue in 2/4 directions and improves hue distance in 4/4 directions while patching only 1-2 prompt color-token embeddings once at the initial prompt forward.
Object-token input-embedding patching gives 0/4 exact restoration and 0/4 hue-distance improvement, supporting color-token specificity relative to a non-color prompt-token input control.
This is the strongest compact decoded prompt-side intervention so far, but it still requires random-color/wrong-color controls before it can be treated as robust decoded prompt-side color control.
```

### 4.3 Prompt-Side Weakness

Prompt-side recovery 较弱：

| component | recovery |
|---|---:|
| highcontrast prompt residual L60 | +0.048844 |
| highcontrast prompt residual L61 | +0.045733 |
| L60 prompt top10 heads | +0.040162 |

cross-task control：

| component | S_gen | S_under |
|---|---:|---:|
| L60 prompt top10 heads | +0.040162 | +0.020725 |

解释：

```text
prompt-side generation heads are weak and partially shared;
they should not be framed as generation-specific.
```

### 4.4 Target-Side Attention Heads

L61 visual-score attention heads show a clearer target-side subset:

| component | recovery |
|---|---:|
| L61 visual-score top10 heads | +0.101426 |
| L61 visual-score bottom10 heads | -0.037579 |
| L61 visual-score self-attn module | +0.083526 |
| L61 visual-score full residual | +0.999021 |

cross-task control:

| component | S_gen | S_under |
|---|---:|---:|
| L61 visual-score top10 heads | +0.101426 | -0.001324 |

解释：

```text
L61 target-side top heads are generation-positive and understanding-near-zero.
But they explain only about 10% of the generation likelihood gap,
so they are not a complete generation circuit.
```

### 4.5 Target-Side MLP Neurons

L61 visual-score MLP module is the strongest single module:

```text
L61 visual-score MLP module: +0.246532
```

Neuron patch ladder:

| component | recovery |
|---|---:|
| top10 MLP neurons | +0.036203 |
| top50 MLP neurons | +0.045597 |
| top200 MLP neurons | +0.073528 |
| top1000 MLP neurons | +0.127074 |
| full MLP module | +0.246532 |

cross-task control:

| component | S_gen | S_under |
|---|---:|---:|
| top10 MLP neurons | +0.036203 | +0.000541 |
| top1000 MLP neurons | +0.127074 | +0.001096 |

解释：

```text
L61 visual-score MLP neurons are rankable and target-side generation-specific,
but useful recovery requires hundreds to thousands of neurons.
This is not sparse generation-neuron evidence.
```

### 4.6 Target-Side MLP Feature Subspace

PCA over clean/corrupt L61 MLP intermediate deltas gives a more compressed feature-subspace view:

| component | recovery |
|---|---:|
| random50 | +0.048127 |
| PCA50 | +0.108716 |
| random200 | +0.036415 |
| PCA200 | +0.169549 |
| top1000 MLP neurons | +0.127074 |
| full MLP module | +0.246532 |

cross-task control:

| component | S_gen | S_under |
|---|---:|---:|
| PCA50 | +0.108716 | -0.000160 |
| PCA200 | +0.169549 | +0.001142 |

解释：

```text
L61 visual-score MLP signal has a structured feature subspace.
PCA200 is the strongest compact generation target-side teacher-forced component so far,
but it still does not reach full MLP module or full residual recovery.
```

## 5. Differential Component Matrix

The consolidated matrix is stored at:

```text
outputs/stage21_differential_component_matrix.csv
outputs/stage21_differential_component_matrix_report.md
```

Core matrix:

| component | S_gen | S_under | classification |
|---|---:|---:|---|
| Understanding L62 heads 36/38/46 | +0.001438 | +0.152832 | understanding-specific |
| Generation L60 prompt top10 heads | +0.040162 | +0.020725 | weak/shared |
| Generation L61 visual-score top10 heads | +0.101426 | -0.001324 | generation target-side specific |
| Generation L61 visual-score top1000 MLP neurons | +0.127074 | +0.001096 | generation target-side feature mass |
| Generation L61 visual-score PCA200 directions | +0.169549 | +0.001142 | generation target-side feature subspace |

## 6. Current Claims

### Supported

```text
Emu3.5 shows task-differentiated causal pathways in the shared decoder backbone.
```

```text
Understanding has a strong late-layer image-to-answer route,
carried by answer-score residual stream and partly localized to L62 heads 36/38/46.
```

```text
Generation has target-side visual-token-likelihood components at L61 visual-score positions.
The strongest compact teacher-forced component so far is L61 visual-score PCA200 MLP directions.
```

```text
Generation target-side heads, top1000 MLP neurons, and PCA200 directions are near-zero on understanding score-position controls.
```

### Partially Supported

```text
Generation and understanding paths are differentiable under the current likelihood-based controls.
```

But this should be qualified:

```text
understanding specificity is answer-score/head-level;
generation specificity is target-side visual-token-likelihood/feature-subspace-level.
```

### Not Supported

```text
sparse generation-selective neurons
sparse understanding-selective neurons
decoded-image robust prompt-to-image generation circuit
complete separation of generation and understanding circuits
small attention-head set explaining the full generation route
color-token-specific generation circuit
```

## 7. Paper-Ready Summary

Short version:

```text
We find asymmetric causal evidence for task-differentiated pathways in Emu3.5.
The understanding pathway is late-residual and partially head-localized at L62 answer-relevant states.
The generation pathway is strongest at target-side visual-token score states, where L61 attention heads and MLP feature subspaces recover visual-token likelihood gaps and show near-zero transfer to understanding controls.
Stage 24-27 add replicated decoded support plus wrong-target and random/other-target controls for full L61 target-side residual patching; Stage 28 shows L61 MLP module alone is insufficient; Stage 29 shows L61 PCA200 is also insufficient; Stage 30 shows L61 top10 attention heads are partially decoded-active but incomplete; Stage 31 adds a bottom10 decoded control supporting weak top-vs-bottom separation; Stage 32 shows top10+MLP combination still fails; Stage 33 shows self-attn+MLP is stronger but still incomplete; Stage 34 shows the full-residual effect spans L60-L62; Stage 35 shows self-attn+MLP does not form that band; Stage 36 extends full-residual restoration to L59-L63; Stage 37 extends it continuously to L56-L63; Stage 38 shows sampled L48/L52 are also positive; Stage 39 shows sampled L32/L40 are also positive; Stage 40 shows sampled L8/L16 are also positive; Stage 41 shows sampled L0 is already positive; Stage 42 shows L0 wrong/random controls do not reproduce exact restoration; Stage 43 shows the same clean-source-specific pattern already at the target-side input-embedding boundary; Stage 44 shows teacher-forced target-side input embeddings have zero recovery while color-token input embeddings nearly fully recover raw-target likelihood; Stage 45 shows decoded color-token input embeddings are a compact prompt-side intervention with 2/4 exact and 4/4 hue-distance improvement; Stage 46 shows object-token input embeddings are a negative decoded control. Thus full-state decoded patching remains best framed as clean-source-specific target trajectory overwrite upper-bound evidence, while compact decoded prompt-side color control is now partially supported and object-token controlled.
```

One-line conclusion:

```text
Understanding specificity is currently answer-score/head-level;
generation specificity is currently target-side visual-token-likelihood/feature-subspace-level,
with replicated decoded support only for full target-side residual patching.
The latest boundary test shows this full-state decoded support already applies at the target-side input-embedding boundary, strengthening the overwrite-upper-bound interpretation.
Teacher-forced input-boundary decomposition shows prompt color-token embeddings can recover raw-target likelihood.
Decoded color-token input-embedding patching now bridges this to partial compact prompt-side decoded support: 2/4 exact and 4/4 improved.
```

## 8. Next Evidence Gates

To strengthen the paper:

1. Test combined attention/MLP/path patching if generation-side compact decoded localization remains important.
3. Add supervised feature directions or CCA-style directions for L61 visual-score MLP.
4. Extend understanding-side MLP decomposition with feature directions rather than sparse neuron sets.
5. Keep all neuron-level claims behind random/bottom controls and cross-task controls.

Stop condition for sparse-neuron claims:

```text
If top-k neurons do not beat random/bottom controls and approach module-level recovery,
the paper should use feature mass / feature subspace language, not neuron language.
```
