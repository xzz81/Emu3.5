# Emu3.5 UMM Generation / Understanding Path 寻找方案

## 1. 核心结论

我们不应该直接从 attention heatmap 声称找到了 path。更合理的路线是：

```text
annotated evidence
-> attention / evidence score
-> sparse candidate heads or edges
-> ablation / activation patching
-> cross-task controls
-> path-level claim
```

这条路线来自 QRHead、Retrieval Head、VLM grounding heads 和 Transformer circuit 工作的共同方法论。

对当前 Emu3.5 项目，最稳妥的判断是：

| 方向 | 当前最可信结论 | 还不能说什么 |
|---|---|---|
| Understanding path | 有明确 late residual route；信息经 post-image question/text states 到 answer score positions；L62 heads 36/38/46 是目前最清晰的 head-level 组件 | 不能说已经找到 sparse understanding neurons |
| Generation path | 有强 target-side visual-score residual / feature-subspace evidence；prompt color-token input embedding 有 compact decoded support | 不能说已经找到 compact generation circuit 或 sparse generation heads/neurons |

因此，下一阶段的目标不是重新发明一套实验，而是在现有 patch-recovery pipeline 前面加一个更干净的 QRHead-style candidate ranking。

## 2. 方法原则

### 2.1 Attention 只用于产生候选

Attention 可以告诉我们“哪些 token 之间可能有信息流”，但它不是解释本身。

必须继续验证：

```text
高 attention 的 head / edge 是否必要？
clean activation patch 到 corrupt run 是否恢复输出？
corrupt activation patch 到 clean run 是否破坏输出？
wrong / random / object-token / bottom-head controls 是否失败？
跨任务 control 是否接近 0？
```

### 2.2 用 full residual 作为 upper bound

已有实验显示：

```text
understanding L61-L63 answer-score full residual patch ≈ full recovery
generation L61 visual-score full residual patch ≈ full recovery
```

所以后续组件或 path 的作用应该报告为：

```text
component recovery / full residual recovery
```

这能避免把一个只解释 10% recovery 的 head set 误写成完整通路。

### 2.3 区分三类 claim

| 证据 | 可以怎么写 | 不能怎么写 |
|---|---|---|
| attention / activation contrast | candidate component | causal path |
| ablation 或 patch 有效 | causal component | task-selective path |
| patch 有效且 controls 失败 | selective causal component | full circuit |
| 能解释 full residual 的主要比例 | partial mechanistic path | complete separation |

## 3. Understanding Path 方案

### 3.1 路径假设

当前最合理的 understanding path 是：

```text
image tokens / EOI
-> post-image question/text residual states
-> answer score positions
-> answer-token logits
```

注意：直接 `answer score -> image token` 的 attention edge 可能较弱，因此不要只盯 answer token 到 image token 的 raw attention。

### 3.2 QRHead-style head ranking

构造或复用已有 hue-control image-read 数据：

```text
image: red circle
question: What color is the circle? Answer with one word.
answer: red
```

定义：

```text
Q_U = post-image question/text tokens 或 answer score positions
E_U = visual tokens / EOI / GT object-region tokens
```

对每个 head 计算：

```text
EvidenceMass(h) =
  mean_{q in Q_U} sum_{e in E_U} Attention_h[q, e]
```

如果有 bbox / mask / patch region，还应计算：

```text
ImageMass(h)
RegionRatio(h) = RegionMass / ImageMass
SpatialEntropy(h)
SelectionFrequency(h)
```

推荐 ranking：

```text
U_score(h) =
  z(EvidenceMass)
  + z(RegionRatio)
  + z(1 - SpatialEntropy)
  + z(SelectionFrequency)
```

### 3.3 因果验证

优先验证已有强候选：

```text
L62 heads 36/38/46
L62 top7 heads
same-layer random heads
bottom / negative heads such as 35 and 32
```

用现有脚本：

```text
scripts/run_understanding_attention_head_patch_recovery.py
scripts/run_understanding_attention_knockout.py
scripts/run_understanding_residual_patch_recovery.py
scripts/run_understanding_module_patch_recovery.py
```

关键实验：

```text
1. clean -> corrupt head-output patch
2. clean run head ablation / knockout
3. corrupt -> clean reverse patch
4. random / bottom head controls
5. generation cross-task control
```

成功标准：

```text
S_under(top_U) > random / bottom controls
S_gen(top_U) ≈ 0
多组 color / shape pairs 上稳定
```

## 4. Generation Path 方案

Generation 需要拆成两个层次。

### 4.1 Prompt-conditioning path

这是最值得寻找的 compact path：

```text
prompt color token
-> internal prompt / residual / attention states
-> visual-score positions
-> visual-token logits
-> decoded hue
```

已有 Stage 45/46 支持 input-boundary 证据：

| intervention | exact HSV | hue-distance improvement |
|---|---:|---:|
| color-token input embedding patch | 2/4 | 4/4 |
| object-token input embedding control | 0/4 | 0/4 |

这说明 color token 对 free generation hue 有因果影响，但还没有说明内部 compact circuit 在哪里。

### 4.2 Target-side trajectory path

已有 generation full residual patch 很强：

```text
L61 visual-score full residual patch ≈ full recovery
```

但这更像 target trajectory overwrite upper bound，不能直接当作 compact generation circuit。

所以它的用途是：

```text
作为 upper bound
作为 component completeness denominator
作为 decoded validation 的强阳性对照
```

### 4.3 Attribute-focused generation head ranking

对 generation，decoder-only 顺序决定了 query 应该是 target-side visual-score positions：

```text
Q_G = visual-score positions
E_G_color = prompt color-token positions
E_G_object = prompt object-token positions
E_G_prompt = all semantic prompt tokens
```

颜色 attention score：

```text
G_color_attn(h) =
  mean_{q in Q_G} sum_{s in E_G_color} Attention_h[q, s]
```

颜色特异性：

```text
G_color_specific(h) =
  G_color_attn(h)
  - G_object_attn(h)
  - G_null_attn(h)
```

只看 attention 仍然不够。还要看这个 head 写入了什么：

```text
WriteScore(h) =
  contribution to clean visual-token logit
  or contribution to clean-vs-corrupt logit margin
```

最终候选分数：

```text
G_candidate(h) =
  z(G_color_specific)
  + z(WriteScore)
  + z(clean-to-corrupt patch recovery)
```

### 4.4 Generation 因果验证

用现有脚本：

```text
scripts/run_generation_attention_head_patch_recovery.py
scripts/run_generation_residual_patch_recovery.py
scripts/run_generation_module_patch_recovery.py
scripts/run_generation_mlp_lowrank_patch_recovery.py
scripts/run_generation_target_residual_patched_decode_hue.py
```

teacher-forced 阶段必须做：

```text
1. top G heads clean -> corrupt patch
2. clean run ablation / mean replacement
3. corrupt -> clean reverse patch
4. random / bottom / attention-matched low-write controls
5. object-token / wrong-color controls
6. understanding cross-task controls
```

只有 teacher-forced 通过后，才做 decoded validation：

```text
1. free corrupt-prompt decode 是否 hue-distance improved
2. exact HSV 是否恢复
3. wrong-target source 是否失败
4. object-token / random-color-token controls 是否失败
5. image quality / format 是否没有整体崩坏
```

## 5. 最小实验集

### U1: Understanding QR-style head census

目标：验证 evidence-ranked heads 是否复现或解释 L62 heads 36/38/46。

```text
task = understanding
layers = 56-63 first, then all layers
query scopes = post-image-text-prompt, score-positions
source scopes = visual tokens, EOI, optional GT region tokens
output = head_scores_understanding.csv
```

成功标准：

```text
top-ranked heads overlap with L62 36/38/46
or attention ranking 与 patch recovery ranking 的差异可解释
```

### U2: L62 heads necessity

目标：证明 L62 heads 36/38/46 不只是 sufficient patch component，也有 necessity。

```text
test heads = 36/38/46, top7, random3/random7, bottom heads 35/32
metric = clean answer NLL / logit margin / answer flip rate
```

成功标准：

```text
ablating top heads hurts clean understanding more than controls
```

### G1: Generation attribute-focused head census

目标：找到 visual-score positions 是否会通过特定 heads 读取 prompt color token。

```text
task = generation
cfg = configs/ume_main_t2i_counterfactual_pairs_highcontrast_seed68.py
target cache = outputs/generation_raw_target_cache_highcontrast_seed68
query = visual-score positions
source = color token, object token, prompt tokens
layers = 0,8,16,32,40,48,52,56-63 first
output = head_scores_generation.csv
```

成功标准：

```text
high color-specific attention heads also have positive write score and patch recovery
```

如果失败，也很有价值：

```text
说明 generation 不是 QRHead-like sparse head route，
应转向 color-token boundary + MLP feature subspace。
```

### G2: Generation color-token edge knockout

目标：验证 `color-token -> visual-score positions` 这条 edge 是否必要。

```text
query = visual-score positions
source = color-token positions
heads = G1 top heads and controls
intervention = add -inf attention mask on selected edges
metric = clean target visual-token NLL / logit margin
```

成功标准：

```text
masking color-token edges hurts clean generation score
more than object-token / random-token edges
```

### G3: Decoded validation

目标：只对 teacher-forced 成功的组件做 free decode。

```text
script = scripts/run_generation_target_residual_patched_decode_hue.py
metric = exact HSV, hue-distance improvement, decoded validity
controls = wrong-target, object-token, random-color-token
```

成功标准：

```text
top component improves decoded hue,
controls fail,
image quality / structure does not collapse
```

## 6. 建议新增脚本

建议新增一个统一 ranking 脚本：

```text
scripts/run_attention_evidence_head_ranking.py
```

参数设计：

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

输出：

```text
head_scores.csv
head_scores_topk.json
attention_maps/
ranking_report.md
```

核心列：

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

## 7. 最终可检验表述

### Understanding

当前可以优先验证的表述：

```text
Emu3.5 hue understanding uses a late-layer image-conditioned text-side route.
L62 heads 36/38/46 are causal, understanding-selective head-level components
under current clean/corrupt controls.
```

还需要补：

```text
QR-style evidence ranking
clean ablation / knockout necessity
more region-level evidence if available
```

### Generation

当前更稳妥的表述：

```text
Emu3.5 hue generation has compact prompt color-token boundary control
and target-side visual-token-likelihood causal components.
The internal generation route is currently better described as residual /
feature-subspace structured rather than sparse-head or sparse-neuron structured.
```

还需要补：

```text
attribute-focused visual-score -> color-token attention ranking
edge knockout
write / logit attribution
decoded controls for any candidate internal component
```

## 8. 停机条件

为了避免过度解释，建议提前设定停机条件：

```text
If attention-ranked heads fail patch / ablation:
  attention is only a candidate signal, not a path.

If top-k heads work in teacher-forced NLL but fail decoded controls:
  claim target-likelihood component, not decoded generation circuit.

If top-k neurons remain weak while PCA / SAE directions work:
  claim distributed feature subspace, not sparse neurons.

If generation heads also help understanding:
  claim shared prompt-side influence, not generation-specific heads.

If full residual works from input embedding to late layers:
  treat it as trajectory overwrite upper bound, not local circuit evidence.
```

