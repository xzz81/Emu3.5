# Emu3.5 UMM Generation / Understanding Path 新版计划

## 1. 总体判断

新版方案采用更保守的证据链：

```text
annotated evidence
-> attention / evidence pre-ranking
-> candidate heads or modules
-> causal patch / ablation
-> cross-task characterization
-> decoded validation
-> partial path claim
```

核心调整是：

```text
Understanding 路线基本保留；
Generation 路线从“寻找 sparse direct attention path”
降级为“先探索是否存在 QRHead-like direct color-token retrieval；
如果没有，就转向 residual / MLP feature-subspace route”。
```

这版方案不再把 raw attention edge、target residual overwrite、raw logit lens 直接解释成 generation path。

## 2. 关键删改

### 2.1 删除 edge-level 主张

删除：

```text
attention / evidence score
-> sparse candidate heads or edges
```

改成：

```text
attention / evidence score
-> candidate heads or modules
```

原因：

当前主要实验是 head ranking、head-output patch、residual patch、module patch、MLP low-rank patch。它们最多支持 head-level component、module-level component、residual route、feature subspace，还不能自然支持 edge-level path。

除非后续真的做 path patching / edge attribution，否则不要写 edge-level path claim。

### 2.2 删除 generation color-token edge knockout 作为主实验

删除原 G2：

```text
query = visual-score positions
source = color-token positions
intervention = add -inf attention mask on selected edges
```

原因：

对 selected attention edge 加 `-inf` 会改变 softmax 归一化结构。它不是干净切断 color-token 信息流，而是强行把注意力质量重新分配给其他 token。这个结果不能直接解释为 color-token edge 必要。

更稳的替代：

```text
head-output patch / ablation
MLP low-rank patch
residual subspace patch
prompt color-token boundary patch
```

如果以后要做 edge-level claim，应改成：

```text
attention-pattern-preserving value patch
path patching
attribution patching
```

### 2.3 删除粗糙的 color-specific attention 公式

删除：

```text
G_color_specific(h) =
  G_color_attn(h)
  - G_object_attn(h)
  - G_null_attn(h)
```

原因：

color token、object token、null token 的数量、位置、BPE 切分和语义作用都不同，直接相减会混入位置偏置、token 数量偏置和 prompt 格式偏置。

改成：

```text
matched / normalized color evidence
```

例如：

```text
attn_to_color(clean_color_prompt)
- attn_to_same_position(corrupt_color_prompt)
```

或：

```text
per-token-normalized color mass
vs.
position-matched non-color semantic token mass
```

也可以更保守：

```text
color-token attention 只作为 candidate feature；
最终以 patch / ablation / reverse patch 验证。
```

### 2.4 降级 raw logit-lens write score

删除：

```text
WriteScore(h) = contribution to clean visual-token logit
```

原因：

中间层 head output 写入 residual stream 后，还会经过后续 attention、MLP、norm、residual mixing。直接用 unembedding 或 logit lens 看 clean visual-token logit contribution，只能作为 proxy，不能写成严格 causal write。

改成：

```text
WriteScore_causal(h) =
  patch / ablation 后 clean-vs-corrupt visual-token logit margin 的变化
```

或：

```text
AttributionScore(h) =
  attribution patching approximation of margin effect
```

### 2.5 分离 candidate ranking 和 causal validation

删除：

```text
G_candidate(h) =
  pre-rank score + clean-to-corrupt patch recovery
```

原因：

QRHead-style ranking 的意义是在 patch 之前用便宜信号找候选。patch recovery 属于验证阶段，不能混入 pre-ranking。

改成两阶段：

```text
Stage 1: candidate ranking
Stage 2: causal validation
```

最后可以汇总：

```text
validated summary table
```

但不要把它叫 pre-rank candidate score。

### 2.6 降级 target residual patched decode

删除：

```text
用 full target residual patched decode 验证 top component
```

改成：

```text
target residual patched decode =
  upper bound
  positive control
  decoded pipeline sanity check
```

原因：

full target residual patch 证明的是 target trajectory overwrite 能恢复 hue，不证明某个 top head、top MLP direction 或 compact generation circuit。

只有能在 free decode 中在线干预的组件，才可以进入 decoded component validation。

### 2.7 cross-task control 改成 characterization

删除硬标准：

```text
S_gen(top_U) ≈ 0
S_under(top_G) ≈ 0
```

改成：

```text
cross-task effect is a label, not a pass/fail criterion
```

解释：

UMM 中 generation 和 understanding 共享部分 prompt-side / residual-side machinery 是合理的。cross-task control 应用于区分：

```text
task-selective component
shared U/G component
```

而不是判断 component 有没有价值。

### 2.8 exact HSV 降为辅助指标

删除：

```text
exact HSV 作为主成功标准
```

改成：

primary decoded metrics:

```text
hue-distance improvement
color classifier / VLM judge color answer
object preservation
decoded validity
```

secondary metric:

```text
exact HSV
```

原因：

free generation 有采样噪声、tokenizer reconstruction noise、光照变化、背景污染和 object boundary ambiguity。exact HSV 太脆弱，不能作为主判据。

## 3. Understanding Path 新方案

### 3.1 路径假设

当前最合理的 understanding path：

```text
image tokens / EOI
-> post-image question/text residual states
-> answer score positions
-> answer-token logits
```

注意：

```text
answer score -> image token
```

的 direct attention edge 可能较弱，因此不要只看 answer score 到 image tokens 的 raw attention。

### 3.2 U1: Understanding Evidence-Head Census

目标：

```text
用 QRHead-style evidence score 重新 ranking understanding heads，
检查是否支持已有 L62 heads 36/38/46。
```

配置：

```text
task = understanding
layers = 56-63 first, then all layers
primary query = answer-score positions
secondary query = post-image question tokens / assistant-format tokens
sources = visual_all, visual_gt_region if available, eoi_only, visual_plus_eoi
output = head_scores_understanding.csv
```

EOI 必须单独报告，不要和普通 visual tokens 混在一起平均。

Pre-rank score：

```text
U_pre_score(h) =
  z(EvidenceMass)
  + z(RegionRatio) if bbox/mask available
  + z(1 - SpatialEntropy) if visual grid available
  + z(SelectionFrequency)
```

这里不能加入 patch recovery。

### 3.3 U2: L62 Heads Necessity + Sufficiency

目标：

```text
验证 L62 heads 36/38/46 是否既 sufficient 又 necessary。
```

测试对象：

```text
L62 heads 36/38/46
L62 top7
random3 / random7
bottom heads 35/32
attention-matched controls if possible
```

Interventions：

```text
clean -> corrupt head-output patch
clean head ablation / mean replacement
corrupt -> clean reverse patch
```

Metrics：

```text
clean answer NLL
answer logit margin
answer flip rate
patch recovery
reverse patch damage
```

结论规则：

```text
如果 U heads 对 generation 近 0：
  understanding-selective

如果 U heads 对 generation 也有正效应：
  shared U/G component
```

可以写：

```text
L62 heads 36/38/46 are causal head-level components
for hue understanding under clean/corrupt patching and ablation controls.
```

暂时不要写：

```text
sparse understanding neurons
complete understanding circuit
fully separated understanding path
```

## 4. Generation Path 新方案

Generation 侧要拆成两个已知证据和一个开放问题。

### 4.1 已知证据 1: Prompt Color-Token Boundary Control

已有 Stage 45/46 支持：

```text
color-token input embedding patch:
  2/4 exact HSV
  4/4 hue-distance improvement

object-token input embedding control:
  0/4 exact HSV
  0/4 hue-distance improvement
```

这说明 prompt color token 对 free generation hue 有因果影响。

但它只证明 input-boundary control，不证明内部 compact circuit。

### 4.2 已知证据 2: Target-Side Visual-Score Residual Trajectory Control

已有 generation full residual patch 很强：

```text
L61 visual-score full residual patch ≈ full recovery
```

但这更像：

```text
target trajectory overwrite upper bound
```

不能直接当成 compact generation circuit。

它的用途是：

```text
upper bound
component completeness denominator
decoded positive control
pipeline sanity check
```

### 4.3 开放问题

Generation 内部 route 目前保持开放：

```text
sparse direct color-token retrieval heads
distributed residual route
MLP / feature-subspace route
target-trajectory-dominant route
```

新版方案的任务是区分这些可能性，而不是预设答案。

## 5. Generation 实验集

### 5.1 G1: Generation Attribute-Attention Exploratory Census

目标：

```text
只回答 generation 是否存在 QRHead-like direct color-token retrieval heads。
```

配置：

```text
task = generation
query = visual-score positions
sources = color-token positions,
          position-matched corrupt-color token,
          object-token positions,
          all semantic prompt tokens
layers = 0,8,16,32,40,48,52,56-63 first
output = head_scores_generation_attention.csv
```

Pre-rank score 只使用 candidate evidence：

```text
normalized color-token evidence
matched color contrast
optional attribution proxy
```

不要加入 patch recovery。

结论规则：

```text
如果成功：
  存在 direct color-token attention candidate heads。

如果失败：
  只说明没有 strong QRHead-like direct retrieval route；
  不说明 generation 没有 color-conditioning path。
```

### 5.2 G2: Generation Component Causal Validation

这是 generation 内部 route 的主验证，替代原来的 edge knockout。

Candidate sources：

```text
G1 top attention heads
high attribution-proxy heads
MLP low-rank directions
residual subspace directions
prompt color-token boundary component
```

Interventions：

```text
clean -> corrupt head-output patch
clean head ablation / mean replacement
corrupt -> clean reverse patch
MLP low-rank patch
residual subspace patch
prompt-boundary patch
```

Controls：

```text
random heads
bottom heads
attention-matched low-effect heads
object-token controls
wrong-color controls
random-color controls if available
```

Metrics：

```text
teacher-forced target visual-token NLL
clean-vs-corrupt visual-token logit margin
hue-token / color-code likelihood if definable
component recovery / full residual recovery
```

结论规则：

```text
如果 top attention heads 弱、MLP low-rank / residual subspace 更强：
  generation color conditioning is better explained by residual /
  feature-subspace structure than sparse attention heads.

如果 G1 失败但 G2 中 MLP / residual subspace 成功：
  generation is not QRHead-like at head level,
  but still has an internal color-conditioning route.
```

### 5.3 G3: Decoded Validation Only For Online Component Intervention

目标：

```text
只对 teacher-forced 通过、且能在 free decode 中在线干预的 component 做 decoded validation。
```

Valid interventions：

```text
online head-output patch
online MLP-subspace patch
online residual-subspace patch
online prompt-boundary intervention
```

Not valid as component proof：

```text
full target residual patched decode
```

它只能作为：

```text
upper bound
positive control
decoded pipeline sanity check
```

Primary decoded metrics：

```text
hue-distance improvement
decoded color classifier / VLM judge accuracy
object preservation
image validity
```

Secondary metric：

```text
exact HSV
```

## 6. 建议新增工具

如果要实现统一 attention evidence ranking，建议新增：

```text
scripts/run_attention_evidence_head_ranking.py
```

它只负责 pre-ranking，不做 patch recovery 综合排序。

建议参数：

```text
--task understanding|generation
--cfg PATH
--manifest PATH
--target-cache-dir PATH
--layers all|60,61,62
--query-scope answer-score|post-image-text|visual-score-positions
--source-scope visual_all|visual_gt_region|eoi_only|visual_plus_eoi|color-token|object-token|prompt
--attn-implementation eager
--out-dir PATH
```

输出：

```text
head_scores.csv
head_scores_topk.json
ranking_report.md
attention_maps/ optional
```

核心列：

```text
task
layer
head
query_scope
source_scope
evidence_mass
region_ratio
spatial_entropy
selection_frequency
pre_rank_score
```

不要把 patch recovery 写入 pre-rank score。

后续可以另做：

```text
validated_component_summary.csv
```

用于汇总 pre-rank、patch recovery、ablation damage、reverse damage、decoded effect。

## 7. 最终可检验表述

### 7.1 Understanding

当前可以保留：

```text
Emu3.5 hue understanding likely uses a late-layer
image-conditioned text-side residual route.

L62 heads 36/38/46 are strong candidate causal
head-level components.
```

如果 U2 通过，可以写：

```text
L62 heads 36/38/46 are causal head-level components
for hue understanding under clean/corrupt patching
and ablation controls.
```

暂时不要写：

```text
sparse understanding neurons
complete understanding circuit
fully separated understanding path
```

### 7.2 Generation

当前更稳的表述：

```text
Emu3.5 hue generation has two currently separable
pieces of evidence:

1. prompt color-token boundary control;
2. late target-side visual-score residual trajectory control.

The internal route should not yet be described as a
sparse attention-head circuit.
```

内部 route 仍是开放问题：

```text
sparse direct color-token retrieval heads
distributed residual route
MLP / feature-subspace route
target-trajectory-dominant route
```

如果 G1 失败，可以写：

```text
We do not find strong evidence for a QRHead-like
direct color-token attention route in generation.
```

不能写：

```text
generation has no compact path
```

如果 G2 中 MLP / subspace 更强，可以写：

```text
generation color conditioning is better explained by
residual / feature-subspace structure than by sparse
attention heads.
```

## 8. 停机条件

为避免过度解释，提前设定以下停机条件：

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

## 9. 一句话收敛版

```text
Understanding:
  保留 QRHead / grounding-head 思路，推进 L62 heads 36/38/46
  的 necessity + sufficiency。

Generation:
  不再预设 sparse attention-head path。
  先探索是否存在 direct color-token retrieval heads；
  若不强，则转向 residual / MLP feature-subspace route。
```
