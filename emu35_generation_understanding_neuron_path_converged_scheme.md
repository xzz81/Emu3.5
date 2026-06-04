# Emu3.5 图像生成与图像理解任务选择性神经元 / 通路探究方案

## 0. 收敛后的核心问题

本研究不再泛泛地问：

```text
Emu3.5 在图像生成或图像理解时，是否有某些神经元被激活？
```

而是收敛为一个可验证的因果问题：

```text
Emu3.5 是否存在对图像生成或图像理解任务具有选择性因果作用的内部组件？
```

这里的“内部组件”按粒度从粗到细包括：

```text
residual stream position
attention head
attention edge
MLP module
MLP intermediate neuron
low-rank feature direction / subspace
```

最终结论必须区分三种强度：

1. `activation-correlated`：任务中激活更强，但尚未证明因果作用。
2. `causal component`：干预该组件会显著改变目标任务输出。
3. `task-selective causal path`：该组件对一个任务有因果作用，对另一个任务影响很弱，并且可以被 path patching / cross-task control 支持。

## 1. 模型结构假设

Emu3.5 是 token-based native multimodal decoder-only UMM，不应套用 diffusion 模型中的：

```text
text encoder -> cross-attention -> UNet / DiT -> image
```

本研究真正要定位的是同一个 Transformer 主干中的 token-to-token 信息流：

```text
text / image tokens
-> self-attention / MLP / residual stream
-> text-token logits or image-token logits
```

因此两个任务的路径定义为：

```text
generation path:
prompt text tokens -> image output positions -> visual-token logits

understanding path:
image input tokens / EOI-like token -> text answer positions -> answer-token logits
```

## 2. 研究假设

### H1: 存在任务差异性激活

图像生成与图像理解会在部分层、attention heads、MLP intermediate neurons 或 residual stream 位置上产生不同激活模式。

### H2: 单个高激活神经元未必是因果神经元

高激活只能作为候选。只有通过 ablation、activation patching、path patching 或 projection-out 后仍能改变任务分数，才能称为因果组件。

### H3: 任务特异性更可能出现在“方向 / 子空间 / head group”而不是单个神经元

大模型能力通常是分布式的。因此本方案把单神经元搜索放在后半段，并与 low-rank feature direction、head group 和 module-level patching 并行验证。

### H4: generation 与 understanding 可能共享主干，但存在方向性不同的信息流

预期不是完全分离的两套神经元，而是：

```text
generation: prompt semantics -> visual token likelihood
understanding: visual evidence -> answer token likelihood
```

二者可能共享一些语义表征，但在 token position、attention edge、head group 和 feature subspace 上出现可分离差异。

## 3. 最小任务集

为了减少变量，本研究第一阶段只使用可控的合成 hue / attribute 数据，不先进入复杂自然图像。

### 3.1 Generation 任务

目标是测试 prompt semantic grounding，而不是测试图像美观度。

示例 prompt：

```text
a red circle
a blue circle
a green square
a purple square
a cyan diamond
a magenta diamond
```

干净 / 扰动 pair 只改变一个属性：

```text
red circle   <-> blue circle
green square <-> purple square
cyan diamond <-> magenta diamond
```

主要观测：

```text
prompt token positions
image output token positions
visual-token logits
```

首选指标：

```text
S_gen = target visual-token log-likelihood recovery
```

如果 decoded image 稳定，再增加辅助指标：

```text
color VQA score
object VQA score
CLIP text-image score
human / rule-based image judgment
```

### 3.2 Understanding 任务

目标是测试 image-to-answer semantic transfer。

示例：

```text
image: red circle
question: What color is the circle? Answer with one word.
answer: red
```

干净 / 扰动 pair 同样只改变一个属性：

```text
red circle image   <-> blue circle image
green square image <-> purple square image
```

主要观测：

```text
image input token positions
EOI-like / boundary token positions
question text positions
answer output positions
answer-token logits
```

首选指标：

```text
S_under = correct answer-token logit / log-likelihood recovery
```

## 4. 核心分数

### 4.1 任务内因果分数

对任意组件 `c`，定义 generation 因果分数：

```text
S_gen(c) =
  recovery of clean target visual-token likelihood
  after patching / restoring component c
```

定义 understanding 因果分数：

```text
S_under(c) =
  recovery of clean answer-token likelihood
  after patching / restoring component c
```

### 4.2 任务选择性分数

generation-selective 分数：

```text
G(c) = S_gen(c) - lambda * S_under(c)
```

understanding-selective 分数：

```text
U(c) = S_under(c) - lambda * S_gen(c)
```

默认 `lambda = 1`。如果两个任务分数尺度不同，需要先用同层随机基线或 full-residual upper bound 归一化。

### 4.3 判定标准

一个组件可以被称为 generation-selective causal component，至少需要满足：

```text
S_gen(c) > same-layer random baseline
S_under(c) is small or negative
effect remains under multiple clean/corrupt pairs
```

一个组件可以被称为 understanding-selective causal component，至少需要满足：

```text
S_under(c) > same-layer random baseline
S_gen(c) is small or negative
effect remains under multiple clean/corrupt pairs
```

若只满足激活差异，不满足因果干预，结论只能写为：

```text
task-correlated activation candidate
```

不能写为：

```text
task-specific neuron
```

## 5. 实验流程

### Stage 0: Hook 与 token taxonomy 校准

目标：

确认 Emu3.5 的 text token、visual token、image boundary token、answer token、generation output token 在 trace 中能被稳定标注。

需要记录：

```text
token string / token id
position index
token type
task phase: prompt / image_input / image_output / question / answer
layer count
hidden size
MLP intermediate size
attention head count
```

停机条件：

```text
能把 generation 的 image output positions 与 understanding 的 answer positions 区分开。
```

### Stage 1: Activation screening

目标：

从全模型中粗筛候选层、head、MLP neuron 和 residual position。

观测位置：

```text
residual stream
attention head output
MLP output
MLP intermediate activation = act(gate_proj(x)) * up_proj(x)
```

聚合维度：

```text
task
sample_id
semantic_attribute
token_type
position_group
layer
component
```

输出候选：

```text
G_activation_candidates
U_activation_candidates
shared_activation_candidates
```

这一阶段只回答：

```text
哪些位置看起来更像 generation / understanding 候选？
```

不回答：

```text
哪些神经元真的控制了生成或理解？
```

### Stage 2: Layer / module causal localization

目标：

先用粗粒度 patching 找到最可能的因果层和模块，避免过早陷入单神经元海量搜索。

Generation clean/corrupt：

```text
clean prompt:   a red circle
corrupt prompt: a blue circle
target:         clean visual tokens or clean decoded attribute
```

Understanding clean/corrupt：

```text
clean image:   red circle
corrupt image: blue circle
question:      What color is the circle?
target:        red
```

干预对象：

```text
residual stream at selected token positions
attention output by layer
MLP output by layer
```

优先排序：

1. residual patching by layer and position group
2. module patching: attention vs MLP
3. attention head patching
4. MLP neuron / subspace patching

输出：

```text
layer_module_recovery_matrix.csv
top_generation_layers.md
top_understanding_layers.md
```

### Stage 3: Attention head 与 edge-level path patching

目标：

把“哪个层有用”推进到“哪些 token 间的信息流有用”。

Generation 重点边：

```text
image output positions <- prompt color token
image output positions <- prompt object token
image output positions <- full prompt text positions
```

Understanding 重点边：

```text
answer positions <- image tokens
answer positions <- EOI-like / image boundary token
answer positions <- question tokens
```

干预方式：

```text
attention knockout
head output patching
path patching
edge masking
```

关键 control：

```text
same-layer random heads
same-token random positions
wrong-attribute pair
unrelated prompt / question pair
```

判定：

如果屏蔽或替换某组 edge 后，目标任务分数显著下降，同时另一个任务影响很小，则该 edge group 可以进入 task-selective path 候选。

### Stage 4: MLP neuron 与 feature subspace localization

目标：

回答用户最关心的“是否有特定神经元被激活 / 起作用”。

本阶段不直接假设单神经元稀疏成立，而是并行比较三种粒度：

```text
top-k MLP neurons by activation difference
top-k MLP neurons by causal patching recovery
low-rank feature directions / PCA / supervised contrastive direction
```

单神经元实验：

```text
ablate top-k generation neurons during generation
ablate top-k generation neurons during understanding
ablate top-k understanding neurons during understanding
ablate top-k understanding neurons during generation
compare with same-layer random-k neurons
```

子空间实验：

```text
fit generation direction from clean-corrupt activation difference
fit understanding direction from clean-corrupt activation difference
projection-out during target task
projection-out during non-target task
activation addition / patching for recovery
```

判定：

```text
若 top-k neurons >> random-k 且跨 pair 稳定:
  可以报告 task-selective neuron group

若 low-rank direction >> top-k neurons:
  报告 distributed task-selective feature subspace

若二者都弱:
  报告未发现稳定神经元级选择性，能力可能更分布式或 metric 不足
```

### Stage 5: Cross-task intervention and claim audit

目标：

验证 generation 候选是否真的 generation-specific，understanding 候选是否真的 understanding-specific。

核心矩阵：

```text
component c | S_gen(c) | S_under(c) | G(c) | U(c) | conclusion
```

必须包含：

```text
generation top heads tested on understanding
understanding top heads tested on generation
generation top MLP neurons tested on understanding
understanding top MLP neurons tested on generation
generation feature subspace tested on understanding
understanding feature subspace tested on generation
```

最终 claim 分级：

```text
Claim A: Emu3.5 has task-correlated activations.
Claim B: Emu3.5 has task-selective causal components.
Claim C: Emu3.5 has sparse task-selective neurons.
Claim D: Emu3.5 has distributed task-selective feature subspaces.
Claim E: Emu3.5 has separable generation and understanding paths.
```

每个 claim 都要写成：

```text
supported / partially supported / not supported / not tested
```

## 6. 最小可执行版本

如果只做一个可投稿前的最小闭环，建议执行以下 8 个实验：

1. 构造 hue clean/corrupt generation 与 understanding pair。
2. 建立 token taxonomy，确认 visual output positions 与 answer positions。
3. 做 layer-position residual patching，分别得到 `S_gen` 与 `S_under` 热图。
4. 对高恢复层做 attention-vs-MLP module patching。
5. 对最高层 / 模块做 attention head patching。
6. 对最高 MLP 层做 top-k neuron ablation / patching，并与 random-k 比较。
7. 对最高 MLP 层做 low-rank feature subspace patching / projection-out。
8. 建立 cross-task matrix，审计所有结论。

完成后能回答：

```text
图像生成任务中是否有特定神经元或组件被激活？
这些激活是否对 visual-token generation 有因果作用？

图像理解任务中是否有特定神经元或组件被激活？
这些激活是否对 answer-token prediction 有因果作用？

generation 与 understanding 的因果组件是否可区分？
这种可区分性是单神经元级、head-level、module-level，还是 feature-subspace-level？
```

## 7. 推荐结果表

### 7.1 Component selectivity matrix

```text
component_id
component_type
layer
position_group
S_gen
S_under
G_score
U_score
random_baseline
upper_bound
conclusion
```

### 7.2 Neuron group table

```text
layer
neuron_group
selection_method
k
generation_effect
understanding_effect
random_k_effect_mean
random_k_effect_std
stability_across_pairs
conclusion
```

### 7.3 Path table

```text
source_token_group
target_token_group
layer
head_or_edge
generation_knockout_effect
understanding_knockout_effect
path_selectivity
conclusion
```

## 8. 预期结论模板

如果单神经元很强：

```text
We identify a small group of MLP neurons that are selectively causal for Emu3.5 visual-token generation / image-to-text understanding.
```

如果 head 或路径更强：

```text
The strongest task selectivity appears at the attention-head / token-edge level rather than at sparse MLP-neuron level.
```

如果子空间更强：

```text
Generation and understanding are better explained by distributed feature subspaces than by individual task-specific neurons.
```

如果只有 activation 差异：

```text
We observe task-correlated activations, but causal interventions do not support stable task-selective neurons under the current benchmark.
```

## 9. 风险与控制

### 风险 1: decoded image 不稳定

优先使用 teacher-forced visual-token likelihood；decoded image 指标只作为补充。

### 风险 2: corruption artifact

使用最小属性扰动 pair，并加入 wrong-pair、random-token、random-head、same-layer random-neuron control。

### 风险 3: 单神经元结论过强

所有 neuron claim 必须和 random-k、module upper bound、low-rank subspace 对比。若 top-k neurons 只能恢复极小比例，不写“特定神经元控制任务”，改写为“存在可排序的弱因果 neuron group”或“分布式特征质量”。

### 风险 4: generation 与 understanding 分数尺度不同

用同任务 full-residual recovery 和同层 random baseline 归一化，再比较选择性。

### 风险 5: token position 误标

Stage 0 必须先完成 token taxonomy audit。没有 token taxonomy，不进入 path claim。

## 10. 最终交付物

建议最终产出以下文件：

```text
outputs/token_taxonomy_audit.md
outputs/stage1_activation_screening.csv
outputs/stage2_layer_module_recovery_matrix.csv
outputs/stage3_attention_edge_path_report.md
outputs/stage4_neuron_subspace_report.md
outputs/stage5_cross_task_selectivity_matrix.csv
outputs/stage5_claim_audit.md
```

最终主报告结构：

```text
1. Problem: task-selective neurons or paths in Emu3.5
2. Model-specific framing: decoder-only token-based UMM
3. Benchmark: matched generation / understanding hue pairs
4. Method: activation screening -> causal patching -> path patching -> subspace intervention
5. Results: component selectivity matrix
6. Claim audit: what is supported and what is not
7. Discussion: sparse neurons vs distributed subspaces
```

## 11. 一句话版本

本方案把“Emu3.5 在图像生成和图像理解时是否有特定神经元被激活”收敛为：

```text
在 matched generation / understanding 任务上，
用 activation screening 找候选，
用 clean/corrupt patching 验证因果，
用 cross-task matrix 验证选择性，
最后判断任务差异来自 sparse neurons、attention paths，还是 distributed feature subspaces。
```
