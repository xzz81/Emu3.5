# Emu3.5 图像生成与图像理解通路探究方案

> 更新说明：本文件是早期研究路线草案。当前收敛版方案与最新实验结论见
> `emu35_task_selective_path_probe_final_plan.md`；
> 最新 cross-task head control 见
> `outputs/cross_task_attention_head_controls_stage14_report.md`。

## 1. 核心问题

本方案探究 Emu3.5 在图像生成和图像理解任务中是否调用了可区分的内部神经通路。

核心问题不是简单地问“是否有某些神经元被激活”，而是进一步问：

1. Emu3.5 是否存在 generation-selective 与 understanding-selective 的内部组件？
2. 这些组件是否对对应任务输出具有因果作用？
3. 屏蔽或替换这些组件后，另一个任务是否受到影响？

因此，实验目标从相关性分析推进到因果通路定位：

```text
activation screening -> causal patching -> cross-task intervention
```

## 2. 研究假设

Emu3.5 是 native multimodal decoder-only UMM，图像、文本都被 token 化后进入同一个 Transformer 主干。因此，图像生成与图像理解并不是 diffusion 模型中的 text encoder 到 cross-attention 到 UNet/DiT 路径，而是：

```text
text/image tokens
-> self-attention / MLP / residual stream
-> text-token logits or image-token logits
```

本方案提出三个假设：

### H1: 任务选择性激活

图像生成任务和图像理解任务会激活不同的层、MLP 中间神经元或 attention heads。

### H2: 任务选择性因果通路

部分高激活组件不仅相关，而且对任务输出有因果作用。generation-selective 组件主要影响 prompt-to-image-token grounding；understanding-selective 组件主要影响 image-token-to-answer-token communication。

### H3: 部分共享但可分离

图像生成和图像理解可能共享部分 backbone 表征，但存在可分离的任务专属通路。对某一类通路进行干预会导致不对称退化。

## 3. 实验对象与任务设置

为了避免模型差异干扰，generation 和 understanding 对照实验应优先使用同一个 Emu3.5 主模型，而不是用 Emu3.5-Image 与 Emu3.5 互相比较。

### Generation 任务

使用 T2I / X2I 任务，重点观察语义 grounding，而不是单纯图像质量。

示例：

```text
a red cube on the left of a blue sphere
```

可测试子能力：

- object presence
- color binding
- spatial relation
- counting
- text rendering
- reference-image copying

### Understanding 任务

使用 image caption / VQA / attribute QA / spatial QA。

示例：

```text
image: red cube
question: what color is the cube?
answer: red
```

可测试子能力：

- object recognition
- attribute recognition
- spatial relation recognition
- counting
- hallucination sensitivity

## 4. Stage 0: Activation Screening

### 目的

先粗筛候选层、神经元、token 类型和模块，得到可能的 generation-selective 与 understanding-selective 组件。

### 观测对象

优先 hook MLP 中间激活：

```text
act(gate_proj(x)) * up_proj(x)
```

原因是该位置比 residual hidden state 更接近 MLP 神经元激活。

同时按以下分组聚合：

```text
task = generation / understanding
token_type = text / visual / structure
layer
neuron
```

### 输出

得到三类候选组件：

```text
G_candidate: generation 中显著高激活
U_candidate: understanding 中显著高激活
Shared_candidate: 两类任务中都高激活
```

### 可运行命令

```bash
CUDA_VISIBLE_DEVICES=0,1 python scripts/run_activation_probe.py \
  --cfg generation=configs/ume_main_t2i_first_image_stop_probe.py \
  --cfg understanding=configs/ume_main_image_read_entropy_seed21.py \
  --max-prompts 8 \
  --layers all \
  --max-seq-len 4096 \
  --out-dir outputs/activation_probe_gen_vs_understanding
```

### 关键输出文件

```text
group_summary.csv
top_active_neurons.csv
contrast_generation_vs_understanding_overall.csv
contrast_understanding_vs_generation_overall.csv
contrast_generation_vs_understanding_text.csv
contrast_understanding_vs_generation_text.csv
contrast_generation_vs_understanding_visual.csv
contrast_understanding_vs_generation_visual.csv
```

### 注意

Stage 0 只能说明 activation correlation，不能作为最终结论。它只用于缩小搜索空间。

## 5. Stage 1: Causal Patching

### 目的

验证 Stage 0 找到的候选组件是否真的对任务输出有因果作用。

核心范式：

```text
clean run
corrupt run
patch clean activation into corrupt run
measure recovery
```

## 5.1 Generation Path Patching

### Clean/Corrupt Pair

构造最小语义扰动：

```text
clean prompt:
a red cube on the left of a blue sphere

corrupt prompt:
a green cube on the left of a blue sphere
```

也可以构造 spatial pair：

```text
clean prompt:
a red cube on the left of a blue sphere

corrupt prompt:
a red cube on the right of a blue sphere
```

### Patching 位置

候选位置包括：

- prompt token residual stream
- image-generation positions residual stream
- MLP intermediate neurons
- attention head outputs
- text-token to image-token attention edges

### Generation 因果分数

优先使用 teacher-forced image-token likelihood：

```text
S_gen(c) = Δ log p(target image tokens)
```

如果没有可靠 ground-truth image tokens，则使用 decoded image 后的辅助指标：

- VQA score
- object detector score
- color classifier score
- spatial relation classifier score
- CLIP text-image score
- human annotation

## 5.2 Understanding Path Patching

### Clean/Corrupt Pair

```text
clean:
image of red cube + "what color is the cube?"

corrupt:
image of green cube + "what color is the cube?"
```

### Patching 位置

候选位置包括：

- visual token residual stream
- EOI-like token residual stream
- text answer token residual stream
- MLP intermediate neurons
- attention head outputs
- image-token to answer-token attention edges

### Understanding 因果分数

```text
S_understand(c) = Δ log p(correct answer token)
```

也可以使用：

- VQA accuracy recovery
- answer margin recovery
- hallucination reduction

## 6. Stage 2: Differential Path Scoring

### 目的

不要只找对 generation 有用或对 understanding 有用的组件，而是找任务差异性组件。

对每个组件 c，定义：

```text
G(c) = S_gen(c) - λ S_understand(c)
```

高 G(c) 表示 generation-specialized component。

反过来：

```text
U(c) = S_understand(c) - λ S_gen(c)
```

高 U(c) 表示 understanding-specialized component。

λ 可以从 0.5、1.0、2.0 做 sensitivity sweep。

最终组件分为：

```text
generation-specialized
understanding-specialized
shared
irrelevant
```

## 7. Stage 3: Cross-Task Intervention

### 目的

检验 generation path 和 understanding path 是否可分离，以及它们是否互相影响。

## 7.1 Mask Generation Path During Understanding

在 I2T / VQA 中屏蔽 generation-specialized components：

```text
mask G(c) during understanding
```

观察：

- VQA accuracy 是否下降
- hallucination 是否下降
- answer grounding 是否变差
- answer fluency 是否保持

可能结论：

```text
若 grounding 不变，说明 generation-specialized path 与 understanding path 可分离。
若 hallucination 下降，说明 generation path 可能参与了视觉幻觉。
若 accuracy 下降，说明 generation path 也被 understanding 调用。
```

## 7.2 Mask Understanding Path During Generation

在 T2I / X2I 中屏蔽 understanding-specialized components：

```text
mask U(c) during generation
```

观察：

- prompt-following 是否下降
- object binding 是否错误
- color binding 是否错误
- spatial relation 是否错误
- image quality 是否基本保持

可能结论：

```text
若图像质量保持但 prompt-following 下降，说明 U(c) 参与语义 grounding。
若生成完全不受影响，说明 understanding path 与 generation path 较可分离。
若图像质量也下降，说明干预位置可能不是语义通路，而是底层 visual-token modeling 通路。
```

## 8. 对照实验

为了避免 activation patching artifact，必须加入以下控制：

### Random Component Control

随机选择相同数量的 layer/head/neuron 进行 mask，与目标组件比较。

### Random Token Span Control

随机 mask text/image token span，排除 token 数量差异导致的效果。

### Clean-Clean Patch

在 clean run 之间 patch，确认 patch 操作本身不会破坏输出。

### Corrupt-Corrupt Patch

在 corrupt run 之间 patch，确认 recovery 不是随机波动。

### Matched Semantic Pair

每组 clean/corrupt pair 只改变一个因素：

```text
red -> green
cube -> sphere
left -> right
two -> three
```

### Token-Type Stratification

区分：

```text
task selectivity
visual-token selectivity
text-token selectivity
structure-token selectivity
```

避免把“视觉 token 激活强”误解成“图像生成任务专属”。

## 9. 评估指标

### Generation 指标

优先级从高到低：

1. teacher-forced target image-token log likelihood
2. decoded image VQA score
3. object/color/spatial classifier score
4. CLIP text-image score
5. human annotation

同时区分：

```text
semantic faithfulness
image quality
visual-token fluency
```

不要只用图像质量指标，因为本研究关心的是 prompt semantic grounding path。

### Understanding 指标

优先级从高到低：

1. correct answer-token log probability
2. answer logit margin
3. VQA accuracy
4. hallucination rate
5. caption faithfulness

## 10. 最小可行实验

如果先做一个最小版本，建议只保留：

```text
任务:
T2I color binding
VQA color answering

语义对:
red cube vs green cube
blue sphere vs yellow sphere

组件:
MLP intermediate neurons
residual stream at selected layers

指标:
S_gen = target image-token likelihood recovery
S_understand = correct answer-token log-prob recovery
```

最小实验流程：

1. 用 activation screening 找 top generation/understanding 候选神经元。
2. 对这些候选做 clean/corrupt causal patching。
3. 计算 G(c) 与 U(c)。
4. mask top G(c) during VQA。
5. mask top U(c) during T2I。
6. 与 random mask 对比。

## 10.1 当前实证进展

当前已经完成从 activation screening 到 understanding-side path decomposition 的一轮漏斗，并开始补 generation-side residual probe。

### Activation / Ablation

已确认 Emu3.5 在 generation output visual tokens 与 understanding output text tokens 上存在 late-layer task-selective MLP activations。

关键输出：

```text
outputs/activation_probe_hue_replay_pilot/stage0_pilot_report.md
outputs/stage1_mlp_ablation_comparison/stage1_top_vs_random_report.md
```

当前支持：

```text
understanding-selected MLP neurons affect answer likelihood more than same-layer random controls.
```

但 generation-selected MLP neurons 的 ablation 证据较弱。

### Understanding Path

理解侧已经从 neuron-level 推进到 residual/module/head-level：

```text
late score-position residual patching at layers 61-63:
near-complete clean answer recovery

direct answer-score -> image/EOI attention knockout:
weak or negative

post-image question/text residual patching:
large recovery, supporting a mediated text-side route

layer 60/62 MLP module and layer 62 self-attn module:
local contribution

layer 62 attention heads 36/38/46 plus supporting heads:
strongest head-level localization so far

small layer 60/62 MLP intermediate-neuron sets:
near-zero recovery, comparable to random controls
```

关键报告：

```text
outputs/understanding_residual_patch_hue_stage2_comparison_report.md
outputs/understanding_attention_knockout_hue_stage3_comparison_report.md
outputs/understanding_mediated_route_hue_stage4_report.md
outputs/understanding_module_patch_hue_stage5_report.md
outputs/understanding_attention_head_patch_hue_stage6_report.md
outputs/understanding_mlp_neuron_patch_hue_stage7_report.md
```

当前理解侧结论：

```text
image-to-answer hue information is recoverable in late residual states,
partly localizable to layer-62 attention heads,
but not carried by the tested small sets of MLP intermediate neurons.
```

### Generation Path

generation-side residual patching 已实现：

```text
scripts/run_generation_residual_patch_recovery.py
```

expanded hue-only 16x16 all-pairs 结果：

```text
cfg = configs/ume_main_t2i_counterfactual_pairs_hueonly_seed69.py
layer = 62
valid directions = 8/8
target cache entries = 8

prompt / visual:
mean recovery -0.013560, helped 4/8

color-token / visual:
mean recovery +0.007586, helped 3/8

prompt / visual-and-structure:
mean recovery -0.437912, helped 1/8

color-token / visual-and-structure:
mean recovery -0.020293, helped 3/8

cached target hue validation:
decoded targets 8/8, simple HSV hue matches 0/8

direct patched decoded-generation:
clean hue matches 0/8
corrupt hue matches 0/8
patched_corrupt hue matches 1/8

CFG-aware direct generation smoke:
clean hue matches 0/2
corrupt hue matches 0/2
patched_corrupt hue matches 0/2

clean hue seed sweep:
decoded 4/4
hue matches 1/4
valid seed found for red circle: 147402

extended red/blue seed sweep:
red: 1/6 valid clean generations, matching seed 147402
blue: 0/6 valid clean generations
combined report: outputs/generation_clean_hue_seed_sweep_pair1_cfg2_combined/combined_clean_hue_seed_sweep_report.md
```

关键报告：

```text
outputs/generation_residual_patch_hue_stage8_report.md
```

当前 generation 侧结论仍然谨慎：

```text
teacher-forced visual-token clean/corrupt gaps can be constructed,
but prompt/color-token residual patching is weak, direction-dependent, and not stabilized by adding structure tokens.
The decoded clean targets themselves often fail simple hue validation, so visual-token NLL is currently a noisy generation grounding metric.
Direct decoded-image validation shows one isolated patched hue restoration case, but still no robust all-direction generation hue recovery.
CFG-aware hook-isolated generation runs, but the tested 16x16 pair still fails hue validation.
Seed filtering can find isolated valid clean hue generations, but the extended red/blue sweep confirms that this pair is not yet a usable bidirectional clean/corrupt generation benchmark.
Generation-side evidence is not yet as strong as understanding-side residual recovery.
```

下一步应转向 generated-image-level validation：

```text
add decoded-image hue validation or VQA/classifier checks;
improve or filter clean generated targets before relying on teacher-forced likelihood;
improve the generation setting itself, for example by using higher-quality valid targets or first filtering prompt/seed pairs with verified clean hue following;
expand seed sweeps to construct filtered clean/corrupt pairs where both sides pass hue validation;
reuse cached targets only for remaining likelihood sweeps.
```

## 11. 预期结果类型

### 结果 A: 完全共享

generation 和 understanding 的高因果组件高度重合。

含义：

```text
Emu3.5 的视觉语义 grounding 主要由共享通路完成。
```

### 结果 B: 部分共享、部分分离

存在 shared components，也存在 G(c) 与 U(c)。

含义：

```text
Emu3.5 的 generation 和 understanding 共用底层语义表征，但在输出阶段调用任务专属通路。
```

这是最可能、也最有论文价值的结果。

### 结果 C: 强分离

G(c) 与 U(c) 基本不重合，跨任务 mask 影响较小。

含义：

```text
统一 backbone 并不意味着统一 circuit；不同任务可能在同一 Transformer 内部形成分离子通路。
```

### 结果 D: 干预无效

activation screening 找到的组件无法通过 patching 造成 recovery。

含义：

```text
原始高激活只是相关性，不是因果通路。
需要转向 residual stream、attention edge 或 SAE feature-level localization。
```

## 12. 推荐引用锚点

主文献不宜过散，建议收敛到四类：

1. Narrow Gate: 用于 understanding path，尤其 image-to-text gate 与 EOI-like token。
2. LocoGen: 用于 generation path localization 的 direct-effect intervention 思路。
3. Best Practices of Activation Patching: 用于 clean/corrupt、metric、baseline 设计。
4. ACDC / path patching: 用于从重要组件升级到信息路径。

其他文献如 Prompt-to-Prompt、Attend-and-Excite、ConceptPrune、SAE for T2I 可放 related work 或 appendix，不作为主实验框架。

## 13. 最终可形成的 Claim

较稳健的主结论：

```text
We localize task-selective causal pathways in Emu3.5 and show that image generation and image understanding rely on partially overlapping but separable internal circuits.
```

更强的主结论：

```text
Generation-specialized components mainly mediate prompt-to-visual-token grounding, while understanding-specialized components mediate image-token-to-answer-token communication; intervening on one pathway produces asymmetric degradation across tasks.
```

## 14. 当前优先级

建议按以下顺序推进：

1. 跑 Stage 0 activation screening，得到候选层和候选神经元。
2. 只选 color binding / color QA 做最小 clean-corrupt causal patching。
3. 建立 S_gen 与 S_understand 两个可复现指标。
4. 计算 G(c) / U(c)，确认是否存在差异性组件。
5. 做跨任务 mask，验证是否出现不对称退化。

在没有完成 Stage 1 和 Stage 3 前，不应把结论写成“发现了生成/理解神经元”。更准确的表述应是：

```text
candidate task-selective activations
causally validated task-selective components
task-selective causal pathways
```
