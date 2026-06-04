# Emu3.5 图像生成与图像理解任务选择性神经元探究方案

## 0. 一句话目标

探究 Emu3.5 在执行图像生成与图像理解任务时，是否存在任务选择性的神经元或内部组件，并进一步验证这些组件是否真的构成因果通路，而不只是激活相关现象。

本方案将问题收敛为：

```text
Do task-selective activations in Emu3.5 correspond to causal generation / understanding pathways?
```

其中：

- 图像生成路径关注 `text prompt -> generated visual tokens`。
- 图像理解路径关注 `image tokens -> text answer tokens`。
- 不把 Emu3.5 误当作 diffusion 模型；核心对象是 decoder-only UMM 内部的 self-attention、MLP、residual stream 和 token-to-token 信息流。

## 1. 核心假设

### H1: 存在任务选择性激活

图像生成与图像理解会在部分层、MLP 中间神经元、attention heads 或 residual stream 位置上产生不同激活模式。

### H2: 任务选择性激活不一定等于因果通路

高激活组件必须经过 ablation、clean/corrupt patching 或 path patching 验证，才能被称为 causal component。

### H3: 更可能是部分共享而非完全分离

Emu3.5 的 generation 与 understanding 可能共享 late-layer visual-token / semantic-token machinery，但在信息方向上存在可区分路径：

```text
generation:    prompt/text tokens -> visual output positions
understanding: visual input tokens -> answer/text output positions
```

## 2. 方法收敛

本研究不采用“大范围泛泛可解释性扫描”，而采用四阶段漏斗：

```text
Stage 0  activation screening
Stage 1  causal ablation
Stage 2  clean/corrupt patching
Stage 3  path-level intervention
```

每一阶段都有明确停机条件：

- Stage 0 只产生候选，不产生结论。
- Stage 1 判断候选是否比随机同层组件更有因果作用。
- Stage 2 判断候选能否恢复 clean/corrupt 语义差异。
- Stage 3 才讨论 generation path 与 understanding path 是否可分离。

## 3. 实验任务

为了降低变量，本阶段只保留 hue / attribute 任务。

### Generation 任务

使用 T2I hue binding：

```text
a red circle
a blue circle
a green square
a purple square
```

观测输出：

```text
generated visual tokens
```

优先指标：

```text
target visual-token NLL / log likelihood
```

### Understanding 任务

使用 image-read hue QA：

```text
image: red circle
question: What color is the circle? Answer with one word.
answer: red
```

观测输出：

```text
answer text tokens
```

优先指标：

```text
correct answer-token NLL / log probability
```

### Clean/Corrupt 对

每组 pair 只改变一个属性：

```text
red circle      <-> blue circle
green square    <-> purple square
yellow triangle <-> cyan triangle
orange diamond  <-> magenta diamond
```

这样可以把问题限定为 attribute information flow，而不是混入复杂场景、计数、布局或图像质量因素。

## 4. Stage 0: Activation Screening

### 目标

找出 generation-over-understanding 和 understanding-over-generation 的候选神经元。

### Hook 位置

优先观测 MLP intermediate：

```text
act(gate_proj(x)) * up_proj(x)
```

这是比 residual stream 更接近“MLP 神经元”的位置。

### 分组

所有激活按以下维度聚合：

```text
task       = generation / understanding
phase      = prompt / output
token_type = text / visual / structure
layer
neuron
```

### 已有结果

当前 replay pilot 已经确认 generation output visual tokens 和 understanding output text tokens 都进入分析，不再是 prefill-only 对比。

已筛出的强候选集中在 late layers，尤其 layer 61-63。

Top generation visual candidates：

```text
(62, 967)
(61, 5624)
(61, 2777)
(63, 17777)
(62, 18848)
```

Top understanding visual candidates：

```text
(63, 6914)
(63, 21052)
(63, 15540)
(63, 17982)
(63, 9339)
```

### 判据

Stage 0 只能支持：

```text
某些神经元在两类任务中存在差异性激活。
```

不能支持：

```text
这些神经元构成因果路径。
```

## 5. Stage 1: Causal Ablation

### 目标

验证 Stage 0 候选是否比同层随机神经元更影响任务输出 likelihood。

### 干预

对选定 MLP 中间神经元做 contribution ablation：

```text
mlp_output' = mlp_output - down_proj(intermediate[selected_neurons])
```

### 分数

```text
delta NLL = ablated NLL - baseline NLL
```

正值表示 ablation 让目标输出更不可能。

### 已有结果

当前 top-vs-random ablation 支持一个较弱但清晰的结论：

```text
Top U MLP neurons have stronger causal effect on image-read answer likelihood than same-layer random neurons.
```

关键结果：

```text
understanding_topU:    +0.005123 NLL/token
understanding_randomU: +0.000055 NLL/token
understanding_topG:    +0.000416 NLL/token
```

但 generation 侧证据较弱：

```text
generation_topG:       +0.000130 NLL/token
generation_randomG:    +0.000091 NLL/token
generation_topU_cross: +0.000561 NLL/token
```

这说明：

- U candidates 对理解答案 likelihood 有因果影响。
- G candidates 还不能被称为 robust generation-causal neurons。
- U candidates 对 generation visual-token likelihood 也有影响，提示 late-layer 组件可能部分共享。

## 6. Stage 2: Clean/Corrupt Patching

### 目标

把问题从“ablation 会不会伤害输出”推进到“clean activation 能否恢复 corrupt run 的正确语义”。

### 通用范式

```text
clean run:   clean image / prompt -> clean target
corrupt run: corrupt image / prompt -> clean target
patched run: corrupt run + clean activation patch
```

恢复分数：

```text
recovery = (corrupt_nll - patched_nll) / (corrupt_nll - clean_nll)
```

### 已有 MLP-neuron patching 结论

理解任务 clean/corrupt 构造有效：

```text
sum_corrupt_minus_clean_nll_sum = 78.53
```

但 top U MLP-neuron patching 没有稳健恢复 clean answer：

```text
topU_prompt mean recovery:    +0.000548
topU_scorepos mean recovery:  -0.000036
random controls:              comparable or better
```

因此当前不能声称：

```text
Top U MLP neurons form a clean/corrupt recoverable image-to-answer causal path.
```

更合理的解释是：

```text
Top U neurons are output-likelihood-sensitive, but the real image-to-answer path is distributed across residual stream, attention heads, or token-to-token edges.
```

### 已有 Residual Stream Patching 结论

在 MLP-neuron patching 失败后，已进一步测试 late-layer residual stream patching。

目标层：

```text
layer 61
layer 62
layer 63
```

目标位置：

```text
score positions
```

对于 one-token answer，score position 是预测答案首 token 的最后 prompt position，而不是 answer token 自身位置。

输出目录：

```text
outputs/understanding_residual_patch_hue_layers61_63_scorepos/
```

关键结果：

```text
layer 61: 8/8 directions helped, mean recovery +1.003642
layer 62: 8/8 directions helped, mean recovery +1.002299
layer 63: 8/8 directions helped, mean recovery +1.000000
```

其中 layer 63 是直接进入 answer-token logits 前的最终表征，主要作为 upper-bound sanity check。更有信息量的是 layer 61/62：它们在 8 个 clean/corrupt 方向上也几乎完全恢复 clean answer NLL。

进一步做 token-scope 对照：

```text
score-positions layer 61: mean recovery +1.003642
score-positions layer 62: mean recovery +1.002299

visual-prompt layer 61:  mean recovery +0.030955
visual-prompt layer 62:  mean recovery +0.009157

eoi-prompt layer 61:     mean recovery +0.025519
eoi-prompt layer 62:     mean recovery +0.020008
```

对比报告：

```text
outputs/understanding_residual_patch_hue_stage2_comparison_report.md
```

这说明 late-layer answer score position 是强信息汇合点；仅 patch visual prompt tokens 或单个 EOI-like token，只能恢复很小一部分 clean/corrupt gap。

因此，当前证据支持：

```text
The image-to-answer causal path exists at the late residual-stream level,
but it is not localized to the tested small set of MLP intermediate neurons.
```

## 7. Stage 3: Path-Level Intervention

由于 residual stream patching 已证明存在可恢复通路，并且 token-scope 对照显示 late visual tokens / EOI-only residual patching 恢复较弱，进一步进入 attention/path-level 分解。

### Understanding Path

参考 Narrow Gate / VLM causal tracing 思路，优先测试：

```text
visual tokens / EOI-like token -> answer score positions
```

候选干预：

- block answer positions attending to image tokens;
- block answer positions attending to EOI-like token;
- patch attention head outputs;
- patch visual-token residual stream into answer score positions.

判据：

```text
若屏蔽少量 image-side token 或 EOI-like token 显著降低 answer likelihood，
且随机 token control 不显著降低，
则该 token/edge 是 understanding gate。
```

当前 residual patching 对 EOI-only gate 的证据较弱：

```text
eoi-prompt residual patching at layer 61/62 recovers only about 2% of the clean/corrupt gap.
```

因此进一步直接测试 attention edge：

```text
answer score position --attends-to--> EOI
answer score position --attends-to--> visual code tokens
```

Stage 3 attention knockout 结果：

```text
visual-prompt individual layer sweep 48/52/56/60/61/62:
max mean harm = +0.007630 at layer 61

eoi-prompt individual layer sweep 48/52/56/60/61/62:
max mean harm = +0.002920 at layer 61

image-prompt individual layer sweep 48/52/56/60/61/62:
max mean harm = +0.004225 at layer 52

image-prompt joint knockout at layers 48+52+56+60+61+62:
mean harm = -0.011454

image-prompt joint knockout at all 64 layers:
mean harm = -0.012591
```

对比报告：

```text
outputs/understanding_attention_knockout_hue_stage3_comparison_report.md
```

解释：

```text
direct answer-score-position -> image/EOI attention edge is not a necessary bottleneck.
```

因此，Stage 2 中几乎完整可恢复的 answer score-position residual state，不能被简单归因于 answer score position 直接 attend 到 visual tokens 或 EOI。更可能的 mediated route 是：

```text
image tokens -> intermediate text/question positions -> answer score position
```

Stage 4 进一步测试 mediated route：

```text
question-text-no-score residual patching:
layer 48 mean recovery +0.503115
layer 52 mean recovery +0.433818
layer 56 mean recovery +0.428705
layer 60 mean recovery +0.575835
layer 61 mean recovery +0.576455
layer 62 mean recovery +0.328967

post-image-text-prompt residual patching:
layer 48 mean recovery +0.926293
layer 52 mean recovery +0.928660
layer 56 mean recovery +0.944795
layer 60 mean recovery +0.996765
layer 61 mean recovery +0.997970
layer 62 mean recovery +0.995240
```

对比报告：

```text
outputs/understanding_mediated_route_hue_stage4_report.md
```

这说明 post-image question text residual states 已经携带大量 image-conditioned hue information。即使排除最终 answer score position，question text tokens 仍可恢复 33%-58% 的 clean/corrupt gap。

对应的 attention knockout 结果更弱且方向依赖：

```text
all-layer image -> score-position attention knockout:
mean harm = -0.012591

all-layer image -> question-text-no-score attention knockout:
mean harm = -0.050808

all-layer image -> post-image-text attention knockout:
mean harm = +0.096136
```

因此当前更准确的理解侧路径假设是：

```text
image tokens influence a distributed text-side residual representation;
the final answer score state reads from this text-side representation;
the route is not captured by a single direct attention edge or EOI bottleneck.
```

Stage 5 进一步做 module-output patching，把 text-side representation 拆到 self-attention output 与 MLP output：

```text
question-text-no-score module patching:
layer 60 MLP mean recovery       +0.096475
layer 60 self-attn mean recovery +0.017457
layer 62 MLP mean recovery       +0.051844
layer 62 self-attn mean recovery +0.006956

post-image-text-prompt module patching:
layer 60 MLP mean recovery       +0.230632
layer 60 self-attn mean recovery +0.037657
layer 62 MLP mean recovery       +0.207666
layer 62 self-attn mean recovery +0.202060
```

对比报告：

```text
outputs/understanding_module_patch_hue_stage5_report.md
```

解释：

```text
single module outputs contribute locally but are much weaker than full residual patching.
```

当前理解侧路径更精确地表述为：

```text
image-conditioned hue information is accumulated in post-image text residual states;
layer 60/62 MLP outputs and layer 62 self-attention output provide local contributions;
the full representation is distributed across residual stream rather than isolated in one module output.
```

Stage 6 进一步把 layer 62 self-attention 拆到 head level：

```text
individual head patching at layer 62, post-image-text-prompt:
head 36 mean recovery +0.051870
head 38 mean recovery +0.047141
head 46 mean recovery +0.042125
head 34 mean recovery +0.023925
head 28 mean recovery +0.023718
head 39 mean recovery +0.023428
head 3  mean recovery +0.022073

joint head patching:
heads 36+38+46 mean recovery                         +0.152832
heads 3+28+34+36+38+39+46 mean recovery              +0.264329
heads 3+25+28+33+34+36+38+39+46+52 mean recovery     +0.290696

full layer 62 self-attn module output mean recovery   +0.202060
```

对比报告：

```text
outputs/understanding_attention_head_patch_hue_stage6_report.md
```

解释：

```text
layer 62 self-attention contribution is concentrated in a small head set,
especially heads 36, 38, and 46.
```

这也是目前最强的 head-level localization。注意 top-10 heads 的联合 patch recovery 高于 full self-attn module patch，提示同层其他 heads 可能含有负贡献或与 clean/corrupt intervention 不一致的贡献。

Stage 7 进一步回到 MLP side，测试 layer 60/62 的 MLP intermediate neuron patching：

```text
patch scope = post-image-text-prompt
candidate source = U-over-G activation contrast / layer-local top activation
control = same-layer deterministic random neurons
```

关键结果：

```text
layer 60 U-over-G contrast top5 mean recovery     +0.001515
layer 60 random5 mean recovery                    +0.000046

layer 62 U-over-G contrast top8 mean recovery     -0.000499
layer 62 random8 mean recovery                    -0.000791

layer 60 topactive10 mean recovery                +0.002243
layer 60 random10 mean recovery                   +0.001160

layer 62 topactive10 mean recovery                +0.000975
layer 62 random10 mean recovery                   -0.000590
```

对比：

```text
layer 60 full MLP module output mean recovery      +0.230632
layer 62 full MLP module output mean recovery      +0.207666
layer 62 top10 attention heads mean recovery       +0.290696
```

对比报告：

```text
outputs/understanding_mlp_neuron_patch_hue_stage7_report.md
```

解释：

```text
tested small MLP intermediate-neuron sets do not explain the MLP module recovery.
```

因此，当前 MLP 侧最准确的说法不是“找到了一小组 understanding path neurons”，而是：

```text
layer 60/62 MLP modules contribute to the recoverable text-side representation,
but their contribution is distributed across many intermediate neurons or feature directions.
Activation-selected sparse neuron sets are not enough to recover clean/corrupt hue information.
```

### Generation Path

参考 LocoGen / Prompt-to-Prompt 的问题定义，但改写到 decoder-only UMM：

```text
which prompt tokens control generated visual-token regions?
```

优先测试：

```text
prompt color token -> visual output positions
prompt object token -> visual output positions
prompt spatial token -> visual output positions
```

候选干预：

- prompt-token residual patching;
- visual-output-position residual patching;
- text-to-visual attention edge knockout;
- attention head output patching.

判据：

```text
若替换 red -> blue 的 clean/corrupt patch 能改变目标 visual-token likelihood，
或 decoded image 的 attribute classifier / VQA 判断随之变化，
则该 path 参与 generation grounding。
```

Stage 8 已启动 generation-side residual patching，对应 understanding Stage 2 的路径存在性测试：

```text
clean prompt -> clean generated visual-token target
corrupt prompt -> same clean visual-token target
patched corrupt prompt -> same clean visual-token target
```

新增脚本：

```text
scripts/run_generation_residual_patch_recovery.py
```

该脚本支持：

```text
patch scope = prompt / color-token / visual-score-positions / all-score-positions
score scope = visual / visual-and-structure / all-target
resolution override = target-height / target-width / image-area
```

重要方法修正：

```text
only directions with corrupt_nll > clean_nll are valid recovery directions.
```

因为 generation 的 teacher-forced visual-token likelihood 会出现方向性不稳定；若 corrupt prompt already assigns lower NLL to clean target，则 recovery denominator 不是语义 clean/corrupt gap。

已完成的 hue-only 16x16 all-pairs sweep：

```text
cfg = configs/ume_main_t2i_counterfactual_pairs_hueonly_seed69.py
layer = 62
pairs = red/blue circle, green/purple square, yellow/cyan triangle, orange/magenta diamond
valid directions = 8/8
target cache entries = 8
```

聚合结果：

```text
prompt / visual:
  mean recovery -0.013560, median +0.001762, helped 4/8

color-token / visual:
  mean recovery +0.007586, median -0.005960, helped 3/8

prompt / visual-and-structure:
  mean recovery -0.437912, median -0.043384, helped 1/8

color-token / visual-and-structure:
  mean recovery -0.020293, median -0.001982, helped 3/8
```

方法改进：

```text
clean generated target token sequences are cached and reused across prompt/color/scope sweeps.
color-token patching supports source-to-target span mapping when color tokenization lengths differ.
```

进一步做 cached target hue validation：

```text
outputs/generation_target_hue_cache_validation_hueonly16/target_hue_validation_report.md
decoded targets = 8/8
simple HSV hue matches = 0/8
```

这说明当前 16x16 clean generated targets 本身并不稳定对应 prompt hue；generation 侧 teacher-forced visual-token likelihood 同时受到 causal path 与 target quality/metric noise 影响。

进一步做 direct patched decoded-generation probe：

```text
scripts/run_generation_patched_decode_hue.py
outputs/generation_patched_decode_hue_layer62_colortoken_allpairs/patched_decode_hue_report.md

layer = 62
patch scope = color-token
target grid = 16x16
variants = clean / corrupt / patched_corrupt
decoded images = 24/24

clean hue matches           = 0/8
corrupt hue matches         = 0/8
patched_corrupt hue matches = 1/8
```

其中唯一正例是：

```text
blue -> red corrupt:
corrupt predicted red, patched_corrupt predicted blue
```

这说明 direct decoded-image 级别出现了一个孤立的 hue restoration case，但仍然不是跨方向稳健的 generation hue-grounding path。

进一步恢复 CFG 并加入 conditional-prefix patch gate：

```text
outputs/generation_patched_decode_hue_layer62_colortoken_pair1_cfg/patched_decode_hue_report.md

generation mode = cfg
classifier-free guidance = 2.0
pair = red circle <-> blue circle

clean hue matches           = 0/2
corrupt hue matches         = 0/2
patched_corrupt hue matches = 0/2
```

CFG-aware hook isolation 能跑通，但 16x16 CFG smoke 仍不能解决 clean hue following 问题。

进一步做 clean hue seed sweep：

```text
scripts/run_generation_clean_hue_seed_sweep.py
outputs/generation_clean_hue_seed_sweep_pair1_cfg2/clean_hue_seed_sweep_report.md

generation mode = cfg
classifier-free guidance = 2.0
pair = red circle / blue circle
seeds per sample = 2

decoded = 4/4
hue matches = 1/4
matching seed: red circle, seed 147402
```

这说明 seed filtering 可以找到有效 clean hue generation，但当前 sweep 还不足以构造双向都有效的 clean/corrupt patch benchmark。

随后把 red/blue circle pair 扩展到每个颜色 6 个 seeds：

```text
outputs/generation_clean_hue_seed_sweep_pair1_cfg2_combined/combined_clean_hue_seed_sweep_report.md

red:  1/6 valid clean generations, matching seed 147402
blue: 0/6 valid clean generations
```

这进一步确认当前 red/blue circle pair 还不是可用的双向 generation patch benchmark。

对比报告：

```text
outputs/generation_residual_patch_hue_stage8_report.md
```

当前 generation-side 结论只能写得很谨慎：

```text
The generation-side residual probe is operational and hue-only 16x16 prompts produce valid visual-token clean/corrupt gaps.
However, layer-62 prompt/color-token residual patching does not robustly recover the visual-token NLL gap across all hue-only directions.
Decoded cached targets also fail simple hue validation, suggesting the current likelihood target is noisy.
Direct decoded generation shows one isolated restoration case but no robust all-pairs recovery.
CFG-aware direct generation also fails hue validation in the tested pair.
Seed filtering can find isolated valid clean generations, but the extended red/blue sweep confirms that this pair is not yet a usable bidirectional generation benchmark.
This is not yet robust evidence for a localized generation hue-grounding path.
```

## 8. 对照实验

每个阶段都必须包含以下 controls：

```text
same-layer random neuron / head control
clean-clean patch
corrupt-corrupt patch
random token span patch
matched semantic pair
```

并区分三类效应：

```text
semantic grounding effect
visual-token fluency effect
general language/output likelihood effect
```

否则容易把“输出 token likelihood 下降”误读成“语义路径被破坏”。

## 9. 最终证据标准

### 可以声称“有特定神经元被激活”

需要满足：

```text
activation contrast > random / baseline
stable across samples
stable across token-type grouping
```

### 可以声称“该神经元有因果作用”

需要满足：

```text
ablation effect > same-layer random control
```

### 可以声称“该神经元属于任务通路”

需要满足更强条件：

```text
clean/corrupt patching shows recovery
and random control does not
and effect is task-specific or direction-specific
```

### 可以声称“generation 与 understanding 通路可分离”

需要满足：

```text
S_gen(c) high while S_understand(c) low
or
S_understand(c) high while S_gen(c) low
```

并通过 cross-task intervention 验证：

```text
mask G path during understanding
mask U path during generation
```

## 10. 当前最稳健结论

基于已完成实验，当前可以写成：

```text
We observe late-layer task-selective MLP activations in Emu3.5.
Understanding-selected MLP neurons show stronger ablation effects on answer likelihood than same-layer random controls.
However, neuron-level clean/corrupt patching does not recover the clean answer.
Late-layer score-position residual patching at layers 61-63 almost fully recovers the clean answer across 8 hue directions, showing that the image-to-answer pathway is present but distributed beyond the tested individual MLP neurons.
Direct attention knockout from answer score positions to image/EOI prompt sources produces only weak or negative harm, so the current evidence points toward a mediated route through intermediate text/question positions rather than a direct answer-to-image attention gate.
Question-text residual patching confirms this mediated route: post-image question text states recover a large fraction of the clean/corrupt gap even when the final score position is excluded.
Module-output patching shows that layer 60/62 MLP outputs and layer 62 self-attention output contribute locally, but none approaches full residual recovery.
Head-level patching further localizes the layer 62 self-attention contribution to heads 36/38/46 plus a small supporting set.
MLP intermediate-neuron patching at layers 60/62, however, does not recover clean answers beyond random controls, so the MLP contribution appears distributed rather than localized to the tested sparse neuron sets.
Generation-side residual patching has now been implemented. Initial hue-only 16x16 teacher-forced visual-token experiments produce valid clean/corrupt gaps, but layer-62 prompt/color-token patching recovers only weak, direction-dependent fractions of the visual-token NLL gap.
```

中文表述：

```text
Emu3.5 在图像生成与图像理解任务中确实出现了 late-layer 任务差异性激活。
其中 understanding-selective MLP 神经元对答案 likelihood 有可测的因果影响。
但这些单个 MLP 神经元还不足以构成可 clean/corrupt 恢复的理解通路；
late-layer score-position residual patching 几乎完全恢复 clean answer，
说明真正的 image-to-answer 通路存在，但更可能分布在 residual stream 或 attention edge 中。
同时，late visual-token residual 和 EOI-only residual patching 恢复很弱，
提示答案相关信息在 late layers 已经主要汇聚到 answer score position。
进一步的 attention knockout 显示，answer score position 直接 attend 到 image/EOI tokens 也不是必要瓶颈；
mediated route 实验显示 image-conditioned hue information 已经进入 post-image question/text residual states；
module patching 表明 layer 60/62 MLP 与 layer 62 self-attention 有局部贡献；
head patching 将 layer 62 self-attention 进一步定位到 heads 36/38/46 及少量支持 head；
但 layer 60/62 的小规模 MLP intermediate neuron patching 不能超过随机对照，
说明 MLP 贡献更可能是分布式特征/子空间，而不是少数单神经元通路。
generation 侧已经跑通 residual patching 和 teacher-forced visual-token NLL 指标；
严格 hue-only 16x16 设置能产生有效 clean/corrupt gap，
但 layer 62 prompt/color-token patching 只显示很弱且方向依赖的恢复，
因此 generation hue-grounding path 目前还没有达到 understanding 侧 residual patching 那种强证据水平。
```

## 11. 推荐下一步执行顺序

### Step 1: Mediated Image-to-Question-to-Answer Route

在 direct answer-score-to-image attention knockout 证据较弱的基础上，下一步测试：

```text
visual/image tokens -> question text positions
question text positions -> answer score position
```

当前 residual patching 已支持 question text positions 携带 image-conditioned information。下一步应定位这些 text-side states 是由哪些模块构建的：
当前 module-output patching 已显示 layer 60/62 MLP 与 layer 62 self-attention 有局部贡献。下一步应继续细化到 head/neuron/feature level：

当前 head-level patching 已定位 layer 62 heads 36/38/46。下一步应继续细化 MLP side：

- 不再把 small top-activation neuron lists 作为主线；
- 转向 layer 60/62 MLP 的低秩方向、PCA/SAE-style feature patching 或更大规模 cumulative neuron sweep；
- 继续把 top heads vs MLP feature directions vs full module vs residual recovery 做成同一套对照。

目标：

```text
middle-to-late layers around 48-62
question text token positions
answer score position
```

### Step 2: Residual Token-Scope Decomposition

把 residual patching 从 score positions 扩展到更细 token scope：

```text
visual-prompt positions only
EOI-like token only
non-visual prompt positions
random visual token spans
```

### Step 3: Generation Residual / Edge Patching

理解侧已经证明“特定少量 MLP 神经元”不是当前最强路径，因此下一阶段优先补 generation 侧因果证据。重新定义 generation 侧目标，不只看 visual-token NLL，而要看 hue binding：

```text
red prompt token -> generated red visual-token region
blue prompt token -> generated blue visual-token region
```

优先使用：

- teacher-forced visual-token likelihood;
- decoded image hue classifier;
- VQA attribute checker。

当前 Stage 8 的下一步应优先让 generation 指标稳定，而不是立刻下钻到神经元：

- 增加 decoded image hue validation 或 VQA/classifier checks；
- 改善或过滤 clean generated targets 后再使用 teacher-forced likelihood；
- 改善 generation setting 本身，例如使用更高质量/更高分辨率 valid targets，或先筛选 clean hue following 稳定的 prompt/seed；
- 扩大 seed sweep，先构造 clean/corrupt 双向都有效的 filtered generation benchmark；
- 继续复用 cached clean targets 做必要的 likelihood sweep；
- visual-score-position patching 只有在对齐控制更清楚后再做；
- 若 residual-level generation recovery 变得稳健，再拆到 attention head、MLP module 或 MLP neuron。

### Step 4: Differential Path Score

对每个组件计算：

```text
G_score = S_gen - lambda * S_understand
U_score = S_understand - lambda * S_gen
```

分类：

```text
generation-specialized
understanding-specialized
shared
irrelevant
```

## 12. 论文级 Claim 梯度

### 弱 Claim

```text
Emu3.5 shows task-selective late-layer activations during image generation and image understanding.
```

### 中等 Claim

```text
Understanding-selected late-layer MLP neurons causally affect answer likelihood, but the recoverable image-to-answer pathway is distributed beyond individual neurons.
```

### 强 Claim

```text
Image generation and image understanding in Emu3.5 rely on partially overlapping but directionally distinguishable pathways: prompt-to-visual-token grounding for generation, and visual-token-to-answer-token communication for understanding.
```

目前证据支持弱 Claim 和部分中等 Claim；强 Claim 需要 residual patching 与 attention/path patching 后才能成立。

## 13. 本方案的收敛点

本方案刻意不再同时追逐所有可能任务和所有模块，而是先回答一个最小但关键的问题：

```text
在 hue generation / hue understanding 中，
late-layer task-selective activations 是否能被推进为 causal path evidence?
```

若答案为是，再扩展到 object、spatial relation、counting。

若答案为否，则结论同样有价值：

```text
Emu3.5 的任务差异不是少数单神经元通路，而是分布式 residual / attention communication pattern。
```
