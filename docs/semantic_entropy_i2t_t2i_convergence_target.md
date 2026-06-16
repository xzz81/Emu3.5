# I2T/T2I 语义熵收敛目标

更新时间：2026-06-06

本文档记录 Emu3.5 UMM 语义熵实验的收敛目标。目标不是证明 I2T 和 T2I 天然相同，也不是证明 text entropy 与 image entropy 可以直接比较，而是证明两条 route 最终被测量的是同一个语义随机变量，并且测量过程没有系统性偏向其中一条 route。

当前 T2I 基线先保留 `model/Emu3.5`，暂不把主实验切到 `Emu3.5-Image`。`Emu3.5-Image` 是更强的专用 T2I/X2I 模型，但旧的 30 张近乎全白或条纹图不应被解释为 `Emu3.5` 必然无法生图；那批结果主要是生成配置跑坏了。

因此，正式语义熵实验不能再使用 `16x16` visual grid 和 `max_new_tokens=360` 这类 smoke-test 参数。正式 T2I 默认参数至少应为 `target_height=64`、`target_width=64`、`image_area=1048576`、`t2i_max_new_tokens=5120`。否则实验会把“模型没按正常配置画出来”混进“语义不确定性”，导致 T2I entropy 被污染。

人话版判断：直接用 `Emu3.5` 生图会比 `Emu3.5-Image` 差，尤其在画质、细节、prompt 跟随、多物体组合和左右关系上更容易出错；但它不应该差到旧 30 张那种几乎全白或条纹的程度。那批图不能作为模型能力结论，只能作为低配 smoke-test 参数跑坏的证据。当前主张是先继续用 `Emu3.5` 做统一模型主实验，把 `Emu3.5-Image` 留作后续模型族对照；但所有正式 T2I 默认值必须使用下面的高配参数，不能再落回 `16x16/360`。

当前工作主张：

```text
Keep Emu3.5 as the default T2I baseline for now.
Do not switch the main result to Emu3.5-Image unless the experiment explicitly asks for a model-family comparison.
Treat the old 30 nearly blank/striped images as a bad-configuration artifact, not as a model-capability conclusion.
Use formal T2I defaults by default: 64x64 visual grid, image_area=1048576, max_new_tokens=5120, CFG=2.0, image_top_k=5120, image_temperature=1.0.
```

核心目标可以写成：

```text
H(S | concept, route = I2T)
H(S | concept, route = T2I)
```

其中 `S` 是同一个有限语义状态空间中的 normalized semantic state，而不是文本 token、图像 token、自由文本回答或像素空间。

## 1. 核心可比性定义

不可接受的 claim：

```text
text entropy ~= image entropy
```

可接受的 protocol-level claim：

```text
I2T and T2I outputs are projected into the same finite semantic state space S,
and entropy is estimated over H(S | concept, route).
```

在当前 UMM 设置中：

```text
S = (object_1, color_1, object_2, color_2, relation, background)
```

每个样本必须满足：

```text
S_{c,r,k} in S

c = concept
r in {I2T, T2I}
k = sample index
```

因此比较对象是同一 concept 条件下的两个 route-level semantic-state 分布：

```text
H(S | c, r = I2T)
H(S | c, r = T2I)
```

不是：

```text
H(text tokens)
H(image tokens)
```

## 2. 必须满足的可比性性质

### 2.1 同一个语义随机变量

I2T 和 T2I 都必须映射到同一个 semantic state `S`。如果 I2T 测的是文本回答，而 T2I 测的是生成图片本身，则不可比；只有二者都被归一化为同一组 semantic slots 后，才有初步可比性。

### 2.2 同一个 concept 条件

每个 comparison 必须在同一个 concept 下做。每个 concept 应该有明确 gold semantic state，例如：

```text
canonical_prompt = red cube left of blue sphere
gold_object_1 = cube
gold_color_1 = red
gold_object_2 = sphere
gold_color_2 = blue
gold_relation = object_1_left_of_object_2
gold_background = white
```

需要保留公开可审计的 `concepts.jsonl` 或 appendix table，至少包含：

```text
concept_id
canonical_prompt
gold_object_1
gold_color_1
gold_object_2
gold_color_2
gold_relation
gold_background
```

### 2.3 同一个 slot schema

当前收敛 schema 固定为 6 个 slots：

```text
object_1
color_1
object_2
color_2
relation
background
```

I2T 和 T2I 必须完全使用同一批 slots。每个 semantic state 必须 exactly 6 slot answers。

严格审计目标：

```text
num_states = 600
num_slot_answers = 3600
3600 = 600 * 6
unique(concept_id, route, sample_idx) = 600
unique(concept_id, route, sample_idx, slot) = 3600
```

### 2.4 同一个有限 value vocabulary

当前收敛 vocabulary：

```text
object slots: cube, sphere, cone, unknown
color slots: red, blue, green, yellow, unknown
relation: object_1_left_of_object_2,
          object_1_right_of_object_2,
          object_1_above_object_2,
          object_1_below_object_2,
          unknown
background: white, other, unknown
```

要求：

- 所有输出必须落在预定义 vocabulary 内。
- invalid parse 不能静默丢弃。
- `unknown` 必须单独统计。
- 低 entropy 不能自动解释为可靠，因为它也可能是稳定输出 `unknown` 或稳定错误。

主表必须同时报告：

```text
entropy
error
unknown rate
invalid parse rate
```

### 2.5 同一个 extractor / parser

I2T 和 T2I 的 semantic-state extraction 必须共享同一套测量流水线：

```text
same forced-choice VQA prompt
same extractor model
same decoding setting
same parser
same normalization rule
same fallback rule
```

extractor 不能成为隐藏变量。当前 strict compare 的收敛目标是让 I2T reference images 和 T2I generated images 都通过同一个 forced-choice VQA extractor 进入同一语义状态空间。

### 2.6 同一个 sampling budget

当前 strict protocol：

```text
30 concepts
2 routes
10 states per concept-route
30 * 2 * 10 = 600 semantic states
```

这只是最低公平性。后续必须补 sample-size sensitivity：

```text
n = 5, 10, 20, 30
```

并报告：

```text
raw entropy
normalized entropy by log(n)
effective number of states = exp(H)
bootstrap confidence interval
```

因为每个 concept-route 只有 10 samples 时，经验最大 state entropy 是 `log(10)`，不是整个 state space 的最大熵。

### 2.7 route 差异是被测对象，不是测量偏差

I2T route：

```text
reference image -> VQA extraction -> semantic state
```

T2I route：

```text
text prompt -> generated image -> VQA extraction -> semantic state
```

两条路线的物理过程并不对称。应明确承认：

```text
T2I route entropy includes semantic variability introduced during image generation,
while I2T route entropy reflects semantic variability in visual understanding under
the same semantic extraction protocol.
```

因此本文比较的是 route-level semantic uncertainty，不是纯粹 internal uncertainty，也不是原始模态 token uncertainty。

## 3. 硬性实验要求

### A. Protocol Audit

最低门槛是 `comparability_audit.md`。如果这一步不过，不进入 entropy 解释。

| Audit item | Expected |
|---|---:|
| concept count | 30 |
| routes | I2T, T2I |
| states per concept-route | exactly 10 |
| total states | 600 |
| slots per state | exactly 6 |
| total slot answers | 3600 |
| legal value rate | 100% |
| invalid parse count | 0 or reported |
| unique state key | 600 |
| unique slot key | 3600 |
| missing state count | 0 |
| duplicated state count | 0 |
| each route-slot answers | 300 |

当前 strict compare 已覆盖这一组要求。

### B. Extractor Validity

下一优先级是证明 semantic state 不是 extractor 幻觉。

最小 human annotation check：

```text
I2T reference states/images: 30-50
T2I generated images: 30-50
manual slots: object_1, color_1, object_2, color_2, relation, background
```

报告：

```text
slot-level accuracy
macro F1
relation accuracy
unknown agreement
Cohen's kappa or Krippendorff's alpha
```

必须单独看：

```text
relation slot
object binding
color-object binding
```

因为这些是最容易被审稿人攻击的点。

可选增强：

- Oracle synthetic image check：程序生成 gold label 确定的 cube/sphere/cone/color/relation/background 图像，验证 extractor 本身是否可靠。
- Cross-extractor agreement：用 Emu3.5、Qwen2.5-VL、GPT-4o、InternVL、LLaVA-like model 或 synthetic rule-based parser 交叉验证趋势。

### C. Entropy Estimator Stability

必须证明 entropy 数值稳定，而不是 `n=10` 的偶然估计。

Sample-size sensitivity：

```text
n = 5, 10, 20, 30
mean H_I2T
mean H_T2I
Delta H = H_T2I - H_I2T
```

Bootstrap：

```text
H(c, I2T)
H(c, T2I)
Delta H(c)
mean Delta H
95% bootstrap CI
median Delta H
Wilcoxon signed-rank test
```

统计单位必须是 concept，而不是 600 个 state。不要把同一 concept 下的 samples 当成独立样本做 t-test。

Normalized entropy：

```text
H_norm(slot j) = H(j) / log(|V_j|)
H_norm(state) = H / log(n)
N_eff = exp(H)
```

因为不同 slot 的 vocabulary size 不同，slot-level entropy 必须归一化后比较。

### D. Error + Unknown 四象限

不能只报 entropy。核心解释必须联合：

```text
route entropy
route error
unknown rate
invalid parse rate
```

四象限解释：

| Entropy | Error | Interpretation |
|---|---|---|
| low | low | 稳定且正确 |
| low | high | 稳定但系统性错误 |
| high | low | 多样但大体正确 |
| high | high | 不稳定且错误 |

`low entropy + high error` 是重要 case：它说明 route 不是不确定，而是稳定地产生错误语义。

### E. Route Fairness Controls

需要证明结果不是 prompt、选项顺序或 extractor bias。

建议补：

- Prompt template robustness：换 forced-choice VQA prompt 模板，检查趋势是否稳定。
- Value order randomization：随机化选项顺序，尤其检查 `unknown` 的位置偏差。
- Paraphrased concept prompts：对 T2I prompt 做 paraphrase，看 T2I 熵是否主要由 wording 诱发。
- Image perturbation robustness：对 I2T reference images 做 resize、compression、small crop、brightness jitter，确认视觉轻扰动下结果稳定。

## 4. Sanity Checks

必须优先补以下 sanity checks。

### 4.1 Random Concept Control

错配 concept 和 image：

```text
prompt: red cube left of blue sphere
image: green cone above yellow cube
```

正常 metric 应该看到 error rate 上升，entropy 可能上升或稳定错误。

### 4.2 Easy vs Hard Concept Split

把 concept 分成 easy/hard，检查 hard concepts 是否有更高 entropy/error。如果所有 concept entropy 都差不多，说明 metric 可能不敏感。

### 4.3 Single-object Ablation

先测单物体：

```text
red cube on white background
blue sphere on white background
```

如果单物体场景都不稳定，两物体 relation 结果更难解释。

### 4.4 Slot Ablation

分别只测 object、color、relation、background，确认差异来源：

```text
T2I high entropy 是否主要来自 relation?
I2T error 是否主要来自 color-object binding?
unknown 是否集中在 background?
```

### 4.5 Duplicate Image / Mode Collapse Check

T2I 低 entropy 可能是稳定，也可能是 mode collapse。需要报告：

```text
generated image duplicate rate
CLIP/image embedding similarity
perceptual hash duplicate rate
```

低 entropy + 高 duplicate rate 必须谨慎解释。

## 5. 三层可比性边界

### 5.1 Protocol Comparability

含义：

```text
I2T and T2I use the same concept set, slot schema, value vocabulary,
extractor, parser, entropy estimator, error definition, and sampling budget.
```

如果 strict audit PASS，可以说：

```text
The two routes are comparable under the same semantic state protocol.
```

### 5.2 Measurement Validity

含义：

```text
The semantic state actually corresponds to the image/text semantics,
instead of being an extractor artifact.
```

需要 extractor validation、human check、oracle synthetic check。

如果通过，可以说：

```text
The shared semantic state space is a valid operationalization of route-level meaning.
```

### 5.3 Scientific Interpretation

含义：

```text
I2T/T2I entropy differences reflect route-specific understanding/generation behavior.
```

需要 concept-level paired statistics、robustness checks、case studies、failure taxonomy，必要时还需要 model comparisons。

如果通过，可以说更强 claim，例如：

```text
T2I exhibits higher semantic dispersion than I2T on relation-sensitive concepts.
```

## 6. Reviewer Concern 与防守目标

潜在弱拒理由：

```text
The paper proposes to compare semantic entropy across I2T and T2I routes.
However, it is unclear whether the two routes induce distributions over the
same random variable. I2T produces textual answers conditioned on images,
whereas T2I produces images conditioned on text. Without a shared semantic
state space, identical slot schema, value vocabulary, extractor, parser, and
sampling budget, the reported entropy values are not directly comparable.
```

strict compare 正在解决这一点。

但还需要继续防守：

```text
Even if the protocol is balanced, the validity of the semantic-state extractor
is not established. The observed entropy may reflect extractor uncertainty,
object-binding errors, option-order bias, or unknown overuse rather than genuine
route-level semantic uncertainty.
```

因此后续收敛必须补 extractor validity、robustness、bootstrap 和四象限 error/unknown analysis。

## 7. 最终论文表格目标

### Table 1: Comparability Audit

| item | expected | observed | pass |
|---|---:|---:|---|
| concepts | 30 | 30 | yes |
| routes | 2 | 2 | yes |
| states per concept-route | 10 | 10 | yes |
| total states | 600 | 600 | yes |
| slots per state | 6 | 6 | yes |
| total slot answers | 3600 | 3600 | yes |
| legal values | 100% | TBD | TBD |
| duplicate states | 0 | TBD | TBD |
| missing states | 0 | TBD | TBD |
| invalid parses | 0 or reported | TBD | TBD |

### Table 2: Extractor Validation

| slot | extractor-human agreement | accuracy | unknown rate |
|---|---:|---:|---:|
| object_1 | TBD | TBD | TBD |
| color_1 | TBD | TBD | TBD |
| object_2 | TBD | TBD | TBD |
| color_2 | TBD | TBD | TBD |
| relation | TBD | TBD | TBD |
| background | TBD | TBD | TBD |

### Table 3: Route Entropy and Error

| route | state entropy | normalized entropy | effective states | error | unknown rate |
|---|---:|---:|---:|---:|---:|
| I2T | TBD | TBD | TBD | TBD | TBD |
| T2I | TBD | TBD | TBD | TBD | TBD |
| Delta T2I-I2T | TBD | TBD | TBD | TBD | TBD |

### Table 4: Slot-level Breakdown

| slot | H I2T | H T2I | Delta H | Err I2T | Err T2I | Delta Err |
|---|---:|---:|---:|---:|---:|---:|
| object_1 | TBD | TBD | TBD | TBD | TBD | TBD |
| color_1 | TBD | TBD | TBD | TBD | TBD | TBD |
| object_2 | TBD | TBD | TBD | TBD | TBD | TBD |
| color_2 | TBD | TBD | TBD | TBD | TBD | TBD |
| relation | TBD | TBD | TBD | TBD | TBD | TBD |
| background | TBD | TBD | TBD | TBD | TBD | TBD |

### Table 5: Robustness

| setting | Delta H | Delta Err | conclusion |
|---|---:|---:|---|
| original prompt | TBD | TBD | TBD |
| paraphrased prompt | TBD | TBD | TBD |
| randomized option order | TBD | TBD | TBD |
| alternative extractor | TBD | TBD | TBD |
| n=5 samples | TBD | TBD | TBD |
| n=20 samples | TBD | TBD | TBD |
| easy concepts only | TBD | TBD | TBD |
| hard concepts only | TBD | TBD | TBD |

## 8. 优先级

如果时间有限，后续最优先补三组实验。

第一优先级：human/extractor validation。

最小版本：随机抽 60 张图，30 reference + 30 generated，人工标注 6 个 slots，并和 extractor 输出比较。

第二优先级：unknown + error + entropy 四象限分析。

建议画图：x-axis 为 normalized entropy，y-axis 为 error，point 为 concept-route，颜色或 marker 表示 route。

第三优先级：sample-size / bootstrap stability。

至少补：

```text
bootstrap CI
concept-level paired test
effective number of states
```

## 9. 最终可接受 Claim

只完成 strict audit 时，可以接受：

```text
Under a controlled semantic-state protocol, I2T and T2I outputs can be
normalized into the same finite semantic space.
```

完成 extractor validation 后，可以接受：

```text
The normalized semantic states provide a valid operational measurement of
route-level semantics.
```

再完成 bootstrap、robustness、error/unknown analysis 后，可以接受：

```text
I2T and T2I route-level semantic entropy are empirically comparable under this
UMM semantic-state protocol, and their differences reveal route-specific
semantic instability rather than raw modality-token uncertainty.
```

仍然不能接受：

```text
Text entropy and image entropy are directly comparable.
UMM has a unified internal entropy space.
```

除非后续进一步补 hidden-state、causal intervention 或 path-level experiments。

## 10. Reviewer Checklist

1. Same concept set.
2. Same gold semantic state.
3. Same slot schema.
4. Same finite value vocabulary.
5. Same extractor model.
6. Same parser and normalization.
7. Same sampling budget.
8. Same entropy estimator.
9. Same error definition.
10. Validated extractor accuracy.
11. Reported unknown / invalid parse rate.
12. Concept-level paired statistics.
13. Bootstrap confidence intervals.
14. Sample-size sensitivity.
15. Prompt-template robustness.
16. Option-order robustness.
17. Alternative-extractor robustness.
18. Object-binding sanity check.
19. Route-specific failure cases.
20. Clear claim boundary: protocol-level semantic comparability, not raw token-level comparability.

当前 strict compare 如果 `AUDIT PASS`，主要覆盖 checklist 的 1-9 和一部分 11。后续最该补的是 10、12、13、14、18；这些补上之后，实验才会从“工程上可比”升级为“审稿人比较难直接否掉的可比”。
