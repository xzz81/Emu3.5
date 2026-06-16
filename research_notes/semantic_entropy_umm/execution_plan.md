# UMM I2T/T2I Semantic Entropy Execution Plan

本文档是一份可执行的目标导向计划，用于在 Emu3.5 fork 中推进统一多模态模型的语义熵研究。

## 0. 最终目标

目标不是证明 image-token entropy 可以和 text-token entropy 比较。目标是建立一个公平框架：

```text
同一个 UMM
同一组语义概念
同一个语义状态空间
分别测 I2T route 和 T2I route 的 semantic entropy
```

核心问题：

```text
当知识相同、语义目标相同、评价空间相同时，
UMM 的图像理解路径 I2T 和图像生成路径 T2I 是否表现出不同的语义不确定性？
```

最终要得到的主要指标：

```text
H_sem(I2T | concept)
H_sem(T2I | concept)
Delta_H = H_sem(T2I | concept) - H_sem(I2T | concept)

Err_sem(I2T | concept)
Err_sem(T2I | concept)
Delta_Err = Err_sem(T2I | concept) - Err_sem(I2T | concept)
```

其中：

- `H_sem` 衡量 route 在语义空间里的不稳定性。
- `Err_sem` 衡量 route 是否偏离目标语义。
- 低熵不等于正确；模型可能稳定地产生错误语义。

## 1. 成功标准

第一阶段成功标准：

1. 构造一个小规模受控概念集。
2. 对同一概念同时跑 I2T 和 T2I。
3. 把两条 route 的输出都解析到同一套语义槽位。
4. 对每个概念计算 `H_sem(I2T)`、`H_sem(T2I)`、`Err_sem(I2T)`、`Err_sem(T2I)`。
5. 产出一张 route 对比表和一份 case 分析。

最小可接受产物：

```text
outputs/semantic_entropy_umm/pilot/
  concepts.jsonl
  i2t_samples.jsonl
  t2i_samples.jsonl
  semantic_slots.jsonl
  route_entropy.csv
  route_error.csv
  case_report.md
```

## 2. 第一阶段只做受控语义空间

先不要做开放式复杂图像。第一阶段只做可控属性：

```text
object: cube, sphere, cone
color: red, blue, green, yellow
position: left, right, center
relation: left_of, right_of, above, below
texture: plain, striped, dotted
```

每个概念写成结构化 target：

```json
{
  "concept_id": "red_cube_left_of_blue_sphere",
  "prompt": "A red cube to the left of a blue sphere on a white background.",
  "target_semantics": {
    "object_1": "cube",
    "color_1": "red",
    "object_2": "sphere",
    "color_2": "blue",
    "relation": "object_1_left_of_object_2",
    "background": "white"
  }
}
```

第一阶段目标不是覆盖真实世界，而是证明 metric 和 pipeline 可以闭环。

## 3. 路线 A：I2T 采样

输入：

```text
reference image + semantic question
```

问题模板：

```text
What is the main object?
What color is the main object?
Where is the main object located?
What object is it next to?
What is the spatial relation between the two objects?
```

执行：

```text
对每张 reference image 重复采样 N 次
固定 decoding config
保存每次 answer 和 token/logprob 信息
```

输出格式：

```json
{
  "concept_id": "red_cube_left_of_blue_sphere",
  "route": "I2T",
  "question": "What color is the cube?",
  "sample_id": 3,
  "raw_output": "The cube is red.",
  "decoded_slots": {
    "color_1": "red"
  }
}
```

## 4. 路线 B：T2I 采样

输入：

```text
same concept prompt
```

执行：

```text
对每个 prompt 重复生成 N 张图
固定 generation config
保存图像、seed、sampling 参数
```

输出格式：

```json
{
  "concept_id": "red_cube_left_of_blue_sphere",
  "route": "T2I",
  "sample_id": 7,
  "image_path": "images/red_cube_left_of_blue_sphere_007.png",
  "prompt": "A red cube to the left of a blue sphere on a white background."
}
```

## 5. 语义解析 phi

这是整个项目最关键的一步。`phi(output)` 必须把 I2T 和 T2I 都映射到同一个语义空间。

I2T:

```text
text answer -> slot labels
```

T2I:

```text
generated image -> VQA / attribute extractor / caption parser -> slot labels
```

第一阶段建议用 rule + VQA 双保险：

1. I2T 文本答案用规则解析，必要时用 LLM judge 归一化。
2. T2I 图像用固定 VQA 问题抽槽位。
3. 对每个 slot 输出 `{value, confidence, evidence}`。

统一格式：

```json
{
  "concept_id": "red_cube_left_of_blue_sphere",
  "route": "T2I",
  "sample_id": 7,
  "slots": {
    "object_1": "cube",
    "color_1": "red",
    "object_2": "sphere",
    "color_2": "blue",
    "relation": "object_1_left_of_object_2"
  },
  "parser": "vqa_slot_extractor_v1"
}
```

## 6. 语义熵计算

对每个概念、每条 route、每个 slot 统计语义分布。

例子：

```text
concept = red_cube
route = T2I
slot = color_1

red: 7
blue: 2
unknown: 1
```

概率：

```text
p(red)=0.7
p(blue)=0.2
p(unknown)=0.1
```

熵：

```text
H_color = - sum_v p(v) log p(v)
```

联合语义状态：

```text
c_i = (object_1, color_1, object_2, color_2, relation)
H_joint = - sum_c p(c) log p(c)
```

必须报告：

```text
H_object
H_color
H_position
H_relation
H_joint
```

## 7. 语义错误计算

熵只表示不稳定，不表示正确性。所以每个 sample 还要和 target semantics 比较。

slot-level error：

```text
err(slot) = 1 if predicted_slot != target_slot else 0
```

concept-level error：

```text
Err_sem = mean_slot_error
```

输出四象限：

```text
low entropy, low error: stable and correct
high entropy, low error: unstable but often correct
low entropy, high error: stable wrong
high entropy, high error: unstable wrong
```

最重要的发现通常是：

```text
T2I low entropy + high error
```

这表示模型稳定生成错误语义，是普通 semantic entropy 容易漏掉的情况。

## 8. 对照实验

至少做三类对照。

### 8.1 Token Entropy Baseline

目的：证明 raw token entropy 不公平。

记录：

```text
I2T text-token entropy
T2I image-token entropy
```

预期结论：

```text
token entropy 受 modality/tokenizer/decoding process 影响，不能解释语义不确定性。
```

### 8.2 Semantic-Equivalent Perturbation

目的：测试 route 对等义输入扰动是否稳定。

I2T：

```text
reference image 做轻微不改语义的变换
question 做 paraphrase
```

T2I：

```text
prompt 做语义等价改写
```

比较：

```text
H_sem(original)
H_sem(perturbed)
```

### 8.3 Contrastive Visual/Language Control

目的：区分视觉证据和语言先验。

I2T：

```text
original image vs distorted image
```

T2I：

```text
full prompt vs prompt with key attribute removed or corrupted
```

看 route semantic distribution 如何变化。

## 9. 内部机制扩展

第一阶段 pipeline 跑通后，再接 Emu3.5 内部信号。

优先级：

1. Hidden-state semantic entropy probe.
2. Vision-aware head divergence.
3. Visual-token uncertainty projected to text/concept logits.
4. Route-specific activation/attention patching.

目标不是替代 `H_sem`，而是解释：

```text
为什么同一个 concept 在 I2T 稳定，在 T2I 不稳定？
为什么某些属性只在生成路径中丢失？
```

## 10. 执行顺序

### Step 1: 建概念集

产物：

```text
concepts.jsonl
```

验收：

```text
至少 30 个概念
每个概念有 prompt 和 target_semantics
覆盖 color/object/position/relation
```

### Step 2: 准备 reference images

产物：

```text
reference_images/
image_manifest.jsonl
```

验收：

```text
每个 concept 有一张 target image
target image 可被 slot extractor 正确识别
```

### Step 3: 跑 I2T 多次采样

产物：

```text
i2t_samples.jsonl
```

验收：

```text
每个 concept 每个 question 至少 N=10 samples
输出可解析到 slots
```

### Step 4: 跑 T2I 多次采样

产物：

```text
t2i_samples.jsonl
generated_images/
```

验收：

```text
每个 concept 至少 N=10 images
保存 seed 和 generation config
```

### Step 5: 统一语义解析

产物：

```text
semantic_slots.jsonl
```

验收：

```text
I2T 和 T2I 共用同一套 slot schema
unknown / ambiguous 必须显式记录
```

### Step 6: 计算熵和错误

产物：

```text
route_entropy.csv
route_error.csv
```

验收：

```text
每个 concept 有 I2T/T2I 对比
每个 slot 有 marginal entropy
每个 concept 有 joint entropy
```

### Step 7: 写 case report

产物：

```text
case_report.md
```

验收：

```text
列出 top route-gap cases
列出 stable-wrong cases
列出 high-entropy-but-correct cases
给出图像和文本样例
```

## 11. 第一轮结果应该回答的问题

第一轮实验结束后，必须能回答：

1. 哪些语义槽位在 T2I 中更不稳定？
2. 哪些语义槽位在 I2T 中更不稳定？
3. route entropy gap 是否集中在 color、position、relation 等特定属性？
4. 是否存在 T2I 稳定错误但低熵的样本？
5. token entropy 和 semantic entropy 是否一致？如果不一致，差在哪里？
6. 同一 UMM 的两个 route 是否表现出 route-specific semantic accessibility？
