# I2T/T2I 统一语义熵执行文档

更新时间：2026-06-05

本文档是 Emu3.5 UMM 统一语义熵实验的可执行推进文档。它不替代理论边界文档 `docs/semantic_entropy_i2t_t2i_convergence_target.md`；理论边界由该文档定义，本文只说明下一步如何在远端项目中执行、验收和整理论文证据。

## 0. 当前状态

远端主工作区：

```bash
ssh lb-gzs
cd /home/wentao/project/Emu3.5
```

当前 `lb-gzs` 已可通过 Tailscale 直连：

```text
wentao@100.69.207.21:22
```

已验证的 Python 环境：

```bash
PY=./.venv-transformers/bin/python
```

不要用裸 `python3` 跑当前语义熵脚本；系统 Python 缺少实验依赖。不要用 `./.venv-vllm/bin/python` 跑当前 strict compare；该环境的 `transformers` 版本与仓库代码不兼容。

运行实验时固定以下环境变量：

```bash
export HF_HOME="$PWD/.cache/huggingface"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTHONPATH="$PWD"
```

当前已完成的基线：

```text
outputs/semantic_entropy_umm/pilot/
outputs/semantic_entropy_umm/strict_compare/
```

`strict_compare` 已完成并通过 comparability audit：

```text
State rows by route: I2T=300, T2I=300
Total semantic states: 600
Total slot answers: 3600
Schema OK: True
Vocabulary OK: True
Verdict: PASS
```

当前 strict compare 的均值结果：

| Metric | Slot | I2T | T2I | Delta T2I-I2T |
|---|---|---:|---:|---:|
| Entropy | joint | 0.0904 | 0.1074 | 0.0170 |
| Entropy | object_1 | 0.0904 | 0.1074 | 0.0170 |
| Error | object_1 | 0.1833 | 0.3200 | 0.1367 |

其他 slots 当前 entropy 和 error 均为 0.0000。这个结果只能说明 protocol-level comparability 已经建立，不能说明 extractor 已经达到人工真值级别。

本文档以以下远端文件为 source of truth：

```text
docs/semantic_entropy_i2t_t2i_convergence_target.md
docs/semantic_entropy_umm_results_summary_2026-06-04.md
research_notes/semantic_entropy_umm/execution_plan.md
outputs/semantic_entropy_umm/pilot/concepts.jsonl
outputs/semantic_entropy_umm/strict_compare/comparability_audit.md
outputs/semantic_entropy_umm/strict_compare/comparability_audit.json
outputs/semantic_entropy_umm/strict_compare/strict_semantic_states.jsonl
outputs/semantic_entropy_umm/strict_compare/strict_slot_answers.jsonl
outputs/semantic_entropy_umm/strict_compare/strict_route_entropy.csv
outputs/semantic_entropy_umm/strict_compare/strict_route_error.csv
```

## 1. 总目标和禁止 claim

本轮目标不是证明：

```text
text entropy ~= image entropy
```

也不是证明：

```text
UMM has a unified internal entropy space
```

当前可执行目标是证明 I2T 和 T2I 都被投影到同一个有限语义状态空间后，可以比较同一个语义随机变量：

```text
H(S | concept, route = I2T)
H(S | concept, route = T2I)
```

其中：

```text
S = (object_1, color_1, object_2, color_2, relation, background)
```

最终论文 claim 必须限定为：

```text
Under a controlled semantic-state protocol, I2T and T2I outputs are normalized
into the same finite semantic space, and route-level semantic entropy is
estimated over H(S | concept, route).
```

任何结果解释都必须同时报告：

```text
entropy
error
unknown rate
invalid parse rate
```

低 entropy 不等于正确；它可能代表稳定正确、稳定错误、稳定输出 unknown，或 mode collapse。

## 2. 固定协议

当前 strict protocol 固定如下：

```text
concepts: 30
routes: I2T, T2I
states per concept-route: 10
semantic states: 30 * 2 * 10 = 600
slots per state: 6
slot answers: 600 * 6 = 3600
```

Slot schema：

```text
object_1
color_1
object_2
color_2
relation
background
```

Value vocabulary：

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

所有后续实验都必须保持：

```text
same concept set
same gold semantic state
same slot schema
same finite value vocabulary
same extractor model or explicitly labeled alternative extractor
same parser and normalization
same entropy estimator
same error definition
concept-level paired statistics
```

统计单位必须是 concept。不要把 600 个 semantic states 当成 600 个独立样本做 t-test。

## 3. 现有基线复现

只在需要复现或修复现有 strict compare 时运行本节命令。默认不要重复大跑，先读取已有结果。

查看已有 audit：

```bash
cd /home/wentao/project/Emu3.5
PY=./.venv-transformers/bin/python

$PY scripts/run_semantic_entropy_strict_compare.py \
  --mode audit \
  --strict-dir outputs/semantic_entropy_umm/strict_compare
```

查看核心输出：

```bash
wc -l outputs/semantic_entropy_umm/strict_compare/strict_semantic_states.jsonl
wc -l outputs/semantic_entropy_umm/strict_compare/strict_slot_answers.jsonl
head outputs/semantic_entropy_umm/strict_compare/strict_route_entropy.csv
head outputs/semantic_entropy_umm/strict_compare/strict_route_error.csv
```

如果必须重跑 strict compare，优先沿用已有 launcher：

```bash
bash outputs/semantic_entropy_umm/strict_compare/launch_strict_workers.sh
```

GPU 规则：

- 吞吐型 inference/evaluation 默认两张 GPU 一个 worker。
- 如果服务器只有 4 张 A6000，使用 `CUDA_VISIBLE_DEVICES=0,1` 和 `CUDA_VISIBLE_DEVICES=2,3` 两个 worker。
- 如果继续复用已有 `launch_strict_workers.sh`，先确认脚本中的设备号在当前机器存在；不要盲目照抄历史 `5,7` 到 4 卡机器。

复现通过门槛：

```text
comparability_audit.md verdict = PASS
strict_semantic_states.jsonl = 600 lines
strict_slot_answers.jsonl = 3600 lines
strict_route_entropy.csv = 421 lines including header
strict_route_error.csv = 361 lines including header
```

## 4. 阶段 A：Human / Extractor Validation

目的：证明 semantic state 不是 automated VQA extractor 的幻觉或选项偏差。

最小执行规模：

```text
I2T reference images: 30
T2I generated images: 30
total images: 60
manual slots per image: 6
manual slot labels: 360
```

抽样规则：

1. 从 30 个 concepts 中每个 concept 抽 1 张 I2T reference image。
2. 从同一批 concepts 中每个 concept 抽 1 张 T2I generated image。
3. T2I image 优先选 `sample_id=0`；如果文件缺失，选该 concept 下最小可用 sample id。
4. 抽样必须固定输出到 manifest，后续人工标注只基于 manifest，不临时换图。

建议产物：

```text
outputs/semantic_entropy_umm/extractor_validation/
  annotation_manifest.csv
  manual_annotations.csv
  extractor_joined_annotations.csv
  extractor_validation_metrics.csv
  extractor_validation_report.md
```

`annotation_manifest.csv` 字段：

```text
annotation_id
concept_id
route
sample_id
image_path
canonical_prompt
gold_object_1
gold_color_1
gold_object_2
gold_color_2
gold_relation
gold_background
```

`manual_annotations.csv` 字段：

```text
annotation_id
annotator_id
object_1
color_1
object_2
color_2
relation
background
notes
```

人工标注规则：

- 只能使用固定 vocabulary。
- 看不清或无法判断时标 `unknown`，不要猜。
- relation 必须以 object_1 和 object_2 的绑定为准，不要只看图中任意两个物体。
- 如果颜色和物体绑定不确定，在 `notes` 中记录。

需要报告的指标：

```text
slot-level accuracy
macro F1
relation accuracy
object binding accuracy
color-object binding accuracy
unknown agreement
route-level accuracy by I2T / T2I
```

最低验收门槛：

- 每个 slot 都有 I2T 和 T2I 的 accuracy。
- relation、object binding、color-object binding 单独报告。
- 所有 disagreement 都能回溯到 image_path、concept_id、route、sample_id。
- 如果任一关键 slot accuracy 明显不足，后续 entropy 解释必须降级为 extractor-limited evidence。

## 5. 阶段 B：Unknown + Error + Entropy 四象限

目的：避免把低 entropy 误读成可靠。

输入：

```text
outputs/semantic_entropy_umm/strict_compare/strict_route_entropy.csv
outputs/semantic_entropy_umm/strict_compare/strict_route_error.csv
outputs/semantic_entropy_umm/strict_compare/strict_semantic_states.jsonl
outputs/semantic_entropy_umm/strict_compare/strict_slot_answers.jsonl
```

必须新增或计算的字段：

```text
unknown_rate
invalid_parse_rate
normalized_entropy
effective_num_states = exp(H)
quadrant
```

四象限定义：

| Entropy | Error | Interpretation |
|---|---|---|
| low | low | 稳定且正确 |
| low | high | 稳定但系统性错误 |
| high | low | 多样但大体正确 |
| high | high | 不稳定且错误 |

阈值默认：

```text
normalized_entropy low/high: dataset median by slot
error low/high: 0.5
```

如果论文主文需要更稳，补 sensitivity：

```text
error threshold = 0.25, 0.5, 0.75
entropy threshold = median, upper tertile
```

建议产物：

```text
outputs/semantic_entropy_umm/quadrant_analysis/
  concept_route_metrics.csv
  slot_quadrant_metrics.csv
  quadrant_cases.csv
  quadrant_scatter.png
  quadrant_report.md
```

`concept_route_metrics.csv` 最小字段：

```text
concept_id
route
slot
entropy
normalized_entropy
effective_num_states
error_rate
unknown_rate
invalid_parse_rate
quadrant
distribution
```

报告必须包含：

- `low entropy + high error` 的代表性 cases。
- T2I error 是否集中在 `object_1`。
- unknown 是否集中在特定 slot。
- I2T 和 T2I 的 quadrant 分布差异。

验收门槛：

- 每个 concept-route-slot 都有 entropy、error、unknown rate。
- joint state 和 6 个 slot 都有 normalized entropy。
- 至少列出每条 route 最典型的 5 个失败 case。

## 6. 阶段 C：Sample-size / Bootstrap Stability

目的：证明结论不是 `n=10` 小样本偶然值。

优先做不重跑模型的 bootstrap：

```text
unit = concept
paired delta = metric(T2I) - metric(I2T)
resampling = concepts with replacement
iterations = 10000
seed = 20270605
```

必须报告：

```text
mean Delta H
median Delta H
95% bootstrap CI
Wilcoxon signed-rank test
effective number of states
```

统计对象：

```text
joint entropy
slot-level entropy
slot-level error
unknown rate
```

建议产物：

```text
outputs/semantic_entropy_umm/bootstrap_stability/
  concept_level_paired_metrics.csv
  bootstrap_summary.csv
  bootstrap_delta_distributions.csv
  bootstrap_report.md
```

如果需要 sample-size sensitivity，优先在已有 10 samples 内做 subsampling：

```text
n = 5, 10
subsample iterations = 1000
seed = 20270605
```

只有当 `n=10` 结果仍不稳定或论文需要更强证据时，再扩大实验到：

```text
n = 20, 30
```

扩大样本前必须先写清：

- 要重跑哪些 routes。
- 是否重新生成 T2I images。
- 是否保留原 strict compare 结果为 baseline。
- GPU 使用和预计运行时间。
- 新输出目录，不覆盖 `strict_compare`。

验收门槛：

- 所有显著性或不确定性结论都以 concept 为统计单位。
- bootstrap CI 覆盖 joint 和 `object_1`，因为当前差异主要出现在这两个指标。
- 如果 CI 跨 0，论文 claim 必须从“路线差异显著”降级为“protocol 可比且当前 pilot 显示弱趋势”。

## 7. 阶段 D：Robustness Controls

目的：防守 prompt、选项顺序、extractor 和 object binding 相关质疑。

优先级 1：option order robustness。

默认设计：

```text
same images
same concepts
same slots
same extractor model
shuffle allowed value order
seeds = 20270605, 20270606, 20270607
```

输出目录：

```text
outputs/semantic_entropy_umm/robustness_option_order/
```

验收：

- 比较 original vs shuffled option order 的 value 分布。
- 单独检查 `unknown` 是否因位置变化而显著变化。
- 报告 relation 和 object_1 的稳定性。

优先级 2：prompt template robustness。

默认设计：

```text
template_v1 = current strict forced-choice VQA prompt
template_v2 = shorter direct label prompt
template_v3 = question-first then options prompt
```

输出目录：

```text
outputs/semantic_entropy_umm/robustness_prompt_template/
```

验收：

- Delta H 和 Delta Err 趋势不能完全由 prompt template 决定。
- 如果趋势随 template 反转，论文只能 claim protocol sensitivity，不能 claim route-specific semantic instability。

优先级 3：object-binding sanity check。

默认设计：

```text
construct or select cases where object_1/object_2 differ in both shape and color
ask relation and color-object binding questions separately
compare predicted binding with gold binding
```

输出目录：

```text
outputs/semantic_entropy_umm/object_binding_sanity/
```

验收：

- relation accuracy 和 color-object binding accuracy 分开报告。
- 列出至少 10 个 binding failure cases，含 image_path 和 raw_output。

优先级 4：alternative extractor。

默认先不接外部付费 API。先尝试本地或已可用的 alternative extractor；如果需要 GPT-4o、Qwen2.5-VL、InternVL 等外部服务，必须另行确认成本、访问方式和输出格式。

## 8. 最终论文表格目标

Table 1：Comparability Audit。

| Item | Expected | Observed | Pass |
|---|---:|---:|---|
| concepts | 30 | 30 | yes |
| routes | 2 | 2 | yes |
| states per concept-route | 10 | 10 | yes |
| total states | 600 | 600 | yes |
| slots per state | 6 | 6 | yes |
| total slot answers | 3600 | 3600 | yes |
| legal values | 100% | current audit PASS | yes |

Table 2：Extractor Validation。

| Slot | Accuracy | Macro F1 | Unknown Agreement | Notes |
|---|---:|---:|---:|---|
| object_1 | TBD | TBD | TBD | TBD |
| color_1 | TBD | TBD | TBD | TBD |
| object_2 | TBD | TBD | TBD | TBD |
| color_2 | TBD | TBD | TBD | TBD |
| relation | TBD | TBD | TBD | binding-critical |
| background | TBD | TBD | TBD | TBD |

Table 3：Route Entropy and Error。

| Route | State Entropy | Normalized Entropy | Effective States | Error | Unknown Rate |
|---|---:|---:|---:|---:|---:|
| I2T | TBD | TBD | TBD | TBD | TBD |
| T2I | TBD | TBD | TBD | TBD | TBD |
| Delta T2I-I2T | TBD | TBD | TBD | TBD | TBD |

Table 4：Slot-level Breakdown。

| Slot | H I2T | H T2I | Delta H | Err I2T | Err T2I | Delta Err |
|---|---:|---:|---:|---:|---:|---:|
| object_1 | 0.0904 | 0.1074 | 0.0170 | 0.1833 | 0.3200 | 0.1367 |
| color_1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| object_2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| color_2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| relation | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| background | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

Table 5：Robustness。

| Setting | Delta H | Delta Err | Conclusion |
|---|---:|---:|---|
| original strict protocol | 0.0170 joint | 0.1367 object_1 | baseline |
| randomized option order | TBD | TBD | TBD |
| prompt template v2 | TBD | TBD | TBD |
| object-binding sanity | TBD | TBD | TBD |
| alternative extractor | TBD | TBD | TBD |

## 9. 论文 claim 边界

只完成当前 strict audit 时，可以写：

```text
Under a controlled semantic-state protocol, I2T and T2I outputs can be
normalized into the same finite semantic space.
```

完成 human/extractor validation 后，可以写：

```text
The normalized semantic states provide an operationally valid measurement of
route-level semantics under the proposed slot schema.
```

完成 bootstrap、robustness、error/unknown 四象限后，才可以写：

```text
I2T and T2I route-level semantic entropy are empirically comparable under this
UMM semantic-state protocol, and observed differences reflect route-specific
semantic instability rather than raw modality-token uncertainty.
```

仍然不能写：

```text
Text entropy and image entropy are directly comparable.
UMM has a unified internal entropy space.
```

除非后续补 hidden-state、causal intervention 或 path-level experiments。

## 10. 执行顺序

建议下一步严格按以下顺序推进：

1. 不重跑模型，先完成 extractor validation 的抽样 manifest 和人工标注表。
2. 完成 60 张图的人工标注，并生成 extractor validation report。
3. 基于现有 strict outputs 完成 unknown + error + entropy 四象限分析。
4. 基于 concept-level paired metrics 完成 bootstrap CI。
5. 只在前四步显示结果有论文价值时，再做 robustness controls。
6. 最后整理五张论文表格和 failure case appendix。

每一步结束时都要更新一个阶段报告，报告必须包括：

```text
input files
commands
output files
sample counts
pass/fail checks
claim allowed after this step
claim still not allowed
```

## 11. Definition of Done

整个统一语义熵实验达到“可写论文初稿”的最低标准，需要同时满足：

- `strict_compare` audit PASS。
- `extractor_validation_report.md` 完成，且人工标注样本可回溯。
- `quadrant_report.md` 完成，且包含 low-entropy high-error cases。
- `bootstrap_report.md` 完成，且统计单位为 concept。
- 至少完成 option-order 或 prompt-template 中一种 robustness。
- 论文 claim 严格限定为 shared semantic state protocol 下的 route-level semantic entropy。

如果只完成 strict audit，则当前状态是“工程上可比”。如果再完成 extractor validation、bootstrap 和 robustness，则可以升级为“审稿人比较难直接否掉的可比”。
