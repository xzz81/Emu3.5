# Emu3.5 UMM 语义熵结果汇总

更新时间：2026-06-04 23:54 CST

本文档汇总本轮 Emu3.5 fork、远端部署、路径清理、UMM pilot、以及 I2T/T2I 严格语义熵可比性验证的当前结果。远端主工作区为 `/home/wentao/project/Emu3.5`，本地工作区为 `/Users/xzz/Documents/emu3.5_fork`。

## 1. 仓库与远端环境

已完成：

- 已将 `https://github.com/xzz81/Emu3.5.git` 克隆到本地 `/Users/xzz/Documents/emu3.5_fork`。
- 已同步到 `lb-gzs:/home/wentao/project/Emu3.5`；当前 `lb-gzs` 可通过 Tailscale 直连 `100.69.207.21:22` 登录。
- 远端 `origin` 已指向 xzz81 fork。
- 远端工作区已切到 commit `32a5ea2`。
- 已为远端项目建立权重与缓存软链接，包括：
  - `weights/Emu3.5 -> /home/wentao/models/Emu3.5/Emu3.5`
  - `weights/Emu3.5-Image`
  - `weights/Emu3.5-VisionTokenizer`
  - `.cache/huggingface`
  - `cache/huggingface`
  - `hf_cache`
  - `modelscope_cache`
- 已清理旧服务器残留的硬编码绝对路径 `/workspace/home/AAAI 2027/...`，改成相对路径，便于本地和远端一致运行。
- 已修复多个脚本里 `torchvision::nms` 重复预注册导致的兼容问题。

当前注意事项：

- 本地和远端都有未提交修改，包括路径清理、NMS 修复、UMM pilot 脚本、strict compare 脚本等。
- 尚未按本轮结果创建 commit；如需发布，需要另行确认提交范围。

## 2. UMM Pilot 结果

远端输出目录：

`/home/wentao/project/Emu3.5/outputs/semantic_entropy_umm/pilot`

Pilot 已完成并通过 `AUDIT PASS`。已确认输出规模如下：

| 文件 | 行数/数量 | 说明 |
|---|---:|---|
| `concepts.jsonl` | 30 | 30 个 UMM concept |
| `image_manifest.jsonl` | 30 | 30 张 reference image |
| `reference_images/` | 30 images | reference 图像 |
| `i2t_samples.jsonl` | 1500 | I2T 采样结果 |
| `t2i_samples.jsonl` | 300 | T2I 采样结果 |
| `generated_images/` | 300 images | T2I 生成图像 |
| `semantic_slots.jsonl` | 1800 | 语义 slot 抽取结果 |
| `route_entropy.csv` | 421 lines | 420 行数据 + header |
| `route_error.csv` | 361 lines | 360 行数据 + header |
| `case_report.md` | 65 lines | case report |

Pilot 的意义：

- 它已经把图像和文本路线都投影到语义 slot 上，而不是直接比较图像 token 和文本 token。
- 它证明了 UMM 上的概念集、I2T 采样、T2I 采样、语义抽取、熵/错误率计算这条流水线可以完整跑通。
- 它可以作为初步证据说明“图片语义熵”不是直接在像素或视觉 token 上算，而是在可解释的语义状态上算。

Pilot 的限制：

- 旧 pilot 里 I2T 是 question-level 结果，T2I 是 image-level 结果，两边粒度不完全一致。
- 因此，旧 pilot 可以说明方向合理，但还不能作为严格证明“文字和图片语义熵完全处在同一个可比级别”的最终证据。

## 3. 三个核心问题的当前回答

### 3.1 图片的语义熵现在是否合理？

结论：作为“语义状态熵”是合理的；strict compare 已把图片 route 的语义熵约束到共享 semantic-state protocol 下。

理由：

- 当前定义避免了直接对图像 token 或像素分布求熵。
- 图像路线先通过同一模型抽取 object、color、relation、background 等语义 slot，再在离散语义状态空间中计算不确定性。
- 这符合当前研究计划里“比较 meanings/semantic states，而不是比较 modality-specific token”的要求。

旧 pilot 仍有粒度问题，所以严格论文证据应以 strict compare 输出为准。strict compare 已在远端完成并通过 `comparability_audit`。

### 3.2 现在文字和图片的语义熵是否可以比较？

结论：原始 token 熵不能比较；旧 pilot 只能弱比较；strict compare 已证明 I2T/T2I 在同一个 UMM semantic state protocol 下可以比较。

不可比较的对象：

- 文本 token 熵 vs 图像 token 熵。
- 自由文本描述熵 vs 图像生成像素/视觉 token 熵。
- 不同粒度的 I2T question-level 熵 vs T2I image-level 熵。

可比较的对象：

- 同一 UMM concept。
- 同一 semantic slot schema。
- 同一 forced-choice value vocabulary。
- 同一 extractor model。
- 同一 parser/normalization。
- 同一 sample count。
- 同一 state-level 聚合方式。

strict compare 正是按这个标准重做：I2T 和 T2I 都转成相同 slot/value 空间里的 semantic state，然后再算 route entropy 与 route error。远端审计已给出 `PASS`。

### 3.3 在 UMM 上有没有实现统一？

结论：工程上已经实现统一 UMM 流水线；protocol-level 的严格统一可比性已经由 strict compare 的 `AUDIT PASS` 支持。

已经统一的部分：

- 同一批 UMM concepts。
- 同一批 reference/generated image 资源。
- 同一套 semantic slot。
- 同一套 value vocabulary。
- 同一套 forced-choice VQA prompt。
- 同一套 extractor/parser。
- 同一套 entropy/error 计算脚本。

已经通过 strict audit 验证的部分：

- I2T/T2I 每个 route 均为 300 个 semantic states。
- 每个 route/slot 均为 300 个 slot answers。
- schema 一致。
- value vocabulary 合法。
- `comparability_audit` verdict 为 `PASS`。

## 4. Strict Compare 设计

新增脚本：

`/home/wentao/project/Emu3.5/scripts/run_semantic_entropy_strict_compare.py`

本地对应：

`/Users/xzz/Documents/emu3.5_fork/scripts/run_semantic_entropy_strict_compare.py`

Strict compare 的目标是修复旧 pilot 的粒度问题，让 I2T 和 T2I 在完全相同的 state-level semantic space 上比较。

核心设计：

- 不重新生成图片，复用 pilot 中已有的 30 张 reference image 和 300 张 generated image。
- 对 I2T 和 T2I 都问同一套 forced-choice VQA 问题。
- 语义 slot 固定为：
  - `object_1`
  - `color_1`
  - `object_2`
  - `color_2`
  - `relation`
  - `background`
- value vocabulary 固定为：
  - object slots: `cube`, `sphere`, `cone`, `unknown`
  - color slots: `red`, `blue`, `green`, `yellow`, `unknown`
  - relation: `object_1_left_of_object_2`, `object_1_right_of_object_2`, `object_1_above_object_2`, `object_1_below_object_2`, `unknown`
  - background: `white`, `other`, `unknown`
- 每个 concept 每个 route 目标为 10 个 semantic states。
- 总目标为：
  - `strict_semantic_states.jsonl`: 600 rows
  - `strict_slot_answers.jsonl`: 3600 rows

Audit 检查项：

- schema 是否一致。
- value vocabulary 是否全部合法。
- 每个 concept 下 I2T/T2I state 数是否都是 10。
- 每个 route/slot 的 answer 数是否都是 300。
- 只有全部通过，`comparability_audit` 才能给出 PASS。

## 5. Strict Compare 最终结果

远端输出目录：

`/home/wentao/project/Emu3.5/outputs/semantic_entropy_umm/strict_compare`

启动方式：

`/home/wentao/project/Emu3.5/outputs/semantic_entropy_umm/strict_compare/launch_strict_workers.sh`

截至 2026-06-04 23:54 CST，3 个 worker 已全部完成，launcher 已自动完成 aggregate 和 audit。

最终输出：

| 文件 | 行数 | 说明 |
|---|---:|---|
| `workers/strict_semantic_states_worker0.jsonl` | 200 | worker0 semantic states |
| `workers/strict_semantic_states_worker1.jsonl` | 200 | worker1 semantic states |
| `workers/strict_semantic_states_worker2.jsonl` | 200 | worker2 semantic states |
| `workers/strict_slot_answers_worker0.jsonl` | 1200 | worker0 slot answers |
| `workers/strict_slot_answers_worker1.jsonl` | 1200 | worker1 slot answers |
| `workers/strict_slot_answers_worker2.jsonl` | 1200 | worker2 slot answers |
| `strict_semantic_states.jsonl` | 600 | merged semantic states |
| `strict_slot_answers.jsonl` | 3600 | merged slot answers |
| `strict_route_entropy.csv` | 421 | 420 data rows + header |
| `strict_route_error.csv` | 361 | 360 data rows + header |
| `comparability_audit.json` | 146 | machine-readable audit |
| `comparability_audit.md` | 48 | human-readable audit |

`comparability_audit.md` verdict：

> PASS

审计确认：

- State rows by route: `{'I2T': 300, 'T2I': 300}`。
- Schema OK: `True`。
- Vocabulary OK: `True`。
- Per-concept route parity failures: `0`。
- Slot-answer parity failures: `0`。

Mean entropy：

| Slot | I2T | T2I | Delta T2I-I2T |
|---|---:|---:|---:|
| joint | 0.0904 | 0.1074 | 0.0170 |
| object_1 | 0.0904 | 0.1074 | 0.0170 |
| color_1 | 0.0000 | 0.0000 | 0.0000 |
| object_2 | 0.0000 | 0.0000 | 0.0000 |
| color_2 | 0.0000 | 0.0000 | 0.0000 |
| relation | 0.0000 | 0.0000 | 0.0000 |
| background | 0.0000 | 0.0000 | 0.0000 |

Mean error：

| Slot | I2T | T2I | Delta T2I-I2T |
|---|---:|---:|---:|
| object_1 | 0.1833 | 0.3200 | 0.1367 |
| color_1 | 0.0000 | 0.0000 | 0.0000 |
| object_2 | 0.0000 | 0.0000 | 0.0000 |
| color_2 | 0.0000 | 0.0000 | 0.0000 |
| relation | 0.0000 | 0.0000 | 0.0000 |
| background | 0.0000 | 0.0000 | 0.0000 |

注意：这个 PASS 证明的是 shared semantic state protocol 下的可比性，不等于证明 automated VQA extractor 已经达到人工真值级别。后续还需要 extractor validation、bootstrap CI、sample-size sensitivity 和 robustness controls。

## 6. 当前最严谨结论

### 已经可以确认

- 仓库已经成功 fork/clone/sync 到远端。
- 远端权重和缓存已经通过软链接接入。
- 旧绝对路径已经清理为相对路径。
- UMM pilot 已完整跑通并通过审计。
- Pilot 已证明图片语义熵可以定义在 semantic slots/state 上，而不是定义在图像 token 上。
- 旧 pilot 暴露出 I2T 与 T2I 粒度不完全一致的问题。
- 已实现 strict compare，用于把 I2T/T2I 拉到相同 semantic state level。
- strict compare 已在远端完成，`comparability_audit` verdict 为 `PASS`。

### 仍需谨慎的边界

- 不能只用旧 pilot 说“文字和图片语义熵已经严格可比”，因为旧 pilot 的 I2T/T2I 粒度不完全一致；严格 claim 应以 strict compare 为准。
- 不能比较原始文本 token 熵和图像 token 熵；后续论文表述必须限定为“共享语义状态空间中的 route entropy”。
- 不能把 protocol-level comparability 说成 UMM 已经具有 unified internal entropy space；后者需要 hidden-state、causal intervention 或 path-level experiments。
- 不能把低 entropy 直接解释为正确，需要联合 error、unknown rate、invalid parse rate 和 extractor validation。

### 推荐论文表述

当前最稳妥的表述是：

> We do not compare raw token-level entropy across modalities. Instead, both I2T and T2I routes are projected into a shared UMM semantic state space defined by the same concepts, slots, finite value vocabulary, extractor, parser, and sampling budget. Entropy is then computed over these normalized semantic states, with route error reported jointly to distinguish stable-but-wrong outputs from genuinely reliable ones.

中文对应：

> 我们不直接比较跨模态 token 级熵，而是将 I2T 与 T2I 两条路线都投影到统一的 UMM 语义状态空间中。该空间使用相同概念、相同 slot schema、相同有限 value vocabulary、相同抽取器、相同解析规则和相同采样预算。随后仅在归一化语义状态上计算 route entropy，并同时报告 route error，以区分“稳定但错误”和“稳定且可靠”的输出。

## 7. 收敛目标与后续动作

已新增收敛目标文档：

`/Users/xzz/Documents/emu3.5_fork/docs/semantic_entropy_i2t_t2i_convergence_target.md`

已新增可直接执行的整理文档：

`/Users/xzz/Documents/emu3.5_fork/docs/semantic_entropy_i2t_t2i_execution_plan.md`

当前 strict audit 已经支持：

> Under a controlled semantic-state protocol, I2T and T2I outputs can be normalized into the same finite semantic space.

后续要把实验从“工程上可比”升级为“审稿人比较难直接否掉的可比”，优先补：

- human/extractor validation：随机抽 30 reference + 30 generated images，人工标注 6 个 slots，对比 extractor。
- unknown + error + entropy 四象限分析：区分稳定正确、稳定错误、多样正确、不稳定错误。
- sample-size / bootstrap stability：补 concept-level paired statistics、95% bootstrap CI、effective number of states。
- robustness controls：prompt template、option order、alternative extractor、object-binding sanity check。
