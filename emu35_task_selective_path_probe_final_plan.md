# Emu3.5 图像生成与图像理解任务选择性通路最终收敛方案

> 汇总报告：`emu35_generation_understanding_path_final_report.md`
>
> 统一证据矩阵：
> `outputs/stage21_differential_component_matrix.csv`
> `outputs/stage21_differential_component_matrix_report.md`
>
> Claim 审计：
> `outputs/stage22_claim_evidence_audit.md`
>
> 复现与证据索引：
> `outputs/stage23_reproducibility_evidence_index.md`
>
> Decoded generation patch validation：
> `outputs/stage24_decoded_generation_patch_validation.md`
> `outputs/stage25_decoded_generation_patch_replication.md`
> `outputs/stage26_decoded_generation_wrong_target_control.md`
> `outputs/stage27_decoded_generation_random_target_control.md`
> `outputs/stage28_decoded_generation_mlp_module_patch.md`

## 1. 收敛后的核心问题

本项目不再泛泛追问“是否有神经元被激活”，而是收敛为一个可验证的因果问题：

```text
Emu3.5 在图像生成与图像理解中出现的任务差异性激活，
是否能被推进为可 clean/corrupt 恢复、可跨任务区分的 causal pathway evidence?
```

对应到两个任务方向：

```text
generation:    text prompt / color token -> generated visual tokens
understanding: image tokens -> post-image text states -> answer score position
```

由于 Emu3.5 是 decoder-only unified multimodal model，研究对象不是 diffusion 模型中的 UNet cross-attention，而是同一个 Transformer 主干内部的：

```text
residual stream
self-attention heads
MLP module / MLP intermediate neurons
token-to-token information flow
```

## 2. 当前最稳健答案

### 图像理解任务

当前证据支持：

```text
理解侧存在明确的 late-layer causal route。
```

但更准确地说，它不是由当前测试的小规模 MLP 神经元集合单独承载，而是分布在 residual stream 与部分 attention heads 中。

关键发现：

| 证据层级 | 当前结果 | 解释 |
|---|---|---|
| activation screening | late layers 61-63 出现 U-over-G 候选神经元 | 有任务差异性激活 |
| MLP neuron ablation | top U neurons 对 answer NLL 影响强于随机 | 有弱因果作用 |
| MLP neuron clean/corrupt patch | 小规模 top U neuron sets 不能稳定恢复 | 不能称为 sparse neuron pathway |
| residual patch | layer 61/62/63 answer score position 几乎完全恢复 | image-to-answer 信息确实在 late residual state 中 |
| attention knockout | answer score -> image/EOI direct edge 弱或负 | 不是单一直接 attention gate |
| mediated route | post-image question/text residual 可恢复大部分 gap | 信息通过 text-side residual 表征中介 |
| module patch | layer 60/62 MLP 与 layer 62 self-attn 有局部贡献 | full residual 表征由多个模块累积 |
| head patch | layer 62 heads 36/38/46 最强，top10 joint recovery 最高 | self-attn contribution 可部分定位到少数 heads |

当前理解侧结论：

```text
Emu3.5 的 hue understanding 通路存在于 late residual stream 中，
经过 post-image text/question states 中介，
layer 62 self-attention heads 36/38/46 是目前最清晰的 head-level 局部化结果；
layer 60/62 MLP modules 有贡献，但不是由当前 sparse activation-selected neurons 单独解释。
```

### 图像生成任务

当前证据支持一个比早期更清楚、但仍然谨慎的结论：

```text
生成侧已经在 retrospective highcontrast raw-target benchmark 上出现弱 causal pathway evidence；
但它主要是 target-side residual 强、prompt-side attention 弱，
Stage 24/25 已把 full L61 target-side residual patch 推进到 decoded validation：
4 个 highcontrast 方向中 exact HSV restoration 3/4，continuous hue-distance improvement 4/4，
Stage 26 wrong-target control 为 0/4 exact restoration 且 4/4 hue-distance 不改善，
Stage 27 random/other-target control 为 0/4 exact restoration，
Stage 28 L61 MLP-module decoded patch 为 0/4 exact restoration，
Stage 29 L61 PCA200 decoded patch 为 0/4 exact restoration 且 0/4 hue-distance improvement，
Stage 30 L61 top10 attention-head decoded patch 为 1/4 exact restoration、2/4 hue-distance improvement，
Stage 31 L61 bottom10 attention-head decoded control 为 0/4 exact restoration、1/4 hue-distance improvement，
Stage 32 L61 top10 attention-head + MLP decoded patch 为 0/4 exact restoration、1/4 hue-distance improvement，
Stage 33 L61 self-attn + MLP decoded patch 为 1/4 exact restoration、3/4 hue-distance improvement，
Stage 34 L60/L61/L62 full residual decoded patch 均为 3/4 exact restoration、4/4 hue-distance improvement，
Stage 35 L60/L61/L62 self-attn+MLP decoded patch 分别为 0/4、1/4、0/4 exact restoration，
Stage 41 L0 sampled full residual decoded patch 为 3/4 exact restoration、4/4 hue-distance improvement；
Stage 42 L0 wrong-target control 为 0/4 exact、0/4 improvement，random-target control 为 0/4 exact、2/4 improvement；
Stage 43 input-embedding clean-target patch 为 3/4 exact、4/4 improvement，wrong-target 为 0/4 exact、0/4 improvement，random-target 为 0/4 exact、2/4 improvement；
Stage 44 teacher-forced input-embedding 分解显示 visual-score target embeddings 为 0 recovery / 0 embedding delta，color-token embeddings 为 mean recovery +1.012833，prompt embeddings 为 +0.949991；
Stage 45 decoded color-token input-embedding patch 为 2/4 exact restoration、4/4 hue-distance improvement，且每方向只 patch 1-2 个 prompt color-token positions；
Stage 46 decoded object-token input-embedding control 为 0/4 exact restoration、0/4 hue-distance improvement；
Stage 40 L8/L16 sampled full residual decoded patch 也均为 3/4 exact restoration、4/4 hue-distance improvement；
Stage 39 L32/L40 sampled full residual decoded patch 也均为 3/4 exact restoration、4/4 hue-distance improvement；
Stage 38 L48/L52 sampled full residual decoded patch 也均为 3/4 exact restoration、4/4 hue-distance improvement；
Stage 37 L56-L63 continuous full residual decoded patch 也均为 3/4 exact restoration、4/4 hue-distance improvement，
还不能称为 generation-selective neurons 或紧凑 color circuit。
```

关键发现：

| 证据层级 | 当前结果 | 解释 |
|---|---|---|
| online decoded benchmark | CFG seed sweep 成本高，red/blue blue side 长 seed 仍失败 | 不能把在线生成作为当前主证据 |
| retrospective HSV validation | hueonly 3/4 usable pairs；highcontrast 2/4 usable pairs | 可用旧 decoded images 建立较干净 raw-target benchmark |
| prompt residual patch | highcontrast L60 +0.048844, L61 +0.045733, L62 +0.010733 | prompt-side 只有弱恢复 |
| color/object token controls | color-token 与 object-token 同量级 | 不能称为颜色词专属通路 |
| target visual-score residual patch | highcontrast L60-L62 约 +0.99 recovery | target-side score residual 几乎完整携带可恢复信息 |
| module patch | prompt L60/L61 self-attn 弱正向；target-side single modules 只恢复 8%-25% | residual 信息由多模块累积 |
| prompt attention head patch | L60 all heads 均为正；top10 joint +0.040162 | prompt-side 有弱 head-level 影响，但不是 generation-specific |
| target visual-score head patch | L61 top10 joint +0.101426；bottom10 joint -0.037579 | target-side self-attn 有有利 head 子集，但远低于 full residual |
| target visual-score MLP neuron patch | L61 top1000 +0.127074；top10 +0.036203 | MLP 信号需要较大 neuron 集合，不是 sparse neuron set |
| target visual-score MLP low-rank patch | L61 PCA200 +0.169549；random200 +0.036415 | MLP delta 有结构化 feature subspace，但仍低于 module |
| target component cross-task controls | L61 top heads: G +0.101426 vs U -0.001324；top1000 neurons: G +0.127074 vs U +0.001096；PCA200: G +0.169549 vs U +0.001142 | target-side components 有初步 generation-specificity |
| decoded target full-state patch | Stage 25 L61 clean-target exact restoration 3/4；Stage 43 input-embedding、Stage 41 L0、Stage 40 L8/L16、Stage 39 L32/L40、Stage 38 L48/L52 sampled layers 与 Stage 37 L56-L63 continuous band 均 3/4 exact、4/4 improvement；Stage 43 input-embedding wrong-target 0/4 exact、0/4 improvement，random-target 0/4 exact、2/4 improvement；Stage 42 L0 wrong-target 0/4 exact、0/4 improvement；random-target 0/4 exact、2/4 improvement；Stage 26 wrong-target 0/4；Stage 27 random-target 0/4 | decoded 支持 clean-source-specific broad target-side trajectory overwrite upper bound，但不是 compact circuit |
| teacher-forced input embedding | Stage 44 visual-score target input embeddings 为 mean recovery 0.000000 且 embedding delta 0；color-token input embeddings 为 +1.012833；prompt input embeddings 为 +0.949991 | raw-target likelihood 中颜色条件已在 prompt color-token input boundary 可恢复；decoded color-token patch 已在 Stage 45 部分验证，但 object-token/random-color control groups 未验证 |
| decoded color-token input embedding | Stage 45 color-token input embedding patch 为 2/4 exact、4/4 improvement；patched_forward_calls=1，patched_positions_total=1-2 | 目前最紧凑的 decoded prompt-side 正向证据；仍需 object-token/random-color 控制 |
| decoded object-token input embedding | Stage 46 object-token input embedding patch 为 0/4 exact、0/4 improvement；patched_forward_calls=1，patched_positions_total=1 | 支持 color-token effect 相对 object-token control 的特异性 |
| decoded MLP-module patch | Stage 28 L61 MLP module exact restoration 0/4，hue-distance improvement 1/4 | MLP module 有 teacher-forced likelihood 贡献，但 decoded control 不足 |
| decoded PCA200 patch | Stage 29 L61 PCA200 exact restoration 0/4，hue-distance improvement 0/4 | PCA200 是 teacher-forced feature subspace，不是 decoded compact circuit |
| decoded top10 attention-head patch | Stage 30 L61 top10 heads exact restoration 1/4，hue-distance improvement 2/4 | top heads 有部分 decoded 影响，但不足以解释 full residual route |
| decoded bottom10 attention-head control | Stage 31 L61 bottom10 heads exact restoration 0/4，hue-distance improvement 1/4 | top10 > bottom10，但 decoded separation 仍较弱 |
| decoded top10 attention-head + MLP patch | Stage 32 exact restoration 0/4，hue-distance improvement 1/4 | teacher-forced 正向组件不能简单叠加成 decoded control |
| decoded self-attn + MLP patch | Stage 33 exact restoration 1/4，hue-distance improvement 3/4 | module-output 组合有更强 decoded 影响，但仍低于 full residual |
| decoded self-attn + MLP band control | Stage 35 L60/L61/L62 exact restoration 0/4、1/4、0/4 | module-output 组合不形成 late-layer band |

当前生成侧结论：

```text
highcontrast generation raw targets 存在真实 clean/corrupt visual-token likelihood gap；
target-side visual score residual 是当前最强的 generation causal evidence；
prompt-side layer60/61 attention 有弱影响，但 head-level 分布较宽且不专属；
target-side layer61 visual-score attention heads 出现 top/bottom 分离，
但 top10 heads 仍只恢复约 10%，远低于 full residual 的约 99%；
target-side layer61 MLP neurons 中 top1000 可恢复约 13%，
但 top10/top50 很弱，仍不能称为 sparse generation neurons；
target-side L61 MLP PCA200 可恢复约 17%，远高于 random200，
说明有结构化 feature subspace，但仍低于 MLP module 的约 25%；
target-side L61 top heads/top1000 neurons/PCA200 directions 在 understanding score-position controls 上近零，
因此 target-side 组件比 prompt-side 组件更接近 generation-specific；
Stage 25 decoded patch 支持 full L61 target-side residual 在小型 highcontrast benchmark 中可恢复 decoded hue/object，
其中 exact HSV restoration 为 3/4，连续 hue-distance improvement 为 4/4；
Stage 26 wrong-target target-side residual patch 不能恢复 clean hue/object，
Stage 27 other-pair random target patch 也不能恢复 exact clean hue/object，
Stage 28 显示 L61 MLP module 不能单独恢复 decoded clean hue/object，
Stage 29 显示 L61 PCA200 也不能恢复 decoded clean hue/object，
Stage 30 显示 L61 top10 attention heads 有部分 decoded 影响但只恢复 1/4 方向，
Stage 31 bottom10 attention-head control 支持 top10 比 bottom10 更强，但差距仍有限，
Stage 32 显示 top10 attention heads + MLP module 组合不增强 decoded restoration，
Stage 33 显示 full self-attn + MLP module output 组合有 3/4 hue-distance improvement 但 exact 仅 1/4，
Stage 34 显示 L60-L62 full residual decoded restoration 几乎一致，
Stage 35 显示 self-attn+MLP module-output 组合不随 L60-L62 形成稳定 band，
Stage 37 显示 full residual decoded band 至少连续扩展到 L56-L63，
Stage 38 显示 sampled L48/L52 也有同样 full-residual restoration，
Stage 39 显示 sampled L32/L40 仍有同样 full-residual restoration，
Stage 40 显示 sampled L8/L16 也有同样 full-residual restoration，
Stage 41 显示 sampled L0 已经有同样 full-residual restoration，
Stage 42 显示 L0 wrong/random controls 不能复现 exact restoration，
Stage 43 显示 input-embedding boundary 已经有同样 clean-source-specific restoration，
Stage 44 显示 teacher-forced color-token input boundary 几乎完整恢复 raw-target likelihood，
Stage 45 显示 decoded color-token input boundary 有 partial exact / full distance-improvement support，
Stage 46 显示 decoded object-token input boundary 为 clean negative control，
但目前仍不支持 generation-selective neurons、color-token-specific circuit，
也不支持少数 attention heads 足以解释完整生成侧通路。
```

## 3. 统一实验漏斗

后续所有实验按四级证据推进，不跳级：

```text
Stage A: activation contrast
Stage B: ablation over random control
Stage C: clean/corrupt patch recovery
Stage D: cross-task specificity
```

各阶段允许的 claim：

| 阶段 | 能说什么 | 不能说什么 |
|---|---|---|
| A | 有任务差异性激活候选 | 有因果通路 |
| B | 组件影响对应输出 likelihood | 组件承载 clean/corrupt 语义 |
| C | 组件参与可恢复 causal path | 任务专属 |
| D | generation / understanding path 可区分或共享 | 绝对模块化分离 |

## 4. Understanding 侧下一步

理解侧已经有强 residual-level evidence，下一步应从“有没有通路”转向“通路如何分解”。

优先顺序：

1. 固化 layer 62 attention heads 36/38/46/top10 的 cross-task control。
2. 对 layer 60/62 MLP 不再只做 top-k neuron patch，改做更大规模 cumulative neuron sweep、低秩方向 patch 或 SAE/PCA-style feature patch。
3. 把 full residual、full module、top heads、MLP feature directions 放到同一张对照表中。
4. 做 generation-side interference control：在 generation benchmark 稳定后测试 U heads/U feature directions 对 T2I 的影响。

停机条件：

```text
如果 sparse MLP neurons 持续不能超过 random control，
论文表述应收敛为 distributed MLP feature / residual representation，
而不是 sparse understanding neurons。
```

## 5. Generation 侧下一步

生成侧当前最大风险不是 patch 技术，而是评估目标不干净。因此下一步先做 benchmark construction。

### 5.1 先构造 verified generation benchmark

目标：

```text
每个 clean/corrupt pair 的 clean side 和 corrupt side 都能在 decoded image 级别通过 hue validation。
```

建议执行：

1. 扩大 seed sweep，但只保留 hue classifier / VQA checker 通过的样本。
2. 优先换更稳定的 prompt/object/resolution，而不是继续在 red/blue circle 16x16 上堆干预。
3. 对每个 pair 保存 prompt、seed、decoded image、HSV hue result、colorful fraction 和 visual-token target cache key。
4. 只在 filtered valid pairs 上跑 teacher-forced NLL patch 和 direct decoded patch。

当前工程更新：

```text
scripts/run_generation_clean_hue_seed_sweep.py 已支持 --sample-id / --pair-id / --expected-color 定向筛选，
并支持 --list-samples 在加载模型前列出候选样本。
脚本现在也支持 --resume 与 --stop-after-matches，
并在每个 seed 完成后增量写出 clean_hue_seed_sweep.csv、summary.csv 和 report。
新增 elapsed_seconds 与 --max-total-seconds，
可记录每个 seed 的生成耗时，并在下一个 seed 前按总耗时预算停止。
当 --resume 发现请求的 sample/seed 已全部存在时，会在加载模型前快速退出。
```

一次 blue-side CFG 定向烟测显示：

```text
cf_hueonly_circle_red_vs_blue__b, seed_idx 6..7, CFG=2.0, max_new_tokens=320
超过五分钟仍无 decoded image 落盘，已中断。
```

这说明后续大规模 CFG seed sweep 应先优化 generation runner 或改成更小、更可恢复的 batch。

已完成一个 1-token persistence smoke：

```text
outputs/generation_clean_hue_seed_sweep_incremental_write_smoke_max1
generated_tokens = 1
decoded = 0
目的仅是验证增量写盘，不作为 hue benchmark 证据。
```

已完成一个 timing / budget smoke：

```text
outputs/generation_clean_hue_seed_sweep_timing_smoke_max1
requested seed_idx 101..102
max_total_seconds = 1
实际写出 seed_idx 101 后，在第二个 seed 前停止
elapsed_seconds = 5.582763
目的仅是验证耗时记录与 stop-before-next 机制，不作为 hue benchmark 证据。
```

同一 timing smoke 目录已验证 resume 快速退出：

```text
--resume 重跑 seed_idx 101 时直接提示 all requested sample/seed rows already exist；
未触发 checkpoint loading。
```

已新增 filtered benchmark manifest 构建脚本：

```text
scripts/build_generation_hue_benchmark_manifest.py
```

同时新增自动聚合入口：

```text
scripts/build_generation_hue_benchmark_from_sweeps.py
```

该脚本会从 `outputs/` 自动发现 `clean_hue_seed_sweep*.csv`，默认排除 `smoke` 与
`generation_hue_filtered_benchmark_manifest` 聚合目录，合并去重后再重建 manifest。
不能用泛泛的 `manifest` 作为排除规则，否则会误伤正式输出目录
`outputs/generation_clean_hue_seed_sweeps_from_manifest`。

它从 seed-sweep CSV 中选择 decoded 且 hue_match 的 clean generations，并输出：

```text
selected_valid_seeds.csv
sample_status.csv
pair_status.csv
missing_sweep_queue.csv
filtered_generation_hue_benchmark_report.md
```

当前 red/blue circle CFG=2.0 manifest：

```text
outputs/generation_hue_filtered_benchmark_manifest_pair1_cfg2
expected samples = 2
selected valid samples = 1
usable bidirectional pairs = 0
selected red seed = 147402
missing sample = cf_hueonly_circle_red_vs_blue__b
```

当前 all-pairs manifest：

```text
outputs/generation_hue_filtered_benchmark_manifest_allpairs_current
expected samples = 8
selected valid samples = 1
usable bidirectional pairs = 0
```

当前自动聚合版 all-pairs manifest：

```text
outputs/generation_hue_filtered_benchmark_manifest_auto_current
discovered CSVs = 4
merged unique rows = 13
expected samples = 8
selected valid samples = 1
usable bidirectional pairs = 0
```

自动聚合入口现在还会写出可执行 shell：

```text
outputs/generation_hue_filtered_benchmark_manifest_auto_current/next_sweep_command.sh
outputs/generation_hue_filtered_benchmark_manifest_auto_current/all_sweep_commands.sh
outputs/generation_hue_filtered_benchmark_manifest_auto_current/run_next_and_refresh.sh
```

`next_sweep_command.sh` 当前指向 blue-side 下一批 seed sweep。
`run_next_and_refresh.sh` 会先运行当前下一条 sweep，再自动重建 auto manifest 和下一条 queue。
长作业日志会写到：

```text
outputs/generation_hue_filtered_benchmark_manifest_auto_current/logs/run_next_*.log
```

当前 missing sweep queue 的首个目标：

```text
sample_id = cf_hueonly_circle_red_vs_blue__b
expected_color = blue
seed_start_idx = 7
seed_count = 1
priority = 0
reason: paired red sample already has selected seed 147402
```

一次 blue-side CFG runtime probe 显示：

```text
outputs/generation_clean_hue_seed_sweep_blue_cfg_idx6_probe_smoke_max8
max_new_tokens = 8
elapsed_seconds = 47.738509
decoded = 0
```

这说明 320-token CFG 生成可能是单 seed 约 30 分钟量级，因此后续推荐按单 seed 长作业调度，而不是一次启动多个 seed。

一次正式 blue-side seed_idx=6 尝试曾在约 15 分 38 秒后被人工中断：

```text
outputs/generation_clean_hue_seed_sweeps_from_manifest/hueonly_circle_red_vs_blue_b_idx6_6/interrupted_attempt_report.md
```

该尝试没有写出 seed CSV 或 decoded image，但 GPU 在中断前仍活跃，说明只靠 seed 间
`--max-total-seconds` 不足以约束单个长 seed。当前推荐命令保留：

```text
--max-total-seconds 2400.0
```

并新增进程内 per-seed deadline：

```text
--per-seed-timeout-seconds 900.0
```

相关实现：

```text
scripts/run_generation_clean_hue_seed_sweep.py
src/utils/generation_utils.py 支持 cfg.extra_stopping_criteria
scripts/run_generation_patched_decode_hue.py 的 no-CFG 路径也尊重同一 extra_stopping_criteria
```

已完成一个 per-seed timeout smoke：

```text
outputs/generation_clean_hue_seed_sweep_timeout_record_smoke
seed_idx = 6
generated_tokens = 1
decoded = 0
stop_reason = per_seed_timeout
```

已完成一次正式 blue-side seed_idx=6 per-seed timeout 尝试：

```text
outputs/generation_clean_hue_seed_sweeps_from_manifest/hueonly_circle_red_vs_blue_b_idx6_6
seed_idx = 6
generated_tokens = 147
elapsed_seconds = 903.809609
decoded = 0
hue_match = 0
stop_reason = per_seed_timeout
```

formal run 后刷新 auto manifest：

```text
cf_hueonly_circle_red_vs_blue__b attempts = 7
decoded = 6
hue_matches = 0
next_seed_start_idx = 7
```

为了防止单个 seed 长时间不返回，manifest 生成的建议命令同时带进程内 per-seed deadline 与外层硬超时：

```text
--per-seed-timeout-seconds 900.0
timeout --foreground 2700.0
```

区别：

```text
--max-total-seconds 只在 seed 之间检查；
--per-seed-timeout-seconds 在 model.generate 内部按 token step 检查并写出 partial row；
timeout --foreground 是最后兜底，可能来不及写 seed CSV。
```

manifest 现在也会给出可直接运行的建议命令。当前 all-pairs 首个命令为：

```bash
CUDA_VISIBLE_DEVICES=0 timeout --foreground 2700.0 python scripts/run_generation_clean_hue_seed_sweep.py --cfg configs/ume_main_t2i_counterfactual_pairs_hueonly_seed69.py --max-pairs 4 --sample-id cf_hueonly_circle_red_vs_blue__b --seed-start-idx 7 --seed-count 1 --generation-mode cfg --target-height 16 --target-width 16 --image-area 65536 --generation-max-new-tokens 320 --classifier-free-guidance 2.0 --resume --out-dir outputs/generation_clean_hue_seed_sweeps_from_manifest/hueonly_circle_red_vs_blue_b_idx7_7 --max-total-seconds 2400.0 --per-seed-timeout-seconds 900.0
```

### 5.1.1 Retrospective raw-target bridge

由于在线 CFG seed sweep 成本过高，已新增一条 retrospective raw-target 路线：

```text
scripts/build_retrospective_generation_image_manifest.py
scripts/build_generation_target_cache_from_raw.py
```

该路线读取旧 T2I runs 中已经保存的 decoded image、raw generation text 与 entropy trace，
先用同一 HSV hue classifier 验证 decoded image，再把 raw generation text 编码成
`run_generation_residual_patch_recovery.py` 可直接读取的 target cache。

retrospective HSV manifest：

```text
outputs/generation_retrospective_image_manifest_hueonly_seed69
samples = 8
usable bidirectional pairs = 3/4

outputs/generation_retrospective_image_manifest_highcontrast_seed68
samples = 8
usable bidirectional pairs = 2/4
```

raw target cache：

```text
outputs/generation_raw_target_cache_hueonly_seed69
cached hue-match samples = 7

outputs/generation_raw_target_cache_highcontrast_seed68
cached hue-match samples = 6
```

已完成 Stage 10 retrospective raw-target patch 报告：

```text
outputs/generation_rawtarget_retrospective_stage10_report.md
```

关键结果：

| benchmark | patch | layers | valid directions | result |
|---|---|---|---:|---|
| hueonly triangle | color-token | 62 | 2 | mean recovery -0.000128，近零 |
| hueonly valid3 | prompt | 60/61/62 | 6 | layer60/61 负或近零，layer62 约 0 |
| highcontrast valid2 | prompt | 60/61/62 | 4 | layer60 +0.048844，layer61 +0.045733，layer62 +0.010733 |

当前解释：

```text
retrospective raw targets 让 generation teacher-forced patch 可以绕过在线生成瓶颈；
highcontrast prompt residual 在 layer60/61 有小但方向一致的正向 recovery；
hueonly prompt/color-token residual 仍然 near-zero；
因此 generation 侧开始出现弱 residual-path 线索，但还不能下钻到 head/module/neuron claim。
```

已完成 Stage 11 highcontrast raw-target control 报告：

```text
outputs/generation_rawtarget_highcontrast_controls_stage11_report.md
```

本轮还修正了 `visual-score-positions` patch 的源/目标位置映射：

```text
clean source score position = clean_prompt_len + target_idx - 1
corrupt target score position = corrupt_prompt_len + target_idx - 1
```

修正前，如果 clean/corrupt prompt 的 token 长度不同，visual score patch 会发生目标 token index 错配。
修正后的 highcontrast valid2 结果：

| scope | layer60 mean recovery | layer61 mean recovery | layer62 mean recovery | interpretation |
|---|---:|---:|---:|---|
| prompt | +0.048844 | +0.045733 | +0.010733 | 弱正向 prompt-side influence |
| color-token | +0.022827 | +0.027874 | -0.010116 | 弱正向但不稳定 |
| object-token | +0.030403 | +0.020750 | -0.002097 | 与 color-token 同量级 |
| visual-score-aligned | +0.985674 | +0.999021 | +0.998530 | target-side residual 上界接近完整 recovery |

当前更精确解释：

```text
highcontrast generation raw targets 具有真实 clean/corrupt visual-token likelihood gap；
target-side visual score residual patch 能几乎完全恢复该 gap；
prompt-side layer60/61 residual patch 只有弱恢复；
color-token patch 与 object-token patch 同量级，暂不能称为颜色词专属通路。
```

已完成 Stage 12 highcontrast raw-target module patch 报告：

```text
outputs/generation_module_patch_highcontrast_stage12_report.md
scripts/run_generation_module_patch_recovery.py
```

关键模块级结果：

| scope | module/layer | mean recovery | interpretation |
|---|---|---:|---|
| prompt | L60 self-attn | +0.054713 | prompt-side 弱恢复主要来自 self-attn |
| prompt | L60 MLP | +0.005645 | 接近零 |
| prompt | L61 self-attn | +0.043160 | 稳定弱正向 |
| prompt | L61 MLP | +0.027157 | 弱贡献 |
| color-token | L60 MLP | +0.056892 | 有弱正向，但非属性专属 |
| object-token | L60 self-attn | +0.058555 | 与 color-token 同量级 |
| visual-score | single modules L60-L62 | +0.083526 到 +0.246532 | 单模块只恢复一部分 |
| visual-score | full residual L60-L62 | 约 +0.99 | full residual 才接近完整恢复 |

当前 generation 侧最稳妥表述：

```text
target-side visual score residual 中有强可恢复信息；
prompt-side layer60/61 self-attn 有弱影响；
MLP 与 token-level patches 只有弱且不专属的贡献；
generation path 更像分布式 residual accumulation，而不是 sparse color neuron 或单一 color-token circuit。
```

已完成 Stage 13 highcontrast raw-target attention head patch 报告：

```text
outputs/generation_attention_head_highcontrast_stage13_report.md
scripts/run_generation_attention_head_patch_recovery.py
```

关键 head-level 结果：

| experiment | result | interpretation |
|---|---:|---|
| L60 prompt all-head individual scan | 64/64 heads mean recovery 为正 | head 影响很宽，不是稀疏尖峰 |
| L60 individual-head mean | mean +0.033932，median +0.033432 | 典型 head 只有弱恢复 |
| L60 best head 16 | +0.068122 | 有较强弱正向 head，但仍是小效应 |
| heads >= +0.05 | 7/64 | top heads 存在，但数量不止一两个 |
| heads helping all directions | 29/64 | 方向一致性也较分散 |
| L60 top10 joint patch | +0.040162 | top10 组合没有超过 L60 self-attn module |
| L60 prompt self-attn module | +0.054713 | 模块级效应不被 top10 heads 简单解释 |

当前 head-level 解释：

```text
generation prompt-side L60 attention 有弱正向影响；
但 all-head scan 与 top10 joint control 显示它不是紧凑 attention-head circuit。
individual head ranking 可作为候选，但不能升级为 generation-selective head claim。
```

已完成 Stage 15 generation target-side visual-score attention head 报告：

```text
outputs/generation_target_visualscore_attention_heads_stage15_report.md
scripts/run_generation_attention_head_patch_recovery.py
```

关键 target-side head 结果：

| experiment | result | interpretation |
|---|---:|---|
| L61 visual-score all-head individual scan | 64/64 heads mean recovery 为正 | target-side head 效应仍有宽分布 |
| L61 individual-head mean | mean +0.034378，median +0.034649 | 典型 single head 是弱恢复 |
| L61 best head 44 | +0.067004 | 单个 head 有弱到中等恢复 |
| heads >= +0.05 | 9/64 | top heads 比底部 heads 更清楚 |
| heads helping all directions | 24/64 | 方向一致性不是稀疏单点 |
| L61 top10 joint visual-score | +0.101426 | top10 是有利子集，超过 L61 self-attn module |
| L61 bottom10 joint visual-score | -0.037579 | 低排名 heads 组合会伤害 recovery |
| L61 visual-score self-attn module | +0.083526 | top10 可超过模块平均效应 |
| L61 visual-score MLP module | +0.246532 | MLP 仍是更强 single-module contribution |
| L61 visual-score full residual | +0.999021 | 完整 target-side 信息主要仍在 residual stream |

当前 target-side head 解释：

```text
generation target-side L61 visual-score attention 中存在有利 top-head subset；
这比 prompt-side L60 heads 更有分辨率，因为 top10 与 bottom10 明显分离。
但 top10 只恢复约 10%，full residual 约 99%，因此它不是完整 generation circuit。
```

已完成 Stage 16 generation target-side visual-score MLP neuron patch 报告：

```text
outputs/generation_visualscore_mlp_neurons_stage16_report.md
scripts/run_generation_mlp_neuron_patch_recovery.py
```

筛选方式：

```text
在 L61 visual-score aligned positions 上收集 MLP intermediate:
act(gate_proj(x)) * up_proj(x)
按 clean/corrupt mean absolute activation difference 排序 neuron，
再 patch selected clean neuron contributions through down_proj。
```

L61 MLP intermediate width:

```text
25,600 neurons
```

关键 neuron-level 结果：

| group | n neurons | mean recovery | interpretation |
|---|---:|---:|---|
| top10 | 10 | +0.036203 | 小规模 sparse set 很弱 |
| top50 | 50 | +0.045597 | 仍明显低于 module |
| top200 | 200 | +0.073528 | 有递增但仍弱 |
| top1000 | 1000 | +0.127074 | 约恢复 L61 MLP module 的一半 |
| bottom1000 | 1000 | +0.017575 | top1000 明显优于 bottom1000 |
| L61 visual-score MLP module | full module | +0.246532 | single-module 上界 |
| L61 visual-score full residual | full residual | +0.999021 | target-side 信息仍主要在 residual |

当前 MLP-neuron 解释：

```text
L61 visual-score MLP neurons 有可排序的 target-side generation signal；
但 top10/top50 远不能解释模块效应，top1000 才恢复约一半 MLP module。
因此目前支持 broader MLP feature mass，不支持 sparse generation-selective neurons。
```

已完成 Stage 18 generation target-side visual-score MLP low-rank feature patch 报告：

```text
outputs/generation_visualscore_mlp_lowrank_stage18_report.md
scripts/run_generation_mlp_lowrank_patch_recovery.py
```

方法：

```text
收集 L61 visual-score aligned positions 的 clean-corrupt MLP intermediate deltas；
对 delta rows 做 PCA；
把 clean-corrupt delta 投影到 top PCA directions 后经 down_proj patch；
同 rank random orthonormal subspace 作为对照。
```

关键 low-rank 结果：

| group | rank | mean recovery | interpretation |
|---|---:|---:|---|
| PCA1 | 1 | +0.053869 | 明显高于 random1 |
| random1 | 1 | +0.011006 | rank1 random weak |
| PCA10 | 10 | +0.056657 | 与 random10 接近 |
| random10 | 10 | +0.049339 | random 子空间也有弱正恢复 |
| PCA50 | 50 | +0.108716 | 明显高于 random50，接近 top1000 neurons |
| random50 | 50 | +0.048127 | 显著低于 PCA50 |
| top1000 neurons | 1000 | +0.127074 | 大 neuron mass 略高于 PCA50 |
| L61 visual-score MLP module | full module | +0.246532 | low-rank/neurons 仍未接近 module |

PCA delta energy:

```text
PCA1  cumulative energy = 0.079451
PCA10 cumulative energy = 0.240801
PCA50 cumulative energy = 0.448880
```

当前 low-rank 解释：

```text
L61 visual-score MLP delta 有低秩结构；
PCA50 明显优于 random50，说明不是任意子空间都有效。
但 PCA50 只恢复约 11%，仍低于 top1000 neurons 和 full MLP module，
因此该 MLP signal 同时具有 low-rank 与 broad-distributed 特征。
```

已完成 Stage 19 generation target-side visual-score MLP higher-rank feature patch 报告：

```text
outputs/generation_visualscore_mlp_lowrank_highrank_stage19_report.md
scripts/run_generation_mlp_lowrank_patch_recovery.py
```

关键 higher-rank 结果：

| group | rank | mean recovery | interpretation |
|---|---:|---:|---|
| PCA100 | 100 | +0.128502 | 约等于 top1000 neurons |
| random100 | 100 | +0.031315 | 明显低于 PCA100 |
| PCA200 | 200 | +0.169549 | 高于 top1000 neurons，仍低于 MLP module |
| random200 | 200 | +0.036415 | 明显低于 PCA200 |
| top1000 neurons | 1000 | +0.127074 | 大 neuron mass 被 PCA100 追平 |
| L61 visual-score MLP module | full module | +0.246532 | PCA200 尚未接近 full module |

PCA high-rank delta energy:

```text
PCA50  cumulative energy = 0.456493
PCA100 cumulative energy = 0.570121
PCA200 cumulative energy = 0.678937
```

当前 high-rank 解释：

```text
L61 visual-score MLP signal 的结构化 feature subspace 比 sparse neuron set 更有压缩效率；
PCA100 约追平 top1000 neurons，PCA200 超过 top1000 neurons。
但 PCA200 仍只恢复约 17%，低于 MLP module 约 25%，远低于 full residual 约 99%。
```

可接受的 benchmark 进入条件：

```text
每个方向至少有一个 clean-valid seed；
clean 与 corrupt prompts 的 decoded hue 均正确；
random seed / random token patch 不产生同等 restoration；
clean/corrupt teacher-forced gap 与 decoded hue 判断方向一致。
```

### 5.2 再做 generation causal decomposition

在 benchmark 稳定后，按以下顺序推进：

```text
residual stream patch
-> attention head output patch
-> MLP module patch
-> MLP neuron / feature patch
-> cross-task intervention
```

优先 patch scope：

```text
color-token residual
object-token residual
prompt-token residual with strict span alignment
visual output score positions
text-to-visual attention heads
```

核心指标：

```text
S_gen_likelihood = clean visual-token NLL recovery
S_gen_decoded    = decoded hue restoration / VQA attribute restoration
```

只有当两个指标至少方向一致时，才继续下钻到神经元。

## 6. Differential Path Score

对同一批组件同时计算：

```text
S_gen(c) = generation patch / ablation effect
S_under(c) = understanding patch / ablation effect

G_score(c) = S_gen(c) - lambda * S_under(c)
U_score(c) = S_under(c) - lambda * S_gen(c)
```

分类：

| 类型 | 判据 |
|---|---|
| generation-specialized | G_score 高，understanding control 低 |
| understanding-specialized | U_score 高，generation control 低 |
| shared | 两边都高 |
| irrelevant/noisy | 两边都低或方向不稳定 |

lambda 建议扫：

```text
0.5, 1.0, 2.0
```

已完成统一 differential component matrix：

```text
outputs/stage21_differential_component_matrix.csv
outputs/stage21_differential_component_matrix_report.md
```

核心矩阵：

| component | S_gen | S_under | current class |
|---|---:|---:|---|
| Understanding L62 heads 36/38/46 | +0.001438 | +0.152832 | understanding-specific head-level route |
| Generation L60 prompt top10 heads | +0.040162 | +0.020725 | weak/shared prompt-side influence |
| Generation L61 visual-score top10 heads | +0.101426 | -0.001324 | generation target-side specific heads |
| Generation L61 visual-score top1000 MLP neurons | +0.127074 | +0.001096 | generation target-side MLP feature mass |
| Generation L61 visual-score PCA200 MLP directions | +0.169549 | +0.001142 | generation target-side feature subspace |

一句话读法：

```text
understanding specificity 目前是 answer-score/head-level；
generation specificity 目前是 target-side visual-token-likelihood/feature-subspace-level。
```

已完成第一批 head-level cross-task control：

```text
outputs/cross_task_attention_head_controls_stage14_report.md
```

关键结果：

| component | S_under | S_gen | current reading |
|---|---:|---:|---|
| Understanding L62 heads 36/38/46 | +0.152832 | +0.001438 | strong U, near-zero G |
| Generation L60 top10 heads | +0.020725 | +0.040162 | weak G, weak U |

对应解释：

```text
理解侧 L62 heads 36/38/46 当前呈现较强 understanding-selective 证据：
它们在 understanding answer likelihood 上有 +0.152832 recovery，
但在 generation prompt-side raw-target likelihood 上只有 +0.001438。

生成侧 L60 top10 heads 当前不能称为 generation-specialized：
它们在 generation 上只有 +0.040162，
在 understanding 上也有 +0.020725 的弱正恢复。
```

已完成 generation target-side components 的 cross-task control：

```text
outputs/cross_task_generation_target_components_stage17_report.md
```

关键结果：

| component | S_gen | S_under | current reading |
|---|---:|---:|---|
| Generation L61 visual-score top10 heads | +0.101426 | -0.001324 | target-side G positive, U near-zero |
| Generation L61 visual-score top10 MLP neurons | +0.036203 | +0.000541 | weak G, U near-zero |
| Generation L61 visual-score top1000 MLP neurons | +0.127074 | +0.001096 | target-side G positive, U near-zero |
| Generation L61 visual-score PCA50 directions | +0.108716 | -0.000160 | target-side G positive, U near-zero |
| Generation L61 visual-score PCA200 directions | +0.169549 | +0.001142 | target-side G positive, U near-zero |

对应解释：

```text
generation prompt-side L60 heads 不专属；
generation target-side L61 visual-score heads、top1000 MLP neurons 与 PCA200 directions
则更接近 generation-specific。
不过这个 specificity 是 raw-target visual-token likelihood 层面的 target-side evidence，
Stage 25 已把 full L61 target-side residual 推进到 4 方向 decoded replication：
exact HSV restoration 3/4，continuous hue-distance improvement 4/4。
Stage 26 wrong-target control 显示该恢复不是任意 target trajectory patch 的泛化效应：
wrong-target exact restoration 0/4，hue-distance improvement 0/4。
Stage 27 random/other-target control 进一步显示 other-pair target trajectory 也不能恢复 clean exact hue：
random-target exact restoration 0/4。
Stage 28 decoded MLP-module patch exact restoration 0/4，说明当前 decoded restoration 还不能降到 MLP-module-level。
Stage 29 decoded PCA200 patch exact restoration 0/4 且 hue-distance improvement 0/4，
说明当前 decoded restoration 也不能降到 compact MLP-subspace-level。
Stage 30 decoded top10 attention-head patch exact restoration 1/4、hue-distance improvement 2/4，
说明 top heads 有 decoded 贡献但仍不能替代 full residual。
Stage 31 decoded bottom10 attention-head control exact restoration 0/4、hue-distance improvement 1/4，
说明 top10/bottom10 有方向性分离但还很弱。
Stage 32 decoded top10 attention-head + MLP patch exact restoration 0/4、hue-distance improvement 1/4，
说明 teacher-forced 正向组件不能简单相加得到 decoded trajectory restoration。
Stage 33 decoded self-attn + MLP patch exact restoration 1/4、hue-distance improvement 3/4，
说明 L61 module-output 组合有部分 path-level 影响，但仍低于 full residual。
Stage 34 decoded L60/L61/L62 full residual patch 均为 exact restoration 3/4、hue-distance improvement 4/4，
说明 decoded route 更像 late residual band，而不是 L61 单层状态。
Stage 35 decoded L60/L61/L62 self-attn+MLP patch 分别为 exact restoration 0/4、1/4、0/4，
说明 residual-band 不能被同层显式 self-attn+MLP module outputs 简单解释。
Stage 41 decoded L0 sampled full residual patch 为 exact restoration 3/4、hue-distance improvement 4/4；
Stage 42 L0 wrong-target patch 为 exact restoration 0/4、hue-distance improvement 0/4，random-target patch 为 exact restoration 0/4、hue-distance improvement 2/4；
Stage 43 input-embedding clean-target patch 为 exact restoration 3/4、hue-distance improvement 4/4，wrong-target patch 为 0/4 exact、0/4 improvement，random-target patch 为 0/4 exact、2/4 improvement；
Stage 44 teacher-forced visual-score input embedding patch 为 mean recovery 0.000000，color-token input embedding patch 为 +1.012833，prompt input embedding patch 为 +0.949991；
Stage 45 decoded color-token input embedding patch 为 exact restoration 2/4、hue-distance improvement 4/4；
Stage 46 decoded object-token input embedding patch 为 exact restoration 0/4、hue-distance improvement 0/4；
Stage 40 L8/L16 sampled full residual patch 也均为 exact restoration 3/4、hue-distance improvement 4/4；
Stage 39 L32/L40 sampled full residual patch 也均为 exact restoration 3/4、hue-distance improvement 4/4；
Stage 38 L48/L52 sampled full residual patch 也均为 exact restoration 3/4、hue-distance improvement 4/4；
Stage 37 L56-L63 continuous full residual patch 也均为 exact restoration 3/4、hue-distance improvement 4/4，
说明 residual-band 边界尚未在 L59/L63 找到。
但它仍不是 compact decoded-image prompt-to-image circuit。
```

低秩方向的 cross-task control 报告：

```text
outputs/cross_task_generation_lowrank_stage20_report.md
scripts/run_understanding_mlp_lowrank_patch_recovery.py
```

## 7. 最终论文 Claim 梯度

当前已经支持的弱 claim：

```text
Emu3.5 在图像生成与图像理解任务中出现 late-layer task-selective activations。
```

当前部分支持的中等 claim：

```text
理解侧存在可 clean/corrupt 恢复的 image-to-answer causal route；
该 route 主要体现在 late residual stream 和部分 layer-62 attention heads，
而不是当前测试的小规模 MLP neuron sets。
其中 layer62 heads 36/38/46 通过了初步 cross-task control：
对 understanding 有明显恢复，对 generation prompt-side raw-target recovery 近零。

生成侧在 highcontrast retrospective raw-target benchmark 中存在可恢复的 visual-token likelihood gap；
target-side visual-score residual 几乎完整携带可恢复信息；
prompt-side layer60/61 attention 只有弱、分布式影响。
generation L60 top10 heads 在 generation 上弱正，在 understanding 上也弱正，
因此尚不能称为 generation-specialized。
target-side L61 visual-score top10 heads 与 bottom10 heads 出现明确分离，
但 top10 recovery 约 +0.101426，仍远低于 full residual 约 +0.999021。
target-side L61 MLP neuron ranking 有递增恢复，
top1000 neurons 约 +0.127074，但 top10/top50 很弱，
因此更像 broader MLP feature mass 而不是 sparse generation neurons。
target-side L61 MLP PCA200 directions 约 +0.169549，明显高于 random200，
说明 MLP signal 有结构化 feature subspace，但仍未接近 module-level recovery。
target-side L61 top heads、top1000 MLP neurons 与 PCA200 directions 在 understanding score-position control 上近零，
因此生成侧 target-side likelihood components 已有初步 cross-task specificity。
Stage 25 进一步显示 full L61 target-side visual-score residual patch 在 4 个 decoded generation 方向中 exact HSV restoration 3/4、hue-distance improvement 4/4，
Stage 26 wrong-target control 为 exact restoration 0/4、hue-distance improvement 0/4，
Stage 27 random/other-target control 为 exact restoration 0/4，
Stage 28 L61 MLP-module decoded patch exact restoration 0/4，
Stage 29 L61 PCA200 decoded patch exact restoration 0/4、hue-distance improvement 0/4，
Stage 30 L61 top10 attention-head decoded patch exact restoration 1/4、hue-distance improvement 2/4，
Stage 31 L61 bottom10 attention-head decoded control exact restoration 0/4、hue-distance improvement 1/4，
Stage 32 L61 top10 attention-head + MLP decoded patch exact restoration 0/4、hue-distance improvement 1/4，
Stage 33 L61 self-attn + MLP decoded patch exact restoration 1/4、hue-distance improvement 3/4，
Stage 34 L60/L61/L62 full residual decoded patch 均为 exact restoration 3/4、hue-distance improvement 4/4，
Stage 35 L60/L61/L62 self-attn+MLP decoded patch exact restoration 0/4、1/4、0/4，
Stage 41 L0 sampled full residual decoded patch 为 exact restoration 3/4、hue-distance improvement 4/4；
Stage 42 L0 wrong-target decoded patch 为 exact restoration 0/4、hue-distance improvement 0/4，random-target decoded patch 为 exact restoration 0/4、hue-distance improvement 2/4；
Stage 43 input-embedding clean-target decoded patch 为 exact restoration 3/4、hue-distance improvement 4/4，wrong-target decoded patch 为 0/4 exact、0/4 improvement，random-target decoded patch 为 0/4 exact、2/4 improvement；
Stage 44 teacher-forced color-token input embedding patch 为 mean recovery +1.012833；
Stage 45 decoded color-token input embedding patch 已测试，为 exact restoration 2/4、hue-distance improvement 4/4；
Stage 40 L8/L16 sampled full residual decoded patch 也均为 exact restoration 3/4、hue-distance improvement 4/4；
Stage 39 L32/L40 sampled full residual decoded patch 也均为 exact restoration 3/4、hue-distance improvement 4/4；
Stage 38 L48/L52 sampled full residual decoded patch 也均为 exact restoration 3/4、hue-distance improvement 4/4；
Stage 37 L56-L63 continuous full residual decoded patch 也均为 exact restoration 3/4、hue-distance improvement 4/4，
但这是 full residual target-trajectory restoration，不是 sparse neuron 或 compact subspace decoded control。
```

暂时不能写的强 claim：

```text
已经发现 generation-selective neurons。
已经证明 generation 与 understanding circuits 可分离。
已经证明少数 generation-specific attention heads 构成完整 prompt-to-image circuit。
```

强 claim 的最低证据门槛：

```text
generation benchmark clean-valid;
generation residual recovery robust;
generation head/module/neuron patch stronger than random;
同一组件在 understanding control 中呈现可解释的不对称性。
```

## 8. 当前推荐行动清单

短期只做三件事：

1. 文档层面：把当前结论写成“理解侧 strong causal route 且已有初步 head-level specificity；生成侧 target residual 强、prompt attention 弱且不专属、target-side components 有初步 specificity”的不对称证据状态。
2. 实验层面：generation 暂停 sparse neuron claim；PCA200 decoded patch 已在 Stage 29 得到负结果；top10 attention-head decoded patch 在 Stage 30 只有部分正向，Stage 31 bottom10 control 支持弱 top-vs-bottom separation，Stage 32 top10+MLP 组合不增强，Stage 33 self-attn+MLP 有 3/4 distance improvement 但 exact 仍 1/4；Stage 34 支持 L60-L62 late residual-band route，Stage 35 显示 self-attn+MLP module-output band 不成立，Stage 36 将 full residual band 扩展到至少 L59-L63，Stage 37 进一步扩展到连续 L56-L63，Stage 38 显示 sampled L48/L52 也强，Stage 39 显示 sampled L32/L40 仍强，Stage 40 显示 sampled L8/L16 也强，Stage 41 显示 sampled L0 已强，Stage 42 显示 L0 wrong/random controls 不复现 exact restoration，Stage 43 显示 input-embedding boundary 已经同样强且 clean-source-specific，Stage 44 显示 teacher-forced target visual-score input embeddings 为零恢复而 color-token input embeddings 几乎完整恢复 raw-target likelihood，Stage 45 显示 decoded color-token input embeddings 有 2/4 exact、4/4 improvement。因此 generation full-state decoded 结果应降格为 clean-source-specific target-trajectory overwrite upper bound；同时 compact prompt-side color-token input control 获得部分 decoded 支持。下一步若继续生成侧，应做 decoded object-token/random-color controls，而不是升级为完整 compact circuit claim。
3. 论文层面：主线先押 understanding causal route；generation 可以报告 replicated decoded support for full target-side residual patching，但仍不能升级到 generation-specific neuron、compact subspace decoded control 或完整 attention-head circuit。

一句话收敛：

```text
理解侧已经进入 causal pathway decomposition；
生成侧已有 highcontrast raw-target causal evidence，但仍是弱 prompt-side、强 target-side、分布式效应；
不要把两侧写成同等成熟度，也不要把 activation-selected MLP neurons 过早写成任务神经元。
```
