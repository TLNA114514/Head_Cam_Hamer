# HaMeR 推理速度与延迟优化

更新时间：2026-07-14。

本文只讨论推理速度、启动延迟、吞吐、显存/内存和输出 I/O。姿态误差实验仍记录在：

- `gloves/glove_local_calibration_experiments.md`
- `gloves/glove_local_calibration_experiments.zh.md`

## 结论

当前建议保留 HaMeR，但使用新的执行链路：

1. 默认 `--sam3-execution per-sequence` 和 `--hamer-execution per-sequence`，两个大模型都只加载一次；
2. 使用 `--hamer-job-batch-size 8` 跨 job 填满模型 batch；
3. `quality` 在 mask 不可读时自动只跑必然胜出的 `1.0` 尺度，有效 mask 仍保留三尺度；
4. 默认不写 SAM3 全帧 debug、778 个 MANO vertices、MANO rotation matrices 和逐帧 overlay；
5. 默认不再执行精度更低的 legacy MANO local refine 和 primary-local fusion；
6. 默认 `quality` profile 仍保持 FP32 和 mesh-mask candidate scoring，不用速度换 pose；
7. 对吞吐敏感时使用 `balanced` profile：单尺度、FP32，只在存在多个候选时做 mesh-mask scoring；
8. `aggressive` 保留三尺度，但使用 FP16、skeleton-mask proxy 和可回退的 `torch.compile`；
9. `fast` profile 使用单尺度 FP16，必须在目标 GPU 上完成精度回归后再用于正式输出；
10. `--frames` 默认按 `--base-dir` 的数据集名自动匹配，避免 right run 误用 left timestamps/FPS。

## 已实现的优化

### 1. SAM3 与 HaMeR 每序列只加载一次

原 pipeline 把相机和时间都切成 chunk，每个 camera/chunk 都会重新加载一次约 `2.50GiB`
的 HaMeR checkpoint。现默认先合并 jobs，再用一个 worker 处理整段序列。
SAM3 image model 同样从每个 camera/chunk 重载改为先处理完整多相机序列，再由轻量 identity/job
阶段按 chunk 读取已有 JSONL 和 masks。

| 序列 | legacy per-chunk | per-camera | 默认 per-sequence |
| --- | ---: | ---: | ---: |
| left 0--442 | 36 次 / 约 90.17GiB checkpoint 读取 | 4 次 | **1 次 / 约 2.50GiB** |
| right 0--477 | 40 次 / 约 100.19GiB checkpoint 读取 | 4 次 | **1 次 / 约 2.50GiB** |

这些是根据实际 job 分片和 checkpoint 大小得到的静态工作量，不是磁盘吞吐 benchmark。
CPU 单次真实模型加载约 `3.17s`；GPU 环境需要重新测量。

SAM3 image detector 在默认 `image`/`posthoc` 路径中也从 left/right 的 `36/40` 次加载降为
每序列 `1` 次。若显式选择 `sam3-native`，image detector 仍只加载一次，但额外的视频 tracker
目前保持按 chunk 运行，以保留重叠窗口和失败恢复粒度；不能把它计入“一次 SAM3 加载”的结论。

如需更细的失败恢复粒度，仍可显式使用 `--hamer-execution per-camera` 或
`--hamer-execution per-chunk`。

### 2. 跨 job 打包候选

旧 worker 在每个 job 内单独建 DataLoader。已知 handedness 的三尺度 job 只有 3 个样本，
无法填满默认 batch size 4。新 worker 会预处理一个小 job window，再保持原顺序统一打包。

静态分析结果：

| 序列 | 逐 job forward batches | camera-packed lower bound | 减少 |
| --- | ---: | ---: | ---: |
| left | 3617 | 2714 | 25.0% |
| right | 3915 | 2937 | 25.0% |

真实 HaMeR CPU A/B（8 jobs、24 candidates、batch size 4）：

- forward batches：`8 -> 6`；
- inference：`11.610s -> 11.284s`，约 `1.03x`；
- 选中尺度和 handedness 完全一致；
- joints 最大绝对差为 `1.19e-7m`。

CPU 上收益较小，因为总 FLOPs 不变；GPU 上通常更依赖 batch 利用率，必须通过 worker config
中的 `timing` 字段实测。

同一个 frame 的左右手 jobs 现在还会在 job window 内共享一次 `cv2.imread` 结果，不再重复
读取和 JPEG 解码。若关闭 vertices/MANO params，worker 也不会把每个候选的 778 vertices 或
rotation matrices 从设备复制到 CPU；skeleton scorer 且不保存 overlay 时完全不初始化 renderer。

### 3. 避免无效 mask render

单尺度且 handedness 已知时只有一个候选，mesh-mask render 不可能改变选择。
`balanced`/`fast` 的 `selection-only` 模式只在候选数大于 1 时评分。
跳过 mesh render 的 singleton 仍会把原始 `sam3_score` 传给同相机重复候选排序，
不会因为 `mask_score=None` 被误当成零质量。

right C0 0--49 的真实 CPU A/B：

| mask scoring | inference | jobs/s | joints 差异 |
| --- | ---: | ---: | ---: |
| all | 79.987s | 1.250 | baseline |
| selection-only | **46.349s** | **2.158** | **0** |

即该已知 handedness 分片为 `1.73x`，且真正获得的 pose 完全不变。存在左右手歧义时仍保留
mask scoring，因此不能把所有相机都按 `1.73x` 外推。

### 4. 无 mask 时自适应缩减尺度

当 SAM3 mask 不存在或路径不可读时，三种尺度的 `mask_score` 都是 `None`，原选择器必然按
tie-break 选择最接近 `1.0` 的尺度。`quality` 现在会在模型前识别这个条件，只生成 `1.0`
候选；有有效 mask 时仍完整保留 `1.0/1.1/1.2`，因此不改变原选择语义。

left 完整序列的 mask 路径不可读，候选从 `10851 -> 3617`，camera-packed forward batch
下界从 `2714 -> 905`。真实 HaMeR CPU A/B（C0 0--3，8 jobs）为：

- candidate samples：`24 -> 8`；
- inference：`11.237s -> 3.751s`，即 **`3.00x`**；
- bbox scale、handedness 和 joints 完全一致，最大关节点差为 `0`。

right 的 mask 全部可读，因此该策略不会偷减其三尺度计算。
新 SAM3 run 若生成了有效 left masks，也会自动回到完整三尺度；`3.00x` 是当前无 mask
分支的真实结果，不是对所有新序列的固定承诺。

### 5. Skeleton-mask aggressive scorer

mesh-mask scoring 的 CPU rasterization 很重。新 `skeleton` scorer 用 HaMeR 已输出的 21 个
投影关节点构造 palm polygon 和 finger tubes，再与 SAM3 mask 计算同结构 overlap score。
right 0--9、70 jobs 的公平 FP32 A/B：

| scorer | inference | mask scoring | candidate 一致率 | 五指 mean |
| --- | ---: | ---: | ---: | ---: |
| mesh | 202.792s | 81.116s | baseline | 60.523mm |
| skeleton | **123.078s** | **0.803s** | 50.0% | 60.705mm |

即端到端 `1.65x`，mean `+0.18mm`。它不是严格等价替代，所以只进入 `aggressive` profile；
扩大到 right 0--49、374 jobs 后，candidate 一致率为 `34.5%`，但多视角融合 mean 从
`34.305 -> 33.271mm`，median 从 `25.428 -> 23.484mm`；代价是 P95
`89.425 -> 90.202mm`、max `120.617 -> 122.754mm`。这说明 proxy 不是 mesh 的近似复刻，
而是一种不同的速度/尾部误差 trade-off，不能因 mean 改善就升级默认。

`aggressive` 还请求 backbone `torch.compile`，若编译建立或首次执行失败，worker 会自动恢复
eager backbone。当前机器无 GPU，compile/FP16 的额外收益尚未计入上述 CPU 数字。

### 6. 单尺度 balanced profile

三尺度会把已知 handedness 的模型样本数从 1 增加到 3。`balanced` 只保留 bbox scale `1.0`。

- left 完整旧输出的 3211 条 prediction 最终全部选择了 scale `1.0`；
- right 0--477 的四相机完整真实重跑覆盖 3461 jobs、478 帧，零失败；
- 单尺度仍需为 454 个 unknown-handedness job 各跑左右手，因此共 3915 candidates、989 个
  model forward batches；
- CPU inference 为 `2108.728s`（`35:08.7`），吞吐 `1.641 jobs/s`；
- quality：mean/median/P95/max = `35.496/30.269/77.334/130.169mm`；
- balanced：`36.061/30.988/80.016/128.528mm`；
- 差值：mean `+0.565mm`，median `+0.719mm`，P95 `+2.682mm`，max `-1.642mm`。

GT 只在推理完成后用于这次离线评估，没有进入模型、候选选择或校准。完整序列证明
`balanced` 的平均退化较小，但 P95 增加约 `2.68mm`，所以 `quality` 仍是默认 profile；
`balanced` 是已完成 full-sequence 验证的速度/精度 trade-off，不冒充无损替代。当前没有同执行链路的
三尺度 full-sequence 计时，因此这里只报告 balanced 的真实绝对耗时，不把候选数比例冒充实测 speedup。

### 7. 输出和后处理 I/O

默认 zero-shot skeleton fusion 不读取 per-view vertices、MANO rotation matrices，也不需要每帧
overlay，因此默认关闭三者。启用 MANO refine 时 pipeline 会自动恢复 vertices 和 MANO params。

| 序列 | 原 JSONL | joint-only JSONL | JSON 减少 | overlay | 合计避免写入 |
| --- | ---: | ---: | ---: | ---: | ---: |
| left | 176.3MiB | 11.5MiB | 93.5% | 930.2MiB | **1095.1MiB** |
| right | 190.5MiB | 13.3MiB | 93.0% | 617.8MiB | **795.0MiB** |

当显式启用 MANO refine 时，pipeline 会自动重新导出 vertices 和 MANO params。需要调试图时使用
`--save-hamer-rendered-overlays`，不要把全量 overlay 当默认产物。

SAM3 的 bbox debug 与 mask debug 也改为默认关闭，真实已有产物的成本为：

| 序列 | bbox debug | mask debug | inference masks（保留） |
| --- | ---: | ---: | ---: |
| left | 689.8MiB | 667.8MiB | 17.1MiB |
| right | 738.5MiB | 715.4MiB | 18.3MiB |

前两列共 `1.36GiB/1.45GiB`，只服务人工查看；第三列会进入 HaMeR 候选选择，不能删除。
需要调试时显式传 `--save-sam3-debug`。

## 四种运行配置

质量优先；mask 可读时保持原三尺度推理，不可读时做严格等价缩减：

```bash
/home/luojiangrui/miniconda3/envs/headcam/bin/python scripts/run_hamer_multiview_pipeline.py \
  --base-dir video/sam3_hamer_left_index \
  --group-range 0-442 \
  --hamer-speed-profile quality \
  --overwrite
```

平衡吞吐与精度：

```bash
/home/luojiangrui/miniconda3/envs/headcam/bin/python scripts/run_hamer_multiview_pipeline.py \
  --base-dir video/sam3_hamer_left_index \
  --group-range 0-442 \
  --hamer-speed-profile balanced \
  --overwrite
```

FP16 快速实验；正式使用前必须在目标 GPU 上和 `quality` 做同序列 A/B：

```bash
/home/luojiangrui/miniconda3/envs/headcam/bin/python scripts/run_hamer_multiview_pipeline.py \
  --base-dir video/sam3_hamer_left_index \
  --group-range 0-442 \
  --hamer-speed-profile fast \
  --overwrite
```

保留三尺度但使用轻量 mask proxy、FP16 和可回退 backbone compile：

```bash
/home/luojiangrui/miniconda3/envs/headcam/bin/python scripts/run_hamer_multiview_pipeline.py \
  --base-dir video/sam3_hamer_right_index \
  --group-range 0-477 \
  --hamer-speed-profile aggressive \
  --overwrite
```

静态分析已有产物：

```bash
python scripts/analyze_hamer_efficiency.py \
  --jobs video/sam3_hamer_left_index/hamer_jobs/hamer_jobs_000000_000442.jsonl \
  --predictions video/sam3_hamer_left_index/hamer_per_view/hamer_predictions_000000_000442.jsonl \
  --rendered-dir video/sam3_hamer_left_index/hamer_per_view/rendered \
  --candidate-scale-policy mask-adaptive
```

每个 worker 的 `hamer_predictions_config_*.json` 会记录模型加载、总 inference、model+output、
mask scoring、overlay、serialization、jobs/s、precision 和实际 batch 数。pipeline 日志也会为每个
子命令打印 `elapsed=...s`。

## 更快模型调研

### WiLoR：最值得下一步实测

WiLoR 官方论文报告：FreiHAND PA-MPJPE `5.5mm`，优于 HaMeR 的 `6.0mm`；HO3Dv2 为
`7.5mm`，优于 HaMeR 的 `7.7mm`。其动态重建指标也明显优于 HaMeR。官方 2026-03 更新的
`--fast` 模式使用 FP16、`torch.compile` 和 backbone layer dropping，声称最高 `1.6x`，
MPJPE 变化约 `0.05mm`，并提供 regressor 与 detector 权重。

- 官方代码与 fast 说明：<https://github.com/rolpotamias/WiLoR>
- CVPR 2025 论文：<https://openaccess.thecvf.com/content/CVPR2025/papers/Potamias_WiLoR_End-to-end_3D_Hand_Localization_and_Reconstruction_in-the-wild_CVPR_2025_paper.pdf>

它是当前最合理的替代候选，但尚未直接替换本 pipeline，原因是：当前 workspace 未安装 WiLoR；
其模型许可证为 `CC-BY-NC-ND`；而且必须在本项目的手套、遮挡、头戴多视角数据上验证输出坐标、
handedness、MANO joints 和跨视角融合兼容性。公开 benchmark 不能代替本地 A/B。

### Fast-HaMeR：接口兼容，但当前权重交付不足

Fast-HaMeR 用轻量 backbone 和知识蒸馏替换 ViT-H。官方仓库声称模型约为原尺寸 35%、
inference `1.5x`，HO3D-v2 只差约 `0.4mm`，代码仍沿用 HaMeR 接口。

- 官方代码：<https://github.com/hunainahmedj/Fast-HaMeR>
- 论文：<https://arxiv.org/abs/2603.16444>

但截至检查日期，官方 README 只说明如何传入 student checkpoint，没有给出可直接下载的已训练
student checkpoint 链接；`fetch_demo_data.sh` 对应原 HaMeR demo data。因此它目前更适合自行训练/
蒸馏后替换，不能在本仓库中声称已经完成可复现替代。

### Hamba：精度强，但速度证据不足

Hamba 官方结果在 FreiHAND 达到 PA-MPVPE `5.3mm`，并强调用更少 token 的 graph-guided Mamba。
但当前官方页面没有给出足够清晰、可与本机 HaMeR 直接对比的端到端 FPS，因此不把“token 更少”
等同于“本 pipeline 一定更快”。

- 官方项目：<https://humansensinglab.github.io/Hamba/>
- 官方代码：<https://github.com/humansensinglab/Hamba>

## 下一步 GPU Gate

目标 GPU 可用后，按以下顺序验证：

1. 同一 100 帧、四相机、同一 jobs，分别运行 `quality`、`balanced`、`fast`、`aggressive`；
2. 比较 config 中的模型加载、jobs/s、peak VRAM 和 forward batch 数；
3. 用 glove 只做推理后的离线 A/B，不把它输入任何 profile 选择器；
4. `fast`/`aggressive` 需要同时满足速度收益、handedness/coverage 不下降、mean/P95 在可接受阈值内；
5. 若获得 WiLoR 安装与许可证许可，再用完全相同 jobs 做 WiLoR adapter 和 full-sequence A/B。
