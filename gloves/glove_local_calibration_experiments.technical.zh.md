# Glove-Local 校准实验：技术附录

> 本文保留完整实验过程、消融和全部结果。方法选择与关键结论见
> [校准方法与实验概览](glove_local_calibration_experiments.zh.md)。

推理优化概览见 `docs/hamer_inference_optimization.md`，完整实验见
`docs/hamer_inference_optimization.technical.md`。

## 摘要与结论

这份文档最核心的结论可以先这样理解：
**HaMeR/MANO 输出的 hand-local 坐标，和 PN glove 的局部坐标/关节定义并不完全一致。**
这个差异是稳定存在的，不只是某几帧抖动，也不是简单做 temporal smoothing 就能解决的问题。

所以，下面的 glove-supervised 校准结果不应该被理解成“HaMeR 在没有 GT 的情况下突然变准了”。
它更像是一层“设备/坐标系对齐”：用 glove GT 学一个从 HaMeR/MANO local hand 到
PN glove local hand 的映射。原始 MANO 输出仍然保留在 `palm_local_joints_m`；
校准后的输出单独写到 `glove_calibrated_palm_local_joints_m`。

### 这些数值怎么看

- `Mean`：平均误差，适合看整体质量。
- `P95`：95% 的点都低于这个误差，适合看大部分坏帧有没有被压住。
- `Max`：最坏点，适合找失败案例，但可能被单个坏帧主导。
- 所有指标都是越低越好。

### 现在该用哪个输出

- 做 in-the-wild / zero-shot 的 palm-local 关节点时，用
  `hamer_palm_local_fused/`。默认主字段是不改写的跨视角均值，任何时序结果都保存在独立字段。
- 做 glove-supervised local 坐标时，用 `hamer_mano_local_refined/` 加静态校准层。
  这只适用于明确拥有 glove 校准片段的任务，不再作为部署默认。
- 如果目标动作空间已经被校准数据密集覆盖，`dense KNN + OOD guard` 的数值最低。
- 做无 GT 的图像侧 viewer/debug 且需要 MANO mesh 时，用
  `hamer_mano_multiview_selected/`；只看 palm-local skeleton 时优先新 zero-shot 输出。
- 不建议把 `hamer_mano_multiview_selected/` 当作 glove-supervised 校准的默认底座：
  它更适合图像侧可视化，但经过 glove 校准后反而更差。

### 推荐的静态校准配置

- 相似变换：包含 scale、rotation、translation；
- 每个 joint 的残差 offset；
- `joint_offset_shrink_k=25`；
- `max_joint_offset_m=0.025`；
- 不启用 bone scale；
- `write-mode=separate`，保留原始输出并额外写校准输出。

### 主结果一眼看懂

下图画的是 mean error，单位是 mm，越短越好。它是纯 Markdown 文本，
所以即使飞书过滤 SVG/HTML，也能正常显示。表格里同时保留了 mean 和 P95。

| 输出 | Mean mm | 文字条形图 |
| --- | ---: | --- |
| left_index 原始 | 30.12 | ███████████████▌ |
| left_index 静态校准 | 14.08 | ███████▎ |
| left_index KNN+OOD | 4.09 | ██ |
| right_index 原始 | 38.83 | ████████████████████ |
| right_index 静态校准 | 20.56 | ██████████▌ |
| right_index KNN+OOD | 5.87 | ███ |

| 序列 | 原始 MANO local mean / P95 | 推荐静态校准 mean / P95 | dense KNN+OOD mean / P95 |
| --- | ---: | ---: | ---: |
| left_index | 30.12 / 59.70 mm | 14.08 / 31.52 mm | 4.09 / 10.06 mm |
| right_index | 38.83 / 89.36 mm | 20.56 / 62.11 mm | 5.87 / 17.64 mm |

需要注意的是，dense KNN 的前提是“目标动作空间已经被校准集密集覆盖”。
如果换到新的动作片段、姿态覆盖不足，就应优先使用静态校准，或者使用带 OOD guard
的低容量 pose residual，而不是盲目相信 KNN。

### 无 GT 图像侧结果

在不使用 glove GT 的图像侧输出里，目前最稳的默认方案是 `physical-pnp`
初始化加 `0.04m` PnP view gate。它不修改相机标定；它只是在某个视角的
physical-K PnP pose 和当前 anchor 明显不一致时，把这个视角排除掉。

| 输出 | Mean mm | 文字条形图 |
| --- | ---: | --- |
| left baseline | 36.71 | ████████████████▌ |
| left PnP-gated | 34.70 | ███████████████▋ |
| right baseline | 44.41 | ████████████████████ |
| right PnP-gated | 42.02 | ███████████████████ |

| 图像侧输出 | left_index 0--442 mean / P95 | right_index 0--477 mean / P95 |
| --- | ---: | ---: |
| baseline image refine | 36.71 / 67.14 mm | 44.41 / 99.42 mm |
| PnP-gated selected 当前默认 | 34.70 / 63.72 mm | 42.02 / 93.20 mm |

当前候选选择规则是：**只要完整的 PnP-gated candidate 存在，就优先选择它；
如果 gated candidate 缺失或失败，再回退 baseline。**

### 更严格的时间顺序测试

最接近部署场景的测试，是只用前半段训练，再在后半段评估。
在这个设置下，pose+velocity residual 是目前 mean 和 P95 最好的版本，
不过它会让 left-index 的最坏点变差。

| 输出 | Mean mm | 文字条形图 |
| --- | ---: | --- |
| left 静态校准 | 15.15 | ██████████████▋ |
| left pose+velocity | 12.52 | ████████████▏ |
| right 静态校准 | 20.60 | ████████████████████ |
| right pose+velocity | 17.84 | █████████████████▎ |

### 已验证但暂不设为默认的方向

- image-space beta 2D refinement：对 left 0--49 只有极小改善，P95 略差，
  不值得增加默认运行时间；
- second-order temporal acceleration：方向略正，但收益太小，右手 max 略差；
- 全局 camera SE(3) correction：held-out 上不稳定，只保留为诊断工具；
- SAM3 boundary loss 的早期版本：没有改善，真正有价值的下一步应是
  双向 silhouette 约束或可微 renderer，而不是继续调旧的单边 mask 项。

## 详细实验记录

本文记录当前从 HaMeR/MANO hand-local 输出到 PN glove 局部坐标的最佳监督校准层。
校准在偶数 `group_id` 帧上训练，并在奇数 `group_id` 帧上评估。

重要说明：这是一层由 glove 监督的设备/局部坐标校准层。它不应被解读为
无 GT 条件下 HaMeR/MANO 精度本身的提升。原始 MANO `palm_local_joints_m`
会保留；校准后的坐标写入 `glove_calibrated_palm_local_joints_m`。

## 为什么需要它

允许 translation 后带来的明显收益，以及校准后再把 wrist 重新置中时的损失，
说明 MANO/HaMeR 和 glove local frame 之间存在稳定的局部坐标/关节定义不匹配。
这个误差源比逐帧 jitter 更大。

## 奇数帧 holdout 结果

### Right Index 序列

`gloves/glove_local/pn3_rightindex_camera_sync_g356p000_c17p000_cut_000000_000477.jsonl`

| 方法 | Mean mm | Median mm | RMSE mm | P95 mm | Max mm |
| --- | ---: | ---: | ---: | ---: | ---: |
| 原始 MANO local | 41.03 | 35.11 | 46.83 | 91.43 | 130.46 |
| 相似变换，允许 translation | 28.24 | 22.87 | 34.05 | 67.38 | 116.87 |
| 相似变换 + joint offsets, k=200, max=30mm | 22.57 | 16.23 | 29.53 | 64.73 | 112.49 |
| 相似变换 + joint offsets, k=25, max=25mm | 20.51 | 13.40 | 28.39 | 63.96 | 109.86 |
| 相似变换 + joint offsets, k=50, max=30mm | 20.68 | 13.71 | 28.42 | 64.02 | 110.41 |
| 相似变换 + wrist 重新置中 | 43.71 | 38.99 | 47.93 | 80.01 | 128.76 |

### Left Index 序列

`gloves/glove_local/pn3_leftindex_camera_sync_g414p000_c47p000_cut_000000_000442.jsonl`

| 方法 | Mean mm | Median mm | RMSE mm | P95 mm | Max mm |
| --- | ---: | ---: | ---: | ---: | ---: |
| 原始 MANO local | 33.75 | 31.70 | 36.85 | 64.39 | 139.68 |
| 相似变换，允许 translation | 22.23 | 21.29 | 24.73 | 39.98 | 105.32 |
| 相似变换 + joint offsets, k=200, max=30mm | 16.74 | 14.85 | 19.47 | 31.87 | 103.74 |
| 相似变换 + joint offsets, k=25, max=25mm | 14.48 | 12.18 | 17.75 | 30.99 | 102.80 |
| 相似变换 + joint offsets, k=50, max=30mm | 14.80 | 12.56 | 17.94 | 30.97 | 102.99 |
| 相似变换 + wrist 重新置中 | 29.23 | 27.59 | 31.40 | 48.37 | 109.60 |

## 全手指奇数帧 holdout 结果

上面的表延续了早先 right/left index 报告的设置，只评估 `thumb,index,middle`。
最佳校准也在五根手指上做了评估。

| 序列 | 方法 | Mean mm | Median mm | RMSE mm | P95 mm | Max mm |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| left_index | 原始 MANO local | 30.12 | 27.45 | 33.61 | 59.70 | 139.68 |
| left_index | 最佳校准 | 14.08 | 11.86 | 17.46 | 31.52 | 102.80 |
| right_index | 原始 MANO local | 38.83 | 32.56 | 45.36 | 89.36 | 130.46 |
| right_index | 最佳校准 | 20.56 | 13.12 | 28.22 | 62.11 | 109.86 |
| left_index | 相似变换 + bone scales + joint offsets | 17.00 | 14.53 | 20.69 | 35.54 | 115.52 |
| right_index | 相似变换 + bone scales + joint offsets | 23.90 | 16.78 | 32.01 | 70.97 | 128.71 |

最佳校准后的逐手指奇数帧 holdout：

| 序列 | Thumb mean/P95 | Index mean/P95 | Middle mean/P95 | Ring mean/P95 | Pinky mean/P95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| left_index | 13.49 / 29.70 | 12.78 / 26.61 | 17.18 / 38.88 | 14.24 / 32.87 | 12.72 / 31.93 |
| right_index | 11.97 / 29.41 | 23.95 / 69.45 | 25.61 / 73.30 | 23.48 / 67.70 | 17.79 / 49.48 |

right 序列仍然存在明显的远端手指尾部误差，尤其在 index/middle/ring 附近。
单纯静态 local calibration 无法解决这个残差。

bone-scale 实验被有意保留在报告中作为负结果。它降低了训练误差，但让两个序列的
奇数帧 holdout 都变差了。许多拟合出的 bone scale 还撞到了保守的 `[0.70, 1.30]`
边界，这说明模型很可能在吸收 MANO-vs-glove 关节定义差异和 pose 分布偏差，
而不是学到稳定的解剖骨长修正。因此 `--bone-scales` 默认应保持关闭。

## 当前推荐

使用以下监督校准：

- 带 scale、rotation、translation 的相似变换
- 每个 joint 的残差 offset
- `joint_offset_shrink_k=25`
- `max_joint_offset_m=0.025`
- `bone_scales=none`
- 写入模式 `separate`

推荐输出为：

- `video/sam3_hamer_right_index/hamer_mano_local_glove_calibrated/mano_local_hands_similarity_translate_jointoffset_k025_m025_even_train_000000_000477.jsonl`
- `video/sam3_hamer_left_index/hamer_mano_local_glove_calibrated/mano_local_hands_similarity_translate_jointoffset_k025_m025_even_train_000000_000442.jsonl`

评估器使用的 space：

```bash
--space glove-calibrated-palm-local
```

## 跨序列迁移

为了测试校准是否只是在记忆某个片段，我们把一个序列上得到的最佳校准应用到另一个序列，
并且不使用目标序列的 glove 数据进行拟合。

| 目标序列 | 校准来源 | Mean mm | Median mm | RMSE mm | P95 mm | Max mm |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| left_index | none/original | 30.15 | 27.44 | 33.65 | 59.52 | 140.17 |
| left_index | right_index calibration | 16.07 | 12.72 | 19.97 | 40.62 | 99.52 |
| left_index | left_index even-train calibration | 14.11 | 11.86 | 17.51 | 31.69 | 102.80 |
| right_index | none/original | 38.87 | 32.62 | 45.41 | 89.39 | 131.97 |
| right_index | left_index calibration | 23.71 | 15.11 | 32.27 | 75.76 | 118.28 |
| right_index | right_index even-train calibration | 20.59 | 13.14 | 28.25 | 62.29 | 109.86 |

这是一个有用的 sanity check。跨序列迁移仍然比原始 MANO local 输出更好，
所以这个 correction 并不只是帧级记忆。不过，同序列校准明显更好，
这说明一部分误差取决于 pose 分布、采集条件或序列特定的 HaMeR bias。

## 带分布保护的 pose-dependent residual

`scripts/calibrate_pose_residual_local.py` 在推荐静态校准之后，加入一个低容量 ridge
回归器：从居中的 21-joint HaMeR/MANO 手型预测剩余 glove-local 残差。
当目标动作来自同一采集序列时，它显著降低了 pose-dependent 的远端手指误差。

模型会存储归一化后的训练 pose prototypes。应用时，它用最近 prototype 距离作为
out-of-distribution (OOD) 测试。完整 residual correction 会保留到 leave-one-out
训练距离的第 75 百分位，并在第 99 百分位线性衰减为 0。因此 OOD pose 会回退到
静态 `k=25, max=25mm` 校准，而不会得到一个不相关的 pose correction。

| 验证设置 | left_index mean / P95 mm | right_index mean / P95 mm |
| --- | ---: | ---: |
| 静态校准，奇数 holdout | 14.08 / 31.52 | 20.56 / 62.11 |
| Pose residual，奇数 holdout | 6.67 / 19.33 | 13.03 / 42.21 |
| Pose residual + OOD guard，奇数 holdout | 6.77 / 19.77 | 13.09 / 42.31 |
| 静态校准，后半连续片段 | 13.80 / 31.00 | 18.25 / 57.51 |
| 前半训练 pose residual + OOD guard，后半评估 | 13.33 / 30.45 | 17.71 / 52.94 |

odd/even 切分对快速迭代仍然有用，但相邻帧会让结果偏乐观。contiguous-half 结果是更保守的检查：
大多数后半段 pose 都被 guard 拒绝了（left: 333/442 hand instances，right: 338/478），
但 in-distribution 的剩余部分仍然改善了整个 held-out 片段。

跨序列应用确认了为什么 guard 是必要的：

| 目标 | 来自另一序列的无保护 pose residual | 来自另一序列的有保护 pose residual | 目标静态校准 |
| --- | ---: | ---: | ---: |
| left_index | 17.08 / 41.92 | 14.09 / 31.54 | 14.11 / 31.69 |
| right_index | 17.55 / 61.22 | 20.36 / 62.32 | 20.59 / 62.29 |

数值为 mean/P95 mm。有保护版本会故意拒绝迁移一个源 pose 分布无法覆盖目标的校准；
它基本回到静态 baseline，而不是像无保护版本那样在 left 上产生明显退化。

对于有本序列 glove calibration clip 的序列，推荐 pose-residual 设置是：
`all-joints`、`ridge_alpha=10`、`correction_shrink=0.75`、
`max_correction_m=0.03` 和 `ood_gating=knn-linear`。生成的两个同序列校准文件是：

- `video/sam3_hamer_left_index/hamer_mano_local_glove_calibrated/pose_residual_ood_alljoints_a010_s075_m030_even_train_000000_000442.json`
- `video/sam3_hamer_right_index/hamer_mano_local_glove_calibrated/pose_residual_ood_alljoints_a010_s075_m030_even_train_000000_000477.json`

## Local Pose-Residual KNN

剩余残差并不是全局线性的。校准脚本现在也支持同一个归一化手部 pose feature space 中的
local KNN regressor。它只使用存储的 HaMeR/MANO pose prototypes 及其 glove residuals；
在应用前仍然会乘以 OOD gate。它的训练诊断采用 leave-one-out，因此不会通过查询当前校准帧本身
报告人为的零误差。

最佳 dense-coverage 设置是 `local-knn`、`k=2`、bandwidth scale `0.5`、
correction shrink `0.75`，以及每 joint `60mm` cap。它在刻意交错的奇数帧 holdout 上大幅改善：

| 序列 | Global ridge + OOD, mean / P95 / max mm | Local KNN + OOD, mean / P95 / max mm |
| --- | ---: | ---: |
| left_index | 6.77 / 19.77 / 89.06 | 4.09 / 10.06 / 89.06 |
| right_index | 13.09 / 42.31 / 81.29 | 5.87 / 17.64 / 67.76 |

逐手指 KNN mean/P95 mm：left thumb 到 pinky 分别为 `3.74/8.06`、`3.74/8.67`、
`4.98/12.26`、`4.21/10.49`、`3.76/10.56`；right 分别为 `3.38/8.90`、
`6.70/18.44`、`7.56/22.87`、`6.76/19.74`、`4.97/14.34`。
right index/middle/ring 的远端尾部仍然是最不可靠区域，但已经比 global ridge 小得多。

这有意被解释为 **dense calibration-coverage** 结果，而不是声称任意早期片段都能预测所有未来手部动作。
在更困难的 first-half-train/later-half-eval 测试中，带保守 30mm cap 的 KNN 与 global ridge 接近
（left 上 `13.43/30.92` vs `13.33/30.45`，right 上 `17.64/52.89` vs `17.71/52.94`）。
把 cap 提到 60mm 对 left 无害，但会把 right later-half P95 提高到 `53.66mm`。
因此，只有当 calibration set 密集覆盖目标 pose space 时才使用 60mm KNN calibration；
否则保留 30mm ridge/KNN calibration 或静态校准。

dense-coverage 输出为：

- `video/sam3_hamer_left_index/hamer_mano_local_glove_calibrated/mano_local_hands_similarity_translate_jointoffset_k025_m025_pose_knn2_bw050_ood_m060_even_train_000000_000442.jsonl`
- `video/sam3_hamer_right_index/hamer_mano_local_glove_calibrated/mano_local_hands_similarity_translate_jointoffset_k025_m025_pose_knn2_bw050_ood_m060_even_train_000000_000477.jsonl`
- `video/sam3_hamer_left_index/hamer_mano_local_glove_calibrated/pose_knn2_bw050_ood_m060_even_train_000000_000442.json`
- `video/sam3_hamer_right_index/hamer_mano_local_glove_calibrated/pose_knn2_bw050_ood_m060_even_train_000000_000477.json`

把任意一个 KNN calibration 应用到另一个序列，在 left 上得到 `14.09/31.58`，
right 上得到 `20.36/62.32`，基本等同于静态校准 baseline。最近 pose OOD gate 让它变安全：
跨序列时，861/886 个 left hand instances 和 711/956 个 right hand instances 收到零 local correction。

## Temporal Smoothing Ablation

smoothing utility 现在支持 `--space glove-calibrated-palm-local`，因此可以作用在 evaluator 实际消费的字段上。
在严格 first-half-train/later-half-eval 切分中，robust Hampel replacement 在 35mm 阈值下没有发现 outlier。
双向 EMA smoothing 平均只改变 joint 约 0.1mm，并把最终误差从 left_index 的 `13.33 / 30.45`
变为 `13.32 / 30.53`，从 right_index 的 `17.71 / 52.94` 变为 `17.68 / 52.82`。
它可能让离线 viewer 略微更平稳，但不是 accuracy-critical 阶段，应保持 optional。

## Pose-Velocity Descriptor Ablation

`all-joints-velocity` 拼接当前居中的 21-joint 手型，以及来自相邻 HaMeR 帧的居中有限差分 motion。
它不使用相邻 glove labels。在 first-half-train/later-half-eval 实验中，ridge alpha 100 时，
它把 OOD-guarded pose-only model 从 left_index 的 `13.33 / 30.45` 改善到 `12.00 / 28.98`，
从 right_index 的 `17.71 / 52.94` 改善到 `16.06 / 46.51`（mean/P95 mm）。

它 **没有** 通过 interleaved odd-frame holdout：left_index 从 `6.77 / 19.77` 变为 `7.65 / 20.97`，
right_index 从 `13.09 / 42.31` 变为 `14.16 / 44.06`。因此这个 descriptor 只保留为实验性的
chronological-clip 选项，而不是默认。任何使用都应由符合“先校准、后使用”工作流的连续验证切分来选择；
不要只根据训练误差或相邻帧切分来选择它。

## Pure Apply 模式

已有 calibration JSON 文件可以应用到新的 HaMeR JSONL，而不需要 glove GT：

```bash
python3 scripts/calibrate_hamer_to_glove_local.py \
  --hamer video/sam3_hamer_left_index/hamer_mano_local_refined/mano_local_hands_000000_000442.jsonl \
  --output video/sam3_hamer_left_index/hamer_mano_local_glove_calibrated/mano_local_hands_apply_right_calibration_k025_m025_000000_000442.jsonl \
  --load-calibration-json video/sam3_hamer_right_index/hamer_mano_local_glove_calibrated/similarity_translate_jointoffset_k025_m025_even_train_000000_000477.json \
  --space palm-local \
  --group-range 0-442 \
  --write-mode separate \
  --overwrite
```

## 解释

结果表明，当前最大的剩余误差来源是稳定的 MANO-to-glove 关节定义不匹配，以及
pose/capture-dependent HaMeR bias 的组合，尤其在远端关节处。时间 jitter 相对较小。
如果生产环境没有目标 session 的 glove GT，只应使用同一 subject/device setup 下估计出的静态校准，
或者应用带 OOD guard 的 pose-residual calibration，让陌生动作可靠地回退到静态结果。

## 相关工作说明

当前方向与更广泛的 MANO/HaMeR 文献一致：

- HaMeR 使用大型 transformer 模型从单目图像回归 MANO hand reconstruction；
  它是很强的拓扑/pose prior，但输出的仍然是模型 convention，而不是 glove-device convention。
  https://arxiv.org/abs/2312.05251
- MANO 本身是从手部扫描中学习得到的低维手模型。它的 joints、root 和 blend-shape conventions
  都是模型定义，不一定精确匹配 glove SDK 的 local joint definitions。
  https://arxiv.org/abs/2201.02610
- 早期 hand mesh recovery 工作同样依赖 differentiable reprojection 和 parametric hand models，
  这进一步说明 image-space fitting 和 model-space calibration 是两件不同的事。
  https://arxiv.org/abs/1902.09305

## Image-Space Multi-View 诊断

image-space MANO refiner 目前仍然有意默认关闭。直接 raw comparison 发现，
之前的 image-space 输出尚未改善 glove-local hand shape：left local MANO mean 为 `30.15mm`，
image-space 为 `31.03mm`；right 为 `38.87mm` vs `39.51mm`。旧 run 几乎没有接受任何
metric multi-view observations，因为其最终 mean reprojection errors 大约是 `180-350px`。

主要原因现在已经明确。HaMeR `cam_t` 定义在 HaMeR 的虚拟 crop focal length 下，
而 refiner 使用物理 rectified camera intrinsics 做投影。把这个 virtual-camera translation
直接当作 physical initialization 会让手处在错误的 metric scale 上。实验性的
`--global-initialization physical-pnp` 现在通过真实 rectified K，从 MANO local joints
和 HaMeR 2D points 初始化 global pose。在一个五帧 left smoke test 上，它把 C1 anchor
reprojection 从约 184px 降到 21px，但 C0/C2/C3 与同一个 physical pose 仍然不一致。
最终 local GT error 没有改善（`39.32mm` vs `36.76mm` baseline），所以在 per-camera
geometry correction 被验证前，PnP 仍保持 opt-in。

这次调查还修复了 `fuse_hamer_jobs.py` 中一个具体的 SAM3 集成 bug：
来自 stabilized tracks 的陈旧相对 mask path 在 relocation 时丢失了 `chunks/` 组件。
因此，在 HaMeR 前，3,211 个 left-sequence SAM3 masks 全部被静默丢弃。resolver 现在能找到
全部 3,211 个文件；0-4 smoke run 确认所有 35 个 fused HaMeR jobs 都获得 mask-blurred 输入。
仅修正 mask 并没有改善五帧 local-MANO sample（`37.33mm` vs `36.76mm`），
所以在 auxiliary-camera geometry 问题解决前，不建议做昂贵的 full left rerun。

另一个 opt-in `--pnp-view-gate-m` 现在会在辅助观测的 physical-K PnP wrist estimate
与 anchor 不一致时拒绝该观测。0.10m 的五帧 mask smoke 把 PnP image refinement
从 `40.79mm` mean 改善到 `38.26mm`，但仍未超过 local MANO（`36.76mm`）。
在启用 gate 时强制 C1/C2 作为 anchor 也更差（`39.52mm`）。这些开关仍是诊断工具，
不是生产默认。下一步真正合理的 image-space 改进需要单独验证 camera SE(3) correction，
或使用更独立的 2D observation，而不是继续调 heuristic loss。

## Mask-Enabled PnP Gate 后续

原始 HaMeR prediction JSONL 文件意外地包含零个可用的 `sam3_mask_path`，
所以它们的 image-space mask loss 实际从未激活。陈旧的 stabilized-track paths 已在
`fuse_hamer_jobs.py` 中修复，left- 和 right-index 0--20 shards 都重新 fusion 并重新跑过 HaMeR：
每个 shard 的 156 个 jobs 都带有有效 SAM3 mask，并使用 blur-masked HaMeR 输入。

在真实 mask-enabled 输入下，决定性改善并不是来自新的 mask term，而是在 optimize 前拒绝
几何不一致的辅助视角。`physical-pnp` initialization 加 `--pnp-view-gate-m 0.04`
与相同 predictions 且 gate disabled 的结果进行了比较：

| 序列 | Gate off mean / P95 / max | 0.04m gate mean / P95 / max |
| --- | --- | --- |
| left-index, groups 0--20 | 38.33 / 69.83 / 118.20 mm | **34.06 / 54.38 / 81.57 mm** |
| right-index, groups 0--20 | 53.37 / 112.98 / 131.62 mm | **49.22 / 110.57 / 126.37 mm** |

权重 `0.10` 的 symmetric SAM3-boundary-to-mesh 实验 loss 没有被提升为默认：
在 left-index 上，它把 mean/P95 从 `38.33/69.83mm` 变为 `38.48/70.45mm`。
它仍然是显式 opt-in 实验，默认权重为 0。

相比之下，尝试从 MANO+HaMeR 2D PnP 拟合单个 global camera SE(3) correction
在真正 held-out 测试中被否定：其表面改善是 hand/reference-specific 的，
并且可能严重恶化另一个 camera pair。该 estimator 只保留为诊断 artifact。
这不同于 PnP view gate；后者不改变 calibration，只在某一帧/视角自身的 physical-K pose
与该帧 anchor 不一致时排除该视角。因此，image-space refinement 仍然是 pipeline opt-in，
但启用时默认现在是 `physical-pnp` 和保守的 `0.04m` PnP gate。

## Baseline/Gated Candidate Selection

最初的 `500px` threshold 规则是有意保守的，但它没有泛化到完整序列。
更广泛的独立评估显示，只要 PnP-gated candidate 存在，它就是更强的默认选择：

| 序列 | Baseline mean / P95 | Always gated mean / P95 |
| --- | --- | --- |
| left-index, groups 0--442 | 36.71 / 67.14 mm | **34.70 / 63.72 mm** |
| right-index, groups 0--49 | 46.62 / 102.41 mm | **43.65 / 100.33 mm** |
| right-index, groups 0--99 | 46.96 / 102.24 mm | **45.11 / 98.92 mm** |
| right-index, groups 0--477 | 44.41 / 99.42 mm | **42.02 / 93.20 mm** |

这组比较在每一对结果中使用相同 predictions、masks 和 calibration；
glove GT 只用于离线评估。先前的 `500px` 规则在 right shard 中只保留 43/100 个 gated hand candidates，
得到较弱的 `46.01mm` mean。有用的信号是 gate 的几何拒绝本身，
而不是很大的 baseline reprojection residual。

因此，`scripts/select_image_refinement_candidates.py` 现在默认偏好所有可用 PnP-gated candidate
（`--min-baseline-max-reprojection-px 0`）。正阈值仍可作为显式实验使用。
缺失或失败的 gated hands 仍会回退到 baseline。pipeline 会把这个 selected result 写到
`hamer_mano_multiview_selected/`，viewer 会优先读取它。image refinement 也默认要求至少 50%
可读取 SAM3 masks，避免把 mask loss 静默失效的 legacy predictions 当作有效实验。

### Pipeline 重建证据

修复后的端到端 pipeline 在五个连续 left-index shards（groups 0--249）上运行，
每个 shard 都有 100% 可读取 SAM3-mask coverage。selected shards 经过严格 duplicate/conflict
检查后合并，并在 4,500 个 glove-evaluated points 上与对应 no-gate baseline 比较：

| 输出 | Mean | Median | RMSE | P95 | Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline image refine | 37.43 mm | 35.09 mm | 40.61 mm | 66.00 mm | 142.48 mm |
| selected baseline/gated image refine | **36.20 mm** | **33.61 mm** | **39.42 mm** | **64.89 mm** | 142.48 mm |

这是一个真实 pipeline-level gain，不是手工构造的 smoke output。selected output 存储在：
`video/sam3_hamer_left_index/hamer_mano_multiview_selected/mano_multiview_local_hands_000000_000249.jsonl`。

完整 left-index 序列（443 frames，groups 0--442）随后被重建为九个不重叠 shard，
并在无重复 group IDs 的情况下合并。所有 886 个 hand records 都有 finite palm-local joints。
在 7,974 个 glove-evaluated points 上，no-gate baseline 与 selected output 的比较为：

| 输出 | Mean | Median | RMSE | P95 | Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline image refine | 36.71 mm | 34.14 mm | 40.09 mm | 67.14 mm | 142.48 mm |
| old 500px selector | 35.66 mm | 33.07 mm | 39.03 mm | 65.26 mm | 142.48 mm |
| PnP-gated candidate（新默认） | **34.70 mm** | **32.40 mm** | **37.76 mm** | **63.72 mm** | 144.04 mm |

旧 selector 为 265/886 个 hand frames 选择了 gated candidate。新默认会选择所有完整的 gated hand candidates。
合并 artifact：
`video/sam3_hamer_left_index/hamer_mano_multiview_selected/mano_multiview_local_hands_000000_000442.jsonl`
已经用该规则重新生成：886/886 个 hand records 现在都来自 PnP-gated candidate，
零 baseline fallbacks，且 palm-local joints 都是 finite。
right-index groups 0--477 的 selected shards 也用同样方式重新生成，并干净合并到：
`video/sam3_hamer_right_index/hamer_mano_multiview_selected/mano_multiview_local_hands_000000_000477.jsonl`。
所有 956/956 个 hand records 现在都来自 PnP-gated candidate，零 baseline fallbacks，
且 palm-local joints 都是 finite。

## Selected 图像输出作为 Glove 校准底座

selected image-space 输出适合作为当前无 GT 图像侧 viewer/result，
但它**不是**更好的 glove-supervised local calibration 底座。我们把同样的
even-train/odd-eval calibration stack 应用于 `hamer_mano_multiview_selected/`，
并与现有的 `hamer_mano_local_refined/` 校准底座进行比较：

| 校准底座 | left_index static mean / P95 | right_index static mean / P95 | left_index KNN mean / P95 | right_index KNN mean / P95 |
| --- | ---: | ---: | ---: | ---: |
| `hamer_mano_local_refined` | **14.08 / 31.52 mm** | **20.56 / 62.11 mm** | **4.09 / 10.05 mm** | **5.87 / 17.64 mm** |
| `hamer_mano_multiview_selected` | 15.80 / 38.73 mm | 21.60 / 62.59 mm | 5.38 / 14.38 mm | 7.57 / 24.05 mm |

因此，即使经过 dense local-KNN residual correction，selected 底座也会让 supervised
输出变差。hand-level odd-frame 诊断显示，selected KNN 只在 left sequence 的
110/442 个 hands 和 right sequence 的 103/478 个 hands 上胜出。
hand-level mean error delta 仍然分别变差 `+1.29mm` 和 `+1.69mm`。
低 image reprojection error 也没有逆转结论：max-reprojection 最低四分位仍然在
left 上差 `+0.59mm`，在 right 上差 `+0.76mm`。

因此，应把两类输出分开使用：

- 无 GT 图像侧可视化/debugging 使用 `hamer_mano_multiview_selected/`；
- glove-supervised local coordinates 继续使用 `hamer_mano_local_refined/`
  作为校准底座，并应用现有 static/KNN calibration stack。

## 严格时间顺序 Glove 校准

前面的 even-train/odd-eval 数字适合衡量插值能力，但它会让 static
similarity/joint-offset calibration 同时看到序列早期和晚期的姿态。
为了去掉这类泄漏，static calibration 脚本现在支持显式
`--train-group-range` 和 `--train-group-ids` 参数。下表中的所有校准层
都只在每条序列的前半段训练，然后在 held-out 后半段评估：

| Calibration output | left_index late-half mean / P95 / max | right_index late-half mean / P95 / max |
| --- | ---: | ---: |
| static similarity + translation + joint offsets | 15.15 / 32.36 / 102.35 mm | 20.60 / 63.16 / 101.09 mm |
| static + pose residual, all-joint features | 14.57 / 31.61 / 102.39 mm | 19.83 / 59.49 / 101.09 mm |
| static + pose residual, fingertip-summary features | 14.23 / 31.55 / 103.80 mm | 19.36 / 56.64 / 101.09 mm |
| static + pose + velocity residual | **12.52 / 29.40** / 111.79 mm | **17.84 / 49.34 / 97.37 mm** |

因此，在严格时间顺序设置下，pose+velocity residual 是目前 mean 和 P95
最强的版本，尤其是在 right-index 上。不过 left-index 的 max 会变差，
这说明它更适合作为 sequence-level calibration option，而不是无条件的 dense 默认。
真正重要的实践变化是：后续如果目标是接近部署场景的 held-out report，
就应该使用显式 training range，而不是只依赖 parity。

## Image-Space Beta 实验

原始 sequence beta estimator 使用 weighted HaMeR-local joint fit。新增的实验路径
`--beta-estimation-space image-2d` 会固定每个 observation 的 HaMeR pose 和 physical-PnP transform，
然后根据 calibrated multi-view HaMeR 2D reprojection 优化共享 10-D beta。
这是一个真正的 image-space shape update，而不是针对 glove GT 的优化。

在 left-index gated shard（groups 0--49）上，它让 left/right beta vectors 的 L2 改变量为
`0.050/0.046`，但 glove 结果基本中性：

| Beta estimator | Mean | Median | RMSE | P95 | Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| HaMeR-local（当前默认） | 32.57 mm | 31.56 mm | 33.87 mm | **48.70 mm** | **79.80 mm** |
| physical-PnP image 2D | **32.53 mm** | **31.51 mm** | **33.84 mm** | 48.75 mm | 79.83 mm |

这个极小的 mean 变化不足以支持切换默认，也不值得增加默认运行时间。
实现保留为显式实验 CLI 路径；下一步有用的 shape signal 应该在序列范围内引入
differentiable SAM3 silhouette term，而不是更多 HaMeR 2D self-consistency。

## Second-Order Temporal Prior 实验

`--temporal-acceleration-weight` 会在已有连续两个 hand states 后加入二阶 local pose/joint prior。
它在独立 50-frame left/right shards 上，以 `0.10` 权重对 PnP-gated candidate 进行了测试：

| 序列 | Current mean / P95 / max | Acceleration mean / P95 / max |
| --- | --- | --- |
| left-index 0--49 | 32.570 / 48.697 / 79.804 mm | **32.560 / 48.667 / 79.575 mm** |
| right-index 0--49 | 43.647 / 100.327 / **127.565 mm** | **43.644 / 100.217** / 127.691 mm |

它在 mean/RMSE/P95 上方向略正，但收益太小，而且 right maximum 略差。
因此它保持默认关闭，而不是基于弱结果被提升为默认。更关键的剩余缺口是一个真正的
双向 SAM3 silhouette objective；当前安装环境没有 PyTorch3D、nvdiffrast 或 Kaolin renderer，
所以这应通过显式 renderer 依赖，或经过仔细验证的轻量替代方案来实现。

## Zero-Shot 直接多视角 Palm Fusion

最终目标是 in-the-wild 和 zero-shot，因此 glove 数据不能进入融合方法。
`scripts/fuse_hamer_palm_local.py` 按这个约束实现了一条新路径：

1. 把每个视角的 HaMeR 结果分别转换到自身 canonical palm frame；
2. 图像质量分数只用于同一相机内重复 hypothesis 的选择；
3. 入选相机保持等权，直接平均有对应关系的 joints；
4. 无条件把该结果保存在 `raw_palm_local_joints_m`；
5. 静态 shape 校准、因果 EMA、离线 Gaussian smoothing 都写入独立字段，除非显式指定
   `--primary-output`，否则绝不覆盖 raw 结果。

生成的 config 会明确写入 `uses_ground_truth: false`、
`cross_view_weighting: equal` 和主输出字段。主 pipeline 默认执行该阶段，但部署安全默认仍是
`--zero-shot-primary-output raw`、bone calibration 为 0、temporal smoothing 为 0。

### 为什么跨视角保持等权

我们把图像质量和跨视角一致性都作为无 GT reliability signal 做过评估，但它们不足以安全 gate pose：

- quality score 与误差的相关系数在 left-index 为 `-0.18`，right-index 为 `+0.10`；
- consensus spread 与误差的相关系数分别只有 `-0.089` 和 `+0.019`；
- per-joint Huber fusion 虽然让 left mean 略降，却明显损害 right 的 mean 和尾部误差，
  因为 C3 有时最准确，同时又是最不像其他视角的那个 view。

因此，quality score 被刻意限制为“同相机重复候选去重”。跨视角 disagreement 会作为诊断量输出，
但不能被解释为 accuracy confidence，也不用于 hard rejection。

### Zero-Shot 结果

下面所有数字都只在 inference 完成后用 glove 做离线评估。融合器本身不会读取 glove 文件、
label、校准参数或 residual model。指标覆盖五根手指，以及完整 left 0--442、right 0--477 序列。

| Zero-shot 输出 | left mean / median / P95 / max | right mean / median / P95 / max |
| --- | ---: | ---: |
| 之前的 MANO local refine | 30.12 / 27.45 / 59.70 / 139.68 mm | 38.83 / 32.56 / 89.36 / 130.46 mm |
| 直接等权跨视角均值，raw | 28.47 / 26.10 / 53.68 / 111.67 mm | 35.50 / 30.27 / 77.33 / 130.17 mm |
| 直接均值 + 因果 EMA `alpha=0.20` | 27.45 / 25.18 / 51.89 / 101.28 mm | 34.55 / 29.51 / 74.26 / 121.15 mm |
| 直接均值 + 自适应因果 One Euro | **27.07 / 24.83 / 51.59 / 101.16 mm** | **34.02 / 28.85 / 73.32 / 117.62 mm** |
| 直接均值 + 离线 Gaussian `radius=10,sigma=4` | **27.37 / 25.08 / 51.47 / 93.92 mm** | **34.31 / 29.40 / 73.06 / 120.25 mm** |

Gaussian 版本需要向后看十帧，只是 offline option，不能冒充 causal deployment 结果。
所有 temporal filter 遇到 detection 缺帧都会重置，不会把旧 pose 穿过长缺口拖到新片段。
无论选择哪种模式，raw 字段始终是 authoritative observation。

自适应因果结果使用时间戳估计的约 25 FPS：`min_cutoff=0.2`、`beta=5.0`、derivative
cutoff `1.0`。参数只在偶数 group 上选择，再到未参与选参的奇数 group 验证；奇数 group 的
left/right mean/P95 为 `27.05/51.46mm` 和 `33.98/73.29mm`，优于固定 EMA 的
`27.44/51.69mm` 和 `34.51/74.11mm`。它不使用未来帧，推理时不读取 glove；pipeline
默认计算这个可选字段，但主输出仍保持 raw。

### 不使用 Pose 监督的静态校准

可选的 zero-shot static bone calibration 会按相机先求中位数，再得到每只手的 camera-balanced
骨长目标；每根骨骼沿该帧原始方向重建。它只修改 shape length，不拟合 pose residual，
也不读取 glove。在两条序列上，它对 mean 和 P95 的影响都不到 `0.1mm`，max 还有正有负，
所以 `--bone-calibration-blend` 默认保持 `0`。它可以作为保守的 shape normalization 实验，
但不能被包装成默认精度收益。

我们也测试了 15 个 MANO joint rotation 的 SO(3) 平均。它能保证输出仍是严格合法的 MANO
参数，但加同样的离线时间窗口后，left 为 `28.74/53.89mm`、right 为
`35.47/78.01mm`（mean/P95），仍不如直接 joint correspondence fusion。
因此它更适合作为 mesh-valid fallback 方向，而不是 joint accuracy 默认方案。

### 几何分支修复与限制

`scripts/triangulate_mediapipe_hands.py` 原来假设 detection 按 `group_id` 连续排列，
但真实 JSONL 是按 camera 分块的。结果每条 record 都被当成单相机 group，最终 0 只手能三角化。
现在 loader 会显式按 `group_id` 聚合全部相机；脚本还支持用
`--tracked-hands` 通过 bbox IoU 做可选的稳定轨迹身份关联，不修改 2D landmarks。

完成分组、track association 和 `0.05--1.0m` 正深度 gate 后，严格 MediaPipe
triangulation 在 left/right 上分别得到 396/415 个 hand instances；完整 pose 的
mean/P95 仍为 `41.31/76.63mm` 和 `48.11/99.06mm`，无论覆盖率还是精度都不足以替代
HaMeR fusion。
直接三角化 HaMeR rendered 2D joints 也被否决：virtual-camera 与 physical-camera 不匹配会产生
严重 depth failure。几何分支暂时保留为诊断，等 identity association 和 physical 2D observation
足够可靠后再进入主融合。

### 命令

部署安全的 raw 融合，同时把可选时序输出保存在独立字段：

```bash
/home/luojiangrui/miniconda3/envs/headcam/bin/python scripts/fuse_hamer_palm_local.py \
  --predictions video/sam3_hamer_left_index/hamer_per_view/hamer_predictions_000000_000442.jsonl \
  --output-dir video/sam3_hamer_left_index/hamer_palm_local_fused \
  --group-range 0-442 \
  --temporal-radius 10 \
  --temporal-sigma 4 \
  --causal-ema-alpha 0.20 \
  --one-euro-min-cutoff 0.2 \
  --one-euro-beta 5.0 \
  --primary-output raw \
  --overwrite
```

经过验证的因果结果使用 `--primary-output adaptive-causal`；只有在明确做 offline 输出时才使用
`--primary-output smoothed`。查看 raw zero-shot skeleton：

```bash
/home/luojiangrui/miniconda3/envs/headcam/bin/python scripts/view_hamer_multiview.py \
  --dataset left_index --range 0-442 --zero-shot
```

## 全部实验结果总表

这张表用于统一索引本文所有已完成实验。除非“指标口径”另有说明，L/R 均按
`mean / P95 / max` 记录，单位为 mm；`—` 表示原实验没有报告该项，而不是 0。
不同数据范围、手指数或切分方式不能只按数值直接排名，必须同时看“范围/切分”。

| ID | 实验族 | 范围/切分 | 方法或变量 | L 指标 | R 指标 | 结论/状态 |
| --- | --- | --- | --- | ---: | ---: | --- |
| C01 | 静态校准基线 | 全序列、五指、even→odd | 原始 MANO local | 30.12 / 59.70 / 139.68 | 38.83 / 89.36 / 130.46 | 未校准基线 |
| C02 | 静态校准主结果 | 全序列、五指、even→odd | similarity + translation + joint offset，`k=25,max=25mm` | **14.08 / 31.52 / 102.80** | **20.56 / 62.11 / 109.86** | **最安全的 glove-supervised 默认** |
| C03 | Bone scale | 全序列、五指、even→odd | C02 + bone scales | 17.00 / 35.54 / 115.52 | 23.90 / 70.97 / 128.71 | 负结果，默认关闭 |
| C04 | 静态变换消融 | thumb/index/middle、even→odd | similarity + translation | 22.23 / 39.98 / 105.32 | 28.24 / 67.38 / 116.87 | translation 必要但不充分 |
| C05 | Joint offset shrink | thumb/index/middle、even→odd | `k=200,max=30mm` | 16.74 / 31.87 / 103.74 | 22.57 / 64.73 / 112.49 | 过度收缩 |
| C06 | Joint offset shrink | thumb/index/middle、even→odd | `k=50,max=30mm` | 14.80 / 30.97 / 102.99 | 20.68 / 64.02 / 110.41 | 接近 C02，但综合略弱 |
| C07 | Wrist recenter | thumb/index/middle、even→odd | similarity 后重新以 wrist 置中 | 29.23 / 48.37 / 109.60 | 43.71 / 80.01 / 128.76 | 明显退化，禁用 |
| C08 | 跨序列静态迁移 | 全序列 | right→left / left→right calibration | 16.07 / 40.62 / 99.52 | 23.71 / 75.76 / 118.28 | 好于原始，但弱于同序列 |
| C09 | 同序列静态迁移对照 | 全序列 | 本序列 even-train calibration | 14.11 / 31.69 / 102.80 | 20.59 / 62.29 / 109.86 | 验证存在序列相关 bias |
| C10 | Global pose residual | 全序列、even→odd | ridge，无 OOD | 6.67 / 19.33 / — | 13.03 / 42.21 / — | 插值切分上有效 |
| C11 | Global pose residual | 全序列、even→odd | ridge + OOD guard | 6.77 / 19.77 / 89.06 | 13.09 / 42.31 / 81.29 | 略保守，可回退静态层 |
| C12 | Pose residual 时序切分 | first-half→later-half | static + all-joint ridge + OOD | 13.33 / 30.45 / — | 17.71 / 52.94 / — | 比 odd/even 更可信 |
| C13 | 跨序列 pose residual | 全序列 | 无 OOD guard | 17.08 / 41.92 / — | 17.55 / 61.22 / — | 新姿态可能错误外推 |
| C14 | 跨序列 pose residual | 全序列 | 有 OOD guard | 14.09 / 31.54 / — | 20.36 / 62.32 / — | 基本安全回退到静态层 |
| C15 | Dense local KNN | 全序列、even→odd | KNN2 + OOD，60mm cap | **4.09 / 10.06 / 89.06** | **5.87 / 17.64 / 67.76** | 仅限目标 pose 被密集覆盖 |
| C16 | Local KNN 时序切分 | first-half→later-half | 保守 30mm cap | 13.43 / 30.92 / — | 17.64 / 52.89 / — | 与 global ridge 接近 |
| C17 | Local KNN cap | first-half→later-half | 60mm cap | left 基本不变 | — / 53.66 / — | right P95 变差，不作时序默认 |
| C18 | Temporal smoothing | first-half→later-half | 双向 EMA | 13.32 / 30.53 / — | 17.68 / 52.82 / — | 变化约 0.1mm，仅 viewer 可选 |
| C19 | Pose+velocity | first-half→later-half | all-joints-velocity，ridge 100 | 12.00 / 28.98 / — | 16.06 / 46.51 / — | 时序切分有效 |
| C20 | Pose+velocity 反证 | even→odd | 同一 descriptor | 7.65 / 20.97 / — | 14.16 / 44.06 / — | 弱于 pose-only，不作全局默认 |
| C21 | Image-space raw | 全序列 | local MANO →旧 image refine | 30.15→31.03 mean | 38.87→39.51 mean | 旧 virtual/physical camera 混用退化 |
| C22 | Physical-PnP smoke | left 5 帧 | local MANO / PnP refine | 36.76 / — / — → 39.32 / — / — | — | reprojection 改善但 local GT 变差 |
| C23 | SAM3 mask path 修复 | left 5 帧 | mask 前/后 | 36.76→37.33 mean | — | 修复输入 bug，但本身不增精度 |
| C24 | PnP view gate | groups 0--20 | gate off → `0.04m` | 38.33/69.83/118.20 → **34.06/54.38/81.57** | 53.37/112.98/131.62 → **49.22/110.57/126.37** | image refine 启用时的保守默认 |
| C25 | SAM3 boundary loss | left groups 0--20 | weight 0 → 0.10 | 38.33/69.83/— → 38.48/70.45/— | — | 负结果，默认 0 |
| C26 | Global camera SE(3) | held-out camera pairs | 单一全局 correction | 不稳定 | 不稳定 | 仅诊断，不进入默认 |
| C27 | Gated candidate selection | 完整/多范围 | baseline → always gated | 36.71/67.14/— → **34.70/63.72/—** | 44.41/99.42/— → **42.02/93.20/—** | 无 GT 图像侧当前默认 |
| C28 | 旧 500px selector | left 0--442 | baseline / old / always gated | 36.71/67.14/142.48 / 35.66/65.26/142.48 / **34.70/63.72/144.04** | — | 大阈值不泛化 |
| C29 | Pipeline 级 gate | left 0--249 | baseline → selected | 37.43/66.00/142.48 → **36.20/64.89/142.48** | — | 真实 4500 点增益 |
| C30 | Selected 作校准底座 | 全序列、even→odd | local-refined → selected，static | **14.08/31.52/—** → 15.80/38.73/— | **20.56/62.11/—** → 21.60/62.59/— | selected 不适合作监督校准底座 |
| C31 | Selected 作 KNN 底座 | 全序列、even→odd | local-refined → selected，KNN | **4.09/10.05/—** → 5.38/14.38/— | **5.87/17.64/—** → 7.57/24.05/— | 继续使用 local-refined 底座 |
| C32 | 严格时间静态层 | first-half→later-half | static similarity + offsets | 15.15 / 32.36 / 102.35 | 20.60 / 63.16 / 101.09 | 最保守时序基线 |
| C33 | 严格时间 pose residual | first-half→later-half | all-joint features | 14.57 / 31.61 / 102.39 | 19.83 / 59.49 / 101.09 | 小幅改善 |
| C34 | 严格时间 pose residual | first-half→later-half | fingertip-summary | 14.23 / 31.55 / 103.80 | 19.36 / 56.64 / 101.09 | 更强但仍非最佳 |
| C35 | 严格时间 pose+velocity | first-half→later-half | pose + velocity | **12.52 / 29.40 / 111.79** | **17.84 / 49.34 / 97.37** | mean/P95 最好；left max 变差，序列级可选 |
| C36 | Image-space beta | left 0--49 | HaMeR-local → physical-PnP 2D | 32.57/48.70/79.80 → 32.53/48.75/79.83 | — | mean 极小改善、P95 略差，默认不切换 |
| C37 | 二阶时间 prior | 0--49 | current → acceleration 0.10 | 32.570/48.697/79.804 → **32.560/48.667/79.575** | 43.647/100.327/**127.565** → **43.644/100.217**/127.691 | 收益过小，默认关闭 |
| C38 | Zero-shot 基线 | 全序列、五指 | MANO local refine | 30.12 / 59.70 / 139.68 | 38.83 / 89.36 / 130.46 | zero-shot 旧基线 |
| C39 | Zero-shot 融合 | 全序列、五指 | 等权多视角 raw | 28.47 / 53.68 / 111.67 | 35.50 / 77.33 / 130.17 | **部署安全主字段** |
| C40 | Zero-shot 时序 | 全序列、五指 | causal EMA `0.20` | 27.45 / 51.89 / 101.28 | 34.55 / 74.26 / 121.15 | 因果可选字段 |
| C41 | Zero-shot 时序 | 全序列、五指 | adaptive causal One Euro | **27.07 / 51.59 / 101.16** | **34.02 / 73.32 / 117.62** | 最佳因果字段，但不覆盖 raw |
| C42 | Zero-shot 时序 | 全序列、五指 | offline Gaussian `r=10,s=4` | 27.37 / **51.47** / **93.92** | 34.31 / **73.06** / 120.25 | 非因果，仅离线 |
| C43 | One Euro 独立 holdout | odd groups | One Euro / fixed EMA mean/P95 | **27.05/51.46** / 27.44/51.69 | **33.98/73.29** / 34.51/74.11 | 参数未用 odd 选取 |
| C44 | Zero-shot bone normalization | 全序列 | bone blend >0 | mean/P95 变化 <0.1 | mean/P95 变化 <0.1 | 无稳定收益，默认 blend 0 |
| C45 | MANO SO(3) 平均 | 全序列、离线窗口 | rotation averaging | 28.74 / 53.89 / — | 35.47 / 78.01 / — | mesh-valid fallback，不作 joint 默认 |
| C46 | MediaPipe triangulation | 全序列 | track + positive-depth gate | 41.31 / 76.63 / —，396 hands | 48.11 / 99.06 / —，415 hands | 覆盖和精度都不足 |
| C47 | HaMeR rendered 2D triangulation | 全序列 | virtual-camera 2D + physical K | 严重 depth failure | 严重 depth failure | 否决 |

安全性优先级：无 glove GT 时使用 C39 的 raw zero-shot 字段；有明确同设备/同 session glove
校准片段时使用 C02；只有 pose 覆盖被验证为密集时才升级到 C15。C35 虽然严格时间切分的
mean/P95 更低，但包含运动描述符且会恶化 left max，因此仍是 sequence-level opt-in。

C02 的调用链与 optimization 路线相互独立：`calibrate_hamer_to_glove_local.py` 仍默认
`--write-mode separate`，不会覆盖 `palm_local_joints_m`。现有 calibration JSON 的 pure-apply
冒烟测试确认原字段逐值不变，并成功写入 4 个 `glove_calibrated_palm_local_joints_m` 手记录。
需要注意两点：优化后的 HaMeR pipeline 默认不再生成 legacy `hamer_mano_local_refined/`，从头重跑时
要显式传 `--run-mano-local-refine`；已有该输入文件时可直接照旧调用。旧 HaMeR/MANO→glove
calibration 也不能未经重新拟合就套到 MobRecon 输出上，因为两者是不同模型域。
