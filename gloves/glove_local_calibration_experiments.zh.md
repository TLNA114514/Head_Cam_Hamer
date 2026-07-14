# Glove-Local 校准方法与实验概览

> 本文汇总 Glove-Local 校准方法、适用条件、关键指标与推荐配置。完整实验过程、参数消融和
> 47 项结果见
> [技术附录](glove_local_calibration_experiments.technical.zh.md)。

## 核心结论

这里的“校准”不是重新训练 HaMeR，也不是让模型凭空变准，而是解决两个系统的定义差异：

- 相机模型输出的是 HaMeR/MANO 定义的手部局部坐标；
- PN glove 输出的是数据手套定义的局部坐标；
- 两者的坐标方向、尺度和每个关节的位置定义并不完全一样。

因此，即使 HaMeR 已经正确识别了手势，也需要一层类似“单位换算 + 坐标对齐”的校准，才能和
glove 数值直接比较。

当前建议如下：

1. **没有 glove 校准数据**：使用 zero-shot 等权多视角 raw 输出；
2. **有同设备、同 session 的 glove 校准片段**：使用静态校准，这是最安全的监督方案；
3. **校准动作已经密集覆盖未来动作**：可以使用 KNN + OOD guard，精度更高；
4. **严格按“前半段校准、后半段使用”**：pose+velocity 的平均指标最好，但 left 最坏点会变差，
   所以仍是可选项；
5. 图像侧 PnP-gated 输出适合 viewer/debug，但不要拿它替代监督校准的 MANO-local 底座。

## 指标说明

所有误差单位都是 mm，并且越低越好。

| 指标 | 定义 | 适合观察什么 |
| --- | --- | --- |
| Mean | 所有点平均错多少 | 整体是否准确 |
| Median | 一半的点低于多少误差 | 典型帧表现 |
| P95 | 95% 的点低于多少误差 | 大部分坏帧是否受控 |
| Max | 最差的一个点错多少 | 极端失败案例 |

只看 Mean 容易掩盖少量严重失败，所以本文通常同时看 Mean 和 P95。

## 整体流程

```text
相机图像
  ├─ HaMeR/MANO → MANO palm-local 手型
  │                  → 有 glove 标定片段？
  │                      ├─ 否：保留 zero-shot raw 输出
  │                      └─ 是：应用 MANO → glove 的监督校准
  │                               ├─ 静态校准：最安全
  │                               ├─ pose residual：只修正熟悉姿态
  │                               └─ local KNN：只用于姿态被密集覆盖时
  └─ MobRecon → 独立的 palm-local 输出；不能直接套用旧 MANO calibration
```

## 方向一：有 Glove 数据的监督校准

### 方法 A：静态校准

静态校准会学习：

- 整体缩放；
- 整体旋转和平移；
- 每个关节一个幅度受限的小偏移。

它不依赖当前动作附近必须有训练样本，因此是最稳妥的默认方案。

推荐参数：

```text
similarity + translation
joint offsets: mean
joint_offset_shrink_k: 25
max_joint_offset_m: 0.025
bone_scales: none
write_mode: separate
```

| 序列 | 原始 Mean / P95 | 静态校准 Mean / P95 |
| --- | ---: | ---: |
| left_index | 30.12 / 59.70 | **14.08 / 31.52** |
| right_index | 38.83 / 89.36 | **20.56 / 62.11** |

### 方法 B：Pose Residual + OOD Guard

静态校准之后，模型仍可能在某些手势上有固定偏差。Pose residual 会根据当前手型再做一次小修正。

风险是：遇到没见过的动作时，它可能修错。OOD guard 的作用就是判断“这个动作是否像校准集”，
不像时自动退回静态校准。

| 切分 | left Mean / P95 | right Mean / P95 |
| --- | ---: | ---: |
| 偶数帧训练、奇数帧评估 | 6.77 / 19.77 | 13.09 / 42.31 |
| 前半段训练、后半段评估 | 13.33 / 30.45 | 17.71 / 52.94 |

奇偶帧相邻、动作相似，所以第一行偏乐观；第二行更接近真实部署。

### 方法 C：Local KNN + OOD Guard

KNN 会寻找校准集中最相似的几个姿态，并参考它们的误差进行修正。它在动作被密集覆盖时非常准确，
但校准集不完整时不应盲目使用。

| 序列 | 静态 Mean / P95 | Dense KNN Mean / P95 |
| --- | ---: | ---: |
| left_index | 14.08 / 31.52 | **4.09 / 10.06** |
| right_index | 20.56 / 62.11 | **5.87 / 17.64** |

这些最佳数字代表“插值”，不代表它能预测任意新动作。

### 方法 D：Pose + Velocity

该方法除了看当前手型，还看最近的运动方向。在严格前半段训练、后半段评估中，它的 Mean/P95 最好：

| 序列 | 静态 Mean / P95 / Max | Pose+velocity Mean / P95 / Max |
| --- | ---: | ---: |
| left_index | 15.15 / 32.36 / 102.35 | **12.52 / 29.40** / 111.79 |
| right_index | 20.60 / 63.16 / 101.09 | **17.84 / 49.34 / 97.37** |

left 的 Max 从 102.35 增加到 111.79，因此它是 sequence-level 可选方案，而不是最安全默认。

## 方向二：没有 Glove 数据的 Zero-Shot 输出

Zero-shot 表示运行时完全不读取 glove GT。

### Raw 等权多视角融合

每个相机先独立得到 palm-local 关节，再对同名关节做等权平均。它简单、可解释，并且不会因为一个
不可靠的“质量分数”错误删除实际最好的相机。

Raw 始终保存在 `raw_palm_local_joints_m`，是最安全、可回退的原始观察值。

### 因果滤波与离线滤波

- EMA：固定平滑强度；
- One Euro：静止时更平滑，快速运动时减少滞后；
- Gaussian：会读取未来帧，只能离线使用。

| 输出 | left Mean / P95 / Max | right Mean / P95 / Max |
| --- | ---: | ---: |
| Raw 等权融合 | 28.47 / 53.68 / 111.67 | 35.50 / 77.33 / 130.17 |
| 因果 EMA | 27.45 / 51.89 / 101.28 | 34.55 / 74.26 / 121.15 |
| 因果 One Euro | **27.07 / 51.59 / 101.16** | **34.02 / 73.32 / 117.62** |
| 离线 Gaussian | 27.37 / **51.47 / 93.92** | 34.31 / **73.06** / 120.25 |

部署默认仍建议保留 raw 为主字段；需要更平稳的实时输出时，再显式选择 One Euro 字段。

## 方向三：图像侧多视角优化

这条路线根据每个相机里的 2D 关节和相机内参，对 3D MANO mesh 做图像空间优化。

### Physical-PnP View Gate

某个辅助相机推算出的手腕位置如果和主视角差得过远，就排除该视角。它不修改相机标定，只拒绝
明显不一致的观察。

| 输出 | left Mean / P95 | right Mean / P95 |
| --- | ---: | ---: |
| Baseline image refine | 36.71 / 67.14 | 44.41 / 99.42 |
| PnP-gated selected | **34.70 / 63.72** | **42.02 / 93.20** |

这适合无 GT viewer/debug，但经过 glove 校准后仍不如 `hamer_mano_local_refined`，所以两条路径要分开。

## 方向四：已经排除或默认关闭的方法

| 方法 | 为什么不作为默认 |
| --- | --- |
| Bone scale | 训练集更好，但两个 holdout 序列都变差 |
| Wrist 重新置中 | 破坏了静态 translation 校准带来的收益 |
| SAM3 boundary loss 旧版本 | Mean/P95 都没有改善 |
| Global camera SE(3) correction | 在不同相机组合上不稳定 |
| Image-space beta | Mean 只改善 0.04mm，P95 略差 |
| 二阶 acceleration prior | 收益太小，right Max 略差 |
| MediaPipe triangulation | 覆盖率和精度都不足 |
| HaMeR rendered 2D triangulation | virtual camera 与 physical camera 不一致，深度失败 |

## 推荐静态校准命令

拟合新的静态校准：

```bash
python scripts/calibrate_hamer_to_glove_local.py \
  --hamer video/sam3_hamer_left_index/hamer_mano_local_refined/mano_local_hands_000000_000442.jsonl \
  --glove gloves/glove_local/pn3_leftindex_camera_sync_g414p000_c47p000_cut_000000_000442.jsonl \
  --output video/sam3_hamer_left_index/hamer_mano_local_glove_calibrated/static_safe.jsonl \
  --calibration-json video/sam3_hamer_left_index/hamer_mano_local_glove_calibrated/static_safe.json \
  --space palm-local \
  --train-parity even \
  --allow-scale \
  --allow-translation \
  --joint-offsets mean \
  --joint-offset-shrink-k 25 \
  --max-joint-offset-m 0.025 \
  --bone-scales none \
  --write-mode separate \
  --overwrite
```

应用已有 calibration JSON 时不需要 glove 文件，改用 `--load-calibration-json` 即可。

### 与 Optimization 路线的兼容性

校准脚本本身没有被新优化路线替换，`--write-mode separate` 仍会保留原始
`palm_local_joints_m`，另写 `glove_calibrated_palm_local_joints_m`。现有 calibration JSON
已经通过 pure-apply 冒烟测试。

但需要区分三件事：

1. 已有 `hamer_mano_local_refined/` 输入时，可以照旧调用；
2. 从优化后的 HaMeR pipeline 重新生成输入时，要显式加入 `--run-mano-local-refine`；
3. HaMeR/MANO 上拟合的 calibration 不能直接套到 MobRecon 输出，切换模型后必须重新拟合和评估。

## 方法与结果总表

下表是方法概览索引；完整 47 项实验及全部消融见技术附录的 C01--C47。

| 方向 | 方法 | 使用前提 | left Mean / P95 | right Mean / P95 | 建议 | 技术编号 |
| --- | --- | --- | ---: | ---: | --- | --- |
| 监督校准 | 原始 MANO local | 无校准 | 30.12 / 59.70 | 38.83 / 89.36 | 仅作基线 | C01 |
| 监督校准 | 静态 similarity + offsets | 有同设备 glove 片段 | **14.08 / 31.52** | **20.56 / 62.11** | **最安全默认** | C02 |
| 监督校准 | Bone scale | 有 glove | 17.00 / 35.54 | 23.90 / 70.97 | 关闭 | C03 |
| 监督校准 | Pose residual + OOD | 熟悉姿态 | 6.77 / 19.77 | 13.09 / 42.31 | 可选 | C11 |
| 监督校准 | Dense KNN + OOD | 姿态密集覆盖 | **4.09 / 10.06** | **5.87 / 17.64** | 条件满足时使用 | C15 |
| 严格时序 | 静态校准 | 前半→后半 | 15.15 / 32.36 | 20.60 / 63.16 | 时序安全基线 | C32 |
| 严格时序 | Pose + velocity | 前半→后半 | **12.52 / 29.40** | **17.84 / 49.34** | 序列级可选 | C35 |
| Zero-shot | MANO local refine | 无 glove | 30.12 / 59.70 | 38.83 / 89.36 | 旧基线 | C38 |
| Zero-shot | 等权多视角 raw | 无 glove | 28.47 / 53.68 | 35.50 / 77.33 | **部署安全主字段** | C39 |
| Zero-shot | 因果 EMA | 无 glove、实时 | 27.45 / 51.89 | 34.55 / 74.26 | 可选字段 | C40 |
| Zero-shot | 因果 One Euro | 无 glove、实时 | **27.07 / 51.59** | **34.02 / 73.32** | 最佳因果字段 | C41 |
| Zero-shot | Gaussian | 无 glove、离线 | 27.37 / **51.47** | 34.31 / **73.06** | 仅离线 | C42 |
| 图像侧 | Baseline image refine | viewer/debug | 36.71 / 67.14 | 44.41 / 99.42 | 基线 | C27 |
| 图像侧 | PnP-gated selected | viewer/debug | **34.70 / 63.72** | **42.02 / 93.20** | 图像侧默认 | C27 |
| 几何分支 | MediaPipe triangulation | 完整多视角检测 | 41.31 / 76.63 | 48.11 / 99.06 | 不采用 | C46 |
