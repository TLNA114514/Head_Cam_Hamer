# HaMeR / MobRecon 推理优化概览

> 本文汇总推理架构、性能瓶颈、优化方法、关键指标与推荐配置。完整性能记录、消融、命令和
> 49 项实验见
> [技术附录](hamer_inference_optimization.technical.md)。

## 核心结论

当前有两条可用路线：

1. **HaMeR 兼容路线**：保留原模型，主要减少重复加载、无效候选和大文件 I/O；
2. **MobRecon 实时路线**：SAM3 找手，光流在中间帧跟踪，MobRecon 估计 3D 手型，再融合多相机结果。

已经通过完整序列验收的实时配置是：

```text
相机：C0,C2,C3
SAM3：GPU，2 个常驻 worker，每 10 帧刷新
中间帧：LK 光流跟踪
MobRecon：CPU FP32，单模型实例
输出：因果 One Euro
```

它在 left/right 完整序列上达到：

| 序列 | 冷启动端到端 FPS | 稳态 FPS | Mean / P95 / Max | SAM3 显存 |
| --- | ---: | ---: | ---: | ---: |
| left | **10.779** | 13.928 | **32.675 / 60.780 / 93.506mm** | 11028MiB |
| right | **11.104** | 14.132 | **34.955 / 72.136 / 111.500mm** | 11028MiB |

如果更重视泛化精度，应保留四路相机的 per-view 结果，而不是永远固定删除 C1；目前只有两个完整样本，
C1 在 left 有帮助、在 right 却有害，还不足以学出普适相机权重。

## 核心术语

| 名词 | 说明 |
| --- | --- |
| 冷启动 FPS | 从启动程序、加载模型开始计时，最严格 |
| 稳态 FPS | 模型已经加载后，持续运行的速度 |
| 延迟 | 一帧从进入系统到得到结果要多久 |
| 显存 reserved | PyTorch 为模型和临时计算保留的 GPU 内存 |
| Keyframe | 真正运行 SAM3 重新找手的帧 |
| 光流 | 根据图像纹理运动，把上一帧的手框移动到下一帧 |
| Mean / P95 / Max | 平均误差 / 95% 误差上界 / 最坏误差，越低越好 |

## 系统架构

```text
多路相机图像
  → SAM3 每隔若干帧重新检测手框
  → 中间帧用光流传播手框
  → MobRecon 对每个手框估计 3D 关节
  → 在 palm-local 坐标中融合多相机结果
  → One Euro 做因果平滑
  → 输出 raw + filtered 两份关节
```

## 方向一：模型选择

HaMeR 精度强、兼容旧流程，但模型大、推理慢。MobRecon 更轻，输入只需 `128×128` 手部 crop。

| 完整序列 | MobRecon Mean / P95 | HaMeR Mean / P95 | MobRecon CPU 序列 FPS |
| --- | ---: | ---: | ---: |
| left | **35.913 / 68.962** | 39.732 / 78.371 | 11.11 |
| right | **36.863 / 72.099** | 41.360 / 84.786 | 13.09 |

这不代表 MobRecon 在所有公开数据集上都比 HaMeR 准，只说明它在当前黑色手套、crop 和 palm-local
评估中更合适。

### MobRecon 放 CPU 还是 GPU

MobRecon 单独运行时，GPU 明显更快：

| 设备 | left 0--99 四相机处理 FPS | P95 延迟 | 显存 | 精度 Mean / P95 |
| --- | ---: | ---: | ---: | ---: |
| CPU FP32 | 12.128 | 95.081ms | 0 | 32.667 / 58.485 |
| GPU FP32 | **19.024** | **46.579ms** | 378MiB | 32.667 / 58.485 |
| GPU FP16 | 20.525 | 48.577ms | 360MiB | 32.537 / 59.954 |

但两个 SAM3 已经同时占用 GPU 时，GPU MobRecon 会和它们争抢计算资源，整条链路反而从
10.748 降到 9.783 FPS。因此：

- 2 个 SAM3 worker：优先 CPU MobRecon；
- 1 个 SAM3 常驻服务：GPU FP32 MobRecon 可以小幅降低延迟；
- 精度优先不使用 FP16 MobRecon，因为 P95 变差约 1.47mm。

## 方向二：SAM3 并行策略

“四个相机”不等于“必须启动四个 SAM3 模型”。一个 SAM3 模型可以顺序处理多路相机，并且只需加载一次。

| SAM3 配置 | 四相机稳态结果 | 结论 |
| --- | ---: | --- |
| 1 SAM3 + CPU MobRecon | 11.061 FPS | 最简单、显存约 5514MiB |
| 1 SAM3 + GPU MobRecon | **11.505 FPS** | 模型常驻时可选 |
| 2 SAM3 + CPU MobRecon | 10.748 FPS | 当前四相机异构并行方案 |
| 2 SAM3 + GPU MobRecon | 9.783 FPS | GPU 计算竞争，不采用 |
| 4 个独立 SAM3 进程 | OOM | 24GB 卡仍不安全 |

四进程失败时 GPU 已占约 23.54GiB，第四个进程再申请 82MiB 就 OOM。不能只用
`4 × 5514MiB < 24GB` 的静态算术判断，因为还有 CUDA context、临时 workspace、桌面和驱动占用。

## 方向三：相机组合选择

视觉上最正、遮挡最少的相机不一定能得到最低 3D 误差。模型域、手套反光、crop、相机标定和手掌朝向
都会影响结果。

| 相机组合 | left Mean / P95 / Max | right Mean / P95 / Max |
| --- | ---: | ---: |
| C1,C2 | 33.957 / 61.298 / 101.843 | 37.527 / 80.692 / 143.577 |
| C0,C2,C3 | 32.636 / 61.136 / **93.436** | **34.982 / 72.196 / 111.538** |
| C0,C1,C2,C3 | **32.482 / 59.114** / 95.137 | 35.570 / 74.737 / 120.967 |

所以当前建议分成两种：

- 已通过 10 FPS 冷启动验收的部署配置：C0,C2,C3；
- 精度研究和新用户/新动作采集：四路都保留，积累更大验证集后再学习相机可靠度。

不要仅凭这两个样本把 C1 永久删除，也不要直接对内部一致性最低的相机做 hard rejection；已有实验表明，
“最不像其他视角”的相机有时反而最准。

## 方向四：Keyframe 频率与光流风险

每 10 帧运行一次 SAM3，意味着中间 9 帧依赖光流。风险确实存在：快速运动、遮挡、运动模糊或手框内
纹理太少时，光流可能失败或静默漂移。

当前数据上的结果：

| Stride | SAM3 keyframe 数 | 稳态 FPS | Mean / P95 / Max |
| ---: | ---: | ---: | ---: |
| 10 | 40 | **10.748** | **32.681 / 58.497 / 93.314** |
| 8 | 52 | 10.352 | 32.901 / 58.766 / 94.511 |

更频繁的 stride 8 没有改善这段精度，反而更慢。完整 left/right 序列各出现 2 次显式光流失败，下一
keyframe 后恢复。

前后向 LK 检查也测试过：框指标只有极小改善，最终 3D 略差，所以保留为诊断开关，默认关闭。

更合理的下一步是**自适应刷新**：平时 stride 10；检测到特征点减少、前后向误差升高、bbox 突变或
多视角 3D 明显不一致时，只对异常相机提前运行 SAM3。当前文件式 producer 还不能反向请求刷新，
这属于下一阶段架构工作。

## 方向五：HaMeR 工程优化

这些优化主要减少重复工作，不改变模型本身：

| 优化 | 说明 | 实测结果 | 精度影响 |
| --- | --- | ---: | --- |
| 每序列只加载一次 | 不再每个相机/分片重复读 2.5GiB checkpoint | left 36→1 次；right 40→1 次 | 无 |
| 跨 job packing | 把多个小 batch 合并 | 11.610→11.284s | 最大差 1.19e-7m |
| Singleton 不做 mask render | 只有一个候选时无需评分 | 79.987→46.349s | joints 差 0 |
| 无 mask 只跑 1.0 尺度 | 三个候选必然选 1.0 时不重复推理 | 11.237→3.751s | joints 差 0 |
| Joint-only JSON | 不保存默认不用的 vertices/rotation | JSON 减少约 93% | 无 |
| 默认关闭 overlay/debug 图 | 避免大量图片写盘 | 每序列避免约 0.8--1.1GiB | 无 |

Skeleton scorer、balanced 和 FP16 等方案虽然更快，但会改变候选或尾部误差，因此只作为显式速度档位。

## 配置选择

| 目标 | 推荐配置 |
| --- | --- |
| 需要已验证的冷启动 ≥10 FPS | C0,C2,C3 + 2 SAM3 GPU + CPU MobRecon FP32 + stride 10 |
| 模型常驻、希望保留四相机 | 2 SAM3 GPU + CPU MobRecon FP32，或 1 SAM3 + GPU MobRecon FP32 |
| 精度优先研究 | 保留四路 per-view、FP32、raw + One Euro 两份输出 |
| 保持 HaMeR 接口 | `quality` profile + per-sequence 加载 + job packing |
| 离线最快但允许精度变化 | 单独验证 `balanced` / `aggressive`，不要直接用于正式输出 |

## 已验收实时命令

```bash
conda run --no-capture-output -n hamer python scripts/run_sparse_sam3_mobrecon.py \
  --base-dir video/sam3_hamer_left_index \
  --frames video/cameras_left_index/frames.jsonl \
  --cameras C0,C2,C3 \
  --group-range 0-442 \
  --keyframe-stride 10 \
  --sam3-workers 2 \
  --sam3-checkpoint /path/to/sam3.pt \
  --sam3-no-hf \
  --sam3-amp-dtype float16 \
  --sam3-torch-threads 2 \
  --mobrecon-device cpu \
  --mobrecon-torch-threads 8 \
  --one-euro-min-cutoff 0.25 \
  --one-euro-beta 0.05 \
  --overwrite
```

## 方法与结果总表

下表是方法概览索引；完整 49 项实验和四相机全部 15 种组合见技术附录 O01--O49。

| 方向 | 方法/配置 | 速度或资源 | left 精度 | right 精度 | 建议 | 技术编号 |
| --- | --- | ---: | ---: | ---: | --- | --- |
| 模型 | HaMeR baseline | 大模型 | 39.732 / 78.371 | 41.360 / 84.786 | 兼容路线 | O02 |
| 模型 | MobRecon CPU | L 11.11 / R 13.09 FPS，不含 SAM3 | **35.913 / 68.962** | **36.863 / 72.099** | 实时 mesh 基线 | O01 |
| 最终链路 | 三相机完整序列 | **10.779 / 11.104 cold FPS** | **32.675 / 60.780** | **34.955 / 72.136** | **已验收默认** | O10--O13 |
| 相机 | C1,C2 | — | 33.957 / 61.298 | 37.527 / 80.692 | 不因视觉好直接采用 | O21 |
| 相机 | C0,C2,C3 | 已验收吞吐 | 32.636 / 61.136 | **34.982 / 72.196** | 当前部署默认 | O26 |
| 相机 | 四相机 | 保留四路结果 | **32.482 / 59.114** | 35.570 / 74.737 | 精度研究默认 | O28 |
| 调度 | 1 SAM3 + CPU MobRecon | 11.061 steady FPS | — | — | 简单稳定 | O31 |
| 调度 | 1 SAM3 + GPU MobRecon | **11.505 steady FPS** | — | — | 常驻服务可选 | O32 |
| 调度 | 2 SAM3 + CPU MobRecon | 10.748 steady FPS | 32.681 / 58.497 | 35.361 / 69.399 | 四相机可用 | O29 |
| 调度 | 2 SAM3 + GPU MobRecon | 9.783 steady FPS | 32.667 / 58.485 | — | 不采用 | O30 |
| 调度 | 4 SAM3 进程 | OOM | — | — | 禁止 | O33 |
| MobRecon | GPU FP32 standalone | 19.024 FPS；378MiB | 32.667 / 58.485 | — | 单独运行最佳 | O35 |
| MobRecon | GPU FP16 standalone | 20.525 FPS | 32.537 / 59.954 | — | P95 退化，不作精度默认 | O36 |
| Keyframe | stride 10 | **10.748 steady FPS** | **32.681 / 58.497** | — | 当前默认 | O37 |
| Keyframe | stride 8 | 10.352 steady FPS | 32.901 / 58.766 | — | 未改善 | O38 |
| 光流 | 前后向 LK 阈值 1.5 | 稍慢 | 32.509 / 59.314 | 35.608 / 75.237 | 默认关闭 | O39 |
| HaMeR 优化 | 单例跳过 mask render | **1.73×** | joints 不变 | — | 默认启用 | O42 |
| HaMeR 优化 | 无 mask 自适应单尺度 | **3.00×** | joints 不变 | — | 条件满足时启用 | O43 |
| 速度档 | Skeleton scorer | 1.65× | — | Mean 略好、P95/Max 变差 | 仅 aggressive | O44--O45 |
| 速度档 | Balanced 单尺度 | 1.641 jobs/s | — | P95 +2.682mm | 非质量默认 | O46 |
| I/O | Joint-only + 关闭 debug | 每序列避免约 0.8--1.5GiB | 不变 | 不变 | 默认启用 | O47--O49 |
