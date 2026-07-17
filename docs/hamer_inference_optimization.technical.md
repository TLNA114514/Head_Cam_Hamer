# HaMeR 推理速度与延迟优化：技术附录

> 本文保留完整性能实验、消融和全部结果。架构选择与关键结论见
> [推理优化概览](hamer_inference_optimization.md)。

更新时间：2026-07-14。

本文只讨论推理速度、启动延迟、吞吐、显存/内存和输出 I/O。姿态误差实验仍记录在：

- `gloves/glove_local_calibration_experiments.md`
- `gloves/glove_local_calibration_experiments.technical.zh.md`

## 结论

“不换模型”这条路已接近当前工程优化的收益上限，但整个任务并没有到速度/精度极限。
2026-07-14 新增的 MobRecon 路线证明：把 HaMeR 换成轻量手网格模型后，手网格阶段已经能在
CPU 上超过 10 个四相机序列帧/秒，而且两段完整手套评估的 mean/P95 都优于现有 HaMeR 输出。
进一步加入稀疏 SAM3 keyframe、光流跟踪、相机消融、CPU 线程隔离和单进程实时 worker 后，
最终 `双 SAM3 GPU + CPU MobRecon + 在线 One-Euro` 链路已经在 left/right 完整序列上分别达到
`10.78/11.10 FPS`；这两个数字都包含模型冷启动、真实 SAM3 推理、跟踪、手网格、融合和输出写入。

因此现在有两条路线：

1. 质量兼容路线继续保留 HaMeR，并使用下面的执行优化；
2. 逼近实时路线使用 `稀疏 SAM3 ROI -> 光流跟踪 -> MobRecon -> palm-local fusion`，完全不使用 MediaPipe；
3. 已验收的吞吐默认仍选 `C0,C2,C3` 三相机；精度优先实验应显式保留四相机预测，不应把两段序列上的相机排名当作固定先验；
4. 双 SAM3 的 `peak_cuda_reserved_mib` 均为 `5514MiB`，合计 `11028MiB`（约 `10.77GiB`），满足 24GiB 约束；四个独立 SAM3 进程实测会在 24GB 卡上 OOM，不能由单进程静态 reserved 简单外推；
5. 默认调度器使用流式重叠：SAM3 每输出一条 keyframe 就立即 flush，CPU worker 跟随消费，不等整段检测完成；
6. SAM3 默认限制为 2 个 CPU thread，MobRecon 使用 8 个 thread，避免两个 GPU producer 抢占 CPU worker；
7. 输出默认使用完全因果的 One-Euro 滤波（`min_cutoff=0.25Hz, beta=0.05`），原始关节仍保存在
   `raw_palm_local_joints_m`，可随时关闭或回退；
8. 当前目标“完整序列、冷启动计入、至少 10 FPS、双 SAM3 总显存不超过 24GiB”已经完成实测验收。

HaMeR 兼容路线建议：

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

## MobRecon 实时手网格路线

### 实现

新增 `scripts/mobrecon_multiview_worker.py`：

- 直接读取现有 SAM3/HaMeR jobs，不改 ROI 生产阶段；
- 使用一个 MobRecon DenseStack 实例跨四相机 batch；
- 输入为 `128x128` crop，默认 SAM3 bbox 外扩 `1.5x`；
- `--image-source rectified` 可直接使用原图 crop，不要求逐帧生成 mask-blurred frame；
- 左手先水平翻转成右手域，输出后恢复镜像；
- 用仓库自带 `j_reg.npy` 从 778 vertices 回归 21 joints，并转为项目使用的 MPII 顺序；
- 输出通用 `hand_mesh_multiview_prediction`，现有 `fuse_hamer_palm_local.py` 已同时兼容 HaMeR 和 MobRecon；
- config 记录模型大小、模型/预处理耗时、jobs/s 和 CUDA peak allocated/reserved；
- unknown handedness 默认只对同帧同相机恰好两个未知框做从左到右的互斥分配，不调用 MediaPipe 姿态模型。

官方 MobRecon DenseStack 权重约 `47MB`，加载后的参数张量约 `46.21MiB`，参数量
`12,114,720`。当前本地权重位置为：

```text
external/HandMesh/pretrained/mobrecon_densestack.pt
```

拓扑索引需要 `openmesh`：

```bash
conda run --no-capture-output -n hamer pip install openmesh
```

若本地没有权重，可用官方 Google Drive 文件 ID 下载：

```bash
mkdir -p external/HandMesh/pretrained
conda run --no-capture-output -n hamer gdown \
  1QKtt5x-8Xe_afjpMTBIk2TI3G5QGk_iu \
  -O external/HandMesh/pretrained/mobrecon_densestack.pt
```

- 官方代码：<https://github.com/SeanChenxy/HandMesh>
- 官方论文：<https://arxiv.org/abs/2112.02753>

### 完整序列实测

所有数字均为当前 i9-12900K、`torch-threads=12`、FP32、batch 8 的真实运行，不含 SAM3 检测，
也没有用 GPU 估算替代实测。每个序列帧包含四相机、最多两手，因此“序列 FPS”按
`jobs/s / (jobs / frames)`计算。

| 序列 | frames | jobs | 总耗时 | jobs/s | 折算序列 FPS | model load |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| left | 443 | 3211 | 39.869s | 80.538 | **11.11** | 0.279s |
| right | 478 | 3461 | 36.523s | 94.763 | **13.09** | 0.262s |

手套评估使用全部匹配帧、左右手和 thumb/index/middle 的 9 个关节点：

| 序列 | 模型 | mean | median | P95 | max |
| --- | --- | ---: | ---: | ---: | ---: |
| left | MobRecon | **35.913mm** | **32.207mm** | **68.962mm** | 140.224mm |
| left | HaMeR baseline | 39.732mm | 35.253mm | 78.371mm | **129.867mm** |
| right | MobRecon | **36.863mm** | **33.141mm** | **72.099mm** | **121.621mm** |
| right | HaMeR baseline | 41.360mm | 36.161mm | 84.786mm | 127.323mm |

结论不是“MobRecon 绝对比 HaMeR 精确”，而是在当前黑色手套、SAM3 crop 和 palm-local 指标上，
轻模型的完整序列 mean/P95 更好。left 的单点 max 比 HaMeR 差 `10.36mm`，因此仍需保留
尾部异常门控或时间滤波。

left 0--49 另做了输入图 A/B：mask-blurred 为 `29.744/55.893mm` mean/P95，直接 rectified
crop 为 `29.273/57.132mm`。原图 mean 略好、P95 `+1.24mm`；这证明实时模式可以只跟踪 bbox，
但是否接受略差的尾部仍应由使用场景决定。

### 运行

```bash
conda run --no-capture-output -n hamer python scripts/mobrecon_multiview_worker.py \
  --jobs video/sam3_hamer_left_index/hamer_jobs/hamer_jobs_000000_000442.jsonl \
  --output-dir video/sam3_hamer_left_index/mobrecon_per_view \
  --group-range 0-442 \
  --batch-size 8 \
  --job-batch-size 64 \
  --device cpu \
  --overwrite

conda run --no-capture-output -n hamer python scripts/fuse_hamer_palm_local.py \
  --predictions video/sam3_hamer_left_index/mobrecon_per_view/mobrecon_predictions_000000_000442.jsonl \
  --output-dir video/sam3_hamer_left_index/mobrecon_palm_local_fused \
  --group-range 0-442 \
  --overwrite
```

实时模式不生成 mask-blurred frame 时，在第一条命令额外加入 `--image-source rectified`。

## 稀疏 SAM3 实时路线

### 实现

新增 `scripts/track_sam3_sparse_keyframes.py`、`scripts/mobrecon_realtime_cpu.py` 和
`scripts/run_sparse_sam3_mobrecon.py`：

- SAM3 只处理每路相机每 10 帧中的 1 帧；准确性默认使用含左右手语义的 `bare` prompt，
  `realtime` 单 `hand` prompt 仅保留为显式速度档；
- 对 `intersection / min(area) >= 0.9` 的嵌套掩码做去重，但不合并空间上分离的两只手；
- 非 keyframe 在 `0.25x` 灰度图上使用 pyramidal LK 光流传播框；
- 离线兼容路径的 jobs 阶段使用 `--no-use-mediapipe --mask-frame-mode none --no-save-debug`；
- 吞吐默认只取 `C0,C2,C3`，减少 25% 图像负载；这是当前两段数据和 10 FPS gate 下的工程配置，不是跨样本最优相机集合的结论；
- 最多启动两个 SAM3 worker，每个进程在所属相机序列内只加载一次模型；
- MobRecon 默认放 CPU，避免和 SAM3 争抢 24GB 显存；
- 实时 CPU worker 把图像读取、光流、crop、MobRecon 和逐帧 palm-local fusion 合并到一个常驻进程，每张图只读一次；
- 未知手框不依赖未来帧：同时运行 Left/Right 两个 MobRecon 假设，再全局最小化同帧多视角掌局部
  形状误差；同一相机的两个独立候选不能落到同一侧，因此一只手和两只手都使用同一套逻辑；
- 已建立轨迹的左右手标签需要两个冲突语义 keyframe 才切换，避免单帧 prompt 误判扩散；
- detector 使用 line-buffered JSONL；默认 `streaming` 调度让双 SAM3 producer 与 CPU consumer 同时运行；
- SAM3/MobRecon 默认分别限制为 2/8 个 CPU thread，避免并行进程争抢 i9-12900K；
- palm-local 输出使用在线 One-Euro 滤波，逐帧更新且不读取未来帧；raw/filtered 两份关节同时保留；
- 最终 config 区分真实 `end_to_end_fps` 与复用 keyframe 时的 `reused_keyframe_pipeline_fps`。

### 跟踪框实测

left 0--99、四相机，以已有 dense SAM3 仅作为 keyframe 来源和离线参考：

| keyframe stride | 跟踪 FPS | 非 keyframe recall@IoU0.5 | 非 keyframe mean IoU | 全部框 P10 IoU |
| ---: | ---: | ---: | ---: | ---: |
| 5 | 32.99 | **0.9767** | **0.9570** | **0.9102** |
| 10 | 32.81 | 0.9543 | 0.9310 | 0.7802 |
| 15 | **33.02** | 0.9629 | 0.9284 | 0.7973 |
| 25 | 32.47 | 0.9515 | 0.8858 | 0.6163 |

选择 stride 10：它把 SAM3 调用降到 10%，非 keyframe recall 仍为 `95.43%`。真实单提示 left 0--99
的 stride 15 A/B 虽然 P95 从 `58.59mm` 略降到 `58.04mm`，但 mean 从 `30.40mm` 增到
`32.28mm`，因此没有为了更少检测牺牲平均精度。表中 FPS 只计算 CPU 跟踪，不含 SAM3 推理。

### 最终姿态精度

最终评估使用真实单提示 SAM3、stride 10、`C0,C2,C3`、在线 handedness 和完全因果 One-Euro；
统计全部匹配帧、左右手以及 thumb/index/middle 的 9 个关节点：

| 序列 | 配置 | mean | median | P95 | max | missing hands |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| left | 最终实时路线 | **32.675mm** | **29.079mm** | **60.780mm** | **93.506mm** | 0 |
| left | HaMeR baseline | 39.732mm | 35.253mm | 78.371mm | 129.867mm | - |
| right | 最终实时路线 | **34.955mm** | **29.811mm** | **72.136mm** | **111.500mm** | 0 |
| right | HaMeR baseline | 41.360mm | 36.161mm | 84.786mm | 127.323mm | - |

关闭滤波时，同一真实单提示输出的 left/right mean/P95/max 分别为
`34.610/66.501/108.940mm` 和 `37.867/80.836/159.130mm`。默认 One-Euro 在两段完整序列上同时改善
mean、median、P95 和 max；它只依赖当前帧和历史状态，不使用未来帧或手套 GT。每只手仍保留 raw 字段，
因此这不是不可逆的数据覆盖。

### 最终 GPU 验收

环境为 RTX 3090 24GB、i9-12900K；SAM3 FP16 autocast、两个 worker，MobRecon CPU FP32、batch 8、
单帧微批。以下均是最终代码的一体化完整序列实测，不是阶段速度相加或 GPU 估算：

| 序列 | frames | 总墙钟 | 冷启动端到端 FPS | 稳态 FPS | startup | P95 帧延迟 | 双 SAM3 peak reserved |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| left | 443 | 41.100s | **10.779** | **13.928** | 9.293s | 98.116ms | 11028MiB |
| right | 478 | 43.047s | **11.104** | **14.132** | 9.223s | 101.677ms | 11028MiB |

两段都没有 missing image、超时或 missing hand；各有 2 次光流失败，但下一关键帧刷新后恢复，最终仍输出
`886/956` 只手记录。每个 SAM3 worker 的 peak reserved 为 `5514MiB`，相加约 `10.77GiB`，只占
24GiB gate 的约 45%。

线程隔离是冷启动端到端过线的关键：未限制时 left 完整序列最好为 `9.95 FPS`；SAM3 限 2 threads、
MobRecon 改为 8 threads 后，SAM3 双进程加载约从 `10s` 降到 `6.3s`。单独复用真实 keyframe 的
8/12/16/20 threads A/B 为 `17.20/16.61/16.49/12.66 FPS`，因此 8 是当前 CPU 的实测默认值。

### 四相机、GPU MobRecon 与刷新频率复核

为检查三相机结论是否被样本选择影响，补跑了 left/right 全序列的真实 C1 SAM3 keyframe，随后使用
相同的 stride 10 光流、GPU FP32 MobRecon 和因果 One-Euro，对同一批 per-view prediction 枚举全部
15 个非空相机组合。代表性结果如下：

| 序列 | 相机 | mean | median | P95 | max |
| --- | --- | ---: | ---: | ---: | ---: |
| left | C0,C1,C2,C3 | **32.482mm** | 29.073mm | **59.114mm** | 95.137mm |
| left | C0,C2,C3 | 32.636mm | **29.005mm** | 61.136mm | **93.436mm** |
| left | C1,C2 | 33.957mm | 31.092mm | 61.298mm | 101.843mm |
| right | C0,C1,C2,C3 | 35.570mm | 30.246mm | 74.737mm | 120.967mm |
| right | C0,C2,C3 | **34.982mm** | **29.734mm** | **72.196mm** | **111.538mm** |
| right | C1,C2 | 37.527mm | 31.753mm | 80.692mm | 143.577mm |

结论有两层：

1. `C1,C2` 画面看起来更正、更完整，不等于 MobRecon 的掌局部 3D 误差更低；视角、遮挡、手套域偏移、
   crop 和标定误差都会影响最终指标；
2. C1 在 left 上有利于 mean/P95，在 right 上却明显有害，因此两段样本不足以把任何固定三相机集合
   宣布为跨动作、跨用户最优。当前 pooled 指标仍支持 `C0,C2,C3` 作为吞吐默认，但精度优先采集应保留
   四路 per-view 结果，并用更多动作/用户的验证集学习相机可靠度，而不是按视觉印象硬删相机。

GPU 调度也做了真实并发测试：

| 配置（left 0--99，四相机） | 稳态 FPS | 结果 |
| --- | ---: | --- |
| 2 SAM3 GPU + MobRecon CPU FP32 | **10.748** | 11028MiB SAM3 reserved，当前最稳的异构并行 |
| 2 SAM3 GPU + MobRecon GPU FP32 | 9.783 | MobRecon 单独虽快，但与 SAM3 抢计算后反而低于 10 FPS |
| 1 SAM3 GPU + MobRecon CPU FP32 | 11.061 | 约 5514MiB，调度简单 |
| 1 SAM3 GPU + MobRecon GPU FP32 | **11.505** | 约 5514+378MiB，常驻模型时延略低 |
| 4 个独立 SAM3 GPU 进程 | OOM | 总占用约 23.54GiB，第四进程再申请 82MiB 时失败 |

MobRecon 单独放 GPU 的确更快：left 0--99 四相机从 CPU `12.128 FPS` 提升到 FP32 GPU
`19.024 FPS`，峰值 reserved 仅 `378MiB`。但显存够不代表整链路更快；两个 SAM3 同时工作时，GPU
MobRecon 的 model time 因计算竞争明显增加。因此当前建议为：两个 SAM3 时保留 CPU MobRecon；若改成
一个常驻 SAM3 服务，可再选择 GPU FP32 MobRecon。FP16 MobRecon 的 P95 在该 A/B 中变差约
`1.47mm`，精度优先不启用。

刷新频率并非越高越好。left 0--99 四相机的真实 A/B 为：

| stride | keyframe images | 稳态 FPS | mean | P95 | max |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 40 | **10.748** | **32.681mm** | **58.497mm** | **93.314mm** |
| 8 | 52 | 10.352 | 32.901mm | 58.766mm | 94.511mm |

stride 8 增加了 30% SAM3 keyframe，却没有改善这段 3D 精度。当前 stride 10 的完整 left/right
各只有 2 次显式光流失败，但单向 LK 仍可能发生未被计数的静默漂移。代码保留了可选
`--max-forward-backward-error` 诊断：`1.5` 阈值在同输入框评估中只有极小收益，最终 3D 的 left/right
mean 反而分别增加 `0.027/0.038mm`，所以默认值为 `0`（关闭），不把离线框指标的小幅提升冒充姿态收益。

精度优先的下一步不是盲目把全局 stride 改成 5 或 8，而是把 SAM3 改为常驻请求服务：正常每 10 帧
刷新；当特征点数/比例、前后向误差、RANSAC 内点率、bbox 尺度跳变或多视角 3D 共识异常时，只对异常
相机提前刷新。当前文件式 producer 只能按固定 stride 预先生产 keyframe，尚不能在 consumer 发现异常后
反向请求，因此该自适应闭环仍是下一阶段工作，不能写成已经完成。

### 运行

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

checkpoint 路径可替换为本机实际 snapshot；若允许 Hugging Face 下载，可去掉 checkpoint 和
`--sam3-no-hf`。输出的
`realtime_config_*.json` 会记录每个阶段耗时、真实端到端 FPS、两个 SAM3 worker 的 peak reserved
及其总和。默认 `--execution-mode streaming`；如需旧的可调试中间 JSONL 链路，可显式使用
`--execution-mode sequential`。首次 GPU run 的 peak 只能在模型实际运行后获得，因此 24GiB gate
会拒绝超限结果，但不能在未知峰值的第一次运行前预知 OOM；后续应把已测峰值作为启动前配置约束。
精度优先复核可把 `--cameras` 改成 `C0,C1,C2,C3`；四相机 0--99 的双 SAM3 + CPU MobRecon
稳态实测为 left/right `10.748/10.618 FPS`，但短序列冷启动端到端只有约 `5.35 FPS`，正式实时服务应
预加载模型，不能把稳态数字当作冷启动验收值。

### 24GB 显存调度

1. 不启动“四相机四模型”，SAM3 worker 数硬限制为 1 或 2；
2. 三相机按 round-robin 分给两个 worker，一个处理 `C0,C3`，另一个处理 `C2`；四相机模式则各处理两路；
3. 一个 CPU MobRecon worker 跟随 keyframe shard，统一完成跟踪、crop、双假设和融合，把 GPU 留给 SAM3；
4. 两个 worker 的 `peak_cuda_reserved_mib` 相加后必须不超过用户指定的 `--vram-budget-gib`；
5. 当前双进程实测为 `5514 + 5514 = 11028MiB`；四进程虽按该静态数字推算小于 24GiB，但真实运行会 OOM，必须保留瞬时 workspace、CUDA context、桌面和驱动余量；
6. 替换模式不同时保留 HaMeR；若未来把 MobRecon 放 GPU，必须重新做整链路 peak gate。

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
conda run --no-capture-output -n headcam python scripts/run_hamer_multiview_pipeline.py \
  --base-dir video/sam3_hamer_left_index \
  --group-range 0-442 \
  --hamer-speed-profile quality \
  --overwrite
```

平衡吞吐与精度：

```bash
conda run --no-capture-output -n headcam python scripts/run_hamer_multiview_pipeline.py \
  --base-dir video/sam3_hamer_left_index \
  --group-range 0-442 \
  --hamer-speed-profile balanced \
  --overwrite
```

FP16 快速实验；正式使用前必须在目标 GPU 上和 `quality` 做同序列 A/B：

```bash
conda run --no-capture-output -n headcam python scripts/run_hamer_multiview_pipeline.py \
  --base-dir video/sam3_hamer_left_index \
  --group-range 0-442 \
  --hamer-speed-profile fast \
  --overwrite
```

保留三尺度但使用轻量 mask proxy、FP16 和可回退 backbone compile：

```bash
conda run --no-capture-output -n headcam python scripts/run_hamer_multiview_pipeline.py \
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

## GPU Gate 验收结果

原定 Gate 已全部完成：

1. left/right 完整序列的 `end_to_end_fps` 分别为 `10.779/11.104`，均不使用单阶段 FPS 代替；
2. `sam3_peak_reserved_sum_mib=11028`，低于 `24576MiB`；
3. CPU config 分别记录 `443/478` frames、`1329/1434` camera images、`2247/2413` predictions、
   `886/956` fused hands，没有 missing image、超时或静默丢帧；
4. 手套评估覆盖 `7974/8604` 个点、missing hand 均为 0，最终 mean/P95 都优于本文 HaMeR baseline；
5. 完整输出和验收配置位于
   `video/sam3_hamer_left_index/sam3_mobrecon_realtime_gpu_filtered_final/` 与
   `video/sam3_hamer_right_index/sam3_mobrecon_realtime_gpu_filtered_final/`；
6. 后续若追求更高质量，可继续做 WiLoR/Fast-HaMeR A/B；它们不再是当前 10 FPS 目标的阻塞项。

## 全部实验结果总表

这张表统一索引本文所有已完成的本地速度、显存与精度实验。精度默认按
`mean / P95 / max`（mm）记录；FPS 必须结合“是否含模型加载/SAM3”列理解，不能把单阶段吞吐
当成端到端结果。`—` 表示该实验没有报告此项。

| ID | 实验族 | 范围/配置 | 速度或资源结果 | L 精度 | R 精度 | 结论/状态 |
| --- | --- | --- | --- | ---: | ---: | --- |
| O01 | MobRecon CPU | 四相机完整序列，FP32，batch 8 | L 11.11 FPS；R 13.09 FPS；不含 SAM3 | 35.913 / 68.962 / 140.224 | 36.863 / 72.099 / 121.621 | 轻量 mesh 基线 |
| O02 | HaMeR baseline | 四相机完整序列 | — | 39.732 / 78.371 / 129.867 | 41.360 / 84.786 / 127.323 | mean/P95 弱于当前 MobRecon |
| O03 | MobRecon 输入图 | left 0--49，mask-blurred | — | 29.744 / 55.893 / — | — | 输入 A/B baseline |
| O04 | MobRecon 输入图 | left 0--49，rectified crop | 避免逐帧 mask-blurred 图 | **29.273** / 57.132 / — | — | mean 更好，P95 +1.24mm |
| O05 | LK 跟踪 | left 0--99，stride 5 | 32.99 tracking FPS；non-key recall 0.9767；mean IoU 0.9570 | — | — | 框精度最佳，SAM3 成本高 |
| O06 | LK 跟踪 | left 0--99，stride 10 | 32.81 tracking FPS；recall 0.9543；mean IoU 0.9310 | — | — | 当前全局默认 |
| O07 | LK 跟踪 | left 0--99，stride 15 | 33.02 tracking FPS；recall 0.9629；mean IoU 0.9284 | — | — | 框指标尚可，3D mean 退化 |
| O08 | LK 跟踪 | left 0--99，stride 25 | 32.47 tracking FPS；recall 0.9515；mean IoU 0.8858 | — | — | P10 IoU 降至 0.6163，不采用 |
| O09 | Stride 15 姿态 A/B | left 0--99，真实 SAM3 | SAM3 调用更少 | 32.28 / 58.04 / —（stride10 为 30.40 / 58.59 / —） | — | P95 微降但 mean 明显变差 |
| O10 | 最终三相机实时 | 完整序列，C0/C2/C3，One Euro | 含真实 SAM3/MobRecon | **32.675 / 60.780 / 93.506** | **34.955 / 72.136 / 111.500** | 已验收精度 |
| O11 | 最终三相机 raw | 同 O10，关闭 One Euro | — | 34.610 / 66.501 / 108.940 | 37.867 / 80.836 / 159.130 | 因果滤波四项均改善 |
| O12 | 最终端到端 | 完整 left，2 SAM3 + CPU MobRecon | **10.779 cold E2E / 13.928 steady FPS**；41.100s；11028MiB | 同 O10 | — | 通过 10 FPS/24GB gate |
| O13 | 最终端到端 | 完整 right，2 SAM3 + CPU MobRecon | **11.104 cold E2E / 14.132 steady FPS**；43.047s；11028MiB | — | 同 O10 | 通过 10 FPS/24GB gate |
| O14 | 四相机组合 | 全序列，C0 | — | 35.400 / 68.196 / 111.501 | 38.211 / 76.134 / 128.685 | 单视角 |
| O15 | 四相机组合 | 全序列，C1 | — | 35.042 / 66.663 / 110.505 | 39.420 / 87.023 / 155.509 | 视觉好不等于 3D 最准 |
| O16 | 四相机组合 | 全序列，C2 | — | 34.796 / 64.140 / 111.107 | 37.376 / 79.872 / 138.891 | 单视角 |
| O17 | 四相机组合 | 全序列，C3 | — | 33.943 / 66.517 / 102.166 | 36.112 / 73.134 / 110.118 | 单视角中较稳 |
| O18 | 四相机组合 | 全序列，C0/C1 | — | 33.597 / 60.912 / 103.485 | 37.035 / 77.417 / 127.777 | 双视角 |
| O19 | 四相机组合 | 全序列，C0/C2 | — | 33.449 / 60.941 / 94.268 | 35.877 / 73.994 / 116.833 | 双视角 |
| O20 | 四相机组合 | 全序列，C0/C3 | — | 33.616 / 64.773 / 101.788 | 36.379 / 72.351 / 128.617 | 双视角 |
| O21 | 四相机组合 | 全序列，C1/C2 | — | 33.957 / 61.298 / 101.843 | 37.527 / 80.692 / 143.577 | 两个“最佳视角”并非最佳指标 |
| O22 | 四相机组合 | 全序列，C1/C3 | — | 33.156 / 61.974 / 104.727 | 37.109 / 80.668 / 128.309 | 双视角 |
| O23 | 四相机组合 | 全序列，C2/C3 | — | 33.560 / 61.850 / 101.315 | 35.790 / 74.400 / 118.179 | 双视角 |
| O24 | 四相机组合 | 全序列，C0/C1/C2 | — | 33.202 / 59.489 / 95.557 | 36.411 / 76.297 / 127.912 | 三视角 |
| O25 | 四相机组合 | 全序列，C0/C1/C3 | — | 32.598 / 60.691 / 102.926 | 35.910 / 75.126 / 127.724 | 三视角 |
| O26 | 四相机组合 | 全序列，C0/C2/C3 | 已验收吞吐默认 | 32.636 / 61.136 / **93.436** | **34.982 / 72.196 / 111.538** | pooled/右序列最稳，但非普适最优 |
| O27 | 四相机组合 | 全序列，C1/C2/C3 | — | 32.846 / 59.664 / 97.932 | 36.164 / 77.017 / 128.369 | 三视角 |
| O28 | 四相机组合 | 全序列，C0/C1/C2/C3 | 四路 prediction 全保留 | **32.482 / 59.114** / 95.137 | 35.570 / 74.737 / 120.967 | left 最佳 mean/P95，right 弱于 C0/C2/C3 |
| O29 | 四相机实时 | 0--99，2 SAM3 + CPU MobRecon | L **10.748**；R **10.618 steady FPS**；11028MiB | 32.681 / 58.497 / 93.314（L） | 35.361 / 69.399 / 121.028（R online） | warm service 可达 10 FPS |
| O30 | GPU 竞争 | left 0--99，2 SAM3 + GPU MobRecon FP32 | **9.783 steady FPS**；MobRecon reserved 378MiB | 32.667 / 58.485 / 93.208 | — | 显存够但计算竞争，慢于 CPU MobRecon |
| O31 | 单 SAM3 调度 | left 0--99，1 SAM3 + CPU MobRecon | 11.061 steady FPS；5514MiB | — | — | 异构并行稳定 |
| O32 | 单 SAM3 调度 | left 0--99，1 SAM3 + GPU MobRecon FP32 | **11.505 steady FPS**；约 5514+378MiB | — | — | 常驻服务可选 |
| O33 | 四进程 SAM3 | left 0--99，4 independent workers | **OOM**；约 23.54GiB 已占用，额外 82MiB 失败 | — | — | 禁止按 4×静态 reserved 外推 |
| O34 | MobRecon device | left 0--99 四相机，CPU FP32 | 12.128 processing FPS；P95 latency 95.081ms | 32.667 / 58.485 / 93.208 | — | standalone baseline |
| O35 | MobRecon device | left 0--99 四相机，GPU FP32 | **19.024 FPS**；P95 46.579ms；378MiB | 32.667 / 58.485 / 93.208 | — | standalone 更快且等精度 |
| O36 | MobRecon precision | left 0--99 四相机，GPU FP16 | **20.525 FPS**；360MiB | 32.537 / 59.954 / 93.745 | — | P95 比 FP32 差约 1.47mm，不作精度默认 |
| O37 | Keyframe 频率 | left 0--99 四相机，stride 10 | **10.748 steady FPS**；40 keyframe images | **32.681 / 58.497 / 93.314** | — | 当前默认 |
| O38 | Keyframe 频率 | left 0--99 四相机，stride 8 | 10.352 steady FPS；52 keyframe images | 32.901 / 58.766 / 94.511 | — | 更慢且精度未改善 |
| O39 | 前后向 LK | 全序列，阈值 1.5 | tracking 略改善；GPU MobRecon L/R 约 22.9/23.2 FPS | 32.509 / 59.314 / 95.770（all4） | 35.608 / 75.237 / 121.238（all4） | 最终 3D 略退化，默认阈值 0 |
| O40 | 模型加载次数 | 完整序列，legacy→per-sequence | L 36→1；R 40→1；checkpoint 读取约 90/100GiB→2.5GiB | 不变 | 不变 | 默认 per-sequence |
| O41 | 跨 job packing | 8 jobs/24 candidates，CPU | 11.610→11.284s；8→6 batches | 最大 joints 差 1.19e-7m | — | 严格近等价，约 1.03x |
| O42 | Singleton mask scoring | right C0 0--49 | 79.987→46.349s；1.250→2.158 jobs/s | joints 差 0 | — | selection-only 默认可用 |
| O43 | 无 mask 尺度缩减 | left C0 0--3 | 24→8 samples；11.237→3.751s，**3.00x** | joints 差 0 | — | 无有效 mask 时严格等价 |
| O44 | Skeleton scorer | right 0--9，70 jobs | 202.792→123.078s；mask 81.116→0.803s | — | 60.523→60.705 mean | 1.65x，但候选一致率仅 50% |
| O45 | Skeleton scorer 扩展 | right 0--49，374 jobs | candidate 一致率 34.5% | — | mean 34.305→33.271；P95 89.425→90.202；max 120.617→122.754 | mean 好、尾部差，仅 aggressive |
| O46 | Balanced profile | right 0--477，单尺度 FP32 | 2108.728s；1.641 jobs/s | — | quality 35.496/77.334/130.169 → balanced 36.061/80.016/128.528 | P95 +2.682mm，非质量默认 |
| O47 | Prediction I/O | 完整序列，joint-only | L 176.3→11.5MiB；R 190.5→13.3MiB | 不变 | 不变 | JSON 减少 93.5%/93.0% |
| O48 | Overlay I/O | 完整序列，默认关闭 | 避免 L 930.2MiB；R 617.8MiB | 不变 | 不变 | debug 时显式开启 |
| O49 | SAM3 debug I/O | 完整序列，bbox/mask debug 默认关闭 | 避免 L 1.36GiB；R 1.45GiB；推理 mask 保留 | 不变 | 不变 | 不删除 inference masks |
| O50 | 单双手与左右手修复 | `video/7.17` 0--383，四相机，`bare`，包含阈值 0.9 | 6.080 cold E2E / 7.508 steady FPS；11 个嵌套候选被抑制 | — | 758 融合手→384/384 帧单 Right；无空帧 | 手数/身份回归通过；尚不是双手 GT 3D 精度实验 |

当前部署选择可直接从表中读取：严格沿用已经通过冷启动 10 FPS gate 的配置时用 O10--O13；
精度优先、模型常驻时保留 O28 的四路 per-view prediction，并在更大验证集上学习相机可靠度；
左右手与手数准确性优先时采用 O50 的 `bare` 默认；
不要采用 O33 的四 SAM3 进程，也不要因为 O35 的 standalone 速度就把双 SAM3 下的 MobRecon
强行迁到同一 GPU，O30 已证明这会降低整链路吞吐。
