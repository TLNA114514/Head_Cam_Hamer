# 头戴相机多视角手部重建

**语言 / Language**: 中文 | [English](README.md)

本项目从同步、已标定的头戴相机图像中重建手部，提供两条可从统一入口切换的流水线：
质量兼容的 HaMeR 路线，以及稀疏 SAM3、光流跟踪和 MobRecon 组成的低延迟路线。

仓库在源码层面可以独立使用：经过验证的 Wrist Cam、HaMeR、SAM3 和 HandMesh
源码都通过 Git submodule 固定版本；安装脚本负责创建隔离环境并下载所有允许公开分发的模型文件。

## 数据流

```text
同步且已标定的多相机图像
  -> 图像去畸变与重映射
  ├─ HaMeR：MediaPipe + dense SAM3 -> HaMeR/MANO -> zero-shot 多视角融合
  └─ MobRecon：sparse SAM3 keyframe -> 光流跟踪 -> MobRecon -> 在线多视角融合
  -> palm-local JSONL 结果和可选调试产物
```

`./scripts/run.sh` 默认仍选择 HaMeR，以兼容已有命令；通过 `--pipeline mobrecon`
切到低延迟路线。两条路线共享输入格式和去畸变缓存，但不是参数完全相同的模型替换：
MobRecon 路线不产生 MANO 参数，也不能运行依赖 MANO 的 refine。

## 支持范围

- 使用 `frames.jsonl` 和兼容 `cameras.yaml` 描述的任意同步相机组合；通过
  `--cameras` 选择相机 ID。
- 裸手和戴手套的 SAM3 prompt 预设。
- 逐帧、后处理关联和 SAM3 原生视频跟踪三种手部跟踪方式。
- 相互隔离的 `headcam`、`hamer`、`sam3hand` Conda 环境。
- 质量优先、均衡、FP16 和编译加速四种推理配置。
- 按 group 和 chunk 分段运行，可复用已经完成的中间结果。
- 可选导出单视角顶点、MANO 参数、渲染叠图和 mask 调试图。

## 不包含的能力

- 不负责从未同步的原始视频完成相机同步或标定。
- 不提供输入录像或私有数据集。
- 不自动再分发受许可证限制的 MANO 模型。
- 不保证仅凭文本 prompt 得到绝对正确的解剖学左右手；多视角和时序约束可以改善，
  但仍需要检查数据和结果。

## 安装

只需要克隆主仓库。即使没有使用 `--recurse-submodules`，`setup.sh` 也会自动初始化依赖：

```bash
git clone https://github.com/TLNA114514/Head_Cam_Hamer.git head_cam
cd head_cam
./scripts/setup.sh
```

安装脚本会：

1. 使用现有 Conda；如果系统没有 Conda，则安装到仓库内的 `.tools/conda`。
2. 创建 `headcam`、`hamer` 和 `sam3hand` 三个环境。
3. 安装固定版本的 HaMeR、SAM3 和 HandMesh 源码。
4. 下载 HaMeR、ViTPose、MobRecon、SAM3 和 SAM3.1 权重。
5. 检查源码路径、Python import 和必需模型文件。

自动环境安装目前面向 Linux x86_64。默认 SAM3 + HaMeR 流水线需要 NVIDIA
驱动能够兼容安装脚本选择的 PyTorch CUDA wheel。

### 必需的上游授权

SAM3 和 SAM3.1 是 Hugging Face 受限仓库。先接受访问条款并在安装前导出 token：

```bash
export HF_TOKEN=hf_...
```

需要镜像时可以设置：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

MANO 使用独立许可证。到 <https://mano.is.tue.mpg.de/> 注册并取得
`MANO_RIGHT.pkl`，然后交给安装脚本：

```bash
HF_TOKEN=hf_... \
MANO_MODEL_PATH=/path/to/MANO_RIGHT.pkl \
./scripts/setup.sh
```

完成这两项一次性授权后，其余安装过程不需要交互。只安装源码和环境、不下载模型时可运行：

```bash
./scripts/setup.sh --skip-models
```

检查现有安装而不修改环境：

```bash
./scripts/setup.sh --check-only
```

## 输入数据

推荐的数据集结构：

```text
dataset/
├── cameras.yaml
├── frames.jsonl
├── C0/
│   ├── 00000000.jpg
│   └── ...
├── C1/
│   └── ...
└── ...
```

`frames.jsonl` 每行描述同步 group 中的一张相机图像：

```json
{"group_id": 0, "camera_id": "C0", "timestamp_unix_ns": 1700000000000000000, "image_path": "C0/00000000.jpg", "width": 1600, "height": 1200}
```

流水线至少使用 `group_id`、`camera_id`、`image_path`、图像尺寸和时间戳。
拥有相同 `group_id` 的图像会被视为同一个同步多视角观测。

`cameras.yaml` 必须包含所有已选相机的标定项，包括内参、畸变参数和 `T_H_C`；
Omni/Mei 模型还需要 `xi`。本仓库使用的外参约定是：

```text
T_H_C 把相机坐标系中的点变换到公共头戴设备坐标系 H。
```

程序会根据 `camera_defaults`（或单相机覆盖项）里的模型字段自动选择去畸变路径：

| 标定字段 | 去畸变后端 | `xi` | 默认焦距缩放 |
| --- | --- | --- | ---: |
| `camera_model: omni`、`projection_model: mei`、`distortion_model: radtan` | OpenCV `omnidir` | 必须提供 | `0.7` |
| `camera_model: pinhole`、`projection_model: pinhole`、`distortion_model: equidistant` | OpenCV `fisheye` | 不使用 | `0.7` |

`video/bad_failure/cameras.yaml` 属于第二条路径；不需要补一个假的 `xi`，也不要把它改写成
Omni 模型。`--rectify-focal-scale` 的全局默认值为 `0.7`；只有在专门测试视场范围时，
才建议改成其他正数。

图像根目录、元数据、标定文件和输出目录彼此独立，不需要使用仓库内预设的数据集名称。

默认的弱左右手先验 `C0:Left,C3:Right` 来自最初的四相机设备。使用其他相机布局时，
应通过 `--camera-handedness-prior none` 关闭，或改成符合真实相机位置的映射。旧版主相机
融合还包含额外的 C0-C3 假设，启用前也需要重新配置。

## 快速开始

### 选择 HaMeR 或 MobRecon

统一入口接受 `--pipeline hamer|mobrecon`；默认值是 `hamer`。列出可用流水线：

```bash
./scripts/run.sh --list-pipelines
```

HaMeR 质量兼容路线：

```bash
./scripts/run.sh \
  --pipeline hamer \
  --image-root /path/to/dataset \
  --frames /path/to/dataset/frames.jsonl \
  --calib /path/to/dataset/cameras.yaml \
  --base-dir outputs/my_run \
  --rectify-focal-scale 0.7 \
  --cameras C0,C1,C2,C3 \
  --group-range 0-999 \
  --hamer-speed-profile quality \
  --camera-handedness-prior none
```

MobRecon 低延迟路线会自行准备或复用 `rectified_for_hamer/` 去畸变缓存，不需要先跑 HaMeR：

```bash
./scripts/run.sh \
  --pipeline mobrecon \
  --image-root /path/to/dataset \
  --frames /path/to/dataset/frames.jsonl \
  --calib /path/to/dataset/cameras.yaml \
  --base-dir outputs/my_realtime_run \
  --rectify-focal-scale 0.7 \
  --cameras C0,C2,C3 \
  --group-range 0-999 \
  --keyframe-stride 10 \
  --sam3-workers 2 \
  --sam3-prompt-preset bare \
  --sam3-duplicate-mask-containment 0.9 \
  --mobrecon-device cpu \
  --mobrecon-precision float32 \
  --mobrecon-torch-threads 8
```

这组 MobRecon 默认值对应当前已验证的“双 SAM3 GPU producer + CPU MobRecon”调度。
固定使用 `C0,C2,C3` 只是现有设备上的吞吐配置；换设备或精度优先时应重新验证相机组合。
一个 SAM3 worker 的常驻服务可测试 GPU FP32 MobRecon；两个 SAM3 worker 并行时，已有实测表明
GPU 计算竞争反而可能降低整条流水线吞吐。

MobRecon 当前以准确性为默认取向：`bare` prompt 会额外提供左右手语义；掩码去重只删除
`intersection / min(area) >= 0.9` 的近乎完全包含候选，因此不会把正常分开的两只手机械合并为一只。
已经建立的轨迹需要连续两个冲突 keyframe 才会切换左右手标签。若更看重速度，可显式改为
`--sam3-prompt-preset realtime`，但该单 prompt 模式缺少直接的左右手语义证据。

姿态输出同样采用准确性优先默认值：bbox 与 crop 的合并倍率约为 `1.32`，接近 MobRecon 的训练裁剪；
跨视角使用 `robust-medoid` 保留一套完整手型，不再逐关节强制平均。最终 `palm_local_joints_m` 默认指向
动作保真的 filtered：在父子骨向量空间做因果 One-Euro，单帧权重不低于 `0.4`，等效响应约 1--2 帧，
不缓存未来帧，并单独稳定骨长。raw 仍保存在 `raw_palm_local_joints_m`。当前默认 `beta` 为 `100.0`；
如需完全关闭平滑，
可显式添加 `--mobrecon-primary-output raw`。

`--image-root` 默认是 `frames.jsonl` 所在目录，`--calib` 默认是同目录下的
`cameras.yaml`，所以通常可以简化为：

```bash
./scripts/run.sh \
  --frames /path/to/dataset/frames.jsonl \
  --base-dir outputs/my_run \
  --rectify-focal-scale 0.7 \
  --cameras C0,C1,C2,C3 \
  --group-range 0-999
```

### 运行 `video/bad_failure`

新标定会被自动识别为 Pinhole/Equidistant。下面的四相机 HaMeR 质量路线会显式写出
默认值 `0.7`：

```bash
./scripts/run.sh \
  --pipeline hamer \
  --image-root video/bad_failure \
  --frames video/bad_failure/frames.jsonl \
  --calib video/bad_failure/cameras.yaml \
  --base-dir outputs/bad_failure_hamer \
  --rectify-focal-scale 0.7 \
  --cameras C0,C1,C2,C3 \
  --group-range 0-619 \
  --hamer-speed-profile quality
```

对应的 MobRecon 低延迟路线：

```bash
./scripts/run.sh \
  --pipeline mobrecon \
  --image-root video/bad_failure \
  --frames video/bad_failure/frames.jsonl \
  --calib video/bad_failure/cameras.yaml \
  --base-dir outputs/bad_failure_mobrecon \
  --rectify-focal-scale 0.7 \
  --cameras C0,C1,C2,C3 \
  --group-range 0-619 \
  --keyframe-stride 10 \
  --sam3-workers 2 \
  --sam3-prompt-preset bare \
  --sam3-duplicate-mask-containment 0.9 \
  --mobrecon-device cpu \
  --mobrecon-precision float32 \
  --mobrecon-torch-threads 8
```

戴手套的数据可以选择对应 prompt 预设：

```bash
./scripts/run.sh \
  --frames /path/to/dataset/frames.jsonl \
  --base-dir outputs/gloved_run \
  --rectify-focal-scale 0.7 \
  --cameras C0,C1,C2,C3 \
  --prompt-preset gloved \
  --group-range 0-999
```

该 `--prompt-preset` 示例属于 HaMeR 路线。MobRecon 使用独立参数
`--sam3-prompt-preset gloved`；其默认值为精度优先的 `bare`，不再内部固定为 `realtime`。

只查看即将执行的命令，不运行推理：

```bash
./scripts/run.sh \
  --frames /path/to/dataset/frames.jsonl \
  --base-dir outputs/test_run \
  --rectify-focal-scale 0.7 \
  --cameras C0,C1 \
  --group-range 0-9 \
  --dry-run
```

只有在确实需要替换已选范围内的现有结果时才加 `--overwrite`；否则支持复用的阶段会保留已有产物。

## HaMeR 推理配置

| 配置 | 用途 | 主要取舍 |
| --- | --- | --- |
| `quality` | 最终处理 | FP32、多尺度候选、完整 mesh-mask 评分 |
| `balanced` | 日常迭代 | 单尺度、FP32 mesh 评分 |
| `fast` | 目标 GPU 上的加速实验 | FP16、单尺度 |
| `aggressive` | 有对照实验的性能测试 | FP16、模型编译、轻量 mask 评分 |

建议从 `quality` 开始。正式使用 `fast` 或 `aggressive` 前，应在目标 GPU 上用同一段序列做结果对照。

常用参数：

```text
--chunk-size N
--group-range START-END
--group-ids 1,5,9
--max-mediapipe-workers N
--max-hamer-workers N
--hand-track-backend image|posthoc|sam3-native
--save-sam3-debug
--save-hamer-rendered-overlays
--hamer-export-vertices
--hamer-export-mano-params
```

查看对应流水线的完整参数：

```bash
./scripts/run.sh --pipeline hamer --help
./scripts/run.sh --pipeline mobrecon --help
```

## 方法开关与当前可用性

两份技术文档中的方法并不都属于同一层开关：

- pipeline 开关决定使用 HaMeR 还是 MobRecon；
- zero-shot 输出开关决定哪个已计算字段复制到主 `palm_local_joints_m`；
- glove-supervised calibration 是先用同步 glove 片段生成 profile，再 pure-apply 到新结果。

按方法族统计，glove-local 的 5 类核心方法（静态映射、ridge residual、local KNN、
pose/velocity residual、校准后 smoothing）已有 5/5 个可执行脚本；zero-shot 的
`raw|static-calibrated|smoothed|causal-smoothed|adaptive-causal` 也已全部接入 HaMeR pipeline。
可执行不等于适合作默认，当前状态如下：

| 方法 | 入口或开关 | 状态 |
| --- | --- | --- |
| HaMeR 执行优化 | `--hamer-speed-profile quality|balanced|fast|aggressive` | 可直接用；默认 `quality` |
| MobRecon 实时路线 | `--pipeline mobrecon` | 可直接用；默认 CPU FP32、stride 10 |
| zero-shot raw / 短缺口补帧 / 因果 / 离线平滑 | `--zero-shot-primary-output ...` | 真实 raw 不改写；默认补齐不超过 2 帧的缺口，并用 5 帧 Gaussian 结果作为主输出 |
| physical-PnP + 0.04m view gate | `--run-mano-multiview-image-refine` | 可选图像侧 MANO refine；仅 HaMeR |
| 静态 glove similarity + joint offsets | `calibrate_hamer_to_glove_local.py` | 有同步 glove 校准片段时的保守默认 |
| ridge / local-KNN + OOD residual | `calibrate_pose_residual_local.py` | 可直接用；KNN 只用于 pose 密集覆盖 |
| pose+velocity residual | 同上，`--feature-mode all-joints-velocity` | 实验性，含相邻帧，不是在线因果输出 |
| 校准后 Hampel/EMA | `smooth_local_hands.py` | 离线 viewer 可选，收益很小 |
| image-2D beta / 二阶时间 prior | `--image-beta-estimation-space image-2d` / `--image-temporal-acceleration-weight` | 已实现但收益不足，默认关闭 |
| camera SE(3)、MediaPipe triangulation | 独立诊断脚本 | 仅诊断，不作为 pipeline switch |
| WiLoR、Fast-HaMeR、Hamba | 无当前本地 worker/profile | 尚不能从本仓库直接切换 |

### 切换 zero-shot 主输出

HaMeR 默认在多视角融合后使用 `radius=2`、`sigma=1.0` 的 5 帧 Gaussian 平滑结果作为
`palm_local_joints_m`，原始等权多视角结果仍完整保存在 `raw_palm_local_joints_m`。这一默认值
适用于离线视频处理，会读取当前帧前后各 2 帧。实时或因果部署可切换到 One-Euro：

```bash
./scripts/run.sh \
  --pipeline hamer \
  --frames /path/to/dataset/frames.jsonl \
  --base-dir outputs/hamer_causal \
  --zero-shot-primary-output adaptive-causal \
  --zero-shot-one-euro-min-cutoff 0.2 \
  --zero-shot-one-euro-beta 5.0
```

默认离线参数等价于：

```text
--zero-shot-primary-output smoothed
--zero-shot-temporal-radius 2
--zero-shot-temporal-sigma 1.0
```

如果需要完全恢复未经时序平滑的主输出：

```text
--zero-shot-primary-output raw
--zero-shot-temporal-radius 0
```

固定 EMA 使用 `causal-smoothed` 和正的 `--zero-shot-causal-ema-alpha`。这里的
`static-calibrated` 只是**不读取 glove GT** 的 zero-shot 骨长归一化，需要同时设置正的
`--zero-shot-bone-calibration-blend`；它不是下面的 glove-supervised profile。

离线融合默认还会补齐最多连续缺失两帧的同一只手。只有缺口前后 handedness 一致、端点关节
最大位移不超过 `0.12 m`、端点骨长相对变化不超过 `20%` 时才会补帧。补出的手会保持
`metric_valid: false`，写入 `temporal_interpolated: true`，并且不会伪造 raw observation。
可通过以下参数调整或关闭：

```text
--zero-shot-temporal-interpolation-max-gap 2
--zero-shot-temporal-interpolation-max-joint-displacement-m 0.12
--zero-shot-temporal-interpolation-max-bone-relative-change 0.20
```

如需严格复现未补帧的 raw 结果，把最大缺口设为 `0`。

### 生成并应用 glove calibration profile

从头生成 glove calibration 底座时，HaMeR pipeline 需要显式加入
`--run-mano-local-refine`。推荐静态 profile 的拟合命令是：

```bash
conda run --no-capture-output -n headcam python scripts/calibrate_hamer_to_glove_local.py \
  --hamer outputs/hamer_run/hamer_mano_local_refined/mano_local_hands_RANGE.jsonl \
  --glove /path/to/synced_glove_local.jsonl \
  --output outputs/calibration/static_calibrated_RANGE.jsonl \
  --calibration-json outputs/calibration/static.profile.json \
  --train-group-range 0-199 \
  --space palm-local \
  --allow-translation \
  --joint-offsets mean \
  --joint-offset-shrink-k 25 \
  --max-joint-offset-m 0.025 \
  --bone-scales none \
  --write-mode separate
```

部署时不再需要 glove GT，直接 pure-apply 已保存的 profile：

```bash
conda run --no-capture-output -n headcam python scripts/calibrate_hamer_to_glove_local.py \
  --hamer outputs/new_run/hamer_mano_local_refined/mano_local_hands_RANGE.jsonl \
  --output outputs/new_run/glove_calibrated_RANGE.jsonl \
  --load-calibration-json outputs/calibration/static.profile.json \
  --space palm-local \
  --write-mode separate
```

`write-mode=separate` 会保留原始 `palm_local_joints_m`，并写入
`glove_calibrated_palm_local_joints_m`。如需 ridge/KNN residual，把上述静态输出作为
`calibrate_pose_residual_local.py` 的输入；使用 `--calibration-json` 生成 residual profile，
部署时用 `--load-calibration-json` 应用。推荐的保守 ridge 参数为：

```text
--space glove-calibrated-palm-local
--regressor ridge
--ridge-alpha 10
--correction-shrink 0.75
--max-correction-m 0.03
--ood-gating knn-linear
--write-mode separate
```

只有校准集密集覆盖目标 pose 时，才切到 `local-knn`、`--knn-k 2`、
`--knn-bandwidth-scale 0.5` 和 `--max-correction-m 0.06`。HaMeR/MANO profile
不能直接应用到 MobRecon 输出；如果要校准 MobRecon，必须用 MobRecon 输出重新拟合并独立验证 profile。

## 输出

所有运行产物都写入 `--base-dir`。主要目录包括：

```text
rectified_for_hamer/          两条路线共享的去畸变缓存（目录名为历史兼容名）
sam3_bboxes/                  mask、bbox 和可选调试图
sam3_tracks_stabilized/       时序稳定后的手部身份
hamer_jobs/                   按相机生成的 HaMeR 任务
hamer_per_view/               原始单视角预测
hamer_palm_local_fused/       默认零样本手部局部坐标融合结果
hamer_mano_multiview_refined/ 可选图像空间优化结果
sam3_mobrecon_realtime/       MobRecon 默认输出根目录
  sam3_keyframes/             稀疏 SAM3 检测
  realtime_cpu/               MobRecon per-view 与在线融合结果
```

HaMeR 面向后续使用的默认结果位于 `hamer_palm_local_fused/`。MobRecon 的最终路径记录在
`sam3_mobrecon_realtime/realtime_config_*.json` 的 `outputs.fused`；raw 与 One-Euro 结果会同时保留。

## 可移植依赖路径

默认源码路径都相对于本仓库：

```text
external/wrist_cam/third_party/hamer
external/wrist_cam/third_party/sam3
external/HandMesh
```

无需修改代码即可覆盖：

```bash
export WRIST_CAM_ROOT=/path/to/wrist_cam
export HAMER_ROOT=/path/to/hamer
export SAM3_ROOT=/path/to/sam3
export MOBRECON_ROOT=/path/to/HandMesh
export CONDA_BIN=/path/to/conda
export HEADCAM_ENV=headcam
export HAMER_ENV=hamer
export MOBRECON_ENV=hamer
export SAM3_ENV=sam3hand
export HEADCAM_PIPELINE=hamer
```

单次运行也可以使用 `--wrist-cam-root`。它等价于临时设置 `WRIST_CAM_ROOT`，并会同时作用于
启动前检查和实际流水线：

```bash
./scripts/run.sh --wrist-cam-root /path/to/wrist_cam [其他流水线参数]
```

支持显式路径参数的命令会优先使用命令行值。

## 更新固定依赖

主仓库记录的是经过测试的精确 submodule commit。更新时应显式选择版本并提交 gitlink：

```bash
git -C external/wrist_cam fetch origin
git -C external/wrist_cam checkout <tested-commit>
git add external/wrist_cam
git commit -m "Update pinned wrist_cam dependency"
```

`external/HandMesh` 使用相同流程。正式运行不建议跟随未经测试的移动分支。

## 常见问题

### submodule 目录为空

```bash
git submodule sync --recursive
git submodule update --init --recursive
```

### SAM3 下载提示没有权限

确认已经接受 `facebook/sam3` 和 `facebook/sam3.1` 的访问条款，然后导出
`HF_TOKEN`，或者在 SAM3 环境内登录。

### 安装提示缺少 MANO

使用 `MANO_MODEL_PATH=/path/to/MANO_RIGHT.pkl` 重新运行安装。许可证不允许仓库自动提交或下载这个文件。

### 运行时使用了错误的数据路径

显式传入 `--frames`、`--image-root` 和 `--calib`。`run.sh` 会先切换到仓库根目录，
所以所有相对路径始终以仓库根目录为基准。

### 检查安装状态

```bash
./scripts/setup.sh --check-only
```

## 其他文档

- 推理优化说明：`docs/hamer_inference_optimization.md`
- 推理优化技术记录：`docs/hamer_inference_optimization.technical.md`
- 手套标定实验：`gloves/glove_local_calibration_experiments.md`
- 手套标定中文概览：`gloves/glove_local_calibration_experiments.zh.md`
- 手套标定中文技术记录：`gloves/glove_local_calibration_experiments.technical.zh.md`

录像、生成图片、模型权重、缓存、本地环境和中间产物均默认排除在 Git 之外。
