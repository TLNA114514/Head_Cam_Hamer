# 头戴相机多视角手部重建

**语言 / Language**: 中文 | [English](README.md)

本项目从同步、已标定的头戴相机图像中重建手部。完整流水线组合了 MediaPipe
关节点、SAM3 分割、HaMeR/MANO 网格恢复、多视角选择、时序身份稳定和手部局部坐标融合。

仓库在源码层面可以独立使用：经过验证的 Wrist Cam、HaMeR、SAM3 和 HandMesh
源码都通过 Git submodule 固定版本；安装脚本负责创建隔离环境并下载所有允许公开分发的模型文件。

## 数据流

```text
同步且已标定的多相机图像
  -> 图像去畸变与重映射
  -> MediaPipe 关节点
  -> SAM3 手部 mask 和 bbox
  -> HaMeR 单视角 MANO 预测
  -> 手部身份时序稳定
  -> 多视角手部局部坐标融合
  -> JSONL 结果和可选调试产物
```

默认路径输出零样本多视角融合结果。MANO 图像空间优化、旧版融合、手套标定和
MobRecon 工具作为可选实验组件保留。

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

图像根目录、元数据、标定文件和输出目录彼此独立，不需要使用仓库内预设的数据集名称。

默认的弱左右手先验 `C0:Left,C3:Right` 来自最初的四相机设备。使用其他相机布局时，
应通过 `--camera-handedness-prior none` 关闭，或改成符合真实相机位置的映射。旧版主相机
融合还包含额外的 C0-C3 假设，启用前也需要重新配置。

## 快速开始

对任意兼容数据集运行完整流水线：

```bash
./scripts/run.sh \
  --image-root /path/to/dataset \
  --frames /path/to/dataset/frames.jsonl \
  --calib /path/to/dataset/cameras.yaml \
  --base-dir outputs/my_run \
  --cameras C0,C1,C2,C3 \
  --group-range 0-999 \
  --hamer-speed-profile quality \
  --camera-handedness-prior none
```

`--image-root` 默认是 `frames.jsonl` 所在目录，`--calib` 默认是同目录下的
`cameras.yaml`，所以通常可以简化为：

```bash
./scripts/run.sh \
  --frames /path/to/dataset/frames.jsonl \
  --base-dir outputs/my_run \
  --cameras C0,C1,C2,C3 \
  --group-range 0-999
```

戴手套的数据可以选择对应 prompt 预设：

```bash
./scripts/run.sh \
  --frames /path/to/dataset/frames.jsonl \
  --base-dir outputs/gloved_run \
  --cameras C0,C1,C2,C3 \
  --prompt-preset gloved \
  --group-range 0-999
```

只查看即将执行的命令，不运行推理：

```bash
./scripts/run.sh \
  --frames /path/to/dataset/frames.jsonl \
  --base-dir outputs/test_run \
  --cameras C0,C1 \
  --group-range 0-9 \
  --dry-run
```

只有在确实需要替换已选范围内的现有结果时才加 `--overwrite`；否则支持复用的阶段会保留已有产物。

## 推理配置

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

完整参数可以运行 `./scripts/run.sh --help` 查看。

## 输出

所有运行产物都写入 `--base-dir`。主要目录包括：

```text
rectified_for_hamer/          去畸变图像缓存和标定元数据
sam3_bboxes/                  mask、bbox 和可选调试图
sam3_tracks_stabilized/       时序稳定后的手部身份
hamer_jobs/                   按相机生成的 HaMeR 任务
hamer_per_view/               原始单视角预测
hamer_palm_local_fused/       默认零样本手部局部坐标融合结果
hamer_mano_multiview_refined/ 可选图像空间优化结果
```

面向后续使用的默认结果位于 `hamer_palm_local_fused/`。单视角结果会继续保留，便于检查或尝试其他融合方法。

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
export SAM3_ENV=sam3hand
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
