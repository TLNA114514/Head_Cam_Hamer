# Head-Camera Multi-View Hand Reconstruction

**Language / 语言**: English | [中文](README.zh.md)

This project reconstructs hands from synchronized, calibrated head-camera
images. It combines MediaPipe landmarks, SAM3 segmentation, HaMeR/MANO mesh
recovery, multi-view selection, and hand-local fusion in one reproducible
pipeline.

The repository is self-contained at the source level: compatible Wrist Cam,
HaMeR, SAM3, and HandMesh sources are pinned as Git submodules. A setup script
creates isolated environments and downloads all publicly distributable model
assets.

## Pipeline

```text
synchronized calibrated camera images
  -> rectification
  -> MediaPipe landmarks
  -> SAM3 hand masks and boxes
  -> HaMeR per-view MANO predictions
  -> temporal identity stabilization
  -> multi-view hand-local fusion
  -> JSONL predictions and optional debug assets
```

The default path produces a zero-shot fused result. Experimental MANO image
refinement, legacy fusion, glove calibration, and MobRecon tools remain
available as opt-in components.

## What it supports

- Any synchronized camera set represented by `frames.jsonl` and a compatible
  `cameras.yaml`; camera IDs are selected with `--cameras`.
- Bare-hand and gloved-hand SAM3 prompt presets.
- Framewise, post-hoc, and native SAM3 video tracking backends.
- Isolated `headcam`, `hamer`, and `sam3hand` Conda environments.
- Quality, balanced, FP16, and compiled inference profiles.
- Resumable intermediate outputs and bounded group/chunk processing.
- Optional per-view vertices, MANO parameters, rendered overlays, and debug
  masks.

## What it does not provide

- Camera calibration or synchronization from raw unsynchronized videos.
- Input recordings or private datasets.
- Automatic redistribution of the licensed MANO model.
- Guaranteed anatomical handedness from text prompts alone; multi-view and
  temporal constraints improve it but do not replace data validation.

## Installation

Clone only the main repository; `setup.sh` initializes missing submodules, so
`--recurse-submodules` is optional:

```bash
git clone https://github.com/TLNA114514/Head_Cam_Hamer.git head_cam
cd head_cam
./scripts/setup.sh
```

The installer:

1. Uses an existing Conda executable or installs a private Miniforge under
   `.tools/conda`.
2. Creates the `headcam`, `hamer`, and `sam3hand` environments.
3. Installs the pinned HaMeR, SAM3, and HandMesh source trees.
4. Downloads HaMeR, ViTPose, MobRecon, SAM3, and SAM3.1 checkpoints.
5. Verifies source paths, Python imports, and model files.

Automatic environment setup currently targets Linux x86_64. An NVIDIA driver
compatible with the installed PyTorch CUDA wheels is required for the default
SAM3 + HaMeR run.

### Required upstream access

SAM3 and SAM3.1 are gated Hugging Face repositories. Accept their access terms
and export a token before setup:

```bash
export HF_TOKEN=hf_...
```

Use a mirror when needed:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

MANO is licensed separately. Register at <https://mano.is.tue.mpg.de/>, obtain
`MANO_RIGHT.pkl`, and pass it to the installer:

```bash
HF_TOKEN=hf_... \
MANO_MODEL_PATH=/path/to/MANO_RIGHT.pkl \
./scripts/setup.sh
```

After these one-time access steps, installation is non-interactive. To install
source and environments without model files, use `./scripts/setup.sh
--skip-models`.

Verify an existing installation without changing it:

```bash
./scripts/setup.sh --check-only
```

## Input data

A dataset normally has this layout:

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

Each JSONL record describes one camera image in a synchronized group:

```json
{"group_id": 0, "camera_id": "C0", "timestamp_unix_ns": 1700000000000000000, "image_path": "C0/00000000.jpg", "width": 1600, "height": 1200}
```

At minimum, the pipeline uses `group_id`, `camera_id`, `image_path`, image
dimensions, and timestamps. All images sharing a `group_id` are treated as a
synchronized multi-view observation.

`cameras.yaml` must contain an entry for every selected camera, including its
intrinsics, distortion parameters, and `T_H_C` transform. Omni/Mei calibration
also requires `xi`. The convention used by this repository is:

```text
T_H_C maps a point from camera coordinates into the common headset frame H.
```

The input root, metadata file, calibration file, and output directory are
independent; none must use a repository-specific dataset name.

The default weak handedness prior (`C0:Left,C3:Right`) reflects the original
four-camera rig. For another camera topology, disable it with
`--camera-handedness-prior none` or provide a mapping that matches the physical
camera placement. Legacy primary-camera fusion has additional C0-C3 assumptions
and should likewise be configured before it is enabled.

## Quick start

Run the complete pipeline on any compatible dataset:

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

`--image-root` defaults to the directory containing `frames.jsonl`, and
`--calib` defaults to `cameras.yaml` in that directory. Therefore the compact
form is usually enough:

```bash
./scripts/run.sh \
  --frames /path/to/dataset/frames.jsonl \
  --base-dir outputs/my_run \
  --cameras C0,C1,C2,C3 \
  --group-range 0-999
```

For gloved hands:

```bash
./scripts/run.sh \
  --frames /path/to/dataset/frames.jsonl \
  --base-dir outputs/gloved_run \
  --cameras C0,C1,C2,C3 \
  --prompt-preset gloved \
  --group-range 0-999
```

Inspect the generated commands without running inference:

```bash
./scripts/run.sh \
  --frames /path/to/dataset/frames.jsonl \
  --base-dir outputs/test_run \
  --cameras C0,C1 \
  --group-range 0-9 \
  --dry-run
```

Use `--overwrite` only when existing artifacts for the selected range should be
replaced. Otherwise completed stages are reused where supported.

## Inference profiles

| Profile | Intended use | Main trade-off |
| --- | --- | --- |
| `quality` | final processing | FP32, multi-scale candidates, full mesh-mask scoring |
| `balanced` | routine iteration | one scale with FP32 mesh scoring |
| `fast` | supported-GPU experiments | FP16, one scale |
| `aggressive` | measured performance tests | FP16, compilation, lightweight mask scoring |

Start with `quality`. Validate `fast` and `aggressive` against the same sequence
on the target GPU before using their results.

Useful controls:

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

Run `./scripts/run.sh --help` for the full option set.

## Outputs

All generated artifacts are written below `--base-dir`. Important directories
include:

```text
rectified_for_hamer/          rectified image cache and calibration metadata
sam3_bboxes/                  masks, boxes, and optional debug images
sam3_tracks_stabilized/       temporally stabilized identities
hamer_jobs/                   per-camera HaMeR work items
hamer_per_view/               raw per-view predictions
hamer_palm_local_fused/       default zero-shot fused hand-local result
hamer_mano_multiview_refined/ optional image-space refinement result
```

The default deployment-oriented output is `hamer_palm_local_fused/`. Per-view
predictions remain available for auditing and alternative fusion experiments.

## Portable dependencies

Default source locations are relative to this repository:

```text
external/wrist_cam/third_party/hamer
external/wrist_cam/third_party/sam3
external/HandMesh
```

They can be overridden without changing code:

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

Explicit command-line paths take precedence where available.

## Updating pinned dependencies

The main repository records exact tested submodule commits. Upgrade them
deliberately and commit the resulting gitlink:

```bash
git -C external/wrist_cam fetch origin
git -C external/wrist_cam checkout <tested-commit>
git add external/wrist_cam
git commit -m "Update pinned wrist_cam dependency"
```

Use the same workflow for `external/HandMesh`. Avoid following an untested
moving branch in production runs.

## Troubleshooting

### A submodule directory is empty

```bash
git submodule sync --recursive
git submodule update --init --recursive
```

### SAM3 download is unauthorized

Accept access for both `facebook/sam3` and `facebook/sam3.1`, then export
`HF_TOKEN` or authenticate inside the SAM3 environment.

### Setup reports a missing MANO file

Rerun setup with `MANO_MODEL_PATH=/path/to/MANO_RIGHT.pkl`. The file cannot be
committed or downloaded automatically because of its license.

### A run uses the wrong dataset path

Pass `--frames`, `--image-root`, and `--calib` explicitly. `run.sh` changes to
the repository root before resolving relative paths, so relative paths are
always repository-relative.

### Check installation state

```bash
./scripts/setup.sh --check-only
```

## Additional documentation

- Inference optimization: `docs/hamer_inference_optimization.md`
- Inference optimization technical notes: `docs/hamer_inference_optimization.technical.md`
- Glove calibration experiments: `gloves/glove_local_calibration_experiments.md`
- Glove calibration overview (Chinese): `gloves/glove_local_calibration_experiments.zh.md`
- Glove calibration technical notes (Chinese): `gloves/glove_local_calibration_experiments.technical.zh.md`

Recordings, generated images, model weights, caches, local environments, and
intermediate outputs are intentionally excluded from Git.
