# Head-Camera Multi-View Hand Reconstruction

**Language / 语言**: English | [中文](README.zh.md)

This project reconstructs hands from synchronized, calibrated head-camera
images. It provides two pipelines behind one switch: a HaMeR path for quality
and compatibility, and a low-latency path built from sparse SAM3, optical-flow
tracking, and MobRecon.

The repository is self-contained at the source level: compatible Wrist Cam,
HaMeR, SAM3, and HandMesh sources are pinned as Git submodules. A setup script
creates isolated environments and downloads all publicly distributable model
assets.

## Pipeline

```text
synchronized calibrated camera images
  -> rectification
  +-- HaMeR: MediaPipe + dense SAM3 -> HaMeR/MANO -> zero-shot multi-view fusion
  `-- MobRecon: sparse SAM3 keyframes -> optical flow -> MobRecon -> online multi-view fusion
  -> palm-local JSONL predictions and optional debug assets
```

`./scripts/run.sh` still selects HaMeR by default for backward compatibility;
use `--pipeline mobrecon` for the low-latency path. Both paths use the same
input format and rectification cache, but they are not parameter-identical
model replacements: MobRecon does not produce MANO parameters and cannot run
MANO-dependent refinement stages.

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

Rectification is selected automatically from the model fields in
`camera_defaults` (or a per-camera override):

| Calibration fields | Rectification backend | `xi` | Default focal scale |
| --- | --- | --- | ---: |
| `camera_model: omni`, `projection_model: mei`, `distortion_model: radtan` | OpenCV `omnidir` | required | `0.7` |
| `camera_model: pinhole`, `projection_model: pinhole`, `distortion_model: equidistant` | OpenCV `fisheye` | not used | `0.7` |

The second path is the format used by `video/bad_failure/cameras.yaml`; do not
add a synthetic `xi` or convert it to an Omni model. The global default for
`--rectify-focal-scale` is `0.7`; pass another positive value only for a
deliberate field-of-view experiment.

The input root, metadata file, calibration file, and output directory are
independent; none must use a repository-specific dataset name.

The default weak handedness prior (`C0:Left,C3:Right`) reflects the original
four-camera rig. For another camera topology, disable it with
`--camera-handedness-prior none` or provide a mapping that matches the physical
camera placement. Legacy primary-camera fusion has additional C0-C3 assumptions
and should likewise be configured before it is enabled.

## Quick start

### Select HaMeR or MobRecon

The unified entry point accepts `--pipeline hamer|mobrecon`; the default is
`hamer`. List the available pipelines with:

```bash
./scripts/run.sh --list-pipelines
```

HaMeR quality/compatibility path:

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

The MobRecon low-latency path prepares or reuses the
`rectified_for_hamer/` cache itself, so HaMeR does not need to run first:

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

These MobRecon defaults match the validated two-SAM3-GPU-producer plus CPU
MobRecon schedule. `C0,C2,C3` is a throughput configuration for the current
rig, not a universal camera subset; validate camera combinations again for a
different rig or a quality-first run. GPU FP32 MobRecon is worth testing with
one resident SAM3 worker, but measured GPU contention reduced total throughput
when two SAM3 workers ran concurrently.

MobRecon now defaults to accuracy-first handedness. The `bare` prompt set adds
semantic left/right evidence. Duplicate suppression removes only candidates
whose masks satisfy `intersection / min(area) >= 0.9`, so spatially separate
hands remain independent. An established track must receive two conflicting
semantic keyframes before its side changes. Use `--sam3-prompt-preset realtime`
explicitly for higher throughput when weaker handedness evidence is acceptable.

`--image-root` defaults to the directory containing `frames.jsonl`, and
`--calib` defaults to `cameras.yaml` in that directory. Therefore the compact
form is usually enough:

```bash
./scripts/run.sh \
  --frames /path/to/dataset/frames.jsonl \
  --base-dir outputs/my_run \
  --rectify-focal-scale 0.7 \
  --cameras C0,C1,C2,C3 \
  --group-range 0-999
```

### Run `video/bad_failure`

The new calibration is detected as Pinhole/Equidistant automatically. The
following full four-camera HaMeR quality example shows the `0.7` default
explicitly:

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

The corresponding MobRecon low-latency run is:

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

For gloved hands:

```bash
./scripts/run.sh \
  --frames /path/to/dataset/frames.jsonl \
  --base-dir outputs/gloved_run \
  --rectify-focal-scale 0.7 \
  --cameras C0,C1,C2,C3 \
  --prompt-preset gloved \
  --group-range 0-999
```

This `--prompt-preset` example belongs to the HaMeR path. MobRecon uses the
separate `--sam3-prompt-preset gloved` option; its default is now the
accuracy-first `bare` preset rather than a fixed `realtime` prompt.

Inspect the generated commands without running inference:

```bash
./scripts/run.sh \
  --frames /path/to/dataset/frames.jsonl \
  --base-dir outputs/test_run \
  --rectify-focal-scale 0.7 \
  --cameras C0,C1 \
  --group-range 0-9 \
  --dry-run
```

Use `--overwrite` only when existing artifacts for the selected range should be
replaced. Otherwise completed stages are reused where supported.

## HaMeR inference profiles

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

Show the option set for either pipeline with:

```bash
./scripts/run.sh --pipeline hamer --help
./scripts/run.sh --pipeline mobrecon --help
```

## Method switches and current availability

The methods described in the two technical reports do not all belong to one
switching layer:

- the pipeline switch selects HaMeR or MobRecon;
- the zero-shot output switch selects which computed field is copied to the
  primary `palm_local_joints_m` field;
- glove-supervised calibration first fits a profile from a synchronized glove
  clip, then pure-applies that profile to new results.

Grouped by method family, all 5 glove-local core families (static mapping,
ridge residual, local KNN, pose/velocity residual, and post-calibration
smoothing) have executable scripts. All 5 zero-shot primary outputs,
`raw|static-calibrated|smoothed|causal-smoothed|adaptive-causal`, are also wired
into the HaMeR pipeline. Runnable does not mean recommended as a default:

| Method | Entry point or switch | Status |
| --- | --- | --- |
| HaMeR execution optimization | `--hamer-speed-profile quality|balanced|fast|aggressive` | Directly usable; `quality` is the default |
| MobRecon realtime path | `--pipeline mobrecon` | Directly usable; CPU FP32 and stride 10 are the defaults |
| Zero-shot raw, bounded gap fill, causal, and offline smoothing | `--zero-shot-primary-output ...` | Raw observations stay untouched; gaps of at most 2 frames are filled and a five-frame Gaussian result is the default primary output |
| Physical-PnP plus 0.04m view gate | `--run-mano-multiview-image-refine` | Optional image-side MANO refinement; HaMeR only |
| Static glove similarity plus joint offsets | `calibrate_hamer_to_glove_local.py` | Conservative default when a synchronized glove calibration clip exists |
| Ridge/local-KNN plus OOD residual | `calibrate_pose_residual_local.py` | Directly usable; KNN requires dense target-pose coverage |
| Pose-plus-velocity residual | Same script with `--feature-mode all-joints-velocity` | Experimental; uses neighboring frames and is not an online causal output |
| Post-calibration Hampel/EMA | `smooth_local_hands.py` | Optional offline viewer output with very small measured gains |
| Image-2D beta / second-order temporal prior | `--image-beta-estimation-space image-2d` / `--image-temporal-acceleration-weight` | Implemented, but gains are insufficient; disabled by default |
| Camera SE(3), MediaPipe triangulation | Standalone diagnostic scripts | Diagnostics only, not pipeline switches |
| WiLoR, Fast-HaMeR, Hamba | No current local worker/profile | Not directly switchable in this repository yet |

### Switch the zero-shot primary output

After multi-view fusion, HaMeR uses a five-frame Gaussian result with
`radius=2` and `sigma=1.0` as `palm_local_joints_m` by default. The original
equal-weight result remains available in `raw_palm_local_joints_m`. This
offline default reads two frames before and after the current frame. A causal
deployment can select One-Euro instead:

```bash
./scripts/run.sh \
  --pipeline hamer \
  --frames /path/to/dataset/frames.jsonl \
  --base-dir outputs/hamer_causal \
  --zero-shot-primary-output adaptive-causal \
  --zero-shot-one-euro-min-cutoff 0.2 \
  --zero-shot-one-euro-beta 5.0
```

The default offline settings are equivalent to:

```text
--zero-shot-primary-output smoothed
--zero-shot-temporal-radius 2
--zero-shot-temporal-sigma 1.0
```

To restore a completely unsmoothed primary output:

```text
--zero-shot-primary-output raw
--zero-shot-temporal-radius 0
```

Fixed EMA uses `causal-smoothed` and a positive
`--zero-shot-causal-ema-alpha`. Here, `static-calibrated` means zero-shot bone
length normalization that **does not read glove ground truth**; it requires a
positive `--zero-shot-bone-calibration-blend`. It is not the glove-supervised
profile below.

Offline fusion also fills isolated missing hands of at most two frames by
default. A gap is filled only when the same handedness exists on both sides,
the endpoint joint displacement is at most `0.12 m`, and endpoint bone lengths
change by at most `20%`. Filled hands keep `metric_valid: false`, set
`temporal_interpolated: true`, and do not fabricate a raw observation. Control
or disable this behavior with:

```text
--zero-shot-temporal-interpolation-max-gap 2
--zero-shot-temporal-interpolation-max-joint-displacement-m 0.12
--zero-shot-temporal-interpolation-max-bone-relative-change 0.20
```

Set the maximum gap to `0` for strict raw-result reproduction.

### Fit and apply a glove calibration profile

When creating a glove-calibration base from scratch, explicitly add
`--run-mano-local-refine` to the HaMeR pipeline. Fit the recommended static
profile with:

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

Deployment no longer needs glove ground truth; pure-apply the stored profile:

```bash
conda run --no-capture-output -n headcam python scripts/calibrate_hamer_to_glove_local.py \
  --hamer outputs/new_run/hamer_mano_local_refined/mano_local_hands_RANGE.jsonl \
  --output outputs/new_run/glove_calibrated_RANGE.jsonl \
  --load-calibration-json outputs/calibration/static.profile.json \
  --space palm-local \
  --write-mode separate
```

`write-mode=separate` preserves `palm_local_joints_m` and writes
`glove_calibrated_palm_local_joints_m`. For a ridge/KNN residual, feed that
static output to `calibrate_pose_residual_local.py`; use `--calibration-json`
to save a residual profile and `--load-calibration-json` to apply it. The
conservative ridge settings are:

```text
--space glove-calibrated-palm-local
--regressor ridge
--ridge-alpha 10
--correction-shrink 0.75
--max-correction-m 0.03
--ood-gating knn-linear
--write-mode separate
```

Only switch to `local-knn`, `--knn-k 2`, `--knn-bandwidth-scale 0.5`, and
`--max-correction-m 0.06` when the calibration set densely covers the target
pose space. A HaMeR/MANO profile must not be applied directly to MobRecon
output; refit and independently validate a separate profile from MobRecon
results.

## Outputs

All generated artifacts are written below `--base-dir`. Important directories
include:

```text
rectified_for_hamer/          shared rectified cache (historical compatibility name)
sam3_bboxes/                  masks, boxes, and optional debug images
sam3_tracks_stabilized/       temporally stabilized identities
hamer_jobs/                   per-camera HaMeR work items
hamer_per_view/               raw per-view predictions
hamer_palm_local_fused/       default zero-shot fused hand-local result
hamer_mano_multiview_refined/ optional image-space refinement result
sam3_mobrecon_realtime/       default MobRecon output root
  sam3_keyframes/             sparse SAM3 detections
  realtime_cpu/               MobRecon per-view and online fused results
```

The default HaMeR deployment-oriented output is `hamer_palm_local_fused/`.
For MobRecon, `outputs.fused` in
`sam3_mobrecon_realtime/realtime_config_*.json` records the final path; raw and
One-Euro results are both preserved.

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
export MOBRECON_ENV=hamer
export SAM3_ENV=sam3hand
export HEADCAM_PIPELINE=hamer
```

For a single run, `--wrist-cam-root` is equivalent to temporarily setting
`WRIST_CAM_ROOT` and applies to both preflight checks and the selected pipeline:

```bash
./scripts/run.sh --wrist-cam-root /path/to/wrist_cam [pipeline options]
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
