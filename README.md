# Head Cam HaMeR

Multi-view head-camera hand reconstruction with MediaPipe, SAM3, HaMeR/MANO,
and hand-local fusion.

The repository pins its working Wrist Cam source snapshot as a Git submodule.
HaMeR and SAM3 are resolved relative to this repository, so no machine-specific
checkout path is required.

## Fresh-machine setup

Clone the main repository. Either cloning style is supported:

```bash
git clone --recurse-submodules https://github.com/TLNA114514/Head_Cam_Hamer.git head_cam
cd head_cam
./scripts/setup.sh
```

or:

```bash
git clone https://github.com/TLNA114514/Head_Cam_Hamer.git head_cam
cd head_cam
./scripts/setup.sh
```

`setup.sh` initializes a missing submodule itself. It then:

1. Uses an existing Conda installation, or installs a private Miniforge under
   `.tools/conda`.
2. Creates the `headcam`, `hamer`, and `sam3hand` environments.
3. Installs the pinned HaMeR and SAM3 source from `external/wrist_cam`, plus
   the optional MobRecon/HandMesh source from its own pinned submodule.
4. Downloads the public HaMeR, ViTPose, MobRecon, SAM3, and SAM3.1 checkpoints.
5. Verifies source paths, Python imports, and required model files.

The setup requires Linux, internet access, and enough disk space for the Conda
environments and model caches. NVIDIA driver/CUDA compatibility is still a host
requirement; PyTorch CUDA wheels are installed automatically.

### Two upstream access requirements

SAM3 and SAM3.1 are gated Hugging Face repositories. Accept their access terms,
then either export a token before setup:

```bash
export HF_TOKEN=hf_...
./scripts/setup.sh
```

or authenticate the environment and rerun setup:

```bash
.tools/conda/bin/conda run -n sam3hand hf auth login
./scripts/setup.sh
```

MANO is licensed separately and cannot be redistributed by this repository.
Register at <https://mano.is.tue.mpg.de/>, download `MANO_RIGHT.pkl`, and provide
it to the same setup command:

```bash
HF_TOKEN=hf_... \
MANO_MODEL_PATH=/path/to/MANO_RIGHT.pkl \
./scripts/setup.sh
```

After those one-time access grants, setup and all public downloads are automatic.
To install only source and environments, use `./scripts/setup.sh --skip-models`.

## One-command run

Place the camera dataset under `video/` (which is intentionally not tracked),
then launch the complete pipeline through the wrapper:

```bash
./scripts/run.sh \
  --base-dir video/sam3_hamer_left_index \
  --group-range 0-442 \
  --chunk-size 50 \
  --hamer-speed-profile quality \
  --overwrite
```

`run.sh` first runs the dependency doctor, then starts the main pipeline in the
`headcam` environment. The main pipeline dispatches SAM3 and HaMeR work to their
own isolated environments.

Check an existing installation without changing it:

```bash
./scripts/setup.sh --check-only
```

For a measured speed/accuracy trade-off, use `--hamer-speed-profile balanced`.
The explicitly gated FP16 + skeleton-mask + compiled-backbone path is available
as `--hamer-speed-profile aggressive`.

Outputs used for in-the-wild deployment are under `hamer_palm_local_fused/`.
Raw equal-view pose remains authoritative. Static calibration, offline
smoothing, and the default causal One Euro result are stored separately.

## Portable path and environment overrides

Defaults require no configuration:

```text
external/wrist_cam/third_party/hamer
external/wrist_cam/third_party/sam3
external/HandMesh
```

Advanced installations can reuse other checkouts or environment names:

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

Explicit command-line options such as `--hamer-root` and `--sam3-root` take
precedence where available.

## Updating the pinned Wrist Cam dependency

The main repository records an exact Wrist Cam commit. Upgrade it deliberately:

```bash
git -C external/wrist_cam fetch origin
git -C external/wrist_cam checkout <tested-commit>
git add external/wrist_cam
git commit -m "Update pinned wrist_cam dependency"
```

Do not point the submodule at an untested moving branch: reproducible source and
model behavior depends on the pinned commit.

## Documentation

- Error and calibration experiments: `gloves/glove_local_calibration_experiments.md`
- Chinese calibration overview: `gloves/glove_local_calibration_experiments.zh.md`
- Chinese calibration technical appendix: `gloves/glove_local_calibration_experiments.technical.zh.md`
- Inference optimization overview: `docs/hamer_inference_optimization.md`
- Inference optimization technical appendix: `docs/hamer_inference_optimization.technical.md`

Data, generated images, model weights, caches, local Conda files, and intermediate
outputs are intentionally ignored by Git.
