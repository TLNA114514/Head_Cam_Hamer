# Head Cam HaMeR

Multi-view head-camera hand reconstruction scripts using MediaPipe, SAM3, HaMeR/MANO, and hand-local fusion.

The pipeline is intended to reuse SAM3/HaMeR code, environments, and checkpoints from:

```bash
/home/luojiangrui/ljr/wrist_cam
```

Quality-first zero-shot run:

```bash
/home/luojiangrui/miniconda3/envs/headcam/bin/python scripts/run_hamer_multiview_pipeline.py \
  --base-dir video/sam3_hamer_left_index \
  --group-range 0-442 \
  --chunk-size 50 \
  --hamer-speed-profile quality \
  --overwrite
```

The default posthoc pipeline loads the SAM3 image detector and HaMeR once per sequence,
packs HaMeR candidates across jobs, keeps FP32 mesh scoring, and drops redundant scales
only when no readable mask could distinguish them. Full SAM3 debug images, HaMeR overlays,
per-view vertices/MANO parameters, legacy primary-local fusion, and MANO refinement are opt-in.
`--frames` is inferred from the dataset named by `--base-dir` (for example,
`sam3_hamer_right_index` uses `cameras_right_index/frames.jsonl`) and remains explicitly overridable.

For a measured speed/accuracy trade-off, use:

```bash
/home/luojiangrui/miniconda3/envs/headcam/bin/python scripts/run_hamer_multiview_pipeline.py \
  --base-dir video/sam3_hamer_left_index \
  --group-range 0-442 \
  --hamer-speed-profile balanced \
  --overwrite
```

For the explicitly gated FP16 + skeleton-mask + compiled-backbone path, use
`--hamer-speed-profile aggressive`; see the speed document for its measured accuracy cost.

Outputs used for in-the-wild deployment are under `hamer_palm_local_fused/`. The raw
equal-view pose remains authoritative. Static calibration, offline smoothing, and the
default-computed causal One Euro result are stored separately.

Interactive visualization:

```bash
/home/luojiangrui/miniconda3/bin/conda run --no-capture-output -n headcam python -s scripts/view_hamer_multiview.py \
  --dataset left_index \
  --range 0-442 \
  --zero-shot
```

Documentation:

- Error and calibration experiments: `gloves/glove_local_calibration_experiments.md`
- Chinese calibration overview: `gloves/glove_local_calibration_experiments.zh.md`
- Chinese calibration technical appendix: `gloves/glove_local_calibration_experiments.technical.zh.md`
- Inference optimization overview: `docs/hamer_inference_optimization.md`
- Inference optimization technical appendix: `docs/hamer_inference_optimization.technical.md`

Data, generated images, model weights, and intermediate outputs are intentionally ignored by git.
