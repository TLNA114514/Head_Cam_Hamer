# Head Cam HaMeR

Multi-view head-camera hand reconstruction scripts using MediaPipe, SAM3, HaMeR/MANO, and hand-local fusion.

The pipeline is intended to reuse SAM3/HaMeR code, environments, and checkpoints from:

```bash
/home/luojiangrui/ljr/wrist_cam
```

Typical segmented run:

```bash
/home/luojiangrui/miniconda3/bin/conda run --no-capture-output -n headcam python -s scripts/run_hamer_multiview_pipeline.py \
  --group-range 1-500 \
  --chunk-size 50 \
  --max-parallel-workers 2 \
  --overwrite
```

Interactive visualization:

```bash
/home/luojiangrui/miniconda3/bin/conda run --no-capture-output -n headcam python -s scripts/view_hamer_multiview.py \
  --group-range 1-500 \
  --space palm-local
```

Data, generated images, model weights, and intermediate outputs are intentionally ignored by git.
