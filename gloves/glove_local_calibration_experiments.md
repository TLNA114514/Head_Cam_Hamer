# Glove-Local Calibration Experiments

Chinese beginner overview: `gloves/glove_local_calibration_experiments.zh.md`.
Chinese technical appendix: `gloves/glove_local_calibration_experiments.technical.zh.md`.
Inference speed, latency, and replacement-model research is kept separately in
`docs/hamer_inference_optimization.md`.

## Plain-Language Summary

The main thing this document shows is simple:
**HaMeR/MANO local hand coordinates and PN-glove local coordinates do not use
exactly the same local frame or joint definition.** That mismatch is stable and
large enough that it dominates ordinary frame-to-frame jitter.

So the best current glove-supervised result is not "HaMeR became magically more
accurate without ground truth." It is a device/local-coordinate calibration
layer trained with glove GT. The raw MANO output is still preserved as
`palm_local_joints_m`; the calibrated output is written separately as
`glove_calibrated_palm_local_joints_m`.

### How To Read The Numbers

- `Mean`: average error. Good for overall quality.
- `P95`: 95% of points are below this error. Good for checking bad frames.
- `Max`: worst point. Useful for finding failures, but one bad frame can
  dominate it.
- Lower is better for every metric.

### What To Use Now

- For in-the-wild / zero-shot palm-local joints, use
  `hamer_palm_local_fused/`. Its default primary field is the unmodified
  cross-view mean; optional temporal results are stored in separate fields.
- For glove-supervised local coordinates, use `hamer_mano_local_refined/` plus
  the static calibration layer only when glove calibration is explicitly part
  of the task. This is not the deployment default.
- For dense in-distribution analysis where the calibration motions are well
  covered, `dense KNN + OOD guard` gives the lowest numbers.
- For no-GT image-side visualization/debugging, use
  `hamer_mano_multiview_selected/` when a MANO mesh is required; use the new
  zero-shot output for the strongest skeleton-only palm-local result.
- Do not use `hamer_mano_multiview_selected/` as the default base for
  glove-supervised calibration; it is better for image-side viewing, but worse
  after glove calibration.

### Recommended Static Calibration

- similarity transform with scale, rotation, and translation;
- per-joint residual offsets;
- `joint_offset_shrink_k=25`;
- `max_joint_offset_m=0.025`;
- no bone-scale fitting;
- `write-mode=separate`.

### Main Result At A Glance

The chart shows mean error in millimeters. It is plain Markdown text, so it
survives Feishu/Lark import even when SVG/HTML is stripped. Lower is better.

| Output | Mean mm | Text bar |
| --- | ---: | --- |
| left_index raw | 30.12 | ███████████████▌ |
| left_index static | 14.08 | ███████▎ |
| left_index KNN+OOD | 4.09 | ██ |
| right_index raw | 38.83 | ████████████████████ |
| right_index static | 20.56 | ██████████▌ |
| right_index KNN+OOD | 5.87 | ███ |

| Sequence | Raw MANO local mean / P95 | Recommended static calibration mean / P95 | Dense KNN+OOD mean / P95 |
| --- | ---: | ---: | ---: |
| left_index | 30.12 / 59.70 mm | 14.08 / 31.52 mm | 4.09 / 10.06 mm |
| right_index | 38.83 / 89.36 mm | 20.56 / 62.11 mm | 5.87 / 17.64 mm |

Important caveat: dense KNN only works this well when the target motion space is
densely covered by calibrated examples. When coverage is uncertain, prefer the
static calibration or a low-capacity pose residual with an OOD guard.

### No-GT Image-Side Result

For image-side output without glove GT, the current best default is
`physical-pnp` initialization plus a conservative `0.04m` PnP view gate. It does
not rewrite camera calibration. It only rejects a view when that view's physical
K PnP pose disagrees with the current anchor.

| Output | Mean mm | Text bar |
| --- | ---: | --- |
| left baseline | 36.71 | ████████████████▌ |
| left PnP-gated | 34.70 | ███████████████▋ |
| right baseline | 44.41 | ████████████████████ |
| right PnP-gated | 42.02 | ███████████████████ |

| Image-side output | left_index 0--442 mean / P95 | right_index 0--477 mean / P95 |
| --- | ---: | ---: |
| baseline image refine | 36.71 / 67.14 mm | 44.41 / 99.42 mm |
| PnP-gated selected current default | 34.70 / 63.72 mm | 42.02 / 93.20 mm |

Current candidate selection rule: **if a complete PnP-gated candidate exists,
prefer it; otherwise fall back to baseline.**

### Strict Chronological Check

The safest deployment-like test is first-half training and later-half
evaluation. Under that split, pose+velocity residual is the strongest variant
so far in mean and P95, although its left-index worst case is worse.

| Output | Mean mm | Text bar |
| --- | ---: | --- |
| left static | 15.15 | ██████████████▋ |
| left pose+velocity | 12.52 | ████████████▏ |
| right static | 20.60 | ████████████████████ |
| right pose+velocity | 17.84 | █████████████████▎ |

### Tested But Not Default

- Image-space beta 2D refinement: the left 0--49 shard changed only slightly,
  and P95 got a little worse, so it is not worth the default runtime cost.
- Second-order temporal acceleration: directionally positive, but too small;
  the right-hand max also got slightly worse.
- Global camera SE(3) correction: unstable on held-out tests, so it remains a
  diagnostic tool only.
- Early SAM3 boundary loss: did not improve the result. The next useful step is
  a two-sided silhouette constraint or differentiable renderer, not more tuning
  of the old one-sided mask term.

## Detailed Experiment Log

This note records the current best supervised calibration layer from HaMeR/MANO
hand-local output to PN glove local coordinates. The calibration is trained on
even `group_id` frames and evaluated on odd `group_id` frames.

Important caveat: this is a glove-supervised device/local-coordinate calibration
layer. It should not be interpreted as a no-GT HaMeR/MANO accuracy improvement.
The original MANO `palm_local_joints_m` is preserved; calibrated coordinates are
written to `glove_calibrated_palm_local_joints_m`.

## Why This Is Needed

The strong gain from allowing translation, and the loss when wrist is re-centered
after calibration, indicates a stable local-coordinate / joint-definition mismatch
between MANO/HaMeR and the glove local frame. This is larger than frame-to-frame
jitter.

## Odd-Holdout Results

### Right Index Sequence

`gloves/glove_local/pn3_rightindex_camera_sync_g356p000_c17p000_cut_000000_000477.jsonl`

| Method | Mean mm | Median mm | RMSE mm | P95 mm | Max mm |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original MANO local | 41.03 | 35.11 | 46.83 | 91.43 | 130.46 |
| Similarity, translation allowed | 28.24 | 22.87 | 34.05 | 67.38 | 116.87 |
| Similarity + joint offsets, k=200, max=30mm | 22.57 | 16.23 | 29.53 | 64.73 | 112.49 |
| Similarity + joint offsets, k=25, max=25mm | 20.51 | 13.40 | 28.39 | 63.96 | 109.86 |
| Similarity + joint offsets, k=50, max=30mm | 20.68 | 13.71 | 28.42 | 64.02 | 110.41 |
| Similarity + wrist re-centered | 43.71 | 38.99 | 47.93 | 80.01 | 128.76 |

### Left Index Sequence

`gloves/glove_local/pn3_leftindex_camera_sync_g414p000_c47p000_cut_000000_000442.jsonl`

| Method | Mean mm | Median mm | RMSE mm | P95 mm | Max mm |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original MANO local | 33.75 | 31.70 | 36.85 | 64.39 | 139.68 |
| Similarity, translation allowed | 22.23 | 21.29 | 24.73 | 39.98 | 105.32 |
| Similarity + joint offsets, k=200, max=30mm | 16.74 | 14.85 | 19.47 | 31.87 | 103.74 |
| Similarity + joint offsets, k=25, max=25mm | 14.48 | 12.18 | 17.75 | 30.99 | 102.80 |
| Similarity + joint offsets, k=50, max=30mm | 14.80 | 12.56 | 17.94 | 30.97 | 102.99 |
| Similarity + wrist re-centered | 29.23 | 27.59 | 31.40 | 48.37 | 109.60 |

## All-Finger Odd-Holdout Results

The tables above follow the earlier right/left index reports and evaluate
`thumb,index,middle`. The best calibration was also evaluated on all five
fingers.

| Sequence | Method | Mean mm | Median mm | RMSE mm | P95 mm | Max mm |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| left_index | Original MANO local | 30.12 | 27.45 | 33.61 | 59.70 | 139.68 |
| left_index | Best calibrated | 14.08 | 11.86 | 17.46 | 31.52 | 102.80 |
| right_index | Original MANO local | 38.83 | 32.56 | 45.36 | 89.36 | 130.46 |
| right_index | Best calibrated | 20.56 | 13.12 | 28.22 | 62.11 | 109.86 |
| left_index | Similarity + bone scales + joint offsets | 17.00 | 14.53 | 20.69 | 35.54 | 115.52 |
| right_index | Similarity + bone scales + joint offsets | 23.90 | 16.78 | 32.01 | 70.97 | 128.71 |

Per-finger odd-holdout after best calibration:

| Sequence | Thumb mean/P95 | Index mean/P95 | Middle mean/P95 | Ring mean/P95 | Pinky mean/P95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| left_index | 13.49 / 29.70 | 12.78 / 26.61 | 17.18 / 38.88 | 14.24 / 32.87 | 12.72 / 31.93 |
| right_index | 11.97 / 29.41 | 23.95 / 69.45 | 25.61 / 73.30 | 23.48 / 67.70 | 17.79 / 49.48 |

The right sequence still has a clear distal-finger tail, especially around
index/middle/ring. That residual is not solved by a static local calibration
alone.

The bone-scale experiment is intentionally left in the report as a negative
result. It reduced training error but worsened odd-frame holdout for both
sequences. Many fitted bone scales also hit the conservative `[0.70, 1.30]`
limits, which suggests the model was absorbing MANO-vs-glove joint-definition
differences and pose distribution bias rather than learning a stable anatomical
bone-length correction. Therefore `--bone-scales` should stay off by default.

## Current Recommendation

Use the supervised calibration:

- similarity transform with scale, rotation, and translation
- per-joint residual offsets
- `joint_offset_shrink_k=25`
- `max_joint_offset_m=0.025`
- `bone_scales=none`
- write mode `separate`

The recommended outputs are:

- `video/sam3_hamer_right_index/hamer_mano_local_glove_calibrated/mano_local_hands_similarity_translate_jointoffset_k025_m025_even_train_000000_000477.jsonl`
- `video/sam3_hamer_left_index/hamer_mano_local_glove_calibrated/mano_local_hands_similarity_translate_jointoffset_k025_m025_even_train_000000_000442.jsonl`

Use evaluator space:

```bash
--space glove-calibrated-palm-local
```

## Cross-Sequence Transfer

To test whether the calibration is only memorizing a segment, the best calibration
from one sequence was applied to the other sequence without using the target
sequence glove data for fitting.

| Target sequence | Calibration source | Mean mm | Median mm | RMSE mm | P95 mm | Max mm |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| left_index | none/original | 30.15 | 27.44 | 33.65 | 59.52 | 140.17 |
| left_index | right_index calibration | 16.07 | 12.72 | 19.97 | 40.62 | 99.52 |
| left_index | left_index even-train calibration | 14.11 | 11.86 | 17.51 | 31.69 | 102.80 |
| right_index | none/original | 38.87 | 32.62 | 45.41 | 89.39 | 131.97 |
| right_index | left_index calibration | 23.71 | 15.11 | 32.27 | 75.76 | 118.28 |
| right_index | right_index even-train calibration | 20.59 | 13.14 | 28.25 | 62.29 | 109.86 |

This is a useful sanity check. Cross-sequence transfer still improves over raw
MANO local output, so the correction is not merely frame memorization. However,
same-sequence calibration is substantially better, which means part of the error
depends on pose distribution, capture conditions, or sequence-specific HaMeR
bias.

## Pose-Dependent Residual With Distribution Guard

`scripts/calibrate_pose_residual_local.py` adds a low-capacity ridge regressor
from the centered 21-joint HaMeR/MANO hand shape to the remaining glove-local
residual. It is fit after the recommended static calibration. This substantially
reduces the pose-dependent distal-finger error when the target motion comes from
the same capture sequence.

The model stores its normalized training-pose prototypes. At application time it
uses the nearest prototype distance as an out-of-distribution (OOD) test. Full
residual correction is retained through the 75th percentile of leave-one-out
training distances and decreases linearly to zero at the 99th percentile. An OOD
pose therefore falls back to the static `k=25, max=25mm` calibration rather than
receiving an unrelated pose correction.

| Validation setup | left_index mean / P95 mm | right_index mean / P95 mm |
| --- | ---: | ---: |
| Static calibration, odd holdout | 14.08 / 31.52 | 20.56 / 62.11 |
| Pose residual, odd holdout | 6.67 / 19.33 | 13.03 / 42.21 |
| Pose residual + OOD guard, odd holdout | 6.77 / 19.77 | 13.09 / 42.31 |
| Static calibration, later contiguous half | 13.80 / 31.00 | 18.25 / 57.51 |
| First-half pose residual + OOD guard, later half | 13.33 / 30.45 | 17.71 / 52.94 |

The odd/even split remains useful for quick iteration, but adjacent frames make
it optimistic. The contiguous-half result is the more conservative check: most
later-half poses were guarded out (left: 333/442 hand instances, right: 338/478)
yet the in-distribution remainder still improved the full held-out segment.

Cross-sequence application confirms why the guard is required:

| Target | Unguarded pose residual from other sequence | Guarded pose residual from other sequence | Static target calibration |
| --- | ---: | ---: | ---: |
| left_index | 17.08 / 41.92 | 14.09 / 31.54 | 14.11 / 31.69 |
| right_index | 17.55 / 61.22 | 20.36 / 62.32 | 20.59 / 62.29 |

Values are mean/P95 mm. The guarded version deliberately declines to transfer a
calibration when its source pose distribution does not cover the target; it
returns essentially the static baseline instead of causing the large left-side
regression seen without a guard.

Recommended pose-residual settings for a sequence with its own glove calibration
clip are `all-joints`, `ridge_alpha=10`, `correction_shrink=0.75`,
`max_correction_m=0.03`, and `ood_gating=knn-linear`. The two generated
same-sequence calibration files are:

- `video/sam3_hamer_left_index/hamer_mano_local_glove_calibrated/pose_residual_ood_alljoints_a010_s075_m030_even_train_000000_000442.json`
- `video/sam3_hamer_right_index/hamer_mano_local_glove_calibrated/pose_residual_ood_alljoints_a010_s075_m030_even_train_000000_000477.json`

## Local Pose-Residual KNN

The remaining residual is not globally linear. The calibration script now also
supports a local KNN regressor in the same normalized hand-pose feature space.
It uses only stored HaMeR/MANO pose prototypes and their glove residuals; it is
still multiplied by the OOD gate before being applied. Its training diagnostic is
leave-one-out, so it cannot report an artificial zero error by querying the
current calibration frame itself.

The best dense-coverage setting is `local-knn`, `k=2`, bandwidth scale `0.5`,
correction shrink `0.75`, and a `60mm` per-joint cap. It is a large improvement
on the deliberately interleaved odd-frame holdout:

| Sequence | Global ridge + OOD, mean / P95 / max mm | Local KNN + OOD, mean / P95 / max mm |
| --- | ---: | ---: |
| left_index | 6.77 / 19.77 / 89.06 | 4.09 / 10.06 / 89.06 |
| right_index | 13.09 / 42.31 / 81.29 | 5.87 / 17.64 / 67.76 |

Per-finger KNN mean/P95 mm is `3.74/8.06`, `3.74/8.67`, `4.98/12.26`,
`4.21/10.49`, `3.76/10.56` for left thumb through pinky, and
`3.38/8.90`, `6.70/18.44`, `7.56/22.87`, `6.76/19.74`, `4.97/14.34` for right.
The right index/middle/ring distal tail remains the least reliable region, but it
is materially smaller than with global ridge.

This is intentionally a **dense calibration-coverage** result, not a claim that
an arbitrary earlier clip can predict all future hand motions. In the harder
first-half-train/later-half-eval test, KNN with a conservative 30mm cap was
similar to the global ridge (`13.43/30.92` vs `13.33/30.45` on left and
`17.64/52.89` vs `17.71/52.94` on right). Raising the cap to 60mm was harmless
on left but increased right later-half P95 to `53.66mm`. Therefore use the 60mm
KNN calibration only when the calibration set densely covers the target pose
space; otherwise retain the 30mm ridge/KNN calibration or static calibration.

The dense-coverage outputs are:

- `video/sam3_hamer_left_index/hamer_mano_local_glove_calibrated/mano_local_hands_similarity_translate_jointoffset_k025_m025_pose_knn2_bw050_ood_m060_even_train_000000_000442.jsonl`
- `video/sam3_hamer_right_index/hamer_mano_local_glove_calibrated/mano_local_hands_similarity_translate_jointoffset_k025_m025_pose_knn2_bw050_ood_m060_even_train_000000_000477.jsonl`
- `video/sam3_hamer_left_index/hamer_mano_local_glove_calibrated/pose_knn2_bw050_ood_m060_even_train_000000_000442.json`
- `video/sam3_hamer_right_index/hamer_mano_local_glove_calibrated/pose_knn2_bw050_ood_m060_even_train_000000_000477.json`

Applying either KNN calibration to the other sequence produced `14.09/31.58`
on left and `20.36/62.32` on right, effectively the static calibration baseline.
The nearest-pose OOD gate is what makes this safe: 861/886 left and 711/956
right cross-sequence hand instances received zero local correction.

## Temporal Smoothing Ablation

The smoothing utility now supports `--space glove-calibrated-palm-local`, so it
can operate on the fields that the evaluator actually consumes. On the strict
first-half-train/later-half-eval split, robust Hampel replacement found no
outliers at a 35mm threshold. Bidirectional EMA smoothing only changed joints by
about 0.1mm on average and changed the final error from `13.33 / 30.45` to
`13.32 / 30.53` for left_index and from `17.71 / 52.94` to `17.68 / 52.82` for
right_index. It may make an offline viewer slightly calmer, but is not an
accuracy-critical stage and should remain optional.

## Pose-Velocity Descriptor Ablation

`all-joints-velocity` concatenates the current centered 21-joint hand shape and
the centered finite-difference motion from its adjacent HaMeR frames. It uses no
neighbouring glove labels. In the first-half-train/later-half-eval experiment,
with ridge alpha 100, it improved the OOD-guarded pose-only model from
`13.33 / 30.45` to `12.00 / 28.98` on left_index and from `17.71 / 52.94` to
`16.06 / 46.51` on right_index (mean/P95 mm).

It did **not** pass the interleaved odd-frame holdout: left_index changed from
`6.77 / 19.77` to `7.65 / 20.97` and right_index from `13.09 / 42.31` to
`14.16 / 44.06`. The descriptor is therefore retained as an experimental
chronological-clip option, not the default. Any use of it should be selected by
a contiguous validation split that matches the intended calibration-then-use
workflow; do not select it from training error or an adjacent-frame split alone.

## Pure Apply Mode

Existing calibration JSON files can be applied to a new HaMeR JSONL without
glove GT:

```bash
python3 scripts/calibrate_hamer_to_glove_local.py \
  --hamer video/sam3_hamer_left_index/hamer_mano_local_refined/mano_local_hands_000000_000442.jsonl \
  --output video/sam3_hamer_left_index/hamer_mano_local_glove_calibrated/mano_local_hands_apply_right_calibration_k025_m025_000000_000442.jsonl \
  --load-calibration-json video/sam3_hamer_right_index/hamer_mano_local_glove_calibrated/similarity_translate_jointoffset_k025_m025_even_train_000000_000477.json \
  --space palm-local \
  --group-range 0-442 \
  --write-mode separate \
  --overwrite
```

## Interpretation

The result suggests the current largest remaining error source is a combination
of stable MANO-to-glove joint-definition mismatch and pose/capture-dependent
HaMeR bias, especially at distal joints. Temporal jitter is comparatively small.
For production without target-session glove GT, use only a static calibration
estimated for the same subject/device setup, or apply a pose-residual calibration
with its OOD guard enabled so unfamiliar motions cleanly fall back to that static
result.

## Related Work Notes

The current direction is consistent with the broader MANO/HaMeR literature:

- HaMeR uses a large transformer model to regress MANO hand reconstruction from
  monocular images; it is a strong topology/pose prior but still outputs a model
  convention, not a glove-device convention.
  https://arxiv.org/abs/2312.05251
- MANO itself is a learned low-dimensional hand model from hand scans. Its joints,
  root, and blend-shape conventions are model definitions and do not have to match
  a glove SDK's local joint definitions exactly.
  https://arxiv.org/abs/2201.02610
- Earlier hand mesh recovery work also relies on differentiable reprojection and
  parametric hand models, reinforcing that image-space fitting and model-space
  calibration are separate concerns.
  https://arxiv.org/abs/1902.09305

## Image-Space Multi-View Diagnostic

The image-space MANO refiner is intentionally still disabled by default. A
direct raw comparison found that the previous image-space outputs did not yet
improve glove-local hand shape: left local MANO was `30.15mm` mean versus
`31.03mm` image-space, while right was `38.87mm` versus `39.51mm`. The old runs
accepted almost no metric multi-view observations because their mean final
reprojection errors were roughly `180–350px`.

The primary cause is now understood. HaMeR `cam_t` is defined under HaMeR's
virtual crop focal length, whereas the refiner projects using the physical
rectified camera intrinsics. Reusing that virtual-camera translation as a
physical initialization puts the hand at the wrong metric scale. An experimental
`--global-initialization physical-pnp` option now initializes global pose from
MANO local joints and HaMeR 2D points through the real rectified K. On a five
frame left smoke test this reduced the C1 anchor reprojection from about 184px
to 21px, but C0/C2/C3 remain inconsistent with that same physical pose. The
resulting local GT error did not improve (`39.32mm` versus `36.76mm` baseline),
so PnP remains opt-in until per-camera geometry correction is validated.

This investigation also fixed a concrete SAM3 integration bug in
`fuse_hamer_jobs.py`: stale relative mask paths from stabilized tracks lost the
`chunks/` component during relocation. All 3,211 left-sequence SAM3 masks were
therefore silently dropped before HaMeR. The resolver now finds all 3,211 files;
a 0–4 smoke run confirmed that all 35 fused HaMeR jobs receive mask-blurred
input. The corrected masks alone did not improve that five-frame local-MANO
sample (`37.33mm` versus `36.76mm`), so no costly full left rerun is recommended
until the auxiliary-camera geometry issue is addressed.

An additional opt-in `--pnp-view-gate-m` now rejects an auxiliary observation
when its physical-K PnP wrist estimate disagrees with the anchor. The 0.10m
five-frame mask smoke improved PnP image refinement from `40.79mm` to `38.26mm`
mean, but still did not beat local MANO (`36.76mm`). Forcing C1/C2 as the anchor
while that gate is active was also worse (`39.52mm`). These switches remain
diagnostic tools, not production defaults. The next legitimate image-space
improvement requires a separately validated camera SE(3) correction or a more
independent 2D observation, rather than further heuristic loss tuning.

## Mask-Enabled PnP Gate Follow-up

The original HaMeR prediction JSONL files unexpectedly contained zero usable
`sam3_mask_path` values, so their image-space mask loss had never actually been
active. The stale stabilized-track paths were fixed in `fuse_hamer_jobs.py`, and
both left- and right-index 0--20 shards were re-fused and re-run through HaMeR:
all 156 jobs per shard then carried a valid SAM3 mask and used the blur-masked
HaMeR input.

With that real mask-enabled input, the decisive improvement was not a new mask
term but rejecting geometrically inconsistent auxiliary views before optimize.
`physical-pnp` initialization plus `--pnp-view-gate-m 0.04` was compared with
the same predictions and the gate disabled:

| Sequence | Gate off mean / P95 / max | 0.04m gate mean / P95 / max |
| --- | --- | --- |
| left-index, groups 0--20 | 38.33 / 69.83 / 118.20 mm | **34.06 / 54.38 / 81.57 mm** |
| right-index, groups 0--20 | 53.37 / 112.98 / 131.62 mm | **49.22 / 110.57 / 126.37 mm** |

The symmetric SAM3-boundary-to-mesh experimental loss at weight `0.10` was not
promoted: on left-index it changed mean/P95 from `38.33/69.83mm` to
`38.48/70.45mm`. It remains an explicit opt-in experiment with a default weight
of zero.

By contrast, attempting to fit a single global camera SE(3) correction from
MANO+HaMeR 2D PnP was rejected by a true held-out test: its apparent improvement
was hand/reference-specific and could severely worsen another camera pair. The
estimator remains a diagnostic artifact only. This is distinct from the PnP view
gate, which does not alter calibration; it only excludes a frame/view when its
own physical-K pose is inconsistent with that frame's anchor. Accordingly,
image-space refinement remains pipeline-opt-in, but when enabled its defaults
are now `physical-pnp` and a conservative `0.04m` PnP gate.

## Baseline/Gated Candidate Selection

The initial `500px` threshold rule was intentionally conservative, but it did
not generalize to the full sequences. A broader independent evaluation showed
that the PnP-gated candidate is the stronger default whenever it exists:

| Sequence | Baseline mean / P95 | Always gated mean / P95 |
| --- | --- | --- |
| left-index, groups 0--442 | 36.71 / 67.14 mm | **34.70 / 63.72 mm** |
| right-index, groups 0--49 | 46.62 / 102.41 mm | **43.65 / 100.33 mm** |
| right-index, groups 0--99 | 46.96 / 102.24 mm | **45.11 / 98.92 mm** |
| right-index, groups 0--477 | 44.41 / 99.42 mm | **42.02 / 93.20 mm** |

This comparison uses the same predictions, masks, and calibration for each
pair; glove GT is used only for offline evaluation. The former `500px` rule
kept only 43/100 gated hand candidates in the right shard and gave a weaker
`46.01mm` mean. The gate's geometric rejection itself, rather than a large
baseline reprojection residual, is the useful signal.

`scripts/select_image_refinement_candidates.py` therefore now prefers every
available PnP-gated candidate by default (`--min-baseline-max-reprojection-px
0`). A positive threshold remains available as an explicit experiment. Missing
or failed gated hands still fall back to baseline. The pipeline writes this
selected result under `hamer_mano_multiview_selected/`, which the viewer reads
first. Image refinement also requires at least 50% readable SAM3 masks by
default, preventing legacy predictions with silently inactive mask losses from
being treated as valid experiments.

### Pipeline Reconstruction Evidence

The repaired end-to-end pipeline was run on five consecutive left-index shards
(groups 0--249), each with 100% readable SAM3-mask coverage. The selected
shards were merged with strict duplicate/conflict checking and compared against
the corresponding no-gate baseline over 4,500 glove-evaluated points:

| Output | Mean | Median | RMSE | P95 | Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline image refine | 37.43 mm | 35.09 mm | 40.61 mm | 66.00 mm | 142.48 mm |
| selected baseline/gated image refine | **36.20 mm** | **33.61 mm** | **39.42 mm** | **64.89 mm** | 142.48 mm |

This is a real pipeline-level gain, not a manually constructed smoke output.
The selected output is stored at
`video/sam3_hamer_left_index/hamer_mano_multiview_selected/mano_multiview_local_hands_000000_000249.jsonl`.

The complete left-index sequence (443 frames, groups 0--442) was subsequently
reconstructed in nine non-overlapping shards and merged with no duplicate
group IDs. All 886 hand records have finite palm-local joints. Across 7,974
glove-evaluated points, the no-gate baseline versus the selected output was:

| Output | Mean | Median | RMSE | P95 | Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline image refine | 36.71 mm | 34.14 mm | 40.09 mm | 67.14 mm | 142.48 mm |
| old 500px selector | 35.66 mm | 33.07 mm | 39.03 mm | 65.26 mm | 142.48 mm |
| PnP-gated candidate (new default) | **34.70 mm** | **32.40 mm** | **37.76 mm** | **63.72 mm** | 144.04 mm |

The former selector chose the gated candidate for 265/886 hand frames. The new
default selects all complete gated hand candidates. The merged artifact at
`video/sam3_hamer_left_index/hamer_mano_multiview_selected/mano_multiview_local_hands_000000_000442.jsonl`
has been regenerated with that rule: 886/886 hand records now come from the
PnP-gated candidate, with zero baseline fallbacks and finite palm-local joints.
The right-index selected shards for groups 0--477 were regenerated the same way
and merge cleanly into
`video/sam3_hamer_right_index/hamer_mano_multiview_selected/mano_multiview_local_hands_000000_000477.jsonl`.
All 956/956 hand records now come from the PnP-gated candidate, with zero
baseline fallbacks and finite palm-local joints.

## Selected Image Output as Glove-Calibration Base

The selected image-space output is useful as the current no-GT image-side
viewer/result, but it is **not** a better base for glove-supervised local
calibration. The same even-train/odd-eval calibration stack was applied to
`hamer_mano_multiview_selected/` and compared with the existing
`hamer_mano_local_refined/` calibration base:

| Calibration base | left_index static mean / P95 | right_index static mean / P95 | left_index KNN mean / P95 | right_index KNN mean / P95 |
| --- | ---: | ---: | ---: | ---: |
| `hamer_mano_local_refined` | **14.08 / 31.52 mm** | **20.56 / 62.11 mm** | **4.09 / 10.05 mm** | **5.87 / 17.64 mm** |
| `hamer_mano_multiview_selected` | 15.80 / 38.73 mm | 21.60 / 62.59 mm | 5.38 / 14.38 mm | 7.57 / 24.05 mm |

The selected base therefore worsens the supervised output even after dense
local-KNN residual correction. A hand-level odd-frame diagnostic showed selected
KNN won only 110/442 left-sequence hands and 103/478 right-sequence hands.
The mean hand-level error delta was still worse by `+1.29mm` on left and
`+1.69mm` on right. Low image reprojection error did not reverse the conclusion:
the lowest max-reprojection quartile was still worse by `+0.59mm` on left and
`+0.76mm` on right.

Accordingly, keep the outputs separated:

- for no-GT image-side visualization/debugging, use
  `hamer_mano_multiview_selected/`;
- for glove-supervised local coordinates, keep
  `hamer_mano_local_refined/` as the calibration base and apply the existing
  static/KNN calibration stack.

## Strict Chronological Glove Calibration

The earlier even-train/odd-eval numbers are useful for measuring interpolation
capacity, but they let the static similarity/joint-offset calibration see both
early and late sequence poses. To remove that leakage, the static calibration
script now supports explicit `--train-group-range` and `--train-group-ids`
arguments. The following table trains all calibration layers only on the first
half of each sequence, then evaluates on the held-out late half:

| Calibration output | left_index late-half mean / P95 / max | right_index late-half mean / P95 / max |
| --- | ---: | ---: |
| static similarity + translation + joint offsets | 15.15 / 32.36 / 102.35 mm | 20.60 / 63.16 / 101.09 mm |
| static + pose residual, all-joint features | 14.57 / 31.61 / 102.39 mm | 19.83 / 59.49 / 101.09 mm |
| static + pose residual, fingertip-summary features | 14.23 / 31.55 / 103.80 mm | 19.36 / 56.64 / 101.09 mm |
| static + pose + velocity residual | **12.52 / 29.40** / 111.79 mm | **17.84 / 49.34 / 97.37 mm** |

The pose+velocity residual is therefore the strongest strictly chronological
variant so far in mean and P95, especially on right-index. The left-index max
does get worse, which suggests this should be treated as a sequence-level
calibration option rather than an unconditional dense default. The important
practical change is that future held-out reports should use the explicit
training range, not parity alone, when the goal is a deployment-like estimate.

## Image-Space Beta Experiment

The original sequence beta estimator uses a weighted HaMeR-local joint fit.
An experimental `--beta-estimation-space image-2d` path was added: it keeps
each observation's HaMeR pose and physical-PnP transform fixed, then optimizes
the shared 10-D beta against calibrated multi-view HaMeR 2D reprojection. This
is a genuine image-space shape update, not an optimization against glove GT.

On the left-index gated shard (groups 0--49), it changed the left/right beta
vectors by L2 `0.050/0.046`, but the glove result was effectively neutral:

| Beta estimator | Mean | Median | RMSE | P95 | Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| HaMeR-local (current default) | 32.57 mm | 31.56 mm | 33.87 mm | **48.70 mm** | **79.80 mm** |
| physical-PnP image 2D | **32.53 mm** | **31.51 mm** | **33.84 mm** | 48.75 mm | 79.83 mm |

The tiny mean change does not justify a default switch or its additional
runtime. The implementation is retained as an explicit experimental CLI path;
the next useful shape signal should incorporate a differentiable SAM3
silhouette term across the sequence, rather than more HaMeR 2D self-consistency.

## Second-Order Temporal Prior Experiment

`--temporal-acceleration-weight` adds a second-order local pose/joint prior
only after two consecutive hand states exist. It was tested at `0.10` against
the PnP-gated candidate over independent 50-frame left/right shards:

| Sequence | Current mean / P95 / max | Acceleration mean / P95 / max |
| --- | --- | --- |
| left-index 0--49 | 32.570 / 48.697 / 79.804 mm | **32.560 / 48.667 / 79.575 mm** |
| right-index 0--49 | 43.647 / 100.327 / **127.565 mm** | **43.644 / 100.217** / 127.691 mm |

The effect is directionally positive in mean/RMSE/P95 but too small, and the
right maximum is marginally worse. It remains disabled by default rather than
being promoted on a weak result. The more consequential remaining gap is a
proper two-sided SAM3 silhouette objective; the installed environment has no
PyTorch3D, nvdiffrast, or Kaolin renderer, so that should be implemented as an
explicit renderer dependency or a carefully validated lightweight alternative.

## Zero-Shot Direct Multi-View Palm Fusion

The deployment goal is in-the-wild and zero-shot, so glove data must not be an
input to the fusion method. `scripts/fuse_hamer_palm_local.py` implements a new
path with that constraint:

1. Convert every per-view HaMeR result into its own canonical palm frame.
2. Use image quality only to choose between duplicate hypotheses from the same
   camera.
3. Give the selected cameras equal weight and average corresponding joints.
4. Preserve that result in `raw_palm_local_joints_m` unconditionally.
5. Store static shape calibration, causal EMA, and offline Gaussian smoothing
   in separate fields; none of them overwrite the raw result unless explicitly
   selected with `--primary-output`.

The generated config writes `uses_ground_truth: false`,
`cross_view_weighting: equal`, and the exact output-field choice. The main
pipeline runs this stage by default, but keeps `--zero-shot-primary-output raw`,
zero bone calibration, and zero temporal smoothing as the deployment-safe
defaults.

### Why Equal View Weights

Image quality and cross-view agreement were evaluated as possible no-GT
reliability signals. Neither was reliable enough to gate the pose:

- quality-score/error correlation was `-0.18` on left-index and `+0.10` on
  right-index;
- consensus-spread/error correlation was `-0.089` and `+0.019`;
- per-joint Huber fusion improved left mean slightly but damaged right mean and
  tail error because C3 was sometimes the most accurate view while also being
  the least consensus-like view.

Therefore quality scores are deliberately limited to within-camera duplicate
selection. Cross-view disagreement is exported as diagnostic uncertainty, not
used as an accuracy confidence or hard rejection gate.

### Zero-Shot Results

All numbers below use glove only after inference as an evaluation ruler. The
fusion output itself never reads glove files, labels, calibration parameters,
or residual models. Metrics cover all five fingers over the full left 0--442
and right 0--477 sequences.

| Zero-shot output | left mean / median / P95 / max | right mean / median / P95 / max |
| --- | ---: | ---: |
| Previous MANO local refine | 30.12 / 27.45 / 59.70 / 139.68 mm | 38.83 / 32.56 / 89.36 / 130.46 mm |
| Direct equal-view mean, raw | 28.47 / 26.10 / 53.68 / 111.67 mm | 35.50 / 30.27 / 77.33 / 130.17 mm |
| Direct mean + causal EMA `alpha=0.20` | 27.45 / 25.18 / 51.89 / 101.28 mm | 34.55 / 29.51 / 74.26 / 121.15 mm |
| Direct mean + adaptive causal One Euro | **27.07 / 24.83 / 51.59 / 101.16 mm** | **34.02 / 28.85 / 73.32 / 117.62 mm** |
| Direct mean + offline Gaussian `radius=10,sigma=4` | **27.37 / 25.08 / 51.47 / 93.92 mm** | **34.31 / 29.40 / 73.06 / 120.25 mm** |

The Gaussian result is an offline option with a ten-frame look-ahead, not a
causal deployment claim. All temporal filters reset at missing detections, so
they do not drag an old hand state across a gap. The raw field remains the
authoritative observation in every mode.

The adaptive causal result uses the timestamp-derived frame rate (about 25 FPS) with
`min_cutoff=0.2`, `beta=5.0`, and derivative cutoff `1.0`. Parameters were selected
on even groups and then checked on held-out odd groups. On odd groups it reached
`27.05/51.46mm` on left and `33.98/73.29mm` on right (mean/P95), versus fixed
EMA's `27.44/51.69mm` and `34.51/74.11mm`. It uses no look-ahead and no glove
input at inference; the pipeline computes this optional field by default while
keeping raw as the primary output.

### Static Calibration Without Pose Supervision

An optional zero-shot static bone calibration estimates a camera-balanced
median bone length per hand and reconstructs every bone along its original
per-frame direction. It changes shape length only; it does not fit pose
residuals or use glove. On these sequences its effect was below `0.1mm` in mean
and P95, while the worst-case effect was mixed, so `--bone-calibration-blend`
remains `0` by default. This is useful as a conservative shape-normalization
experiment, not as evidence for a default accuracy gain.

SO(3) averaging of the 15 MANO joint rotations was also tested. It retains a
strictly valid MANO parameterization, but even with the same offline temporal
window it reached `28.74/53.89mm` on left and `35.47/78.01mm` on right
(mean/P95), worse than direct joint correspondence fusion. It is therefore a
mesh-valid fallback direction rather than the joint-accuracy default.

### Geometry Branch Fixes and Limits

`scripts/triangulate_mediapipe_hands.py` previously assumed detections were
contiguous by `group_id`, while the actual JSONL is grouped by camera. Every
record was consequently treated as a one-camera group and zero hands could be
triangulated. The loader now groups all records by `group_id` explicitly, and
the script can optionally use stabilized SAM3 tracks for bbox-only hand
association via `--tracked-hands`.

After grouping, track association, and a `0.05--1.0m` positive-depth gate,
strict MediaPipe triangulation produced 396 left-sequence and 415 right-sequence
hand instances. Complete-pose mean/P95 remained `41.31/76.63mm` and
`48.11/99.06mm`, so this is still not accurate or complete enough to replace
HaMeR fusion. Direct triangulation of
HaMeR-rendered 2D joints was also rejected: virtual-camera/physical-camera
mismatch created severe depth failures. Geometry is retained as a diagnostic
branch until identity association and physical 2D observations are reliable.

### Commands

Deployment-safe raw fusion with optional outputs retained separately:

```bash
/home/luojiangrui/miniconda3/envs/headcam/bin/python scripts/fuse_hamer_palm_local.py \
  --predictions video/sam3_hamer_left_index/hamer_per_view/hamer_predictions_000000_000442.jsonl \
  --output-dir video/sam3_hamer_left_index/hamer_palm_local_fused \
  --group-range 0-442 \
  --temporal-radius 10 \
  --temporal-sigma 4 \
  --causal-ema-alpha 0.20 \
  --one-euro-min-cutoff 0.2 \
  --one-euro-beta 5.0 \
  --primary-output raw \
  --overwrite
```

Use `--primary-output adaptive-causal` for the validated causal result, or
`--primary-output smoothed` only for an explicitly offline result. View the
raw zero-shot skeleton with:

```bash
/home/luojiangrui/miniconda3/envs/headcam/bin/python scripts/view_hamer_multiview.py \
  --dataset left_index --range 0-442 --zero-shot
```
