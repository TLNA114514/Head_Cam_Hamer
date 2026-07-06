#!/usr/bin/env python3
"""Run HaMeR on fused multi-view hand jobs and export structured predictions."""

from __future__ import annotations

import argparse
import builtins
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from hamer_multiview_utils import DEFAULT_BASE_DIR, iter_jsonl, load_mask, parse_group_ids, range_suffix
from progress_utils import tqdm


WRIST_CAM_ROOT = Path("/home/luojiangrui/ljr/wrist_cam")
DEFAULT_HAMER_ROOT = WRIST_CAM_ROOT / "third_party" / "hamer"
LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)
_ORIGINAL_PRINT = builtins.print


def install_noise_filters() -> None:
    warnings.filterwarnings("ignore")

    def quiet_print(*args: Any, **kwargs: Any) -> None:
        text = " ".join(str(arg) for arg in args)
        if text.startswith("downsampling_factor="):
            return
        _ORIGINAL_PRINT(*args, **kwargs)

    builtins.print = quiet_print


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=Path, help="HaMeR jobs JSONL. Defaults to hamer_jobs by range.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_BASE_DIR / "hamer_per_view")
    parser.add_argument("--hamer-root", type=Path, default=DEFAULT_HAMER_ROOT)
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--group-range")
    parser.add_argument("--group-ids")
    parser.add_argument("--camera-id", help="Optional single camera shard.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--rescale-factor", type=float, default=2.0)
    parser.add_argument("--candidate-bbox-scales", default="1.0,1.1,1.2")
    parser.add_argument("--save-rendered-overlay", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-debug-only", action="store_true")
    parser.add_argument("--progress-every", type=int, default=10, help="Print line-based progress every N jobs; 0 disables.")
    parser.add_argument("--progress-position", type=int, default=int(os.environ.get("TQDM_POSITION", "0")), help="tqdm terminal row position.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def parse_float_list(value: str) -> list[float]:
    out = []
    for part in value.split(","):
        part = part.strip()
        if part:
            out.append(float(part))
    return out or [1.0]


def scale_bbox_xyxy(bbox: list[float], image_shape: tuple[int, int, int], scale: float) -> np.ndarray:
    img_h, img_w = image_shape[:2]
    x1, y1, x2, y2 = [float(v) for v in bbox]
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    w = max(x2 - x1, 2.0) * scale
    h = max(y2 - y1, 2.0) * scale
    return np.asarray(
        [
            np.clip(cx - w * 0.5, 0, img_w - 1),
            np.clip(cy - h * 0.5, 0, img_h - 1),
            np.clip(cx + w * 0.5, 0, img_w - 1),
            np.clip(cy + h * 0.5, 0, img_h - 1),
        ],
        dtype=np.float32,
    )


def resolve_hamer_checkpoint(hamer_root: Path, checkpoint: str | None) -> Path:
    if checkpoint:
        checkpoint_path = Path(checkpoint).expanduser()
        if not checkpoint_path.is_absolute():
            checkpoint_path = hamer_root / checkpoint_path
    else:
        checkpoint_path = hamer_root / "_DATA" / "hamer_ckpts" / "checkpoints" / "hamer.ckpt"
    return checkpoint_path.resolve()


def iter_jobs(
    path: Path,
    camera_id: str | None,
    include_debug_only: bool,
    group_ids: set[int] | None = None,
) -> list[dict[str, Any]]:
    jobs = []
    for record in iter_jsonl(path):
        if record.get("type") != "hamer_multiview_jobs":
            continue
        group_id = int(record["group_id"])
        if group_ids is not None and group_id not in group_ids:
            continue
        if camera_id and record.get("camera_id") != camera_id:
            continue
        for job in record.get("jobs") or []:
            if job.get("debug_only") and not include_debug_only:
                continue
            jobs.append(job)
    return jobs


def candidate_records(job: dict[str, Any], image_shape: tuple[int, int, int], scales: list[float]) -> list[dict[str, Any]]:
    handedness = job.get("handedness")
    if handedness == "Left":
        sides = [(0, "Left", "known")]
    elif handedness == "Right":
        sides = [(1, "Right", "known")]
    else:
        sides = [(0, "Left", "unknown_hypothesis"), (1, "Right", "unknown_hypothesis")]
    records = []
    for is_right, side, status in sides:
        for scale in scales:
            records.append(
                {
                    "bbox": scale_bbox_xyxy(job["bbox_rectified_px"], image_shape, scale),
                    "is_right": int(is_right),
                    "handedness": side,
                    "hypothesis_status": status,
                    "bbox_scale": float(scale),
                }
            )
    return records


def choose_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda item: (
            item.get("mask_score") if item.get("mask_score") is not None else -1.0,
            -abs(float(item.get("bbox_scale", 1.0)) - 1.0),
            item.get("hypothesis_status") == "known",
        ),
    )


def score_candidates_with_mask(
    candidates: list[dict[str, Any]],
    mask_path: str | None,
    renderer: Any,
    image_shape: tuple[int, int, int],
    focal_length: Any,
) -> None:
    mask = load_mask(mask_path)
    if mask is None:
        return
    for candidate in candidates:
        try:
            rgba = renderer.render_rgba_multiple(
                [candidate["verts"]],
                cam_t=[candidate["cam_t"]],
                render_res=[image_shape[1], image_shape[0]],
                is_right=[candidate["is_right"]],
                focal_length=focal_length,
            )
        except Exception:
            continue
        mesh_mask = rgba[:, :, 3] > 0.05
        inter = float(np.logical_and(mask, mesh_mask).sum())
        union = float(np.logical_or(mask, mesh_mask).sum())
        sam_area = float(mask.sum())
        mesh_area = float(mesh_mask.sum())
        iou = inter / union if union > 0 else 0.0
        sam_cov = inter / sam_area if sam_area > 0 else 0.0
        mesh_cov = inter / mesh_area if mesh_area > 0 else 0.0
        candidate["mask_iou"] = iou
        candidate["mask_sam_coverage"] = sam_cov
        candidate["mask_mesh_coverage"] = mesh_cov
        candidate["mask_score"] = 0.45 * iou + 0.45 * sam_cov + 0.10 * mesh_cov


def main() -> None:
    install_noise_filters()
    args = parse_args()
    original_cwd = Path.cwd()
    group_ids = parse_group_ids(args.group_range, args.group_ids)
    suffix = range_suffix(group_ids)
    jobs_path = args.jobs or (DEFAULT_BASE_DIR / "hamer_jobs" / f"hamer_jobs_{suffix}.jsonl")
    if not jobs_path.is_absolute():
        jobs_path = original_cwd / jobs_path
    camera_suffix = f"_{args.camera_id}" if args.camera_id else ""
    if not args.output_dir.is_absolute():
        args.output_dir = original_cwd / args.output_dir
    output_path = args.output_dir / f"hamer_predictions_{suffix}{camera_suffix}.jsonl"
    if args.dry_run:
        jobs = iter_jobs(jobs_path, args.camera_id, args.include_debug_only, group_ids) if jobs_path.exists() else []
        print(json.dumps({"jobs": len(jobs), "jobs_path": str(jobs_path), "output_path": str(output_path)}, indent=2))
        return
    if output_path.exists() and not args.overwrite:
        raise SystemExit(f"{output_path} exists; pass --overwrite to replace it")
    if not jobs_path.exists():
        raise SystemExit(f"jobs not found: {jobs_path}")

    args.hamer_root = args.hamer_root.expanduser().resolve()
    if not args.hamer_root.exists():
        raise SystemExit(f"HaMeR root not found: {args.hamer_root}. Run/setup wrist_cam HaMeR first.")
    sys.path.insert(0, str(args.hamer_root))

    from hamer.datasets.vitdet_dataset import ViTDetDataset  # type: ignore
    from hamer.models import load_hamer  # type: ignore
    from hamer.utils import recursive_to  # type: ignore
    from hamer.utils.renderer import Renderer, cam_crop_to_full  # type: ignore

    checkpoint_path = resolve_hamer_checkpoint(args.hamer_root, args.checkpoint)
    model_config_path = checkpoint_path.parent.parent / "model_config.yaml"
    if not checkpoint_path.exists():
        raise SystemExit(f"HaMeR checkpoint not found: {checkpoint_path}. Reuse/setup wrist_cam HaMeR first.")
    if not model_config_path.exists():
        raise SystemExit(f"HaMeR model_config.yaml not found next to checkpoint: {model_config_path}")
    print(f"[HaMeR {args.camera_id or 'all'}] loading model checkpoint={checkpoint_path}", flush=True)
    os.chdir(args.hamer_root)
    try:
        model, model_cfg = load_hamer(str(checkpoint_path))
    finally:
        os.chdir(original_cwd)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device).eval()
    renderer = Renderer(model_cfg, faces=model.mano.faces)
    scales = parse_float_list(args.candidate_bbox_scales)
    jobs = iter_jobs(jobs_path, args.camera_id, args.include_debug_only, group_ids)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    render_dir = args.output_dir / "rendered"
    stats = {"jobs": len(jobs), "predictions": 0, "failed_jobs": 0, "unknown_jobs": 0}

    print(f"[HaMeR {args.camera_id or 'all'}] start jobs={len(jobs)} output={output_path}", flush=True)
    with output_path.open("w", encoding="utf-8") as out:
        progress = tqdm(jobs, desc=f"HaMeR {args.camera_id or 'all'}", unit="job", position=args.progress_position)
        for index, job in enumerate(progress, start=1):
            frame_path = Path(job["hamer_frame_path"])
            img_cv2 = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if img_cv2 is None:
                stats["failed_jobs"] += 1
                continue
            records = candidate_records(job, img_cv2.shape, scales)
            if job.get("handedness") == "unknown":
                stats["unknown_jobs"] += 1
            boxes = np.stack([item["bbox"] for item in records])
            right = np.asarray([item["is_right"] for item in records], dtype=np.int64)
            dataset = ViTDetDataset(model_cfg, img_cv2, boxes, right, rescale_factor=args.rescale_factor)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
            candidates = []
            record_offset = 0
            last_scaled_focal_length = None
            for batch in dataloader:
                batch = recursive_to(batch, device)
                with torch.no_grad():
                    pred = model(batch)
                multiplier = 2 * batch["right"] - 1
                pred_cam = pred["pred_cam"]
                pred_cam[:, 1] = multiplier * pred_cam[:, 1]
                box_center = batch["box_center"].float()
                box_size = batch["box_size"].float()
                img_size = batch["img_size"].float()
                scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
                pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()
                last_scaled_focal_length = scaled_focal_length

                batch_size = batch["img"].shape[0]
                for n in range(batch_size):
                    rec = records[record_offset + n]
                    is_right = int(batch["right"][n].detach().cpu().item())
                    verts = pred["pred_vertices"][n].detach().cpu().numpy()
                    joints = pred["pred_keypoints_3d"][n].detach().cpu().numpy()
                    verts[:, 0] = (2 * is_right - 1) * verts[:, 0]
                    joints[:, 0] = (2 * is_right - 1) * joints[:, 0]
                    candidates.append(
                        {
                            "handedness": rec["handedness"],
                            "is_right": is_right,
                            "hypothesis_status": rec["hypothesis_status"],
                            "bbox_scale": rec["bbox_scale"],
                            "verts": verts,
                            "joints": joints,
                            "raw_verts": pred["pred_vertices"][n].detach().cpu().numpy(),
                            "cam_t": pred_cam_t_full[n],
                            "pred_cam_t": pred["pred_cam_t"][n].detach().cpu().numpy(),
                            "batch_img": batch["img"][n].detach().cpu(),
                            "mask_score": None,
                        }
                    )
                record_offset += batch_size

            score_candidates_with_mask(
                candidates,
                job.get("sam3_mask_path"),
                renderer,
                img_cv2.shape,
                last_scaled_focal_length,
            )
            selected = choose_candidate(candidates)
            if selected is None:
                stats["failed_jobs"] += 1
                continue

            render_path = None
            if args.save_rendered_overlay:
                try:
                    rgba = renderer.render_rgba_multiple(
                        [selected["verts"]],
                        cam_t=[selected["cam_t"]],
                        render_res=[img_cv2.shape[1], img_cv2.shape[0]],
                        is_right=[selected["is_right"]],
                        focal_length=last_scaled_focal_length,
                    )
                    input_img = img_cv2.astype(np.float32)[:, :, ::-1] / 255.0
                    input_img = np.concatenate([input_img, np.ones_like(input_img[:, :, :1])], axis=2)
                    overlay = input_img[:, :, :3] * (1 - rgba[:, :, 3:]) + rgba[:, :, :3] * rgba[:, :, 3:]
                    render_path = render_dir / str(job["camera_id"]) / f"{int(job['group_id']):08d}_{int(job['job_index']):02d}.jpg"
                    render_path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(render_path), 255 * overlay[:, :, ::-1])
                except Exception as exc:
                    print(f"render failed for job {job.get('group_id')} {job.get('camera_id')}: {exc}", flush=True)

            prediction = {
                "type": "hamer_multiview_prediction",
                "group_id": int(job["group_id"]),
                "camera_id": job["camera_id"],
                "job_index": int(job["job_index"]),
                "handedness": selected["handedness"],
                "handedness_source": job.get("handedness_source"),
                "is_right": int(selected["is_right"]),
                "bbox_rectified_px": job["bbox_rectified_px"],
                "source_detector": job.get("source_detector"),
                "rectified_image_path": job.get("rectified_image_path"),
                "hamer_frame_path": job.get("hamer_frame_path"),
                "used_mask_blur": bool(job.get("used_mask_blur")),
                "hamer_joints_cam": selected["joints"].tolist(),
                "hamer_vertices_cam": selected["verts"].tolist(),
                "hamer_cam_t": selected["cam_t"].tolist(),
                "hamer_pred_cam_t": selected["pred_cam_t"].tolist(),
                "bbox_scale": float(selected["bbox_scale"]),
                "mask_score": selected.get("mask_score"),
                "mask_iou": selected.get("mask_iou"),
                "rendered_overlay_path": str(render_path) if render_path else None,
                "hypothesis_status": selected["hypothesis_status"],
            }
            out.write(json.dumps(prediction, ensure_ascii=False, separators=(",", ":")) + "\n")
            stats["predictions"] += 1
            if args.progress_every > 0 and (index % args.progress_every == 0 or index == len(jobs)):
                progress.set_postfix_str(f"group={job['group_id']} pred={stats['predictions']} failed={stats['failed_jobs']}")

    with (args.output_dir / f"hamer_predictions_config_{suffix}{camera_suffix}.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "jobs": str(jobs_path),
                "output_path": str(output_path),
                "hamer_root": str(args.hamer_root),
                "checkpoint": str(checkpoint_path),
                "candidate_bbox_scales": scales,
                "stats": stats,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
        f.write("\n")
    print("Summary")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print(f"wrote: {output_path}")


if __name__ == "__main__":
    main()
