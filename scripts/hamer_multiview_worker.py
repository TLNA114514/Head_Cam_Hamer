#!/usr/bin/env python3
"""Run HaMeR on fused multi-view hand jobs and export structured predictions."""

from __future__ import annotations

import argparse
import builtins
import json
import os
import sys
import time
import warnings
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from hamer_multiview_utils import HAND_CONNECTIONS, DEFAULT_BASE_DIR, iter_jsonl, load_mask, parse_group_ids, range_suffix
from progress_utils import tqdm


from dependency_paths import DEFAULT_HAMER_ROOT
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
    parser.add_argument("--rectified-config", type=Path, help="Rectified cache config JSON with per-camera new intrinsics.")
    parser.add_argument("--group-range")
    parser.add_argument("--group-ids")
    parser.add_argument("--camera-id", help="Optional single camera shard.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--job-batch-size", type=int, default=8, help="Number of jobs to preprocess and pack together.")
    parser.add_argument("--rescale-factor", type=float, default=2.0)
    parser.add_argument("--candidate-bbox-scales", default="1.0,1.1,1.2")
    parser.add_argument(
        "--candidate-scale-policy",
        choices=["fixed", "mask-adaptive"],
        default="fixed",
        help="Use all configured scales, or collapse to the scale nearest 1.0 when no readable SAM3 mask exists.",
    )
    parser.add_argument("--precision", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--allow-tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--mask-scoring",
        choices=["all", "selection-only", "none"],
        default="all",
        help="Render candidate masks always, only when selection is ambiguous, or never.",
    )
    parser.add_argument(
        "--mask-score-method",
        choices=["mesh", "skeleton"],
        default="mesh",
        help="Score SAM3 overlap with a rendered mesh or a lightweight projected skeleton proxy.",
    )
    parser.add_argument("--compile-backbone", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--compile-mode",
        choices=["default", "reduce-overhead", "max-autotune"],
        default="reduce-overhead",
    )
    parser.add_argument(
        "--export-vertices",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write 778 MANO vertices per prediction. Disable for joint-only fusion to reduce JSONL I/O.",
    )
    parser.add_argument(
        "--export-mano-params",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write MANO rotation matrices and betas. Joint-only fusion does not require them.",
    )
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


def load_rectified_intrinsics(path: Path | None) -> tuple[dict[str, list[list[float]]], str | None]:
    if path is None or not path.exists():
        return {}, None
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    intrinsics = data.get("new_intrinsics") or {}
    return {str(key): value for key, value in intrinsics.items()}, str(path)


def bbox_area_score(bbox: list[float] | np.ndarray | None, image_shape: tuple[int, int, int]) -> float:
    if bbox is None or len(bbox) != 4:
        return 0.0
    img_h, img_w = image_shape[:2]
    x1, y1, x2, y2 = [float(v) for v in bbox]
    area_ratio = max(0.0, x2 - x1) * max(0.0, y2 - y1) / float(max(1, img_w * img_h))
    if area_ratio <= 0.0:
        return 0.0
    if area_ratio < 0.003:
        return float(np.clip(area_ratio / 0.003, 0.0, 1.0))
    if area_ratio > 0.12:
        return float(np.clip(0.12 / area_ratio, 0.0, 1.0))
    return 1.0


def bbox_edge_score(bbox: list[float] | np.ndarray | None, image_shape: tuple[int, int, int]) -> float:
    if bbox is None or len(bbox) != 4:
        return 0.0
    img_h, img_w = image_shape[:2]
    x1, y1, x2, y2 = [float(v) for v in bbox]
    margin = min(x1, y1, float(img_w) - x2, float(img_h) - y2)
    return float(np.clip((margin + 20.0) / 80.0, 0.0, 1.0))


def keypoints_2d_to_full_image(
    pred_keypoints_2d: torch.Tensor,
    box_center: torch.Tensor,
    box_size: torch.Tensor,
    is_right: int,
) -> np.ndarray:
    points = pred_keypoints_2d.detach().cpu().float().clone()
    points[:, 0] = float(2 * int(is_right) - 1) * points[:, 0]
    center = box_center.detach().cpu().float()
    size = box_size.detach().cpu().float()
    return (points * size.reshape(1, 1) + center.reshape(1, 2)).numpy()


def joint_confidences(
    points: np.ndarray,
    bbox: list[float] | np.ndarray,
    mask: np.ndarray | None,
    mask_score: float | None,
    image_shape: tuple[int, int, int],
) -> list[float]:
    img_h, img_w = image_shape[:2]
    bbox_score = bbox_area_score(bbox, image_shape)
    edge_score = bbox_edge_score(bbox, image_shape)
    mask_component = float(np.clip(mask_score if mask_score is not None else 0.35, 0.0, 1.0))
    base = 0.20 + 0.35 * bbox_score + 0.20 * edge_score + 0.25 * mask_component
    confidences = []
    for x, y in points[:21]:
        in_bounds = 0.0 <= float(x) < float(img_w) and 0.0 <= float(y) < float(img_h)
        if not in_bounds:
            confidences.append(0.0)
            continue
        mask_factor = 1.0
        if mask is not None:
            xi = int(round(float(x)))
            yi = int(round(float(y)))
            xi = int(np.clip(xi, 0, img_w - 1))
            yi = int(np.clip(yi, 0, img_h - 1))
            mask_factor = 1.0 if bool(mask[yi, xi]) else 0.45
        confidences.append(float(np.clip(base * mask_factor, 0.0, 1.0)))
    return confidences


def set_mask_overlap_score(candidate: dict[str, Any], reference_mask: np.ndarray, candidate_mask: np.ndarray) -> None:
    inter = float(np.logical_and(reference_mask, candidate_mask).sum())
    union = float(np.logical_or(reference_mask, candidate_mask).sum())
    sam_area = float(reference_mask.sum())
    candidate_area = float(candidate_mask.sum())
    iou = inter / union if union > 0 else 0.0
    sam_cov = inter / sam_area if sam_area > 0 else 0.0
    candidate_cov = inter / candidate_area if candidate_area > 0 else 0.0
    candidate["mask_iou"] = iou
    candidate["mask_sam_coverage"] = sam_cov
    candidate["mask_mesh_coverage"] = candidate_cov
    candidate["mask_score"] = 0.45 * iou + 0.45 * sam_cov + 0.10 * candidate_cov


def score_candidates_with_mesh_mask(
    candidates: list[dict[str, Any]],
    mask: np.ndarray | None,
    renderer: Any,
    image_shape: tuple[int, int, int],
    focal_length: Any,
) -> None:
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
        set_mask_overlap_score(candidate, mask, mesh_mask)


def score_candidates_with_skeleton_mask(
    candidates: list[dict[str, Any]],
    mask: np.ndarray | None,
    image_shape: tuple[int, int, int],
) -> None:
    if mask is None:
        return
    image_height, image_width = image_shape[:2]
    palm_indices = np.asarray([0, 5, 9, 13, 17], dtype=np.int32)
    for candidate in candidates:
        points = np.asarray(candidate.get("keypoints_2d"), dtype=np.float32)[:21]
        if points.shape != (21, 2) or not np.isfinite(points).all():
            continue
        points[:, 0] = np.clip(points[:, 0], 0, image_width - 1)
        points[:, 1] = np.clip(points[:, 1], 0, image_height - 1)
        points_i = np.rint(points).astype(np.int32)
        thickness = max(3, int(round(float(candidate.get("bbox_size", 100.0)) * 0.035)))
        skeleton_mask = np.zeros((image_height, image_width), dtype=np.uint8)
        palm = cv2.convexHull(points_i[palm_indices])
        cv2.fillConvexPoly(skeleton_mask, palm, 1)
        for start, end in HAND_CONNECTIONS:
            cv2.line(
                skeleton_mask,
                tuple(points_i[start]),
                tuple(points_i[end]),
                1,
                thickness=thickness,
                lineType=cv2.LINE_AA,
            )
        joint_radius = max(2, thickness // 2)
        for point in points_i:
            cv2.circle(skeleton_mask, tuple(point), joint_radius, 1, thickness=-1, lineType=cv2.LINE_AA)
        set_mask_overlap_score(candidate, mask, skeleton_mask.astype(bool))


def detach_mano_params(pred: dict[str, Any], index: int) -> dict[str, Any]:
    params = pred.get("pred_mano_params") or {}
    out = {}
    for key in ("global_orient", "hand_pose", "betas"):
        value = params.get(key)
        if value is None:
            continue
        out[key] = value[index].detach().cpu().numpy().tolist()
    return out


def main() -> None:
    install_noise_filters()
    args = parse_args()
    if args.batch_size < 1 or args.job_batch_size < 1:
        raise SystemExit("--batch-size and --job-batch-size must be positive")
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
    if args.rectified_config is not None and not args.rectified_config.is_absolute():
        args.rectified_config = original_cwd / args.rectified_config
    rectified_intrinsics, rectified_intrinsics_source = load_rectified_intrinsics(args.rectified_config)

    args.hamer_root = args.hamer_root.expanduser().resolve()
    if not args.hamer_root.exists():
        raise SystemExit(f"HaMeR root not found: {args.hamer_root}. Run scripts/setup.sh first.")
    sys.path.insert(0, str(args.hamer_root))

    from hamer.datasets.vitdet_dataset import ViTDetDataset  # type: ignore
    from hamer.models import load_hamer  # type: ignore
    from hamer.utils import recursive_to  # type: ignore
    from hamer.utils.renderer import Renderer, cam_crop_to_full  # type: ignore

    checkpoint_path = resolve_hamer_checkpoint(args.hamer_root, args.checkpoint)
    model_config_path = checkpoint_path.parent.parent / "model_config.yaml"
    if not checkpoint_path.exists():
        raise SystemExit(f"HaMeR checkpoint not found: {checkpoint_path}. Run scripts/setup.sh first.")
    if not model_config_path.exists():
        raise SystemExit(f"HaMeR model_config.yaml not found next to checkpoint: {model_config_path}")
    print(f"[HaMeR {args.camera_id or 'all'}] loading model checkpoint={checkpoint_path}", flush=True)
    load_started = time.perf_counter()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    os.chdir(args.hamer_root)
    try:
        model, model_cfg = load_hamer(str(checkpoint_path), map_location=device)
    finally:
        os.chdir(original_cwd)
    model = model.to(device).eval()
    eager_backbone = model.backbone
    compile_requested = bool(args.compile_backbone)
    compile_enabled = False
    compile_error = None
    if compile_requested:
        try:
            if device.type != "cuda":
                raise RuntimeError("backbone compile is enabled only on CUDA")
            if not hasattr(torch, "compile"):
                raise RuntimeError("torch.compile is unavailable")
            model.backbone = torch.compile(eager_backbone, mode=args.compile_mode)
            compile_enabled = True
        except Exception as exc:
            compile_error = f"{type(exc).__name__}: {exc}"
            print(f"[HaMeR] backbone compile disabled: {compile_error}", flush=True)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(args.allow_tf32)
        torch.backends.cudnn.allow_tf32 = bool(args.allow_tf32)
    renderer_required = bool(
        args.save_rendered_overlay
        or (args.mask_scoring != "none" and args.mask_score_method == "mesh")
    )
    renderer = Renderer(model_cfg, faces=model.mano.faces) if renderer_required else None
    model_load_seconds = time.perf_counter() - load_started
    scales = parse_float_list(args.candidate_bbox_scales)
    jobs = iter_jobs(jobs_path, args.camera_id, args.include_debug_only, group_ids)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    render_dir = args.output_dir / "rendered"
    stats = {
        "jobs": len(jobs),
        "predictions": 0,
        "failed_jobs": 0,
        "unknown_jobs": 0,
        "mask_adaptive_jobs": 0,
        "mask_adaptive_candidates_avoided": 0,
        "image_reads": 0,
        "image_cache_hits": 0,
        "candidate_samples": 0,
        "model_forward_batches": 0,
    }
    timing = {
        "model_load_seconds": model_load_seconds,
        "image_and_dataset_setup_seconds": 0.0,
        "model_and_output_seconds": 0.0,
        "mask_scoring_seconds": 0.0,
        "overlay_seconds": 0.0,
        "serialize_seconds": 0.0,
    }
    autocast_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}.get(args.precision)
    need_candidate_vertices = bool(
        args.export_vertices
        or args.save_rendered_overlay
        or (args.mask_scoring != "none" and args.mask_score_method == "mesh")
    )
    inference_started = time.perf_counter()

    print(f"[HaMeR {args.camera_id or 'all'}] start jobs={len(jobs)} output={output_path}", flush=True)
    with output_path.open("w", encoding="utf-8") as out:
        progress = tqdm(total=len(jobs), desc=f"HaMeR {args.camera_id or 'all'}", unit="job", position=args.progress_position)
        for job_start in range(0, len(jobs), args.job_batch_size):
            contexts: list[dict[str, Any]] = []
            samples: list[dict[str, Any]] = []
            sample_refs: list[tuple[int, dict[str, Any]]] = []
            image_cache: dict[Path, np.ndarray] = {}
            for job_offset, job in enumerate(jobs[job_start : job_start + args.job_batch_size]):
                index = job_start + job_offset + 1
                preprocess_started = time.perf_counter()
                frame_path = Path(job["hamer_frame_path"])
                img_cv2 = image_cache.get(frame_path)
                if img_cv2 is None:
                    img_cv2 = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
                    stats["image_reads"] += 1
                    if img_cv2 is not None:
                        image_cache[frame_path] = img_cv2
                else:
                    stats["image_cache_hits"] += 1
                if img_cv2 is None:
                    stats["failed_jobs"] += 1
                    progress.update(1)
                    continue
                sam_mask = load_mask(
                    job.get("sam3_mask_path"),
                    image_size=(img_cv2.shape[1], img_cv2.shape[0]),
                )
                job_scales = scales
                scale_scoring_available = sam_mask is not None and args.mask_scoring != "none"
                if args.candidate_scale_policy == "mask-adaptive" and not scale_scoring_available and len(scales) > 1:
                    job_scales = [min(scales, key=lambda scale: abs(float(scale) - 1.0))]
                    side_count = 1 if job.get("handedness") in {"Left", "Right"} else 2
                    stats["mask_adaptive_jobs"] += 1
                    stats["mask_adaptive_candidates_avoided"] += side_count * (len(scales) - len(job_scales))
                records = candidate_records(job, img_cv2.shape, job_scales)
                stats["candidate_samples"] += len(records)
                if job.get("handedness") == "unknown":
                    stats["unknown_jobs"] += 1
                boxes = np.stack([item["bbox"] for item in records])
                right = np.asarray([item["is_right"] for item in records], dtype=np.int64)
                dataset = ViTDetDataset(model_cfg, img_cv2, boxes, right, rescale_factor=args.rescale_factor)
                context_index = len(contexts)
                contexts.append(
                    {"index": index, "job": job, "image": img_cv2, "sam_mask": sam_mask, "candidates": []}
                )
                for record_index, record in enumerate(records):
                    samples.append(dataset[record_index])
                    sample_refs.append((context_index, record))
                timing["image_and_dataset_setup_seconds"] += time.perf_counter() - preprocess_started

            dataloader = torch.utils.data.DataLoader(samples, batch_size=args.batch_size, shuffle=False, num_workers=0)
            record_offset = 0
            for batch in dataloader:
                stats["model_forward_batches"] += 1
                batch = recursive_to(batch, device)
                forward_started = time.perf_counter()
                autocast_context = (
                    torch.autocast(device_type="cuda", dtype=autocast_dtype)
                    if device.type == "cuda" and autocast_dtype is not None
                    else nullcontext()
                )
                with torch.inference_mode(), autocast_context:
                    try:
                        pred = model(batch)
                    except Exception as exc:
                        if not compile_enabled:
                            raise
                        compile_error = f"runtime {type(exc).__name__}: {exc}"
                        print(f"[HaMeR] compiled backbone failed; retrying eager: {compile_error}", flush=True)
                        model.backbone = eager_backbone
                        compile_enabled = False
                        pred = model(batch)
                multiplier = 2 * batch["right"] - 1
                raw_pred_cam = pred["pred_cam"]
                pred_cam = torch.stack(
                    [raw_pred_cam[:, 0], multiplier * raw_pred_cam[:, 1], raw_pred_cam[:, 2]], dim=-1
                )
                box_center = batch["box_center"].float()
                box_size = batch["box_size"].float()
                img_size = batch["img_size"].float()
                scaled_focal_lengths = (
                    model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max(dim=1).values
                )
                pred_cam_t_full = cam_crop_to_full(
                    pred_cam, box_center, box_size, img_size, scaled_focal_lengths
                ).detach().cpu().numpy()

                current_batch_size = batch["img"].shape[0]
                for sample_index in range(current_batch_size):
                    context_index, record = sample_refs[record_offset + sample_index]
                    is_right = int(batch["right"][sample_index].detach().cpu().item())
                    verts = (
                        pred["pred_vertices"][sample_index].detach().cpu().numpy()
                        if need_candidate_vertices
                        else None
                    )
                    joints = pred["pred_keypoints_3d"][sample_index].detach().cpu().numpy()
                    keypoints_2d = keypoints_2d_to_full_image(
                        pred["pred_keypoints_2d"][sample_index],
                        box_center[sample_index],
                        box_size[sample_index],
                        is_right,
                    )
                    if verts is not None:
                        verts[:, 0] = (2 * is_right - 1) * verts[:, 0]
                    joints[:, 0] = (2 * is_right - 1) * joints[:, 0]
                    contexts[context_index]["candidates"].append(
                        {
                            "handedness": record["handedness"],
                            "is_right": is_right,
                            "hypothesis_status": record["hypothesis_status"],
                            "bbox": record["bbox"],
                            "bbox_scale": record["bbox_scale"],
                            "verts": verts,
                            "joints": joints,
                            "cam_t": pred_cam_t_full[sample_index],
                            "pred_cam_t": pred["pred_cam_t"][sample_index].detach().cpu().numpy(),
                            "keypoints_2d": keypoints_2d,
                            "bbox_center": box_center[sample_index].detach().cpu().numpy(),
                            "bbox_size": float(box_size[sample_index].detach().cpu().item()),
                            "scaled_focal_length": float(scaled_focal_lengths[sample_index].detach().cpu().item()),
                            "mano_params_rotmat": (
                                detach_mano_params(pred, sample_index) if args.export_mano_params else None
                            ),
                            "mask_score": None,
                        }
                    )
                timing["model_and_output_seconds"] += time.perf_counter() - forward_started
                record_offset += current_batch_size

            for context in contexts:
                index = int(context["index"])
                job = context["job"]
                img_cv2 = context["image"]
                candidates = context["candidates"]
                mask_started = time.perf_counter()
                should_score_mask = args.mask_scoring == "all" or (
                    args.mask_scoring == "selection-only" and len(candidates) > 1
                )
                if should_score_mask:
                    if args.mask_score_method == "mesh":
                        score_candidates_with_mesh_mask(
                            candidates,
                            context["sam_mask"],
                            renderer,
                            img_cv2.shape,
                            candidates[0]["scaled_focal_length"] if candidates else None,
                        )
                    else:
                        score_candidates_with_skeleton_mask(candidates, context["sam_mask"], img_cv2.shape)
                timing["mask_scoring_seconds"] += time.perf_counter() - mask_started
                selected = choose_candidate(candidates)
                if selected is None:
                    stats["failed_jobs"] += 1
                    progress.update(1)
                    continue

                render_path = None
                if args.save_rendered_overlay:
                    overlay_started = time.perf_counter()
                    try:
                        rgba = renderer.render_rgba_multiple(
                            [selected["verts"]],
                            cam_t=[selected["cam_t"]],
                            render_res=[img_cv2.shape[1], img_cv2.shape[0]],
                            is_right=[selected["is_right"]],
                            focal_length=selected["scaled_focal_length"],
                        )
                        input_img = img_cv2.astype(np.float32)[:, :, ::-1] / 255.0
                        input_img = np.concatenate([input_img, np.ones_like(input_img[:, :, :1])], axis=2)
                        overlay = input_img[:, :, :3] * (1 - rgba[:, :, 3:]) + rgba[:, :, :3] * rgba[:, :, 3:]
                        render_path = render_dir / str(job["camera_id"]) / f"{int(job['group_id']):08d}_{int(job['job_index']):02d}.jpg"
                        render_path.parent.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(render_path), 255 * overlay[:, :, ::-1])
                    except Exception as exc:
                        print(f"render failed for job {job.get('group_id')} {job.get('camera_id')}: {exc}", flush=True)
                    timing["overlay_seconds"] += time.perf_counter() - overlay_started

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
                    "sam3_mask_path": job.get("sam3_mask_path"),
                    "sam3_score": job.get("sam3_score"),
                    "track_id": job.get("track_id"),
                    "locked_handedness": job.get("locked_handedness"),
                    "handedness_confidence": job.get("handedness_confidence"),
                    "hamer_joints_cam": selected["joints"].tolist(),
                    "hamer_vertices_cam": selected["verts"].tolist() if args.export_vertices else None,
                    "hamer_joints_2d_rectified_px": selected["keypoints_2d"][:21].tolist(),
                    "hamer_joints_2d_conf": joint_confidences(
                        selected["keypoints_2d"],
                        selected.get("bbox", job["bbox_rectified_px"]),
                        context["sam_mask"],
                        selected.get("mask_score"),
                        img_cv2.shape,
                    ),
                    "hamer_cam_t": selected["cam_t"].tolist(),
                    "hamer_pred_cam_t": selected["pred_cam_t"].tolist(),
                    "hamer_focal_length": selected["scaled_focal_length"],
                    "hamer_bbox_center": selected["bbox_center"].tolist(),
                    "hamer_bbox_size": float(selected["bbox_size"]),
                    "mano_params_rotmat": selected.get("mano_params_rotmat"),
                    "mano_param_source": "hamer" if selected.get("mano_params_rotmat") else None,
                    "bbox_scale": float(selected["bbox_scale"]),
                    "rectified_K": rectified_intrinsics.get(str(job["camera_id"])),
                    "rectified_K_source": rectified_intrinsics_source,
                    "mask_score": selected.get("mask_score"),
                    "mask_iou": selected.get("mask_iou"),
                    "candidate_mask_scoring": args.mask_scoring,
                    "candidate_mask_score_method": args.mask_score_method,
                    "candidate_scale_policy": args.candidate_scale_policy,
                    "rendered_overlay_path": str(render_path) if render_path else None,
                    "hypothesis_status": selected["hypothesis_status"],
                }
                serialize_started = time.perf_counter()
                out.write(json.dumps(prediction, ensure_ascii=False, separators=(",", ":")) + "\n")
                timing["serialize_seconds"] += time.perf_counter() - serialize_started
                stats["predictions"] += 1
                progress.update(1)
                if args.progress_every > 0 and (index % args.progress_every == 0 or index == len(jobs)):
                    progress.set_postfix_str(f"group={job['group_id']} pred={stats['predictions']} failed={stats['failed_jobs']}")
        progress.close()

    inference_seconds = time.perf_counter() - inference_started
    timing["inference_total_seconds"] = inference_seconds
    timing["jobs_per_second"] = len(jobs) / inference_seconds if inference_seconds > 0.0 else None
    timing["predictions_per_second"] = stats["predictions"] / inference_seconds if inference_seconds > 0.0 else None
    with (args.output_dir / f"hamer_predictions_config_{suffix}{camera_suffix}.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "jobs": str(jobs_path),
                "output_path": str(output_path),
                "hamer_root": str(args.hamer_root),
                "checkpoint": str(checkpoint_path),
                "candidate_bbox_scales": scales,
                "candidate_scale_policy": args.candidate_scale_policy,
                "batch_size": args.batch_size,
                "job_batch_size": args.job_batch_size,
                "precision": args.precision,
                "effective_precision": args.precision if device.type == "cuda" else "float32",
                "allow_tf32": args.allow_tf32,
                "mask_scoring": args.mask_scoring,
                "mask_score_method": args.mask_score_method,
                "compile_backbone_requested": compile_requested,
                "compile_backbone_enabled": compile_enabled,
                "compile_mode": args.compile_mode,
                "compile_error": compile_error,
                "export_vertices": args.export_vertices,
                "export_mano_params": args.export_mano_params,
                "candidate_vertices_materialized": need_candidate_vertices,
                "save_rendered_overlay": args.save_rendered_overlay,
                "renderer_initialized": renderer_required,
                "timing": timing,
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
    print(f"  model_load_seconds: {timing['model_load_seconds']:.3f}")
    print(f"  inference_total_seconds: {timing['inference_total_seconds']:.3f}")
    print(f"  jobs_per_second: {timing['jobs_per_second']:.3f}")
    print(f"wrote: {output_path}")


if __name__ == "__main__":
    main()
