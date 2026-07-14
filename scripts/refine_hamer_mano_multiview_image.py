#!/usr/bin/env python3
"""Refine HaMeR MANO pose/beta with multi-view image-space 2D and SAM3 mask losses."""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml

from hamer_multiview_utils import (
    DEFAULT_BASE_DIR,
    DEFAULT_CALIB,
    HAND_CONNECTIONS,
    PRIMARY_CAMERAS,
    iter_jsonl,
    load_mask,
    parse_group_ids,
    range_suffix,
)
from progress_utils import tqdm


from dependency_paths import DEFAULT_HAMER_ROOT
LANDMARK_NAMES = [
    "wrist",
    "thumb_cmc",
    "thumb_mcp",
    "thumb_ip",
    "thumb_tip",
    "index_mcp",
    "index_pip",
    "index_dip",
    "index_tip",
    "middle_mcp",
    "middle_pip",
    "middle_dip",
    "middle_tip",
    "ring_mcp",
    "ring_pip",
    "ring_dip",
    "ring_tip",
    "pinky_mcp",
    "pinky_pip",
    "pinky_dip",
    "pinky_tip",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", action="append", type=Path)
    parser.add_argument("--predictions-glob", default=str(DEFAULT_BASE_DIR / "hamer_per_view" / "hamer_predictions_*.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_BASE_DIR / "hamer_mano_multiview_refined")
    parser.add_argument("--hamer-root", type=Path, default=DEFAULT_HAMER_ROOT)
    parser.add_argument("--checkpoint")
    parser.add_argument("--calib", type=Path, default=DEFAULT_CALIB)
    parser.add_argument("--rectified-config", type=Path)
    parser.add_argument("--projection-correction", type=Path, help="Optional per-camera rectified-pixel affine correction JSON.")
    parser.add_argument("--rectify-focal-scale", type=float, default=0.30)
    parser.add_argument(
        "--global-initialization",
        choices=["hamer-virtual", "physical-pnp"],
        default="physical-pnp",
        help="Global pose initializer. physical-pnp uses rectified physical intrinsics and is the recommended image-space path.",
    )
    parser.add_argument("--group-range")
    parser.add_argument("--group-ids")
    parser.add_argument("--image-width", type=int, default=1600)
    parser.add_argument("--image-height", type=int, default=1200)
    parser.add_argument("--use-mediapipe-2d", choices=["never", "weak", "auto"], default="never")
    parser.add_argument("--mediapipe", type=Path)
    parser.add_argument("--mediapipe-min-score", type=float, default=0.85)
    parser.add_argument("--mediapipe-loss-weight", type=float, default=0.12)
    parser.add_argument("--window-size", type=int, default=7)
    parser.add_argument("--optimize-mano-pose", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--optimize-mano-betas", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-frame-beta-delta", action="store_true")
    parser.add_argument("--max-frame-beta-delta-norm", type=float, default=0.12)
    parser.add_argument("--max-iters", type=int, default=70)
    parser.add_argument("--beta-iters", type=int, default=100)
    parser.add_argument("--max-beta-observations", type=int, default=400)
    parser.add_argument(
        "--beta-estimation-space",
        choices=["hamer-local", "image-2d"],
        default="hamer-local",
        help="Sequence beta estimator. image-2d adds physical-camera multi-view reprojection refinement after the HaMeR-local initialization.",
    )
    parser.add_argument("--image-beta-max-observations", type=int, default=240)
    parser.add_argument("--image-beta-prior-weight", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=0.018)
    parser.add_argument("--beta-learning-rate", type=float, default=0.025)
    parser.add_argument("--keypoint-loss-weight", type=float, default=1.0)
    parser.add_argument("--mask-loss-weight", type=float, default=0.18)
    parser.add_argument(
        "--min-readable-sam3-mask-ratio",
        type=float,
        default=0.50,
        help="Fail image refinement when too few input predictions carry a readable SAM3 mask; set 0 to allow mask-free legacy predictions.",
    )
    parser.add_argument(
        "--mask-boundary-loss-weight",
        type=float,
        default=0.0,
        help="Experimental symmetric SAM3-boundary-to-mesh loss; 0 keeps the established one-way mask loss.",
    )
    parser.add_argument("--mask-boundary-samples", type=int, default=128)
    parser.add_argument("--mask-boundary-softmin-px", type=float, default=10.0)
    parser.add_argument("--mask-boundary-vertex-subsample", type=int, default=4)
    parser.add_argument("--pose-prior-weight", type=float, default=0.30)
    parser.add_argument("--beta-prior-weight", type=float, default=1.0)
    parser.add_argument("--temporal-pose-weight", type=float, default=0.42)
    parser.add_argument("--temporal-joint-weight", type=float, default=0.20)
    parser.add_argument(
        "--temporal-acceleration-weight",
        type=float,
        default=0.0,
        help="Optional second-order local-pose prior. Keep 0 until validated on a sequence with held-out GT.",
    )
    parser.add_argument("--global-prior-weight", type=float, default=0.02)
    parser.add_argument("--keypoint-cauchy-scale-px", type=float, default=18.0)
    parser.add_argument("--mask-cauchy-scale-px", type=float, default=28.0)
    parser.add_argument("--max-reprojection-error-px", type=float, default=36.0)
    parser.add_argument("--max-mask-distance-px", type=float, default=32.0)
    parser.add_argument("--soft-reprojection-error-px", type=float, default=90.0)
    parser.add_argument("--soft-mask-distance-px", type=float, default=90.0)
    parser.add_argument("--min-view-weight", type=float, default=0.12)
    parser.add_argument("--anchor-view-weight", type=float, default=1.45)
    parser.add_argument("--min-soft-used-weight", type=float, default=0.08)
    parser.add_argument("--min-metric-used-weight", type=float, default=0.28)
    parser.add_argument(
        "--primary-anchor-score-margin",
        type=float,
        default=0.08,
        help="With PnP gating, keep C1/C2 as anchor unless another view's quality score exceeds it by this margin.",
    )
    parser.add_argument(
        "--pnp-view-gate-m",
        type=float,
        default=0.04,
        help="Reject auxiliary views whose physical-K PnP wrist differs from the anchor; 0 disables the gate.",
    )
    parser.add_argument("--vertex-subsample", type=int, default=14)
    parser.add_argument("--min-candidate-score", type=float, default=-0.15)
    parser.add_argument("--primary-prior-bonus", type=float, default=0.04)
    parser.add_argument("--temporal-error-cap-m", type=float, default=0.10)
    parser.add_argument("--temporal-reject-threshold-m", type=float, default=0.16)
    parser.add_argument("--smoothing-alpha", type=float, default=0.65)
    parser.add_argument("--debug-every", type=int, default=25)
    parser.add_argument("--save-debug-overlays", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--cuda-cache-clear-every",
        type=int,
        default=1,
        help="Release cached CUDA allocations every N completed frames; 0 disables it.",
    )
    parser.add_argument("--progress-position", type=int, default=int(os.environ.get("TQDM_POSITION", "0")))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def finite_array(value: Any, shape_tail: tuple[int, ...] | None = None) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64)
    if shape_tail is not None and tuple(arr.shape[-len(shape_tail):]) != shape_tail:
        return None
    if not np.all(np.isfinite(arr)):
        return None
    return arr


def is_xyz_list(value: Any) -> bool:
    return isinstance(value, list) and len(value) > 0 and isinstance(value[0], list) and len(value[0]) == 3


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def weighted_median(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    out = []
    weights = np.maximum(np.asarray(weights, dtype=np.float64), 1e-8)
    for dim in range(values.shape[1]):
        order = np.argsort(values[:, dim])
        sorted_values = values[order, dim]
        sorted_weights = weights[order]
        cutoff = 0.5 * sorted_weights.sum()
        index = int(np.searchsorted(np.cumsum(sorted_weights), cutoff, side="left"))
        out.append(float(sorted_values[min(index, len(sorted_values) - 1)]))
    return np.asarray(out, dtype=np.float64)


def local_points(points: Any) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float64)
    return arr - arr[0:1]


def palm_frame_np(joints: np.ndarray) -> np.ndarray:
    wrist = joints[0]
    x_axis = joints[5] - wrist
    y_hint = joints[17] - wrist
    if np.linalg.norm(x_axis) < 1e-8:
        x_axis = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        x_axis = x_axis / np.linalg.norm(x_axis)
    z_axis = np.cross(x_axis, y_hint)
    if np.linalg.norm(z_axis) < 1e-8:
        z_axis = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        z_axis = z_axis / np.linalg.norm(z_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= max(np.linalg.norm(y_axis), 1e-8)
    return np.stack([x_axis, y_axis, z_axis], axis=1)


def palm_local_np(points: np.ndarray, joints_for_frame: np.ndarray | None = None) -> np.ndarray:
    joints = points if joints_for_frame is None else joints_for_frame
    basis = palm_frame_np(joints)
    return (points - joints[0:1]) @ basis


def palm_local_torch(points: torch.Tensor, joints_for_frame: torch.Tensor | None = None) -> torch.Tensor:
    joints = points if joints_for_frame is None else joints_for_frame
    wrist = joints[:, 0:1, :]
    x_axis = torch.nn.functional.normalize(joints[:, 5, :] - joints[:, 0, :], dim=-1, eps=1e-8)
    y_hint = joints[:, 17, :] - joints[:, 0, :]
    z_axis = torch.nn.functional.normalize(torch.cross(x_axis, y_hint, dim=-1), dim=-1, eps=1e-8)
    y_axis = torch.nn.functional.normalize(torch.cross(z_axis, x_axis, dim=-1), dim=-1, eps=1e-8)
    basis = torch.stack([x_axis, y_axis, z_axis], dim=-1)
    return torch.matmul(points - wrist, basis)


def kabsch_transform(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    n = min(len(source), len(target))
    src = source[:n]
    tgt = target[:n]
    src_c = src.mean(axis=0)
    tgt_c = tgt.mean(axis=0)
    h = (src - src_c).T @ (tgt - tgt_c)
    u, _s, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1] *= -1
        r = vt.T @ u.T
    aligned = (source - src_c) @ r.T + tgt_c
    err = float(np.mean(np.linalg.norm(aligned[:n] - target[:n], axis=1)))
    return aligned, r, src_c, tgt_c, err


def rotmat_to_6d(rotmat: torch.Tensor) -> torch.Tensor:
    return torch.cat([rotmat[..., :, 0], rotmat[..., :, 1]], dim=-1)


def rot6d_to_rotmat(value: torch.Tensor) -> torch.Tensor:
    a1 = value[..., 0:3]
    a2 = value[..., 3:6]
    b1 = torch.nn.functional.normalize(a1, dim=-1, eps=1e-8)
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = torch.nn.functional.normalize(b2, dim=-1, eps=1e-8)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def smooth_rotmat(current: np.ndarray, previous: np.ndarray | None, alpha: float) -> np.ndarray:
    if previous is None:
        return current
    current_6d = rotmat_to_6d(torch.tensor(current, dtype=torch.float32))
    previous_6d = rotmat_to_6d(torch.tensor(previous, dtype=torch.float32))
    mixed = float(alpha) * current_6d + (1.0 - float(alpha)) * previous_6d
    return rot6d_to_rotmat(mixed).detach().cpu().numpy()


def cauchy_loss(value: torch.Tensor, scale: float) -> torch.Tensor:
    return torch.log1p((value / max(float(scale), 1e-8)) ** 2)


def weighted_torch_mean(values: list[torch.Tensor], weights: list[torch.Tensor], device: torch.device) -> torch.Tensor:
    if not values:
        return torch.tensor(0.0, dtype=torch.float32, device=device)
    stacked_values = torch.stack(values)
    stacked_weights = torch.stack(weights).to(dtype=stacked_values.dtype, device=device)
    return (stacked_values * stacked_weights).sum() / torch.clamp(stacked_weights.sum(), min=1e-6)


def robust_scalar_weight(value: float | None, scale: float) -> float:
    if value is None:
        return 1.0
    if not math.isfinite(float(value)):
        return 0.0
    normalized = max(float(value), 0.0) / max(float(scale), 1e-8)
    return float(1.0 / (1.0 + normalized * normalized))


def observation_quality_weight(item: dict[str, Any], camera_id: str, anchor_camera: str, args: argparse.Namespace) -> float:
    score = max(float(item.get("_score", 0.0)), float(args.min_view_weight))
    if item.get("_temporal_pose_error_m") is not None:
        temporal = max(float(item["_temporal_pose_error_m"]), 0.0)
        score *= robust_scalar_weight(temporal, args.temporal_error_cap_m)
    if camera_id == anchor_camera:
        score *= float(args.anchor_view_weight)
    return float(max(score, args.min_view_weight))


def resolve_hamer_checkpoint(hamer_root: Path, checkpoint: str | None) -> Path:
    if checkpoint:
        path = Path(checkpoint).expanduser()
        if not path.is_absolute():
            path = hamer_root / path
    else:
        path = hamer_root / "_DATA" / "hamer_ckpts" / "checkpoints" / "hamer.ckpt"
    return path.resolve()


def load_rectified_intrinsics(rectified_config: Path | None, calib_path: Path, rectify_focal_scale: float) -> dict[str, np.ndarray]:
    if rectified_config and rectified_config.exists():
        with rectified_config.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return {camera_id: np.asarray(k, dtype=np.float64) for camera_id, k in (data.get("new_intrinsics") or {}).items()}
    with calib_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    out = {}
    for camera_id, cam in data["cameras"].items():
        k = np.asarray(cam["intrinsics"], dtype=np.float64).copy()
        k[0, 0] *= rectify_focal_scale
        k[1, 1] *= rectify_focal_scale
        out[camera_id] = k
    return out


def load_camera_geometry(calib_path: Path, intrinsics: dict[str, np.ndarray]) -> dict[str, dict[str, np.ndarray]]:
    with calib_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    cameras = {}
    for camera_id, cam in data["cameras"].items():
        if camera_id not in intrinsics:
            continue
        t_h_c = np.asarray(cam["T_H_C"], dtype=np.float64)
        cameras[camera_id] = {
            "K": intrinsics[camera_id],
            "R_H_C": t_h_c[:3, :3],
            "t_H_C": t_h_c[:3, 3],
        }
    return cameras


def load_projection_corrections(path: Path | None) -> dict[str, np.ndarray]:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    raw = data.get("camera_projection_corrections") or data.get("projection_corrections") or {}
    corrections = {}
    for camera_id, item in raw.items():
        affine = item.get("affine_projected_to_rectified_px") if isinstance(item, dict) else item
        arr = np.asarray(affine, dtype=np.float64)
        if arr.shape == (2, 3) and np.all(np.isfinite(arr)):
            corrections[str(camera_id)] = arr
    return corrections


def apply_projection_correction_torch(xy: torch.Tensor, affine: torch.Tensor | None) -> torch.Tensor:
    if affine is None:
        return xy
    ones = torch.ones((xy.shape[0], 1), dtype=xy.dtype, device=xy.device)
    homo = torch.cat([xy, ones], dim=1)
    return homo @ affine.transpose(0, 1)


def apply_projection_correction_np(xy: np.ndarray, affine: np.ndarray | None) -> np.ndarray:
    if affine is None:
        return xy
    homo = np.concatenate([xy, np.ones((xy.shape[0], 1), dtype=xy.dtype)], axis=1)
    return homo @ affine.T


def undo_projection_correction_np(xy: np.ndarray, affine: np.ndarray | None) -> np.ndarray:
    """Map corrected rectified pixels back to the physical-K pixel plane."""
    if affine is None:
        return xy
    linear = np.asarray(affine[:, :2], dtype=np.float64)
    if abs(float(np.linalg.det(linear))) < 1e-10:
        return xy
    return (xy - affine[:, 2].reshape(1, 2)) @ np.linalg.inv(linear).T


def bbox_area_score(bbox: list[float] | None, image_width: int, image_height: int) -> float:
    if not bbox or len(bbox) != 4:
        return 0.0
    x1, y1, x2, y2 = [float(v) for v in bbox]
    area_ratio = max(0.0, x2 - x1) * max(0.0, y2 - y1) / float(max(1, image_width * image_height))
    if area_ratio <= 0.0:
        return 0.0
    if area_ratio < 0.003:
        return clamp01(area_ratio / 0.003)
    if area_ratio > 0.12:
        return clamp01(0.12 / area_ratio)
    return 1.0


def bbox_edge_score(bbox: list[float] | None, image_width: int, image_height: int) -> float:
    if not bbox or len(bbox) != 4:
        return 0.0
    x1, y1, x2, y2 = [float(v) for v in bbox]
    margin = min(x1, y1, float(image_width) - x2, float(image_height) - y2)
    return clamp01((margin + 20.0) / 80.0)


def source_bonus(source: str | None) -> float:
    if source == "mediapipe+sam3":
        return 0.06
    if source in {"sam3", "mediapipe"}:
        return 0.03
    return 0.0


def derive_2d_from_hamer_camera(rec: dict[str, Any], args: argparse.Namespace) -> np.ndarray | None:
    joints = finite_array(rec.get("hamer_joints_cam"), (3,))
    cam_t = finite_array(rec.get("hamer_cam_t"))
    if joints is None or cam_t is None or cam_t.shape != (3,):
        return None
    points = joints[:21] + cam_t.reshape(1, 3)
    z = np.maximum(points[:, 2], 1e-5)
    k = finite_array(rec.get("rectified_K"), (3, 3))
    if k is None:
        k = getattr(args, "_rectified_intrinsics", {}).get(str(rec.get("camera_id")))
    if k is not None:
        fx, fy, cx, cy = k[0, 0], k[1, 1], k[0, 2], k[1, 2]
    else:
        focal = rec.get("hamer_focal_length")
        if not isinstance(focal, (int, float)) or not math.isfinite(float(focal)):
            return None
        fx = fy = float(focal)
        cx = args.image_width * 0.5
        cy = args.image_height * 0.5
    return np.stack([fx * points[:, 0] / z + cx, fy * points[:, 1] / z + cy], axis=1)


def in_mask_ratio(points: np.ndarray, mask_path: str | None, image_width: int, image_height: int) -> float | None:
    mask = load_mask(mask_path, image_size=(image_width, image_height))
    if mask is None:
        return None
    hits = 0
    valid = 0
    for x, y in points[:21]:
        if not (0 <= x < image_width and 0 <= y < image_height):
            continue
        valid += 1
        ix = min(max(int(round(x)), 0), image_width - 1)
        iy = min(max(int(round(y)), 0), image_height - 1)
        hits += int(mask[iy, ix])
    return hits / valid if valid else 0.0


def prediction_quality(rec: dict[str, Any], points_2d: np.ndarray, args: argparse.Namespace) -> tuple[float, dict[str, float]]:
    conf = finite_array(rec.get("hamer_joints_2d_conf"))
    conf_score = float(np.mean(conf[:21])) if conf is not None and len(conf) >= 21 else 0.45
    in_bounds = np.logical_and.reduce(
        (
            points_2d[:21, 0] >= 0,
            points_2d[:21, 0] < args.image_width,
            points_2d[:21, 1] >= 0,
            points_2d[:21, 1] < args.image_height,
        )
    )
    in_bounds_score = float(np.mean(in_bounds))
    mask_raw = rec.get("mask_score")
    mask_score = clamp01(float(mask_raw)) if isinstance(mask_raw, (int, float)) else 0.30
    mask_joint_ratio = in_mask_ratio(points_2d, rec.get("sam3_mask_path"), args.image_width, args.image_height)
    mask_joint_score = 0.45 if mask_joint_ratio is None else float(mask_joint_ratio)
    bbox_area = bbox_area_score(rec.get("bbox_rectified_px"), args.image_width, args.image_height)
    edge = bbox_edge_score(rec.get("bbox_rectified_px"), args.image_width, args.image_height)
    primary = args.primary_prior_bonus if rec.get("camera_id") == PRIMARY_CAMERAS.get(rec.get("handedness")) else 0.0
    known = 0.04 if rec.get("hypothesis_status") == "known" else 0.0
    score = (
        0.32 * conf_score
        + 0.23 * mask_score
        + 0.16 * mask_joint_score
        + 0.12 * bbox_area
        + 0.08 * edge
        + 0.06 * in_bounds_score
        + source_bonus(rec.get("source_detector"))
        + primary
        + known
        - 0.03 * abs(float(rec.get("bbox_scale", 1.0)) - 1.0)
    )
    return float(score), {
        "2d_conf": conf_score,
        "mask": mask_score,
        "2d_in_mask": mask_joint_score,
        "bbox_area": bbox_area,
        "edge": edge,
        "in_bounds": in_bounds_score,
        "primary_prior": float(primary),
        "known": float(known),
    }


def candidate_from_prediction(
    rec: dict[str, Any],
    previous_palm: np.ndarray | None,
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    params = rec.get("mano_params_rotmat") or {}
    hand_pose = finite_array(params.get("hand_pose"), (3, 3))
    betas = finite_array(params.get("betas"))
    global_orient = finite_array(params.get("global_orient"), (3, 3))
    joints = finite_array(rec.get("hamer_joints_cam"), (3,))
    verts = finite_array(rec.get("hamer_vertices_cam"), (3,))
    points_2d = finite_array(rec.get("hamer_joints_2d_rectified_px"), (2,))
    if points_2d is None:
        points_2d = derive_2d_from_hamer_camera(rec, args)
    if hand_pose is None or betas is None or joints is None or verts is None or points_2d is None:
        return None
    if hand_pose.shape != (15, 3, 3) or betas.shape != (10,) or len(points_2d) < 21:
        return None
    if global_orient is not None and global_orient.shape == (1, 3, 3):
        global_orient = global_orient[0]
    if global_orient is None or global_orient.shape != (3, 3):
        global_orient = np.eye(3, dtype=np.float64)
    local_joints = local_points(joints)
    local_verts = local_points(verts)
    palm_joints = palm_local_np(local_joints)
    score, parts = prediction_quality(rec, points_2d, args)
    temporal_error = None
    if previous_palm is not None:
        temporal_error = float(np.mean(np.linalg.norm(palm_joints - previous_palm, axis=1)))
        score -= min(temporal_error, args.temporal_error_cap_m) / max(args.temporal_error_cap_m, 1e-8) * 0.20
    out = dict(rec)
    out.update(
        {
            "_hand_pose": hand_pose,
            "_betas": betas,
            "_global_orient": global_orient,
            "_raw_joints": joints,
            "_local_joints": local_joints,
            "_local_vertices": local_verts,
            "_palm_joints": palm_joints,
            "_points_2d": points_2d[:21],
            "_conf_2d": finite_array(rec.get("hamer_joints_2d_conf"))[:21] if finite_array(rec.get("hamer_joints_2d_conf")) is not None and len(finite_array(rec.get("hamer_joints_2d_conf"))) >= 21 else np.full(21, 0.45),
            "_score": float(score),
            "_quality_parts": parts,
            "_temporal_pose_error_m": temporal_error,
        }
    )
    return out


def load_predictions(paths: list[Path], group_ids: set[int] | None, args: argparse.Namespace) -> dict[int, dict[str, dict[str, list[dict[str, Any]]]]]:
    data: dict[int, dict[str, dict[str, list[dict[str, Any]]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for path in paths:
        for rec in iter_jsonl(path):
            if rec.get("type") != "hamer_multiview_prediction":
                continue
            group_id = int(rec["group_id"])
            if group_ids is not None and group_id not in group_ids:
                continue
            handedness = rec.get("handedness")
            camera_id = rec.get("camera_id")
            if handedness not in {"Left", "Right"} or not camera_id:
                continue
            if not is_xyz_list(rec.get("hamer_joints_cam")) or not rec.get("mano_params_rotmat"):
                continue
            data[group_id][handedness][camera_id].append(rec)
    return data


def load_mediapipe_2d(path: Path | None, group_ids: set[int] | None, min_score: float) -> dict[int, dict[str, dict[str, dict[str, Any]]]]:
    data: dict[int, dict[str, dict[str, dict[str, Any]]]] = defaultdict(lambda: defaultdict(dict))
    if path is None or not path.exists():
        return data
    for rec in iter_jsonl(path):
        group_id = int(rec.get("group_id"))
        if group_ids is not None and group_id not in group_ids:
            continue
        camera_id = rec.get("camera_id")
        if not camera_id:
            continue
        for hand in rec.get("hands") or []:
            handedness = hand.get("handedness")
            score = hand.get("handedness_score")
            if handedness not in {"Left", "Right"} or not isinstance(score, (int, float)) or float(score) < min_score:
                continue
            points = []
            for point in hand.get("landmarks_rectified_px") or []:
                x = point.get("x")
                y = point.get("y")
                if isinstance(x, (int, float)) and isinstance(y, (int, float)) and math.isfinite(float(x)) and math.isfinite(float(y)):
                    points.append([float(x), float(y)])
            if len(points) >= 21:
                data[group_id][handedness][str(camera_id)] = {
                    "points": np.asarray(points[:21], dtype=np.float64),
                    "score": float(score),
                }
    return data


def attach_mediapipe_2d(
    selected: dict[str, dict[str, Any]],
    mediapipe: dict[int, dict[str, dict[str, dict[str, Any]]]],
    group_id: int,
    handedness: str,
    args: argparse.Namespace,
) -> None:
    if args.use_mediapipe_2d == "never":
        return
    by_camera = mediapipe.get(group_id, {}).get(handedness, {})
    for camera_id, item in selected.items():
        mp = by_camera.get(camera_id)
        if not mp:
            continue
        item["_mediapipe_2d"] = mp["points"]
        item["_mediapipe_conf"] = np.full(21, float(mp["score"]), dtype=np.float64)


def choose_per_camera(
    records_by_camera: dict[str, list[dict[str, Any]]],
    previous_palm: np.ndarray | None,
    args: argparse.Namespace,
) -> dict[str, dict[str, Any]]:
    selected = {}
    for camera_id, items in records_by_camera.items():
        candidates = [item for rec in items if (item := candidate_from_prediction(rec, previous_palm, args)) is not None]
        if not candidates:
            continue
        best = max(candidates, key=lambda item: item["_score"])
        if best["_score"] >= args.min_candidate_score:
            selected[camera_id] = best
    return selected


def select_anchor(selected: dict[str, dict[str, Any]], args: argparse.Namespace) -> tuple[str, dict[str, Any], dict[str, Any]]:
    # C1/C2 are a useful hand-specific prior, not a hard rule. A side view may
    # become anchor only with a clear image-quality advantage; PnP then gates
    # the remaining views against that selected anchor.
    handedness = next(iter(selected.values())).get("handedness")
    preferred_camera = PRIMARY_CAMERAS.get(handedness)
    best_camera, best_item = max(sorted(selected.items()), key=lambda pair: pair[1]["_score"])
    if args.pnp_view_gate_m > 0.0 and preferred_camera in selected:
        preferred_item = selected[preferred_camera]
        preferred_score = float(preferred_item["_score"])
        best_score = float(best_item["_score"])
        if best_camera == preferred_camera or best_score <= preferred_score + args.primary_anchor_score_margin:
            camera_id = str(preferred_camera)
            item = preferred_item
            reason = "pnp_primary_prior"
        else:
            camera_id = str(best_camera)
            item = best_item
            reason = "quality_exceeds_primary_prior"
    else:
        camera_id, item = best_camera, best_item
        reason = "highest_image_quality_score"
    return camera_id, item, {
        "anchor_selection_reason": reason,
        "candidate_scores": {key: float(value["_score"]) for key, value in selected.items()},
        "candidate_quality_parts": {key: value["_quality_parts"] for key, value in selected.items()},
        "candidate_temporal_errors_m": {key: value["_temporal_pose_error_m"] for key, value in selected.items()},
    }


def mano_forward_local(mano: Any, hand_pose_rotmat: torch.Tensor, betas: torch.Tensor, is_right: int) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = hand_pose_rotmat.shape[0]
    global_orient = torch.eye(3, dtype=hand_pose_rotmat.dtype, device=hand_pose_rotmat.device).reshape(1, 1, 3, 3).repeat(batch_size, 1, 1, 1)
    output = mano(global_orient=global_orient.float(), hand_pose=hand_pose_rotmat.float(), betas=betas.float(), pose2rot=False)
    joints = output.joints[:, :21, :].clone()
    verts = output.vertices.clone()
    multiplier = float(2 * int(is_right) - 1)
    joints[:, :, 0] *= multiplier
    verts[:, :, 0] *= multiplier
    wrist = joints[:, 0:1, :].clone()
    return joints - wrist, verts - wrist


def estimate_sequence_beta(
    mano: Any,
    predictions: dict[int, dict[str, dict[str, list[dict[str, Any]]]]],
    handedness: str,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, Any]]:
    observations = []
    for _group_id, by_hand in predictions.items():
        for _camera_id, items in by_hand.get(handedness, {}).items():
            for rec in items:
                item = candidate_from_prediction(rec, None, args)
                if item is not None:
                    observations.append(item)
    if not observations:
        return np.zeros(10, dtype=np.float64), {"beta_source": "zeros_no_observations", "beta_observations": 0, "beta_fit_cost": None}
    observations = sorted(observations, key=lambda item: item["_score"], reverse=True)[: max(1, args.max_beta_observations)]
    betas = np.stack([item["_betas"] for item in observations], axis=0)
    weights = np.asarray([max(item["_score"], 1e-3) for item in observations], dtype=np.float64)
    beta_prior = np.clip(weighted_median(betas, weights), -3.0, 3.0)
    if not args.optimize_mano_betas:
        return beta_prior, {"beta_source": "weighted_median", "beta_observations": len(observations), "beta_fit_cost": None}

    pose_batch = torch.tensor(np.stack([item["_hand_pose"] for item in observations], axis=0), dtype=torch.float32, device=device)
    target_batch = torch.tensor(np.stack([item["_palm_joints"] for item in observations], axis=0), dtype=torch.float32, device=device)
    weight_batch = torch.tensor(weights / max(weights.mean(), 1e-8), dtype=torch.float32, device=device)
    is_right = int(observations[0].get("is_right", 1 if handedness == "Right" else 0))
    beta = torch.tensor(beta_prior, dtype=torch.float32, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([beta], lr=args.beta_learning_rate)
    final_loss = None
    for _ in range(max(0, args.beta_iters)):
        optimizer.zero_grad(set_to_none=True)
        pred_joints, _pred_verts = mano_forward_local(mano, pose_batch, beta.reshape(1, 10).repeat(len(observations), 1), is_right)
        pred_palm = palm_local_torch(pred_joints)
        per_obs = ((pred_palm - target_batch) ** 2).mean(dim=(1, 2))
        beta_prior_loss = ((beta - torch.tensor(beta_prior, dtype=torch.float32, device=device)) ** 2).mean()
        loss = (per_obs * weight_batch).mean() + args.beta_prior_weight * beta_prior_loss
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            beta.clamp_(-3.0, 3.0)
        final_loss = float(loss.detach().cpu().item())
    return beta.detach().cpu().numpy().astype(np.float64), {
        "beta_source": "optimized_sequence_hamer_local",
        "beta_observations": len(observations),
        "beta_fit_cost": final_loss,
        "beta_prior": beta_prior.tolist(),
    }


def refine_sequence_beta_image_2d(
    mano: Any,
    predictions: dict[int, dict[str, dict[str, list[dict[str, Any]]]]],
    handedness: str,
    beta_init: np.ndarray,
    beta_meta: dict[str, Any],
    cameras: dict[str, dict[str, np.ndarray]],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Refine one hand's shared shape against calibrated multi-view 2D observations.

    Pose and the physical-PnP global transform are deliberately held fixed in
    this stage. That makes beta a sequence-level variable instead of letting a
    noisy frame absorb its reprojection residual by changing shape.
    """
    if not args.optimize_mano_betas:
        return beta_init, beta_meta

    expected_is_right = int(handedness == "Right")
    candidates: list[dict[str, Any]] = []
    for by_hand in predictions.values():
        for records in by_hand.get(handedness, {}).values():
            for rec in records:
                item = candidate_from_prediction(rec, None, args)
                if item is None or int(item.get("is_right", expected_is_right)) != expected_is_right:
                    continue
                candidates.append(item)
    if not candidates:
        return beta_init, {**beta_meta, "image_beta_source": "skipped_no_candidates", "image_beta_observations": 0}

    candidates = sorted(candidates, key=lambda item: item["_score"], reverse=True)[: max(1, args.image_beta_max_observations)]
    observations: list[dict[str, Any]] = []
    beta_tensor = torch.tensor(beta_init, dtype=torch.float32, device=device)
    for item in candidates:
        camera_id = str(item.get("camera_id"))
        camera = cameras.get(camera_id)
        if camera is None:
            continue
        with torch.no_grad():
            init_joints, _ = mano_forward_local(
                mano,
                torch.tensor(item["_hand_pose"], dtype=torch.float32, device=device).reshape(1, 15, 3, 3),
                beta_tensor.reshape(1, 10),
                expected_is_right,
            )
        orient_h, trans_h, source = initial_global_from_anchor(
            item,
            init_joints[0].detach().cpu().numpy(),
            cameras,
            args._projection_corrections.get(camera_id),
            True,
        )
        # Virtual-crop translation is not metrically compatible with the
        # rectified K, so it must not supervise a physical image-space beta.
        if source != "physical_pnp":
            continue
        observations.append(
            {
                "pose": item["_hand_pose"],
                "orient": orient_h,
                "trans": trans_h,
                "camera": camera,
                "points_2d": item["_points_2d"],
                "conf_2d": item["_conf_2d"],
                "weight": max(float(item["_score"]), 1e-3),
            }
        )

    if len(observations) < 6:
        return beta_init, {
            **beta_meta,
            "image_beta_source": "skipped_insufficient_physical_pnp",
            "image_beta_observations": len(observations),
        }

    pose_batch = torch.tensor(np.stack([item["pose"] for item in observations]), dtype=torch.float32, device=device)
    orient_batch = torch.tensor(np.stack([item["orient"] for item in observations]), dtype=torch.float32, device=device)
    trans_batch = torch.tensor(np.stack([item["trans"] for item in observations]), dtype=torch.float32, device=device)
    r_h_c = torch.tensor(np.stack([item["camera"]["R_H_C"] for item in observations]), dtype=torch.float32, device=device)
    t_h_c = torch.tensor(np.stack([item["camera"]["t_H_C"] for item in observations]), dtype=torch.float32, device=device)
    k_batch = torch.tensor(np.stack([item["camera"]["K"] for item in observations]), dtype=torch.float32, device=device)
    target_2d = torch.tensor(np.stack([item["points_2d"] for item in observations]), dtype=torch.float32, device=device)
    conf_2d = torch.tensor(np.stack([item["conf_2d"] for item in observations]), dtype=torch.float32, device=device)
    weights = torch.tensor(np.asarray([item["weight"] for item in observations]), dtype=torch.float32, device=device)
    weights = weights / torch.clamp(weights.sum(), min=1e-6)

    beta_prior = torch.tensor(beta_init, dtype=torch.float32, device=device)
    beta = beta_prior.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([beta], lr=args.beta_learning_rate)
    final_loss = None
    for _ in range(max(0, args.beta_iters)):
        optimizer.zero_grad(set_to_none=True)
        local_joints, _ = mano_forward_local(
            mano,
            pose_batch,
            beta.reshape(1, 10).repeat(len(observations), 1),
            expected_is_right,
        )
        points_h = torch.matmul(local_joints, orient_batch.transpose(1, 2)) + trans_batch[:, None, :]
        points_c = torch.matmul(points_h - t_h_c[:, None, :], r_h_c)
        depth = points_c[:, :, 2]
        projected = torch.stack(
            (
                k_batch[:, 0, 0, None] * points_c[:, :, 0] / torch.clamp(depth, min=1e-5) + k_batch[:, 0, 2, None],
                k_batch[:, 1, 1, None] * points_c[:, :, 1] / torch.clamp(depth, min=1e-5) + k_batch[:, 1, 2, None],
            ),
            dim=-1,
        )
        residual = torch.linalg.norm(projected - target_2d, dim=-1)
        valid = (depth > 1e-4).float() * conf_2d
        per_observation = (cauchy_loss(residual, args.keypoint_cauchy_scale_px) * valid).sum(dim=1) / torch.clamp(valid.sum(dim=1), min=1e-6)
        reprojection_loss = (per_observation * weights).sum()
        prior_loss = ((beta - beta_prior) ** 2).mean()
        loss = reprojection_loss + args.image_beta_prior_weight * prior_loss
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            beta.clamp_(-3.0, 3.0)
        final_loss = float(loss.detach().cpu().item())

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return beta.detach().cpu().numpy().astype(np.float64), {
        **beta_meta,
        "beta_source": "image_2d_pnp_refined",
        "image_beta_source": "physical_pnp_fixed_pose_reprojection",
        "image_beta_observations": len(observations),
        "image_beta_fit_cost": final_loss,
        "image_beta_prior": beta_init.tolist(),
    }


def torch_camera(camera: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "K": torch.tensor(camera["K"], dtype=torch.float32, device=device),
        "R_H_C": torch.tensor(camera["R_H_C"], dtype=torch.float32, device=device),
        "t_H_C": torch.tensor(camera["t_H_C"], dtype=torch.float32, device=device),
    }


def transform_to_camera(points_local: torch.Tensor, orient_h: torch.Tensor, trans_h: torch.Tensor, camera: dict[str, torch.Tensor]) -> torch.Tensor:
    points_h = torch.matmul(points_local, orient_h.transpose(0, 1)) + trans_h.reshape(1, 3)
    return torch.matmul(points_h - camera["t_H_C"].reshape(1, 3), camera["R_H_C"])


def project_camera_points(points_c: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    z = torch.clamp(points_c[:, 2], min=1e-5)
    x = k[0, 0] * points_c[:, 0] / z + k[0, 2]
    y = k[1, 1] * points_c[:, 1] / z + k[1, 2]
    return torch.stack([x, y], dim=1), z


def load_mask_tensors(
    mask_path: str | None,
    image_width: int,
    image_height: int,
    device: torch.device,
    boundary_samples: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    mask = load_mask(mask_path, image_size=(image_width, image_height))
    if mask is None:
        return None, None, None
    mask_u8 = mask.astype(np.uint8)
    outside = (1 - mask_u8).astype(np.uint8)
    dist = cv2.distanceTransform(outside, cv2.DIST_L2, 3).astype(np.float32)
    eroded = cv2.erode(mask_u8, np.ones((3, 3), dtype=np.uint8), iterations=1)
    boundary_yx = np.argwhere((mask_u8 > 0) & (eroded == 0))
    if len(boundary_yx) and boundary_samples > 0:
        indexes = np.linspace(0, len(boundary_yx) - 1, min(int(boundary_samples), len(boundary_yx)), dtype=np.int64)
        boundary_xy = boundary_yx[indexes][:, ::-1].astype(np.float32)
        boundary_tensor = torch.tensor(boundary_xy, dtype=torch.float32, device=device)
    else:
        boundary_tensor = None
    mask_tensor = torch.tensor(mask.astype(np.float32), dtype=torch.float32, device=device).reshape(1, 1, image_height, image_width)
    dist_tensor = torch.tensor(dist, dtype=torch.float32, device=device).reshape(1, 1, image_height, image_width)
    return mask_tensor, dist_tensor, boundary_tensor


def soft_boundary_to_mesh_loss(boundary_xy: torch.Tensor, mesh_xy: torch.Tensor, temperature_px: float) -> torch.Tensor:
    """Differentiable soft-Chamfer term from SAM3 mask boundary to projected mesh."""
    if len(boundary_xy) == 0 or len(mesh_xy) == 0:
        return torch.zeros((), dtype=mesh_xy.dtype, device=mesh_xy.device)
    distances = torch.cdist(boundary_xy.unsqueeze(0), mesh_xy.unsqueeze(0)).squeeze(0)
    temperature = max(float(temperature_px), 1e-3)
    # Use log-mean-exp rather than log-sum-exp so an exact match remains zero
    # regardless of the number of sampled mesh vertices.
    return (
        -temperature
        * (torch.logsumexp(-distances / temperature, dim=1) - math.log(float(mesh_xy.shape[0])))
    ).mean()


def sample_image_tensor(image_tensor: torch.Tensor, xy: torch.Tensor, image_width: int, image_height: int) -> torch.Tensor:
    x = 2.0 * xy[:, 0] / max(float(image_width - 1), 1.0) - 1.0
    y = 2.0 * xy[:, 1] / max(float(image_height - 1), 1.0) - 1.0
    grid = torch.stack([x, y], dim=1).reshape(1, -1, 1, 2)
    return torch.nn.functional.grid_sample(image_tensor, grid, mode="bilinear", padding_mode="border", align_corners=True).reshape(-1)


def initial_global_from_anchor(
    anchor: dict[str, Any],
    local_joints_init: np.ndarray,
    cameras: dict[str, dict[str, np.ndarray]],
    projection_affine: np.ndarray | None = None,
    use_physical_pnp: bool = False,
) -> tuple[np.ndarray, np.ndarray, str]:
    camera_id = str(anchor["camera_id"])
    camera = cameras[camera_id]
    if use_physical_pnp:
        target_2d = undo_projection_correction_np(np.asarray(anchor["_points_2d"], dtype=np.float64), projection_affine)
        confidence = np.asarray(anchor.get("_conf_2d"), dtype=np.float64)
        valid = (
            np.isfinite(target_2d).all(axis=1)
            & np.isfinite(local_joints_init).all(axis=1)
            & (confidence[: len(target_2d)] > 1e-4)
        )
    else:
        valid = np.zeros(len(local_joints_init), dtype=bool)
    if int(np.sum(valid)) >= 6:
        object_points = np.asarray(local_joints_init[valid], dtype=np.float64).reshape(-1, 1, 3)
        image_points = np.asarray(target_2d[valid], dtype=np.float64).reshape(-1, 1, 2)
        try:
            ok, rvec, tvec = cv2.solvePnP(
                object_points,
                image_points,
                camera["K"].astype(np.float64),
                None,
                flags=cv2.SOLVEPNP_EPNP,
            )
            if ok:
                # A short iterative pass is noticeably more stable for the
                # near-planar palm poses than EPNP alone.
                ok_iter, rvec, tvec = cv2.solvePnP(
                    object_points,
                    image_points,
                    camera["K"].astype(np.float64),
                    None,
                    rvec,
                    tvec,
                    useExtrinsicGuess=True,
                    flags=cv2.SOLVEPNP_ITERATIVE,
                )
                if ok_iter and np.all(np.isfinite(tvec)) and float(tvec[2]) > 1e-4:
                    r_c_hand, _ = cv2.Rodrigues(rvec)
                    r_h_hand = camera["R_H_C"] @ r_c_hand
                    t_h = camera["R_H_C"] @ tvec.reshape(3) + camera["t_H_C"]
                    return r_h_hand.astype(np.float64), t_h.astype(np.float64), "physical_pnp"
        except cv2.error:
            pass

    # This is only a compatibility fallback for incomplete/invalid 2D input.
    # HaMeR's cam_t is expressed under its virtual crop camera, so it is not a
    # good physical initialization when PnP is available.
    _aligned, r_c_hand, _src_c, _tgt_c, _err = kabsch_transform(local_joints_init, anchor["_local_joints"])
    wrist_c = np.asarray(anchor.get("hamer_cam_t"), dtype=np.float64).reshape(3) + anchor["_raw_joints"][0]
    r_h_hand = camera["R_H_C"] @ r_c_hand
    t_h = camera["R_H_C"] @ wrist_c + camera["t_H_C"]
    return r_h_hand.astype(np.float64), t_h.astype(np.float64), "hamer_virtual_camera_fallback"


def pnp_wrist_headset(
    item: dict[str, Any],
    camera: dict[str, np.ndarray],
    projection_affine: np.ndarray | None,
) -> np.ndarray | None:
    """Estimate the camera-observed wrist in H without using virtual HaMeR depth."""
    joints = np.asarray(item.get("_raw_joints"), dtype=np.float64)
    points = undo_projection_correction_np(np.asarray(item.get("_points_2d"), dtype=np.float64), projection_affine)
    confidence = np.asarray(item.get("_conf_2d"), dtype=np.float64)
    if joints.shape[0] < 6 or points.shape[0] < 6:
        return None
    count = min(len(joints), len(points), len(confidence), 21)
    valid = np.isfinite(joints[:count]).all(axis=1) & np.isfinite(points[:count]).all(axis=1) & (confidence[:count] > 1e-4)
    if int(np.sum(valid)) < 6:
        return None
    try:
        ok, _rvec, tvec = cv2.solvePnP(
            (joints[:count] - joints[0:1]) [valid].reshape(-1, 1, 3),
            points[:count][valid].reshape(-1, 1, 2),
            camera["K"].astype(np.float64),
            None,
            flags=cv2.SOLVEPNP_EPNP,
        )
    except cv2.error:
        return None
    if not ok or not np.all(np.isfinite(tvec)) or float(tvec[2]) <= 1e-4:
        return None
    return camera["R_H_C"] @ tvec.reshape(3) + camera["t_H_C"]


def mean_reprojection_error(projected: np.ndarray, target: np.ndarray, conf: np.ndarray) -> float:
    valid = conf > 1e-4
    if not np.any(valid):
        return float("inf")
    return float(np.average(np.linalg.norm(projected[valid] - target[valid], axis=1), weights=conf[valid]))


def mask_distance_for_points(points: np.ndarray, mask_path: str | None, image_width: int, image_height: int) -> float | None:
    mask = load_mask(mask_path, image_size=(image_width, image_height))
    if mask is None:
        return None
    dist = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 3)
    values = []
    for x, y in points:
        if not (0 <= x < image_width and 0 <= y < image_height):
            values.append(max(image_width, image_height) * 0.25)
            continue
        ix = min(max(int(round(x)), 0), image_width - 1)
        iy = min(max(int(round(y)), 0), image_height - 1)
        values.append(float(dist[iy, ix]))
    return float(np.mean(values)) if values else None


def refine_hand(
    mano: Any,
    group_id: int,
    handedness: str,
    selected: dict[str, dict[str, Any]],
    sequence_beta: np.ndarray,
    beta_info: dict[str, Any],
    previous: dict[str, Any] | None,
    previous_previous: dict[str, Any] | None,
    cameras_np: dict[str, dict[str, np.ndarray]],
    cameras_torch: dict[str, dict[str, torch.Tensor]],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any] | None:
    previous_palm = None if previous is None else np.asarray(previous.get("palm_local_joints_m"), dtype=np.float64)
    if not selected:
        if previous is None:
            return None
        hand = dict(previous)
        hand["group_id"] = group_id
        hand["mode"] = "mano_multiview_image_temporal_fallback"
        hand["metric_valid"] = False
        hand["fit_success"] = False
        hand["fallback_reason"] = "missing_candidates"
        return hand

    anchor_camera, anchor, selection_info = select_anchor(selected, args)
    if previous_palm is not None and anchor.get("_temporal_pose_error_m") is not None and anchor["_temporal_pose_error_m"] > args.temporal_reject_threshold_m:
        alternatives = [item for item in selected.values() if item.get("_temporal_pose_error_m") is not None and item["_temporal_pose_error_m"] <= args.temporal_reject_threshold_m]
        if alternatives:
            anchor = max(alternatives, key=lambda item: item["_score"])
            anchor_camera = str(anchor["camera_id"])
            selection_info["anchor_selection_reason"] = "temporal_reject_switched_anchor"
        elif previous is not None:
            hand = dict(previous)
            hand["group_id"] = group_id
            hand["mode"] = "mano_multiview_image_temporal_fallback"
            hand["metric_valid"] = False
            hand["fit_success"] = False
            hand["fallback_reason"] = "temporal_reject"
            return hand

    anchor_pose = torch.tensor(anchor["_hand_pose"], dtype=torch.float32, device=device)
    beta_base = torch.tensor(sequence_beta, dtype=torch.float32, device=device)
    with torch.no_grad():
        init_joints_t, _init_verts_t = mano_forward_local(mano, anchor_pose.reshape(1, 15, 3, 3), beta_base.reshape(1, 10), int(anchor.get("is_right", 1 if handedness == "Right" else 0)))
    init_joints_np = init_joints_t[0].detach().cpu().numpy()
    init_orient_np, init_trans_np, initialization_source = initial_global_from_anchor(
        anchor,
        init_joints_np,
        cameras_np,
        args._projection_corrections.get(anchor_camera),
        args.global_initialization == "physical-pnp",
    )

    pose_6d = rotmat_to_6d(anchor_pose).detach().clone().requires_grad_(bool(args.optimize_mano_pose))
    orient_6d = rotmat_to_6d(torch.tensor(init_orient_np, dtype=torch.float32, device=device)).detach().clone().requires_grad_(True)
    trans_h = torch.tensor(init_trans_np, dtype=torch.float32, device=device, requires_grad=True)
    beta_delta = torch.zeros(10, dtype=torch.float32, device=device, requires_grad=bool(args.allow_frame_beta_delta))
    params = [orient_6d, trans_h]
    if args.optimize_mano_pose:
        params.append(pose_6d)
    if args.allow_frame_beta_delta:
        params.append(beta_delta)

    prev_pose_tensor = None
    prev_local_joints_tensor = None
    prev_prev_pose_tensor = None
    prev_prev_local_joints_tensor = None
    if previous is not None:
        prev_pose = previous.get("mano_params_refined", {}).get("hand_pose")
        if prev_pose is not None:
            prev_pose_tensor = torch.tensor(prev_pose, dtype=torch.float32, device=device)
        prev_local = previous.get("local_joints_m")
        if prev_local is not None:
            prev_local_joints_tensor = torch.tensor(prev_local, dtype=torch.float32, device=device).reshape(1, 21, 3)
    if previous_previous is not None:
        prev_prev_pose = previous_previous.get("mano_params_refined", {}).get("hand_pose")
        if prev_prev_pose is not None:
            prev_prev_pose_tensor = torch.tensor(prev_prev_pose, dtype=torch.float32, device=device)
        prev_prev_local = previous_previous.get("local_joints_m")
        if prev_prev_local is not None:
            prev_prev_local_joints_tensor = torch.tensor(prev_prev_local, dtype=torch.float32, device=device).reshape(1, 21, 3)

    observations = []
    pre_rejected: list[str] = []
    anchor_affine = args._projection_corrections.get(anchor_camera)
    anchor_wrist_h = pnp_wrist_headset(anchor, cameras_np[anchor_camera], anchor_affine)
    for camera_id, item in sorted(selected.items()):
        if camera_id not in cameras_torch:
            continue
        affine_np = args._projection_corrections.get(camera_id)
        pnp_distance = None
        if anchor_wrist_h is not None:
            wrist_h = pnp_wrist_headset(item, cameras_np[camera_id], affine_np)
            if wrist_h is not None:
                pnp_distance = float(np.linalg.norm(wrist_h - anchor_wrist_h))
        item["_pnp_wrist_distance_m"] = pnp_distance
        if (
            args.pnp_view_gate_m > 0.0
            and camera_id != anchor_camera
            and (pnp_distance is None or pnp_distance > args.pnp_view_gate_m)
        ):
            pre_rejected.append(camera_id)
            continue
        mask_tensor, dist_tensor, boundary_tensor = load_mask_tensors(
            item.get("sam3_mask_path"),
            args.image_width,
            args.image_height,
            device,
            args.mask_boundary_samples,
        )
        view_weight = observation_quality_weight(item, camera_id, anchor_camera, args)
        observations.append(
            {
                "camera_id": camera_id,
                "item": item,
                "camera": cameras_torch[camera_id],
                "projection_affine": torch.tensor(args._projection_corrections[camera_id], dtype=torch.float32, device=device)
                if camera_id in args._projection_corrections
                else None,
                "projection_affine_np": args._projection_corrections.get(camera_id),
                "target_2d": torch.tensor(item["_points_2d"], dtype=torch.float32, device=device),
                "conf_2d": torch.tensor(item["_conf_2d"], dtype=torch.float32, device=device),
                "view_weight": torch.tensor(view_weight, dtype=torch.float32, device=device),
                "view_weight_float": view_weight,
                "mediapipe_2d": torch.tensor(item["_mediapipe_2d"], dtype=torch.float32, device=device) if item.get("_mediapipe_2d") is not None else None,
                "mediapipe_conf": torch.tensor(item["_mediapipe_conf"], dtype=torch.float32, device=device) if item.get("_mediapipe_conf") is not None else None,
                "mask": mask_tensor,
                "dist": dist_tensor,
                "mask_boundary": boundary_tensor,
            }
        )
    if not observations:
        return None

    optimizer = torch.optim.Adam(params, lr=args.learning_rate)
    init_orient = torch.tensor(init_orient_np, dtype=torch.float32, device=device)
    init_trans = torch.tensor(init_trans_np, dtype=torch.float32, device=device)
    final_loss = None
    final_parts: dict[str, float] = {}
    vertex_step = max(1, int(args.vertex_subsample))
    boundary_vertex_step = max(1, int(args.mask_boundary_vertex_subsample))
    for _ in range(max(1, args.max_iters)):
        optimizer.zero_grad(set_to_none=True)
        pose_rot = rot6d_to_rotmat(pose_6d).reshape(1, 15, 3, 3)
        orient_h = rot6d_to_rotmat(orient_6d).reshape(3, 3)
        beta = (beta_base + beta_delta).reshape(1, 10)
        local_joints, local_verts = mano_forward_local(mano, pose_rot, beta, int(anchor.get("is_right", 1 if handedness == "Right" else 0)))

        kp_losses = []
        kp_weights = []
        mp_losses = []
        mp_weights = []
        mask_losses = []
        mask_weights = []
        boundary_losses = []
        boundary_weights = []
        for obs in observations:
            points_c = transform_to_camera(local_joints[0], orient_h, trans_h, obs["camera"])
            projected, depth = project_camera_points(points_c, obs["camera"]["K"])
            projected = apply_projection_correction_torch(projected, obs["projection_affine"])
            residual = torch.linalg.norm(projected - obs["target_2d"], dim=1)
            valid = (depth > 1e-4).float() * obs["conf_2d"]
            kp_losses.append((cauchy_loss(residual, args.keypoint_cauchy_scale_px) * valid).sum() / torch.clamp(valid.sum(), min=1e-6))
            kp_weights.append(obs["view_weight"])
            if obs["mediapipe_2d"] is not None and obs["mediapipe_conf"] is not None:
                mp_residual = torch.linalg.norm(projected - obs["mediapipe_2d"], dim=1)
                mp_valid = (depth > 1e-4).float() * obs["mediapipe_conf"]
                mp_losses.append((cauchy_loss(mp_residual, args.keypoint_cauchy_scale_px) * mp_valid).sum() / torch.clamp(mp_valid.sum(), min=1e-6))
                mp_weights.append(obs["view_weight"])

            if obs["dist"] is not None and args.mask_loss_weight > 0:
                verts_c = transform_to_camera(local_verts[0, ::vertex_step], orient_h, trans_h, obs["camera"])
                verts_px, verts_depth = project_camera_points(verts_c, obs["camera"]["K"])
                verts_px = apply_projection_correction_torch(verts_px, obs["projection_affine"])
                in_front = verts_depth > 1e-4
                if torch.any(in_front):
                    dist_values = sample_image_tensor(obs["dist"], verts_px[in_front], args.image_width, args.image_height)
                    mask_losses.append(cauchy_loss(dist_values, args.mask_cauchy_scale_px).mean())
                    mask_weights.append(obs["view_weight"])
            if obs["mask_boundary"] is not None and args.mask_boundary_loss_weight > 0:
                boundary_verts_c = transform_to_camera(local_verts[0, ::boundary_vertex_step], orient_h, trans_h, obs["camera"])
                boundary_verts_px, boundary_verts_depth = project_camera_points(boundary_verts_c, obs["camera"]["K"])
                boundary_verts_px = apply_projection_correction_torch(boundary_verts_px, obs["projection_affine"])
                in_front = boundary_verts_depth > 1e-4
                if torch.any(in_front):
                    boundary_losses.append(
                        cauchy_loss(
                            soft_boundary_to_mesh_loss(
                                obs["mask_boundary"],
                                boundary_verts_px[in_front],
                                args.mask_boundary_softmin_px,
                            ),
                            args.mask_cauchy_scale_px,
                        )
                    )
                    boundary_weights.append(obs["view_weight"])

        keypoint_loss = weighted_torch_mean(kp_losses, kp_weights, device)
        mediapipe_loss = weighted_torch_mean(mp_losses, mp_weights, device)
        mask_loss = weighted_torch_mean(mask_losses, mask_weights, device)
        boundary_loss = weighted_torch_mean(boundary_losses, boundary_weights, device)
        pose_prior = ((pose_rot.reshape(15, 3, 3) - anchor_pose) ** 2).mean()
        beta_prior = (beta_delta ** 2).mean()
        global_prior = ((orient_h - init_orient) ** 2).mean() + ((trans_h - init_trans) ** 2).mean()
        temporal_pose = torch.tensor(0.0, dtype=torch.float32, device=device)
        temporal_joint = torch.tensor(0.0, dtype=torch.float32, device=device)
        temporal_acceleration = torch.tensor(0.0, dtype=torch.float32, device=device)
        if prev_pose_tensor is not None:
            temporal_pose = ((pose_rot.reshape(15, 3, 3) - prev_pose_tensor) ** 2).mean()
        if prev_local_joints_tensor is not None:
            temporal_joint = ((local_joints - prev_local_joints_tensor) ** 2).mean()
        if prev_pose_tensor is not None and prev_prev_pose_tensor is not None:
            pose_acceleration = pose_rot.reshape(15, 3, 3) - 2.0 * prev_pose_tensor + prev_prev_pose_tensor
            temporal_acceleration = (pose_acceleration**2).mean()
            if prev_local_joints_tensor is not None and prev_prev_local_joints_tensor is not None:
                joint_acceleration = local_joints - 2.0 * prev_local_joints_tensor + prev_prev_local_joints_tensor
                temporal_acceleration = temporal_acceleration + (joint_acceleration**2).mean()
        loss = (
            args.keypoint_loss_weight * keypoint_loss
            + args.mediapipe_loss_weight * mediapipe_loss
            + args.mask_loss_weight * mask_loss
            + args.mask_boundary_loss_weight * boundary_loss
            + args.pose_prior_weight * pose_prior
            + args.beta_prior_weight * beta_prior
            + args.temporal_pose_weight * temporal_pose
            + args.temporal_joint_weight * temporal_joint
            + args.temporal_acceleration_weight * temporal_acceleration
            + args.global_prior_weight * global_prior
        )
        loss.backward()
        optimizer.step()
        if args.allow_frame_beta_delta:
            with torch.no_grad():
                norm = torch.linalg.norm(beta_delta)
                if norm > args.max_frame_beta_delta_norm:
                    beta_delta.mul_(args.max_frame_beta_delta_norm / (norm + 1e-8))
                beta_delta.clamp_(-0.20, 0.20)
        final_loss = float(loss.detach().cpu().item())
        final_parts = {
            "keypoint": float(keypoint_loss.detach().cpu().item()),
            "mediapipe": float(mediapipe_loss.detach().cpu().item()),
            "mask": float(mask_loss.detach().cpu().item()),
            "mask_boundary": float(boundary_loss.detach().cpu().item()),
            "pose_prior": float(pose_prior.detach().cpu().item()),
            "temporal_pose": float(temporal_pose.detach().cpu().item()),
            "temporal_joint": float(temporal_joint.detach().cpu().item()),
            "temporal_acceleration": float(temporal_acceleration.detach().cpu().item()),
        }

    pose_rot_np = rot6d_to_rotmat(pose_6d).detach().cpu().numpy()
    pose_rot_np = smooth_rotmat(pose_rot_np, None if previous is None else np.asarray(previous.get("mano_params_refined", {}).get("hand_pose"), dtype=np.float64), args.smoothing_alpha)
    orient_np = rot6d_to_rotmat(orient_6d).detach().cpu().numpy()
    trans_np = trans_h.detach().cpu().numpy().astype(np.float64)
    beta_np = np.clip((beta_base + beta_delta).detach().cpu().numpy(), -3.0, 3.0).astype(np.float64)
    with torch.no_grad():
        pose_tensor = torch.tensor(pose_rot_np[None], dtype=torch.float32, device=device)
        beta_tensor = torch.tensor(beta_np[None], dtype=torch.float32, device=device)
        local_joints_t, local_verts_t = mano_forward_local(mano, pose_tensor, beta_tensor, int(anchor.get("is_right", 1 if handedness == "Right" else 0)))
    local_joints = local_joints_t[0].detach().cpu().numpy().astype(np.float64)
    local_vertices = local_verts_t[0].detach().cpu().numpy().astype(np.float64)
    palm_joints = palm_local_np(local_joints)
    palm_vertices = palm_local_np(local_vertices, local_joints)

    projection_debug = {}
    optimized_used = []
    metric_used = []
    rejected = list(pre_rejected)
    orient_t = torch.tensor(orient_np, dtype=torch.float32, device=device)
    trans_t = torch.tensor(trans_np, dtype=torch.float32, device=device)
    local_joints_eval = torch.tensor(local_joints, dtype=torch.float32, device=device)
    local_verts_eval = torch.tensor(local_vertices[::vertex_step], dtype=torch.float32, device=device)
    for obs in observations:
        camera_id = obs["camera_id"]
        with torch.no_grad():
            points_c = transform_to_camera(local_joints_eval, orient_t, trans_t, obs["camera"])
            projected_t, _depth = project_camera_points(points_c, obs["camera"]["K"])
            projected_t = apply_projection_correction_torch(projected_t, obs["projection_affine"])
            projected = projected_t.detach().cpu().numpy()
            verts_c = transform_to_camera(local_verts_eval, orient_t, trans_t, obs["camera"])
            verts_px_t, _verts_depth = project_camera_points(verts_c, obs["camera"]["K"])
            verts_px_t = apply_projection_correction_torch(verts_px_t, obs["projection_affine"])
            verts_px = verts_px_t.detach().cpu().numpy()
        item = obs["item"]
        reproj = mean_reprojection_error(projected, item["_points_2d"], item["_conf_2d"])
        mask_dist = mask_distance_for_points(verts_px, item.get("sam3_mask_path"), args.image_width, args.image_height)
        reproj_weight = robust_scalar_weight(reproj, args.soft_reprojection_error_px)
        mask_weight = robust_scalar_weight(mask_dist, args.soft_mask_distance_px)
        residual_weight = reproj_weight * mask_weight
        final_view_weight = float(obs["view_weight_float"]) * residual_weight
        metric_ok = (
            reproj <= args.max_reprojection_error_px
            and (mask_dist is None or mask_dist <= args.max_mask_distance_px)
            and final_view_weight >= args.min_metric_used_weight
        )
        soft_ok = final_view_weight >= args.min_soft_used_weight
        if soft_ok:
            optimized_used.append(camera_id)
        else:
            if camera_id not in rejected:
                rejected.append(camera_id)
        if metric_ok:
            metric_used.append(camera_id)
        projection_debug[camera_id] = {
            "joints_2d_px": projected.tolist(),
            "hamer_joints_2d_px": item["_points_2d"].tolist(),
            "mean_reprojection_error_px": reproj,
            "mean_mask_distance_px": mask_dist,
            "view_weight_prior": float(obs["view_weight_float"]),
            "reprojection_weight": reproj_weight,
            "mask_weight": mask_weight,
            "final_view_weight": final_view_weight,
            "used": bool(soft_ok),
            "metric_used": bool(metric_ok),
            "source_detector": item.get("source_detector"),
            "mask_path": item.get("sam3_mask_path"),
            "rectified_image_path": item.get("rectified_image_path"),
            "projection_correction_affine": None
            if obs["projection_affine_np"] is None
            else obs["projection_affine_np"].tolist(),
            "pnp_wrist_distance_m": item.get("_pnp_wrist_distance_m"),
        }
    if anchor_camera not in optimized_used and anchor_camera in projection_debug:
        optimized_used.append(anchor_camera)
        rejected = [camera_id for camera_id in rejected if camera_id != anchor_camera]
        projection_debug[anchor_camera]["used"] = True
        projection_debug[anchor_camera]["forced_anchor_used"] = True

    temporal_error = None
    if previous_palm is not None:
        temporal_error = float(np.mean(np.linalg.norm(palm_joints - previous_palm, axis=1)))
    if len(metric_used) > 1:
        mode = "mano_multiview_image_refined_fused"
        metric_valid = True
    elif len(optimized_used) > 1:
        mode = "mano_multiview_image_soft_fused"
        metric_valid = False
    else:
        mode = "mano_multiview_image_single_view"
        metric_valid = False
    return {
        "group_id": group_id,
        "handedness": handedness,
        "mode": mode,
        "metric_valid": metric_valid,
        "fit_success": True,
        "fit_cost": final_loss,
        "fit_loss_parts": final_parts,
        "anchor_camera": anchor_camera,
        "primary_camera": anchor_camera,
        "used_cameras": optimized_used,
        "optimized_cameras": optimized_used,
        "metric_cameras": metric_used,
        "rejected_cameras": rejected,
        "anchor_score": float(anchor["_score"]),
        "initialization_source": initialization_source,
        "pose_error_m": temporal_error,
        "beta_source": beta_info.get("beta_source"),
        "beta_observations": beta_info.get("beta_observations"),
        "local_joints_m": local_joints.tolist(),
        "local_vertices_m": local_vertices.tolist(),
        "palm_local_joints_m": palm_joints.tolist(),
        "palm_local_vertices_m": palm_vertices.tolist(),
        "projection_debug": projection_debug,
        "mano_params_refined": {
            "hand_pose": pose_rot_np.tolist(),
            "betas": beta_np.tolist(),
            "global_orient": orient_np.reshape(1, 3, 3).tolist(),
            "global_t_headset_m": trans_np.tolist(),
        },
        "mano_params_anchor": {
            "hand_pose": anchor["_hand_pose"].tolist(),
            "betas": anchor["_betas"].tolist(),
            "global_orient": anchor["_global_orient"].reshape(1, 3, 3).tolist(),
        },
        **selection_info,
    }


def output_joints(hand: dict[str, Any]) -> list[dict[str, Any]]:
    joints = []
    local = hand.get("local_joints_m") or []
    palm = hand.get("palm_local_joints_m") or []
    for index, position in enumerate(local):
        if not isinstance(position, list) or len(position) != 3:
            continue
        palm_position = palm[index] if index < len(palm) else position
        joints.append(
            {
                "index": index,
                "joint_index": index,
                "name": LANDMARK_NAMES[index] if index < len(LANDMARK_NAMES) else f"joint_{index}",
                "valid": True,
                "metric_valid": bool(hand.get("metric_valid")),
                "position": position,
                "root_relative_headset_m": position,
                "palm_local_m": palm_position,
                "source_cameras": hand.get("used_cameras", []),
                "rejected_cameras": hand.get("rejected_cameras", []),
                "reconstruction_mode": hand.get("mode"),
            }
        )
    return joints


def draw_points(image: np.ndarray, points: list[list[float]], color: tuple[int, int, int], label: str | None = None) -> None:
    valid: dict[int, tuple[int, int]] = {}
    for index, point in enumerate(points[:21]):
        if not isinstance(point, list) or len(point) < 2:
            continue
        x, y = int(round(float(point[0]))), int(round(float(point[1])))
        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
            valid[index] = (x, y)
    for start, end in HAND_CONNECTIONS:
        if start in valid and end in valid:
            cv2.line(image, valid[start], valid[end], color, 2, cv2.LINE_AA)
    for index, xy in valid.items():
        cv2.circle(image, xy, 4 if index else 6, color, -1, cv2.LINE_AA)
    if label and 0 in valid:
        cv2.putText(image, label, (valid[0][0] + 8, max(18, valid[0][1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def save_debug_overlays(frame: dict[str, Any], output_dir: Path, suffix: str, debug_every: int) -> None:
    if debug_every <= 0 or int(frame["group_id"]) % debug_every != 0:
        return
    by_camera: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for hand in frame.get("hands") or []:
        for camera_id, info in (hand.get("projection_debug") or {}).items():
            by_camera[camera_id].append((hand, info))
    for camera_id, items in by_camera.items():
        image_path = None
        for _hand, info in items:
            if info.get("rectified_image_path"):
                image_path = Path(info["rectified_image_path"])
                break
        if image_path is None:
            continue
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        for hand, info in items:
            if info.get("mask_path"):
                mask = load_mask(info.get("mask_path"), image_size=(image.shape[1], image.shape[0]))
                if mask is not None:
                    image[mask] = (0.55 * image[mask] + 0.45 * np.asarray([60, 170, 60])).astype(np.uint8)
            color = (220, 180, 45) if hand.get("handedness") == "Left" else (70, 90, 235)
            draw_points(image, info.get("hamer_joints_2d_px") or [], (120, 120, 120), "HaMeR2D")
            draw_points(image, info.get("joints_2d_px") or [], color, f"{hand.get('handedness')} {'used' if info.get('used') else 'rej'}")
        out_path = output_dir / "debug" / suffix / camera_id / f"{int(frame['group_id']):08d}.jpg"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), image)


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    return value


def update_camera_diagnostics(camera_diag: dict[str, dict[str, float]], hand: dict[str, Any]) -> None:
    for camera_id, info in (hand.get("projection_debug") or {}).items():
        diag = camera_diag.setdefault(
            str(camera_id),
            {
                "observations": 0.0,
                "soft_used": 0.0,
                "metric_used": 0.0,
                "rejected": 0.0,
                "reprojection_error_sum_px": 0.0,
                "mask_distance_sum_px": 0.0,
                "mask_distance_count": 0.0,
                "final_view_weight_sum": 0.0,
            },
        )
        diag["observations"] += 1.0
        if info.get("used"):
            diag["soft_used"] += 1.0
        else:
            diag["rejected"] += 1.0
        if info.get("metric_used"):
            diag["metric_used"] += 1.0
        reproj = info.get("mean_reprojection_error_px")
        if isinstance(reproj, (int, float)) and math.isfinite(float(reproj)):
            diag["reprojection_error_sum_px"] += float(reproj)
        mask_dist = info.get("mean_mask_distance_px")
        if isinstance(mask_dist, (int, float)) and math.isfinite(float(mask_dist)):
            diag["mask_distance_sum_px"] += float(mask_dist)
            diag["mask_distance_count"] += 1.0
        view_weight = info.get("final_view_weight")
        if isinstance(view_weight, (int, float)) and math.isfinite(float(view_weight)):
            diag["final_view_weight_sum"] += float(view_weight)


def finalize_camera_diagnostics(camera_diag: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for camera_id, diag in sorted(camera_diag.items()):
        observations = max(diag.get("observations", 0.0), 1.0)
        mask_count = max(diag.get("mask_distance_count", 0.0), 1.0)
        item = dict(diag)
        item["mean_reprojection_error_px"] = diag.get("reprojection_error_sum_px", 0.0) / observations
        item["mean_mask_distance_px"] = diag.get("mask_distance_sum_px", 0.0) / mask_count
        item["mean_final_view_weight"] = diag.get("final_view_weight_sum", 0.0) / observations
        item["soft_used_rate"] = diag.get("soft_used", 0.0) / observations
        item["metric_used_rate"] = diag.get("metric_used", 0.0) / observations
        out[camera_id] = item
    return out


def main() -> None:
    warnings.filterwarnings("ignore")
    args = parse_args()
    if args.max_iters < 0 or args.beta_iters < 0:
        raise SystemExit("--max-iters/--beta-iters must be non-negative")
    if args.window_size < 1:
        raise SystemExit("--window-size must be positive")
    group_ids = parse_group_ids(args.group_range, args.group_ids)
    suffix = range_suffix(group_ids)
    if args.predictions:
        prediction_paths = args.predictions
    else:
        ranged_prediction = DEFAULT_BASE_DIR / "hamer_per_view" / f"hamer_predictions_{suffix}.jsonl"
        prediction_paths = [ranged_prediction] if ranged_prediction.exists() else [Path(item) for item in sorted(glob.glob(args.predictions_glob))]
    output_path = args.output_dir / f"mano_multiview_local_hands_{suffix}.jsonl"
    stats_path = args.output_dir / f"refine_stats_{suffix}.json"
    partial_output_path = output_path.with_name(f".{output_path.name}.partial")
    partial_stats_path = stats_path.with_name(f".{stats_path.name}.partial")
    if args.dry_run:
        print(json.dumps({"prediction_files": [str(path) for path in prediction_paths], "output_path": str(output_path)}, indent=2))
        return
    if not prediction_paths:
        raise SystemExit("no prediction files found")
    if output_path.exists() and not args.overwrite:
        raise SystemExit(f"{output_path} exists; pass --overwrite to replace it")
    partial_output_path.unlink(missing_ok=True)
    partial_stats_path.unlink(missing_ok=True)

    predictions = load_predictions(prediction_paths, group_ids, args)
    prediction_records = [
        record
        for by_hand in predictions.values()
        for by_camera in by_hand.values()
        for records in by_camera.values()
        for record in records
    ]
    readable_mask_count = sum(
        bool(record.get("sam3_mask_path")) and Path(str(record["sam3_mask_path"])).exists()
        for record in prediction_records
    )
    mask_ratio = readable_mask_count / max(len(prediction_records), 1)
    if mask_ratio < args.min_readable_sam3_mask_ratio:
        raise SystemExit(
            "Readable SAM3 mask coverage is too low "
            f"({readable_mask_count}/{len(prediction_records)}={mask_ratio:.1%}); "
            "re-run fusion and HaMeR predictions after resolving SAM3 mask paths, "
            "or explicitly pass --min-readable-sam3-mask-ratio 0 for legacy mask-free input."
        )

    args.hamer_root = args.hamer_root.expanduser().resolve()
    checkpoint_path = resolve_hamer_checkpoint(args.hamer_root, args.checkpoint)
    if not checkpoint_path.exists():
        raise SystemExit(f"HaMeR checkpoint not found: {checkpoint_path}")
    sys.path.insert(0, str(args.hamer_root))
    from hamer.models import load_hamer  # type: ignore

    original_cwd = Path.cwd()
    os.chdir(args.hamer_root)
    try:
        model, _model_cfg = load_hamer(str(checkpoint_path))
    finally:
        os.chdir(original_cwd)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    mano = model.mano.to(device).eval()
    for param in mano.parameters():
        param.requires_grad_(False)

    intrinsics = load_rectified_intrinsics(args.rectified_config, args.calib, args.rectify_focal_scale)
    args._rectified_intrinsics = intrinsics
    args._projection_corrections = load_projection_corrections(args.projection_correction)
    cameras_np = load_camera_geometry(args.calib, intrinsics)
    cameras_torch = {camera_id: torch_camera(camera, device) for camera_id, camera in cameras_np.items()}
    mediapipe = load_mediapipe_2d(args.mediapipe, group_ids, args.mediapipe_min_score) if args.use_mediapipe_2d != "never" else {}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    beta_by_hand = {}
    beta_meta = {}
    for handedness in ("Left", "Right"):
        beta_by_hand[handedness], beta_meta[handedness] = estimate_sequence_beta(mano, predictions, handedness, args, device)
        if args.beta_estimation_space == "image-2d":
            beta_by_hand[handedness], beta_meta[handedness] = refine_sequence_beta_image_2d(
                mano,
                predictions,
                handedness,
                beta_by_hand[handedness],
                beta_meta[handedness],
                cameras_np,
                args,
                device,
            )

    stats = defaultdict(int)
    camera_diag: dict[str, dict[str, float]] = {}
    previous_by_hand: dict[str, dict[str, Any]] = {}
    previous_previous_by_hand: dict[str, dict[str, Any]] = {}
    # Keep incomplete work explicitly marked as partial. A hard CUDA/process
    # termination must never leave a valid-looking but truncated final JSONL.
    with partial_output_path.open("w", encoding="utf-8") as out:
        for group_id in tqdm(sorted(predictions), desc="MANO image refine", unit="frame", position=args.progress_position):
            hands = []
            frame_record = {"type": "hamer_mano_multiview_image_refined_frame", "group_id": group_id, "hands": hands}
            for handedness in ("Left", "Right"):
                previous = previous_by_hand.get(handedness)
                previous_previous = previous_previous_by_hand.get(handedness)
                previous_palm = None if previous is None else np.asarray(previous.get("palm_local_joints_m"), dtype=np.float64)
                selected = choose_per_camera(predictions[group_id].get(handedness, {}), previous_palm, args)
                attach_mediapipe_2d(selected, mediapipe, group_id, handedness, args)
                hand = refine_hand(
                    mano,
                    group_id,
                    handedness,
                    selected,
                    beta_by_hand[handedness],
                    beta_meta[handedness],
                    previous,
                    previous_previous,
                    cameras_np,
                    cameras_torch,
                    args,
                    device,
                )
                if hand is None:
                    continue
                hand["joints"] = output_joints(hand)
                hand["metric_joint_count"] = len(hand["joints"]) if hand.get("metric_valid") else 0
                hand["temporal_fallback_joint_count"] = 0 if hand.get("metric_valid") else len(hand["joints"])
                hands.append(hand)
                if previous is not None:
                    previous_previous_by_hand[handedness] = previous
                previous_by_hand[handedness] = hand
                stats["hands"] += 1
                stats[f"mode:{hand['mode']}"] += 1
                stats[f"anchor_camera:{hand.get('anchor_camera')}"] += 1
                if hand.get("metric_valid"):
                    stats["metric_hands"] += 1
                stats["used_camera_observations"] += len(hand.get("used_cameras") or [])
                stats["metric_camera_observations"] += len(hand.get("metric_cameras") or [])
                stats["rejected_camera_observations"] += len(hand.get("rejected_cameras") or [])
                update_camera_diagnostics(camera_diag, hand)
            out.write(json.dumps(frame_record, ensure_ascii=False, separators=(",", ":")) + "\n")
            if args.save_debug_overlays:
                save_debug_overlays(frame_record, args.output_dir, suffix, args.debug_every)
            stats["frames"] += 1
            # The optimizer creates several per-view image tensors. Releasing the
            # allocator cache between frames avoids long-range fragmentation when
            # a full multi-camera sequence is processed in one CUDA process.
            if (
                args.cuda_cache_clear_every > 0
                and stats["frames"] % args.cuda_cache_clear_every == 0
                and torch.cuda.is_available()
            ):
                torch.cuda.empty_cache()

    with partial_stats_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "prediction_files": [str(path) for path in prediction_paths],
                "readable_sam3_masks": readable_mask_count,
                "prediction_count": len(prediction_records),
                "readable_sam3_mask_ratio": mask_ratio,
                "output_path": str(output_path),
                "checkpoint": str(checkpoint_path),
                "rectified_config": str(args.rectified_config) if args.rectified_config else None,
                "calib": str(args.calib),
                "beta_by_hand": {key: value.tolist() for key, value in beta_by_hand.items()},
                "beta_meta": beta_meta,
                "args": json_safe(vars(args)),
                "stats": dict(stats),
                "camera_diagnostics": finalize_camera_diagnostics(camera_diag),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
        f.write("\n")
    os.replace(partial_output_path, output_path)
    os.replace(partial_stats_path, stats_path)
    print("Summary")
    for key in sorted(stats):
        print(f"  {key}: {stats[key]}")
    print(f"wrote: {output_path}")


if __name__ == "__main__":
    main()
