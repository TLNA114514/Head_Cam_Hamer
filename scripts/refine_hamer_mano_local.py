#!/usr/bin/env python3
"""Refine multi-view HaMeR predictions in hand-local MANO pose/beta space."""

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

import numpy as np
import torch

from hamer_multiview_utils import DEFAULT_BASE_DIR, iter_jsonl, parse_group_ids, range_suffix
from progress_utils import tqdm


WRIST_CAM_ROOT = Path("/home/luojiangrui/ljr/wrist_cam")
DEFAULT_HAMER_ROOT = WRIST_CAM_ROOT / "third_party" / "hamer"
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
    parser.add_argument("--local-hands", type=Path, help="Level1 hamer_local_hands JSONL. Defaults by selected range.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_BASE_DIR / "hamer_mano_local_refined")
    parser.add_argument("--hamer-root", type=Path, default=DEFAULT_HAMER_ROOT)
    parser.add_argument("--checkpoint")
    parser.add_argument("--group-range")
    parser.add_argument("--group-ids")
    parser.add_argument("--image-width", type=int, default=1600)
    parser.add_argument("--image-height", type=int, default=1200)
    parser.add_argument("--local-consistency-threshold-m", type=float, default=0.025)
    parser.add_argument("--temporal-error-cap-m", type=float, default=0.08)
    parser.add_argument("--temporal-reject-threshold-m", type=float, default=0.12)
    parser.add_argument("--anchor-switch-margin", type=float, default=0.04)
    parser.add_argument("--min-candidate-score", type=float, default=-0.25)
    parser.add_argument("--smoothing-alpha", type=float, default=0.55)
    parser.add_argument("--pose-prior-weight", type=float, default=0.25)
    parser.add_argument("--beta-prior-weight", type=float, default=1.0)
    parser.add_argument("--temporal-pose-weight", type=float, default=0.35)
    parser.add_argument("--vertex-loss-weight", type=float, default=0.10)
    parser.add_argument("--joint-loss-weight", type=float, default=1.0)
    parser.add_argument("--level1-score-weight", type=float, default=0.15)
    parser.add_argument("--quality-mask-weight", type=float, default=0.55)
    parser.add_argument("--quality-bbox-weight", type=float, default=0.15)
    parser.add_argument("--quality-edge-weight", type=float, default=0.12)
    parser.add_argument("--quality-source-bonus", type=float, default=0.06)
    parser.add_argument("--quality-known-bonus", type=float, default=0.05)
    parser.add_argument("--optimize-mano-pose", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--optimize-mano-betas", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-frame-beta-delta", action="store_true")
    parser.add_argument("--max-frame-beta-delta-norm", type=float, default=0.15)
    parser.add_argument("--max-iters", type=int, default=80)
    parser.add_argument("--beta-iters", type=int, default=120)
    parser.add_argument("--max-beta-observations", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.025)
    parser.add_argument("--beta-learning-rate", type=float, default=0.03)
    parser.add_argument("--vertex-subsample", type=int, default=8)
    parser.add_argument("--progress-position", type=int, default=int(os.environ.get("TQDM_POSITION", "0")))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def is_xyz_list(value: Any) -> bool:
    return isinstance(value, list) and len(value) > 0 and isinstance(value[0], list) and len(value[0]) == 3


def finite_array(value: Any, shape_tail: tuple[int, ...] | None = None) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64)
    if shape_tail is not None and tuple(arr.shape[-len(shape_tail):]) != shape_tail:
        return None
    if not np.all(np.isfinite(arr)):
        return None
    return arr


def local_points(points: Any) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float64)
    return arr - arr[0:1]


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


def apply_kabsch(points: np.ndarray, r: np.ndarray, src_c: np.ndarray, tgt_c: np.ndarray) -> np.ndarray:
    return (points - src_c) @ r.T + tgt_c


def palm_frame_np(joints: np.ndarray) -> np.ndarray:
    wrist = joints[0]
    x_axis = joints[5] - wrist
    y_hint = joints[17] - wrist
    x_norm = np.linalg.norm(x_axis)
    if x_norm < 1e-8:
        x_axis = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        x_axis = x_axis / x_norm
    z_axis = np.cross(x_axis, y_hint)
    z_norm = np.linalg.norm(z_axis)
    if z_norm < 1e-8:
        z_axis = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        z_axis = z_axis / z_norm
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
    x_axis = joints[:, 5, :] - joints[:, 0, :]
    y_hint = joints[:, 17, :] - joints[:, 0, :]
    x_axis = torch.nn.functional.normalize(x_axis, dim=-1, eps=1e-8)
    z_axis = torch.cross(x_axis, y_hint, dim=-1)
    z_axis = torch.nn.functional.normalize(z_axis, dim=-1, eps=1e-8)
    y_axis = torch.cross(z_axis, x_axis, dim=-1)
    y_axis = torch.nn.functional.normalize(y_axis, dim=-1, eps=1e-8)
    basis = torch.stack([x_axis, y_axis, z_axis], dim=-1)
    return torch.matmul(points - wrist, basis)


def weighted_median(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    out = []
    weights = np.asarray(weights, dtype=np.float64)
    weights = np.maximum(weights, 1e-8)
    for dim in range(values.shape[1]):
        order = np.argsort(values[:, dim])
        sorted_values = values[order, dim]
        sorted_weights = weights[order]
        cutoff = 0.5 * sorted_weights.sum()
        index = int(np.searchsorted(np.cumsum(sorted_weights), cutoff, side="left"))
        out.append(float(sorted_values[min(index, len(sorted_values) - 1)]))
    return np.asarray(out, dtype=np.float64)


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def bbox_area_score(bbox: list[float] | None, image_width: int, image_height: int) -> float:
    if not bbox or len(bbox) != 4:
        return 0.0
    x1, y1, x2, y2 = [float(v) for v in bbox]
    area_ratio = max(0.0, x2 - x1) * max(0.0, y2 - y1) / float(image_width * image_height)
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


def source_bonus(source: str | None, args: argparse.Namespace) -> float:
    if source == "mediapipe+sam3":
        return float(args.quality_source_bonus)
    if source in {"sam3", "mediapipe"}:
        return 0.5 * float(args.quality_source_bonus)
    return 0.0


def prediction_quality(rec: dict[str, Any], args: argparse.Namespace) -> tuple[float, dict[str, float]]:
    mask_raw = rec.get("mask_score")
    mask_input = clamp01(float(mask_raw)) if isinstance(mask_raw, (int, float)) else 0.28
    mask_part = args.quality_mask_weight * mask_input
    bbox_part = args.quality_bbox_weight * bbox_area_score(rec.get("bbox_rectified_px"), args.image_width, args.image_height)
    edge_part = args.quality_edge_weight * bbox_edge_score(rec.get("bbox_rectified_px"), args.image_width, args.image_height)
    source_part = source_bonus(rec.get("source_detector"), args)
    known_part = args.quality_known_bonus if rec.get("hypothesis_status") == "known" else 0.0
    scale_penalty = -0.04 * abs(float(rec.get("bbox_scale", 1.0)) - 1.0)
    score = mask_part + bbox_part + edge_part + source_part + known_part + scale_penalty
    return float(score), {
        "mask": float(mask_part),
        "bbox_area": float(bbox_part),
        "edge": float(edge_part),
        "source": float(source_part),
        "known": float(known_part),
        "bbox_scale_penalty": float(scale_penalty),
    }


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


def resolve_hamer_checkpoint(hamer_root: Path, checkpoint: str | None) -> Path:
    if checkpoint:
        path = Path(checkpoint).expanduser()
        if not path.is_absolute():
            path = hamer_root / path
    else:
        path = hamer_root / "_DATA" / "hamer_ckpts" / "checkpoints" / "hamer.ckpt"
    return path.resolve()


def load_predictions(paths: list[Path], group_ids: set[int] | None) -> dict[int, dict[str, dict[str, list[dict[str, Any]]]]]:
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
            if not is_xyz_list(rec.get("hamer_joints_cam")):
                continue
            if not rec.get("mano_params_rotmat"):
                continue
            data[group_id][handedness][camera_id].append(rec)
    return data


def load_level1(path: Path | None, group_ids: set[int] | None) -> dict[int, dict[str, dict[str, Any]]]:
    data: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    if path is None or not path.exists():
        return data
    for rec in iter_jsonl(path):
        if rec.get("type") != "hamer_primary_local_frame":
            continue
        group_id = int(rec["group_id"])
        if group_ids is not None and group_id not in group_ids:
            continue
        for hand in rec.get("hands") or []:
            handedness = hand.get("handedness")
            if handedness in {"Left", "Right"}:
                data[group_id][handedness] = hand
    return data


def candidate_from_prediction(
    rec: dict[str, Any],
    level1_hand: dict[str, Any] | None,
    previous_palm_joints: np.ndarray | None,
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    params = rec.get("mano_params_rotmat") or {}
    hand_pose = finite_array(params.get("hand_pose"), (3, 3))
    beta = finite_array(params.get("betas"))
    joints = finite_array(rec.get("hamer_joints_cam"), (3,))
    verts = finite_array(rec.get("hamer_vertices_cam"), (3,))
    if hand_pose is None or beta is None or joints is None or verts is None:
        return None
    if hand_pose.shape != (15, 3, 3) or beta.shape != (10,):
        return None
    local_joints = local_points(joints)
    local_verts = local_points(verts)
    palm_joints = palm_local_np(local_joints)
    palm_verts = palm_local_np(local_verts, local_joints)
    base_score, parts = prediction_quality(rec, args)
    level1_score = 0.0
    if level1_hand:
        scores = level1_hand.get("anchor_selection_scores") or level1_hand.get("temporal_selection_scores") or {}
        raw = scores.get(rec.get("camera_id"))
        if isinstance(raw, (int, float)) and math.isfinite(float(raw)):
            level1_score = float(args.level1_score_weight) * float(raw)
    temporal_error = None
    temporal_penalty = 0.0
    if previous_palm_joints is not None:
        temporal_error = float(np.mean(np.linalg.norm(palm_joints - previous_palm_joints, axis=1)))
        temporal_penalty = -min(temporal_error, args.temporal_error_cap_m) / max(args.temporal_error_cap_m, 1e-8)
    score = base_score + level1_score + 0.25 * temporal_penalty
    out = dict(rec)
    out.update(
        {
            "_hand_pose": hand_pose,
            "_betas": beta,
            "_local_joints": local_joints,
            "_local_vertices": local_verts,
            "_palm_joints": palm_joints,
            "_palm_vertices": palm_verts,
            "_score": float(score),
            "_quality_score": float(base_score),
            "_quality_parts": parts,
            "_level1_score_bonus": float(level1_score),
            "_temporal_pose_error_m": temporal_error,
        }
    )
    return out


def choose_per_camera_candidates(
    records_by_camera: dict[str, list[dict[str, Any]]],
    level1_hand: dict[str, Any] | None,
    previous_palm_joints: np.ndarray | None,
    args: argparse.Namespace,
) -> dict[str, dict[str, Any]]:
    selected = {}
    for camera_id, items in records_by_camera.items():
        candidates = [
            item
            for rec in items
            if (item := candidate_from_prediction(rec, level1_hand, previous_palm_joints, args)) is not None
        ]
        if not candidates:
            continue
        selected[camera_id] = max(candidates, key=lambda item: item["_score"])
    return selected


def select_anchor_candidate(selected: dict[str, dict[str, Any]], level1_hand: dict[str, Any] | None, args: argparse.Namespace) -> tuple[str, dict[str, Any], dict[str, Any]]:
    if not selected:
        raise ValueError("no candidates")
    level1_anchor = level1_hand.get("anchor_camera") if level1_hand else None
    best_camera, best = max(sorted(selected.items()), key=lambda item: item[1]["_score"])
    reason = "highest_score"
    if level1_anchor in selected and best_camera != level1_anchor:
        anchor_score = selected[level1_anchor]["_score"]
        if best["_score"] <= anchor_score + args.anchor_switch_margin:
            best_camera = str(level1_anchor)
            best = selected[best_camera]
            reason = "level1_anchor_within_margin"
        else:
            reason = "non_level1_anchor_higher_score"
    elif level1_anchor == best_camera:
        reason = "level1_anchor_highest_score"
    return best_camera, best, {
        "anchor_selection_reason": reason,
        "candidate_scores": {camera_id: float(item["_score"]) for camera_id, item in selected.items()},
        "candidate_quality_scores": {camera_id: float(item["_quality_score"]) for camera_id, item in selected.items()},
        "candidate_temporal_errors_m": {camera_id: item["_temporal_pose_error_m"] for camera_id, item in selected.items()},
    }


def build_targets(anchor: dict[str, Any], selected: dict[str, dict[str, Any]], args: argparse.Namespace) -> tuple[list[str], list[str], dict[str, float], np.ndarray, np.ndarray]:
    used = [str(anchor["camera_id"])]
    rejected = []
    errors = {}
    weighted_joints = [anchor["_palm_joints"] * max(anchor["_score"], 1e-3)]
    weights = [max(anchor["_score"], 1e-3)]
    target_vertices = anchor["_palm_vertices"]
    for camera_id, item in sorted(selected.items()):
        if camera_id == anchor["camera_id"]:
            continue
        _aligned, r, src_c, tgt_c, err = kabsch_transform(item["_local_joints"], anchor["_local_joints"])
        errors[camera_id] = err
        if err <= args.local_consistency_threshold_m:
            used.append(camera_id)
            aligned_joints = apply_kabsch(item["_local_joints"], r, src_c, tgt_c)
            aligned_palm = palm_local_np(aligned_joints)
            w = max(item["_score"], 1e-3)
            weighted_joints.append(aligned_palm * w)
            weights.append(w)
        else:
            rejected.append(camera_id)
    target_joints = np.sum(np.stack(weighted_joints, axis=0), axis=0) / float(np.sum(weights))
    return used, rejected, errors, target_joints, target_vertices


def mano_forward_local(
    mano: Any,
    hand_pose_rotmat: torch.Tensor,
    betas: torch.Tensor,
    is_right: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = hand_pose_rotmat.shape[0]
    global_orient = torch.eye(3, dtype=hand_pose_rotmat.dtype, device=hand_pose_rotmat.device).reshape(1, 1, 3, 3).repeat(batch_size, 1, 1, 1)
    output = mano(global_orient=global_orient.float(), hand_pose=hand_pose_rotmat.float(), betas=betas.float(), pose2rot=False)
    joints = output.joints[:, :21, :]
    verts = output.vertices
    multiplier = float(2 * int(is_right) - 1)
    joints = joints.clone()
    verts = verts.clone()
    joints[:, :, 0] *= multiplier
    verts[:, :, 0] *= multiplier
    wrist = joints[:, 0:1, :].clone()
    joints = joints - wrist
    verts = verts - wrist
    return joints, verts


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
                item = candidate_from_prediction(rec, None, None, args)
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
        "beta_source": "optimized_sequence",
        "beta_observations": len(observations),
        "beta_fit_cost": final_loss,
        "beta_prior": beta_prior.tolist(),
    }


def smooth_rotmat(current: np.ndarray, previous: np.ndarray | None, alpha: float) -> np.ndarray:
    if previous is None:
        return current
    current_6d = rotmat_to_6d(torch.tensor(current, dtype=torch.float32))
    previous_6d = rotmat_to_6d(torch.tensor(previous, dtype=torch.float32))
    mixed = float(alpha) * current_6d + (1.0 - float(alpha)) * previous_6d
    return rot6d_to_rotmat(mixed).detach().cpu().numpy()


def refine_hand(
    mano: Any,
    group_id: int,
    handedness: str,
    selected: dict[str, dict[str, Any]],
    level1_hand: dict[str, Any] | None,
    sequence_beta: np.ndarray,
    beta_info: dict[str, Any],
    previous: dict[str, Any] | None,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any] | None:
    previous_palm = None if previous is None else np.asarray(previous.get("palm_local_joints_m"), dtype=np.float64)
    if not selected:
        if previous is None:
            return None
        hand = dict(previous)
        hand["group_id"] = group_id
        hand["mode"] = "mano_pose_beta_temporal_fallback"
        hand["metric_valid"] = False
        hand["fit_success"] = False
        hand["fit_cost"] = None
        hand["fallback_reason"] = "missing_candidates"
        return hand

    anchor_camera, anchor, selection_info = select_anchor_candidate(selected, level1_hand, args)
    if anchor["_score"] < args.min_candidate_score and previous is not None:
        hand = dict(previous)
        hand["group_id"] = group_id
        hand["mode"] = "mano_pose_beta_temporal_fallback"
        hand["metric_valid"] = False
        hand["fit_success"] = False
        hand["fit_cost"] = None
        hand["fallback_reason"] = "low_candidate_score"
        return hand
    if previous_palm is not None and anchor.get("_temporal_pose_error_m") is not None and anchor["_temporal_pose_error_m"] > args.temporal_reject_threshold_m:
        alternatives = [item for item in selected.values() if item.get("_temporal_pose_error_m") is not None and item["_temporal_pose_error_m"] <= args.temporal_reject_threshold_m]
        if alternatives:
            anchor = max(alternatives, key=lambda item: item["_score"])
            anchor_camera = str(anchor["camera_id"])
            selection_info["anchor_selection_reason"] = "temporal_reject_switched_anchor"
        else:
            hand = dict(previous)
            hand["group_id"] = group_id
            hand["mode"] = "mano_pose_beta_temporal_fallback"
            hand["metric_valid"] = False
            hand["fit_success"] = False
            hand["fit_cost"] = None
            hand["fallback_reason"] = "temporal_reject"
            return hand

    used, rejected, consistency_errors, target_joints_np, target_vertices_np = build_targets(anchor, selected, args)
    target_joints = torch.tensor(target_joints_np[None], dtype=torch.float32, device=device)
    vertex_step = max(1, int(args.vertex_subsample))
    target_vertices = torch.tensor(target_vertices_np[None, ::vertex_step], dtype=torch.float32, device=device)
    anchor_pose = torch.tensor(anchor["_hand_pose"], dtype=torch.float32, device=device)
    pose_6d = rotmat_to_6d(anchor_pose).detach().clone().requires_grad_(bool(args.optimize_mano_pose))
    beta_base = torch.tensor(sequence_beta, dtype=torch.float32, device=device)
    beta_delta = torch.zeros(10, dtype=torch.float32, device=device, requires_grad=bool(args.allow_frame_beta_delta))
    params = []
    if args.optimize_mano_pose:
        params.append(pose_6d)
    if args.allow_frame_beta_delta:
        params.append(beta_delta)
    optimizer = torch.optim.Adam(params, lr=args.learning_rate) if params else None
    prev_pose_tensor = None
    if previous is not None and previous.get("mano_params_refined", {}).get("hand_pose") is not None:
        prev_pose_tensor = torch.tensor(previous["mano_params_refined"]["hand_pose"], dtype=torch.float32, device=device)
    final_loss = None
    iters = max(1, args.max_iters if params else 1)
    for _ in range(iters):
        if optimizer:
            optimizer.zero_grad(set_to_none=True)
        pose_rot = rot6d_to_rotmat(pose_6d).reshape(1, 15, 3, 3)
        beta = (beta_base + beta_delta).reshape(1, 10)
        pred_joints, pred_verts = mano_forward_local(mano, pose_rot, beta, int(anchor.get("is_right", 1 if handedness == "Right" else 0)))
        pred_palm_joints = palm_local_torch(pred_joints)
        pred_palm_vertices = palm_local_torch(pred_verts[:, ::vertex_step], pred_joints)
        joint_loss = ((pred_palm_joints - target_joints) ** 2).mean()
        vertex_loss = ((pred_palm_vertices - target_vertices) ** 2).mean()
        pose_prior = ((pose_rot.reshape(15, 3, 3) - anchor_pose) ** 2).mean()
        beta_prior = (beta_delta ** 2).mean()
        temporal_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
        if prev_pose_tensor is not None:
            temporal_loss = ((pose_rot.reshape(15, 3, 3) - prev_pose_tensor) ** 2).mean()
        loss = (
            args.joint_loss_weight * joint_loss
            + args.vertex_loss_weight * vertex_loss
            + args.pose_prior_weight * pose_prior
            + args.beta_prior_weight * beta_prior
            + args.temporal_pose_weight * temporal_loss
        )
        if optimizer:
            loss.backward()
            optimizer.step()
            if args.allow_frame_beta_delta:
                with torch.no_grad():
                    norm = torch.linalg.norm(beta_delta)
                    if norm > args.max_frame_beta_delta_norm:
                        beta_delta.mul_(args.max_frame_beta_delta_norm / (norm + 1e-8))
                    beta_delta.clamp_(-0.25, 0.25)
        final_loss = float(loss.detach().cpu().item())

    pose_rot = rot6d_to_rotmat(pose_6d).detach().cpu().numpy()
    pose_rot = smooth_rotmat(pose_rot, None if previous is None else np.asarray(previous["mano_params_refined"]["hand_pose"], dtype=np.float64), args.smoothing_alpha)
    beta_np = np.clip((beta_base + beta_delta).detach().cpu().numpy(), -3.0, 3.0)
    with torch.no_grad():
        pose_tensor = torch.tensor(pose_rot[None], dtype=torch.float32, device=device)
        beta_tensor = torch.tensor(beta_np[None], dtype=torch.float32, device=device)
        joints_t, verts_t = mano_forward_local(mano, pose_tensor, beta_tensor, int(anchor.get("is_right", 1 if handedness == "Right" else 0)))
    local_joints = joints_t[0].detach().cpu().numpy().astype(np.float64)
    local_vertices = verts_t[0].detach().cpu().numpy().astype(np.float64)
    palm_joints = palm_local_np(local_joints)
    palm_vertices = palm_local_np(local_vertices, local_joints)
    temporal_error = None
    if previous_palm is not None:
        temporal_error = float(np.mean(np.linalg.norm(palm_joints - previous_palm, axis=1)))
    mode = "mano_pose_beta_refined_fused" if len(used) > 1 else "mano_pose_beta_single_view"
    metric_valid = len(used) > 1
    return {
        "group_id": group_id,
        "handedness": handedness,
        "mode": mode,
        "metric_valid": bool(metric_valid),
        "fit_success": True,
        "fit_cost": final_loss,
        "anchor_camera": anchor_camera,
        "primary_camera": anchor_camera,
        "used_cameras": used,
        "rejected_cameras": rejected,
        "consistency_errors_m": consistency_errors,
        "anchor_score": float(anchor["_score"]),
        "anchor_quality_score": float(anchor["_quality_score"]),
        "pose_error_m": temporal_error,
        "beta_source": beta_info.get("beta_source"),
        "beta_observations": beta_info.get("beta_observations"),
        "local_joints_m": local_joints.tolist(),
        "local_vertices_m": local_vertices.tolist(),
        "palm_local_joints_m": palm_joints.tolist(),
        "palm_local_vertices_m": palm_vertices.tolist(),
        "mano_params_refined": {
            "hand_pose": pose_rot.tolist(),
            "betas": beta_np.tolist(),
            "global_orient": np.eye(3, dtype=np.float64).reshape(1, 3, 3).tolist(),
        },
        "mano_params_anchor": {
            "hand_pose": anchor["_hand_pose"].tolist(),
            "betas": anchor["_betas"].tolist(),
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


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    return value


def main() -> None:
    warnings.filterwarnings("ignore")
    args = parse_args()
    if args.max_iters < 0 or args.beta_iters < 0:
        raise SystemExit("--max-iters/--beta-iters must be non-negative")
    if args.local_consistency_threshold_m <= 0:
        raise SystemExit("--local-consistency-threshold-m must be positive")
    group_ids = parse_group_ids(args.group_range, args.group_ids)
    suffix = range_suffix(group_ids)
    if args.predictions:
        prediction_paths = args.predictions
    else:
        ranged_prediction = DEFAULT_BASE_DIR / "hamer_per_view" / f"hamer_predictions_{suffix}.jsonl"
        prediction_paths = [ranged_prediction] if ranged_prediction.exists() else [Path(item) for item in sorted(glob.glob(args.predictions_glob))]
    local_path = args.local_hands or (DEFAULT_BASE_DIR / "hamer_primary_local" / f"hamer_local_hands_{suffix}.jsonl")
    output_path = args.output_dir / f"mano_local_hands_{suffix}.jsonl"
    stats_path = args.output_dir / f"refine_stats_{suffix}.json"
    if args.dry_run:
        print(
            json.dumps(
                {
                    "prediction_files": [str(path) for path in prediction_paths],
                    "local_hands": str(local_path),
                    "output_path": str(output_path),
                },
                indent=2,
            )
        )
        return
    if not prediction_paths:
        raise SystemExit("no prediction files found")
    if output_path.exists() and not args.overwrite:
        raise SystemExit(f"{output_path} exists; pass --overwrite to replace it")

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

    predictions = load_predictions(prediction_paths, group_ids)
    level1 = load_level1(local_path, group_ids)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    beta_by_hand = {}
    beta_meta = {}
    for handedness in ("Left", "Right"):
        beta_by_hand[handedness], beta_meta[handedness] = estimate_sequence_beta(mano, predictions, handedness, args, device)

    stats = defaultdict(int)
    previous_by_hand: dict[str, dict[str, Any]] = {}
    with output_path.open("w", encoding="utf-8") as out:
        for group_id in tqdm(sorted(predictions), desc="MANO local refine", unit="frame", position=args.progress_position):
            hands = []
            for handedness in ("Left", "Right"):
                selected = choose_per_camera_candidates(
                    predictions[group_id].get(handedness, {}),
                    level1.get(group_id, {}).get(handedness),
                    None if handedness not in previous_by_hand else np.asarray(previous_by_hand[handedness].get("palm_local_joints_m"), dtype=np.float64),
                    args,
                )
                hand = refine_hand(
                    mano,
                    group_id,
                    handedness,
                    selected,
                    level1.get(group_id, {}).get(handedness),
                    beta_by_hand[handedness],
                    beta_meta[handedness],
                    previous_by_hand.get(handedness),
                    args,
                    device,
                )
                if hand is None:
                    continue
                hand["joints"] = output_joints(hand)
                hand["metric_joint_count"] = len(hand["joints"]) if hand.get("metric_valid") else 0
                hand["temporal_fallback_joint_count"] = 0 if hand.get("metric_valid") else len(hand["joints"])
                hands.append(hand)
                previous_by_hand[handedness] = hand
                stats["hands"] += 1
                stats[f"mode:{hand['mode']}"] += 1
                stats[f"anchor_camera:{hand.get('anchor_camera')}"] += 1
                if hand.get("metric_valid"):
                    stats["metric_hands"] += 1
            out.write(
                json.dumps(
                    {
                        "type": "hamer_mano_local_refined_frame",
                        "group_id": group_id,
                        "hands": hands,
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )
            stats["frames"] += 1

    with stats_path.open("w", encoding="utf-8") as f:
        serializable_args = json_safe(vars(args))
        json.dump(
            {
                "prediction_files": [str(path) for path in prediction_paths],
                "local_hands": str(local_path),
                "output_path": str(output_path),
                "checkpoint": str(checkpoint_path),
                "beta_by_hand": {key: value.tolist() for key, value in beta_by_hand.items()},
                "beta_meta": beta_meta,
                "args": serializable_args,
                "stats": dict(stats),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
        f.write("\n")
    print("Summary")
    for key in sorted(stats):
        print(f"  {key}: {stats[key]}")
    print(f"wrote: {output_path}")


if __name__ == "__main__":
    main()
