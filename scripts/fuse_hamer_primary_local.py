#!/usr/bin/env python3
"""Fuse per-view HaMeR predictions into primary-view hand-local outputs."""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from hamer_multiview_utils import (
    BACKUP_PRIMARY_CAMERAS,
    DEFAULT_BASE_DIR,
    PRIMARY_CAMERAS,
    iter_jsonl,
    parse_group_ids,
    range_suffix,
)
from progress_utils import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", action="append", type=Path, help="Prediction JSONL. Can repeat.")
    parser.add_argument("--predictions-glob", default=str(DEFAULT_BASE_DIR / "hamer_per_view" / "hamer_predictions_*.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_BASE_DIR / "hamer_primary_local")
    parser.add_argument("--group-range")
    parser.add_argument("--group-ids")
    parser.add_argument("--consistency-threshold-m", type=float, default=0.025)
    parser.add_argument("--temporal", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--temporal-selection-weight", type=float, default=0.45)
    parser.add_argument("--temporal-error-cap-m", type=float, default=0.08)
    parser.add_argument("--temporal-metric-alpha", type=float, default=0.75)
    parser.add_argument("--temporal-primary-alpha", type=float, default=0.45)
    parser.add_argument("--temporal-backup-alpha", type=float, default=0.40)
    parser.add_argument("--temporal-nonprimary-alpha", type=float, default=0.30)
    parser.add_argument("--temporal-quality-anchor-alpha", type=float, default=0.42)
    parser.add_argument("--image-width", type=int, default=1600)
    parser.add_argument("--image-height", type=int, default=1200)
    parser.add_argument("--quality-mask-weight", type=float, default=0.55)
    parser.add_argument("--quality-bbox-weight", type=float, default=0.15)
    parser.add_argument("--quality-edge-weight", type=float, default=0.12)
    parser.add_argument("--quality-source-bonus", type=float, default=0.06)
    parser.add_argument("--quality-known-bonus", type=float, default=0.05)
    parser.add_argument("--primary-prior-bonus", type=float, default=0.08)
    parser.add_argument("--backup-prior-bonus", type=float, default=0.04)
    parser.add_argument("--consensus-selection-weight", type=float, default=0.35)
    parser.add_argument("--consensus-error-cap-m", type=float, default=0.06)
    parser.add_argument("--anchor-switch-margin", type=float, default=0.04)
    parser.add_argument("--min-anchor-score", type=float, default=-1.0)
    parser.add_argument("--backup-primary-cameras", default="Left:C0,Right:C3")
    parser.add_argument("--backup-min-mask-score", type=float, default=0.25)
    parser.add_argument("--backup-min-view-quality", type=float, default=0.45)
    parser.add_argument("--backup-max-temporal-error-m", type=float, default=0.06)
    parser.add_argument("--backup-require-known", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--progress-position", type=int, default=int(os.environ.get("TQDM_POSITION", "0")))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def is_xyz_list(value: Any) -> bool:
    return isinstance(value, list) and len(value) > 0 and isinstance(value[0], list) and len(value[0]) == 3


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
            data[group_id][handedness][camera_id].append(rec)
    return data


def parse_backup_primary_cameras(value: str | None) -> dict[str, str]:
    if not value or value.strip().lower() in {"none", "off", "false", "0"}:
        return {}
    mapping = dict(BACKUP_PRIMARY_CAMERAS)
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if ":" in part:
            handedness, camera_id = part.split(":", 1)
        elif "=" in part:
            handedness, camera_id = part.split("=", 1)
        else:
            raise ValueError(f"Invalid backup primary mapping: {part}")
        handedness = handedness.strip()
        camera_id = camera_id.strip()
        if handedness not in {"Left", "Right"}:
            raise ValueError(f"Invalid backup primary handedness: {handedness}")
        if not camera_id:
            raise ValueError(f"Invalid backup primary camera for {handedness}")
        mapping[handedness] = camera_id
    return mapping


def local_points(points: list[list[float]]) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float64)
    return arr - arr[0:1]


def kabsch_align(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, float]:
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
    return aligned, err


def pose_error(local_joints: np.ndarray, previous_joints: np.ndarray | None) -> float | None:
    if previous_joints is None or len(local_joints) == 0 or len(previous_joints) == 0:
        return None
    _aligned, err = kabsch_align(local_joints, previous_joints)
    return err


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def bbox_area_score(bbox: list[float] | None, image_width: int, image_height: int) -> float:
    if not bbox or len(bbox) != 4 or image_width <= 0 or image_height <= 0:
        return 0.0
    x1, y1, x2, y2 = [float(v) for v in bbox]
    area_ratio = max(0.0, x2 - x1) * max(0.0, y2 - y1) / float(image_width * image_height)
    # Hands that are extremely tiny are usually weak HaMeR crops; very large crops
    # often include too much arm/background. Keep this as a soft score, not a gate.
    good_min = 0.003
    good_max = 0.12
    if area_ratio <= 0.0:
        return 0.0
    if area_ratio < good_min:
        return clamp01(area_ratio / good_min)
    if area_ratio > good_max:
        return clamp01(good_max / area_ratio)
    return 1.0


def bbox_edge_score(bbox: list[float] | None, image_width: int, image_height: int) -> float:
    if not bbox or len(bbox) != 4 or image_width <= 0 or image_height <= 0:
        return 0.0
    x1, y1, x2, y2 = [float(v) for v in bbox]
    margin = min(x1, y1, float(image_width) - x2, float(image_height) - y2)
    # A little negative margin is tolerated because padded crops can cross the
    # image boundary; deep negative/edge-hugging boxes are likely truncated.
    return clamp01((margin + 20.0) / 80.0)


def source_detector_bonus(source: str | None, args: argparse.Namespace) -> float:
    if source == "mediapipe+sam3":
        return float(args.quality_source_bonus)
    if source in {"sam3", "mediapipe"}:
        return 0.5 * float(args.quality_source_bonus)
    return 0.0


def base_view_quality(rec: dict[str, Any], args: argparse.Namespace) -> tuple[float, dict[str, float]]:
    mask_score_raw = rec.get("mask_score")
    if isinstance(mask_score_raw, (int, float)):
        mask_component_input = clamp01(float(mask_score_raw))
    elif rec.get("used_mask_blur"):
        mask_component_input = 0.0
    else:
        # No SAM mask does not mean the crop is bad; it just lacks silhouette
        # evidence. Give MediaPipe-only jobs a modest neutral baseline.
        mask_component_input = 0.28
    mask_component = float(args.quality_mask_weight) * mask_component_input
    bbox_component = float(args.quality_bbox_weight) * bbox_area_score(
        rec.get("bbox_rectified_px"),
        int(args.image_width),
        int(args.image_height),
    )
    edge_component = float(args.quality_edge_weight) * bbox_edge_score(
        rec.get("bbox_rectified_px"),
        int(args.image_width),
        int(args.image_height),
    )
    source_component = source_detector_bonus(rec.get("source_detector"), args)
    known_component = float(args.quality_known_bonus) if rec.get("hypothesis_status") == "known" else 0.0
    bbox_scale_penalty = 0.04 * abs(float(rec.get("bbox_scale", 1.0)) - 1.0)
    score = mask_component + bbox_component + edge_component + source_component + known_component - bbox_scale_penalty
    parts = {
        "mask": mask_component,
        "bbox_area": bbox_component,
        "edge": edge_component,
        "source": source_component,
        "known": known_component,
        "bbox_scale_penalty": -bbox_scale_penalty,
    }
    return float(score), parts


def temporal_candidate_score(
    rec: dict[str, Any],
    previous_joints: np.ndarray | None,
    args: argparse.Namespace,
) -> tuple[float, float | None]:
    score, _parts = base_view_quality(rec, args)
    err = None
    if args.temporal and previous_joints is not None:
        err = pose_error(local_points(rec["hamer_joints_cam"]), previous_joints)
        if err is not None:
            normalized = min(err, args.temporal_error_cap_m) / max(args.temporal_error_cap_m, 1e-6)
            score -= args.temporal_selection_weight * normalized
    return score, err


def choose_prediction(
    items: list[dict[str, Any]],
    previous_joints: np.ndarray | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    best = None
    best_score = -float("inf")
    best_temporal_error = None
    for rec in items:
        score, temporal_error = temporal_candidate_score(rec, previous_joints, args)
        if score > best_score:
            best = rec
            best_score = score
            best_temporal_error = temporal_error
    assert best is not None
    best = dict(best)
    best["_temporal_selection_score"] = float(best_score)
    best["_temporal_pose_error_m"] = best_temporal_error
    base_score, base_parts = base_view_quality(best, args)
    best["_view_quality_score"] = float(base_score)
    best["_view_quality_parts"] = base_parts
    return best


def smoothing_alpha(mode: str, args: argparse.Namespace) -> float:
    if mode == "primary_hamer_local_fused":
        return float(args.temporal_metric_alpha)
    if mode == "primary_hamer_local":
        return float(args.temporal_primary_alpha)
    if mode in {"backup_primary_hamer_local", "backup_primary_hamer_local_fused"}:
        return float(args.temporal_backup_alpha)
    if mode in {"quality_anchor_hamer_local", "quality_anchor_hamer_local_fused"}:
        return float(args.temporal_quality_anchor_alpha)
    return float(args.temporal_nonprimary_alpha)


def backup_gate(
    candidate: dict[str, Any] | None,
    previous_joints: np.ndarray | None,
    args: argparse.Namespace,
) -> tuple[bool, dict[str, Any]]:
    info: dict[str, Any] = {
        "backup_primary_gate_passed": False,
        "backup_primary_gate_reason": None,
        "backup_primary_mask_score": None,
        "backup_primary_view_quality_score": None,
        "backup_primary_temporal_error_m": None,
    }
    if candidate is None:
        info["backup_primary_gate_reason"] = "missing_candidate"
        return False, info
    if args.backup_require_known and candidate.get("hypothesis_status") != "known":
        info["backup_primary_gate_reason"] = "not_known_hypothesis"
        return False, info
    mask_score = candidate.get("mask_score")
    mask_score_value = float(mask_score) if isinstance(mask_score, (int, float)) else 0.0
    view_quality_value = float(candidate.get("_view_quality_score", 0.0))
    info["backup_primary_mask_score"] = mask_score_value
    info["backup_primary_view_quality_score"] = view_quality_value
    if mask_score_value < args.backup_min_mask_score and view_quality_value < args.backup_min_view_quality:
        info["backup_primary_gate_reason"] = "low_mask_score"
        return False, info
    temporal_error = candidate.get("_temporal_pose_error_m")
    if temporal_error is None and previous_joints is not None:
        temporal_error = pose_error(local_points(candidate["hamer_joints_cam"]), previous_joints)
    info["backup_primary_temporal_error_m"] = temporal_error
    if temporal_error is not None and temporal_error > args.backup_max_temporal_error_m:
        info["backup_primary_gate_reason"] = "temporal_discontinuity"
        return False, info
    info["backup_primary_gate_passed"] = True
    info["backup_primary_gate_reason"] = "passed" if mask_score_value >= args.backup_min_mask_score else "passed_by_view_quality"
    return True, info


def camera_role(camera_id: str, canonical_primary_camera: str, backup_primary_camera: str | None) -> str:
    if camera_id == canonical_primary_camera:
        return "primary"
    if backup_primary_camera and camera_id == backup_primary_camera:
        return "backup_primary"
    return "quality_anchor"


def role_prior_bonus(role: str, args: argparse.Namespace) -> float:
    if role == "primary":
        return float(args.primary_prior_bonus)
    if role == "backup_primary":
        return float(args.backup_prior_bonus)
    return 0.0


def consensus_errors(selected: dict[str, dict[str, Any]]) -> dict[str, float | None]:
    errors: dict[str, list[float]] = {camera_id: [] for camera_id in selected}
    local_by_camera = {
        camera_id: local_points(pred["hamer_joints_cam"])
        for camera_id, pred in selected.items()
        if is_xyz_list(pred.get("hamer_joints_cam"))
    }
    cameras = sorted(local_by_camera)
    for index, camera_a in enumerate(cameras):
        for camera_b in cameras[index + 1:]:
            _aligned, err = kabsch_align(local_by_camera[camera_a], local_by_camera[camera_b])
            errors[camera_a].append(err)
            errors[camera_b].append(err)
    return {
        camera_id: float(np.median(values)) if values else None
        for camera_id, values in errors.items()
    }


def consensus_score(error_m: float | None, args: argparse.Namespace) -> float:
    if error_m is None:
        return 0.0
    cap = max(float(args.consensus_error_cap_m), 1e-6)
    return float(args.consensus_selection_weight) * max(0.0, 1.0 - min(float(error_m), cap) / cap)


def select_anchor(
    selected: dict[str, dict[str, Any]],
    handedness: str,
    canonical_primary_camera: str,
    backup_primary_camera: str | None,
    args: argparse.Namespace,
) -> tuple[str, dict[str, Any], str, dict[str, Any]]:
    consensus_by_camera = consensus_errors(selected)
    scored: dict[str, dict[str, Any]] = {}
    for camera_id, pred in selected.items():
        role = camera_role(camera_id, canonical_primary_camera, backup_primary_camera)
        view_quality = float(pred.get("_view_quality_score", pred.get("_temporal_selection_score", 0.0)))
        temporal_score = float(pred.get("_temporal_selection_score", view_quality))
        temporal_penalty = temporal_score - view_quality
        consensus_bonus = consensus_score(consensus_by_camera.get(camera_id), args)
        prior_bonus = role_prior_bonus(role, args)
        anchor_score = view_quality + temporal_penalty + consensus_bonus + prior_bonus
        scored[camera_id] = {
            "role": role,
            "view_quality_score": view_quality,
            "view_quality_parts": pred.get("_view_quality_parts", {}),
            "temporal_score": temporal_score,
            "temporal_penalty": temporal_penalty,
            "temporal_pose_error_m": pred.get("_temporal_pose_error_m"),
            "consensus_error_m": consensus_by_camera.get(camera_id),
            "consensus_bonus": consensus_bonus,
            "role_prior_bonus": prior_bonus,
            "anchor_score": float(anchor_score),
        }

    best_camera, best_info = max(
        sorted(scored.items()),
        key=lambda item: item[1]["anchor_score"],
    )
    canonical_info = scored.get(canonical_primary_camera)
    reason = "highest_quality_score"
    if best_camera != canonical_primary_camera and canonical_info is not None:
        margin = float(args.anchor_switch_margin)
        if best_info["anchor_score"] <= canonical_info["anchor_score"] + margin:
            best_camera = canonical_primary_camera
            best_info = canonical_info
            reason = "canonical_primary_within_margin"
        else:
            reason = "non_primary_higher_quality"
    elif best_camera == canonical_primary_camera:
        reason = "canonical_primary_highest_score"
    elif best_info["role"] == "backup_primary":
        reason = "backup_primary_higher_quality"
    if best_info["anchor_score"] < float(args.min_anchor_score):
        reason = f"below_min_anchor_score:{reason}"

    anchor = selected[best_camera]
    role = str(best_info["role"])
    return best_camera, anchor, role, {
        "anchor_selection_reason": reason,
        "anchor_score": best_info["anchor_score"],
        "anchor_score_margin_to_canonical": (
            None
            if canonical_info is None
            else float(best_info["anchor_score"] - canonical_info["anchor_score"])
        ),
        "view_quality_scores": {camera_id: info["view_quality_score"] for camera_id, info in scored.items()},
        "view_quality_parts": {camera_id: info["view_quality_parts"] for camera_id, info in scored.items()},
        "anchor_selection_scores": {camera_id: info["anchor_score"] for camera_id, info in scored.items()},
        "anchor_selection_details": scored,
        "selection_consensus_errors_m": {
            camera_id: info["consensus_error_m"] for camera_id, info in scored.items()
        },
    }


def apply_temporal_smoothing(
    current_joints: np.ndarray,
    previous_joints: np.ndarray | None,
    mode: str,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, Any]]:
    info: dict[str, Any] = {
        "temporal_smoothing_applied": False,
        "temporal_alpha": None,
        "temporal_pose_error_m": None,
    }
    if not args.temporal or previous_joints is None or len(current_joints) == 0:
        return current_joints, info
    aligned_previous, err = kabsch_align(previous_joints, current_joints)
    alpha = max(0.0, min(1.0, smoothing_alpha(mode, args)))
    smoothed = alpha * current_joints + (1.0 - alpha) * aligned_previous
    smoothed[0] = 0.0
    info.update(
        {
            "temporal_smoothing_applied": True,
            "temporal_alpha": alpha,
            "temporal_pose_error_m": err,
        }
    )
    return smoothed, info


def fuse_hand(
    group_id: int,
    handedness: str,
    by_camera: dict[str, list[dict[str, Any]]],
    previous_joints: np.ndarray | None,
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    canonical_primary_camera = PRIMARY_CAMERAS[handedness]
    backup_primary_camera = args.backup_primary_map.get(handedness)
    selected = {camera_id: choose_prediction(items, previous_joints, args) for camera_id, items in by_camera.items() if items}
    backup_gate_info: dict[str, Any] = {
        "backup_primary_camera": backup_primary_camera,
        "backup_primary_gate_passed": None,
        "backup_primary_gate_reason": None,
        "backup_primary_mask_score": None,
        "backup_primary_view_quality_score": None,
        "backup_primary_temporal_error_m": None,
    }

    if not selected:
        return None

    anchor_camera, anchor, anchor_role, anchor_selection_info = select_anchor(
        selected,
        handedness,
        canonical_primary_camera,
        backup_primary_camera,
        args,
    )

    if anchor_role == "backup_primary":
        passed, backup_gate_info = backup_gate(anchor, previous_joints, args)
        backup_gate_info["backup_primary_camera"] = backup_primary_camera
        if not passed and canonical_primary_camera not in selected:
            anchor_role = "quality_anchor"
            anchor_selection_info["anchor_selection_reason"] = (
                f"backup_gate_failed_but_no_canonical:{backup_gate_info['backup_primary_gate_reason']}"
            )
        elif not passed:
            anchor_camera = canonical_primary_camera
            anchor = selected[canonical_primary_camera]
            anchor_role = "primary"
            anchor_selection_info["anchor_selection_reason"] = (
                f"backup_gate_failed_reverted_to_canonical:{backup_gate_info['backup_primary_gate_reason']}"
            )
            anchor_selection_info["anchor_score"] = anchor_selection_info["anchor_selection_scores"].get(anchor_camera)
            anchor_selection_info["anchor_score_margin_to_canonical"] = 0.0

    if anchor_role == "quality_anchor" and anchor_camera not in {canonical_primary_camera, backup_primary_camera}:
        # Free non-primary anchors are useful for debugging/visual continuity, but
        # they should not silently become metric output unless another view agrees.
        pass

    if anchor is None:
        if not selected:
            return None
        camera_id, candidate = max(
            sorted(selected.items()),
            key=lambda item: item[1].get("_temporal_selection_score", -float("inf")),
        )
        joints = local_points(candidate["hamer_joints_cam"])
        verts = local_points(candidate["hamer_vertices_cam"]) if is_xyz_list(candidate.get("hamer_vertices_cam")) else None
        joints, temporal_info = apply_temporal_smoothing(joints, previous_joints, "nonprimary_hamer_candidate", args)
        return {
            "group_id": group_id,
            "handedness": handedness,
            "primary_camera": camera_id,
            "canonical_primary_camera": canonical_primary_camera,
            "anchor_camera": camera_id,
            "anchor_camera_role": "nonprimary_candidate",
            "used_cameras": [camera_id],
            "rejected_cameras": [],
            "mode": "nonprimary_hamer_candidate",
            "metric_valid": False,
            "local_shape_valid": False,
            "local_joints_m": joints.tolist(),
            "local_vertices_m": verts.tolist() if verts is not None else None,
            "consistency_errors_m": {},
            "source_predictions": {camera_id: candidate.get("rendered_overlay_path")},
            "temporal_selection_errors_m": {camera_id: candidate.get("_temporal_pose_error_m")},
            "temporal_selection_scores": {camera_id: candidate.get("_temporal_selection_score")},
            **anchor_selection_info,
            **backup_gate_info,
            **temporal_info,
        }

    anchor_joints = local_points(anchor["hamer_joints_cam"])
    anchor_verts = local_points(anchor["hamer_vertices_cam"]) if is_xyz_list(anchor.get("hamer_vertices_cam")) else None
    aligned_joints = [anchor_joints]
    used = [anchor_camera]
    rejected = []
    errors = {}
    source_predictions = {anchor_camera: anchor.get("rendered_overlay_path")}

    for camera_id, pred in sorted(selected.items()):
        if camera_id == anchor_camera:
            continue
        joints = local_points(pred["hamer_joints_cam"])
        aligned, err = kabsch_align(joints, anchor_joints)
        errors[camera_id] = err
        source_predictions[camera_id] = pred.get("rendered_overlay_path")
        if err <= args.consistency_threshold_m:
            aligned_joints.append(aligned)
            used.append(camera_id)
        else:
            rejected.append(camera_id)

    if len(aligned_joints) > 1:
        fused_joints = np.median(np.stack(aligned_joints, axis=0), axis=0)
        if anchor_role == "primary":
            mode = "primary_hamer_local_fused"
        elif anchor_role == "backup_primary":
            mode = "backup_primary_hamer_local_fused"
        else:
            mode = "quality_anchor_hamer_local_fused"
        metric_valid = True
    else:
        fused_joints = anchor_joints
        if anchor_role == "primary":
            mode = "primary_hamer_local"
        elif anchor_role == "backup_primary":
            mode = "backup_primary_hamer_local"
        else:
            mode = "quality_anchor_hamer_local"
        metric_valid = False
    raw_fused_joints = fused_joints.copy()
    fused_joints, temporal_info = apply_temporal_smoothing(fused_joints, previous_joints, mode, args)

    return {
        "group_id": group_id,
        "handedness": handedness,
        "primary_camera": anchor_camera,
        "canonical_primary_camera": canonical_primary_camera,
        "anchor_camera": anchor_camera,
        "anchor_camera_role": anchor_role,
        "used_cameras": used,
        "rejected_cameras": rejected,
        "mode": mode,
        "metric_valid": metric_valid,
        "local_shape_valid": bool(metric_valid or anchor_role in {"primary", "backup_primary", "quality_anchor"}),
        "local_joints_m": fused_joints.tolist(),
        "raw_local_joints_m": raw_fused_joints.tolist(),
        "local_vertices_m": anchor_verts.tolist() if anchor_verts is not None else None,
        "primary_camera_joints_cam": anchor.get("hamer_joints_cam"),
        "primary_camera_cam_t": anchor.get("hamer_cam_t"),
        "consistency_errors_m": errors,
        "source_predictions": source_predictions,
        "temporal_selection_errors_m": {
            camera_id: pred.get("_temporal_pose_error_m") for camera_id, pred in sorted(selected.items())
        },
        "temporal_selection_scores": {
            camera_id: pred.get("_temporal_selection_score") for camera_id, pred in sorted(selected.items())
        },
        **anchor_selection_info,
        **backup_gate_info,
        **temporal_info,
    }


def main() -> None:
    args = parse_args()
    if args.temporal_error_cap_m <= 0.0:
        raise SystemExit("--temporal-error-cap-m must be positive")
    if args.backup_min_mask_score < 0.0:
        raise SystemExit("--backup-min-mask-score must be non-negative")
    if args.backup_min_view_quality < 0.0:
        raise SystemExit("--backup-min-view-quality must be non-negative")
    if args.backup_max_temporal_error_m <= 0.0:
        raise SystemExit("--backup-max-temporal-error-m must be positive")
    args.backup_primary_map = parse_backup_primary_cameras(args.backup_primary_cameras)
    for name in (
        "temporal_metric_alpha",
        "temporal_primary_alpha",
        "temporal_backup_alpha",
        "temporal_nonprimary_alpha",
        "temporal_quality_anchor_alpha",
    ):
        value = float(getattr(args, name))
        if not 0.0 <= value <= 1.0:
            raise SystemExit(f"--{name.replace('_', '-')} must be in [0, 1]")
    for name in (
        "quality_mask_weight",
        "quality_bbox_weight",
        "quality_edge_weight",
        "quality_source_bonus",
        "quality_known_bonus",
        "primary_prior_bonus",
        "backup_prior_bonus",
        "consensus_selection_weight",
        "anchor_switch_margin",
    ):
        if float(getattr(args, name)) < 0.0:
            raise SystemExit(f"--{name.replace('_', '-')} must be non-negative")
    if args.consensus_error_cap_m <= 0.0:
        raise SystemExit("--consensus-error-cap-m must be positive")
    if args.image_width <= 0 or args.image_height <= 0:
        raise SystemExit("--image-width/--image-height must be positive")
    group_ids = parse_group_ids(args.group_range, args.group_ids)
    suffix = range_suffix(group_ids)
    paths = args.predictions or [Path(item) for item in sorted(glob.glob(args.predictions_glob))]
    output_path = args.output_dir / f"hamer_local_hands_{suffix}.jsonl"
    if args.dry_run:
        print(json.dumps({"prediction_files": [str(p) for p in paths], "output_path": str(output_path)}, indent=2))
        return
    if output_path.exists() and not args.overwrite:
        raise SystemExit(f"{output_path} exists; pass --overwrite to replace it")
    if not paths:
        raise SystemExit("no prediction files found")

    predictions = load_predictions(paths, group_ids)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stats = defaultdict(int)
    previous_joints_by_hand: dict[str, np.ndarray] = {}
    with output_path.open("w", encoding="utf-8") as out:
        group_order = sorted(predictions)
        for group_id in tqdm(group_order, desc="local fusion", unit="frame", position=args.progress_position):
            hands = []
            for handedness in ("Left", "Right"):
                hand = fuse_hand(
                    group_id,
                    handedness,
                    predictions[group_id].get(handedness, {}),
                    previous_joints_by_hand.get(handedness),
                    args,
                )
                if hand is not None:
                    hands.append(hand)
                    previous_joints_by_hand[handedness] = np.asarray(hand["local_joints_m"], dtype=np.float64)
                    stats["hands"] += 1
                    stats[f"mode:{hand['mode']}"] += 1
                    stats[f"anchor_camera:{hand.get('anchor_camera')}"] += 1
                    if hand.get("temporal_smoothing_applied"):
                        stats["temporal_smoothed_hands"] += 1
                        stats[f"temporal_smoothed_mode:{hand['mode']}"] += 1
                    stats[f"anchor_role:{hand.get('anchor_camera_role')}"] += 1
                    if hand.get("backup_primary_gate_reason"):
                        stats[f"backup_gate:{hand['backup_primary_gate_reason']}"] += 1
                    if hand["metric_valid"]:
                        stats["metric_hands"] += 1
            out.write(
                json.dumps(
                    {
                        "type": "hamer_primary_local_frame",
                        "group_id": group_id,
                        "hands": hands,
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )
            stats["frames"] += 1

    with (args.output_dir / f"hamer_local_config_{suffix}.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "predictions": [str(path) for path in paths],
                "output_path": str(output_path),
                "consistency_threshold_m": args.consistency_threshold_m,
                "temporal": bool(args.temporal),
                "temporal_selection_weight": args.temporal_selection_weight,
                "temporal_error_cap_m": args.temporal_error_cap_m,
                "temporal_metric_alpha": args.temporal_metric_alpha,
                "temporal_primary_alpha": args.temporal_primary_alpha,
                "temporal_backup_alpha": args.temporal_backup_alpha,
                "temporal_nonprimary_alpha": args.temporal_nonprimary_alpha,
                "temporal_quality_anchor_alpha": args.temporal_quality_anchor_alpha,
                "image_width": args.image_width,
                "image_height": args.image_height,
                "quality_mask_weight": args.quality_mask_weight,
                "quality_bbox_weight": args.quality_bbox_weight,
                "quality_edge_weight": args.quality_edge_weight,
                "quality_source_bonus": args.quality_source_bonus,
                "quality_known_bonus": args.quality_known_bonus,
                "primary_prior_bonus": args.primary_prior_bonus,
                "backup_prior_bonus": args.backup_prior_bonus,
                "consensus_selection_weight": args.consensus_selection_weight,
                "consensus_error_cap_m": args.consensus_error_cap_m,
                "anchor_switch_margin": args.anchor_switch_margin,
                "min_anchor_score": args.min_anchor_score,
                "backup_primary_cameras": args.backup_primary_map,
                "backup_min_mask_score": args.backup_min_mask_score,
                "backup_min_view_quality": args.backup_min_view_quality,
                "backup_max_temporal_error_m": args.backup_max_temporal_error_m,
                "backup_require_known": args.backup_require_known,
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
