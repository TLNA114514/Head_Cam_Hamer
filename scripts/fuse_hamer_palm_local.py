#!/usr/bin/env python3
"""Fuse per-view HaMeR predictions directly in canonical palm-local space."""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from hamer_multiview_utils import DEFAULT_BASE_DIR, iter_jsonl, parse_group_ids, range_suffix
from progress_utils import tqdm


HAND_BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", action="append", type=Path, help="Prediction JSONL. Can repeat.")
    parser.add_argument("--predictions-glob", default=str(DEFAULT_BASE_DIR / "hamer_per_view" / "hamer_predictions_*.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_BASE_DIR / "hamer_palm_local_fused")
    parser.add_argument("--group-range")
    parser.add_argument("--group-ids")
    parser.add_argument("--image-width", type=int, default=1600)
    parser.add_argument("--image-height", type=int, default=1200)
    parser.add_argument("--quality-mask-weight", type=float, default=0.55)
    parser.add_argument("--quality-bbox-weight", type=float, default=0.15)
    parser.add_argument("--quality-edge-weight", type=float, default=0.12)
    parser.add_argument("--quality-source-bonus", type=float, default=0.06)
    parser.add_argument("--quality-known-bonus", type=float, default=0.05)
    parser.add_argument(
        "--view-fusion",
        choices=["mean", "quality-weighted", "robust-medoid"],
        default="mean",
        help="Cross-view shape fusion. robust-medoid preserves one coherent predicted hand instead of averaging poses.",
    )
    parser.add_argument(
        "--bone-calibration-blend",
        type=float,
        default=0.0,
        help="Zero-shot static bone-length normalization strength. It preserves per-frame bone directions.",
    )
    parser.add_argument("--bone-calibration-min-observations", type=int, default=25)
    parser.add_argument("--temporal-radius", type=int, default=0, help="Offline Gaussian smoothing radius in frames; zero disables it.")
    parser.add_argument("--temporal-sigma", type=float, default=4.0)
    parser.add_argument(
        "--temporal-interpolation-max-gap",
        type=int,
        default=2,
        help="Fill bounded offline gaps up to this many frames; zero disables interpolation.",
    )
    parser.add_argument("--temporal-interpolation-max-joint-displacement-m", type=float, default=0.12)
    parser.add_argument("--temporal-interpolation-max-bone-relative-change", type=float, default=0.20)
    parser.add_argument("--causal-ema-alpha", type=float, default=0.0, help="Causal EMA alpha; zero disables it.")
    parser.add_argument("--one-euro-min-cutoff", type=float, default=0.0, help="One Euro minimum cutoff in Hz; zero disables it.")
    parser.add_argument("--one-euro-beta", type=float, default=0.0, help="One Euro speed coefficient.")
    parser.add_argument("--one-euro-derivative-cutoff", type=float, default=1.0)
    parser.add_argument(
        "--one-euro-min-alpha",
        type=float,
        default=0.0,
        help="Minimum new-frame weight. Positive values bound filter lag during slow deliberate motion.",
    )
    parser.add_argument(
        "--one-euro-bone-space",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Filter parent-to-child bone vectors and reconstruct stable bone lengths instead of filtering joints directly.",
    )
    parser.add_argument("--frame-rate", type=float, default=25.0)
    parser.add_argument(
        "--primary-output",
        choices=["raw", "static-calibrated", "smoothed", "causal-smoothed", "adaptive-causal"],
        default="raw",
        help="Field copied to palm_local_joints_m. Raw output is always retained separately.",
    )
    parser.add_argument("--include-vertices", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--progress-position", type=int, default=int(os.environ.get("TQDM_POSITION", "0")))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def is_xyz_list(value: Any, minimum_length: int = 1) -> bool:
    if not isinstance(value, list) or len(value) < minimum_length:
        return False
    arr = np.asarray(value, dtype=np.float64)
    return arr.ndim == 2 and arr.shape[1] == 3 and np.all(np.isfinite(arr))


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def bbox_area_score(bbox: list[float] | None, image_width: int, image_height: int) -> float:
    if not bbox or len(bbox) != 4 or image_width <= 0 or image_height <= 0:
        return 0.0
    x1, y1, x2, y2 = [float(value) for value in bbox]
    area_ratio = max(0.0, x2 - x1) * max(0.0, y2 - y1) / float(image_width * image_height)
    if area_ratio <= 0.0:
        return 0.0
    if area_ratio < 0.003:
        return clamp01(area_ratio / 0.003)
    if area_ratio > 0.12:
        return clamp01(0.12 / area_ratio)
    return 1.0


def bbox_edge_score(bbox: list[float] | None, image_width: int, image_height: int) -> float:
    if not bbox or len(bbox) != 4 or image_width <= 0 or image_height <= 0:
        return 0.0
    x1, y1, x2, y2 = [float(value) for value in bbox]
    margin = min(x1, y1, float(image_width) - x2, float(image_height) - y2)
    return clamp01((margin + 20.0) / 80.0)


def prediction_quality(record: dict[str, Any], args: argparse.Namespace) -> tuple[float, dict[str, float]]:
    mask_score = record.get("mask_score")
    if isinstance(mask_score, (int, float)):
        mask_input = clamp01(float(mask_score))
    elif isinstance(record.get("sam3_score"), (int, float)):
        mask_input = clamp01(float(record["sam3_score"]))
    elif record.get("used_mask_blur"):
        mask_input = 0.0
    else:
        mask_input = 0.28
    mask_part = float(args.quality_mask_weight) * mask_input
    bbox_part = float(args.quality_bbox_weight) * bbox_area_score(
        record.get("bbox_rectified_px"), int(args.image_width), int(args.image_height)
    )
    edge_part = float(args.quality_edge_weight) * bbox_edge_score(
        record.get("bbox_rectified_px"), int(args.image_width), int(args.image_height)
    )
    source = record.get("source_detector")
    source_multiplier = 1.0 if source == "mediapipe+sam3" else (0.5 if source in {"mediapipe", "sam3"} else 0.0)
    source_part = float(args.quality_source_bonus) * source_multiplier
    known_part = float(args.quality_known_bonus) if record.get("hypothesis_status") == "known" else 0.0
    scale_penalty = 0.04 * abs(float(record.get("bbox_scale", 1.0)) - 1.0)
    score = mask_part + bbox_part + edge_part + source_part + known_part - scale_penalty
    return float(score), {
        "mask": float(mask_part),
        "bbox_area": float(bbox_part),
        "edge": float(edge_part),
        "source": float(source_part),
        "known": float(known_part),
        "bbox_scale_penalty": float(-scale_penalty),
    }


def palm_frame(joints: np.ndarray) -> np.ndarray | None:
    wrist = joints[0]
    x_axis = joints[5] - wrist
    y_hint = joints[17] - wrist
    x_norm = float(np.linalg.norm(x_axis))
    if x_norm <= 1e-6:
        return None
    x_axis /= x_norm
    z_axis = np.cross(x_axis, y_hint)
    z_norm = float(np.linalg.norm(z_axis))
    if z_norm <= 1e-6:
        return None
    z_axis /= z_norm
    y_axis = np.cross(z_axis, x_axis)
    y_norm = float(np.linalg.norm(y_axis))
    if y_norm <= 1e-6:
        return None
    y_axis /= y_norm
    return np.stack([x_axis, y_axis, z_axis], axis=1)


def palm_local(points: np.ndarray, joints: np.ndarray) -> np.ndarray:
    frame = palm_frame(joints)
    if frame is None:
        raise ValueError("degenerate palm frame")
    return (points - joints[0:1]) @ frame


def candidate_from_record(record: dict[str, Any], args: argparse.Namespace) -> dict[str, Any] | None:
    joints_value = record.get("hand_mesh_joints_cam") or record.get("hamer_joints_cam")
    if not is_xyz_list(joints_value, minimum_length=21):
        return None
    joints_cam = np.asarray(joints_value, dtype=np.float64)
    try:
        local_joints = palm_local(joints_cam[:21], joints_cam[:21])
    except ValueError:
        return None
    local_vertices = None
    vertices_value = record.get("hand_mesh_vertices_cam") or record.get("hamer_vertices_cam")
    if args.include_vertices and is_xyz_list(vertices_value):
        vertices_cam = np.asarray(vertices_value, dtype=np.float64)
        try:
            local_vertices = palm_local(vertices_cam, joints_cam[:21])
        except ValueError:
            return None
    score, score_parts = prediction_quality(record, args)
    return {
        "record": record,
        "model_name": str(record.get("model_name") or "hamer"),
        "camera_id": str(record["camera_id"]),
        "joints": local_joints,
        "vertices": local_vertices,
        "quality_score": score,
        "quality_parts": score_parts,
    }


def load_candidates(
    paths: list[Path], group_ids: set[int] | None, args: argparse.Namespace
) -> dict[int, dict[str, dict[str, list[dict[str, Any]]]]]:
    candidates: dict[int, dict[str, dict[str, list[dict[str, Any]]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for path in paths:
        for record in iter_jsonl(path):
            if record.get("type") not in {"hamer_multiview_prediction", "hand_mesh_multiview_prediction"}:
                continue
            group_id = int(record["group_id"])
            if group_ids is not None and group_id not in group_ids:
                continue
            handedness = record.get("handedness")
            camera_id = record.get("camera_id")
            if handedness not in {"Left", "Right"} or not camera_id:
                continue
            candidate = candidate_from_record(record, args)
            if candidate is not None:
                candidates[group_id][str(handedness)][str(camera_id)].append(candidate)
    return candidates


def select_per_camera(
    candidates: dict[int, dict[str, dict[str, list[dict[str, Any]]]]]
) -> dict[tuple[int, str], list[dict[str, Any]]]:
    selected: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for group_id, by_hand in candidates.items():
        for handedness, by_camera in by_hand.items():
            items = [max(camera_items, key=lambda item: item["quality_score"]) for camera_items in by_camera.values() if camera_items]
            if items:
                selected[(group_id, handedness)] = sorted(items, key=lambda item: item["camera_id"])
    return selected


def bone_lengths(joints: np.ndarray) -> np.ndarray:
    return np.asarray([np.linalg.norm(joints[child] - joints[parent]) for parent, child in HAND_BONES], dtype=np.float64)


def joints_to_bone_vectors(joints: np.ndarray) -> np.ndarray:
    vectors = np.zeros_like(joints, dtype=np.float64)
    vectors[0] = joints[0]
    for parent, child in HAND_BONES:
        vectors[child] = joints[child] - joints[parent]
    return vectors


def bone_vectors_to_joints(
    vectors: np.ndarray,
    lengths: np.ndarray,
    fallback_vectors: np.ndarray,
) -> np.ndarray:
    joints = np.zeros_like(vectors, dtype=np.float64)
    joints[0] = vectors[0]
    for bone_index, (parent, child) in enumerate(HAND_BONES):
        direction = vectors[child]
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-8:
            direction = fallback_vectors[child]
            norm = float(np.linalg.norm(direction))
        if norm <= 1e-8:
            joints[child] = joints[parent]
            continue
        joints[child] = joints[parent] + direction / norm * float(lengths[bone_index])
    return joints


def estimate_static_bone_lengths(
    selected: dict[tuple[int, str], list[dict[str, Any]]], min_observations: int
) -> dict[str, np.ndarray | None]:
    by_hand_camera: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    for (_group_id, handedness), items in selected.items():
        for item in items:
            by_hand_camera[(handedness, item["camera_id"])].append(bone_lengths(item["joints"]))
    targets: dict[str, np.ndarray | None] = {}
    for handedness in ("Left", "Right"):
        camera_medians = []
        for (item_hand, _camera_id), observations in sorted(by_hand_camera.items()):
            if item_hand != handedness or len(observations) < min_observations:
                continue
            camera_medians.append(np.median(np.stack(observations, axis=0), axis=0))
        targets[handedness] = np.median(np.stack(camera_medians), axis=0) if camera_medians else None
    return targets


def apply_static_bone_lengths(joints: np.ndarray, target_lengths: np.ndarray | None, blend: float) -> np.ndarray:
    if target_lengths is None or blend <= 0.0:
        return joints.copy()
    output = joints.copy()
    for bone_index, (parent, child) in enumerate(HAND_BONES):
        direction = joints[child] - joints[parent]
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-8:
            continue
        calibrated = output[parent] + direction / norm * float(target_lengths[bone_index])
        original_length_position = output[parent] + direction
        output[child] = (1.0 - blend) * original_length_position + blend * calibrated
    output[0] = 0.0
    return output


def fusion_weights(items: list[dict[str, Any]]) -> np.ndarray:
    weights = np.asarray([max(float(item.get("quality_score", 0.0)), 0.05) for item in items], dtype=np.float64)
    return weights / float(np.sum(weights))


def robust_medoid_index(items: list[dict[str, Any]], stack: np.ndarray, weights: np.ndarray) -> int:
    pairwise = np.mean(np.linalg.norm(stack[:, None, :, :] - stack[None, :, :, :], axis=3), axis=2)
    costs = pairwise @ weights
    return min(
        range(len(items)),
        key=lambda index: (float(costs[index]), -float(weights[index]), str(items[index]["camera_id"])),
    )


def fuse_views(
    items: list[dict[str, Any]],
    target_lengths: np.ndarray | None,
    bone_blend: float,
    include_vertices: bool,
    method: str = "mean",
) -> dict[str, Any]:
    if method not in {"mean", "quality-weighted", "robust-medoid"}:
        raise ValueError(f"unknown view fusion method: {method}")
    raw_stack = np.stack([item["joints"] for item in items], axis=0)
    calibrated_stack = np.stack(
        [apply_static_bone_lengths(item["joints"], target_lengths, bone_blend) for item in items], axis=0
    )
    weights = fusion_weights(items)
    source_index = None
    if method == "robust-medoid":
        source_index = robust_medoid_index(items, raw_stack, weights)
        raw_joints = raw_stack[source_index].copy()
        calibrated_joints = calibrated_stack[source_index].copy()
    elif method == "quality-weighted":
        raw_joints = np.average(raw_stack, axis=0, weights=weights)
        calibrated_joints = np.average(calibrated_stack, axis=0, weights=weights)
    else:
        raw_joints = np.mean(raw_stack, axis=0)
        calibrated_joints = np.mean(calibrated_stack, axis=0)
    raw_residuals = np.linalg.norm(raw_stack - raw_joints[None, :, :], axis=2)
    camera_errors = {
        item["camera_id"]: float(np.mean(raw_residuals[index]))
        for index, item in enumerate(items)
    }
    vertices = None
    if include_vertices:
        vertex_items = [item["vertices"] for item in items if item["vertices"] is not None]
        if len(vertex_items) == len(items):
            vertex_stack = np.stack(vertex_items, axis=0)
            if source_index is not None:
                vertices = vertex_stack[source_index].copy()
            elif method == "quality-weighted":
                vertices = np.average(vertex_stack, axis=0, weights=weights)
            else:
                vertices = np.mean(vertex_stack, axis=0)
    return {
        "raw_joints": raw_joints,
        "static_joints": calibrated_joints,
        "vertices": vertices,
        "joint_std_m": np.std(raw_stack, axis=0).mean(axis=1),
        "camera_errors_m": camera_errors,
        "mean_consensus_error_m": float(np.mean(raw_residuals)),
        "p95_consensus_error_m": float(np.percentile(raw_residuals, 95)),
        "fusion_method": method,
        "fusion_source_camera": items[source_index]["camera_id"] if source_index is not None else None,
        "fusion_source_track_id": (
            items[source_index]["record"].get("track_id") if source_index is not None else None
        ),
        "fusion_weights": {
            item["camera_id"]: float(weights[index]) for index, item in enumerate(items)
        },
    }


def interpolate_short_gaps(
    fused_by_key: dict[tuple[int, str], dict[str, Any]],
    selected: dict[tuple[int, str], list[dict[str, Any]]],
    allowed_group_ids: set[int] | None,
    max_gap: int,
    max_joint_displacement_m: float,
    max_bone_relative_change: float,
) -> tuple[dict[tuple[int, str], dict[str, Any]], dict[str, int]]:
    if max_gap <= 0:
        return {}, {}

    interpolated: dict[tuple[int, str], dict[str, Any]] = {}
    stats: dict[str, int] = defaultdict(int)
    for handedness in ("Left", "Right"):
        observed_groups = sorted(group_id for group_id, hand in fused_by_key if hand == handedness)
        for previous_group_id, next_group_id in zip(observed_groups, observed_groups[1:]):
            gap_size = next_group_id - previous_group_id - 1
            if gap_size < 1 or gap_size > max_gap:
                continue
            stats["candidate_gaps"] += 1
            missing_group_ids = list(range(previous_group_id + 1, next_group_id))
            if allowed_group_ids is not None and any(group_id not in allowed_group_ids for group_id in missing_group_ids):
                stats["rejected_not_requested"] += 1
                continue

            previous = fused_by_key[(previous_group_id, handedness)]
            following = fused_by_key[(next_group_id, handedness)]
            joint_displacements = np.linalg.norm(following["raw_joints"] - previous["raw_joints"], axis=1)
            endpoint_max_joint_displacement_m = float(np.max(joint_displacements))
            if endpoint_max_joint_displacement_m > max_joint_displacement_m:
                stats["rejected_joint_displacement"] += 1
                continue

            previous_bones = bone_lengths(previous["raw_joints"])
            following_bones = bone_lengths(following["raw_joints"])
            bone_denominator = np.maximum(0.5 * (previous_bones + following_bones), 1e-8)
            endpoint_max_bone_relative_change = float(
                np.max(np.abs(following_bones - previous_bones) / bone_denominator)
            )
            if endpoint_max_bone_relative_change > max_bone_relative_change:
                stats["rejected_bone_change"] += 1
                continue

            previous_vertices = previous.get("vertices")
            following_vertices = following.get("vertices")
            can_interpolate_vertices = (
                isinstance(previous_vertices, np.ndarray)
                and isinstance(following_vertices, np.ndarray)
                and previous_vertices.shape == following_vertices.shape
            )
            for group_id in missing_group_ids:
                alpha = (group_id - previous_group_id) / float(next_group_id - previous_group_id)
                interpolated[(group_id, handedness)] = {
                    "raw_joints": (1.0 - alpha) * previous["raw_joints"] + alpha * following["raw_joints"],
                    "static_joints": (1.0 - alpha) * previous["static_joints"] + alpha * following["static_joints"],
                    "vertices": (
                        (1.0 - alpha) * previous_vertices + alpha * following_vertices
                        if can_interpolate_vertices
                        else None
                    ),
                    "previous_group_id": previous_group_id,
                    "next_group_id": next_group_id,
                    "gap_size": gap_size,
                    "alpha": float(alpha),
                    "endpoint_max_joint_displacement_m": endpoint_max_joint_displacement_m,
                    "endpoint_max_bone_relative_change": endpoint_max_bone_relative_change,
                    "endpoint_view_counts": [
                        len(selected[(previous_group_id, handedness)]),
                        len(selected[(next_group_id, handedness)]),
                    ],
                    "endpoint_used_cameras": [
                        [item["camera_id"] for item in selected[(previous_group_id, handedness)]],
                        [item["camera_id"] for item in selected[(next_group_id, handedness)]],
                    ],
                    "source_models": sorted(
                        {
                            item["model_name"]
                            for endpoint_group_id in (previous_group_id, next_group_id)
                            for item in selected[(endpoint_group_id, handedness)]
                        }
                    ),
                }
                stats["interpolated_hands"] += 1
            stats["filled_gaps"] += 1
    return interpolated, dict(stats)


def gaussian_smooth(
    sequence: dict[tuple[int, str], np.ndarray], radius: int, sigma: float
) -> dict[tuple[int, str], np.ndarray]:
    if radius <= 0:
        return {}
    output: dict[tuple[int, str], np.ndarray] = {}
    groups_by_hand: dict[str, set[int]] = defaultdict(set)
    for group_id, handedness in sequence:
        groups_by_hand[handedness].add(group_id)
    for (group_id, handedness), current in sequence.items():
        values = []
        weights = []
        for offset in range(-radius, radius + 1):
            key = (group_id + offset, handedness)
            if key not in sequence:
                continue
            segment_start = min(group_id, group_id + offset)
            segment_end = max(group_id, group_id + offset)
            if not all(group in groups_by_hand[handedness] for group in range(segment_start, segment_end + 1)):
                continue
            values.append(sequence[key])
            weights.append(math.exp(-0.5 * (float(offset) / sigma) ** 2))
        output[(group_id, handedness)] = np.average(np.stack(values, axis=0), axis=0, weights=weights)
    return output


def causal_ema(
    sequence: dict[tuple[int, str], np.ndarray], alpha: float
) -> dict[tuple[int, str], np.ndarray]:
    if alpha <= 0.0:
        return {}
    output: dict[tuple[int, str], np.ndarray] = {}
    state: dict[str, np.ndarray] = {}
    previous_group: dict[str, int] = {}
    for group_id in sorted({group for group, _hand in sequence}):
        for handedness in ("Left", "Right"):
            key = (group_id, handedness)
            if key not in sequence:
                continue
            current = sequence[key]
            previous = state.get(handedness)
            contiguous = previous_group.get(handedness) == group_id - 1
            value = current if previous is None or not contiguous else alpha * current + (1.0 - alpha) * previous
            output[key] = value
            state[handedness] = value
            previous_group[handedness] = group_id
    return output


def lowpass_alpha(cutoff_hz: np.ndarray | float, frame_rate: float) -> np.ndarray:
    cutoff = np.asarray(cutoff_hz, dtype=np.float64)
    return 1.0 / (1.0 + frame_rate / (2.0 * math.pi * cutoff))


def one_euro_smooth(
    sequence: dict[tuple[int, str], np.ndarray],
    min_cutoff: float,
    beta: float,
    derivative_cutoff: float,
    frame_rate: float,
    min_alpha: float = 0.0,
    bone_space: bool = False,
) -> dict[tuple[int, str], np.ndarray]:
    if min_cutoff <= 0.0:
        return {}
    output: dict[tuple[int, str], np.ndarray] = {}
    filtered_state: dict[str, np.ndarray] = {}
    raw_state: dict[str, np.ndarray] = {}
    derivative_state: dict[str, np.ndarray] = {}
    length_state: dict[str, np.ndarray] = {}
    previous_group: dict[str, int] = {}
    derivative_alpha = float(lowpass_alpha(derivative_cutoff, frame_rate))
    for group_id in sorted({group for group, _hand in sequence}):
        for handedness in ("Left", "Right"):
            key = (group_id, handedness)
            if key not in sequence:
                continue
            current_joints = sequence[key]
            current = joints_to_bone_vectors(current_joints) if bone_space else current_joints
            current_lengths = bone_lengths(current_joints) if bone_space else None
            contiguous = previous_group.get(handedness) == group_id - 1
            if handedness not in filtered_state or not contiguous:
                filtered = current.copy()
                derivative = np.zeros_like(current)
                if current_lengths is not None:
                    length_state[handedness] = current_lengths.copy()
            else:
                raw_derivative = (current - raw_state[handedness]) * frame_rate
                derivative = (
                    derivative_alpha * raw_derivative
                    + (1.0 - derivative_alpha) * derivative_state[handedness]
                )
                joint_speed = np.linalg.norm(derivative, axis=1, keepdims=True)
                cutoff = min_cutoff + beta * joint_speed
                alpha = np.maximum(lowpass_alpha(cutoff, frame_rate), min_alpha)
                filtered = alpha * current + (1.0 - alpha) * filtered_state[handedness]
                if current_lengths is not None:
                    length_alpha = max(float(lowpass_alpha(min_cutoff, frame_rate)), 0.5 * min_alpha)
                    length_state[handedness] = (
                        length_alpha * current_lengths
                        + (1.0 - length_alpha) * length_state[handedness]
                    )
            value = (
                bone_vectors_to_joints(filtered, length_state[handedness], current)
                if bone_space
                else filtered
            )
            output[key] = value
            filtered_state[handedness] = filtered
            raw_state[handedness] = current
            derivative_state[handedness] = derivative
            previous_group[handedness] = group_id
    return output


def choose_primary_joints(
    key: tuple[int, str],
    fused: dict[str, Any],
    smoothed: dict[tuple[int, str], np.ndarray],
    causal: dict[tuple[int, str], np.ndarray],
    adaptive_causal: dict[tuple[int, str], np.ndarray],
    args: argparse.Namespace,
) -> tuple[np.ndarray, str]:
    if args.primary_output == "raw":
        return fused["raw_joints"], "raw"
    if args.primary_output == "static-calibrated":
        return fused["static_joints"], "static-calibrated"
    if args.primary_output == "smoothed":
        if key not in smoothed:
            raise ValueError("--primary-output smoothed requires --temporal-radius > 0")
        return smoothed[key], "smoothed"
    if args.primary_output == "causal-smoothed":
        if key not in causal:
            raise ValueError("--primary-output causal-smoothed requires --causal-ema-alpha > 0")
        return causal[key], "causal-smoothed"
    if key not in adaptive_causal:
        raise ValueError("--primary-output adaptive-causal requires --one-euro-min-cutoff > 0")
    return adaptive_causal[key], "adaptive-causal"


def validate_args(args: argparse.Namespace) -> None:
    if args.image_width <= 0 or args.image_height <= 0:
        raise SystemExit("--image-width/--image-height must be positive")
    if not 0.0 <= args.bone_calibration_blend <= 1.0:
        raise SystemExit("--bone-calibration-blend must be in [0, 1]")
    if args.bone_calibration_min_observations < 1:
        raise SystemExit("--bone-calibration-min-observations must be positive")
    if args.temporal_radius < 0:
        raise SystemExit("--temporal-radius must be non-negative")
    if args.temporal_radius > 0 and args.temporal_sigma <= 0.0:
        raise SystemExit("--temporal-sigma must be positive when smoothing is enabled")
    if args.temporal_interpolation_max_gap < 0:
        raise SystemExit("--temporal-interpolation-max-gap must be non-negative")
    if args.temporal_interpolation_max_joint_displacement_m <= 0.0:
        raise SystemExit("--temporal-interpolation-max-joint-displacement-m must be positive")
    if args.temporal_interpolation_max_bone_relative_change <= 0.0:
        raise SystemExit("--temporal-interpolation-max-bone-relative-change must be positive")
    if not 0.0 <= args.causal_ema_alpha <= 1.0:
        raise SystemExit("--causal-ema-alpha must be in [0, 1]")
    if args.one_euro_min_cutoff < 0.0 or args.one_euro_beta < 0.0:
        raise SystemExit("One Euro cutoff and beta must be non-negative")
    if not 0.0 <= args.one_euro_min_alpha <= 1.0:
        raise SystemExit("--one-euro-min-alpha must be in [0, 1]")
    if args.one_euro_derivative_cutoff <= 0.0 or args.frame_rate <= 0.0:
        raise SystemExit("One Euro derivative cutoff and frame rate must be positive")
    if args.primary_output == "static-calibrated" and args.bone_calibration_blend <= 0.0:
        raise SystemExit("--primary-output static-calibrated requires --bone-calibration-blend > 0")
    if args.primary_output == "smoothed" and args.temporal_radius <= 0:
        raise SystemExit("--primary-output smoothed requires --temporal-radius > 0")
    if args.primary_output == "causal-smoothed" and args.causal_ema_alpha <= 0.0:
        raise SystemExit("--primary-output causal-smoothed requires --causal-ema-alpha > 0")
    if args.primary_output == "adaptive-causal" and args.one_euro_min_cutoff <= 0.0:
        raise SystemExit("--primary-output adaptive-causal requires --one-euro-min-cutoff > 0")


def main() -> None:
    args = parse_args()
    validate_args(args)
    group_ids = parse_group_ids(args.group_range, args.group_ids)
    suffix = range_suffix(group_ids)
    paths = args.predictions or [Path(path) for path in sorted(glob.glob(args.predictions_glob))]
    output_path = args.output_dir / f"palm_local_hands_{suffix}.jsonl"
    config_path = args.output_dir / f"palm_local_config_{suffix}.json"
    if args.dry_run:
        print(json.dumps({"predictions": [str(path) for path in paths], "output_path": str(output_path)}, indent=2))
        return
    if not paths:
        raise SystemExit("no prediction files found")
    if output_path.exists() and not args.overwrite:
        raise SystemExit(f"{output_path} exists; pass --overwrite to replace it")

    candidates = load_candidates(paths, group_ids, args)
    selected = select_per_camera(candidates)
    static_bone_lengths = estimate_static_bone_lengths(selected, args.bone_calibration_min_observations)
    fused_by_key = {
        key: fuse_views(
            items,
            static_bone_lengths.get(key[1]),
            args.bone_calibration_blend,
            args.include_vertices,
            args.view_fusion,
        )
        for key, items in selected.items()
    }
    interpolated_by_key, interpolation_stats = interpolate_short_gaps(
        fused_by_key,
        selected,
        group_ids,
        args.temporal_interpolation_max_gap,
        args.temporal_interpolation_max_joint_displacement_m,
        args.temporal_interpolation_max_bone_relative_change,
    )
    static_sequence = {key: fused["static_joints"] for key, fused in fused_by_key.items()}
    static_sequence.update({key: item["static_joints"] for key, item in interpolated_by_key.items()})
    smoothed = gaussian_smooth(static_sequence, args.temporal_radius, args.temporal_sigma)
    causal = causal_ema(static_sequence, args.causal_ema_alpha)
    adaptive_causal = one_euro_smooth(
        static_sequence,
        args.one_euro_min_cutoff,
        args.one_euro_beta,
        args.one_euro_derivative_cutoff,
        args.frame_rate,
        args.one_euro_min_alpha,
        args.one_euro_bone_space,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stats: dict[str, int] = defaultdict(int)
    with output_path.open("w", encoding="utf-8") as output_file:
        group_order = sorted(
            {group_id for group_id, _handedness in selected}
            | {group_id for group_id, _handedness in interpolated_by_key}
        )
        for group_id in tqdm(group_order, desc="palm-local fusion", unit="frame", position=args.progress_position):
            hands = []
            for handedness in ("Left", "Right"):
                key = (group_id, handedness)
                if key not in selected and key not in interpolated_by_key:
                    continue
                if key in interpolated_by_key:
                    interpolation = interpolated_by_key[key]
                    primary_joints, base_primary_source = choose_primary_joints(
                        key, interpolation, smoothed, causal, adaptive_causal, args
                    )
                    primary_source = f"temporal-interpolated:{base_primary_source}"
                    hand = {
                        "group_id": group_id,
                        "handedness": handedness,
                        "mode": f"zero_shot_{primary_source}",
                        "metric_valid": False,
                        "local_shape_valid": True,
                        "temporal_interpolated": True,
                        "fusion_view_count": 0,
                        "used_cameras": [],
                        "source_models": interpolation["source_models"],
                        "palm_local_joints_m": primary_joints.tolist(),
                        "raw_palm_local_joints_m": None,
                        "static_calibrated_palm_local_joints_m": None,
                        "temporal_interpolated_raw_palm_local_joints_m": interpolation["raw_joints"].tolist(),
                        "temporal_interpolated_static_palm_local_joints_m": interpolation["static_joints"].tolist(),
                        "smoothed_palm_local_joints_m": smoothed[key].tolist() if key in smoothed else None,
                        "causal_smoothed_palm_local_joints_m": causal[key].tolist() if key in causal else None,
                        "adaptive_causal_palm_local_joints_m": (
                            adaptive_causal[key].tolist() if key in adaptive_causal else None
                        ),
                        "palm_local_vertices_m": (
                            interpolation["vertices"].tolist() if interpolation["vertices"] is not None else None
                        ),
                        "joint_consensus_std_m": None,
                        "mean_consensus_error_m": None,
                        "p95_consensus_error_m": None,
                        "per_camera_consensus_error_m": {},
                        "per_camera_quality_scores": {},
                        "per_camera_quality_parts": {},
                        "interpolation_previous_group_id": interpolation["previous_group_id"],
                        "interpolation_next_group_id": interpolation["next_group_id"],
                        "interpolation_gap_size": interpolation["gap_size"],
                        "interpolation_alpha": interpolation["alpha"],
                        "interpolation_endpoint_view_counts": interpolation["endpoint_view_counts"],
                        "interpolation_endpoint_used_cameras": interpolation["endpoint_used_cameras"],
                        "interpolation_endpoint_max_joint_displacement_m": interpolation[
                            "endpoint_max_joint_displacement_m"
                        ],
                        "interpolation_endpoint_max_bone_relative_change": interpolation[
                            "endpoint_max_bone_relative_change"
                        ],
                        "bone_calibration_blend": args.bone_calibration_blend,
                        "temporal_radius": args.temporal_radius,
                        "temporal_sigma": args.temporal_sigma if args.temporal_radius > 0 else None,
                        "causal_ema_alpha": args.causal_ema_alpha if args.causal_ema_alpha > 0 else None,
                        "one_euro_min_cutoff": args.one_euro_min_cutoff if args.one_euro_min_cutoff > 0 else None,
                        "one_euro_beta": args.one_euro_beta if args.one_euro_min_cutoff > 0 else None,
                        "one_euro_min_alpha": args.one_euro_min_alpha if args.one_euro_min_cutoff > 0 else None,
                        "one_euro_bone_space": args.one_euro_bone_space if args.one_euro_min_cutoff > 0 else None,
                        "one_euro_derivative_cutoff": (
                            args.one_euro_derivative_cutoff if args.one_euro_min_cutoff > 0 else None
                        ),
                        "frame_rate": args.frame_rate if args.one_euro_min_cutoff > 0 else None,
                        "primary_output_source": primary_source,
                    }
                    hands.append(hand)
                    stats["hands"] += 1
                    stats["temporal_interpolated_hands"] += 1
                    stats["view_count:0"] += 1
                    stats[f"primary_output:{primary_source}"] += 1
                    continue
                items = selected[key]
                fused = fused_by_key[key]
                primary_joints, primary_source = choose_primary_joints(
                    key, fused, smoothed, causal, adaptive_causal, args
                )
                hand = {
                    "group_id": group_id,
                    "handedness": handedness,
                    "mode": f"zero_shot_multiview_{fused['fusion_method']}:{primary_source}",
                    "metric_valid": len(items) >= 2,
                    "local_shape_valid": True,
                    "temporal_interpolated": False,
                    "fusion_view_count": len(items),
                    "used_cameras": [item["camera_id"] for item in items],
                    "source_models": sorted({item["model_name"] for item in items}),
                    "palm_local_joints_m": primary_joints.tolist(),
                    "raw_palm_local_joints_m": fused["raw_joints"].tolist(),
                    "static_calibrated_palm_local_joints_m": fused["static_joints"].tolist(),
                    "smoothed_palm_local_joints_m": smoothed[key].tolist() if key in smoothed else None,
                    "causal_smoothed_palm_local_joints_m": causal[key].tolist() if key in causal else None,
                    "adaptive_causal_palm_local_joints_m": (
                        adaptive_causal[key].tolist() if key in adaptive_causal else None
                    ),
                    "palm_local_vertices_m": fused["vertices"].tolist() if fused["vertices"] is not None else None,
                    "joint_consensus_std_m": fused["joint_std_m"].tolist(),
                    "mean_consensus_error_m": fused["mean_consensus_error_m"],
                    "p95_consensus_error_m": fused["p95_consensus_error_m"],
                    "per_camera_consensus_error_m": fused["camera_errors_m"],
                    "per_camera_quality_scores": {item["camera_id"]: item["quality_score"] for item in items},
                    "per_camera_quality_parts": {item["camera_id"]: item["quality_parts"] for item in items},
                    "view_fusion_method": fused["fusion_method"],
                    "view_fusion_source_camera": fused["fusion_source_camera"],
                    "view_fusion_source_track_id": fused["fusion_source_track_id"],
                    "view_fusion_weights": fused["fusion_weights"],
                    "bone_calibration_blend": args.bone_calibration_blend,
                    "temporal_radius": args.temporal_radius,
                    "temporal_sigma": args.temporal_sigma if args.temporal_radius > 0 else None,
                    "causal_ema_alpha": args.causal_ema_alpha if args.causal_ema_alpha > 0 else None,
                    "one_euro_min_cutoff": args.one_euro_min_cutoff if args.one_euro_min_cutoff > 0 else None,
                    "one_euro_beta": args.one_euro_beta if args.one_euro_min_cutoff > 0 else None,
                    "one_euro_min_alpha": args.one_euro_min_alpha if args.one_euro_min_cutoff > 0 else None,
                    "one_euro_bone_space": args.one_euro_bone_space if args.one_euro_min_cutoff > 0 else None,
                    "one_euro_derivative_cutoff": (
                        args.one_euro_derivative_cutoff if args.one_euro_min_cutoff > 0 else None
                    ),
                    "frame_rate": args.frame_rate if args.one_euro_min_cutoff > 0 else None,
                    "primary_output_source": primary_source,
                }
                hands.append(hand)
                stats["hands"] += 1
                stats[f"view_count:{len(items)}"] += 1
                stats[f"primary_output:{primary_source}"] += 1
                if hand["metric_valid"]:
                    stats["metric_hands"] += 1
            output_file.write(
                json.dumps(
                    {"type": "hamer_palm_local_fused_frame", "group_id": group_id, "hands": hands},
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )
            stats["frames"] += 1

    for key, value in interpolation_stats.items():
        stats[f"interpolation:{key}"] = value

    config = {
        "predictions": [str(path) for path in paths],
        "output_path": str(output_path),
        "group_range": args.group_range,
        "group_ids": args.group_ids,
        "primary_output": args.primary_output,
        "view_fusion": args.view_fusion,
        "bone_calibration_blend": args.bone_calibration_blend,
        "bone_calibration_min_observations": args.bone_calibration_min_observations,
        "static_bone_lengths_m": {
            handedness: lengths.tolist() if lengths is not None else None
            for handedness, lengths in static_bone_lengths.items()
        },
        "temporal_radius": args.temporal_radius,
        "temporal_sigma": args.temporal_sigma,
        "temporal_interpolation_max_gap": args.temporal_interpolation_max_gap,
        "temporal_interpolation_max_joint_displacement_m": args.temporal_interpolation_max_joint_displacement_m,
        "temporal_interpolation_max_bone_relative_change": args.temporal_interpolation_max_bone_relative_change,
        "temporal_interpolation_stats": interpolation_stats,
        "causal_ema_alpha": args.causal_ema_alpha,
        "one_euro_min_cutoff": args.one_euro_min_cutoff,
        "one_euro_beta": args.one_euro_beta,
        "one_euro_min_alpha": args.one_euro_min_alpha,
        "one_euro_bone_space": args.one_euro_bone_space,
        "one_euro_derivative_cutoff": args.one_euro_derivative_cutoff,
        "frame_rate": args.frame_rate,
        "include_vertices": args.include_vertices,
        "cross_view_weighting": args.view_fusion,
        "quality_score_usage": (
            "within-camera duplicate selection and cross-view fusion"
            if args.view_fusion != "mean"
            else "within-camera duplicate selection only"
        ),
        "uses_ground_truth": False,
        "stats": dict(stats),
    }
    with config_path.open("w", encoding="utf-8") as config_file:
        json.dump(config, config_file, ensure_ascii=False, indent=2)
        config_file.write("\n")
    print("Summary")
    for key in sorted(stats):
        print(f"  {key}: {stats[key]}")
    print(f"wrote: {output_path}")


if __name__ == "__main__":
    main()
