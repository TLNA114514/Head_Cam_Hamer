#!/usr/bin/env python3
"""Robust hand-local refinement for primary-view MediaPipe hand reconstructions."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix


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

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]

PALM_CONNECTIONS = [
    (0, 5), (0, 9), (0, 13), (0, 17),
    (5, 9), (9, 13), (13, 17), (5, 17),
]

PRIMARY_CAMERAS = {
    "Left": "C1",
    "Right": "C2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--detections",
        type=Path,
        default=Path("video/mediapipe_hands_scale_0p30_handedness_fixed/landmarks.jsonl"),
        help="MediaPipe per-camera landmarks JSONL.",
    )
    parser.add_argument(
        "--triangulated",
        type=Path,
        default=Path("video/mediapipe_hands_scale_0p30_handedness_fixed/triangulated_primary_strict/triangulated_hands.jsonl"),
        help="Primary-strict triangulated JSONL used as refinement seed.",
    )
    parser.add_argument(
        "--calib",
        type=Path,
        default=Path("video/cameras/cameras.yaml"),
        help="Camera calibration YAML containing T_H_C.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("video/mediapipe_hands_scale_0p30_handedness_fixed/hand_local_refined"),
        help="Output directory for local_hands.jsonl and stats.",
    )
    parser.add_argument("--min-handedness-score", type=float, default=0.7)
    parser.add_argument("--max-groups", type=int, help="Only process the first N triangulated groups.")
    parser.add_argument("--stride", type=int, default=1, help="Use every Nth triangulated group.")
    parser.add_argument("--pre-gate-ray-error-m", type=float, default=0.06)
    parser.add_argument("--max-mean-ray-error-m", type=float, default=0.03)
    parser.add_argument("--max-ray-error-m", type=float, default=0.05)
    parser.add_argument("--primary-ray-weight", type=float, default=2.0)
    parser.add_argument("--aux-ray-weight", type=float, default=1.0)
    parser.add_argument("--bone-length-weight", type=float, default=1.5)
    parser.add_argument("--palm-rigidity-weight", type=float, default=1.0)
    parser.add_argument("--temporal-weight", type=float, default=1.2)
    parser.add_argument("--cauchy-f-scale", type=float, default=0.025)
    parser.add_argument("--max-nfev", type=int, default=20)
    parser.add_argument(
        "--allow-invalid-original",
        action="store_true",
        help="Use rays whose rectified point did not map to a valid original pixel.",
    )
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc


def is_xyz(value: object) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 3
        and all(isinstance(item, (int, float)) and math.isfinite(float(item)) for item in value)
    )


def unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return vector * 0.0
    return vector / norm


def load_camera_poses(calib_path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    with calib_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    poses = {}
    for camera_id, cam in data["cameras"].items():
        t_h_c = np.asarray(cam["T_H_C"], dtype=np.float64)
        poses[camera_id] = (t_h_c[:3, 3], t_h_c[:3, :3])
    return poses


def ray_to_headset(
    camera_id: str,
    ray_cam: dict[str, float],
    camera_poses: dict[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    origin_h, rotation_h_c = camera_poses[camera_id]
    direction_c = np.asarray([ray_cam["x"], ray_cam["y"], ray_cam["z"]], dtype=np.float64)
    direction_h = rotation_h_c @ unit(direction_c)
    return origin_h, unit(direction_h)


def ray_residual(point: np.ndarray, origin: np.ndarray, direction: np.ndarray) -> np.ndarray:
    direction = unit(direction)
    return point - origin - direction * float(np.dot(point - origin, direction))


def ray_distance(point: np.ndarray, origin: np.ndarray, direction: np.ndarray) -> float:
    return float(np.linalg.norm(ray_residual(point, origin, direction)))


def triangulate_rays(rays: list[tuple[str, np.ndarray, np.ndarray]]) -> np.ndarray | None:
    if len(rays) < 2:
        return None
    a = np.zeros((3, 3), dtype=np.float64)
    b = np.zeros(3, dtype=np.float64)
    for _, origin, direction in rays:
        direction = unit(direction)
        projector = np.eye(3) - np.outer(direction, direction)
        a += projector
        b += projector @ origin
    try:
        point = np.linalg.lstsq(a, b, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(point)):
        return None
    return point


def choose_hands(record: dict[str, Any], min_score: float) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for hand in record.get("hands") or []:
        handedness = hand.get("handedness") or "unknown"
        score = hand.get("handedness_score")
        score = float(score) if isinstance(score, (int, float)) else -1.0
        if score < min_score:
            continue
        old = selected.get(handedness)
        old_score = old.get("handedness_score") if old else None
        old_score = float(old_score) if isinstance(old_score, (int, float)) else -1.0
        if old is None or score > old_score:
            selected[handedness] = hand
    return selected


def load_detections(
    path: Path,
    min_score: float,
    allow_invalid_original: bool,
    camera_poses: dict[str, tuple[np.ndarray, np.ndarray]],
    group_ids: set[int] | None = None,
) -> dict[int, dict[str, dict[str, dict[int, tuple[np.ndarray, np.ndarray, float]]]]]:
    groups: dict[int, dict[str, dict[str, dict[int, tuple[np.ndarray, np.ndarray, float]]]]] = defaultdict(lambda: defaultdict(dict))
    for record in iter_jsonl(path):
        group_id = int(record["group_id"])
        if group_ids is not None and group_id not in group_ids:
            continue
        camera_id = record["camera_id"]
        if camera_id not in camera_poses:
            continue
        for handedness, hand in choose_hands(record, min_score).items():
            rays = hand.get("rectified_ray_cam") or []
            original_points = hand.get("landmarks_original_px") or []
            score = float(hand.get("handedness_score", 0.0))
            joint_rays: dict[int, tuple[np.ndarray, np.ndarray, float]] = {}
            for joint_index, ray in enumerate(rays[: len(LANDMARK_NAMES)]):
                if not allow_invalid_original:
                    if joint_index >= len(original_points) or not original_points[joint_index].get("valid"):
                        continue
                origin, direction = ray_to_headset(camera_id, ray, camera_poses)
                joint_rays[joint_index] = (origin, direction, score)
            if joint_rays:
                groups[group_id][camera_id][handedness] = joint_rays
    return groups


def load_triangulated_frames(path: Path, max_groups: int | None, stride: int) -> list[dict[str, Any]]:
    frames = []
    seen = 0
    for source_index, record in enumerate(iter_jsonl(path)):
        if record.get("type") != "triangulated_mediapipe_hand_frame":
            continue
        if source_index % stride != 0:
            continue
        frames.append(record)
        seen += 1
        if max_groups is not None and seen >= max_groups:
            break
    return frames


def triangulated_seed(hand: dict[str, Any]) -> dict[int, np.ndarray]:
    seed = {}
    for joint in hand.get("joints", []):
        index = joint.get("index")
        position = joint.get("position")
        if isinstance(index, int) and joint.get("valid") and is_xyz(position):
            seed[index] = np.asarray(position, dtype=np.float64)
    return seed


def frame_triangulated_hands(frame: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result = {}
    for hand in frame.get("hands", []):
        handedness = hand.get("handedness")
        if handedness:
            result[handedness] = hand
    return result


def root_relative(points: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    wrist = points.get(0)
    if wrist is None:
        return {}
    return {index: point - wrist for index, point in points.items()}


def palm_frame(points: dict[int, np.ndarray]) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    wrist = points.get(0)
    index_mcp = points.get(5)
    pinky_mcp = points.get(17)
    if wrist is None or index_mcp is None or pinky_mcp is None:
        return None, None

    x_axis = unit(index_mcp - pinky_mcp)
    y_axis = unit(((index_mcp + pinky_mcp) * 0.5) - wrist)
    z_axis = unit(np.cross(x_axis, y_axis))
    if np.linalg.norm(x_axis) <= 1e-9 or np.linalg.norm(y_axis) <= 1e-9 or np.linalg.norm(z_axis) <= 1e-9:
        return None, None
    y_axis = unit(np.cross(z_axis, x_axis))
    rotation = np.column_stack([x_axis, y_axis, z_axis])
    return wrist, rotation


def target_distance(points: dict[int, np.ndarray], previous_root: dict[int, np.ndarray], a: int, b: int) -> float | None:
    if a in previous_root and b in previous_root:
        return float(np.linalg.norm(previous_root[a] - previous_root[b]))
    if a in points and b in points:
        return float(np.linalg.norm(points[a] - points[b]))
    return None


def make_initial_points(
    observations: dict[int, list[dict[str, Any]]],
    seed_points: dict[int, np.ndarray],
    previous_world: dict[int, np.ndarray],
    previous_root: dict[int, np.ndarray],
    primary_camera: str,
) -> dict[int, np.ndarray]:
    points = dict(seed_points)

    for joint_index, obs_list in observations.items():
        if joint_index in points:
            continue
        multi = [(obs["camera_id"], obs["origin"], obs["direction"]) for obs in obs_list]
        point = triangulate_rays(multi)
        if point is not None:
            points[joint_index] = point
            continue
        if joint_index in previous_world:
            points[joint_index] = previous_world[joint_index]
            continue

    wrist = points.get(0)
    if wrist is None:
        wrist_obs = [obs for obs in observations.get(0, []) if obs["camera_id"] == primary_camera]
        if wrist_obs and 0 in previous_world:
            points[0] = previous_world[0]
            wrist = points[0]

    if wrist is not None:
        for joint_index, rel in previous_root.items():
            if joint_index not in points:
                points[joint_index] = wrist + rel

    return points


def accepted_observations(
    observations: dict[int, list[dict[str, Any]]],
    seed_points: dict[int, np.ndarray],
    previous_world: dict[int, np.ndarray],
    primary_camera: str,
    args: argparse.Namespace,
    stats: Counter,
) -> tuple[dict[int, list[dict[str, Any]]], dict[int, list[str]]]:
    accepted: dict[int, list[dict[str, Any]]] = defaultdict(list)
    rejected: dict[int, list[str]] = defaultdict(list)

    for joint_index, obs_list in observations.items():
        gate_point = seed_points.get(joint_index)
        if gate_point is None:
            gate_point = previous_world.get(joint_index)
        for obs in obs_list:
            camera_id = obs["camera_id"]
            if camera_id == primary_camera:
                accepted[joint_index].append(obs)
                continue
            if gate_point is not None:
                distance = ray_distance(gate_point, obs["origin"], obs["direction"])
                if distance > args.pre_gate_ray_error_m:
                    rejected[joint_index].append(camera_id)
                    stats["pregate_rejected_observations"] += 1
                    stats[f"pregate_rejected:{camera_id}"] += 1
                    continue
            accepted[joint_index].append(obs)
    return accepted, rejected


def optimize_points(
    initial_points: dict[int, np.ndarray],
    accepted: dict[int, list[dict[str, Any]]],
    previous_root: dict[int, np.ndarray],
    args: argparse.Namespace,
) -> tuple[dict[int, np.ndarray], Any]:
    variable_indices = sorted(index for index in initial_points if accepted.get(index) or index == 0)
    if not variable_indices:
        return {}, None
    index_to_offset = {index: offset for offset, index in enumerate(variable_indices)}
    x0 = np.concatenate([initial_points[index] for index in variable_indices])

    seed_root = root_relative(initial_points)
    bone_targets = {}
    for edge in HAND_CONNECTIONS:
        target = target_distance(initial_points, previous_root, *edge)
        if target is not None and target > 1e-5:
            bone_targets[edge] = target
    palm_targets = {}
    for edge in PALM_CONNECTIONS:
        target = target_distance(initial_points, previous_root, *edge)
        if target is not None and target > 1e-5:
            palm_targets[edge] = target

    def unpack(params: np.ndarray) -> dict[int, np.ndarray]:
        return {
            index: params[offset * 3: offset * 3 + 3]
            for index, offset in index_to_offset.items()
        }

    accepted_items = [(joint_index, accepted[joint_index]) for joint_index in sorted(accepted)]
    bone_items = list(bone_targets.items())
    palm_items = list(palm_targets.items())
    if previous_root and 0 in index_to_offset:
        temporal_items = [(joint_index, previous_root[joint_index]) for joint_index in sorted(previous_root) if joint_index in index_to_offset]
        temporal_scale = args.temporal_weight
    elif seed_root and 0 in index_to_offset:
        temporal_items = [(joint_index, seed_root[joint_index]) for joint_index in sorted(seed_root) if joint_index in index_to_offset]
        temporal_scale = args.temporal_weight * 0.35
    else:
        temporal_items = []
        temporal_scale = 0.0

    def variable_columns(joint_index: int) -> list[int]:
        offset = index_to_offset[joint_index] * 3
        return [offset, offset + 1, offset + 2]

    dependencies: list[list[int]] = []
    for joint_index, obs_list in accepted_items:
        if joint_index not in index_to_offset:
            continue
        columns = variable_columns(joint_index)
        for _obs in obs_list:
            dependencies.extend([columns, columns, columns])
    for (a, b), _target in bone_items:
        if a in index_to_offset and b in index_to_offset:
            dependencies.append(variable_columns(a) + variable_columns(b))
    for (a, b), _target in palm_items:
        if a in index_to_offset and b in index_to_offset:
            dependencies.append(variable_columns(a) + variable_columns(b))
    for joint_index, _prev_rel in temporal_items:
        columns = variable_columns(joint_index) + variable_columns(0)
        dependencies.extend([columns, columns, columns])

    sparsity = lil_matrix((len(dependencies), len(x0)), dtype=np.int8)
    for row, columns in enumerate(dependencies):
        for column in columns:
            sparsity[row, column] = 1

    def residuals(params: np.ndarray) -> np.ndarray:
        points = unpack(params)
        values = []
        for joint_index, obs_list in accepted_items:
            point = points.get(joint_index)
            if point is None:
                continue
            for obs in obs_list:
                weight = args.primary_ray_weight if obs["is_primary"] else args.aux_ray_weight
                values.extend((ray_residual(point, obs["origin"], obs["direction"]) * weight).tolist())

        for (a, b), target in bone_items:
            if a in points and b in points:
                values.append((float(np.linalg.norm(points[a] - points[b])) - target) * args.bone_length_weight)

        for (a, b), target in palm_items:
            if a in points and b in points:
                values.append((float(np.linalg.norm(points[a] - points[b])) - target) * args.palm_rigidity_weight)

        if temporal_items and 0 in points:
            wrist = points[0]
            for joint_index, prev_rel in temporal_items:
                if joint_index in points:
                    values.extend(((points[joint_index] - wrist - prev_rel) * temporal_scale).tolist())

        return np.asarray(values, dtype=np.float64)

    result = least_squares(
        residuals,
        x0,
        loss="cauchy",
        f_scale=args.cauchy_f_scale,
        max_nfev=args.max_nfev,
        jac_sparsity=sparsity.tocsr(),
        tr_solver="lsmr",
    )
    return unpack(result.x), result


def build_joint_output(
    joint_index: int,
    optimized_points: dict[int, np.ndarray],
    accepted: dict[int, list[dict[str, Any]]],
    rejected: dict[int, list[str]],
    palm_origin: np.ndarray | None,
    palm_rotation: np.ndarray | None,
    primary_camera: str,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], str | None]:
    point = optimized_points.get(joint_index)
    obs_list = accepted.get(joint_index, [])
    rejected_cameras = sorted(set(rejected.get(joint_index, [])))

    if point is None or not np.all(np.isfinite(point)):
        mode = "local_rejected_outlier" if rejected_cameras else "missing"
        return {
            "joint_index": joint_index,
            "joint_name": LANDMARK_NAMES[joint_index],
            "index": joint_index,
            "name": LANDMARK_NAMES[joint_index],
            "valid": False,
            "metric_valid": False,
            "reconstruction_mode": mode,
            "position": None,
            "position_world_m": None,
            "root_relative_headset_m": None,
            "palm_local_m": None,
            "source_cameras": [],
            "rejected_cameras": rejected_cameras,
            "mean_ray_error_m": None,
            "max_ray_error_m": None,
        }, mode

    source_cameras = sorted({obs["camera_id"] for obs in obs_list})
    distances = [ray_distance(point, obs["origin"], obs["direction"]) for obs in obs_list]
    mean_error = float(np.mean(distances)) if distances else None
    max_error = float(max(distances)) if distances else None
    has_primary = primary_camera in source_cameras
    has_aux = any(camera_id != primary_camera for camera_id in source_cameras)
    metric_valid = (
        has_primary
        and has_aux
        and mean_error is not None
        and max_error is not None
        and mean_error <= args.max_mean_ray_error_m
        and max_error <= args.max_ray_error_m
    )

    if metric_valid:
        mode = "local_refined_metric"
    elif has_primary:
        mode = "primary_temporal_local_fallback"
    elif rejected_cameras:
        mode = "local_rejected_outlier"
    else:
        mode = "missing"

    valid = mode in {"local_refined_metric", "primary_temporal_local_fallback"}
    if not valid:
        point_out = None
        root_out = None
        palm_out = None
    else:
        point_out = point.tolist()
        root = optimized_points.get(0)
        root_out = (point - root).tolist() if root is not None else None
        if palm_origin is not None and palm_rotation is not None:
            palm = palm_rotation.T @ (point - palm_origin)
            if joint_index == 0:
                palm = np.zeros(3, dtype=np.float64)
            palm_out = palm.tolist()
        else:
            palm_out = root_out

    return {
        "joint_index": joint_index,
        "joint_name": LANDMARK_NAMES[joint_index],
        "index": joint_index,
        "name": LANDMARK_NAMES[joint_index],
        "valid": valid,
        "metric_valid": bool(metric_valid),
        "reconstruction_mode": mode,
        "position": point_out,
        "position_world_m": point_out,
        "root_relative_headset_m": root_out,
        "palm_local_m": palm_out,
        "source_cameras": source_cameras if valid else [],
        "rejected_cameras": rejected_cameras,
        "mean_ray_error_m": mean_error if metric_valid else None,
        "max_ray_error_m": max_error if metric_valid else None,
    }, mode


def refine_hand(
    group_id: int,
    handedness: str,
    detection_group: dict[str, dict[str, dict[int, tuple[np.ndarray, np.ndarray, float]]]],
    triangulated_hand: dict[str, Any] | None,
    previous_state: dict[str, dict[str, dict[int, np.ndarray]]],
    args: argparse.Namespace,
    stats: Counter,
) -> dict[str, Any] | None:
    primary_camera = PRIMARY_CAMERAS.get(handedness)
    if primary_camera is None:
        return None
    primary_camera_hands = detection_group.get(primary_camera, {})
    if handedness not in primary_camera_hands:
        stats["skipped_no_primary_hand"] += 1
        stats[f"skipped_no_primary:{handedness}"] += 1
        return None

    observations: dict[int, list[dict[str, Any]]] = defaultdict(list)
    camera_ids = sorted(camera_id for camera_id, hands in detection_group.items() if handedness in hands)
    for camera_id in camera_ids:
        joint_rays = detection_group[camera_id][handedness]
        for joint_index, (origin, direction, score) in joint_rays.items():
            observations[joint_index].append(
                {
                    "camera_id": camera_id,
                    "origin": origin,
                    "direction": direction,
                    "score": score,
                    "is_primary": camera_id == primary_camera,
                }
            )

    seed_points = triangulated_seed(triangulated_hand or {})
    prev_world = previous_state.get("world", {}).get(handedness, {})
    prev_root = previous_state.get("root", {}).get(handedness, {})
    accepted, rejected = accepted_observations(observations, seed_points, prev_world, primary_camera, args, stats)
    initial_points = make_initial_points(accepted, seed_points, prev_world, prev_root, primary_camera)

    if 0 not in initial_points:
        stats["skipped_no_wrist_seed"] += 1
        return None

    optimized_points, result = optimize_points(initial_points, accepted, prev_root, args)
    if not optimized_points:
        stats["skipped_optimization_empty"] += 1
        return None

    palm_origin, palm_rotation = palm_frame(optimized_points)
    joints = []
    metric_count = 0
    fallback_count = 0
    outlier_count = 0
    missing_count = 0
    used_cameras = set()
    rejected_cameras = set()

    for joint_index in range(len(LANDMARK_NAMES)):
        joint, mode = build_joint_output(
            joint_index,
            optimized_points,
            accepted,
            rejected,
            palm_origin,
            palm_rotation,
            primary_camera,
            args,
        )
        joints.append(joint)
        used_cameras.update(joint.get("source_cameras") or [])
        rejected_cameras.update(joint.get("rejected_cameras") or [])
        if mode == "local_refined_metric":
            metric_count += 1
        elif mode == "primary_temporal_local_fallback":
            fallback_count += 1
        elif mode == "local_rejected_outlier":
            outlier_count += 1
        elif mode == "missing":
            missing_count += 1

    if metric_count == 0 and fallback_count == 0:
        stats["skipped_no_valid_refined_joints"] += 1
        return None

    valid_world = {
        joint["joint_index"]: np.asarray(joint["position_world_m"], dtype=np.float64)
        for joint in joints
        if joint.get("valid") and is_xyz(joint.get("position_world_m"))
    }
    valid_root = root_relative(valid_world)
    if valid_world and valid_root:
        previous_state.setdefault("world", {})[handedness] = valid_world
        previous_state.setdefault("root", {})[handedness] = valid_root

    palm_frame_out = None
    if palm_origin is not None and palm_rotation is not None:
        palm_frame_out = {
            "origin": palm_origin.tolist(),
            "rotation_headset_from_palm": palm_rotation.tolist(),
        }

    wrist_world = optimized_points.get(0)
    stats["hands"] += 1
    stats[f"hands:{handedness}"] += 1
    stats["metric_joints"] += metric_count
    stats["temporal_fallback_joints"] += fallback_count
    stats["outlier_joints"] += outlier_count
    stats["missing_joints"] += missing_count
    for camera_id in used_cameras:
        stats[f"used_camera:{camera_id}"] += 1
    for camera_id in rejected_cameras:
        stats[f"rejected_camera:{camera_id}"] += 1

    return {
        "handedness": handedness,
        "primary_camera": primary_camera,
        "used_cameras": sorted(used_cameras),
        "rejected_cameras": sorted(rejected_cameras),
        "optimization_success": bool(result.success) if result is not None else False,
        "optimization_cost": float(result.cost) if result is not None else None,
        "metric_joint_count": metric_count,
        "temporal_fallback_joint_count": fallback_count,
        "outlier_joint_count": outlier_count,
        "missing_joint_count": missing_count,
        "wrist_world_m": wrist_world.tolist() if wrist_world is not None else None,
        "palm_frame_in_headset": palm_frame_out,
        "joints": joints,
        "group_id": group_id,
    }


def refine_frames(args: argparse.Namespace, output_jsonl: Path) -> Counter:
    camera_poses = load_camera_poses(args.calib)
    triangulated_frames = load_triangulated_frames(args.triangulated, args.max_groups, args.stride)
    target_group_ids = {int(frame["group_id"]) for frame in triangulated_frames}
    detections = load_detections(
        args.detections,
        args.min_handedness_score,
        args.allow_invalid_original,
        camera_poses,
        target_group_ids,
    )
    stats: Counter = Counter()
    previous_state: dict[str, dict[str, dict[int, np.ndarray]]] = {"world": {}, "root": {}}
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    with output_jsonl.open("w", encoding="utf-8") as out:
        for frame in triangulated_frames:
            group_id = int(frame["group_id"])
            detection_group = detections.get(group_id, {})
            tri_hands = frame_triangulated_hands(frame)
            output_hands = []
            for handedness in ("Left", "Right"):
                refined = refine_hand(
                    group_id,
                    handedness,
                    detection_group,
                    tri_hands.get(handedness),
                    previous_state,
                    args,
                    stats,
                )
                if refined is not None:
                    output_hands.append(refined)

            output_frame = {
                "type": "hand_local_refined_frame",
                "source": "headcam_mediapipe_hand_local_refinement",
                "group_id": group_id,
                "timestamp_unix_ns": frame.get("timestamp_unix_ns"),
                "hands": output_hands,
            }
            stats["frames"] += 1
            if not output_hands:
                stats["frames_without_hands"] += 1
            out.write(json.dumps(output_frame, ensure_ascii=False, separators=(",", ":")) + "\n")
    return stats


def write_config(args: argparse.Namespace, output_dir: Path, output_jsonl: Path, stats: Counter) -> None:
    config = {
        "detections": str(args.detections),
        "triangulated": str(args.triangulated),
        "calib": str(args.calib),
        "output_jsonl": str(output_jsonl),
        "primary_cameras": PRIMARY_CAMERAS,
        "min_handedness_score": args.min_handedness_score,
        "pre_gate_ray_error_m": args.pre_gate_ray_error_m,
        "max_mean_ray_error_m": args.max_mean_ray_error_m,
        "max_ray_error_m": args.max_ray_error_m,
        "primary_ray_weight": args.primary_ray_weight,
        "aux_ray_weight": args.aux_ray_weight,
        "bone_length_weight": args.bone_length_weight,
        "palm_rigidity_weight": args.palm_rigidity_weight,
        "temporal_weight": args.temporal_weight,
        "cauchy_f_scale": args.cauchy_f_scale,
        "max_nfev": args.max_nfev,
        "allow_invalid_original": bool(args.allow_invalid_original),
        "max_groups": args.max_groups,
        "stride": args.stride,
        "stats": dict(stats),
    }
    with (output_dir / "refinement_config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
        f.write("\n")


def write_stats(output_dir: Path, stats: Counter) -> None:
    with (output_dir / "refinement_stats.json").open("w", encoding="utf-8") as f:
        json.dump(dict(sorted(stats.items())), f, ensure_ascii=False, indent=2)
        f.write("\n")


def main() -> None:
    args = parse_args()
    if args.stride < 1:
        raise SystemExit("--stride must be at least 1")
    for path_name in ("detections", "triangulated", "calib"):
        path = getattr(args, path_name)
        if not path.exists():
            raise SystemExit(f"{path_name} not found: {path}")
    if args.max_mean_ray_error_m <= 0.0 or args.max_ray_error_m <= 0.0:
        raise SystemExit("ray error thresholds must be positive")
    if args.pre_gate_ray_error_m <= 0.0:
        raise SystemExit("--pre-gate-ray-error-m must be positive")

    output_jsonl = args.output_dir / "local_hands.jsonl"
    stats = refine_frames(args, output_jsonl)
    write_config(args, args.output_dir, output_jsonl, stats)
    write_stats(args.output_dir, stats)

    print("Summary")
    for key in sorted(stats):
        print(f"  {key}: {stats[key]}")
    print(f"wrote: {output_jsonl}")


if __name__ == "__main__":
    main()
