#!/usr/bin/env python3
"""Triangulate MediaPipe hand landmarks from multi-camera ray detections."""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.animation import FuncAnimation, PillowWriter


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

DEFAULT_TRACK_JOINTS = [
    "wrist",
    "thumb_tip",
    "index_tip",
    "middle_tip",
    "ring_tip",
    "pinky_tip",
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
        default=Path("video/sam3_hamer_left_index/landmarks.jsonl"),
        help="Input MediaPipe multi-camera landmarks JSONL.",
    )
    parser.add_argument(
        "--calib",
        type=Path,
        default=Path("video/cameras_left_index/cameras.yaml"),
        help="Camera calibration YAML containing T_H_C.",
    )
    parser.add_argument(
        "--tracked-hands",
        type=Path,
        help="Optional stabilized SAM3 tracks used only to associate MediaPipe hands across cameras.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory. Defaults to DETECTIONS parent / triangulated.",
    )
    parser.add_argument("--min-views", type=int, default=2, help="Minimum camera rays per 3D landmark.")
    parser.add_argument(
        "--min-handedness-score",
        type=float,
        default=0.7,
        help="Drop MediaPipe hands with handedness confidence below this threshold.",
    )
    parser.add_argument(
        "--max-mean-ray-error-m",
        type=float,
        default=0.03,
        help="Reject triangulated landmarks whose mean ray distance exceeds this many meters.",
    )
    parser.add_argument(
        "--max-ray-error-m",
        type=float,
        default=0.05,
        help="Reject triangulated landmarks whose worst ray distance exceeds this many meters.",
    )
    parser.add_argument("--min-depth-m", type=float, default=0.05, help="Reject intersections behind or extremely close to a camera.")
    parser.add_argument("--max-depth-m", type=float, default=1.0, help="Reject ill-conditioned intersections beyond this head-camera working depth.")
    parser.add_argument(
        "--match-key",
        choices=["handedness", "single"],
        default="handedness",
        help="How to associate hands across cameras in a synchronized group.",
    )
    parser.add_argument(
        "--allow-invalid-original",
        action="store_true",
        help="Use landmarks whose rectified point did not map to a valid original image pixel.",
    )
    parser.add_argument("--max-groups", type=int, help="Only process the first N group ids.")
    parser.add_argument("--stride", type=int, default=1, help="Use every Nth group for triangulation.")
    parser.add_argument("--save-plot", type=Path, help="Static trajectory plot path.")
    parser.add_argument("--save-gif", type=Path, help="Optional skeleton animation GIF path.")
    parser.add_argument("--animation-max-frames", type=int, default=250)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--show-cameras", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--title", help="Plot title.")
    return parser.parse_args()


def load_camera_poses(calib_path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    with calib_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    poses = {}
    for camera_id, cam in data["cameras"].items():
        t_h_c = np.asarray(cam["T_H_C"], dtype=np.float64)
        rotation = t_h_c[:3, :3]
        origin = t_h_c[:3, 3]
        poses[camera_id] = (origin, rotation)
    return poses


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


def group_detections(path: Path, max_groups: int | None, stride: int) -> Iterable[tuple[int, list[dict[str, Any]]]]:
    records_by_group: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in iter_jsonl(path):
        group_id = int(record["group_id"])
        records_by_group[group_id].append(record)

    yielded = 0
    for group_id in sorted(records_by_group):
        if group_id % stride != 0:
            continue
        yield group_id, records_by_group[group_id]
        yielded += 1
        if max_groups is not None and yielded >= max_groups:
            return


def hand_bbox(hand: dict[str, Any]) -> list[float] | None:
    landmarks = hand.get("landmarks_rectified_px") or []
    points = [
        (float(point["x"]), float(point["y"]))
        for point in landmarks
        if isinstance(point, dict)
        and isinstance(point.get("x"), (int, float))
        and isinstance(point.get("y"), (int, float))
    ]
    if not points:
        return None
    xs, ys = zip(*points)
    return [min(xs), min(ys), max(xs), max(ys)]


def bbox_iou(a: list[float] | None, b: list[float] | None) -> float:
    if not a or not b or len(a) != 4 or len(b) != 4:
        return 0.0
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(a[2]) - float(a[0])) * max(0.0, float(a[3]) - float(a[1]))
    area_b = max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))
    union = area_a + area_b - intersection
    return intersection / union if union > 0.0 else 0.0


def load_tracked_hands(path: Path | None) -> dict[tuple[int, str], list[dict[str, Any]]]:
    tracked: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    if path is None:
        return tracked
    for record in iter_jsonl(path):
        group_id = int(record["group_id"])
        camera_id = str(record["camera_id"])
        tracked[(group_id, camera_id)].extend(record.get("hands") or [])
    return tracked


def tracked_handedness(hand: dict[str, Any], tracks: list[dict[str, Any]], min_iou: float = 0.25) -> str | None:
    bbox = hand_bbox(hand)
    best_handedness = None
    best_iou = 0.0
    for track in tracks:
        handedness = track.get("locked_handedness") or track.get("handedness")
        if handedness not in {"Left", "Right"}:
            continue
        overlap = bbox_iou(bbox, track.get("bbox"))
        if overlap > best_iou:
            best_iou = overlap
            best_handedness = str(handedness)
    return best_handedness if best_iou >= min_iou else None


def choose_hands(
    record: dict[str, Any],
    match_key: str,
    min_handedness_score: float,
    tracks: list[dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    hands = record.get("hands") or []
    selected: dict[str, dict[str, Any]] = {}

    for hand in hands:
        track_handedness = tracked_handedness(hand, tracks or [])
        if match_key == "single":
            key = "hand"
        else:
            key = track_handedness or hand.get("handedness") or "unknown"
        score = hand.get("handedness_score")
        score = float(score) if isinstance(score, (int, float)) else -1.0
        if track_handedness is None and score < min_handedness_score:
            continue
        old = selected.get(key)
        old_score = old.get("handedness_score") if old else None
        old_score = float(old_score) if isinstance(old_score, (int, float)) else -1.0
        if old is None or score > old_score:
            selected[key] = hand
    return selected


def ray_to_headset(
    camera_id: str,
    ray_cam: dict[str, float],
    camera_poses: dict[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    origin_h, rotation_h_c = camera_poses[camera_id]
    direction_c = np.asarray([ray_cam["x"], ray_cam["y"], ray_cam["z"]], dtype=np.float64)
    direction_c /= np.linalg.norm(direction_c)
    direction_h = rotation_h_c @ direction_c
    direction_h /= np.linalg.norm(direction_h)
    return origin_h, direction_h


def triangulate_rays(rays: list[tuple[str, np.ndarray, np.ndarray]]) -> tuple[np.ndarray, float, float]:
    a = np.zeros((3, 3), dtype=np.float64)
    b = np.zeros(3, dtype=np.float64)
    for _, origin, direction in rays:
        direction = direction / np.linalg.norm(direction)
        projector = np.eye(3) - np.outer(direction, direction)
        a += projector
        b += projector @ origin

    point = np.linalg.lstsq(a, b, rcond=None)[0]
    distances = []
    for _, origin, direction in rays:
        distances.append(float(np.linalg.norm(np.cross(direction, point - origin))))
    return point, float(np.mean(distances)), float(max(distances))


def hand_primary_camera(handedness: str) -> str | None:
    return PRIMARY_CAMERAS.get(handedness)


def ray_depth(point: np.ndarray, origin: np.ndarray, direction: np.ndarray) -> float:
    return float(np.dot(point - origin, direction / np.linalg.norm(direction)))


def select_best_triangulation(
    rays: list[tuple[str, np.ndarray, np.ndarray]],
    primary_camera: str | None,
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    if len(rays) < args.min_views:
        return None

    has_primary = primary_camera is not None and any(camera_id == primary_camera for camera_id, _, _ in rays)
    if primary_camera is not None and not has_primary:
        return None
    candidates = []
    rejected = []
    for count in range(args.min_views, len(rays) + 1):
        for combo in combinations(rays, count):
            if has_primary and primary_camera not in {camera_id for camera_id, _, _ in combo}:
                continue
            point, mean_error, max_error = triangulate_rays(list(combo))
            depths = [ray_depth(point, origin, direction) for _camera_id, origin, direction in combo]
            depth_valid = all(args.min_depth_m <= depth <= args.max_depth_m for depth in depths)
            item = {
                "rays": list(combo),
                "point": point,
                "mean_ray_error_m": mean_error,
                "max_ray_error_m": max_error,
                "source_cameras": [camera_id for camera_id, _, _ in combo],
                "min_depth_m": float(min(depths)),
                "max_depth_m": float(max(depths)),
            }
            if depth_valid and mean_error <= args.max_mean_ray_error_m and max_error <= args.max_ray_error_m:
                candidates.append(item)
            else:
                item["rejected_reason"] = "depth" if not depth_valid else "ray_error"
                rejected.append(item)

    if not candidates:
        if not rejected:
            return None
        best_rejected = min(
            rejected,
            key=lambda item: (
                item["rejected_reason"] == "depth",
                item["mean_ray_error_m"] > args.max_mean_ray_error_m,
                item["max_ray_error_m"] > args.max_ray_error_m,
                item["mean_ray_error_m"],
                item["max_ray_error_m"],
            ),
        )
        return best_rejected

    return min(
        candidates,
        key=lambda item: (
            -len(item["rays"]),
            item["mean_ray_error_m"],
            item["max_ray_error_m"],
        ),
    )


def triangulate_group(
    group_id: int,
    records: list[dict[str, Any]],
    camera_poses: dict[str, tuple[np.ndarray, np.ndarray]],
    args: argparse.Namespace,
    depth_history: dict[tuple[str, int], float],
    tracked_hands: dict[tuple[int, str], list[dict[str, Any]]],
) -> dict[str, Any]:
    hands_by_key: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    timestamps = []

    for record in records:
        camera_id = record["camera_id"]
        if camera_id not in camera_poses:
            continue
        if record.get("timestamp_unix_ns") is not None:
            timestamps.append(int(record["timestamp_unix_ns"]))
        tracks = tracked_hands.get((group_id, camera_id), [])
        for key, hand in choose_hands(record, args.match_key, args.min_handedness_score, tracks).items():
            hands_by_key[key][camera_id] = hand

    output_hands = []
    for key in sorted(hands_by_key):
        camera_hands = hands_by_key[key]
        primary_camera = hand_primary_camera(key)
        joints = []
        triangulated_count = 0
        fallback_count = 0
        rejected_count = 0

        for landmark_index, landmark_name in enumerate(LANDMARK_NAMES):
            rays = []
            source_cameras = []
            source_scores = []
            for camera_id, hand in sorted(camera_hands.items()):
                original_points = hand.get("landmarks_original_px") or []
                if not args.allow_invalid_original:
                    if landmark_index >= len(original_points) or not original_points[landmark_index].get("valid"):
                        continue
                ray_items = hand.get("rectified_ray_cam") or []
                if landmark_index >= len(ray_items):
                    continue
                origin_h, direction_h = ray_to_headset(camera_id, ray_items[landmark_index], camera_poses)
                rays.append((camera_id, origin_h, direction_h))
                source_cameras.append(camera_id)
                score = hand.get("handedness_score")
                if isinstance(score, (int, float)):
                    source_scores.append(float(score))

            mean_score = float(sum(source_scores) / len(source_scores)) if source_scores else None
            selected = select_best_triangulation(rays, primary_camera, args)
            if selected is not None and "rejected_reason" not in selected:
                point = selected["point"]
                selected_source_cameras = selected["source_cameras"]
                primary_depth = None
                if primary_camera in selected_source_cameras:
                    for camera_id, origin_h, direction_h in selected["rays"]:
                        if camera_id == primary_camera:
                            primary_depth = ray_depth(point, origin_h, direction_h)
                            if primary_depth > 0.0:
                                depth_history[(key, landmark_index)] = primary_depth
                            break

                if primary_depth is None:
                    for camera_id, origin_h, direction_h in selected["rays"]:
                        primary_depth = ray_depth(point, origin_h, direction_h)
                        if primary_depth > 0.0:
                            depth_history[(key, landmark_index)] = primary_depth
                            break

                if landmark_index == 0 and primary_depth is not None and primary_depth > 0.0:
                    depth_history[(key, 0)] = primary_depth

                triangulated_count += 1
                joints.append(
                    {
                        "name": landmark_name,
                        "index": landmark_index,
                        "valid": True,
                        "metric_valid": True,
                        "reconstruction_mode": "triangulated",
                        "position": point.tolist(),
                        "primary_camera": primary_camera,
                        "source_cameras": selected_source_cameras,
                        "view_count": len(selected["rays"]),
                        "mean_ray_error_m": selected["mean_ray_error_m"],
                        "max_ray_error_m": selected["max_ray_error_m"],
                        "mean_handedness_score": mean_score,
                    }
                )
                continue

            if selected is not None and selected.get("rejected_reason"):
                rejected_count += 1

            primary_ray = None
            if primary_camera is not None and len(rays) == 1 and rays[0][0] == primary_camera:
                primary_ray = rays[0]
            if primary_ray is not None:
                depth = depth_history.get((key, landmark_index))
                if depth is None:
                    depth = depth_history.get((key, 0))
                if depth is not None and depth > 0.0:
                    _, origin_h, direction_h = primary_ray
                    point = origin_h + depth * direction_h
                    fallback_count += 1
                    joints.append(
                        {
                            "name": landmark_name,
                            "index": landmark_index,
                            "valid": True,
                            "metric_valid": False,
                            "reconstruction_mode": "primary_depth_fallback",
                            "position": point.tolist(),
                            "primary_camera": primary_camera,
                            "source_cameras": [primary_camera],
                            "view_count": 1,
                            "mean_handedness_score": mean_score,
                        }
                    )
                    continue

            if selected is not None and selected.get("rejected_reason"):
                joints.append(
                    {
                        "name": landmark_name,
                        "index": landmark_index,
                        "valid": False,
                        "metric_valid": False,
                        "reconstruction_mode": "rejected",
                        "position": None,
                        "primary_camera": primary_camera,
                        "source_cameras": selected["source_cameras"],
                        "view_count": len(selected["rays"]),
                        "mean_ray_error_m": selected["mean_ray_error_m"],
                        "max_ray_error_m": selected["max_ray_error_m"],
                        "mean_handedness_score": mean_score,
                        "rejected_reason": selected["rejected_reason"],
                    }
                )
            else:
                joints.append(
                    {
                        "name": landmark_name,
                        "index": landmark_index,
                        "valid": False,
                        "metric_valid": False,
                        "reconstruction_mode": "missing",
                        "position": None,
                        "primary_camera": primary_camera,
                        "source_cameras": source_cameras,
                        "view_count": len(rays),
                    }
                )

        if triangulated_count or fallback_count:
            output_hands.append(
                {
                    "handedness": key,
                    "primary_camera": primary_camera,
                    "source_camera_count": len(camera_hands),
                    "triangulated_joint_count": triangulated_count,
                    "fallback_joint_count": fallback_count,
                    "rejected_joint_count": rejected_count,
                    "joints": joints,
                }
            )

    return {
        "type": "triangulated_mediapipe_hand_frame",
        "source": "headcam_mediapipe_multiview",
        "group_id": group_id,
        "timestamp_unix_ns": min(timestamps) if timestamps else None,
        "hands": output_hands,
    }


def triangulate_file(args: argparse.Namespace, output_jsonl: Path) -> Counter:
    camera_poses = load_camera_poses(args.calib)
    tracked_hands = load_tracked_hands(args.tracked_hands)
    stats: Counter = Counter()
    depth_history: dict[tuple[str, int], float] = {}
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    with output_jsonl.open("w", encoding="utf-8") as out:
        for group_id, records in group_detections(args.detections, args.max_groups, args.stride):
            frame = triangulate_group(group_id, records, camera_poses, args, depth_history, tracked_hands)
            stats["frames"] += 1
            stats["hands"] += len(frame["hands"])
            for hand in frame["hands"]:
                stats[f"hands:{hand['handedness']}"] += 1
                stats["triangulated_joints"] += hand["triangulated_joint_count"]
                stats["fallback_joints"] += hand.get("fallback_joint_count", 0)
                stats["rejected_joints"] += hand.get("rejected_joint_count", 0)
            out.write(json.dumps(frame, ensure_ascii=False, separators=(",", ":")) + "\n")

    return stats


def is_xyz(value: object) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 3
        and all(isinstance(item, (int, float)) and math.isfinite(float(item)) for item in value)
    )


def iter_triangulated_frames(path: Path) -> Iterable[dict[str, Any]]:
    for record in iter_jsonl(path):
        if record.get("type") == "triangulated_mediapipe_hand_frame":
            yield record


def collect_plot_data(path: Path) -> tuple[dict[tuple[str, str], list[tuple[int, list[float]]]], list[dict]]:
    tracks: dict[tuple[str, str], list[tuple[int, list[float]]]] = defaultdict(list)
    frames = []
    track_names = set(DEFAULT_TRACK_JOINTS)

    for frame in iter_triangulated_frames(path):
        group_id = int(frame["group_id"])
        frame_joints = {}
        for hand in frame.get("hands", []):
            handedness = hand.get("handedness", "unknown")
            for joint in hand.get("joints", []):
                name = joint.get("name")
                position = joint.get("position")
                if joint.get("valid") and is_xyz(position):
                    frame_joints[(handedness, name)] = position
                    if name in track_names:
                        tracks[(handedness, name)].append((group_id, position))
        frames.append({"group_id": group_id, "joints": frame_joints})

    return tracks, frames


def set_axes_equal(ax) -> None:
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    max_range = max(
        abs(x_limits[1] - x_limits[0]),
        abs(y_limits[1] - y_limits[0]),
        abs(z_limits[1] - z_limits[0]),
    )
    if max_range == 0:
        max_range = 1.0
    centers = [
        sum(x_limits) / 2,
        sum(y_limits) / 2,
        sum(z_limits) / 2,
    ]
    radius = max_range / 2
    ax.set_xlim3d(centers[0] - radius, centers[0] + radius)
    ax.set_ylim3d(centers[1] - radius, centers[1] + radius)
    ax.set_zlim3d(centers[2] - radius, centers[2] + radius)


def set_axes_from_points(ax, points: list[list[float]]) -> None:
    if not points:
        return
    arr = np.asarray(points, dtype=np.float64)
    low = np.percentile(arr, 1, axis=0)
    high = np.percentile(arr, 99, axis=0)
    margin = np.maximum((high - low) * 0.08, 0.02)
    ax.set_xlim(float(low[0] - margin[0]), float(high[0] + margin[0]))
    ax.set_ylim(float(low[1] - margin[1]), float(high[1] + margin[1]))
    ax.set_zlim(float(low[2] - margin[2]), float(high[2] + margin[2]))
    set_axes_equal(ax)


def draw_camera_rig(ax, calib_path: Path) -> None:
    camera_poses = load_camera_poses(calib_path)
    for camera_id, (origin, rotation) in sorted(camera_poses.items()):
        ax.scatter([origin[0]], [origin[1]], [origin[2]], color="black", s=25)
        ax.text(origin[0], origin[1], origin[2], f" {camera_id}", fontsize=8)
        forward = rotation @ np.asarray([0.0, 0.0, 0.05])
        endpoint = origin + forward
        ax.plot([origin[0], endpoint[0]], [origin[1], endpoint[1]], [origin[2], endpoint[2]], color="black", linewidth=1)


def plot_static(path: Path, output_path: Path, args: argparse.Namespace) -> None:
    tracks, frames = collect_plot_data(path)
    all_points = [position for frame in frames for position in frame["joints"].values()]
    if not all_points:
        raise SystemExit("no triangulated joints to plot")

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")

    colors = {"Left": "#2d9cdb", "Right": "#eb5757", "hand": "#27ae60", "unknown": "#9b51e0"}
    for (handedness, name), samples in sorted(tracks.items()):
        if not samples:
            continue
        positions = [position for _, position in samples]
        color = colors.get(handedness, "#9b51e0")
        ax.plot(
            [p[0] for p in positions],
            [p[1] for p in positions],
            [p[2] for p in positions],
            linewidth=1.1,
            alpha=0.75,
            color=color,
            label=f"{handedness}:{name}",
        )
        ax.scatter([positions[-1][0]], [positions[-1][1]], [positions[-1][2]], s=18, color=color)

    if args.show_cameras:
        draw_camera_rig(ax, args.calib)

    ax.set_title(args.title or f"Triangulated MediaPipe hand trajectories: {args.detections.parent.name}")
    ax.set_xlabel("Headset X meters")
    ax.set_ylabel("Headset Y meters")
    ax.set_zlabel("Headset Z meters")
    ax.grid(True)
    set_axes_from_points(ax, all_points)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0, fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    print(f"saved plot: {output_path}")


def animate(path: Path, output_path: Path, args: argparse.Namespace) -> None:
    _, frames = collect_plot_data(path)
    frames = [frame for frame in frames if frame["joints"]]
    if not frames:
        raise SystemExit("no triangulated joints to animate")

    stride = max(1, math.ceil(len(frames) / args.animation_max_frames))
    indices = list(range(0, len(frames), stride))
    if indices[-1] != len(frames) - 1:
        indices.append(len(frames) - 1)

    all_points = [position for frame in frames for position in frame["joints"].values()]
    keys = sorted({key for frame in frames for key in frame["joints"]})
    colors = {"Left": "#2d9cdb", "Right": "#eb5757", "hand": "#27ae60", "unknown": "#9b51e0"}

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    set_axes_from_points(ax, all_points)
    ax.set_xlabel("Headset X meters")
    ax.set_ylabel("Headset Y meters")
    ax.set_zlabel("Headset Z meters")
    ax.grid(True)
    if args.show_cameras:
        draw_camera_rig(ax, args.calib)

    artists = {}
    for key in keys:
        color = colors.get(key[0], "#9b51e0")
        line, = ax.plot([], [], [], linewidth=2.0, color=color, alpha=0.85)
        dot = ax.scatter([], [], [], s=22, color=color)
        artists[key] = (line, dot)

    panel = fig.text(
        0.74,
        0.55,
        "",
        ha="left",
        va="center",
        family="monospace",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "white", "edgecolor": "0.75", "alpha": 0.92},
    )
    fig.subplots_adjust(right=0.70)

    def set_line(line, points: list[list[float]]) -> None:
        line.set_data([p[0] for p in points], [p[1] for p in points])
        line.set_3d_properties([p[2] for p in points])

    def set_dot(dot, point: list[float] | None) -> None:
        if point is None:
            dot._offsets3d = ([], [], [])
        else:
            dot._offsets3d = ([point[0]], [point[1]], [point[2]])

    def update(step: int):
        frame = frames[indices[step]]
        ax.set_title(args.title or f"Triangulated hands group={frame['group_id']}")
        lines = [f"group {frame['group_id']}", ""]

        by_hand: dict[str, dict[int, list[float]]] = defaultdict(dict)
        for (handedness, name), position in frame["joints"].items():
            by_hand[handedness][LANDMARK_NAMES.index(name)] = position

        for key, (line, dot) in artists.items():
            position = frame["joints"].get(key)
            set_dot(dot, position)
            set_line(line, [])

        for handedness, indexed_points in by_hand.items():
            color = colors.get(handedness, "#9b51e0")
            for start, end in HAND_CONNECTIONS:
                a = indexed_points.get(start)
                b = indexed_points.get(end)
                if a is None or b is None:
                    continue
                key = (handedness, LANDMARK_NAMES[end])
                line, _ = artists[key]
                set_line(line, [a, b])
                line.set_color(color)
            wrist = indexed_points.get(0)
            if wrist:
                lines.extend([handedness, f"  wrist x={wrist[0]: .3f}", f"        y={wrist[1]: .3f}", f"        z={wrist[2]: .3f}", ""])

        panel.set_text("\n".join(lines[:24]))
        return [item for pair in artists.values() for item in pair] + [panel]

    anim = FuncAnimation(fig, update, frames=len(indices), interval=1000 / args.fps, blit=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = PillowWriter(fps=args.fps)
    anim.save(output_path, writer=writer, dpi=args.dpi)
    print(f"saved animation: {output_path}")
    print(f"source frames={len(frames)} rendered frames={len(indices)} stride={stride}")


def write_config(args: argparse.Namespace, output_dir: Path, output_jsonl: Path, stats: Counter) -> None:
    config = {
        "detections": str(args.detections),
        "calib": str(args.calib),
        "tracked_hands": str(args.tracked_hands) if args.tracked_hands else None,
        "output_jsonl": str(output_jsonl),
        "min_views": args.min_views,
        "min_handedness_score": args.min_handedness_score,
        "max_mean_ray_error_m": args.max_mean_ray_error_m,
        "max_ray_error_m": args.max_ray_error_m,
        "min_depth_m": args.min_depth_m,
        "max_depth_m": args.max_depth_m,
        "primary_cameras": PRIMARY_CAMERAS,
        "match_key": args.match_key,
        "allow_invalid_original": bool(args.allow_invalid_original),
        "max_groups": args.max_groups,
        "stride": args.stride,
        "stats": dict(stats),
    }
    run_config = args.detections.parent / "run_config.json"
    if run_config.exists():
        config["detection_run_config"] = json.loads(run_config.read_text(encoding="utf-8"))
    with (output_dir / "triangulation_config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
        f.write("\n")


def main() -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    args = parse_args()
    if args.min_views < 2:
        raise SystemExit("--min-views must be at least 2")
    if args.min_handedness_score < 0.0:
        raise SystemExit("--min-handedness-score must be non-negative")
    if args.max_mean_ray_error_m <= 0.0 or args.max_ray_error_m <= 0.0:
        raise SystemExit("ray error thresholds must be positive")
    if args.min_depth_m <= 0.0 or args.max_depth_m <= args.min_depth_m:
        raise SystemExit("depth thresholds must satisfy 0 < min-depth < max-depth")
    if args.stride < 1:
        raise SystemExit("--stride must be at least 1")
    if not args.detections.exists():
        raise SystemExit(f"detections not found: {args.detections}")
    if not args.calib.exists():
        raise SystemExit(f"calib not found: {args.calib}")
    if args.tracked_hands is not None and not args.tracked_hands.exists():
        raise SystemExit(f"tracked hands not found: {args.tracked_hands}")

    output_dir = args.output_dir or (args.detections.parent / "triangulated_primary_strict")
    output_jsonl = output_dir / "triangulated_hands.jsonl"
    stats = triangulate_file(args, output_jsonl)
    write_config(args, output_dir, output_jsonl, stats)

    print("Summary")
    for key in sorted(stats):
        print(f"  {key}: {stats[key]}")
    print(f"wrote: {output_jsonl}")

    plot_path = args.save_plot or (output_dir / "triangulated_overview.png")
    plot_static(output_jsonl, plot_path, args)
    if args.save_gif:
        animate(output_jsonl, args.save_gif, args)


if __name__ == "__main__":
    main()
