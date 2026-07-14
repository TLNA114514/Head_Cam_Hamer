#!/usr/bin/env python3
"""Interactive RGB + 3D skeleton viewer for triangulated/refined MediaPipe hands."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec


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

CAMERA_LAYOUT = {
    "C1": (0, 0),
    "C0": (1, 0),
    "C2": (0, 2),
    "C3": (1, 2),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--triangulated",
        type=Path,
        default=Path("video/sam3_hamer_left_index/hand_local_refined/local_hands.jsonl"),
        help="Triangulated or hand-local refined hand JSONL.",
    )
    parser.add_argument(
        "--frames",
        type=Path,
        default=Path("video/cameras_left_index/frames.jsonl"),
        help="Synchronized frame metadata JSONL.",
    )
    parser.add_argument(
        "--detections",
        type=Path,
        help="MediaPipe per-camera landmarks JSONL. Defaults to TRIANGULATED parent parent / landmarks.jsonl.",
    )
    parser.add_argument("--hamer-predictions", type=Path, help="HaMeR per-view prediction JSONL for rectified 2D overlays.")
    parser.add_argument("--no-hamer-overlay", action="store_true", help="Do not draw HaMeR/refined 2D overlays on RGB panes.")
    parser.add_argument("--image-root", type=Path, default=Path("video/cameras_left_index"), help="Image root.")
    parser.add_argument("--calib", type=Path, default=Path("video/cameras_left_index/cameras.yaml"), help="Camera calibration YAML.")
    parser.add_argument("--no-mediapipe-overlay", action="store_true", help="Show raw RGB images without MediaPipe 2D overlay.")
    parser.add_argument("--show-camera-rig", action=argparse.BooleanOptionalAction, default=True, help="Draw camera positions and forward directions in the 3D view.")
    parser.add_argument("--camera-axis-length", type=float, default=0.08, help="Camera direction marker length in meters.")
    parser.add_argument(
        "--space",
        choices=["world", "root-relative", "palm-local"],
        default="palm-local",
        help="3D coordinate space for refined hand-local inputs.",
    )
    parser.add_argument("--stride", type=int, default=1, help="Playback source-frame stride.")
    parser.add_argument("--group-range", help="Optional inclusive group range such as 1-100.")
    parser.add_argument("--max-frames", type=int, help="Maximum playback frames after stride.")
    parser.add_argument("--interval-ms", type=int, default=80, help="Playback interval.")
    parser.add_argument("--fixed-range", type=float, default=0.75, help="3D axis cube size in meters.")
    parser.add_argument("--joint-size", type=float, default=28, help="3D joint marker size.")
    parser.add_argument("--image-downscale", type=int, default=2, help="Integer RGB display downscale.")
    parser.add_argument("--elev", type=float, default=20, help="Initial 3D elevation.")
    parser.add_argument("--azim", type=float, default=-65, help="Initial 3D azimuth.")
    parser.add_argument("--render-mode", choices=["skeleton", "mesh", "mesh+skeleton"], default="skeleton", help="3D hand rendering mode for MANO outputs.")
    parser.add_argument("--mesh-stride", type=int, default=8, help="Subsample MANO vertices by this stride in the viewer.")
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


def load_frame_images(frames_path: Path) -> dict[int, dict[str, str]]:
    images: dict[int, dict[str, str]] = {}
    for record in iter_jsonl(frames_path):
        group_id = int(record["group_id"])
        camera_id = record["camera_id"]
        images.setdefault(group_id, {})[camera_id] = record["image_path"]
    return images


def load_mediapipe_detections(detections_path: Path) -> dict[int, dict[str, dict[str, Any]]]:
    detections: dict[int, dict[str, dict[str, Any]]] = {}
    for record in iter_jsonl(detections_path):
        group_id = int(record["group_id"])
        camera_id = record["camera_id"]
        detections.setdefault(group_id, {})[camera_id] = record
    return detections


def load_hamer_predictions(path: Path | None) -> dict[int, dict[str, list[dict[str, Any]]]]:
    predictions: dict[int, dict[str, list[dict[str, Any]]]] = {}
    if path is None or not path.exists():
        return predictions
    for record in iter_jsonl(path):
        if record.get("type") != "hamer_multiview_prediction":
            continue
        group_id = int(record["group_id"])
        camera_id = str(record["camera_id"])
        predictions.setdefault(group_id, {}).setdefault(camera_id, []).append(record)
    return predictions


def load_camera_poses(calib_path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    with calib_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    poses = {}
    for camera_id, cam in data["cameras"].items():
        t_h_c = np.asarray(cam["T_H_C"], dtype=np.float64)
        poses[camera_id] = (t_h_c[:3, 3], t_h_c[:3, :3])
    return poses


def parse_group_range(value: str | None) -> set[int] | None:
    if not value:
        return None
    ids = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            ids.update(range(start, end + 1))
        else:
            ids.add(int(part))
    return ids


def normalize_hamer_local_frame(record: dict[str, Any]) -> dict[str, Any]:
    hands = []
    for hand in record.get("hands", []):
        joints = []
        local_values = hand.get("local_joints_m") or hand.get("palm_local_joints_m") or []
        for index, position in enumerate(local_values):
            if not is_xyz(position):
                continue
            offset = -0.09 if hand.get("handedness") == "Left" else 0.09
            display_position = [float(position[0]) + offset, float(position[1]), float(position[2])]
            palm_position = None
            palm_values = hand.get("palm_local_joints_m") or []
            if index < len(palm_values) and is_xyz(palm_values[index]):
                palm_position = [
                    float(palm_values[index][0]) + offset,
                    float(palm_values[index][1]),
                    float(palm_values[index][2]),
                ]
            joints.append(
                {
                    "index": index,
                    "name": LANDMARK_NAMES[index] if index < len(LANDMARK_NAMES) else f"joint_{index}",
                    "valid": True,
                    "metric_valid": bool(hand.get("metric_valid")),
                    "position": display_position,
                    "root_relative_headset_m": display_position,
                    "palm_local_m": palm_position or display_position,
                    "source_cameras": hand.get("used_cameras", []),
                    "rejected_cameras": hand.get("rejected_cameras", []),
                    "reconstruction_mode": hand.get("mode"),
                    "_display_offset_applied": True,
                }
            )
        item = dict(hand)
        item["joints"] = joints
        item["metric_joint_count"] = len([joint for joint in joints if joint.get("metric_valid")])
        item["temporal_fallback_joint_count"] = len([joint for joint in joints if not joint.get("metric_valid")])
        hands.append(item)
    out = dict(record)
    out["hands"] = hands
    return out


def load_triangulated_frames(path: Path, stride: int, max_frames: int | None, group_ids: set[int] | None = None) -> list[dict[str, Any]]:
    frames = []
    for source_index, record in enumerate(iter_jsonl(path)):
        record_type = record.get("type")
        if record_type not in {"triangulated_mediapipe_hand_frame", "hand_local_refined_frame", "hamer_primary_local_frame", "hamer_palm_local_fused_frame", "hamer_mano_local_refined_frame", "hamer_mano_multiview_image_refined_frame", "glove_local_frame"}:
            continue
        if group_ids is not None and int(record["group_id"]) not in group_ids:
            continue
        if source_index % stride != 0:
            continue
        if record_type in {"hamer_primary_local_frame", "hamer_palm_local_fused_frame", "hamer_mano_local_refined_frame", "hamer_mano_multiview_image_refined_frame"}:
            record = normalize_hamer_local_frame(record)
        frames.append(record)
        if max_frames is not None and len(frames) >= max_frames:
            break
    return frames


def is_refined_frame(frame: dict[str, Any]) -> bool:
    return frame.get("type") in {"hand_local_refined_frame", "hamer_primary_local_frame", "hamer_mano_local_refined_frame", "hamer_mano_multiview_image_refined_frame", "glove_local_frame"}


def joint_position_for_space(hand: dict[str, Any], joint: dict[str, Any], space: str, refined: bool) -> list[float] | None:
    if refined:
        key = {
            "world": "position_world_m",
            "root-relative": "root_relative_headset_m",
            "palm-local": "palm_local_m",
        }[space]
        position = joint.get(key)
        if position is None:
            position = joint.get("position")
        if space == "palm-local" and is_xyz(position):
            offset_x = 0.0 if joint.get("_display_offset_applied") else (-0.09 if hand.get("handedness") == "Left" else 0.09)
            return [float(position[0]) + offset_x, float(position[1]), float(position[2])]
    else:
        position = joint.get("position")
    return position if is_xyz(position) else None


def valid_hand_points(hand: dict[str, Any], space: str, refined: bool) -> dict[int, dict[str, Any]]:
    points = {}
    for joint in hand.get("joints", []):
        index = joint.get("index", joint.get("joint_index"))
        position = joint_position_for_space(hand, joint, space, refined)
        if isinstance(index, int) and joint.get("valid") and position is not None:
            item = dict(joint)
            item["position"] = position
            item["_display_offset_applied"] = True
            points[index] = item
    return points


def frame_points(frame: dict[str, Any], space: str) -> list[list[float]]:
    points = []
    refined = is_refined_frame(frame)
    for hand in frame.get("hands", []):
        points.extend(joint["position"] for joint in valid_hand_points(hand, space, refined).values())
    return points


def hand_vertices_for_space(hand: dict[str, Any], space: str, stride: int) -> list[list[float]]:
    if space == "palm-local" and hand.get("palm_local_vertices_m"):
        vertices = hand.get("palm_local_vertices_m") or []
    else:
        vertices = hand.get("local_vertices_m") or []
    step = max(1, int(stride))
    offset = -0.09 if hand.get("handedness") == "Left" else 0.09
    out = []
    for vertex in vertices[::step]:
        if is_xyz(vertex):
            out.append([float(vertex[0]) + offset, float(vertex[1]), float(vertex[2])])
    return out


def frame_source_camera_counts(frame: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for hand in frame.get("hands", []):
        for joint in hand.get("joints", []):
            if not joint.get("valid"):
                continue
            for camera_id in joint.get("source_cameras", []):
                counts[camera_id] = counts.get(camera_id, 0) + 1
    return counts


def frame_rejected_camera_counts(frame: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for hand in frame.get("hands", []):
        for joint in hand.get("joints", []):
            for camera_id in joint.get("rejected_cameras", []):
                counts[camera_id] = counts.get(camera_id, 0) + 1
    return counts


def frame_anchor_camera_labels(frame: dict[str, Any]) -> dict[str, list[str]]:
    labels: dict[str, list[str]] = {}
    for hand in frame.get("hands", []):
        camera_id = hand.get("anchor_camera") or hand.get("primary_camera")
        handedness = hand.get("handedness", "hand")
        if not camera_id:
            continue
        labels.setdefault(str(camera_id), []).append(str(handedness)[:1])
    return labels


def frame_anchor_pose_errors(frame: dict[str, Any]) -> dict[str, list[float]]:
    values: dict[str, list[float]] = {}
    for hand in frame.get("hands", []):
        camera_id = hand.get("anchor_camera") or hand.get("primary_camera")
        pose_error = hand.get("pose_error_m", hand.get("temporal_pose_error_m"))
        if camera_id and isinstance(pose_error, (int, float)):
            values.setdefault(str(camera_id), []).append(float(pose_error))
    return values


def frame_anchor_beta_sources(frame: dict[str, Any]) -> dict[str, list[str]]:
    values: dict[str, list[str]] = {}
    for hand in frame.get("hands", []):
        camera_id = hand.get("anchor_camera") or hand.get("primary_camera")
        beta_source = hand.get("beta_source")
        if camera_id and beta_source:
            values.setdefault(str(camera_id), []).append(str(beta_source).replace("optimized_", "opt_"))
    return values


def set_axes_centered(ax, points: list[list[float]], fixed_range: float) -> None:
    if points:
        arr = np.asarray(points, dtype=np.float64)
        center = np.median(arr, axis=0)
    else:
        center = np.asarray([0.25, 0.0, -0.25], dtype=np.float64)
    radius = fixed_range / 2
    ax.set_xlim(float(center[0] - radius), float(center[0] + radius))
    ax.set_ylim(float(center[1] - radius), float(center[1] + radius))
    ax.set_zlim(float(center[2] - radius), float(center[2] + radius))


def draw_mediapipe_overlay_bgr(image: np.ndarray, detection: dict[str, Any] | None) -> int:
    if not detection:
        return 0

    colors = {
        "Left": (219, 156, 45),
        "Right": (87, 87, 235),
        "unknown": (224, 81, 155),
        None: (224, 81, 155),
    }
    hand_count = 0
    for hand in detection.get("hands") or []:
        points = hand.get("landmarks_original_px") or []
        if not points:
            continue
        handedness = hand.get("handedness")
        color = colors.get(handedness, colors["unknown"])
        valid_points: dict[int, tuple[int, int]] = {}
        for index, point in enumerate(points):
            if not point.get("valid"):
                continue
            x = point.get("x")
            y = point.get("y")
            if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
                continue
            valid_points[index] = (int(round(x)), int(round(y)))

        if not valid_points:
            continue
        hand_count += 1
        for start, end in HAND_CONNECTIONS:
            a = valid_points.get(start)
            b = valid_points.get(end)
            if a is None or b is None:
                continue
            import cv2

            cv2.line(image, a, b, color, 4, cv2.LINE_AA)
        for index, point in valid_points.items():
            import cv2

            radius = 7 if index == 0 else 5
            cv2.circle(image, point, radius, (0, 90, 255), -1, cv2.LINE_AA)
            cv2.circle(image, point, radius + 1, color, 1, cv2.LINE_AA)

        wrist = valid_points.get(0)
        if wrist:
            import cv2

            score = hand.get("handedness_score")
            score_text = f" {float(score):.2f}" if isinstance(score, (int, float)) else ""
            label = f"{handedness or 'hand'}{score_text}"
            cv2.putText(
                image,
                label,
                (wrist[0] + 10, wrist[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                image,
                label,
                (wrist[0] + 10, wrist[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
                cv2.LINE_AA,
            )
    return hand_count


def draw_2d_hand_bgr(image: np.ndarray, points: list[Any], color: tuple[int, int, int], label: str | None = None, radius: int = 4) -> int:
    valid_points: dict[int, tuple[int, int]] = {}
    for index, point in enumerate(points[:21]):
        if isinstance(point, dict):
            x = point.get("x")
            y = point.get("y")
        elif isinstance(point, list) and len(point) >= 2:
            x, y = point[0], point[1]
        else:
            continue
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            continue
        x_i = int(round(float(x)))
        y_i = int(round(float(y)))
        if 0 <= x_i < image.shape[1] and 0 <= y_i < image.shape[0]:
            valid_points[index] = (x_i, y_i)
    if not valid_points:
        return 0
    import cv2

    for start, end in HAND_CONNECTIONS:
        a = valid_points.get(start)
        b = valid_points.get(end)
        if a is not None and b is not None:
            cv2.line(image, a, b, color, 2, cv2.LINE_AA)
    for index, xy in valid_points.items():
        cv2.circle(image, xy, radius + (1 if index == 0 else 0), color, -1, cv2.LINE_AA)
    if label and 0 in valid_points:
        wrist = valid_points[0]
        cv2.putText(image, label, (wrist[0] + 8, max(18, wrist[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(image, label, (wrist[0] + 8, max(18, wrist[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    return 1


def draw_hamer_overlay_bgr(
    image: np.ndarray,
    predictions: list[dict[str, Any]],
    frame: dict[str, Any],
    camera_id: str,
) -> int:
    count = 0
    for pred in predictions:
        handedness = pred.get("handedness", "hand")
        color = (219, 156, 45) if handedness == "Left" else (87, 87, 235)
        points = pred.get("hamer_joints_2d_rectified_px") or []
        count += draw_2d_hand_bgr(image, points, (120, 120, 120), f"H2D {handedness}", radius=3)

    for hand in frame.get("hands") or []:
        projection = (hand.get("projection_debug") or {}).get(camera_id)
        if not projection:
            continue
        handedness = hand.get("handedness", "hand")
        used = bool(projection.get("used"))
        color = (220, 180, 45) if handedness == "Left" else (70, 90, 235)
        if not used:
            color = (90, 90, 90)
        err = projection.get("mean_reprojection_error_px")
        err_text = f"{float(err):.1f}px" if isinstance(err, (int, float)) else "-"
        label = f"ref {handedness} {'used' if used else 'rej'} {err_text}"
        count += draw_2d_hand_bgr(image, projection.get("joints_2d_px") or [], color, label, radius=4)
    return count


class RgbSkeletonViewer:
    def __init__(
        self,
        args: argparse.Namespace,
        frames: list[dict[str, Any]],
        frame_images: dict[int, dict[str, str]],
        detections: dict[int, dict[str, dict[str, Any]]],
        hamer_predictions: dict[int, dict[str, list[dict[str, Any]]]],
        camera_poses: dict[str, tuple[np.ndarray, np.ndarray]],
    ) -> None:
        self.args = args
        self.frames = frames
        self.frame_images = frame_images
        self.detections = detections
        self.hamer_predictions = hamer_predictions
        self.camera_poses = camera_poses
        self.index = 0
        self.paused = False
        self.image_artists = {}
        self.image_axes = {}
        self.skeleton_artists = []
        self.camera_artists = []
        self.status_text = None

        self.fig = plt.figure(figsize=(15, 8))
        grid = GridSpec(2, 3, figure=self.fig, width_ratios=[1.05, 1.8, 1.05], wspace=0.04, hspace=0.08)

        for camera_id, (row, col) in CAMERA_LAYOUT.items():
            ax = self.fig.add_subplot(grid[row, col])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(camera_id, fontsize=11, pad=4)
            image_artist = ax.imshow(np.zeros((600, 800, 3), dtype=np.uint8))
            self.image_axes[camera_id] = ax
            self.image_artists[camera_id] = image_artist

        self.ax3d = self.fig.add_subplot(grid[:, 1], projection="3d")
        self.ax3d.set_xlabel("Headset X meters")
        self.ax3d.set_ylabel("Headset Y meters")
        self.ax3d.set_zlabel("Headset Z meters")
        self.ax3d.grid(True)
        self.ax3d.view_init(elev=args.elev, azim=args.azim)

        self.status_text = self.fig.text(
            0.50,
            0.02,
            "",
            ha="center",
            va="bottom",
            family="monospace",
            fontsize=9,
        )

        self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)
        if args.space == "world":
            self.draw_camera_rig()

    def on_key_press(self, event: Any) -> None:
        if event.key == " ":
            self.paused = not self.paused
            state = "paused" if self.paused else "running"
            print(f"viewer {state}")
        elif event.key in {"right", "d"}:
            self.index = min(self.index + 1, len(self.frames) - 1)
            self.draw_current()
        elif event.key in {"left", "a"}:
            self.index = max(self.index - 1, 0)
            self.draw_current()
        elif event.key == "home":
            self.index = 0
            self.draw_current()
        elif event.key == "end":
            self.index = len(self.frames) - 1
            self.draw_current()

    def read_rgb(self, frame: dict[str, Any], camera_id: str) -> tuple[np.ndarray, int]:
        import cv2

        group_id = int(frame["group_id"])
        hamer_items = [] if self.args.no_hamer_overlay else self.hamer_predictions.get(group_id, {}).get(camera_id, [])
        rectified_path = None
        for item in hamer_items:
            if item.get("rectified_image_path"):
                rectified_path = Path(item["rectified_image_path"])
                break
        rel_path = self.frame_images.get(group_id, {}).get(camera_id)
        if rectified_path is not None:
            path = rectified_path
        elif rel_path is not None:
            path = self.args.image_root / rel_path
        else:
            return np.zeros((600, 800, 3), dtype=np.uint8), 0
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            return np.zeros((600, 800, 3), dtype=np.uint8), 0
        hand_count = 0
        if hamer_items:
            hand_count = draw_hamer_overlay_bgr(bgr, hamer_items, frame, camera_id)
        elif not self.args.no_mediapipe_overlay:
            detection = self.detections.get(group_id, {}).get(camera_id)
            hand_count = draw_mediapipe_overlay_bgr(bgr, detection)
        if self.args.image_downscale > 1:
            width = bgr.shape[1] // self.args.image_downscale
            height = bgr.shape[0] // self.args.image_downscale
            bgr = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), hand_count

    def clear_skeleton(self) -> None:
        for artist in self.skeleton_artists:
            artist.remove()
        self.skeleton_artists = []

    def draw_camera_rig(self) -> None:
        if not self.args.show_camera_rig:
            return
        colors = {"C0": "#333333", "C1": "#111111", "C2": "#111111", "C3": "#333333"}
        for camera_id, (origin, rotation) in sorted(self.camera_poses.items()):
            color = colors.get(camera_id, "#111111")
            point = self.ax3d.scatter([origin[0]], [origin[1]], [origin[2]], color=color, s=35, depthshade=True)
            self.camera_artists.append(point)
            label = self.ax3d.text(origin[0], origin[1], origin[2], f" {camera_id}", color=color, fontsize=9)
            self.camera_artists.append(label)

            forward = rotation @ np.asarray([0.0, 0.0, self.args.camera_axis_length])
            right = rotation @ np.asarray([self.args.camera_axis_length * 0.55, 0.0, 0.0])
            down = rotation @ np.asarray([0.0, self.args.camera_axis_length * 0.35, 0.0])
            end = origin + forward
            right_end = end + right
            left_end = end - right
            lower_end = end + down
            for a, b, width in [
                (origin, end, 2.2),
                (end, right_end, 1.3),
                (end, left_end, 1.3),
                (end, lower_end, 1.1),
            ]:
                line, = self.ax3d.plot(
                    [a[0], b[0]],
                    [a[1], b[1]],
                    [a[2], b[2]],
                    color=color,
                    linewidth=width,
                    alpha=0.9,
                )
                self.camera_artists.append(line)

    def draw_current(self) -> None:
        frame = self.frames[self.index]
        group_id = int(frame["group_id"])
        used_camera_counts = frame_source_camera_counts(frame)
        rejected_camera_counts = frame_rejected_camera_counts(frame)
        anchor_labels = frame_anchor_camera_labels(frame)
        pose_errors = frame_anchor_pose_errors(frame)
        beta_sources = frame_anchor_beta_sources(frame)

        for camera_id, artist in self.image_artists.items():
            rgb, hand_count = self.read_rgb(frame, camera_id)
            artist.set_data(rgb)
            anchor_text = "".join(anchor_labels.get(camera_id, [])) or "-"
            err_values = pose_errors.get(camera_id, [])
            err_text = f"{np.mean(err_values):.3f}" if err_values else "-"
            beta_text = ",".join(sorted(set(beta_sources.get(camera_id, [])))) or "-"
            self.image_axes[camera_id].set_title(
                f"{camera_id}  group {group_id}  2D hands={hand_count}  "
                f"anchor={anchor_text}  used={used_camera_counts.get(camera_id, 0)}  rej={rejected_camera_counts.get(camera_id, 0)}  "
                f"err={err_text} beta={beta_text}",
                fontsize=11,
                pad=4,
            )

        self.clear_skeleton()
        self.draw_skeleton(frame)
        self.update_status(frame)
        self.fig.canvas.draw_idle()

    def draw_skeleton(self, frame: dict[str, Any]) -> None:
        colors = {"Left": "#2d9cdb", "Right": "#eb5757", "hand": "#27ae60", "unknown": "#9b51e0"}
        refined = is_refined_frame(frame)
        all_points = frame_points(frame, self.args.space)
        set_axes_centered(self.ax3d, all_points, self.args.fixed_range)
        title = "Glove palm-local skeleton" if frame.get("type") == "glove_local_frame" else ("Hand-local refined skeleton" if refined else "Triangulated 3D skeleton")
        self.ax3d.set_title(f"{title} ({self.args.space})", fontsize=12)

        for hand in frame.get("hands", []):
            handedness = hand.get("handedness", "unknown")
            color = colors.get(handedness, "#9b51e0")
            points = valid_hand_points(hand, self.args.space, refined)
            if not points:
                points = {}

            if self.args.render_mode in {"mesh", "mesh+skeleton"}:
                vertices = hand_vertices_for_space(hand, self.args.space, self.args.mesh_stride)
                if vertices:
                    scatter = self.ax3d.scatter(
                        [p[0] for p in vertices],
                        [p[1] for p in vertices],
                        [p[2] for p in vertices],
                        s=max(1.0, self.args.joint_size * 0.12),
                        color=color,
                        alpha=0.20,
                        depthshade=True,
                    )
                    self.skeleton_artists.append(scatter)

            if self.args.render_mode == "mesh":
                continue
            if not points:
                continue

            triangulated = [joint["position"] for joint in points.values() if joint.get("metric_valid", True)]
            fallback = [joint["position"] for joint in points.values() if not joint.get("metric_valid", True)]
            if triangulated:
                scatter = self.ax3d.scatter(
                    [p[0] for p in triangulated],
                    [p[1] for p in triangulated],
                    [p[2] for p in triangulated],
                    s=self.args.joint_size,
                    color=color,
                    depthshade=True,
                    label=f"{handedness} triangulated",
                )
                self.skeleton_artists.append(scatter)
            if fallback:
                scatter = self.ax3d.scatter(
                    [p[0] for p in fallback],
                    [p[1] for p in fallback],
                    [p[2] for p in fallback],
                    s=self.args.joint_size * 0.75,
                    color=color,
                    alpha=0.34,
                    depthshade=True,
                    label=f"{handedness} fallback",
                )
                self.skeleton_artists.append(scatter)

            for start, end in HAND_CONNECTIONS:
                a_joint = points.get(start)
                b_joint = points.get(end)
                if a_joint is None or b_joint is None:
                    continue
                a = a_joint["position"]
                b = b_joint["position"]
                is_fallback = not a_joint.get("metric_valid", True) or not b_joint.get("metric_valid", True)
                line, = self.ax3d.plot(
                    [a[0], b[0]],
                    [a[1], b[1]],
                    [a[2], b[2]],
                    color=color,
                    linewidth=1.7 if is_fallback else 2.2,
                    alpha=0.34 if is_fallback else 0.92,
                    linestyle="--" if is_fallback else "-",
                )
                self.skeleton_artists.append(line)

            wrist_joint = points.get(0)
            if wrist_joint:
                wrist = wrist_joint["position"]
                anchor_camera = hand.get("anchor_camera") or hand.get("primary_camera")
                label = f" {handedness}"
                if anchor_camera:
                    label += f" @{anchor_camera}"
                if not wrist_joint.get("metric_valid", True):
                    label += " fallback"
                text = self.ax3d.text(wrist[0], wrist[1], wrist[2], label, color=color, fontsize=9)
                self.skeleton_artists.append(text)

    def update_status(self, frame: dict[str, Any]) -> None:
        hands = frame.get("hands", [])
        joint_count = sum(
            hand.get("triangulated_joint_count", hand.get("metric_joint_count", 0))
            for hand in hands
        )
        fallback_count = sum(
            hand.get("fallback_joint_count", hand.get("temporal_fallback_joint_count", 0))
            for hand in hands
        )
        state = "paused" if self.paused else "playing"
        self.status_text.set_text(
            f"{state} | frame {self.index + 1}/{len(self.frames)} | "
            f"group_id={frame.get('group_id')} | hands={len(hands)} metric={joint_count} fallback={fallback_count} | "
            "Space pause, Left/Right step, drag middle 3D view"
        )

    def update(self, _step: int):
        if not self.paused:
            self.index = (self.index + 1) % len(self.frames)
            self.draw_current()
        return list(self.image_artists.values()) + self.skeleton_artists


def main() -> None:
    args = parse_args()
    if args.stride < 1:
        raise SystemExit("--stride must be at least 1")
    if args.image_downscale < 1:
        raise SystemExit("--image-downscale must be at least 1")
    if not args.triangulated.exists():
        raise SystemExit(f"triangulated file not found: {args.triangulated}")
    if not args.frames.exists():
        raise SystemExit(f"frames file not found: {args.frames}")
    if args.show_camera_rig and not args.calib.exists():
        raise SystemExit(f"calib file not found: {args.calib}")
    if args.detections is None:
        args.detections = args.triangulated.parent.parent / "landmarks.jsonl"
    if not args.no_mediapipe_overlay and not args.detections.exists() and (args.hamer_predictions is None or not args.hamer_predictions.exists()):
        raise SystemExit(f"detections file not found: {args.detections}")

    frames = load_triangulated_frames(args.triangulated, args.stride, args.max_frames, parse_group_range(args.group_range))
    frames = [frame for frame in frames if frame.get("hands")]
    if not frames:
        raise SystemExit("no triangulated hand frames found")
    frame_images = load_frame_images(args.frames)
    detections = {} if args.no_mediapipe_overlay or not args.detections.exists() else load_mediapipe_detections(args.detections)
    hamer_predictions = load_hamer_predictions(args.hamer_predictions)
    camera_poses = load_camera_poses(args.calib) if args.show_camera_rig else {}

    viewer = RgbSkeletonViewer(args, frames, frame_images, detections, hamer_predictions, camera_poses)
    viewer.draw_current()
    animation = FuncAnimation(viewer.fig, viewer.update, interval=args.interval_ms, blit=False)
    viewer.animation = animation
    plt.show()


if __name__ == "__main__":
    main()
