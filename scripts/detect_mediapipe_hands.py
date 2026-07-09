#!/usr/bin/env python3
"""Detect MediaPipe hand landmarks on synchronized multi-camera images."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import yaml
from progress_utils import tqdm


HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
)


RECTIFY_MODES = {
    "perspective": cv2.omnidir.RECTIFY_PERSPECTIVE,
    "cylindrical": cv2.omnidir.RECTIFY_CYLINDRICAL,
    "longlat": cv2.omnidir.RECTIFY_LONGLATI,
    "stereographic": cv2.omnidir.RECTIFY_STEREOGRAPHIC,
}


@dataclass(frozen=True)
class CameraCalibration:
    camera_id: str
    image_size: tuple[int, int]
    k: np.ndarray
    d: np.ndarray
    xi: np.ndarray
    new_k: np.ndarray
    map_x: np.ndarray
    map_y: np.ndarray
    full_view_map_x: np.ndarray | None = None
    full_view_map_y: np.ndarray | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MediaPipe Hands on omnidir-rectified synchronized camera images."
    )
    parser.add_argument("--input", default="video/cameras_left_index", type=Path, help="Camera image root.")
    parser.add_argument("--calib", default="video/cameras_left_index/cameras.yaml", type=Path, help="Camera YAML.")
    parser.add_argument("--frames", default="video/cameras_left_index/frames.jsonl", type=Path, help="Frame JSONL.")
    parser.add_argument("--output", default="video/mediapipe_left_index", type=Path, help="Output directory.")
    parser.add_argument("--cameras", default="C0,C1,C2,C3", help="Comma-separated camera ids.")
    parser.add_argument("--max-num-hands", default=2, type=int, help="Maximum hands per image.")
    parser.add_argument("--min-detection-confidence", default=0.5, type=float)
    parser.add_argument("--min-tracking-confidence", default=0.5, type=float)
    parser.add_argument(
        "--model-complexity",
        default=1,
        type=int,
        choices=(0, 1),
        help="MediaPipe Hands model complexity.",
    )
    parser.add_argument(
        "--debug-every",
        default=100,
        type=int,
        help="Save one debug overlay every N group ids per camera; 0 disables debug images.",
    )
    parser.add_argument("--progress-every", default=100, type=int, help="Print progress every N images; 0 disables.")
    parser.add_argument("--progress-position", default=int(os.environ.get("TQDM_POSITION", "0")), type=int, help="tqdm terminal row position.")
    parser.add_argument(
        "--rectify-focal-scale",
        default=1.0,
        type=float,
        help=(
            "Scale the virtual perspective focal length. Values below 1.0 show a wider "
            "rectified view, with more black border and more perspective distortion."
        ),
    )
    parser.add_argument(
        "--write-full-view-debug",
        action="store_true",
        help=(
            "Also save longlat undistorted debug images under debug_full/. These are for "
            "visual inspection of the full camera content and are not used for MediaPipe."
        ),
    )
    parser.add_argument(
        "--swap-handedness",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Swap MediaPipe Left/Right labels. MediaPipe Solutions Hands assumes mirrored "
            "selfie input; these headset camera images are not mirrored, so swapping is "
            "enabled by default."
        ),
    )
    parser.add_argument(
        "--repair-duplicate-handedness",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If exactly two detected hands have the same Left/Right label, keep the higher-score label and flip the lower-score hand.",
    )
    parser.add_argument(
        "--limit-groups",
        default=None,
        type=int,
        help="Process only the first N group ids, useful for smoke tests.",
    )
    parser.add_argument("--group-range", help='Process specific group ids/ranges, e.g. "1-100" or "1-20,50".')
    parser.add_argument("--group-ids", help='Process specific comma-separated group ids, e.g. "1,7,42".')
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing landmarks.jsonl.",
    )
    return parser.parse_args()


def load_mediapipe() -> Any:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    try:
        import mediapipe as mp  # type: ignore
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency: mediapipe. Install it with:\n"
            "  python3 -m pip install mediapipe"
        ) from exc
    if not hasattr(mp, "solutions") or not hasattr(mp.solutions, "hands"):
        version = getattr(mp, "__version__", "unknown")
        raise SystemExit(
            "This script needs the classic MediaPipe Solutions Hands API, "
            f"but mediapipe {version} does not expose mp.solutions.hands.\n"
            "Install a compatible version, for example:\n"
            "  python3 -m pip install mediapipe==0.10.21 protobuf==4.25.9"
        )
    return mp


def build_undistort_maps(
    k: np.ndarray,
    d: np.ndarray,
    xi: np.ndarray,
    new_k: np.ndarray,
    image_size: tuple[int, int],
    rectify_mode: int,
) -> tuple[np.ndarray, np.ndarray]:
    return cv2.omnidir.initUndistortRectifyMap(
        k,
        d,
        xi,
        np.eye(3, dtype=np.float64),
        new_k,
        image_size,
        cv2.CV_32FC1,
        rectify_mode,
    )


def load_calibrations(
    calib_path: Path,
    camera_ids: set[str],
    rectify_focal_scale: float,
    write_full_view_debug: bool,
) -> dict[str, CameraCalibration]:
    with calib_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if rectify_focal_scale <= 0.0:
        raise ValueError("--rectify-focal-scale must be positive")

    default_size = tuple(int(v) for v in data["camera_defaults"]["image_size"])
    if len(default_size) != 2:
        raise ValueError(f"camera_defaults.image_size must be [width, height]: {default_size}")

    calibrations: dict[str, CameraCalibration] = {}
    for camera_id in sorted(camera_ids):
        if camera_id not in data["cameras"]:
            raise KeyError(f"Camera {camera_id!r} not found in {calib_path}")
        cam = data["cameras"][camera_id]
        image_size = tuple(int(v) for v in cam.get("image_size", default_size))
        k = np.asarray(cam["intrinsics"], dtype=np.float64)
        d = np.asarray(cam["distortion"], dtype=np.float64).reshape(1, -1)
        xi = np.asarray([[cam["xi"]]], dtype=np.float64)
        new_k = k.copy()
        new_k[0, 0] *= rectify_focal_scale
        new_k[1, 1] *= rectify_focal_scale
        map_x, map_y = build_undistort_maps(
            k, d, xi, new_k, image_size, RECTIFY_MODES["perspective"]
        )

        full_view_map_x = None
        full_view_map_y = None
        if write_full_view_debug:
            full_view_map_x, full_view_map_y = build_undistort_maps(
                k, d, xi, k.copy(), image_size, RECTIFY_MODES["longlat"]
            )

        calibrations[camera_id] = CameraCalibration(
            camera_id=camera_id,
            image_size=(int(image_size[0]), int(image_size[1])),
            k=k,
            d=d,
            xi=xi,
            new_k=new_k,
            map_x=map_x,
            map_y=map_y,
            full_view_map_x=full_view_map_x,
            full_view_map_y=full_view_map_y,
        )
    return calibrations


def parse_group_ids(group_range: str | None, group_ids: str | None) -> set[int] | None:
    ids: set[int] = set()
    if group_range:
        for part in group_range.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                start_s, end_s = part.split("-", 1)
                start = int(start_s)
                end = int(end_s)
                if end < start:
                    raise ValueError(f"Invalid group range: {part}")
                ids.update(range(start, end + 1))
            else:
                ids.add(int(part))
    if group_ids:
        ids.update(int(part.strip()) for part in group_ids.split(",") if part.strip())
    return ids or None


def iter_frame_records(
    frames_path: Path,
    camera_ids: set[str],
    limit_groups: int | None,
    group_ids: set[int] | None,
) -> Iterable[dict[str, Any]]:
    seen_groups: set[int] = set()
    with frames_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("camera_id") not in camera_ids:
                continue
            group_id = int(record["group_id"])
            if group_ids is not None and group_id not in group_ids:
                continue
            if limit_groups is not None:
                if group_id not in seen_groups and len(seen_groups) >= limit_groups:
                    continue
                seen_groups.add(group_id)
            record["_line_no"] = line_no
            yield record


def bilinear_sample(map_img: np.ndarray, x: float, y: float) -> tuple[float, bool]:
    height, width = map_img.shape[:2]
    if not (0.0 <= x <= width - 1 and 0.0 <= y <= height - 1):
        return float("nan"), False

    x0 = int(math.floor(x))
    y0 = int(math.floor(y))
    x1 = min(x0 + 1, width - 1)
    y1 = min(y0 + 1, height - 1)
    wx = x - x0
    wy = y - y0

    top = (1.0 - wx) * float(map_img[y0, x0]) + wx * float(map_img[y0, x1])
    bottom = (1.0 - wx) * float(map_img[y1, x0]) + wx * float(map_img[y1, x1])
    return (1.0 - wy) * top + wy * bottom, True


def rectified_to_original(calib: CameraCalibration, x: float, y: float) -> dict[str, Any]:
    orig_x, ok_x = bilinear_sample(calib.map_x, x, y)
    orig_y, ok_y = bilinear_sample(calib.map_y, x, y)
    in_source = (
        ok_x
        and ok_y
        and 0.0 <= orig_x < calib.image_size[0]
        and 0.0 <= orig_y < calib.image_size[1]
    )
    return {
        "x": none_if_nan(orig_x),
        "y": none_if_nan(orig_y),
        "valid": bool(in_source),
    }


def rectified_ray(calib: CameraCalibration, x: float, y: float) -> dict[str, float]:
    fx = float(calib.new_k[0, 0])
    fy = float(calib.new_k[1, 1])
    cx = float(calib.new_k[0, 2])
    cy = float(calib.new_k[1, 2])
    ray = np.asarray([(x - cx) / fx, (y - cy) / fy, 1.0], dtype=np.float64)
    ray /= np.linalg.norm(ray)
    return {"x": float(ray[0]), "y": float(ray[1]), "z": float(ray[2])}


def none_if_nan(value: float) -> float | None:
    return None if math.isnan(value) else float(value)


def draw_debug_overlay(image: np.ndarray, hands: list[dict[str, Any]]) -> np.ndarray:
    overlay = image.copy()
    for hand in hands:
        points = hand["landmarks_rectified_px"]
        for start, end in HAND_CONNECTIONS:
            p0 = points[start]
            p1 = points[end]
            cv2.line(
                overlay,
                (int(round(p0["x"])), int(round(p0["y"]))),
                (int(round(p1["x"])), int(round(p1["y"]))),
                (0, 220, 255),
                2,
                cv2.LINE_AA,
            )
        for idx, point in enumerate(points):
            radius = 5 if idx == 0 else 3
            cv2.circle(
                overlay,
                (int(round(point["x"])), int(round(point["y"]))),
                radius,
                (0, 80, 255),
                -1,
                cv2.LINE_AA,
            )
        label = f'{hand.get("handedness", "hand")} {hand.get("handedness_score", 0.0):.2f}'
        wrist = points[0]
        cv2.putText(
            overlay,
            label,
            (int(round(wrist["x"])) + 8, int(round(wrist["y"])) - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return overlay


def swap_handedness_label(label: str | None) -> str | None:
    if label == "Left":
        return "Right"
    if label == "Right":
        return "Left"
    return label


def mediapipe_hands_to_dicts(results: Any, calib: CameraCalibration, swap_handedness: bool) -> list[dict[str, Any]]:
    if not results.multi_hand_landmarks:
        return []

    handedness_items = results.multi_handedness or []
    hands: list[dict[str, Any]] = []
    width, height = calib.image_size
    for hand_index, hand_landmarks in enumerate(results.multi_hand_landmarks):
        handedness = None
        handedness_score = None
        if hand_index < len(handedness_items) and handedness_items[hand_index].classification:
            cls = handedness_items[hand_index].classification[0]
            handedness = cls.label
            if swap_handedness:
                handedness = swap_handedness_label(handedness)
            handedness_score = float(cls.score)

        landmarks_norm: list[dict[str, float]] = []
        landmarks_rectified_px: list[dict[str, float]] = []
        landmarks_original_px: list[dict[str, Any]] = []
        rays: list[dict[str, float]] = []

        for landmark in hand_landmarks.landmark:
            norm = {
                "x": float(landmark.x),
                "y": float(landmark.y),
                "z": float(landmark.z),
            }
            if hasattr(landmark, "visibility"):
                norm["visibility"] = float(landmark.visibility)
            landmarks_norm.append(norm)

            x = float(landmark.x) * width
            y = float(landmark.y) * height
            rectified = {"x": x, "y": y, "z": float(landmark.z)}
            if hasattr(landmark, "visibility"):
                rectified["visibility"] = float(landmark.visibility)
            landmarks_rectified_px.append(rectified)
            landmarks_original_px.append(rectified_to_original(calib, x, y))
            rays.append(rectified_ray(calib, x, y))

        hands.append(
            {
                "hand_index": hand_index,
                "handedness": handedness,
                "handedness_score": handedness_score,
                "landmarks_rectified_px": landmarks_rectified_px,
                "landmarks_original_px": landmarks_original_px,
                "landmarks_rectified_norm": landmarks_norm,
                "rectified_ray_cam": rays,
            }
        )
    return hands


def opposite_handedness(label: str | None) -> str | None:
    if label == "Left":
        return "Right"
    if label == "Right":
        return "Left"
    return label


def repair_duplicate_handedness(hands: list[dict[str, Any]]) -> int:
    if len(hands) != 2:
        return 0
    labels = [hand.get("handedness") for hand in hands]
    if labels[0] not in {"Left", "Right"} or labels[0] != labels[1]:
        return 0
    scores = [
        float(hand.get("handedness_score")) if isinstance(hand.get("handedness_score"), (int, float)) else -1.0
        for hand in hands
    ]
    flip_index = 0 if scores[0] < scores[1] else 1
    original = hands[flip_index].get("handedness")
    hands[flip_index]["original_handedness"] = original
    hands[flip_index]["handedness"] = opposite_handedness(original)
    hands[flip_index]["handedness_repair_reason"] = "duplicate_two_hands_flip_lower_score"
    return 1


def summarize(stats: dict[str, dict[str, float]]) -> None:
    print("\nSummary")
    total_frames = 0
    for camera_id in sorted(stats):
        item = stats[camera_id]
        frames = int(item["frames"])
        total_frames += frames
        detected_frames = int(item["detected_frames"])
        failed_frames = int(item["failed_frames"])
        total_hands = int(item["total_hands"])
        repaired = int(item.get("repaired_duplicate_handedness", 0))
        avg_hands = total_hands / frames if frames else 0.0
        print(
            f"  {camera_id}: frames={frames} detected_frames={detected_frames} "
            f"failed_frames={failed_frames} total_hands={total_hands} "
            f"avg_hands={avg_hands:.3f} repaired_duplicate_handedness={repaired}"
        )
    print(f"  total_frames={total_frames}")


def write_run_config(
    output_dir: Path,
    args: argparse.Namespace,
    calibrations: dict[str, CameraCalibration],
) -> None:
    config = {
        "input": str(args.input),
        "calib": str(args.calib),
        "frames": str(args.frames),
        "cameras": sorted(calibrations),
        "rectify_mode": "perspective",
        "rectify_focal_scale": args.rectify_focal_scale,
        "write_full_view_debug": bool(args.write_full_view_debug),
        "swap_handedness": bool(args.swap_handedness),
        "repair_duplicate_handedness": bool(args.repair_duplicate_handedness),
        "max_num_hands": args.max_num_hands,
        "min_detection_confidence": args.min_detection_confidence,
        "min_tracking_confidence": args.min_tracking_confidence,
        "model_complexity": args.model_complexity,
        "new_intrinsics": {
            camera_id: calib.new_k.tolist()
            for camera_id, calib in sorted(calibrations.items())
        },
    }
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
        f.write("\n")


def main() -> int:
    warnings.filterwarnings("ignore")
    args = parse_args()
    mp = load_mediapipe()
    group_ids = parse_group_ids(args.group_range, args.group_ids)

    camera_ids = {item.strip() for item in args.cameras.split(",") if item.strip()}
    if not camera_ids:
        raise SystemExit("--cameras must include at least one camera id")

    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "landmarks.jsonl"
    if output_path.exists() and not args.overwrite:
        raise SystemExit(f"{output_path} already exists; pass --overwrite to replace it.")

    calibrations = load_calibrations(
        args.calib,
        camera_ids,
        args.rectify_focal_scale,
        args.write_full_view_debug,
    )
    write_run_config(output_dir, args, calibrations)
    stats = {
        camera_id: {"frames": 0, "detected_frames": 0, "failed_frames": 0, "total_hands": 0, "repaired_duplicate_handedness": 0}
        for camera_id in camera_ids
    }

    hands_solution = mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=args.max_num_hands,
        model_complexity=args.model_complexity,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )

    with hands_solution as hands, output_path.open("w", encoding="utf-8") as out:
        frame_records = list(iter_frame_records(args.frames, camera_ids, args.limit_groups, group_ids))
        progress = tqdm(frame_records, desc=f"MediaPipe {','.join(sorted(camera_ids))}", unit="image", position=args.progress_position)
        for index, frame in enumerate(progress, start=1):
            camera_id = frame["camera_id"]
            group_id = int(frame["group_id"])
            calib = calibrations[camera_id]
            image_path = args.input / frame["image_path"]
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(f"Could not read image from frames line {frame['_line_no']}: {image_path}")

            rectified = cv2.remap(
                image,
                calib.map_x,
                calib.map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )
            rgb = cv2.cvtColor(rectified, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = hands.process(rgb)
            hand_dicts = mediapipe_hands_to_dicts(results, calib, args.swap_handedness)
            if args.repair_duplicate_handedness:
                stats[camera_id]["repaired_duplicate_handedness"] += repair_duplicate_handedness(hand_dicts)

            stats[camera_id]["frames"] += 1
            stats[camera_id]["total_hands"] += len(hand_dicts)
            if hand_dicts:
                stats[camera_id]["detected_frames"] += 1
            else:
                stats[camera_id]["failed_frames"] += 1

            record = {
                "group_id": group_id,
                "camera_id": camera_id,
                "image_path": frame["image_path"],
                "timestamp_unix_ns": frame.get("timestamp_unix_ns"),
                "skew_us": frame.get("skew_us"),
                "hands": hand_dicts,
            }
            out.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")

            if args.debug_every > 0 and group_id % args.debug_every == 0:
                debug_dir = output_dir / "debug" / camera_id
                debug_dir.mkdir(parents=True, exist_ok=True)
                debug_path = debug_dir / Path(frame["image_path"]).name
                cv2.imwrite(str(debug_path), draw_debug_overlay(rectified, hand_dicts))

                if calib.full_view_map_x is not None and calib.full_view_map_y is not None:
                    full_view = cv2.remap(
                        image,
                        calib.full_view_map_x,
                        calib.full_view_map_y,
                        interpolation=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT,
                    )
                    full_view_dir = output_dir / "debug_full" / camera_id
                    full_view_dir.mkdir(parents=True, exist_ok=True)
                    full_view_path = full_view_dir / Path(frame["image_path"]).name
                    cv2.imwrite(str(full_view_path), full_view)

            if args.progress_every > 0 and index % args.progress_every == 0:
                progress.set_postfix_str(f"group={group_id} camera={camera_id} hands={len(hand_dicts)}")

    summarize(stats)
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
