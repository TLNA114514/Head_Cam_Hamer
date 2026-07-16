#!/usr/bin/env python3
"""Shared helpers for the multi-view SAM3 + HaMeR pipeline."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import yaml


DEFAULT_BASE_DIR = Path("video/sam3_hamer_left_index")
DEFAULT_IMAGE_ROOT = Path("video/cameras_left_index")
DEFAULT_FRAMES = Path("video/cameras_left_index/frames.jsonl")
DEFAULT_CALIB = Path("video/cameras_left_index/cameras.yaml")
DEFAULT_CAMERAS = ("C0", "C1", "C2", "C3")
DEFAULT_RECTIFY_FOCAL_SCALE = 0.7
IMAGE_SIZE = (1600, 1200)
PRIMARY_CAMERAS = {"Left": "C1", "Right": "C2"}
BACKUP_PRIMARY_CAMERAS = {"Left": "C0", "Right": "C3"}
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
)


@dataclass(frozen=True)
class RectifyCalibration:
    camera_id: str
    image_size: tuple[int, int]
    camera_model: str
    projection_model: str
    distortion_model: str
    rectify_backend: str
    rectify_focal_scale: float
    map_x: np.ndarray
    map_y: np.ndarray
    new_k: np.ndarray


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


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def parse_cameras(value: str | None) -> set[str]:
    if not value:
        return set(DEFAULT_CAMERAS)
    cameras = {item.strip() for item in value.split(",") if item.strip()}
    if not cameras:
        raise ValueError("--cameras must contain at least one camera id")
    return cameras


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


def range_suffix(group_ids: set[int] | None) -> str:
    if not group_ids:
        return "all"
    ordered = sorted(group_ids)
    if ordered == list(range(ordered[0], ordered[-1] + 1)):
        return f"{ordered[0]:06d}_{ordered[-1]:06d}"
    if len(ordered) <= 4:
        return "_".join(f"{item:06d}" for item in ordered)
    return f"{ordered[0]:06d}_{ordered[-1]:06d}_{len(ordered)}ids"


def filter_frame_records(
    frames_path: Path,
    cameras: set[str],
    group_ids: set[int] | None,
) -> list[dict[str, Any]]:
    records = []
    for record in iter_jsonl(frames_path):
        camera_id = record.get("camera_id")
        group_id = int(record["group_id"])
        if camera_id not in cameras:
            continue
        if group_ids is not None and group_id not in group_ids:
            continue
        records.append(record)
    return records


def build_rectify_calibrations(
    calib_path: Path,
    cameras: set[str],
    rectify_focal_scale: float | None = DEFAULT_RECTIFY_FOCAL_SCALE,
) -> dict[str, RectifyCalibration]:
    with calib_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if rectify_focal_scale is not None and rectify_focal_scale <= 0.0:
        raise ValueError("--rectify-focal-scale must be positive")

    defaults = data.get("camera_defaults") or {}
    default_size = tuple(int(v) for v in defaults["image_size"])
    if len(default_size) != 2:
        raise ValueError(f"camera_defaults.image_size must be [width, height]: {default_size}")

    calibrations = {}
    for camera_id in sorted(cameras):
        if camera_id not in data["cameras"]:
            raise KeyError(f"Camera {camera_id!r} not found in {calib_path}")
        cam = data["cameras"][camera_id]
        image_size = tuple(int(v) for v in cam.get("image_size", default_size))
        if len(image_size) != 2:
            raise ValueError(f"{camera_id}.image_size must be [width, height]: {image_size}")
        k = np.asarray(cam["intrinsics"], dtype=np.float64)
        if k.shape != (3, 3):
            raise ValueError(f"{camera_id}.intrinsics must be a 3x3 matrix, got {k.shape}")
        d = np.asarray(cam["distortion"], dtype=np.float64).reshape(-1)

        inferred_omni = "xi" in cam
        camera_model = str(
            cam.get("camera_model", defaults.get("camera_model", "omni" if inferred_omni else "pinhole"))
        ).strip().lower()
        projection_model = str(
            cam.get("projection_model", defaults.get("projection_model", "mei" if inferred_omni else "pinhole"))
        ).strip().lower()
        distortion_model = str(
            cam.get("distortion_model", defaults.get("distortion_model", "radtan"))
        ).strip().lower().replace("_", "-")

        if camera_model in {"omni", "omnidir", "mei"} or projection_model in {"omni", "omnidir", "mei"}:
            rectify_backend = "opencv-omnidir"
        elif camera_model == "pinhole" and distortion_model in {"equidistant", "fisheye"}:
            rectify_backend = "opencv-fisheye"
        elif camera_model == "pinhole" and distortion_model in {
            "radtan",
            "radial-tangential",
            "plumb-bob",
            "opencv",
        }:
            rectify_backend = "opencv-pinhole"
        else:
            raise ValueError(
                f"Unsupported camera model for {camera_id}: camera_model={camera_model!r}, "
                f"projection_model={projection_model!r}, distortion_model={distortion_model!r}"
            )

        focal_scale = DEFAULT_RECTIFY_FOCAL_SCALE if rectify_focal_scale is None else rectify_focal_scale
        new_k = k.copy()
        new_k[0, 0] *= focal_scale
        new_k[1, 1] *= focal_scale
        identity = np.eye(3, dtype=np.float64)

        if rectify_backend == "opencv-omnidir":
            if "xi" not in cam:
                raise ValueError(f"{camera_id} uses Omni/Mei projection but does not define xi")
            xi = np.asarray([[cam["xi"]]], dtype=np.float64)
            map_x, map_y = cv2.omnidir.initUndistortRectifyMap(
                k,
                d.reshape(1, -1),
                xi,
                identity,
                new_k,
                image_size,
                cv2.CV_32FC1,
                cv2.omnidir.RECTIFY_PERSPECTIVE,
            )
        elif rectify_backend == "opencv-fisheye":
            if d.size != 4:
                raise ValueError(f"{camera_id}.distortion must contain 4 equidistant coefficients, got {d.size}")
            map_x, map_y = cv2.fisheye.initUndistortRectifyMap(
                k,
                d.reshape(4, 1),
                identity,
                new_k,
                image_size,
                cv2.CV_32FC1,
            )
        else:
            map_x, map_y = cv2.initUndistortRectifyMap(
                k,
                d.reshape(-1, 1),
                identity,
                new_k,
                image_size,
                cv2.CV_32FC1,
            )

        calibrations[camera_id] = RectifyCalibration(
            camera_id=camera_id,
            image_size=(int(image_size[0]), int(image_size[1])),
            camera_model=camera_model,
            projection_model=projection_model,
            distortion_model=distortion_model,
            rectify_backend=rectify_backend,
            rectify_focal_scale=float(focal_scale),
            map_x=map_x,
            map_y=map_y,
            new_k=new_k,
        )
    return calibrations


def rectified_rel_path(record: dict[str, Any]) -> Path:
    return Path(record["camera_id"]) / Path(record["image_path"]).name


def clamp_bbox(bbox: Iterable[float], image_size: tuple[int, int] = IMAGE_SIZE) -> list[float]:
    width, height = image_size
    x1, y1, x2, y2 = [float(v) for v in bbox]
    x1 = max(0.0, min(width - 1.0, x1))
    y1 = max(0.0, min(height - 1.0, y1))
    x2 = max(0.0, min(width - 1.0, x2))
    y2 = max(0.0, min(height - 1.0, y2))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def expand_bbox(
    bbox: Iterable[float],
    pad: float,
    image_size: tuple[int, int] = IMAGE_SIZE,
) -> list[float]:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    side = max(x2 - x1, y2 - y1, 2.0) * pad
    half = side * 0.5
    return clamp_bbox([cx - half, cy - half, cx + half, cy + half], image_size)


def union_bbox(a: Iterable[float], b: Iterable[float]) -> list[float]:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    return [min(ax1, bx1), min(ay1, by1), max(ax2, bx2), max(ay2, by2)]


def bbox_iou(a: Iterable[float], b: Iterable[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def bbox_area(bbox: Iterable[float]) -> float:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def mediapipe_bbox(hand: dict[str, Any], pad: float = 1.25) -> list[float] | None:
    points = hand.get("landmarks_rectified_px") or []
    xy = []
    for point in points:
        x = point.get("x")
        y = point.get("y")
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            if math.isfinite(float(x)) and math.isfinite(float(y)):
                xy.append((float(x), float(y)))
    if len(xy) < 4:
        return None
    xs = [item[0] for item in xy]
    ys = [item[1] for item in xy]
    return expand_bbox([min(xs), min(ys), max(xs), max(ys)], pad)


def handedness_to_is_right(handedness: str | None) -> int | None:
    if handedness == "Right":
        return 1
    if handedness == "Left":
        return 0
    return None


def opposite_handedness(handedness: str) -> str:
    if handedness == "Left":
        return "Right"
    if handedness == "Right":
        return "Left"
    return "unknown"


def dilate_mask(mask: np.ndarray, pixels: int) -> np.ndarray:
    if pixels <= 0:
        return mask.astype(bool)
    kernel = np.ones((int(pixels), int(pixels)), dtype=np.uint8)
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)


def feather_mask(mask: np.ndarray, pixels: int) -> np.ndarray:
    if pixels <= 0:
        return mask.astype(np.float32)
    k = max(1, int(pixels))
    if k % 2 == 0:
        k += 1
    return cv2.GaussianBlur(mask.astype(np.float32), (k, k), 0)


def masked_blur_frame(
    image: np.ndarray,
    mask: np.ndarray,
    dilate: int = 9,
    feather: int = 11,
) -> np.ndarray:
    mask = dilate_mask(mask, dilate)
    alpha = feather_mask(mask, feather)
    alpha = np.clip(alpha, 0.0, 1.0)[..., None]
    blur = cv2.GaussianBlur(image, (0, 0), sigmaX=14.0, sigmaY=14.0)
    out = image.astype(np.float32) * alpha + blur.astype(np.float32) * (1.0 - alpha)
    return np.clip(out, 0, 255).astype(np.uint8)


def load_mask(mask_path: str | None, image_size: tuple[int, int] = IMAGE_SIZE) -> np.ndarray | None:
    if not mask_path:
        return None
    if not Path(mask_path).exists():
        return None
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    width, height = image_size
    if mask.shape[:2] != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    return mask > 0


def draw_bbox(image: np.ndarray, bbox: Iterable[float], color: tuple[int, int, int], label: str) -> None:
    x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        image,
        label,
        (x1, max(20, y1 - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )
