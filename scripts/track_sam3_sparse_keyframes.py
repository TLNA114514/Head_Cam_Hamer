#!/usr/bin/env python3
"""Propagate sparse SAM3 hand boxes with lightweight pyramidal optical flow."""

from __future__ import annotations

import argparse
import itertools
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from hamer_multiview_utils import (
    DEFAULT_BASE_DIR,
    DEFAULT_CAMERAS,
    DEFAULT_FRAMES,
    bbox_iou,
    filter_frame_records,
    iter_jsonl,
    parse_cameras,
    parse_group_ids,
    range_suffix,
    rectified_rel_path,
)
from progress_utils import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=Path, default=DEFAULT_FRAMES)
    parser.add_argument("--rectified-dir", type=Path, default=DEFAULT_BASE_DIR / "rectified_for_hamer")
    parser.add_argument("--keyframe-sam3", type=Path, required=True)
    parser.add_argument("--reference-sam3", type=Path, help="Optional dense SAM3 used only for offline evaluation.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_BASE_DIR / "sam3_sparse_tracks")
    parser.add_argument("--cameras", default=",".join(DEFAULT_CAMERAS))
    parser.add_argument("--group-range")
    parser.add_argument("--group-ids")
    parser.add_argument("--keyframe-stride", type=int, default=5)
    parser.add_argument("--flow-scale", type=float, default=0.25)
    parser.add_argument("--max-corners", type=int, default=60)
    parser.add_argument("--min-corners", type=int, default=8)
    parser.add_argument("--quality-level", type=float, default=0.01)
    parser.add_argument("--min-distance", type=float, default=3.0)
    parser.add_argument("--max-flow-error", type=float, default=25.0)
    parser.add_argument("--max-forward-backward-error", type=float, default=0.0)
    parser.add_argument("--max-hands", type=int, default=2)
    parser.add_argument("--evaluation-iou", type=float, default=0.5)
    parser.add_argument("--progress-position", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_hands(
    path: Path | None,
    cameras: set[str],
    group_ids: set[int] | None,
) -> dict[tuple[int, str], list[dict[str, Any]]]:
    result: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    if path is None or not path.exists():
        return result
    for record in iter_jsonl(path):
        group_id = int(record["group_id"])
        camera_id = str(record["camera_id"])
        if camera_id not in cameras or (group_ids is not None and group_id not in group_ids):
            continue
        for hand in record.get("hands") or []:
            bbox = hand.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            item = dict(hand)
            item["bbox"] = [float(value) for value in bbox]
            result[(group_id, camera_id)].append(item)
    return result


def scaled_gray(image: np.ndarray, scale: float) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if scale == 1.0:
        return gray
    return cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def clip_bbox(bbox: np.ndarray, width: int, height: int) -> np.ndarray | None:
    x1, y1, x2, y2 = [float(value) for value in bbox]
    x1 = float(np.clip(x1, 0.0, width - 2.0))
    y1 = float(np.clip(y1, 0.0, height - 2.0))
    x2 = float(np.clip(x2, x1 + 2.0, width - 1.0))
    y2 = float(np.clip(y2, y1 + 2.0, height - 1.0))
    if x2 - x1 < 4.0 or y2 - y1 < 4.0:
        return None
    return np.asarray([x1, y1, x2, y2], dtype=np.float32)


def bbox_corners(bbox: np.ndarray, scale: float) -> np.ndarray:
    x1, y1, x2, y2 = bbox * scale
    return np.asarray([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)


def detect_points(gray: np.ndarray, bbox: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    height, width = gray.shape[:2]
    scaled = bbox * args.flow_scale
    x1 = int(np.clip(np.floor(scaled[0]), 0, width - 1))
    y1 = int(np.clip(np.floor(scaled[1]), 0, height - 1))
    x2 = int(np.clip(np.ceil(scaled[2]), x1 + 1, width))
    y2 = int(np.clip(np.ceil(scaled[3]), y1 + 1, height))
    mask = np.zeros_like(gray, dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255
    points = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=args.max_corners,
        qualityLevel=args.quality_level,
        minDistance=args.min_distance,
        mask=mask,
        blockSize=5,
    )
    if points is None:
        return np.empty((0, 2), dtype=np.float32)
    return points.reshape(-1, 2).astype(np.float32)


def transform_bbox(
    bbox: np.ndarray,
    previous_points: np.ndarray,
    current_points: np.ndarray,
    args: argparse.Namespace,
    width: int,
    height: int,
) -> np.ndarray | None:
    matrix = None
    if len(previous_points) >= 4:
        matrix, _inliers = cv2.estimateAffinePartial2D(
            previous_points,
            current_points,
            method=cv2.RANSAC,
            ransacReprojThreshold=2.0,
            maxIters=100,
            confidence=0.95,
        )
    if matrix is None:
        if not len(previous_points):
            return None
        displacement = np.median(current_points - previous_points, axis=0)
        matrix = np.asarray([[1.0, 0.0, displacement[0]], [0.0, 1.0, displacement[1]]], dtype=np.float32)
    linear = matrix[:, :2]
    scale = float(np.sqrt(abs(np.linalg.det(linear))))
    if not np.isfinite(scale) or not 0.75 <= scale <= 1.35:
        displacement = np.median(current_points - previous_points, axis=0)
        matrix = np.asarray([[1.0, 0.0, displacement[0]], [0.0, 1.0, displacement[1]]], dtype=np.float32)
    corners = bbox_corners(bbox, args.flow_scale)
    transformed = cv2.transform(corners[None, :, :], matrix)[0] / args.flow_scale
    output = np.asarray(
        [transformed[:, 0].min(), transformed[:, 1].min(), transformed[:, 0].max(), transformed[:, 1].max()],
        dtype=np.float32,
    )
    return clip_bbox(output, width, height)


def propagate_state(
    state: dict[str, Any],
    previous_gray: np.ndarray,
    current_gray: np.ndarray,
    width: int,
    height: int,
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    points = state.get("points")
    if points is None or len(points) < args.min_corners:
        points = detect_points(previous_gray, state["bbox"], args)
    if len(points) < 2:
        return None
    next_points, status, errors = cv2.calcOpticalFlowPyrLK(
        previous_gray,
        current_gray,
        points.reshape(-1, 1, 2),
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01),
    )
    if next_points is None or status is None:
        return None
    valid = status.reshape(-1).astype(bool)
    if errors is not None:
        valid &= errors.reshape(-1) <= args.max_flow_error
    forward_backward_error = None
    if args.max_forward_backward_error > 0.0:
        backward_points, backward_status, _backward_errors = cv2.calcOpticalFlowPyrLK(
            current_gray,
            previous_gray,
            next_points,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01),
        )
        if backward_points is None or backward_status is None:
            return None
        valid &= backward_status.reshape(-1).astype(bool)
        forward_backward_error = np.linalg.norm(backward_points.reshape(-1, 2) - points, axis=1)
        valid &= forward_backward_error <= args.max_forward_backward_error
    previous_valid = points[valid]
    current_valid = next_points.reshape(-1, 2)[valid]
    if len(previous_valid) < 2:
        return None
    bbox = transform_bbox(state["bbox"], previous_valid, current_valid, args, width, height)
    if bbox is None:
        return None
    refreshed = detect_points(current_gray, bbox, args)
    output = dict(state)
    output["bbox"] = bbox
    output["points"] = refreshed if len(refreshed) >= args.min_corners else current_valid
    output["track_age"] = int(state.get("track_age", 0)) + 1
    output["score"] = float(state.get("score", 1.0)) * 0.98
    output["flow_point_count"] = int(len(previous_valid))
    output["flow_point_ratio"] = float(len(previous_valid) / len(points))
    output["flow_forward_backward_mean"] = (
        float(np.mean(forward_backward_error[valid])) if forward_backward_error is not None else None
    )
    return output


def match_refresh(
    detections: list[dict[str, Any]],
    states: dict[int, dict[str, Any]],
    gray: np.ndarray,
    args: argparse.Namespace,
    next_track_id: int,
) -> tuple[dict[int, dict[str, Any]], int]:
    state_ids = list(states)
    candidates = sorted(
        (
            (bbox_iou(states[state_id]["bbox"].tolist(), detection["bbox"]), state_id, detection_index)
            for state_id in state_ids
            for detection_index, detection in enumerate(detections)
        ),
        reverse=True,
    )
    assigned_states: set[int] = set()
    assigned_detections: set[int] = set()
    detection_to_track: dict[int, int] = {}
    for iou, state_id, detection_index in candidates:
        if iou < 0.05 or state_id in assigned_states or detection_index in assigned_detections:
            continue
        assigned_states.add(state_id)
        assigned_detections.add(detection_index)
        detection_to_track[detection_index] = state_id
    refreshed = {}
    for detection_index, detection in enumerate(detections[: args.max_hands]):
        track_id = detection_to_track.get(detection_index)
        if track_id is None:
            track_id = next_track_id
            next_track_id += 1
        bbox = np.asarray(detection["bbox"], dtype=np.float32)
        state = dict(detection)
        state.update(
            {
                "bbox": bbox,
                "points": detect_points(gray, bbox, args),
                "track_id": track_id,
                "track_age": 0,
                "keyframe_score": float(detection.get("score", 1.0) or 1.0),
                "score": float(detection.get("score", 1.0) or 1.0),
            }
        )
        refreshed[track_id] = state
    return refreshed, next_track_id


def state_to_hand(state: dict[str, Any], keyframe: bool, group_id: int, camera_id: str) -> dict[str, Any]:
    ignored = {"points", "bbox", "mask_path"}
    hand = {key: value for key, value in state.items() if key not in ignored and not key.startswith("_")}
    hand.update(
        {
            "bbox": state["bbox"].tolist(),
            "track_id": f"{camera_id}_s{int(state['track_id']):02d}",
            "track_numeric_id": int(state["track_id"]),
            "group_id": group_id,
            "camera_id": camera_id,
            "track_backend": "sam3_keyframe" if keyframe else "opencv_lk_sparse_sam3",
            "is_keyframe": keyframe,
            "mask_path": state.get("mask_path") if keyframe else None,
        }
    )
    return hand


def best_iou_matching(predicted: list[dict[str, Any]], reference: list[dict[str, Any]]) -> list[float]:
    if not predicted or not reference:
        return []
    best_pairs: list[tuple[float, int, int]] = []
    count = min(len(predicted), len(reference))
    for predicted_indices in itertools.permutations(range(len(predicted)), count):
        for reference_indices in itertools.permutations(range(len(reference)), count):
            values = [
                bbox_iou(predicted[predicted_index]["bbox"], reference[reference_index]["bbox"])
                for predicted_index, reference_index in zip(predicted_indices, reference_indices)
            ]
            score = sum(values)
            if not best_pairs or score > best_pairs[0][0]:
                best_pairs = [(score, predicted_indices, reference_indices)]
    if not best_pairs:
        return []
    _score, predicted_indices, reference_indices = best_pairs[0]
    return [
        bbox_iou(predicted[predicted_index]["bbox"], reference[reference_index]["bbox"])
        for predicted_index, reference_index in zip(predicted_indices, reference_indices)
    ]


def evaluate_records(
    records: list[dict[str, Any]],
    reference: dict[tuple[int, str], list[dict[str, Any]]],
    threshold: float,
) -> dict[str, Any]:
    ious = []
    reference_hands = 0
    matched_hands = 0
    recalled_hands = 0
    nonkey_ious = []
    nonkey_reference_hands = 0
    nonkey_recalled_hands = 0
    for record in records:
        key = (int(record["group_id"]), str(record["camera_id"]))
        reference_items = reference.get(key, [])
        values = best_iou_matching(record.get("hands") or [], reference_items)
        reference_hands += len(reference_items)
        matched_hands += len(values)
        recalled_hands += sum(value >= threshold for value in values)
        ious.extend(values)
        if not record.get("is_keyframe"):
            nonkey_reference_hands += len(reference_items)
            nonkey_recalled_hands += sum(value >= threshold for value in values)
            nonkey_ious.extend(values)
    return {
        "reference_hands": reference_hands,
        "matched_hands": matched_hands,
        "recall_at_iou": recalled_hands / reference_hands if reference_hands else None,
        "mean_iou": float(np.mean(ious)) if ious else None,
        "median_iou": float(np.median(ious)) if ious else None,
        "p10_iou": float(np.percentile(ious, 10)) if ious else None,
        "nonkey_reference_hands": nonkey_reference_hands,
        "nonkey_recall_at_iou": nonkey_recalled_hands / nonkey_reference_hands if nonkey_reference_hands else None,
        "nonkey_mean_iou": float(np.mean(nonkey_ious)) if nonkey_ious else None,
        "nonkey_median_iou": float(np.median(nonkey_ious)) if nonkey_ious else None,
        "iou_threshold": threshold,
    }


def main() -> None:
    args = parse_args()
    if args.keyframe_stride < 1 or not 0.0 < args.flow_scale <= 1.0:
        raise SystemExit("--keyframe-stride must be positive and --flow-scale must be in (0, 1]")
    if args.min_corners < 2 or args.max_corners < args.min_corners:
        raise SystemExit("corner limits are invalid")
    if args.max_flow_error <= 0.0 or args.max_forward_backward_error < 0.0:
        raise SystemExit("max flow error must be positive and forward-backward error must be non-negative")
    cameras = parse_cameras(args.cameras)
    group_ids = parse_group_ids(args.group_range, args.group_ids)
    suffix = range_suffix(group_ids)
    frame_records = filter_frame_records(args.frames, cameras, group_ids)
    by_camera: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in frame_records:
        by_camera[str(record["camera_id"])].append(record)
    for records in by_camera.values():
        records.sort(key=lambda record: int(record["group_id"]))
    keyframe_hands = load_hands(args.keyframe_sam3, cameras, group_ids)
    reference_path = args.reference_sam3 or args.keyframe_sam3
    reference_hands = load_hands(reference_path, cameras, group_ids)
    output_path = args.output_dir / f"sam3_sparse_tracks_{suffix}.jsonl"
    config_path = args.output_dir / f"sam3_sparse_tracks_config_{suffix}.json"
    if args.dry_run:
        print(
            json.dumps(
                {
                    "records": len(frame_records),
                    "cameras": sorted(cameras),
                    "keyframe_stride": args.keyframe_stride,
                    "output": str(output_path),
                },
                indent=2,
            )
        )
        return
    if output_path.exists() and not args.overwrite:
        raise SystemExit(f"{output_path} exists; pass --overwrite to replace it")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    output_records = []
    stats = {"records": 0, "keyframes": 0, "tracked_frames": 0, "missing_images": 0, "tracking_failures": 0}
    started = time.perf_counter()
    for camera_index, camera_id in enumerate(sorted(by_camera)):
        records = by_camera[camera_id]
        states: dict[int, dict[str, Any]] = {}
        next_track_id = 1
        previous_gray = None
        progress = tqdm(
            records,
            desc=f"sparse SAM3 {camera_id}",
            unit="frame",
            position=args.progress_position + camera_index,
        )
        for sequence_index, frame_record in enumerate(progress):
            group_id = int(frame_record["group_id"])
            image_path = args.rectified_dir / rectified_rel_path(frame_record)
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                stats["missing_images"] += 1
                continue
            gray = scaled_gray(image, args.flow_scale)
            height, width = image.shape[:2]
            scheduled_keyframe = sequence_index % args.keyframe_stride == 0 or previous_gray is None
            detections = keyframe_hands.get((group_id, camera_id), []) if scheduled_keyframe else []
            is_keyframe = bool(scheduled_keyframe and detections)
            if is_keyframe:
                states, next_track_id = match_refresh(detections, states, gray, args, next_track_id)
                stats["keyframes"] += 1
            elif previous_gray is not None:
                propagated = {}
                for track_id, state in states.items():
                    updated = propagate_state(state, previous_gray, gray, width, height, args)
                    if updated is None:
                        stats["tracking_failures"] += 1
                        continue
                    propagated[track_id] = updated
                states = propagated
                stats["tracked_frames"] += 1
            hands = [state_to_hand(state, is_keyframe, group_id, camera_id) for state in states.values()]
            hands.sort(key=lambda hand: int(hand["track_numeric_id"]))
            hands = hands[: args.max_hands]
            output_records.append(
                {
                    "type": "sam3_sparse_tracks",
                    "group_id": group_id,
                    "camera_id": camera_id,
                    "rectified_image_path": str(image_path),
                    "is_keyframe": is_keyframe,
                    "hands": hands,
                }
            )
            stats["records"] += 1
            previous_gray = gray
        progress.close()
    elapsed = time.perf_counter() - started
    output_records.sort(key=lambda record: (int(record["group_id"]), str(record["camera_id"])))
    with output_path.open("w", encoding="utf-8") as output_file:
        for record in output_records:
            output_file.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    evaluation = evaluate_records(output_records, reference_hands, args.evaluation_iou)
    timing = {
        "tracking_seconds": elapsed,
        "camera_images_per_second": stats["records"] / elapsed if elapsed else None,
        "multiview_frames_per_second": stats["records"] / max(1, len(cameras)) / elapsed if elapsed else None,
        "excludes_sam3_keyframe_inference": True,
    }
    with config_path.open("w", encoding="utf-8") as config_file:
        json.dump(
            {
                "frames": str(args.frames),
                "rectified_dir": str(args.rectified_dir),
                "keyframe_sam3": str(args.keyframe_sam3),
                "reference_sam3": str(reference_path),
                "output": str(output_path),
                "cameras": sorted(cameras),
                "keyframe_stride": args.keyframe_stride,
                "flow_scale": args.flow_scale,
                "max_flow_error": args.max_flow_error,
                "max_forward_backward_error": args.max_forward_backward_error,
                "stats": stats,
                "timing": timing,
                "evaluation": evaluation,
            },
            config_file,
            ensure_ascii=False,
            indent=2,
        )
        config_file.write("\n")
    print(json.dumps({"stats": stats, "timing": timing, "evaluation": evaluation}, ensure_ascii=False, indent=2))
    print(f"wrote: {output_path}")


if __name__ == "__main__":
    main()
