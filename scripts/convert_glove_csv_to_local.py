#!/usr/bin/env python3
"""Convert PN glove CSV exports into hand-local JSONL for the existing viewer."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Iterable

import numpy as np

from progress_utils import tqdm


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

FINGER_PREFIX = {
    "Left": {
        "wrist": "LeftHand",
        "thumb": ["LeftHandThumb1", "LeftHandThumb2", "LeftHandThumb3"],
        "index": ["LeftInHandIndex", "LeftHandIndex1", "LeftHandIndex2", "LeftHandIndex3"],
        "middle": ["LeftInHandMiddle", "LeftHandMiddle1", "LeftHandMiddle2", "LeftHandMiddle3"],
        "ring": ["LeftInHandRing", "LeftHandRing1", "LeftHandRing2", "LeftHandRing3"],
        "pinky": ["LeftInHandPinky", "LeftHandPinky1", "LeftHandPinky2", "LeftHandPinky3"],
    },
    "Right": {
        "wrist": "RightHand",
        "thumb": ["RightHandThumb1", "RightHandThumb2", "RightHandThumb3"],
        "index": ["RightInHandIndex", "RightHandIndex1", "RightHandIndex2", "RightHandIndex3"],
        "middle": ["RightInHandMiddle", "RightHandMiddle1", "RightHandMiddle2", "RightHandMiddle3"],
        "ring": ["RightInHandRing", "RightHandRing1", "RightHandRing2", "RightHandRing3"],
        "pinky": ["RightInHandPinky", "RightHandPinky1", "RightHandPinky2", "RightHandPinky3"],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="+", type=Path, help="PN glove CSV file(s).")
    parser.add_argument("--output-dir", type=Path, default=Path("gloves/glove_local"))
    parser.add_argument("--output", type=Path, help="Single output JSONL path. Valid only with one input CSV.")
    parser.add_argument("--hands", default="Left,Right", help="Comma-separated hands to export: Left,Right.")
    parser.add_argument("--group-offset", type=int, default=1, help="group_id = Frame-No + group_offset.")
    parser.add_argument("--scale", type=float, default=1.0, help="Scale factor applied to CSV joint positions.")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress-position", type=int, default=int(os.environ.get("TQDM_POSITION", "0")))
    return parser.parse_args()


def parse_hands(value: str) -> list[str]:
    hands = [item.strip() for item in value.split(",") if item.strip()]
    for hand in hands:
        if hand not in {"Left", "Right"}:
            raise ValueError(f"Invalid hand: {hand}")
    return hands


def read_header(path: Path) -> tuple[int | None, list[str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        first = f.readline().strip()
        declared = int(first) if first.isdigit() else None
        if first.startswith("Frame-No"):
            return None, next(csv.reader([first]))
        return declared, next(csv.reader(f))


def finite_point(values: Iterable[float]) -> bool:
    return all(math.isfinite(float(value)) for value in values)


def joint_position(row: dict[str, str], joint: str, scale: float) -> np.ndarray | None:
    keys = [f"{joint}-Joint-Posi-x", f"{joint}-Joint-Posi-y", f"{joint}-Joint-Posi-z"]
    try:
        values = [float(row[key]) * scale for key in keys]
    except (KeyError, ValueError):
        return None
    if not finite_point(values):
        return None
    return np.asarray(values, dtype=np.float64)


def extrapolate_tip(points: list[np.ndarray]) -> np.ndarray:
    if len(points) >= 2:
        return points[-1] + (points[-1] - points[-2])
    return points[-1].copy()


def build_hand_joints(row: dict[str, str], handedness: str, scale: float) -> np.ndarray | None:
    spec = FINGER_PREFIX[handedness]
    wrist = joint_position(row, spec["wrist"], scale)
    if wrist is None:
        return None

    thumb = [joint_position(row, name, scale) for name in spec["thumb"]]
    index = [joint_position(row, name, scale) for name in spec["index"]]
    middle = [joint_position(row, name, scale) for name in spec["middle"]]
    ring = [joint_position(row, name, scale) for name in spec["ring"]]
    pinky = [joint_position(row, name, scale) for name in spec["pinky"]]
    if any(point is None for point in thumb + index + middle + ring + pinky):
        return None

    thumb_points = [point for point in thumb if point is not None]
    joints = [
        wrist,
        thumb_points[0],
        thumb_points[1],
        thumb_points[2],
        extrapolate_tip(thumb_points),
    ]
    for finger in (index, middle, ring, pinky):
        joints.extend(point for point in finger if point is not None)
    if len(joints) != 21:
        return None
    return np.stack(joints, axis=0)


def palm_frame_np(joints: np.ndarray) -> np.ndarray:
    wrist = joints[0]
    x_axis = joints[5] - wrist
    y_hint = joints[17] - wrist
    x_axis = x_axis / max(np.linalg.norm(x_axis), 1e-8)
    z_axis = np.cross(x_axis, y_hint)
    z_axis = z_axis / max(np.linalg.norm(z_axis), 1e-8)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / max(np.linalg.norm(y_axis), 1e-8)
    return np.stack([x_axis, y_axis, z_axis], axis=1)


def palm_local_np(points: np.ndarray) -> np.ndarray:
    basis = palm_frame_np(points)
    return (points - points[0:1]) @ basis


def hand_record(handedness: str, joints_world: np.ndarray, source_csv: Path) -> dict:
    local = joints_world - joints_world[0:1]
    palm = palm_local_np(joints_world)
    joints = []
    for index, name in enumerate(LANDMARK_NAMES):
        joints.append(
            {
                "joint_index": index,
                "index": index,
                "joint_name": name,
                "name": name,
                "valid": True,
                "metric_valid": True,
                "reconstruction_mode": "glove_csv",
                "position_world_m": joints_world[index].tolist(),
                "root_relative_headset_m": local[index].tolist(),
                "palm_local_m": palm[index].tolist(),
                "source_cameras": [],
                "rejected_cameras": [],
            }
        )
    return {
        "handedness": handedness,
        "mode": "glove_csv_palm_local",
        "source": "glove_csv",
        "source_csv": str(source_csv),
        "metric_valid": True,
        "local_joints_m": local.tolist(),
        "palm_local_joints_m": palm.tolist(),
        "joints": joints,
        "metric_joint_count": len(joints),
        "temporal_fallback_joint_count": 0,
    }


def convert_one(path: Path, output_path: Path, hands: list[str], args: argparse.Namespace) -> int:
    declared, header = read_header(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not args.overwrite:
        raise SystemExit(f"{output_path} exists; pass --overwrite")

    count = 0
    skipped = 0
    with path.open("r", encoding="utf-8", newline="") as f, output_path.open("w", encoding="utf-8") as out:
        first = f.readline()
        if not first.startswith("Frame-No"):
            f.readline()
        reader = csv.DictReader(f, fieldnames=header)
        total = declared
        progress = tqdm(reader, total=total, desc=f"glove {path.stem}", unit="frame", position=args.progress_position)
        for source_index, row in enumerate(progress):
            if args.stride > 1 and source_index % args.stride != 0:
                continue
            frame_no = int(float(row.get("Frame-No", source_index)))
            record_hands = []
            for handedness in hands:
                joints = build_hand_joints(row, handedness, args.scale)
                if joints is None:
                    skipped += 1
                    continue
                record_hands.append(hand_record(handedness, joints, path))
            if not record_hands:
                continue
            out.write(
                json.dumps(
                    {
                        "type": "glove_local_frame",
                        "group_id": frame_no + args.group_offset,
                        "glove_frame_no": frame_no,
                        "source_csv": str(path),
                        "hands": record_hands,
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )
            count += 1
            if args.max_frames is not None and count >= args.max_frames:
                break
    print(f"wrote {count} frames -> {output_path} (skipped_hands={skipped})")
    return count


def main() -> None:
    args = parse_args()
    hands = parse_hands(args.hands)
    if args.output and len(args.csv) != 1:
        raise SystemExit("--output can only be used with one input CSV")
    total = 0
    for csv_path in args.csv:
        output_path = args.output or (args.output_dir / f"{csv_path.stem}_glove_local.jsonl")
        total += convert_one(csv_path, output_path, hands, args)
    print(f"total frames: {total}")


if __name__ == "__main__":
    main()
