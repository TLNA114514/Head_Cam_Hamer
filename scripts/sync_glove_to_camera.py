#!/usr/bin/env python3
"""Resample PN glove CSV frames onto camera group_id timestamps."""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import os
from pathlib import Path

import numpy as np

from convert_glove_csv_to_local import build_hand_joints, hand_record, parse_hands, read_header
from hamer_multiview_utils import DEFAULT_FRAMES
from progress_utils import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="+", type=Path, help="PN glove CSV file(s).")
    parser.add_argument("--frames", type=Path, default=DEFAULT_FRAMES)
    parser.add_argument("--output-dir", type=Path, default=Path("gloves/glove_local"))
    parser.add_argument("--output", type=Path, help="Single output JSONL path. Valid only with one input CSV.")
    parser.add_argument("--hands", default="Left,Right", help="Comma-separated hands to export: Left,Right.")
    parser.add_argument("--glove-fps", type=float, default=60.0)
    parser.add_argument("--camera-fps", type=float, default=25.0)
    parser.add_argument(
        "--anchor-glove-frame",
        type=float,
        required=True,
        help="Glove frame number aligned to --anchor-camera-frame. By default this is CSV Frame-No.",
    )
    parser.add_argument(
        "--anchor-glove-is-one-based",
        action="store_true",
        help="Treat --anchor-glove-frame as human ordinal counting and subtract 1 before syncing.",
    )
    parser.add_argument(
        "--anchor-camera-frame",
        type=float,
        required=True,
        help="Camera zero-based group_id/frame index aligned to --anchor-glove-frame.",
    )
    parser.add_argument("--group-range", help="Inclusive camera group range, e.g. 1-100.")
    parser.add_argument("--range", dest="range_alias", help="Short alias for --group-range.")
    parser.add_argument("--group-ids", help="Comma-separated camera group ids.")
    parser.add_argument("--scale", type=float, default=1.0, help="Scale factor applied to CSV joint positions.")
    parser.add_argument("--interpolation", choices=["linear", "nearest"], default="linear")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress-position", type=int, default=int(os.environ.get("TQDM_POSITION", "0")))
    return parser.parse_args()


def parse_group_range(value: str | None) -> tuple[int, int] | None:
    if not value:
        return None
    start_text, end_text = value.split("-", 1)
    start = int(start_text)
    end = int(end_text)
    if end < start:
        raise ValueError(f"Invalid group range: {value}")
    return start, end


def parse_group_ids(value: str | None) -> set[int] | None:
    if not value:
        return None
    return {int(item.strip()) for item in value.split(",") if item.strip()}


def load_camera_group_ids(path: Path, group_range: tuple[int, int] | None, selected_ids: set[int] | None) -> list[int]:
    group_ids: set[int] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            group_id = int(record["group_id"])
            if group_range and not (group_range[0] <= group_id <= group_range[1]):
                continue
            if selected_ids is not None and group_id not in selected_ids:
                continue
            group_ids.add(group_id)
    return sorted(group_ids)


def load_glove_samples(path: Path, hands: list[str], scale: float) -> tuple[list[int], dict[int, dict[str, np.ndarray]]]:
    declared, header = read_header(path)
    samples: dict[int, dict[str, np.ndarray]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        first = f.readline()
        if not first.startswith("Frame-No"):
            f.readline()
        reader = csv.DictReader(f, fieldnames=header)
        progress = tqdm(reader, total=declared, desc=f"load {path.stem}", unit="frame")
        for source_index, row in enumerate(progress):
            frame_no = int(float(row.get("Frame-No", source_index)))
            hand_samples = {}
            for handedness in hands:
                joints = build_hand_joints(row, handedness, scale)
                if joints is not None:
                    hand_samples[handedness] = joints
            if hand_samples:
                samples[frame_no] = hand_samples
    return sorted(samples), samples


def sample_hand(
    frame_numbers: list[int],
    samples: dict[int, dict[str, np.ndarray]],
    handedness: str,
    glove_frame_float: float,
    interpolation: str,
) -> tuple[np.ndarray | None, int | None, int | None, float | None]:
    if not frame_numbers or not math.isfinite(glove_frame_float):
        return None, None, None, None
    if glove_frame_float < frame_numbers[0] or glove_frame_float > frame_numbers[-1]:
        return None, None, None, None

    if interpolation == "nearest":
        index = bisect.bisect_left(frame_numbers, glove_frame_float)
        candidates = []
        if index < len(frame_numbers):
            candidates.append(frame_numbers[index])
        if index > 0:
            candidates.append(frame_numbers[index - 1])
        nearest = min(candidates, key=lambda frame_no: abs(frame_no - glove_frame_float))
        joints = samples.get(nearest, {}).get(handedness)
        return joints, nearest, nearest, 0.0 if joints is not None else None

    upper_index = bisect.bisect_left(frame_numbers, glove_frame_float)
    if upper_index < len(frame_numbers) and frame_numbers[upper_index] == glove_frame_float:
        frame_no = frame_numbers[upper_index]
        joints = samples.get(frame_no, {}).get(handedness)
        return joints, frame_no, frame_no, 0.0 if joints is not None else None
    if upper_index == 0 or upper_index >= len(frame_numbers):
        return None, None, None, None

    lower = frame_numbers[upper_index - 1]
    upper = frame_numbers[upper_index]
    lower_joints = samples.get(lower, {}).get(handedness)
    upper_joints = samples.get(upper, {}).get(handedness)
    if lower_joints is None or upper_joints is None:
        return None, lower, upper, None
    alpha = float((glove_frame_float - lower) / (upper - lower))
    joints = (1.0 - alpha) * lower_joints + alpha * upper_joints
    return joints, lower, upper, alpha


def default_output_path(output_dir: Path, csv_path: Path, anchor_glove_frame: float, anchor_camera_frame: float) -> Path:
    glove_text = f"{anchor_glove_frame:.3f}".replace(".", "p")
    camera_text = f"{anchor_camera_frame:.3f}".replace(".", "p")
    return output_dir / f"{csv_path.stem}_camera_sync_g{glove_text}_c{camera_text}.jsonl"


def sync_one(path: Path, output_path: Path, group_ids: list[int], hands: list[str], args: argparse.Namespace) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not args.overwrite:
        raise SystemExit(f"{output_path} exists; pass --overwrite")

    anchor_glove_frame = args.anchor_glove_frame - 1.0 if args.anchor_glove_is_one_based else args.anchor_glove_frame
    glove_per_camera = args.glove_fps / args.camera_fps
    frame_numbers, samples = load_glove_samples(path, hands, args.scale)

    written = 0
    skipped_groups = 0
    skipped_hands = 0
    progress = tqdm(group_ids, total=len(group_ids), desc=f"sync {path.stem}", unit="group", position=args.progress_position)
    with output_path.open("w", encoding="utf-8") as out:
        for group_id in progress:
            glove_frame_float = anchor_glove_frame + (float(group_id) - args.anchor_camera_frame) * glove_per_camera
            record_hands = []
            for handedness in hands:
                joints, lower, upper, alpha = sample_hand(
                    frame_numbers,
                    samples,
                    handedness,
                    glove_frame_float,
                    args.interpolation,
                )
                if joints is None:
                    skipped_hands += 1
                    continue
                hand = hand_record(handedness, joints, path)
                hand["mode"] = "glove_csv_camera_synced"
                hand["source"] = "glove_csv_camera_synced"
                hand["sync"] = {
                    "glove_frame_float": glove_frame_float,
                    "glove_frame_floor": lower,
                    "glove_frame_ceil": upper,
                    "alpha": alpha,
                    "interpolation": args.interpolation,
                }
                record_hands.append(hand)
            if not record_hands:
                skipped_groups += 1
                continue
            out.write(
                json.dumps(
                    {
                        "type": "glove_local_frame",
                        "group_id": group_id,
                        "camera_frame_index": group_id,
                        "glove_frame_float": glove_frame_float,
                        "source_csv": str(path),
                        "sync": {
                            "glove_fps": args.glove_fps,
                            "camera_fps": args.camera_fps,
                            "anchor_glove_frame_input": args.anchor_glove_frame,
                            "anchor_glove_frame_zero_based": anchor_glove_frame,
                            "anchor_glove_is_one_based": bool(args.anchor_glove_is_one_based),
                            "anchor_camera_frame": args.anchor_camera_frame,
                            "formula": "glove_frame=anchor_glove+(camera_group-anchor_camera)*glove_fps/camera_fps",
                        },
                        "hands": record_hands,
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )
            written += 1
    print(
        f"wrote {written} synced camera frames -> {output_path} "
        f"(skipped_groups={skipped_groups}, skipped_hands={skipped_hands})",
        flush=True,
    )
    return written


def main() -> None:
    args = parse_args()
    if args.range_alias and not args.group_range:
        args.group_range = args.range_alias
    if args.output and len(args.csv) != 1:
        raise SystemExit("--output can only be used with one input CSV")
    if args.glove_fps <= 0 or args.camera_fps <= 0:
        raise SystemExit("FPS values must be positive")

    hands = parse_hands(args.hands)
    group_ids = load_camera_group_ids(args.frames, parse_group_range(args.group_range), parse_group_ids(args.group_ids))
    if not group_ids:
        raise SystemExit("No camera group ids selected")

    anchor_glove_frame = args.anchor_glove_frame - 1.0 if args.anchor_glove_is_one_based else args.anchor_glove_frame
    print(
        "sync formula: "
        f"glove_frame = {anchor_glove_frame:.6f} + (camera_group - {args.anchor_camera_frame:.6f}) "
        f"* {args.glove_fps:.6f}/{args.camera_fps:.6f}",
        flush=True,
    )
    total = 0
    for csv_path in args.csv:
        output_path = args.output or default_output_path(args.output_dir, csv_path, anchor_glove_frame, args.anchor_camera_frame)
        total += sync_one(csv_path, output_path, group_ids, hands, args)
    print(f"total synced frames: {total}", flush=True)


if __name__ == "__main__":
    main()
