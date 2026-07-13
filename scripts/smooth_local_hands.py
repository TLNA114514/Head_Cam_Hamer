#!/usr/bin/env python3
"""Robust temporal smoothing for hand-local HaMeR/MANO JSONL outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from evaluate_hamer_vs_glove import hand_positions
from hamer_multiview_utils import iter_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--space",
        choices=["palm-local", "glove-calibrated-palm-local"],
        default="palm-local",
        help="Coordinate field to smooth. Calibrated output is written back to calibrated fields.",
    )
    parser.add_argument("--window-radius", type=int, default=2)
    parser.add_argument("--outlier-threshold-m", type=float, default=0.055)
    parser.add_argument("--mad-scale", type=float, default=4.0)
    parser.add_argument("--ema-alpha", type=float, default=0.72)
    parser.add_argument("--blend", type=float, default=0.55)
    parser.add_argument("--hands", default="Left,Right")
    parser.add_argument("--no-smooth-vertices", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def as_array(value: Any, shape_tail: tuple[int, ...]) -> np.ndarray | None:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim < len(shape_tail) or tuple(arr.shape[-len(shape_tail):]) != shape_tail:
        return None
    if not np.all(np.isfinite(arr)):
        return None
    return arr


def robust_hampel(sequence: np.ndarray, radius: int, threshold: float, mad_scale: float) -> tuple[np.ndarray, int]:
    out = sequence.copy()
    replacements = 0
    n = len(sequence)
    for index in range(n):
        start = max(0, index - radius)
        end = min(n, index + radius + 1)
        window = sequence[start:end]
        median = np.median(window, axis=0)
        distances = np.linalg.norm(window - median.reshape(1, *median.shape), axis=-1)
        current_dist = np.linalg.norm(sequence[index] - median, axis=-1)
        mad = np.median(np.abs(distances - np.median(distances, axis=0, keepdims=True)), axis=0)
        joint_threshold = np.maximum(threshold, mad_scale * 1.4826 * mad)
        replace = current_dist > joint_threshold
        if np.any(replace):
            out[index, replace] = median[replace]
            replacements += int(np.sum(replace))
    return out, replacements


def bidirectional_ema(sequence: np.ndarray, alpha: float) -> np.ndarray:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    forward = sequence.copy()
    for index in range(1, len(sequence)):
        forward[index] = alpha * forward[index] + (1.0 - alpha) * forward[index - 1]
    backward = sequence.copy()
    for index in range(len(sequence) - 2, -1, -1):
        backward[index] = alpha * backward[index] + (1.0 - alpha) * backward[index + 1]
    return 0.5 * (forward + backward)


def smooth_sequence(sequence: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, dict[str, Any]]:
    filtered, replacements = robust_hampel(
        sequence,
        max(0, int(args.window_radius)),
        float(args.outlier_threshold_m),
        float(args.mad_scale),
    )
    ema = bidirectional_ema(filtered, float(args.ema_alpha))
    blend = float(np.clip(args.blend, 0.0, 1.0))
    smoothed = (1.0 - blend) * filtered + blend * ema
    delta = np.linalg.norm(smoothed - sequence, axis=-1)
    return smoothed, {
        "hampel_replacements": replacements,
        "mean_delta_m": float(np.mean(delta)),
        "p95_delta_m": float(np.percentile(delta, 95)),
        "max_delta_m": float(np.max(delta)),
    }


def update_joints_list(hand: dict[str, Any], palm: np.ndarray, local: np.ndarray | None, joint_key: str) -> None:
    joints = hand.get("joints")
    if not isinstance(joints, list):
        return
    for index, joint in enumerate(joints):
        if not isinstance(joint, dict) or index >= len(palm):
            continue
        joint[joint_key] = palm[index].tolist()
        if local is not None and index < len(local):
            joint["position"] = local[index].tolist()
            joint["root_relative_headset_m"] = local[index].tolist()
        joint["temporal_smoothed"] = True


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"{args.output} exists; pass --overwrite to replace it")
    handedness_set = {item.strip() for item in args.hands.split(",") if item.strip()}
    source_key = "glove_calibrated_palm_local_joints_m" if args.space == "glove-calibrated-palm-local" else "palm_local_joints_m"
    target_joint_key = "glove_calibrated_palm_local_m" if args.space == "glove-calibrated-palm-local" else "palm_local_m"
    frames = list(iter_jsonl(args.input))
    hand_indices: dict[str, list[tuple[int, int]]] = {hand: [] for hand in handedness_set}
    for frame_index, frame in enumerate(frames):
        for hand_index, hand in enumerate(frame.get("hands") or []):
            handedness = hand.get("handedness")
            if handedness in hand_indices:
                palm = hand_positions(hand, args.space)
                local = as_array(hand.get("local_joints_m"), (3,))
                if palm is not None and palm.shape == (21, 3) and (
                    args.space == "glove-calibrated-palm-local" or (local is not None and local.shape == (21, 3))
                ):
                    hand_indices[handedness].append((frame_index, hand_index))

    stats = {
        "input": str(args.input),
        "output": str(args.output),
        "frames": len(frames),
        "hands": {},
        "args": vars(args) | {"input": str(args.input), "output": str(args.output)},
    }
    for handedness, positions in hand_indices.items():
        if not positions:
            continue
        palm_seq = np.stack(
            [hand_positions(frames[frame_index]["hands"][hand_index], args.space) for frame_index, hand_index in positions],
            axis=0,
        )
        palm_smoothed, palm_stats = smooth_sequence(palm_seq, args)
        local_smoothed: np.ndarray | None = None
        local_stats: dict[str, Any] | None = None
        if args.space == "palm-local":
            local_seq = np.stack(
                [as_array(frames[frame_index]["hands"][hand_index].get("local_joints_m"), (3,)) for frame_index, hand_index in positions],
                axis=0,
            )
            local_smoothed, local_stats = smooth_sequence(local_seq, args)
        for seq_index, (frame_index, hand_index) in enumerate(positions):
            hand = frames[frame_index]["hands"][hand_index]
            hand[source_key] = palm_smoothed[seq_index].tolist()
            if local_smoothed is not None:
                hand["local_joints_m"] = local_smoothed[seq_index].tolist()
            hand["mode"] = f"{hand.get('mode', 'hand_local')}_temporal_smoothed"
            hand["temporal_smoothing"] = {
                "window_radius": int(args.window_radius),
                "outlier_threshold_m": float(args.outlier_threshold_m),
                "mad_scale": float(args.mad_scale),
                "ema_alpha": float(args.ema_alpha),
                "blend": float(args.blend),
            }
            update_joints_list(
                hand,
                palm_smoothed[seq_index],
                local_smoothed[seq_index] if local_smoothed is not None else None,
                target_joint_key,
            )
        stats["hands"][handedness] = {
            "count": len(positions),
            "palm": palm_stats,
            "local": local_stats,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for frame in frames:
            f.write(json.dumps(frame, ensure_ascii=False, separators=(",", ":")) + "\n")
    stats_path = args.output.with_suffix(".stats.json")
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"wrote: {args.output}")
    print(f"stats: {stats_path}")
    print(json.dumps(stats["hands"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
