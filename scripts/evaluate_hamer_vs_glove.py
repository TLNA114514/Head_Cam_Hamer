#!/usr/bin/env python3
"""Evaluate hand-local HaMeR/MANO output against synchronized PN glove joints."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from convert_glove_csv_to_local import LANDMARK_NAMES, parse_hands


FINGER_JOINTS = {
    "thumb": [4, 3, 2],
    "index": [8, 7, 6],
    "middle": [12, 11, 10],
    "ring": [16, 15, 14],
    "pinky": [20, 19, 18],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hamer", type=Path, required=True, help="HaMeR local JSONL.")
    parser.add_argument("--compare-hamer", action="append", type=Path, help="Additional HaMeR JSONL to evaluate against the same glove GT.")
    parser.add_argument("--glove", type=Path, required=True, help="Camera-synced glove local JSONL.")
    parser.add_argument(
        "--space",
        choices=[
            "palm-local",
            "raw-palm-local",
            "zero-shot-static-palm-local",
            "smoothed-palm-local",
            "causal-smoothed-palm-local",
            "adaptive-causal-palm-local",
            "root-relative",
            "glove-calibrated-palm-local",
            "glove-calibrated-root-relative",
        ],
        default="palm-local",
    )
    parser.add_argument("--hands", default="Left,Right")
    parser.add_argument("--fingers", default="thumb,index,middle")
    parser.add_argument("--group-range", help="Inclusive range, e.g. 1-100.")
    parser.add_argument("--group-ids", help="Comma-separated group ids.")
    parser.add_argument("--group-parity", choices=["all", "even", "odd"], default="all")
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument(
        "--require-hamer-metric-valid",
        action="store_true",
        help="Skip HaMeR hands whose hand-level metric_valid is false.",
    )
    return parser.parse_args()


def parse_group_filter(group_range: str | None, group_ids: str | None) -> set[int] | None:
    selected: set[int] = set()
    if group_range:
        for part in group_range.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                start_text, end_text = part.split("-", 1)
                selected.update(range(int(start_text), int(end_text) + 1))
            else:
                selected.add(int(part))
    if group_ids:
        selected.update(int(item.strip()) for item in group_ids.split(",") if item.strip())
    return selected or None


def apply_group_parity(group_filter: set[int] | None, parity: str) -> set[int] | None:
    if parity == "all":
        return group_filter
    if group_filter is None:
        raise SystemExit("--group-parity requires --group-range or --group-ids")
    wanted = 0 if parity == "even" else 1
    return {group_id for group_id in group_filter if group_id % 2 == wanted}


def parse_fingers(value: str) -> list[str]:
    fingers = [item.strip() for item in value.split(",") if item.strip()]
    invalid = [finger for finger in fingers if finger not in FINGER_JOINTS]
    if invalid:
        raise SystemExit(f"Invalid fingers: {invalid}. Valid: {sorted(FINGER_JOINTS)}")
    return fingers


def finite_xyz(value: Any) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) == 3
        and all(isinstance(item, (int, float)) and math.isfinite(float(item)) for item in value)
    )


def hand_positions(hand: dict[str, Any], space: str, allow_specialized_fallback: bool = False) -> np.ndarray | None:
    if space == "palm-local":
        direct_keys = ["palm_local_joints_m"]
        joint_keys = ["palm_local_m"]
    elif space == "raw-palm-local":
        direct_keys = ["raw_palm_local_joints_m"]
        joint_keys = ["raw_palm_local_m"]
    elif space == "zero-shot-static-palm-local":
        direct_keys = ["static_calibrated_palm_local_joints_m"]
        joint_keys = ["static_calibrated_palm_local_m"]
    elif space == "smoothed-palm-local":
        direct_keys = ["smoothed_palm_local_joints_m"]
        joint_keys = ["smoothed_palm_local_m"]
    elif space == "causal-smoothed-palm-local":
        direct_keys = ["causal_smoothed_palm_local_joints_m"]
        joint_keys = ["causal_smoothed_palm_local_m"]
    elif space == "adaptive-causal-palm-local":
        direct_keys = ["adaptive_causal_palm_local_joints_m"]
        joint_keys = ["adaptive_causal_palm_local_m"]
    elif space == "root-relative":
        direct_keys = ["local_joints_m"]
        joint_keys = ["root_relative_headset_m"]
    elif space == "glove-calibrated-palm-local":
        direct_keys = ["glove_calibrated_palm_local_joints_m", "palm_local_joints_m"]
        joint_keys = ["glove_calibrated_palm_local_m", "palm_local_m"]
    elif space == "glove-calibrated-root-relative":
        direct_keys = ["glove_calibrated_root_relative_joints_m", "local_joints_m"]
        joint_keys = ["glove_calibrated_root_relative_m", "root_relative_headset_m"]
    else:
        raise ValueError(f"Unsupported space: {space}")

    if allow_specialized_fallback and space in {
        "raw-palm-local",
        "zero-shot-static-palm-local",
        "smoothed-palm-local",
        "causal-smoothed-palm-local",
        "adaptive-causal-palm-local",
    }:
        direct_keys.append("palm_local_joints_m")
        joint_keys.append("palm_local_m")

    for direct_key in direct_keys:
        direct = hand.get(direct_key)
        if isinstance(direct, list) and len(direct) >= 21 and all(finite_xyz(item) for item in direct[:21]):
            return np.asarray(direct[:21], dtype=np.float64)

    joints = hand.get("joints") or []
    out = np.full((21, 3), np.nan, dtype=np.float64)
    found = 0
    for joint in joints:
        index = joint.get("index", joint.get("joint_index"))
        position = None
        for joint_key in joint_keys:
            candidate = joint.get(joint_key)
            if finite_xyz(candidate):
                position = candidate
                break
        if isinstance(index, int) and 0 <= index < 21 and finite_xyz(position):
            out[index] = np.asarray(position, dtype=np.float64)
            found += 1
    if found >= 21:
        return out
    return None


def load_hands_by_group(
    path: Path,
    space: str,
    group_filter: set[int] | None,
    allow_specialized_fallback: bool = False,
) -> dict[int, dict[str, dict[str, Any]]]:
    frames: dict[int, dict[str, dict[str, Any]]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if "group_id" not in record:
                continue
            group_id = int(record["group_id"])
            if group_filter is not None and group_id not in group_filter:
                continue
            hands = {}
            for hand in record.get("hands", []):
                handedness = hand.get("handedness")
                if handedness not in {"Left", "Right"}:
                    continue
                positions = hand_positions(hand, space, allow_specialized_fallback)
                if positions is None:
                    continue
                item = dict(hand)
                item["_positions"] = positions
                hands[handedness] = item
            if hands:
                frames[group_id] = hands
    return frames


def stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "mean_m": None,
            "median_m": None,
            "rmse_m": None,
            "p90_m": None,
            "p95_m": None,
            "max_m": None,
            "mean_mm": None,
            "median_mm": None,
            "rmse_mm": None,
            "p90_mm": None,
            "p95_mm": None,
            "max_mm": None,
        }
    arr = np.asarray(values, dtype=np.float64)
    rmse = float(np.sqrt(np.mean(arr * arr)))
    out = {
        "count": int(arr.size),
        "mean_m": float(np.mean(arr)),
        "median_m": float(np.median(arr)),
        "rmse_m": rmse,
        "p90_m": float(np.percentile(arr, 90)),
        "p95_m": float(np.percentile(arr, 95)),
        "max_m": float(np.max(arr)),
    }
    out.update({key.replace("_m", "_mm"): value * 1000.0 for key, value in out.items() if key.endswith("_m")})
    return out


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        error = float(row["error_m"])
        buckets["overall"].append(error)
        buckets[f"hand:{row['handedness']}"].append(error)
        buckets[f"finger:{row['finger']}"].append(error)
        buckets[f"joint:{row['joint_name']}"].append(error)
        buckets[f"hand_finger:{row['handedness']}:{row['finger']}"].append(error)
    return {name: stats(values) for name, values in sorted(buckets.items())}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "group_id",
        "handedness",
        "finger",
        "joint_index",
        "joint_name",
        "error_m",
        "error_mm",
        "hamer_x",
        "hamer_y",
        "hamer_z",
        "glove_x",
        "glove_y",
        "glove_z",
        "hamer_mode",
        "hamer_metric_valid",
        "hamer_anchor_camera",
        "hamer_used_cameras",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_one(
    hamer_path: Path,
    glove_frames: dict[int, dict[str, dict[str, Any]]],
    group_filter: set[int] | None,
    space: str,
    hands: list[str],
    fingers: list[str],
    target_joints: list[tuple[str, int]],
    require_metric_valid: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    hamer_frames = load_hands_by_group(hamer_path, space, group_filter)
    common_groups = sorted(set(hamer_frames) & set(glove_frames))

    rows: list[dict[str, Any]] = []
    missing_hamer = 0
    missing_glove = 0
    skipped_nonmetric = 0
    for group_id in common_groups:
        for handedness in hands:
            hamer_hand = hamer_frames[group_id].get(handedness)
            glove_hand = glove_frames[group_id].get(handedness)
            if hamer_hand is None:
                missing_hamer += 1
                continue
            if glove_hand is None:
                missing_glove += 1
                continue
            if require_metric_valid and not hamer_hand.get("metric_valid"):
                skipped_nonmetric += 1
                continue
            hamer_positions = hamer_hand["_positions"]
            glove_positions = glove_hand["_positions"]
            for finger, joint_index in target_joints:
                delta = hamer_positions[joint_index] - glove_positions[joint_index]
                error_m = float(np.linalg.norm(delta))
                joint_name = LANDMARK_NAMES[joint_index] if joint_index < len(LANDMARK_NAMES) else f"joint_{joint_index}"
                rows.append(
                    {
                        "group_id": group_id,
                        "handedness": handedness,
                        "finger": finger,
                        "joint_index": joint_index,
                        "joint_name": joint_name,
                        "error_m": error_m,
                        "error_mm": error_m * 1000.0,
                        "hamer_x": float(hamer_positions[joint_index, 0]),
                        "hamer_y": float(hamer_positions[joint_index, 1]),
                        "hamer_z": float(hamer_positions[joint_index, 2]),
                        "glove_x": float(glove_positions[joint_index, 0]),
                        "glove_y": float(glove_positions[joint_index, 1]),
                        "glove_z": float(glove_positions[joint_index, 2]),
                        "hamer_mode": hamer_hand.get("mode"),
                        "hamer_metric_valid": bool(hamer_hand.get("metric_valid")),
                        "hamer_anchor_camera": hamer_hand.get("anchor_camera"),
                        "hamer_used_cameras": ",".join(hamer_hand.get("used_cameras") or []),
                    }
                )

    summary = {
        "hamer": str(hamer_path),
        "space": space,
        "hands": hands,
        "fingers": fingers,
        "joint_indices": {finger: FINGER_JOINTS[finger] for finger in fingers},
        "matched_groups": len(common_groups),
        "evaluated_points": len(rows),
        "missing_hamer_hands": missing_hamer,
        "missing_glove_hands": missing_glove,
        "skipped_nonmetric_hamer_hands": skipped_nonmetric,
        "stats": summarize(rows),
    }
    return summary, rows


def main() -> None:
    args = parse_args()
    group_filter = apply_group_parity(parse_group_filter(args.group_range, args.group_ids), args.group_parity)
    hands = parse_hands(args.hands)
    fingers = parse_fingers(args.fingers)
    target_joints = [(finger, index) for finger in fingers for index in FINGER_JOINTS[finger]]

    glove_frames = load_hands_by_group(args.glove, args.space, group_filter, allow_specialized_fallback=True)
    summary, rows = evaluate_one(
        args.hamer,
        glove_frames,
        group_filter,
        args.space,
        hands,
        fingers,
        target_joints,
        args.require_hamer_metric_valid,
    )
    summary["glove"] = str(args.glove)
    summary["group_parity"] = args.group_parity
    comparisons = []
    for compare_path in args.compare_hamer or []:
        compare_summary, _compare_rows = evaluate_one(
            compare_path,
            glove_frames,
            group_filter,
            args.space,
            hands,
            fingers,
            target_joints,
            args.require_hamer_metric_valid,
        )
        comparisons.append(compare_summary)
    if comparisons:
        summary["comparisons"] = comparisons

    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.output_csv:
        write_csv(args.output_csv, rows)


if __name__ == "__main__":
    main()
