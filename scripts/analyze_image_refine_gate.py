#!/usr/bin/env python3
"""Diagnose when PnP-gated image refinement improves local hand accuracy.

This is an evaluation-only aid: glove GT is used to analyze candidate selection,
never to alter a production MANO output.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from evaluate_hamer_vs_glove import FINGER_JOINTS, load_hands_by_group


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--gated", type=Path, required=True)
    parser.add_argument("--glove", type=Path, required=True)
    parser.add_argument("--group-range", help="Inclusive range, e.g. 0-20.")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def parse_group_range(value: str | None) -> set[int] | None:
    if not value:
        return None
    selected: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            selected.update(range(int(start), int(end) + 1))
        else:
            selected.add(int(part))
    return selected or None


def finite_float(value: Any) -> float | None:
    if isinstance(value, (int, float)) and np.isfinite(float(value)):
        return float(value)
    return None


def diagnostics(hand: dict[str, Any]) -> dict[str, float | int | None]:
    views = hand.get("projection_debug") or {}
    pnp_values, reprojection_values, mask_values = [], [], []
    for item in views.values():
        if not isinstance(item, dict):
            continue
        value = finite_float(item.get("pnp_wrist_distance_m"))
        if value is not None:
            pnp_values.append(value)
        value = finite_float(item.get("mean_reprojection_error_px"))
        if value is not None:
            reprojection_values.append(value)
        value = finite_float(item.get("mean_mask_distance_px"))
        if value is not None:
            mask_values.append(value)
    return {
        "camera_count": len(views),
        "used_camera_count": len(hand.get("used_cameras") or []),
        "max_pnp_wrist_distance_m": max(pnp_values) if pnp_values else None,
        "max_reprojection_error_px": max(reprojection_values) if reprojection_values else None,
        "max_mask_distance_px": max(mask_values) if mask_values else None,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    gains = np.asarray([row["gated_gain_mm"] for row in rows], dtype=np.float64)
    return {
        "count": int(len(rows)),
        "mean_gated_gain_mm": float(gains.mean()) if len(gains) else None,
        "median_gated_gain_mm": float(np.median(gains)) if len(gains) else None,
        "gated_better_rate": float(np.mean(gains > 0.0)) if len(gains) else None,
    }


def main() -> None:
    args = parse_args()
    groups = parse_group_range(args.group_range)
    baseline = load_hands_by_group(args.baseline, "palm-local", groups)
    gated = load_hands_by_group(args.gated, "palm-local", groups)
    glove = load_hands_by_group(args.glove, "palm-local", groups)
    joint_indices = [index for finger in ("thumb", "index", "middle") for index in FINGER_JOINTS[finger]]

    rows = []
    for group_id in sorted(set(baseline) & set(gated) & set(glove)):
        for handedness in ("Left", "Right"):
            base_hand = baseline[group_id].get(handedness)
            gated_hand = gated[group_id].get(handedness)
            glove_hand = glove[group_id].get(handedness)
            if base_hand is None or gated_hand is None or glove_hand is None:
                continue
            base_error = np.linalg.norm(base_hand["_positions"][joint_indices] - glove_hand["_positions"][joint_indices], axis=1)
            gated_error = np.linalg.norm(gated_hand["_positions"][joint_indices] - glove_hand["_positions"][joint_indices], axis=1)
            row = {
                "group_id": group_id,
                "handedness": handedness,
                "baseline_error_mm": float(np.mean(base_error) * 1000.0),
                "gated_error_mm": float(np.mean(gated_error) * 1000.0),
                "gated_gain_mm": float((np.mean(base_error) - np.mean(gated_error)) * 1000.0),
                "baseline_anchor_camera": base_hand.get("anchor_camera"),
                "gated_anchor_camera": gated_hand.get("anchor_camera"),
                **diagnostics(base_hand),
            }
            rows.append(row)

    bins: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        value = row["max_pnp_wrist_distance_m"]
        if value is None:
            key = "pnp_missing"
        elif value <= 0.04:
            key = "pnp_le_0p04"
        elif value <= 0.08:
            key = "pnp_0p04_0p08"
        else:
            key = "pnp_gt_0p08"
        bins[key].append(row)
    by_anchor: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_used_cameras: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_reprojection: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_anchor[str(row["baseline_anchor_camera"] or "missing")].append(row)
        by_used_cameras[str(row["used_camera_count"])].append(row)
        reprojection = row["max_reprojection_error_px"]
        if reprojection is None:
            reprojection_key = "reprojection_missing"
        elif reprojection <= 100.0:
            reprojection_key = "reprojection_le_100"
        elif reprojection <= 500.0:
            reprojection_key = "reprojection_100_500"
        else:
            reprojection_key = "reprojection_gt_500"
        by_reprojection[reprojection_key].append(row)
    payload = {
        "type": "image_refine_gate_diagnostic",
        "baseline": str(args.baseline),
        "gated": str(args.gated),
        "glove": str(args.glove),
        "rows": rows,
        "overall": summarize(rows),
        "by_max_pnp_wrist_distance": {key: summarize(value) for key, value in sorted(bins.items())},
        "by_baseline_anchor": {key: summarize(value) for key, value in sorted(by_anchor.items())},
        "by_used_camera_count": {key: summarize(value) for key, value in sorted(by_used_cameras.items())},
        "by_max_reprojection": {key: summarize(value) for key, value in sorted(by_reprojection.items())},
    }
    print(json.dumps({key: value for key, value in payload.items() if key != "rows"}, ensure_ascii=False, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
