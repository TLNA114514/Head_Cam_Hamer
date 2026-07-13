#!/usr/bin/env python3
"""Estimate conservative rectified-pixel projection corrections from refinement diagnostics."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from hamer_multiview_utils import iter_jsonl, parse_group_ids, range_suffix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refined", action="append", type=Path, required=True, help="Refined JSONL with projection_debug.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--group-range")
    parser.add_argument("--group-ids")
    parser.add_argument("--image-width", type=float, default=1600.0)
    parser.add_argument("--image-height", type=float, default=1200.0)
    parser.add_argument("--max-input-error-px", type=float, default=500.0)
    parser.add_argument("--min-points", type=int, default=200)
    parser.add_argument("--huber-scale-px", type=float, default=90.0)
    parser.add_argument("--max-scale-delta", type=float, default=0.18)
    parser.add_argument("--max-shift-px", type=float, default=180.0)
    parser.add_argument("--min-mean-improvement-px", type=float, default=8.0)
    parser.add_argument("--iters", type=int, default=8)
    parser.add_argument("--mode", choices=["axis-aligned", "translation"], default="axis-aligned")
    return parser.parse_args()


def finite_points(value: Any) -> np.ndarray | None:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 2 or len(arr) == 0:
        return None
    arr = arr[:, :2]
    if not np.all(np.isfinite(arr)):
        return None
    return arr


def collect_pairs(paths: list[Path], group_ids: set[int] | None, args: argparse.Namespace) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    by_camera: dict[str, list[tuple[np.ndarray, np.ndarray]]] = defaultdict(list)
    for path in paths:
        for frame in iter_jsonl(path):
            group_id = int(frame.get("group_id", -1))
            if group_ids is not None and group_id not in group_ids:
                continue
            for hand in frame.get("hands") or []:
                for camera_id, info in (hand.get("projection_debug") or {}).items():
                    projected = finite_points(info.get("joints_2d_px"))
                    target = finite_points(info.get("hamer_joints_2d_px"))
                    if projected is None or target is None:
                        continue
                    n = min(len(projected), len(target), 21)
                    if n < 6:
                        continue
                    projected = projected[:n]
                    target = target[:n]
                    in_bounds = (
                        (projected[:, 0] >= -args.image_width * 0.25)
                        & (projected[:, 0] <= args.image_width * 1.25)
                        & (projected[:, 1] >= -args.image_height * 0.25)
                        & (projected[:, 1] <= args.image_height * 1.25)
                        & (target[:, 0] >= 0)
                        & (target[:, 0] < args.image_width)
                        & (target[:, 1] >= 0)
                        & (target[:, 1] < args.image_height)
                    )
                    residual = np.linalg.norm(projected - target, axis=1)
                    keep = in_bounds & (residual <= args.max_input_error_px)
                    if np.any(keep):
                        by_camera[str(camera_id)].append((projected[keep], target[keep]))
    return by_camera


def weighted_percentile(values: np.ndarray, q: float) -> float:
    if len(values) == 0:
        return float("nan")
    return float(np.percentile(values, q))


def residual_stats(source: np.ndarray, target: np.ndarray, affine: np.ndarray) -> dict[str, float]:
    pred = apply_affine(source, affine)
    residual = np.linalg.norm(pred - target, axis=1)
    return {
        "mean_px": float(np.mean(residual)),
        "median_px": float(np.median(residual)),
        "p95_px": weighted_percentile(residual, 95),
        "max_px": float(np.max(residual)),
    }


def apply_affine(points: np.ndarray, affine: np.ndarray) -> np.ndarray:
    homo = np.concatenate([points[:, :2], np.ones((len(points), 1), dtype=np.float64)], axis=1)
    return homo @ affine.T


def fit_translation(source: np.ndarray, target: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    delta = target - source
    weights = np.ones(len(source), dtype=np.float64)
    shift = np.zeros(2, dtype=np.float64)
    for _ in range(max(1, args.iters)):
        shift = np.average(delta, axis=0, weights=weights)
        shift = np.clip(shift, -args.max_shift_px, args.max_shift_px)
        residual = np.linalg.norm(source + shift.reshape(1, 2) - target, axis=1)
        weights = 1.0 / np.maximum(1.0, residual / max(args.huber_scale_px, 1e-6))
    return np.asarray([[1.0, 0.0, shift[0]], [0.0, 1.0, shift[1]]], dtype=np.float64)


def fit_axis_aligned(source: np.ndarray, target: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    weights = np.ones(len(source), dtype=np.float64)
    params = np.asarray([1.0, 0.0, 1.0, 0.0], dtype=np.float64)
    x_design = np.stack([source[:, 0], np.ones(len(source))], axis=1)
    y_design = np.stack([source[:, 1], np.ones(len(source))], axis=1)
    for _ in range(max(1, args.iters)):
        sqrt_w = np.sqrt(np.maximum(weights, 1e-8))
        ax, bx = np.linalg.lstsq(x_design * sqrt_w[:, None], target[:, 0] * sqrt_w, rcond=None)[0]
        ay, by = np.linalg.lstsq(y_design * sqrt_w[:, None], target[:, 1] * sqrt_w, rcond=None)[0]
        ax = float(np.clip(ax, 1.0 - args.max_scale_delta, 1.0 + args.max_scale_delta))
        ay = float(np.clip(ay, 1.0 - args.max_scale_delta, 1.0 + args.max_scale_delta))
        bx = float(np.clip(bx, -args.max_shift_px, args.max_shift_px))
        by = float(np.clip(by, -args.max_shift_px, args.max_shift_px))
        params = np.asarray([ax, bx, ay, by], dtype=np.float64)
        affine = np.asarray([[ax, 0.0, bx], [0.0, ay, by]], dtype=np.float64)
        residual = np.linalg.norm(apply_affine(source, affine) - target, axis=1)
        weights = 1.0 / np.maximum(1.0, residual / max(args.huber_scale_px, 1e-6))
    return np.asarray([[params[0], 0.0, params[1]], [0.0, params[2], params[3]]], dtype=np.float64)


def identity_affine() -> np.ndarray:
    return np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)


def main() -> None:
    args = parse_args()
    group_ids = parse_group_ids(args.group_range, args.group_ids)
    pairs = collect_pairs(args.refined, group_ids, args)
    corrections = {}
    diagnostics = {}
    for camera_id, chunks in sorted(pairs.items()):
        source = np.concatenate([item[0] for item in chunks], axis=0)
        target = np.concatenate([item[1] for item in chunks], axis=0)
        before = residual_stats(source, target, identity_affine())
        if len(source) < args.min_points:
            affine = identity_affine()
            status = "insufficient_points"
        elif args.mode == "translation":
            affine = fit_translation(source, target, args)
            status = "estimated_translation"
        else:
            affine = fit_axis_aligned(source, target, args)
            status = "estimated_axis_aligned"
        after = residual_stats(source, target, affine)
        if after["mean_px"] > before["mean_px"] - args.min_mean_improvement_px:
            affine = identity_affine()
            after = residual_stats(source, target, affine)
            status = "identity_no_reliable_improvement"
        corrections[camera_id] = {
            "affine_projected_to_rectified_px": affine.tolist(),
            "mode": args.mode,
            "status": status,
        }
        diagnostics[camera_id] = {
            "point_count": int(len(source)),
            "frame_hand_observation_count": int(len(chunks)),
            "before": before,
            "after": after,
            "mean_improvement_px": before["mean_px"] - after["mean_px"] if math.isfinite(before["mean_px"]) else None,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "type": "rectified_projection_correction",
                "range_suffix": range_suffix(group_ids),
                "source_refined": [str(path) for path in args.refined],
                "camera_projection_corrections": corrections,
                "diagnostics": diagnostics,
                "args": vars(args) | {"refined": [str(path) for path in args.refined], "output": str(args.output)},
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
        f.write("\n")
    print(f"wrote: {args.output}")
    for camera_id, diag in diagnostics.items():
        print(
            f"{camera_id}: n={diag['point_count']} "
            f"mean {diag['before']['mean_px']:.1f}->{diag['after']['mean_px']:.1f}px "
            f"p95 {diag['before']['p95_px']:.1f}->{diag['after']['p95_px']:.1f}px"
        )


if __name__ == "__main__":
    main()
