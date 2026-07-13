#!/usr/bin/env python3
"""Fit a low-capacity pose-dependent residual calibration in hand-local space."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from evaluate_hamer_vs_glove import apply_group_parity, hand_positions, load_hands_by_group, parse_group_filter
from hamer_multiview_utils import iter_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hamer", type=Path, required=True, help="Input HaMeR/local JSONL, usually after static glove calibration.")
    parser.add_argument("--glove", type=Path, help="Glove GT JSONL. Required unless --load-calibration-json is used.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration-json", type=Path)
    parser.add_argument("--load-calibration-json", type=Path)
    parser.add_argument(
        "--space",
        choices=["palm-local", "root-relative", "glove-calibrated-palm-local", "glove-calibrated-root-relative"],
        default="glove-calibrated-palm-local",
    )
    parser.add_argument("--hands", default="Left,Right")
    parser.add_argument("--group-range")
    parser.add_argument("--group-ids")
    parser.add_argument("--train-parity", choices=["all", "even", "odd"], default="even")
    parser.add_argument(
        "--train-group-range",
        help="Explicit inclusive groups used only to fit the calibration; supersedes --train-parity.",
    )
    parser.add_argument(
        "--train-group-ids",
        help="Explicit comma-separated groups used only to fit the calibration; supersedes --train-parity.",
    )
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument(
        "--regressor",
        choices=["ridge", "local-knn", "ridge-knn"],
        default="ridge",
        help="Residual predictor. KNN variants retain the same OOD guard.",
    )
    parser.add_argument("--knn-k", type=int, default=8)
    parser.add_argument("--knn-bandwidth-scale", type=float, default=1.0)
    parser.add_argument(
        "--knn-blend",
        type=float,
        default=0.5,
        help="Local-KNN weight for --regressor ridge-knn.",
    )
    parser.add_argument("--correction-shrink", type=float, default=0.75)
    parser.add_argument("--max-correction-m", type=float, default=0.025)
    parser.add_argument(
        "--ood-gating",
        choices=["none", "knn-linear"],
        default="knn-linear",
        help="Attenuate pose residuals outside the pose distribution seen during calibration.",
    )
    parser.add_argument(
        "--ood-full-quantile",
        type=float,
        default=0.75,
        help="Leave-one-out nearest-neighbor distance quantile that still receives full correction.",
    )
    parser.add_argument(
        "--ood-zero-quantile",
        type=float,
        default=0.99,
        help="Leave-one-out nearest-neighbor distance quantile at which correction reaches zero.",
    )
    parser.add_argument(
        "--ood-min-gate",
        type=float,
        default=0.0,
        help="Minimum correction multiplier for out-of-distribution poses.",
    )
    parser.add_argument(
        "--feature-mode",
        choices=["all-joints", "all-joints-velocity", "fingertip-summary"],
        default="all-joints",
        help="Pose descriptor used by the residual regressor.",
    )
    parser.add_argument(
        "--write-mode",
        choices=["separate", "overwrite"],
        default="separate",
        help="Write corrected coordinates to glove_calibrated_* fields by default.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_hands(value: str) -> list[str]:
    hands = [item.strip() for item in value.split(",") if item.strip()]
    invalid = [item for item in hands if item not in {"Left", "Right"}]
    if invalid:
        raise SystemExit(f"Invalid hands: {invalid}")
    return hands


def output_keys(space: str) -> tuple[str, str]:
    if "root-relative" in space:
        return "glove_calibrated_root_relative_joints_m", "glove_calibrated_root_relative_m"
    return "glove_calibrated_palm_local_joints_m", "glove_calibrated_palm_local_m"


def feature_vector(
    points: np.ndarray,
    mode: str,
    context_frames: dict[int, dict[str, dict[str, Any]]] | None = None,
    group_id: int | None = None,
    handedness: str | None = None,
) -> np.ndarray:
    centered = points - points[0:1]
    if mode == "all-joints":
        return centered.reshape(-1)
    if mode == "all-joints-velocity":
        previous = points
        following = points
        if context_frames is not None and group_id is not None and handedness is not None:
            previous_hand = context_frames.get(group_id - 1, {}).get(handedness)
            following_hand = context_frames.get(group_id + 1, {}).get(handedness)
            if previous_hand is not None:
                candidate = previous_hand.get("_positions")
                if isinstance(candidate, np.ndarray) and candidate.shape == (21, 3):
                    previous = candidate
            if following_hand is not None:
                candidate = following_hand.get("_positions")
                if isinstance(candidate, np.ndarray) and candidate.shape == (21, 3):
                    following = candidate
        previous = previous - previous[0:1]
        following = following - following[0:1]
        velocity = 0.5 * (following - previous)
        return np.concatenate([centered.reshape(-1), velocity.reshape(-1)])
    if mode == "fingertip-summary":
        tips = centered[[4, 8, 12, 16, 20]]
        mcps = centered[[2, 5, 9, 13, 17]]
        tip_norms = np.linalg.norm(tips, axis=1)
        mcp_norms = np.linalg.norm(mcps, axis=1)
        return np.concatenate([tips.reshape(-1), mcps.reshape(-1), tip_norms, mcp_norms])
    raise ValueError(f"Unsupported feature mode: {mode}")


def collect_training_matrices(
    hamer_frames: dict[int, dict[str, dict[str, Any]]],
    glove_frames: dict[int, dict[str, dict[str, Any]]],
    handedness: str,
    feature_mode: str,
    context_frames: dict[int, dict[str, dict[str, Any]]] | None,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    xs = []
    ys = []
    groups = []
    for group_id in sorted(set(hamer_frames) & set(glove_frames)):
        hamer_hand = hamer_frames[group_id].get(handedness)
        glove_hand = glove_frames[group_id].get(handedness)
        if hamer_hand is None or glove_hand is None:
            continue
        hamer_pos = hamer_hand["_positions"]
        glove_pos = glove_hand["_positions"]
        if hamer_pos.shape != (21, 3) or glove_pos.shape != (21, 3):
            continue
        xs.append(feature_vector(hamer_pos, feature_mode, context_frames, group_id, handedness))
        ys.append((glove_pos - hamer_pos).reshape(-1))
        groups.append(group_id)
    if not xs:
        return np.empty((0, 0), dtype=np.float64), np.empty((0, 63), dtype=np.float64), []
    return np.stack(xs, axis=0), np.stack(ys, axis=0), groups


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> dict[str, Any]:
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y):
        raise ValueError(f"Bad training shapes: x={x.shape}, y={y.shape}")
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-8] = 1.0
    x_norm = (x - mean.reshape(1, -1)) / scale.reshape(1, -1)
    design = np.concatenate([np.ones((len(x_norm), 1), dtype=np.float64), x_norm], axis=1)
    reg = np.eye(design.shape[1], dtype=np.float64) * float(alpha)
    reg[0, 0] = 0.0
    weights = np.linalg.solve(design.T @ design + reg, design.T @ y)
    pred = design @ weights
    residual = y - pred
    # The calibration clips corrections outside its observed pose manifold.  A
    # nearest-neighbor distance is deliberately used instead of a Gaussian
    # covariance: hand-pose features are highly correlated and sample counts
    # are small enough that a full covariance estimate is unstable.
    if len(x_norm) > 1:
        delta = x_norm[:, None, :] - x_norm[None, :, :]
        distances = np.sqrt(np.mean(delta * delta, axis=2))
        np.fill_diagonal(distances, np.inf)
        nearest_train_distance = np.min(distances, axis=1)
    else:
        nearest_train_distance = np.zeros(len(x_norm), dtype=np.float64)
    return {
        "feature_mean": mean,
        "feature_scale": scale,
        "weights": weights,
        "train_pred": pred,
        "train_residual": residual,
        "feature_prototypes_norm": x_norm,
        "nearest_train_distance": nearest_train_distance,
        "target_residuals": y,
    }


def local_knn_prediction(
    feature_norm: np.ndarray,
    model: dict[str, Any],
    k: int,
    bandwidth_scale: float,
    exclude_index: int | None = None,
) -> np.ndarray:
    prototypes = np.asarray(model["feature_prototypes_norm"], dtype=np.float64)
    targets = np.asarray(model["target_residuals"], dtype=np.float64)
    if prototypes.ndim != 2 or targets.ndim != 2 or len(prototypes) != len(targets) or len(prototypes) == 0:
        raise ValueError("Local KNN calibration is missing compatible prototypes/targets")
    distances = np.sqrt(np.mean((prototypes - feature_norm.reshape(1, -1)) ** 2, axis=1))
    if exclude_index is not None and 0 <= exclude_index < len(distances):
        distances[exclude_index] = np.inf
    finite_indices = np.flatnonzero(np.isfinite(distances))
    if finite_indices.size == 0:
        return np.zeros(targets.shape[1], dtype=np.float64)
    count = min(max(1, int(k)), int(finite_indices.size))
    order = finite_indices[np.argsort(distances[finite_indices])[:count]]
    selected_distances = distances[order]
    bandwidth = max(float(selected_distances[-1]) * float(bandwidth_scale), 1e-6)
    weights = np.exp(-0.5 * (selected_distances / bandwidth) ** 2)
    weights /= max(float(np.sum(weights)), 1e-12)
    return np.sum(targets[order] * weights.reshape(-1, 1), axis=0)


def selected_training_prediction(
    model: dict[str, Any],
    regressor: str,
    knn_k: int,
    knn_bandwidth_scale: float,
    knn_blend: float,
) -> np.ndarray:
    ridge_prediction = np.asarray(model["train_pred"], dtype=np.float64)
    if regressor == "ridge":
        return ridge_prediction
    prototypes = np.asarray(model["feature_prototypes_norm"], dtype=np.float64)
    knn_prediction = np.stack(
        [
            local_knn_prediction(feature, model, knn_k, knn_bandwidth_scale, exclude_index=index)
            for index, feature in enumerate(prototypes)
        ],
        axis=0,
    )
    if regressor == "local-knn":
        return knn_prediction
    return (1.0 - float(knn_blend)) * ridge_prediction + float(knn_blend) * knn_prediction


def ood_gate(
    feature_norm: np.ndarray,
    model: dict[str, Any],
    mode: str,
    full_quantile: float,
    zero_quantile: float,
    min_gate: float,
) -> tuple[float, float, float, float]:
    if mode == "none" or "feature_prototypes_norm" not in model or "nearest_train_distance" not in model:
        return 1.0, 0.0, 0.0, 0.0
    prototypes = np.asarray(model["feature_prototypes_norm"], dtype=np.float64)
    train_distances = np.asarray(model["nearest_train_distance"], dtype=np.float64)
    if prototypes.ndim != 2 or len(prototypes) == 0 or train_distances.size == 0:
        return 1.0, 0.0, 0.0, 0.0
    full_threshold = float(np.quantile(train_distances, full_quantile))
    zero_threshold = float(np.quantile(train_distances, zero_quantile))
    zero_threshold = max(zero_threshold, full_threshold + 1e-8)
    distance = float(np.min(np.sqrt(np.mean((prototypes - feature_norm.reshape(1, -1)) ** 2, axis=1))))
    gate = (zero_threshold - distance) / (zero_threshold - full_threshold)
    return float(np.clip(gate, min_gate, 1.0)), distance, full_threshold, zero_threshold


def predict_correction(
    points: np.ndarray,
    model: dict[str, Any],
    feature_mode: str,
    shrink: float,
    max_norm: float,
    gating_mode: str,
    ood_full_quantile: float,
    ood_zero_quantile: float,
    ood_min_gate: float,
    context_frames: dict[int, dict[str, dict[str, Any]]] | None,
    group_id: int,
    handedness: str,
    regressor: str,
    knn_k: int,
    knn_bandwidth_scale: float,
    knn_blend: float,
) -> tuple[np.ndarray, dict[str, float | str]]:
    mean = np.asarray(model["feature_mean"], dtype=np.float64)
    scale = np.asarray(model["feature_scale"], dtype=np.float64)
    weights = np.asarray(model["weights"], dtype=np.float64)
    feat = feature_vector(points, feature_mode, context_frames, group_id, handedness)
    feat_norm = (feat - mean) / scale
    design = np.concatenate([[1.0], feat_norm])
    ridge_correction = design @ weights
    if regressor == "ridge":
        predicted_correction = ridge_correction
    else:
        knn_correction = local_knn_prediction(feat_norm, model, knn_k, knn_bandwidth_scale)
        if regressor == "local-knn":
            predicted_correction = knn_correction
        else:
            predicted_correction = (1.0 - float(knn_blend)) * ridge_correction + float(knn_blend) * knn_correction
    correction = predicted_correction.reshape(21, 3)
    gate, distance, full_threshold, zero_threshold = ood_gate(
        feat_norm,
        model,
        gating_mode,
        ood_full_quantile,
        ood_zero_quantile,
        ood_min_gate,
    )
    correction *= float(shrink) * gate
    if max_norm > 0.0:
        norms = np.linalg.norm(correction, axis=1)
        over = norms > max_norm
        correction[over] *= (max_norm / np.maximum(norms[over], 1e-12)).reshape(-1, 1)
    return correction, {
        "mode": gating_mode,
        "gate": gate,
        "nearest_distance": distance,
        "full_threshold": full_threshold,
        "zero_threshold": zero_threshold,
        "regressor": regressor,
    }


def apply_to_joints_list(hand: dict[str, Any], corrected: np.ndarray, joint_key: str) -> None:
    joints = hand.get("joints")
    if not isinstance(joints, list):
        return
    for joint in joints:
        index = joint.get("index", joint.get("joint_index"))
        if isinstance(index, int) and 0 <= index < 21:
            joint[joint_key] = corrected[index].tolist()


def serialize_model(model: dict[str, Any]) -> dict[str, Any]:
    return {
        "feature_mean": np.asarray(model["feature_mean"]).tolist(),
        "feature_scale": np.asarray(model["feature_scale"]).tolist(),
        "weights": np.asarray(model["weights"]).tolist(),
        "feature_prototypes_norm": np.asarray(model.get("feature_prototypes_norm", []), dtype=np.float64).tolist(),
        "nearest_train_distance": np.asarray(model.get("nearest_train_distance", []), dtype=np.float64).tolist(),
        "target_residuals": np.asarray(model.get("target_residuals", []), dtype=np.float64).tolist(),
    }


def deserialize_model(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "feature_mean": np.asarray(raw["feature_mean"], dtype=np.float64),
        "feature_scale": np.asarray(raw["feature_scale"], dtype=np.float64),
        "weights": np.asarray(raw["weights"], dtype=np.float64),
        "feature_prototypes_norm": np.asarray(raw.get("feature_prototypes_norm", []), dtype=np.float64),
        "nearest_train_distance": np.asarray(raw.get("nearest_train_distance", []), dtype=np.float64),
        "target_residuals": np.asarray(raw.get("target_residuals", []), dtype=np.float64),
    }


def main() -> None:
    args = parse_args()
    if not (0.0 <= args.ood_full_quantile <= args.ood_zero_quantile <= 1.0):
        raise SystemExit("Require 0 <= --ood-full-quantile <= --ood-zero-quantile <= 1")
    if not (0.0 <= args.ood_min_gate <= 1.0):
        raise SystemExit("--ood-min-gate must be in [0, 1]")
    if args.knn_k < 1:
        raise SystemExit("--knn-k must be >= 1")
    if args.knn_bandwidth_scale <= 0.0:
        raise SystemExit("--knn-bandwidth-scale must be > 0")
    if not (0.0 <= args.knn_blend <= 1.0):
        raise SystemExit("--knn-blend must be in [0, 1]")
    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"{args.output} exists; pass --overwrite to replace it")
    if args.load_calibration_json is None and args.glove is None:
        raise SystemExit("--glove is required unless --load-calibration-json is used")

    hands = parse_hands(args.hands)
    all_group_filter = parse_group_filter(args.group_range, args.group_ids)
    direct_key, joint_key = output_keys(args.space)

    diagnostics: dict[str, Any] = {}
    models: dict[str, dict[str, Any]] = {}
    # This map contains only HaMeR input and is used for optional temporal
    # features; it never reads glove labels from neighbouring frames.
    context_frames = load_hands_by_group(args.hamer, args.space, all_group_filter)
    if args.load_calibration_json is not None:
        data = json.loads(args.load_calibration_json.read_text(encoding="utf-8"))
        if data.get("type") != "pose_residual_local_calibration":
            raise SystemExit(f"Unsupported calibration type in {args.load_calibration_json}: {data.get('type')}")
        args.feature_mode = data.get("feature_mode", args.feature_mode)
        args.ridge_alpha = float(data.get("ridge_alpha", args.ridge_alpha))
        args.regressor = data.get("regressor", args.regressor)
        args.knn_k = int(data.get("knn_k", args.knn_k))
        args.knn_bandwidth_scale = float(data.get("knn_bandwidth_scale", args.knn_bandwidth_scale))
        args.knn_blend = float(data.get("knn_blend", args.knn_blend))
        args.correction_shrink = float(data.get("correction_shrink", args.correction_shrink))
        args.max_correction_m = float(data.get("max_correction_m", args.max_correction_m))
        args.ood_gating = data.get("ood_gating", args.ood_gating)
        args.ood_full_quantile = float(data.get("ood_full_quantile", args.ood_full_quantile))
        args.ood_zero_quantile = float(data.get("ood_zero_quantile", args.ood_zero_quantile))
        args.ood_min_gate = float(data.get("ood_min_gate", args.ood_min_gate))
        models = {hand: deserialize_model(raw) for hand, raw in data.get("models", {}).items() if hand in hands}
        diagnostics = {hand: {"status": "loaded_existing_calibration", "source": str(args.load_calibration_json)} for hand in hands}
    else:
        explicit_train_filter = parse_group_filter(args.train_group_range, args.train_group_ids)
        train_filter = explicit_train_filter if explicit_train_filter is not None else apply_group_parity(all_group_filter, args.train_parity)
        hamer_train = load_hands_by_group(args.hamer, args.space, train_filter)
        glove_train = load_hands_by_group(args.glove, args.space, train_filter)
        for handedness in hands:
            x, y, groups = collect_training_matrices(
                hamer_train,
                glove_train,
                handedness,
                args.feature_mode,
                context_frames,
            )
            if len(groups) < 20:
                diagnostics[handedness] = {"status": "skipped_not_enough_training_groups", "train_groups": len(groups)}
                continue
            model = fit_ridge(x, y, args.ridge_alpha)
            selected_pred = selected_training_prediction(
                model,
                args.regressor,
                args.knn_k,
                args.knn_bandwidth_scale,
                args.knn_blend,
            )
            pred = np.asarray(selected_pred, dtype=np.float64).reshape(len(groups), 21, 3)
            target = y.reshape(len(groups), 21, 3)
            correction = pred * float(args.correction_shrink)
            if args.max_correction_m > 0.0:
                norms = np.linalg.norm(correction, axis=2)
                correction[norms > args.max_correction_m] *= (
                    args.max_correction_m / np.maximum(norms[norms > args.max_correction_m], 1e-12)
                ).reshape(-1, 1)
            before = np.linalg.norm(target, axis=2).reshape(-1)
            after = np.linalg.norm(target - correction, axis=2).reshape(-1)
            models[handedness] = {
                "feature_mean": model["feature_mean"],
                "feature_scale": model["feature_scale"],
                "weights": model["weights"],
                "feature_prototypes_norm": model["feature_prototypes_norm"],
                "nearest_train_distance": model["nearest_train_distance"],
                "target_residuals": model["target_residuals"],
            }
            diagnostics[handedness] = {
                "status": "estimated_pose_residual",
                "train_groups": len(groups),
                "feature_dim": int(x.shape[1]),
                "target_dim": int(y.shape[1]),
                "regressor": args.regressor,
                "before_mean_mm": float(np.mean(before) * 1000.0),
                "after_mean_mm": float(np.mean(after) * 1000.0),
                "before_p95_mm": float(np.percentile(before, 95) * 1000.0),
                "after_p95_mm": float(np.percentile(after, 95) * 1000.0),
                "mean_correction_norm_mm": float(np.mean(np.linalg.norm(correction, axis=2)) * 1000.0),
                "max_correction_norm_mm": float(np.max(np.linalg.norm(correction, axis=2)) * 1000.0),
            }

    frames = []
    correction_norms: dict[str, list[float]] = {hand: [] for hand in hands}
    gates: dict[str, list[float]] = {hand: [] for hand in hands}
    ood_distances: dict[str, list[float]] = {hand: [] for hand in hands}
    for frame in iter_jsonl(args.hamer):
        group_id = int(frame.get("group_id", -1))
        if all_group_filter is not None and group_id not in all_group_filter:
            continue
        out_frame = dict(frame)
        out_hands = []
        for raw_hand in frame.get("hands", []):
            hand = dict(raw_hand)
            handedness = hand.get("handedness")
            points = hand_positions(hand, args.space)
            if handedness in models and points is not None:
                correction, gate_info = predict_correction(
                    points,
                    models[handedness],
                    args.feature_mode,
                    args.correction_shrink,
                    args.max_correction_m,
                    args.ood_gating,
                    args.ood_full_quantile,
                    args.ood_zero_quantile,
                    args.ood_min_gate,
                    context_frames,
                    group_id,
                    handedness,
                    args.regressor,
                    args.knn_k,
                    args.knn_bandwidth_scale,
                    args.knn_blend,
                )
                corrected = points + correction
                if args.write_mode == "overwrite" and args.space == "palm-local":
                    hand["palm_local_joints_m"] = corrected.tolist()
                    apply_to_joints_list(hand, corrected, "palm_local_m")
                elif args.write_mode == "overwrite" and args.space == "root-relative":
                    hand["local_joints_m"] = corrected.tolist()
                    apply_to_joints_list(hand, corrected, "root_relative_headset_m")
                else:
                    hand[direct_key] = corrected.tolist()
                    apply_to_joints_list(hand, corrected, joint_key)
                hand["pose_residual_calibration"] = {
                    "feature_mode": args.feature_mode,
                    "ridge_alpha": float(args.ridge_alpha),
                    "regressor": args.regressor,
                    "knn_k": int(args.knn_k),
                    "knn_bandwidth_scale": float(args.knn_bandwidth_scale),
                    "knn_blend": float(args.knn_blend),
                    "correction_shrink": float(args.correction_shrink),
                    "max_correction_m": float(args.max_correction_m),
                    "ood_gating": gate_info,
                }
                correction_norms[handedness].extend(np.linalg.norm(correction, axis=1).tolist())
                gates[handedness].append(float(gate_info["gate"]))
                ood_distances[handedness].append(float(gate_info["nearest_distance"]))
            out_hands.append(hand)
        out_frame["hands"] = out_hands
        frames.append(out_frame)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for frame in frames:
            f.write(json.dumps(frame, ensure_ascii=False, separators=(",", ":")) + "\n")

    config = {
        "type": "pose_residual_local_calibration",
        "input": str(args.hamer),
        "glove": str(args.glove) if args.glove else None,
        "space": args.space,
        "feature_mode": args.feature_mode,
        "ridge_alpha": float(args.ridge_alpha),
        "regressor": args.regressor,
        "knn_k": int(args.knn_k),
        "knn_bandwidth_scale": float(args.knn_bandwidth_scale),
        "knn_blend": float(args.knn_blend),
        "correction_shrink": float(args.correction_shrink),
        "max_correction_m": float(args.max_correction_m),
        "ood_gating": args.ood_gating,
        "ood_full_quantile": float(args.ood_full_quantile),
        "ood_zero_quantile": float(args.ood_zero_quantile),
        "ood_min_gate": float(args.ood_min_gate),
        "train_parity": args.train_parity,
        "train_group_range": args.train_group_range,
        "train_group_ids": args.train_group_ids,
        "group_range": args.group_range,
        "group_ids": args.group_ids,
        "models": {hand: serialize_model(model) for hand, model in models.items()},
        "diagnostics": diagnostics,
        "apply_stats": {
            hand: {
                "mean_correction_norm_mm": float(np.mean(values) * 1000.0) if values else 0.0,
                "p95_correction_norm_mm": float(np.percentile(values, 95) * 1000.0) if values else 0.0,
                "max_correction_norm_mm": float(np.max(values) * 1000.0) if values else 0.0,
                "mean_ood_gate": float(np.mean(gates[hand])) if gates[hand] else 1.0,
                "p05_ood_gate": float(np.percentile(gates[hand], 5)) if gates[hand] else 1.0,
                "fully_attenuated_frame_count": int(np.sum(np.asarray(gates[hand]) <= 1e-8)) if gates[hand] else 0,
                "mean_ood_distance": float(np.mean(ood_distances[hand])) if ood_distances[hand] else 0.0,
            }
            for hand, values in correction_norms.items()
        },
    }
    if args.calibration_json:
        args.calibration_json.parent.mkdir(parents=True, exist_ok=True)
        args.calibration_json.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"wrote: {args.output}")
    if args.calibration_json:
        print(f"calibration: {args.calibration_json}")
    print(json.dumps(diagnostics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
