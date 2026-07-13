#!/usr/bin/env python3
"""Fit a conservative hand-local similarity calibration from HaMeR/MANO output to glove GT."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from evaluate_hamer_vs_glove import apply_group_parity, hand_positions, load_hands_by_group, parse_group_filter
from hamer_multiview_utils import iter_jsonl


HAND_BONES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hamer", type=Path, required=True)
    parser.add_argument("--glove", type=Path, help="Glove GT JSONL. Required unless --load-calibration-json is used.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration-json", type=Path)
    parser.add_argument(
        "--load-calibration-json",
        type=Path,
        help="Apply an existing calibration JSON instead of fitting from glove GT.",
    )
    parser.add_argument("--space", choices=["palm-local", "root-relative"], default="palm-local")
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
    parser.add_argument("--fit-joints", default="0,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20")
    parser.add_argument("--min-pairs", type=int, default=200)
    parser.add_argument("--allow-scale", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-translation", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--write-mode",
        choices=["separate", "overwrite"],
        default="separate",
        help="Write calibrated coordinates into separate glove_calibrated_* fields by default.",
    )
    parser.add_argument(
        "--recenter-wrist-after-transform",
        action="store_true",
        help="Subtract transformed wrist from every calibrated hand, preserving wrist-origin local semantics.",
    )
    parser.add_argument(
        "--joint-offsets",
        choices=["none", "mean"],
        default="none",
        help="Optionally fit conservative per-joint residual offsets after the global similarity transform.",
    )
    parser.add_argument(
        "--joint-offset-shrink-k",
        type=float,
        default=200.0,
        help="Shrink joint offsets by n/(n+k) to reduce overfitting.",
    )
    parser.add_argument(
        "--max-joint-offset-m",
        type=float,
        default=0.03,
        help="Clamp each per-joint residual offset norm.",
    )
    parser.add_argument(
        "--bone-scales",
        choices=["none", "median"],
        default="none",
        help="Optionally fit conservative per-bone length scale factors after the global similarity transform.",
    )
    parser.add_argument(
        "--bone-scale-shrink-k",
        type=float,
        default=100.0,
        help="Shrink bone length scale factors toward 1 by n/(n+k).",
    )
    parser.add_argument("--min-bone-scale", type=float, default=0.70)
    parser.add_argument("--max-bone-scale", type=float, default=1.30)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_hands(value: str) -> list[str]:
    hands = [item.strip() for item in value.split(",") if item.strip()]
    invalid = [item for item in hands if item not in {"Left", "Right"}]
    if invalid:
        raise SystemExit(f"Invalid hands: {invalid}")
    return hands


def parse_joint_indices(value: str) -> list[int]:
    indices = [int(item.strip()) for item in value.split(",") if item.strip()]
    invalid = [item for item in indices if item < 0 or item >= 21]
    if invalid:
        raise SystemExit(f"Invalid joint indices: {invalid}")
    return indices


def umeyama_similarity(source: np.ndarray, target: np.ndarray, allow_scale: bool, allow_translation: bool) -> dict[str, Any]:
    src_mean = source.mean(axis=0)
    tgt_mean = target.mean(axis=0)
    src_c = source - src_mean
    tgt_c = target - tgt_mean
    cov = (tgt_c.T @ src_c) / max(len(source), 1)
    u, singular, vt = np.linalg.svd(cov)
    sign = np.ones(3, dtype=np.float64)
    if np.linalg.det(u @ vt) < 0:
        sign[-1] = -1.0
    r = u @ np.diag(sign) @ vt
    if allow_scale:
        var = float(np.mean(np.sum(src_c * src_c, axis=1)))
        scale = float(np.sum(singular * sign) / max(var, 1e-12))
    else:
        scale = 1.0
    if allow_translation:
        t = tgt_mean - scale * (r @ src_mean)
    else:
        t = np.zeros(3, dtype=np.float64)
    return {"scale": scale, "rotation": r, "translation": t}


def apply_similarity(points: np.ndarray, transform: dict[str, Any]) -> np.ndarray:
    scale = float(transform["scale"])
    rotation = np.asarray(transform["rotation"], dtype=np.float64)
    translation = np.asarray(transform["translation"], dtype=np.float64)
    return scale * (points @ rotation.T) + translation.reshape(1, 3)


def load_existing_calibration(
    path: Path,
    hands: list[str],
) -> tuple[dict[str, dict[str, Any]], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    transforms = {}
    for handedness in hands:
        raw = (data.get("transforms") or {}).get(handedness)
        if raw is None:
            transforms[handedness] = {
                "scale": 1.0,
                "rotation": np.eye(3, dtype=np.float64),
                "translation": np.zeros(3, dtype=np.float64),
            }
            continue
        transforms[handedness] = {
            "scale": float(raw.get("scale", 1.0)),
            "rotation": np.asarray(raw.get("rotation", np.eye(3).tolist()), dtype=np.float64),
            "translation": np.asarray(raw.get("translation", [0.0, 0.0, 0.0]), dtype=np.float64),
        }

    residual_offsets = {}
    raw_offsets = data.get("residual_offsets") or {}
    for handedness in hands:
        offsets = np.asarray(raw_offsets.get(handedness, np.zeros((21, 3)).tolist()), dtype=np.float64)
        if offsets.shape != (21, 3):
            raise SystemExit(f"Invalid residual_offsets shape for {handedness} in {path}: {offsets.shape}")
        residual_offsets[handedness] = offsets

    bone_scales = {}
    raw_bone_scales = data.get("bone_scales_by_hand") or {}
    for handedness in hands:
        scales = np.asarray(raw_bone_scales.get(handedness, np.ones(len(HAND_BONES)).tolist()), dtype=np.float64)
        if scales.shape != (len(HAND_BONES),):
            raise SystemExit(f"Invalid bone_scales_by_hand shape for {handedness} in {path}: {scales.shape}")
        bone_scales[handedness] = scales
    return transforms, residual_offsets, bone_scales, data


def collect_training_pairs(
    hamer_frames: dict[int, dict[str, dict[str, Any]]],
    glove_frames: dict[int, dict[str, dict[str, Any]]],
    handedness: str,
    fit_joints: list[int],
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    source = []
    target = []
    groups = []
    for group_id in sorted(set(hamer_frames) & set(glove_frames)):
        hamer_hand = hamer_frames[group_id].get(handedness)
        glove_hand = glove_frames[group_id].get(handedness)
        if hamer_hand is None or glove_hand is None:
            continue
        hamer_pos = hamer_hand["_positions"]
        glove_pos = glove_hand["_positions"]
        source.append(hamer_pos[fit_joints])
        target.append(glove_pos[fit_joints])
        groups.append(group_id)
    if not source:
        return np.empty((0, 3)), np.empty((0, 3)), []
    return np.concatenate(source, axis=0), np.concatenate(target, axis=0), groups


def estimate_joint_offsets(
    hamer_frames: dict[int, dict[str, dict[str, Any]]],
    glove_frames: dict[int, dict[str, dict[str, Any]]],
    handedness: str,
    transform: dict[str, Any],
    fit_joints: list[int],
    shrink_k: float,
    max_norm_m: float,
    bone_scales: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    sums = np.zeros((21, 3), dtype=np.float64)
    counts = np.zeros(21, dtype=np.int64)
    for group_id in sorted(set(hamer_frames) & set(glove_frames)):
        hamer_hand = hamer_frames[group_id].get(handedness)
        glove_hand = glove_frames[group_id].get(handedness)
        if hamer_hand is None or glove_hand is None:
            continue
        predicted = apply_similarity(hamer_hand["_positions"], transform)
        if bone_scales is not None:
            predicted = apply_bone_scales(predicted, bone_scales)
        residual = glove_hand["_positions"] - predicted
        for joint_index in fit_joints:
            if np.all(np.isfinite(residual[joint_index])):
                sums[joint_index] += residual[joint_index]
                counts[joint_index] += 1

    offsets = np.zeros((21, 3), dtype=np.float64)
    for joint_index, count in enumerate(counts):
        if count <= 0:
            continue
        mean = sums[joint_index] / float(count)
        shrink = float(count) / max(float(count) + float(shrink_k), 1e-12)
        offset = mean * shrink
        norm = float(np.linalg.norm(offset))
        if max_norm_m > 0.0 and norm > max_norm_m:
            offset = offset * (max_norm_m / max(norm, 1e-12))
        offsets[joint_index] = offset

    norms = np.linalg.norm(offsets, axis=1)
    diagnostics = {
        "counts": counts.tolist(),
        "mean_offset_norm_mm": float(np.mean(norms[fit_joints]) * 1000.0) if fit_joints else 0.0,
        "max_offset_norm_mm": float(np.max(norms) * 1000.0),
        "shrink_k": float(shrink_k),
        "max_joint_offset_m": float(max_norm_m),
    }
    return offsets, diagnostics


def estimate_bone_scales(
    hamer_frames: dict[int, dict[str, dict[str, Any]]],
    glove_frames: dict[int, dict[str, dict[str, Any]]],
    handedness: str,
    transform: dict[str, Any],
    shrink_k: float,
    min_scale: float,
    max_scale: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    ratios_by_bone: list[list[float]] = [[] for _ in HAND_BONES]
    for group_id in sorted(set(hamer_frames) & set(glove_frames)):
        hamer_hand = hamer_frames[group_id].get(handedness)
        glove_hand = glove_frames[group_id].get(handedness)
        if hamer_hand is None or glove_hand is None:
            continue
        source = apply_similarity(hamer_hand["_positions"], transform)
        target = glove_hand["_positions"]
        for bone_index, (parent, child) in enumerate(HAND_BONES):
            source_len = float(np.linalg.norm(source[child] - source[parent]))
            target_len = float(np.linalg.norm(target[child] - target[parent]))
            if np.isfinite(source_len) and np.isfinite(target_len) and source_len > 1e-6 and target_len > 1e-6:
                ratios_by_bone[bone_index].append(target_len / source_len)

    scales = np.ones(len(HAND_BONES), dtype=np.float64)
    counts = []
    raw_medians = []
    for bone_index, ratios in enumerate(ratios_by_bone):
        counts.append(len(ratios))
        if not ratios:
            raw_medians.append(1.0)
            continue
        median_ratio = float(np.median(np.asarray(ratios, dtype=np.float64)))
        shrink = float(len(ratios)) / max(float(len(ratios)) + float(shrink_k), 1e-12)
        scale = 1.0 + shrink * (median_ratio - 1.0)
        scales[bone_index] = float(np.clip(scale, min_scale, max_scale))
        raw_medians.append(median_ratio)

    diagnostics = {
        "bones": [{"parent": parent, "child": child} for parent, child in HAND_BONES],
        "counts": counts,
        "raw_median_scales": raw_medians,
        "scales": scales.tolist(),
        "mean_abs_delta_percent": float(np.mean(np.abs(scales - 1.0)) * 100.0),
        "max_abs_delta_percent": float(np.max(np.abs(scales - 1.0)) * 100.0),
        "shrink_k": float(shrink_k),
        "min_bone_scale": float(min_scale),
        "max_bone_scale": float(max_scale),
    }
    return scales, diagnostics


def apply_bone_scales(points: np.ndarray, bone_scales: np.ndarray) -> np.ndarray:
    if bone_scales.shape != (len(HAND_BONES),):
        raise ValueError(f"bone_scales must have shape {(len(HAND_BONES),)}, got {bone_scales.shape}")
    out = np.asarray(points, dtype=np.float64).copy()
    for bone_index, (parent, child) in enumerate(HAND_BONES):
        out[child] = out[parent] + bone_scales[bone_index] * (points[child] - points[parent])
    return out


def set_hand_positions(hand: dict[str, Any], space: str, positions: np.ndarray, write_mode: str) -> None:
    if space == "palm-local":
        separate_direct_key = "glove_calibrated_palm_local_joints_m"
        separate_joint_key = "glove_calibrated_palm_local_m"
        base_direct_key = "palm_local_joints_m"
        base_joint_key = "palm_local_m"
    else:
        separate_direct_key = "glove_calibrated_root_relative_joints_m"
        separate_joint_key = "glove_calibrated_root_relative_m"
        base_direct_key = "local_joints_m"
        base_joint_key = "root_relative_headset_m"

    hand[separate_direct_key] = positions.tolist()
    if write_mode == "overwrite":
        hand[base_direct_key] = positions.tolist()
    for index, joint in enumerate(hand.get("joints") or []):
        if isinstance(joint, dict) and index < len(positions):
            joint[separate_joint_key] = positions[index].tolist()
            if write_mode == "overwrite":
                joint[base_joint_key] = positions[index].tolist()
    hand["glove_local_calibrated"] = True


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"{args.output} exists; pass --overwrite to replace it")
    if args.load_calibration_json is None and args.glove is None:
        raise SystemExit("--glove is required when fitting a new calibration")
    hands = parse_hands(args.hands)
    fit_joints = parse_joint_indices(args.fit_joints)
    all_group_filter = parse_group_filter(args.group_range, args.group_ids)
    loaded_calibration = None
    diagnostics = {}
    if args.load_calibration_json is not None:
        transforms, joint_offsets, bone_scales, loaded_calibration = load_existing_calibration(args.load_calibration_json, hands)
        active_joint_offsets = loaded_calibration.get("joint_offsets") == "mean"
        active_bone_scales = loaded_calibration.get("bone_scales") == "median"
        diagnostics = {
            handedness: {
                "status": "loaded_existing_calibration",
                "source": str(args.load_calibration_json),
            }
            for handedness in hands
        }
    else:
        active_joint_offsets = args.joint_offsets == "mean"
        active_bone_scales = args.bone_scales == "median"
        explicit_train_filter = parse_group_filter(args.train_group_range, args.train_group_ids)
        train_filter = explicit_train_filter if explicit_train_filter is not None else apply_group_parity(all_group_filter, args.train_parity)
        hamer_train = load_hands_by_group(args.hamer, args.space, train_filter)
        glove_train = load_hands_by_group(args.glove, args.space, train_filter)
        transforms = {}
        joint_offsets = {}
        bone_scales = {}
        for handedness in hands:
            source, target, groups = collect_training_pairs(hamer_train, glove_train, handedness, fit_joints)
            if len(source) < args.min_pairs:
                transforms[handedness] = {
                    "scale": 1.0,
                    "rotation": np.eye(3, dtype=np.float64),
                    "translation": np.zeros(3, dtype=np.float64),
                }
                status = "identity_insufficient_pairs"
            else:
                transforms[handedness] = umeyama_similarity(source, target, args.allow_scale, args.allow_translation)
                status = "estimated_similarity"
            corrected = apply_similarity(source, transforms[handedness]) if len(source) else source
            before = np.linalg.norm(source - target, axis=1) if len(source) else np.asarray([])
            after = np.linalg.norm(corrected - target, axis=1) if len(source) else np.asarray([])
            offsets = np.zeros((21, 3), dtype=np.float64)
            offset_diagnostics = None
            scales = np.ones(len(HAND_BONES), dtype=np.float64)
            bone_diagnostics = None
            if active_bone_scales and len(source):
                scales, bone_diagnostics = estimate_bone_scales(
                    hamer_train,
                    glove_train,
                    handedness,
                    transforms[handedness],
                    args.bone_scale_shrink_k,
                    args.min_bone_scale,
                    args.max_bone_scale,
                )
                corrected_bone = np.concatenate(
                    [
                        apply_bone_scales(apply_similarity(hamer_train[group_id][handedness]["_positions"], transforms[handedness]), scales)[fit_joints]
                        for group_id in groups
                        if handedness in hamer_train.get(group_id, {}) and handedness in glove_train.get(group_id, {})
                    ],
                    axis=0,
                )
                target_bone = np.concatenate(
                    [
                        glove_train[group_id][handedness]["_positions"][fit_joints]
                        for group_id in groups
                        if handedness in hamer_train.get(group_id, {}) and handedness in glove_train.get(group_id, {})
                    ],
                    axis=0,
                )
                if len(corrected_bone):
                    after = np.linalg.norm(corrected_bone - target_bone, axis=1)
            if active_joint_offsets and len(source):
                offsets, offset_diagnostics = estimate_joint_offsets(
                    hamer_train,
                    glove_train,
                    handedness,
                    transforms[handedness],
                    fit_joints,
                    args.joint_offset_shrink_k,
                    args.max_joint_offset_m,
                    scales if active_bone_scales else None,
                )
                if len(source):
                    # Recompute flattened train diagnostics after offset fitting.
                    corrected_with_offsets = []
                    target_with_offsets = []
                    for group_id in groups:
                        hamer_hand = hamer_train[group_id].get(handedness)
                        glove_hand = glove_train[group_id].get(handedness)
                        if hamer_hand is None or glove_hand is None:
                            continue
                        predicted = apply_similarity(hamer_hand["_positions"], transforms[handedness])
                        if active_bone_scales:
                            predicted = apply_bone_scales(predicted, scales)
                        predicted = predicted + offsets
                        corrected_with_offsets.append(predicted[fit_joints])
                        target_with_offsets.append(glove_hand["_positions"][fit_joints])
                    if corrected_with_offsets:
                        corrected_flat = np.concatenate(corrected_with_offsets, axis=0)
                        target_flat = np.concatenate(target_with_offsets, axis=0)
                        after = np.linalg.norm(corrected_flat - target_flat, axis=1)
            joint_offsets[handedness] = offsets
            bone_scales[handedness] = scales
            diagnostics[handedness] = {
                "status": status,
                "train_groups": len(groups),
                "train_points": int(len(source)),
                "fit_joints": fit_joints,
                "before_mean_mm": float(np.mean(before) * 1000.0) if len(before) else None,
                "after_mean_mm": float(np.mean(after) * 1000.0) if len(after) else None,
                "before_p95_mm": float(np.percentile(before, 95) * 1000.0) if len(before) else None,
                "after_p95_mm": float(np.percentile(after, 95) * 1000.0) if len(after) else None,
                "bone_scales": bone_diagnostics,
                "joint_offsets": offset_diagnostics,
            }

    frames = list(iter_jsonl(args.hamer))
    for frame in frames:
        group_id = int(frame.get("group_id", -1))
        if all_group_filter is not None and group_id not in all_group_filter:
            continue
        for hand in frame.get("hands") or []:
            handedness = hand.get("handedness")
            if handedness not in transforms:
                continue
            positions = hand_positions(hand, args.space)
            if positions is None:
                continue
            calibrated = apply_similarity(positions, transforms[handedness])
            if active_bone_scales:
                calibrated = apply_bone_scales(calibrated, bone_scales[handedness])
            if active_joint_offsets:
                calibrated = calibrated + joint_offsets[handedness]
            if args.recenter_wrist_after_transform:
                calibrated = calibrated - calibrated[0:1]
            set_hand_positions(hand, args.space, calibrated, args.write_mode)
            hand["glove_local_calibration"] = {
                "source": str(args.load_calibration_json or args.calibration_json or args.output.with_suffix(".calibration.json")),
                "train_parity": args.train_parity,
                "train_group_range": args.train_group_range,
                "train_group_ids": args.train_group_ids,
                "mode": "similarity",
                "write_mode": args.write_mode,
                "recenter_wrist_after_transform": bool(args.recenter_wrist_after_transform),
                "joint_offsets": args.joint_offsets if args.load_calibration_json is None else loaded_calibration.get("joint_offsets"),
                "bone_scales": args.bone_scales if args.load_calibration_json is None else loaded_calibration.get("bone_scales"),
            }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for frame in frames:
            f.write(json.dumps(frame, ensure_ascii=False, separators=(",", ":")) + "\n")

    calib_path = args.calibration_json or args.output.with_suffix(".calibration.json")
    if args.load_calibration_json is None:
        calib_path.parent.mkdir(parents=True, exist_ok=True)
        with calib_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "type": "hamer_to_glove_local_similarity_calibration",
                    "hamer": str(args.hamer),
                    "glove": str(args.glove),
                    "space": args.space,
                    "group_range": args.group_range,
                    "train_parity": args.train_parity,
                    "train_group_range": args.train_group_range,
                    "train_group_ids": args.train_group_ids,
                    "allow_scale": args.allow_scale,
                    "allow_translation": args.allow_translation,
                    "write_mode": args.write_mode,
                    "recenter_wrist_after_transform": bool(args.recenter_wrist_after_transform),
                    "joint_offsets": args.joint_offsets,
                    "joint_offset_shrink_k": float(args.joint_offset_shrink_k),
                    "max_joint_offset_m": float(args.max_joint_offset_m),
                    "bone_scales": args.bone_scales,
                    "bone_scale_shrink_k": float(args.bone_scale_shrink_k),
                    "min_bone_scale": float(args.min_bone_scale),
                    "max_bone_scale": float(args.max_bone_scale),
                    "bone_edges": [{"parent": parent, "child": child} for parent, child in HAND_BONES],
                    "transforms": {
                        handedness: {
                            "scale": float(transform["scale"]),
                            "rotation": np.asarray(transform["rotation"]).tolist(),
                            "translation": np.asarray(transform["translation"]).tolist(),
                        }
                        for handedness, transform in transforms.items()
                    },
                    "residual_offsets": {
                        handedness: np.asarray(offsets).tolist() for handedness, offsets in joint_offsets.items()
                    },
                    "bone_scales_by_hand": {
                        handedness: np.asarray(scales).tolist() for handedness, scales in bone_scales.items()
                    },
                    "diagnostics": diagnostics,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
            f.write("\n")
    print(f"wrote: {args.output}")
    print(f"calibration: {args.load_calibration_json or calib_path}")
    print(json.dumps(diagnostics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
