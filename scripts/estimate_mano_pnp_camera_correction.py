#!/usr/bin/env python3
"""Estimate conservative per-camera SE(3) corrections from MANO+2D PnP pairs.

The result is deliberately written as a separate JSON artifact. It never
overwrites the calibration YAML; a correction should only be consumed after its
holdout diagnostics show a genuine cross-view consistency improvement.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml

from hamer_multiview_utils import iter_jsonl, parse_group_ids, range_suffix
from progress_utils import tqdm


WRIST_CAM_ROOT = Path("/home/luojiangrui/ljr/wrist_cam")
DEFAULT_HAMER_ROOT = WRIST_CAM_ROOT / "third_party" / "hamer"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", action="append", type=Path)
    parser.add_argument("--predictions-glob", default="video/sam3_hamer_left_index/hamer_per_view/hamer_predictions_*.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hamer-root", type=Path, default=DEFAULT_HAMER_ROOT)
    parser.add_argument("--checkpoint")
    parser.add_argument("--calib", type=Path, required=True)
    parser.add_argument("--group-range")
    parser.add_argument("--group-ids")
    parser.add_argument("--reference-cameras", default="Left:C1,Right:C2")
    parser.add_argument("--min-2d-conf", type=float, default=0.25)
    parser.add_argument("--max-reprojection-error-px", type=float, default=8.0)
    parser.add_argument("--max-translation-residual-m", type=float, default=0.12)
    parser.add_argument("--max-rotation-residual-deg", type=float, default=15.0)
    parser.add_argument("--min-inliers", type=int, default=40)
    parser.add_argument("--max-observations", type=int, default=0, help="0 means no cap.")
    parser.add_argument(
        "--evaluate-correction",
        type=Path,
        help="Existing correction JSON to evaluate on this independent frame range. Does not alter cameras.yaml.",
    )
    parser.add_argument("--progress-position", type=int, default=int(os.environ.get("TQDM_POSITION", "0")))
    return parser.parse_args()


def parse_references(value: str) -> dict[str, str]:
    out = {"Left": "C1", "Right": "C2"}
    for part in value.split(","):
        if not part.strip():
            continue
        hand, camera = part.split(":", 1)
        hand, camera = hand.strip(), camera.strip()
        if hand not in {"Left", "Right"} or not camera:
            raise SystemExit(f"Invalid --reference-cameras item: {part}")
        out[hand] = camera
    return out


def resolve_checkpoint(root: Path, checkpoint: str | None) -> Path:
    if checkpoint:
        path = Path(checkpoint).expanduser()
        return path if path.is_absolute() else root / path
    return root / "_DATA" / "hamer_ckpts" / "checkpoints" / "hamer.ckpt"


def as_array(value: Any, shape: tuple[int, ...]) -> np.ndarray | None:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != shape or not np.all(np.isfinite(arr)):
        return None
    return arr


def candidate_score(record: dict[str, Any]) -> tuple[float, float, float]:
    mask = float(record.get("mask_score", -1.0)) if isinstance(record.get("mask_score"), (int, float)) else -1.0
    confidence = float(record.get("handedness_confidence", 0.0)) if isinstance(record.get("handedness_confidence"), (int, float)) else 0.0
    known = 1.0 if record.get("hypothesis_status") == "known" else 0.0
    return mask, known, confidence


def load_records(paths: list[Path], group_ids: set[int] | None) -> dict[int, dict[str, dict[str, dict[str, Any]]]]:
    grouped: dict[int, dict[str, dict[str, list[dict[str, Any]]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for path in paths:
        for record in iter_jsonl(path):
            if record.get("type") != "hamer_multiview_prediction":
                continue
            group_id = int(record.get("group_id", -1))
            if group_ids is not None and group_id not in group_ids:
                continue
            hand, camera = record.get("handedness"), record.get("camera_id")
            if hand in {"Left", "Right"} and camera:
                grouped[group_id][hand][str(camera)].append(record)
    selected: dict[int, dict[str, dict[str, dict[str, Any]]]] = {}
    for group_id, by_hand in grouped.items():
        selected[group_id] = {}
        for hand, by_camera in by_hand.items():
            selected[group_id][hand] = {camera: max(items, key=candidate_score) for camera, items in by_camera.items()}
    return selected


def mano_local_joints(mano: Any, record: dict[str, Any], device: torch.device) -> np.ndarray | None:
    params = record.get("mano_params_rotmat") or {}
    pose = as_array(params.get("hand_pose"), (15, 3, 3))
    betas = as_array(params.get("betas"), (10,))
    if pose is None or betas is None:
        return None
    is_right = int(record.get("is_right", 1 if record.get("handedness") == "Right" else 0))
    with torch.no_grad():
        pose_t = torch.tensor(pose[None], dtype=torch.float32, device=device)
        beta_t = torch.tensor(betas[None], dtype=torch.float32, device=device)
        orient = torch.eye(3, dtype=torch.float32, device=device).reshape(1, 1, 3, 3)
        output = mano(global_orient=orient, hand_pose=pose_t, betas=beta_t, pose2rot=False)
        joints = output.joints[0, :21].detach().cpu().numpy().astype(np.float64)
    joints[:, 0] *= float(2 * is_right - 1)
    return joints - joints[0:1]


def solve_pnp(local_joints: np.ndarray, record: dict[str, Any], min_conf: float, max_error: float) -> np.ndarray | None:
    image = np.asarray(record.get("hamer_joints_2d_rectified_px"), dtype=np.float64)
    k = np.asarray(record.get("rectified_K"), dtype=np.float64)
    confidence = np.asarray(record.get("hamer_joints_2d_conf"), dtype=np.float64)
    if image.shape[0] < 21 or k.shape != (3, 3):
        return None
    if confidence.shape[0] < 21:
        confidence = np.ones(21, dtype=np.float64)
    valid = np.isfinite(image[:21]).all(axis=1) & np.isfinite(local_joints[:21]).all(axis=1) & (confidence[:21] >= min_conf)
    if int(valid.sum()) < 6:
        return None
    obj = local_joints[:21][valid].reshape(-1, 1, 3)
    img = image[:21][valid].reshape(-1, 1, 2)
    try:
        ok, rvec, tvec = cv2.solvePnP(obj, img, k, None, flags=cv2.SOLVEPNP_EPNP)
        if not ok:
            return None
        ok, rvec, tvec = cv2.solvePnP(obj, img, k, None, rvec, tvec, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok or not np.isfinite(tvec).all() or float(tvec.reshape(-1)[2]) <= 1e-4:
            return None
    except cv2.error:
        return None
    r, _ = cv2.Rodrigues(rvec)
    projected = (k @ np.concatenate([r, tvec], axis=1) @ np.concatenate([obj[:, 0], np.ones((len(obj), 1))], axis=1).T).T
    projected = projected[:, :2] / np.maximum(projected[:, 2:3], 1e-8)
    error = float(np.mean(np.linalg.norm(projected - img[:, 0], axis=1)))
    if not np.isfinite(error) or error > max_error:
        return None
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = r
    transform[:3, 3] = tvec.reshape(3)
    return transform


def rotation_distance_deg(a: np.ndarray, b: np.ndarray) -> float:
    relative = a @ b.T
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def robust_delta(transforms: list[np.ndarray], args: argparse.Namespace) -> tuple[np.ndarray | None, np.ndarray]:
    if not transforms:
        return None, np.zeros(0, dtype=bool)
    rvecs = np.stack([cv2.Rodrigues(item[:3, :3])[0].reshape(3) for item in transforms])
    translations = np.stack([item[:3, 3] for item in transforms])
    median_rvec = np.median(rvecs, axis=0)
    median_translation = np.median(translations, axis=0)
    r_med, _ = cv2.Rodrigues(median_rvec.reshape(3, 1))
    rot_error = np.asarray([rotation_distance_deg(item[:3, :3], r_med) for item in transforms])
    trans_error = np.linalg.norm(translations - median_translation.reshape(1, 3), axis=1)
    keep = (rot_error <= args.max_rotation_residual_deg) & (trans_error <= args.max_translation_residual_m)
    if int(keep.sum()) < args.min_inliers:
        return None, keep
    rvec = np.median(rvecs[keep], axis=0)
    translation = np.median(translations[keep], axis=0)
    rotation, _ = cv2.Rodrigues(rvec.reshape(3, 1))
    delta = np.eye(4, dtype=np.float64)
    delta[:3, :3] = rotation
    delta[:3, 3] = translation
    return delta, keep


def load_corrections(path: Path | None) -> dict[str, np.ndarray]:
    if path is None:
        return {}
    if not path.exists():
        raise SystemExit(f"Correction report not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8")).get("camera_corrections", {})
    result: dict[str, np.ndarray] = {}
    for camera, item in raw.items():
        value = item.get("delta_T_H") if isinstance(item, dict) else None
        matrix = np.asarray(value, dtype=np.float64)
        if matrix.shape == (4, 4) and np.all(np.isfinite(matrix)):
            result[str(camera)] = matrix
    return result


def summary(values: list[float]) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(array)),
        "median": float(np.median(array)) if len(array) else None,
        "p95": float(np.percentile(array, 95)) if len(array) else None,
        "mean": float(np.mean(array)) if len(array) else None,
    }


def cross_view_pose_consistency(
    samples: list[dict[str, Any]],
    camera: str,
    raw_transforms: dict[str, np.ndarray],
    corrected_transforms: dict[str, np.ndarray],
) -> dict[str, Any]:
    before_wrist, after_wrist = [], []
    before_rotation, after_rotation = [], []
    for sample in samples:
        reference = sample["reference_camera"]
        reference_pose = sample["reference_pose"]
        camera_pose = sample["camera_pose"]
        raw_reference = raw_transforms[reference] @ reference_pose
        raw_camera = raw_transforms[camera] @ camera_pose
        corrected_reference = corrected_transforms[reference] @ reference_pose
        corrected_camera = corrected_transforms[camera] @ camera_pose
        before_wrist.append(float(np.linalg.norm(raw_reference[:3, 3] - raw_camera[:3, 3])))
        after_wrist.append(float(np.linalg.norm(corrected_reference[:3, 3] - corrected_camera[:3, 3])))
        before_rotation.append(rotation_distance_deg(raw_reference[:3, :3], raw_camera[:3, :3]))
        after_rotation.append(rotation_distance_deg(corrected_reference[:3, :3], corrected_camera[:3, :3]))
    return {
        "wrist_distance_m": {"before": summary(before_wrist), "after": summary(after_wrist)},
        "rotation_distance_deg": {"before": summary(before_rotation), "after": summary(after_rotation)},
    }


def main() -> None:
    args = parse_args()
    group_ids = parse_group_ids(args.group_range, args.group_ids)
    paths = args.predictions or [Path(item) for item in sorted(glob.glob(args.predictions_glob))]
    if not paths:
        raise SystemExit("No prediction JSONL files found")
    references = parse_references(args.reference_cameras)
    calib = yaml.safe_load(args.calib.read_text(encoding="utf-8"))["cameras"]
    base_transforms = {camera: np.asarray(info["T_H_C"], dtype=np.float64) for camera, info in calib.items()}

    args.hamer_root = args.hamer_root.expanduser().resolve()
    checkpoint = resolve_checkpoint(args.hamer_root, args.checkpoint)
    if not checkpoint.exists():
        raise SystemExit(f"HaMeR checkpoint not found: {checkpoint}")
    sys.path.insert(0, str(args.hamer_root))
    from hamer.models import load_hamer  # type: ignore

    cwd = Path.cwd()
    os.chdir(args.hamer_root)
    try:
        model, _cfg = load_hamer(str(checkpoint))
    finally:
        os.chdir(cwd)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mano = model.mano.to(device).eval()
    for parameter in mano.parameters():
        parameter.requires_grad_(False)

    records = load_records(paths, group_ids)
    poses: dict[tuple[int, str, str], np.ndarray] = {}
    for group_id in tqdm(sorted(records), desc="MANO PnP poses", unit="frame", position=args.progress_position):
        for hand, by_camera in records[group_id].items():
            for camera, record in by_camera.items():
                if camera not in base_transforms:
                    continue
                joints = mano_local_joints(mano, record, device)
                if joints is None:
                    continue
                pose = solve_pnp(joints, record, args.min_2d_conf, args.max_reprojection_error_px)
                if pose is not None:
                    poses[(group_id, hand, camera)] = pose

    observations: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for group_id, by_hand in records.items():
        for hand, by_camera in by_hand.items():
            reference = references[hand]
            reference_pose = poses.get((group_id, hand, reference))
            if reference_pose is None or reference not in base_transforms:
                continue
            h_hand = base_transforms[reference] @ reference_pose
            for camera in by_camera:
                if camera == reference or camera not in base_transforms:
                    continue
                camera_pose = poses.get((group_id, hand, camera))
                if camera_pose is None:
                    continue
                estimated_h_c = h_hand @ np.linalg.inv(camera_pose)
                delta = estimated_h_c @ np.linalg.inv(base_transforms[camera])
                observations[camera].append(
                    {
                        "delta": delta,
                        "handedness": hand,
                        "reference_camera": reference,
                        "reference_pose": reference_pose,
                        "camera_pose": camera_pose,
                    }
                )

    corrections: dict[str, Any] = {}
    fitted_deltas: dict[str, np.ndarray] = {}
    for camera, samples in sorted(observations.items()):
        items = [sample["delta"] for sample in samples]
        if args.max_observations > 0:
            items = items[: args.max_observations]
        delta, keep = robust_delta(items, args)
        if delta is not None:
            fitted_deltas[camera] = delta
        corrections[camera] = {
            "status": "estimated" if delta is not None else "insufficient_consistent_inliers",
            "delta_T_H": None if delta is None else delta.tolist(),
            "inliers": int(keep.sum()),
            "observations": len(items),
            "rotation_residual_deg": None if delta is None else float(np.median([rotation_distance_deg(item[:3, :3], delta[:3, :3]) for item in items])),
            "translation_residual_m": None if delta is None else float(np.median([np.linalg.norm(item[:3, 3] - delta[:3, 3]) for item in items])),
        }

    external_deltas = load_corrections(args.evaluate_correction)
    active_deltas = external_deltas if external_deltas else fitted_deltas
    corrected_transforms = dict(base_transforms)
    for camera, delta in active_deltas.items():
        if camera in corrected_transforms:
            corrected_transforms[camera] = delta @ corrected_transforms[camera]

    for camera, samples in sorted(observations.items()):
        by_hand = {
            hand: cross_view_pose_consistency(
                [sample for sample in samples if sample["handedness"] == hand],
                camera,
                base_transforms,
                corrected_transforms,
            )
            for hand in ("Left", "Right")
            if any(sample["handedness"] == hand for sample in samples)
        }
        corrections[camera]["cross_view_pose_consistency"] = {
            "evaluation_mode": "external_holdout" if external_deltas else "in_sample_fit",
            "reference_cameras": sorted({str(sample["reference_camera"]) for sample in samples}),
            **cross_view_pose_consistency(samples, camera, base_transforms, corrected_transforms),
            "by_handedness": by_hand,
        }
        if external_deltas:
            corrections[camera]["evaluated_delta_source"] = (
                "external" if camera in external_deltas else "identity_missing_from_external_report"
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "type": "mano_pnp_camera_correction",
        "prediction_files": [str(path) for path in paths],
        "calib": str(args.calib),
        "reference_cameras": references,
        "camera_corrections": corrections,
        "active_correction_source": str(args.evaluate_correction) if args.evaluate_correction else "fitted_on_this_range",
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    # `argparse` may contain Path lists (for repeated --predictions).  Keep the
    # report robust to those metadata-only values instead of failing after PnP.
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    print(f"wrote: {args.output}")
    for camera, item in corrections.items():
        print(f"{camera}: {item['status']} inliers={item['inliers']}/{item['observations']}")


if __name__ == "__main__":
    main()
