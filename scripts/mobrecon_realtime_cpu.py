#!/usr/bin/env python3
"""Run sparse-box tracking and MobRecon as one low-latency CPU pipeline."""

from __future__ import annotations

import argparse
import itertools
import json
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import torch

from fuse_hamer_jobs import best_handedness_from_sam3
from fuse_hamer_palm_local import candidate_from_record, fuse_views, lowpass_alpha
from hamer_multiview_utils import (
    DEFAULT_BASE_DIR,
    DEFAULT_FRAMES,
    expand_bbox,
    filter_frame_records,
    parse_cameras,
    parse_group_ids,
    range_suffix,
    rectified_rel_path,
)
from mobrecon_multiview_worker import (
    DEFAULT_MOBRECON_ROOT,
    choose_device,
    load_model,
    prediction_record,
    prepare_sample,
    restore_outputs,
)
from progress_utils import tqdm
from track_sam3_sparse_keyframes import load_hands, match_refresh, propagate_state, scaled_gray, state_to_hand


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=Path, default=DEFAULT_FRAMES)
    parser.add_argument("--rectified-dir", type=Path, default=DEFAULT_BASE_DIR / "rectified_for_hamer")
    parser.add_argument("--keyframe-sam3", type=Path)
    parser.add_argument("--keyframe-shard", action="append", type=Path)
    parser.add_argument("--follow-keyframes", action="store_true")
    parser.add_argument("--keyframe-timeout", type=float, default=600.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_BASE_DIR / "mobrecon_realtime_cpu")
    parser.add_argument("--mobrecon-root", type=Path, default=DEFAULT_MOBRECON_ROOT)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--cameras", default="C0,C2,C3")
    parser.add_argument("--group-range")
    parser.add_argument("--group-ids")
    parser.add_argument("--keyframe-stride", type=int, default=10)
    parser.add_argument("--flow-scale", type=float, default=0.25)
    parser.add_argument("--max-corners", type=int, default=60)
    parser.add_argument("--min-corners", type=int, default=8)
    parser.add_argument("--quality-level", type=float, default=0.01)
    parser.add_argument("--min-distance", type=float, default=3.0)
    parser.add_argument("--max-flow-error", type=float, default=25.0)
    parser.add_argument("--max-forward-backward-error", type=float, default=0.0)
    parser.add_argument("--max-hands", type=int, default=2)
    parser.add_argument("--bbox-pad", type=float, default=1.15)
    parser.add_argument("--crop-scale", type=float, default=1.5)
    parser.add_argument("--input-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--microbatch-groups", type=int, default=1)
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument("--one-euro-min-cutoff", type=float, default=0.25)
    parser.add_argument("--one-euro-beta", type=float, default=0.05)
    parser.add_argument("--one-euro-derivative-cutoff", type=float, default=1.0)
    parser.add_argument("--frame-rate", type=float, default=25.0)
    parser.add_argument(
        "--handedness-switch-confirm-keyframes",
        type=int,
        default=2,
        help="Require this many conflicting semantic keyframes before changing an established track side.",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    parser.add_argument("--precision", choices=["float32", "float16"], default="float32")
    parser.add_argument("--export-vertices", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--write-intermediates", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--progress-position", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.keyframe_stride < 1 or not 0.0 < args.flow_scale <= 1.0:
        raise SystemExit("keyframe stride must be positive and flow scale must be in (0, 1]")
    if args.batch_size < 1 or args.microbatch_groups < 1 or args.input_size < 2:
        raise SystemExit("batch sizes must be positive and input size must be at least 2")
    if args.min_corners < 2 or args.max_corners < args.min_corners:
        raise SystemExit("corner limits are invalid")
    if args.max_flow_error <= 0.0 or args.max_forward_backward_error < 0.0:
        raise SystemExit("max flow error must be positive and forward-backward error must be non-negative")
    if args.bbox_pad <= 0.0 or args.crop_scale <= 0.0:
        raise SystemExit("bbox and crop scales must be positive")
    if (
        args.one_euro_min_cutoff < 0.0
        or args.one_euro_beta < 0.0
        or args.one_euro_derivative_cutoff <= 0.0
        or args.frame_rate <= 0.0
    ):
        raise SystemExit("One-Euro cutoffs/frame rate must be positive and beta must be non-negative")
    source_count = int(args.keyframe_sam3 is not None) + int(bool(args.keyframe_shard))
    if source_count != 1:
        raise SystemExit("provide exactly one --keyframe-sam3 or one or more --keyframe-shard paths")
    if args.follow_keyframes and not args.keyframe_shard:
        raise SystemExit("--follow-keyframes requires --keyframe-shard")
    if args.keyframe_timeout <= 0.0:
        raise SystemExit("--keyframe-timeout must be positive")
    if args.handedness_switch_confirm_keyframes < 1:
        raise SystemExit("--handedness-switch-confirm-keyframes must be positive")


def group_frame_records(records: list[dict[str, Any]]) -> list[tuple[int, dict[str, dict[str, Any]]]]:
    grouped: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        grouped[int(record["group_id"])][str(record["camera_id"])] = record
    return sorted(grouped.items())


class KeyframeSource:
    def get(self, group_id: int, camera_id: str) -> list[dict[str, Any]]:
        raise NotImplementedError


class StaticKeyframeSource(KeyframeSource):
    def __init__(self, paths: list[Path], cameras: set[str], group_ids: set[int] | None) -> None:
        self.hands: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
        for path in paths:
            loaded = load_hands(path, cameras, group_ids)
            for key, values in loaded.items():
                self.hands[key].extend(values)

    def get(self, group_id: int, camera_id: str) -> list[dict[str, Any]]:
        return self.hands.get((group_id, camera_id), [])


class FollowingKeyframeSource(KeyframeSource):
    def __init__(self, paths: list[Path], timeout: float) -> None:
        self.paths = paths
        self.timeout = timeout
        self.files: dict[Path, Any] = {}
        self.records: dict[tuple[int, str], list[dict[str, Any]]] = {}

    def read_available(self) -> int:
        added = 0
        for path in self.paths:
            if path not in self.files:
                if not path.exists():
                    continue
                self.files[path] = path.open("r", encoding="utf-8")
            source = self.files[path]
            while True:
                position = source.tell()
                line = source.readline()
                if not line:
                    source.seek(position)
                    break
                if not line.endswith("\n"):
                    source.seek(position)
                    break
                record = json.loads(line)
                key = (int(record["group_id"]), str(record["camera_id"]))
                self.records[key] = [dict(hand) for hand in record.get("hands") or []]
                added += 1
        return added

    def get(self, group_id: int, camera_id: str) -> list[dict[str, Any]]:
        key = (group_id, camera_id)
        started = time.perf_counter()
        while key not in self.records:
            self.read_available()
            if key in self.records:
                break
            if time.perf_counter() - started > self.timeout:
                raise TimeoutError(f"timed out waiting for SAM3 keyframe group={group_id} camera={camera_id}")
            time.sleep(0.01)
        return self.records.pop(key)

    def close(self) -> None:
        for source in self.files.values():
            source.close()


def tracker_state() -> dict[str, Any]:
    return {"states": {}, "next_track_id": 1, "previous_gray": None, "sequence_index": 0}


def track_image(
    image: np.ndarray,
    group_id: int,
    camera_id: str,
    detections: list[dict[str, Any]],
    state: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], bool, int]:
    gray = scaled_gray(image, args.flow_scale)
    height, width = image.shape[:2]
    scheduled_keyframe = state["sequence_index"] % args.keyframe_stride == 0 or state["previous_gray"] is None
    is_keyframe = bool(scheduled_keyframe and detections)
    tracking_failures = 0
    if is_keyframe:
        state["states"], state["next_track_id"] = match_refresh(
            detections,
            state["states"],
            gray,
            args,
            state["next_track_id"],
        )
    elif state["previous_gray"] is not None:
        propagated = {}
        for track_id, track in state["states"].items():
            updated = propagate_state(track, state["previous_gray"], gray, width, height, args)
            if updated is None:
                tracking_failures += 1
                continue
            propagated[track_id] = updated
        state["states"] = propagated
    hands = [state_to_hand(track, is_keyframe, group_id, camera_id) for track in state["states"].values()]
    hands.sort(key=lambda hand: int(hand["track_numeric_id"]))
    state["previous_gray"] = gray
    state["sequence_index"] += 1
    return hands[: args.max_hands], is_keyframe, tracking_failures


def hand_job(
    hand: dict[str, Any], image_path: Path, group_id: int, camera_id: str, job_index: int, bbox_pad: float
) -> dict[str, Any]:
    handedness, handedness_source, handedness_score = best_handedness_from_sam3(hand)
    return {
        "job_index": job_index,
        "group_id": group_id,
        "camera_id": camera_id,
        "rectified_image_path": str(image_path),
        "hamer_frame_path": str(image_path),
        "used_mask_blur": False,
        "bbox_rectified_px": expand_bbox(hand["bbox"], bbox_pad),
        "handedness": handedness or "unknown",
        "handedness_source": handedness_source or "sam3_only_unknown",
        "handedness_score": handedness_score,
        "is_right": int(handedness == "Right") if handedness in {"Left", "Right"} else None,
        "source_detector": "sam3",
        "sam3_mask_path": hand.get("mask_path"),
        "sam3_score": hand.get("score"),
        "track_id": hand.get("track_id"),
        "locked_handedness": hand.get("locked_handedness"),
        "handedness_confidence": hand.get("handedness_confidence"),
        "track_is_keyframe": bool(hand.get("is_keyframe")),
        "debug_only": False,
    }


def assign_online_handedness(
    jobs: list[dict[str, Any]],
    track_handedness: dict[tuple[str, str], str],
    track_conflicts: dict[tuple[str, str], tuple[str, int]] | None = None,
    switch_confirm_keyframes: int = 2,
) -> tuple[list[dict[str, Any]], int]:
    if track_conflicts is None:
        track_conflicts = {}
    for handedness in ("Left", "Right"):
        same_side = [job for job in jobs if str(job.get("handedness") or "").title() == handedness]
        if len(same_side) <= 1:
            continue

        def evidence_key(job: dict[str, Any]) -> tuple[int, float, int]:
            track_id = job.get("track_id")
            track_key = (str(job["camera_id"]), str(track_id)) if track_id is not None else None
            inherited = track_handedness.get(track_key) if track_key is not None else None
            confidence = job.get("handedness_score")
            confidence_value = float(confidence) if isinstance(confidence, (int, float)) else -1.0
            return int(inherited == handedness), confidence_value, -int(job.get("job_index", 0))

        strongest = max(same_side, key=evidence_key)
        for job in same_side:
            if job is strongest:
                continue
            job["observed_handedness"] = handedness
            job["observed_handedness_source"] = job.get("handedness_source")
            job["handedness"] = "unknown"
            job["handedness_source"] = "online_same_side_collision"
            job["is_right"] = None
    unresolved = []
    for job in jobs:
        handedness = str(job.get("handedness") or "").title()
        track_id = job.get("track_id")
        track_key = (str(job["camera_id"]), str(track_id)) if track_id is not None else None
        if handedness in {"Left", "Right"}:
            inherited = track_handedness.get(track_key) if track_key is not None else None
            if inherited is not None and inherited != handedness:
                observed_source = job.get("handedness_source")
                confirmed = False
                if bool(job.get("track_is_keyframe")):
                    previous_side, previous_count = track_conflicts.get(track_key, ("", 0))
                    conflict_count = previous_count + 1 if previous_side == handedness else 1
                    track_conflicts[track_key] = (handedness, conflict_count)
                    confirmed = conflict_count >= switch_confirm_keyframes
                if not confirmed:
                    job["observed_handedness"] = handedness
                    job["observed_handedness_source"] = observed_source
                    job["handedness"] = inherited
                    job["handedness_source"] = "online_track_conflict_hold"
                    job["is_right"] = int(inherited == "Right")
                    continue
            if track_key is not None:
                track_handedness[track_key] = handedness
                track_conflicts.pop(track_key, None)
        else:
            unresolved.append(job)
    for job in unresolved:
        track_id = job.get("track_id")
        inherited = track_handedness.get((str(job["camera_id"]), str(track_id))) if track_id is not None else None
        if inherited:
            job["handedness"] = inherited
            job["handedness_source"] = "online_track_inherit"
            job["is_right"] = int(inherited == "Right")
    resolved = [job for job in jobs if str(job.get("handedness") or "").title() in {"Left", "Right"}]
    ambiguous = [job for job in jobs if job not in resolved]
    for job in ambiguous:
        hypothesis_id = f"{job['camera_id']}:{job['group_id']}:{job.get('track_id')}:{job['job_index']}"
        for handedness in ("Left", "Right"):
            hypothesis = dict(job)
            hypothesis["handedness"] = handedness
            hypothesis["handedness_source"] = "online_dual_hypothesis"
            hypothesis["is_right"] = int(handedness == "Right")
            hypothesis["ambiguous_handedness"] = True
            hypothesis["handedness_hypothesis_id"] = hypothesis_id
            resolved.append(hypothesis)
    return resolved, len(ambiguous)


def fusion_args() -> SimpleNamespace:
    return SimpleNamespace(
        image_width=1600,
        image_height=1200,
        quality_mask_weight=0.55,
        quality_bbox_weight=0.15,
        quality_edge_weight=0.12,
        quality_source_bonus=0.06,
        quality_known_bonus=0.05,
        include_vertices=False,
    )


def selected_camera_candidates(
    candidates: dict[str, dict[str, list[dict[str, Any]]]], handedness: str
) -> list[dict[str, Any]]:
    return [
        max(values, key=lambda item: item["quality_score"])
        for values in candidates.get(handedness, {}).values()
        if values
    ]


def handedness_assignment_cost(
    candidates: dict[str, dict[str, list[dict[str, Any]]]], shape_history: dict[str, np.ndarray]
) -> tuple[float, int]:
    cost = 0.0
    active_hands = 0
    for handedness in ("Left", "Right"):
        items = selected_camera_candidates(candidates, handedness)
        if not items:
            continue
        active_hands += 1
        stacked = np.stack([item["joints"] for item in items], axis=0)
        mean_shape = np.mean(stacked, axis=0)
        if len(items) >= 2:
            cost += float(np.mean(np.linalg.norm(stacked - mean_shape[None, :, :], axis=2)))
        history = shape_history.get(handedness)
        if history is not None:
            cost += 0.35 * float(np.mean(np.linalg.norm(mean_shape - history, axis=1)))
    cost += 0.02 * max(0, active_hands - 1)
    return cost, active_hands


def select_ambiguous_hypotheses(
    candidates: dict[str, dict[str, list[dict[str, Any]]]],
    ambiguous: dict[str, list[tuple[str, dict[str, Any]]]],
    shape_history: dict[str, np.ndarray],
) -> list[tuple[str, dict[str, Any]]]:
    if not ambiguous:
        return []
    hypothesis_ids = sorted(ambiguous)
    option_sets = [
        sorted(ambiguous[hypothesis_id], key=lambda item: (item[0] != "Right", item[0]))
        for hypothesis_id in hypothesis_ids
    ]
    base = {
        handedness: {camera_id: list(values) for camera_id, values in candidates.get(handedness, {}).items()}
        for handedness in ("Left", "Right")
    }
    best_selection: list[tuple[str, dict[str, Any]]] | None = None
    best_key: tuple[float, int] | None = None
    for selection in itertools.product(*option_sets):
        trial = {
            handedness: {camera_id: list(values) for camera_id, values in base[handedness].items()}
            for handedness in ("Left", "Right")
        }
        occupied = {
            (handedness, camera_id)
            for handedness in ("Left", "Right")
            for camera_id, values in trial[handedness].items()
            if values
        }
        valid = True
        for handedness, candidate in selection:
            key = (handedness, candidate["camera_id"])
            if key in occupied:
                valid = False
                break
            occupied.add(key)
            trial[handedness].setdefault(candidate["camera_id"], []).append(candidate)
        if not valid:
            continue
        key = handedness_assignment_cost(trial, shape_history)
        if best_key is None or key < best_key:
            best_key = key
            best_selection = list(selection)
    if best_selection is not None:
        return best_selection
    return [options[0] for options in option_sets if options]


def fuse_group(
    group_id: int,
    predictions: list[dict[str, Any]],
    args: SimpleNamespace,
    shape_history: dict[str, np.ndarray],
    track_handedness: dict[tuple[str, str], str],
) -> tuple[dict[str, Any], int]:
    candidates: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    ambiguous: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for prediction in predictions:
        candidate = candidate_from_record(prediction, args)
        if candidate is None:
            continue
        hypothesis_id = prediction.get("handedness_hypothesis_id")
        handedness = str(prediction["handedness"])
        if prediction.get("ambiguous_handedness") and hypothesis_id:
            ambiguous[str(hypothesis_id)].append((handedness, candidate))
        else:
            candidates[handedness][str(prediction["camera_id"])].append(candidate)
    selected = select_ambiguous_hypotheses(candidates, ambiguous, shape_history)
    for handedness, candidate in selected:
        candidates[handedness][candidate["camera_id"]].append(candidate)
        track_id = candidate["record"].get("track_id")
        if track_id is not None:
            track_handedness[(candidate["camera_id"], str(track_id))] = handedness
    selected_hypotheses = len(selected)
    hands = []
    for handedness in ("Left", "Right"):
        items = selected_camera_candidates(candidates, handedness)
        if not items:
            continue
        items.sort(key=lambda item: item["camera_id"])
        fused = fuse_views(items, None, 0.0, False)
        shape_history[handedness] = fused["raw_joints"].copy()
        hands.append(
            {
                "group_id": group_id,
                "handedness": handedness,
                "mode": "zero_shot_multiview_mean:raw",
                "metric_valid": len(items) >= 2,
                "local_shape_valid": True,
                "fusion_view_count": len(items),
                "used_cameras": [item["camera_id"] for item in items],
                "source_models": sorted({item["model_name"] for item in items}),
                "palm_local_joints_m": fused["raw_joints"].tolist(),
                "raw_palm_local_joints_m": fused["raw_joints"].tolist(),
                "static_calibrated_palm_local_joints_m": fused["raw_joints"].tolist(),
                "smoothed_palm_local_joints_m": None,
                "causal_smoothed_palm_local_joints_m": None,
                "adaptive_causal_palm_local_joints_m": None,
                "palm_local_vertices_m": None,
                "joint_consensus_std_m": fused["joint_std_m"].tolist(),
                "mean_consensus_error_m": fused["mean_consensus_error_m"],
                "p95_consensus_error_m": fused["p95_consensus_error_m"],
                "per_camera_consensus_error_m": fused["camera_errors_m"],
                "per_camera_quality_scores": {item["camera_id"]: item["quality_score"] for item in items},
                "per_camera_quality_parts": {item["camera_id"]: item["quality_parts"] for item in items},
                "bone_calibration_blend": 0.0,
                "temporal_radius": 0,
                "temporal_sigma": None,
                "causal_ema_alpha": None,
                "one_euro_min_cutoff": None,
                "one_euro_beta": None,
                "one_euro_derivative_cutoff": None,
                "frame_rate": None,
                "primary_output_source": "raw",
            }
        )
    return {"type": "hamer_palm_local_fused_frame", "group_id": group_id, "hands": hands}, selected_hypotheses


def apply_one_euro(
    fused_record: dict[str, Any],
    filtered_state: dict[str, np.ndarray],
    raw_state: dict[str, np.ndarray],
    derivative_state: dict[str, np.ndarray],
    previous_group: dict[str, int],
    args: argparse.Namespace,
) -> None:
    if args.one_euro_min_cutoff <= 0.0:
        return
    group_id = int(fused_record["group_id"])
    derivative_alpha = float(lowpass_alpha(args.one_euro_derivative_cutoff, args.frame_rate))
    for hand in fused_record["hands"]:
        handedness = str(hand["handedness"])
        current = np.asarray(hand["raw_palm_local_joints_m"], dtype=np.float64)
        contiguous = previous_group.get(handedness) == group_id - 1
        if handedness not in filtered_state or not contiguous:
            filtered = current.copy()
            derivative = np.zeros_like(current)
        else:
            raw_derivative = (current - raw_state[handedness]) * args.frame_rate
            derivative = (
                derivative_alpha * raw_derivative
                + (1.0 - derivative_alpha) * derivative_state[handedness]
            )
            joint_speed = np.linalg.norm(derivative, axis=1, keepdims=True)
            cutoff = args.one_euro_min_cutoff + args.one_euro_beta * joint_speed
            alpha = lowpass_alpha(cutoff, args.frame_rate)
            filtered = alpha * current + (1.0 - alpha) * filtered_state[handedness]
        filtered_state[handedness] = filtered
        raw_state[handedness] = current
        derivative_state[handedness] = derivative
        previous_group[handedness] = group_id
        filtered_list = filtered.tolist()
        hand["mode"] = "zero_shot_multiview_mean:adaptive-causal"
        hand["palm_local_joints_m"] = filtered_list
        hand["adaptive_causal_palm_local_joints_m"] = filtered_list
        hand["one_euro_min_cutoff"] = args.one_euro_min_cutoff
        hand["one_euro_beta"] = args.one_euro_beta
        hand["one_euro_derivative_cutoff"] = args.one_euro_derivative_cutoff
        hand["frame_rate"] = args.frame_rate
        hand["primary_output_source"] = "adaptive-causal"


def percentile(values: list[float], quantile: float) -> float | None:
    return float(np.percentile(np.asarray(values, dtype=np.float64), quantile)) if values else None


def main() -> None:
    args = parse_args()
    validate_args(args)
    torch.set_num_threads(max(1, args.torch_threads))
    cameras = sorted(parse_cameras(args.cameras))
    group_ids = parse_group_ids(args.group_range, args.group_ids)
    suffix = range_suffix(group_ids)
    frame_records = filter_frame_records(args.frames, set(cameras), group_ids)
    grouped_frames = group_frame_records(frame_records)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "groups": len(grouped_frames),
                    "records": len(frame_records),
                    "cameras": cameras,
                    "keyframe_stride": args.keyframe_stride,
                    "microbatch_groups": args.microbatch_groups,
                },
                indent=2,
            )
        )
        return
    keyframe_paths = [args.keyframe_sam3] if args.keyframe_sam3 is not None else list(args.keyframe_shard or [])
    if not args.follow_keyframes:
        missing_keyframes = [path for path in keyframe_paths if not path.exists()]
        if missing_keyframes:
            raise SystemExit(f"keyframe SAM3 file not found: {missing_keyframes[0]}")
    checkpoint = (args.checkpoint or (args.mobrecon_root / "pretrained" / "mobrecon_densestack.pt")).expanduser().resolve()
    output_paths = {
        "tracks": args.output_dir / f"sam3_sparse_tracks_{suffix}.jsonl",
        "jobs": args.output_dir / f"hamer_jobs_{suffix}.jsonl",
        "predictions": args.output_dir / f"mobrecon_predictions_{suffix}.jsonl",
        "fused": args.output_dir / f"palm_local_hands_{suffix}.jsonl",
        "config": args.output_dir / f"mobrecon_realtime_config_{suffix}.json",
    }
    for name, output_path in output_paths.items():
        if name != "config" and output_path.exists() and not args.overwrite:
            raise SystemExit(f"{output_path} exists; pass --overwrite to replace it")

    keyframe_source: KeyframeSource
    if args.follow_keyframes:
        keyframe_source = FollowingKeyframeSource(keyframe_paths, args.keyframe_timeout)
    else:
        keyframe_source = StaticKeyframeSource(keyframe_paths, set(cameras), group_ids)
    device = choose_device(args.device)
    if args.precision == "float16" and device.type != "cuda":
        raise SystemExit("float16 MobRecon inference requires CUDA")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    load_started = time.perf_counter()
    model, joint_regressor, model_mib = load_model(args.mobrecon_root.expanduser().resolve(), checkpoint, device)
    model_load_seconds = time.perf_counter() - load_started
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    states = {camera_id: tracker_state() for camera_id in cameras}
    track_handedness: dict[tuple[str, str], str] = {}
    track_handedness_conflicts: dict[tuple[str, str], tuple[str, int]] = {}
    fused_shape_history: dict[str, np.ndarray] = {}
    one_euro_filtered_state: dict[str, np.ndarray] = {}
    one_euro_raw_state: dict[str, np.ndarray] = {}
    one_euro_derivative_state: dict[str, np.ndarray] = {}
    one_euro_previous_group: dict[str, int] = {}
    stats: dict[str, int] = defaultdict(int)
    timing: dict[str, float] = defaultdict(float)
    group_latencies = []
    pending_packets: list[dict[str, Any]] = []
    fusion_config = fusion_args()
    processing_started = time.perf_counter()

    tracks_file = output_paths["tracks"].open("w", encoding="utf-8", buffering=1) if args.write_intermediates else nullcontext()
    jobs_file = output_paths["jobs"].open("w", encoding="utf-8", buffering=1) if args.write_intermediates else nullcontext()
    with (
        tracks_file as tracks_output,
        jobs_file as jobs_output,
        output_paths["predictions"].open("w", encoding="utf-8", buffering=1) as predictions_output,
        output_paths["fused"].open("w", encoding="utf-8", buffering=1) as fused_output,
    ):
        progress = tqdm(grouped_frames, desc="MobRecon realtime CPU", unit="frame", position=args.progress_position)

        def flush_pending() -> None:
            if not pending_packets:
                return
            samples = [sample for packet in pending_packets for sample in packet["samples"]]
            predictions_by_group: dict[int, list[dict[str, Any]]] = defaultdict(list)
            for batch_start in range(0, len(samples), args.batch_size):
                batch_samples = samples[batch_start : batch_start + args.batch_size]
                if not batch_samples:
                    continue
                inputs = torch.stack([sample["input"] for sample in batch_samples]).to(device, non_blocking=True)
                model_started = time.perf_counter()
                autocast_context = (
                    torch.autocast(device_type="cuda", dtype=torch.float16)
                    if args.precision == "float16"
                    else nullcontext()
                )
                with torch.inference_mode(), autocast_context:
                    outputs = model(inputs)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                timing["model_seconds"] += time.perf_counter() - model_started
                stats["model_batches"] += 1
                meshes = outputs["mesh_pred"].detach().float().cpu().numpy()
                points = outputs["uv_pred"].detach().float().cpu().numpy()
                for sample_index, sample in enumerate(batch_samples):
                    joints_m, mesh_m, points_2d = restore_outputs(
                        meshes[sample_index], points[sample_index], sample, joint_regressor
                    )
                    record = prediction_record(
                        sample["job"],
                        sample,
                        joints_m,
                        mesh_m,
                        points_2d,
                        args.export_vertices,
                        args.crop_scale,
                        "online_spatial",
                    )
                    predictions_by_group[int(record["group_id"])].append(record)
                    predictions_output.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                    stats["predictions"] += 1
            fusion_started = time.perf_counter()
            for packet in pending_packets:
                group_id = int(packet["group_id"])
                fused_record, selected_hypotheses = fuse_group(
                    group_id,
                    predictions_by_group.get(group_id, []),
                    fusion_config,
                    fused_shape_history,
                    track_handedness,
                )
                apply_one_euro(
                    fused_record,
                    one_euro_filtered_state,
                    one_euro_raw_state,
                    one_euro_derivative_state,
                    one_euro_previous_group,
                    args,
                )
                fused_output.write(json.dumps(fused_record, ensure_ascii=False, separators=(",", ":")) + "\n")
                stats["fused_hands"] += len(fused_record["hands"])
                stats["selected_dual_hypotheses"] += selected_hypotheses
                stats["frames"] += 1
                group_latencies.append(time.perf_counter() - packet["started"])
            timing["fusion_seconds"] += time.perf_counter() - fusion_started
            pending_packets.clear()

        for group_id, by_camera in progress:
            group_started = time.perf_counter()
            samples = []
            for camera_id in cameras:
                frame_record = by_camera.get(camera_id)
                if frame_record is None:
                    stats["missing_frame_records"] += 1
                    continue
                image_path = args.rectified_dir / rectified_rel_path(frame_record)
                read_started = time.perf_counter()
                image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                timing["image_read_seconds"] += time.perf_counter() - read_started
                if image is None:
                    stats["missing_images"] += 1
                    states[camera_id]["sequence_index"] += 1
                    continue
                sequence_index = int(states[camera_id]["sequence_index"])
                scheduled = sequence_index % args.keyframe_stride == 0 or states[camera_id]["previous_gray"] is None
                wait_started = time.perf_counter()
                detections = keyframe_source.get(group_id, camera_id) if scheduled else []
                keyframe_wait_seconds = time.perf_counter() - wait_started
                timing["keyframe_wait_seconds"] += keyframe_wait_seconds
                if scheduled and group_id == grouped_frames[0][0]:
                    timing["initial_keyframe_wait_seconds"] += keyframe_wait_seconds
                else:
                    timing["subsequent_keyframe_wait_seconds"] += keyframe_wait_seconds
                tracking_started = time.perf_counter()
                hands, is_keyframe, failures = track_image(
                    image, group_id, camera_id, detections, states[camera_id], args
                )
                timing["tracking_seconds"] += time.perf_counter() - tracking_started
                stats["tracking_failures"] += failures
                stats["keyframe_images"] += int(is_keyframe)
                track_record = {
                    "type": "sam3_sparse_tracks",
                    "group_id": group_id,
                    "camera_id": camera_id,
                    "rectified_image_path": str(image_path),
                    "is_keyframe": is_keyframe,
                    "hands": hands,
                }
                if tracks_output is not None:
                    tracks_output.write(json.dumps(track_record, ensure_ascii=False, separators=(",", ":")) + "\n")

                jobs_started = time.perf_counter()
                jobs = [hand_job(hand, image_path, group_id, camera_id, index, args.bbox_pad) for index, hand in enumerate(hands)]
                jobs, unresolved_count = assign_online_handedness(
                    jobs,
                    track_handedness,
                    track_handedness_conflicts,
                    args.handedness_switch_confirm_keyframes,
                )
                stats["unresolved_handedness"] += unresolved_count
                stats["handedness_conflicts_held"] += sum(
                    job.get("handedness_source") == "online_track_conflict_hold" for job in jobs
                )
                job_record = {"type": "hamer_multiview_jobs", "group_id": group_id, "camera_id": camera_id, "jobs": jobs}
                if jobs_output is not None:
                    jobs_output.write(json.dumps(job_record, ensure_ascii=False, separators=(",", ":")) + "\n")
                timing["jobs_seconds"] += time.perf_counter() - jobs_started

                preprocess_started = time.perf_counter()
                for job in jobs:
                    samples.append(prepare_sample(image, job, str(job["handedness"]), args))
                timing["preprocess_seconds"] += time.perf_counter() - preprocess_started
                stats["jobs"] += len(jobs)
                stats["camera_images"] += 1
            pending_packets.append({"group_id": group_id, "samples": samples, "started": group_started})
            if len(pending_packets) >= args.microbatch_groups:
                flush_pending()
        flush_pending()
        progress.close()

    processing_seconds = time.perf_counter() - processing_started
    if isinstance(keyframe_source, FollowingKeyframeSource):
        keyframe_source.close()
    peak_allocated_mib = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None
    peak_reserved_mib = torch.cuda.max_memory_reserved(device) / 2**20 if device.type == "cuda" else None
    frame_count = int(stats["frames"])
    warm_processing_seconds = max(
        0.0,
        processing_seconds - float(timing.get("initial_keyframe_wait_seconds", 0.0)),
    )
    outputs = {
        "predictions": str(output_paths["predictions"]),
        "fused": str(output_paths["fused"]),
    }
    if args.write_intermediates:
        outputs.update({"tracks": str(output_paths["tracks"]), "jobs": str(output_paths["jobs"])})
    config = {
        "frames_metadata": str(args.frames),
        "keyframe_sam3": str(args.keyframe_sam3) if args.keyframe_sam3 is not None else None,
        "keyframe_shards": [str(path) for path in args.keyframe_shard or []],
        "follow_keyframes": args.follow_keyframes,
        "cameras": cameras,
        "group_range": args.group_range,
        "group_ids": args.group_ids,
        "keyframe_stride": args.keyframe_stride,
        "flow_scale": args.flow_scale,
        "max_flow_error": args.max_flow_error,
        "max_forward_backward_error": args.max_forward_backward_error,
        "microbatch_groups": args.microbatch_groups,
        "batch_size": args.batch_size,
        "device": str(device),
        "precision": args.precision,
        "torch_threads": args.torch_threads,
        "one_euro_min_cutoff": args.one_euro_min_cutoff,
        "one_euro_beta": args.one_euro_beta,
        "one_euro_derivative_cutoff": args.one_euro_derivative_cutoff,
        "frame_rate": args.frame_rate,
        "handedness_switch_confirm_keyframes": args.handedness_switch_confirm_keyframes,
        "model_mib": model_mib,
        "peak_cuda_allocated_mib": peak_allocated_mib,
        "peak_cuda_reserved_mib": peak_reserved_mib,
        "stats": dict(stats),
        "timing": {
            "model_load_seconds": model_load_seconds,
            "processing_seconds": processing_seconds,
            "processing_fps": frame_count / processing_seconds if processing_seconds else None,
            "warm_processing_seconds": warm_processing_seconds,
            "warm_processing_fps": frame_count / warm_processing_seconds if warm_processing_seconds else None,
            "wall_seconds_with_model_load": model_load_seconds + processing_seconds,
            "wall_fps_with_model_load": frame_count / (model_load_seconds + processing_seconds),
            "mean_group_latency_seconds": float(np.mean(group_latencies)) if group_latencies else None,
            "p95_group_latency_seconds": percentile(group_latencies, 95),
            **dict(timing),
        },
        "outputs": outputs,
        "keyframe_wait_included": args.follow_keyframes,
    }
    with output_paths["config"].open("w", encoding="utf-8") as config_file:
        json.dump(config, config_file, ensure_ascii=False, indent=2)
        config_file.write("\n")
    print(json.dumps(config, ensure_ascii=False, indent=2))
    print(f"wrote: {output_paths['config']}")


if __name__ == "__main__":
    main()
