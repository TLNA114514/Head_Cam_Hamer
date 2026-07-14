#!/usr/bin/env python3
"""Orchestrate the SAM3 + MediaPipe-guided multi-view HaMeR pipeline."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from statistics import median
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import Queue
from threading import Lock

from hamer_multiview_utils import DEFAULT_BASE_DIR, DEFAULT_CAMERAS, DEFAULT_FRAMES, iter_jsonl, parse_cameras, parse_group_ids, range_suffix
from progress_utils import tqdm


WRIST_CAM_ROOT = Path("/home/luojiangrui/ljr/wrist_cam")
LOG_LOCK = Lock()
TTY_STREAM = None
PROC_LOCK = Lock()
ACTIVE_PROCS: set[subprocess.Popen] = set()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument(
        "--frames",
        type=Path,
        help="Frame metadata JSONL. Defaults to the cameras_<dataset>/frames.jsonl matching --base-dir.",
    )
    parser.add_argument("--mediapipe", type=Path, help="MediaPipe landmarks JSONL. Defaults to base-dir/landmarks.jsonl.")
    parser.add_argument("--rectify-focal-scale", type=float, default=0.30)
    parser.add_argument("--group-range")
    parser.add_argument("--range", dest="range_alias", help="Short alias for --group-range.")
    parser.add_argument("--group-ids")
    parser.add_argument("--cameras", default=",".join(DEFAULT_CAMERAS))
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--max-parallel-workers", type=int, default=2)
    parser.add_argument("--max-mediapipe-workers", type=int, default=4)
    parser.add_argument("--max-hamer-workers", type=int, default=1)
    parser.add_argument("--conda-bin", default="/home/luojiangrui/miniconda3/bin/conda")
    parser.add_argument("--sam3-conda-env", default="sam3hand")
    parser.add_argument("--hamer-conda-env", default="hamer")
    parser.add_argument("--sam3-root", type=Path, default=WRIST_CAM_ROOT / "third_party" / "sam3")
    parser.add_argument("--sam3-checkpoint", type=Path, help="Optional local SAM3 checkpoint. Prevents HuggingFace download when provided.")
    parser.add_argument("--sam3-no-hf", action="store_true", help="Do not allow SAM3 to download checkpoints from HuggingFace.")
    parser.add_argument("--sam3-hf-endpoint", default="https://hf-mirror.com", help="HF endpoint used by SAM3 downloads.")
    parser.add_argument("--sam3-version", choices=["sam3", "sam3.1"], default="sam3.1", help="SAM3 native video tracker version.")
    parser.add_argument(
        "--sam3-execution",
        choices=["per-sequence", "per-chunk"],
        default="per-sequence",
        help="Load the SAM3 image model once per sequence or once per camera/chunk.",
    )
    parser.add_argument(
        "--save-sam3-debug",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save full-frame SAM3 bbox and mask debug images.",
    )
    parser.add_argument("--hamer-root", type=Path, default=WRIST_CAM_ROOT / "third_party" / "hamer")
    parser.add_argument(
        "--hamer-execution",
        choices=["per-sequence", "per-camera", "per-chunk"],
        default="per-sequence",
        help="Load HaMeR once per sequence (fastest), per camera, or per camera/chunk (legacy resume granularity).",
    )
    parser.add_argument(
        "--hamer-speed-profile",
        choices=["quality", "balanced", "fast", "aggressive"],
        default="quality",
        help="quality keeps mesh scoring/FP32; aggressive keeps three scales but uses FP16, skeleton scoring, and compile.",
    )
    parser.add_argument("--hamer-batch-size", type=int, default=4)
    parser.add_argument("--hamer-job-batch-size", type=int, default=8)
    parser.add_argument("--hamer-candidate-bbox-scales")
    parser.add_argument("--hamer-candidate-scale-policy", choices=["fixed", "mask-adaptive"])
    parser.add_argument("--hamer-precision", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--hamer-allow-tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hamer-mask-scoring", choices=["all", "selection-only", "none"])
    parser.add_argument("--hamer-mask-score-method", choices=["mesh", "skeleton"])
    parser.add_argument("--hamer-compile-backbone", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--hamer-compile-mode",
        choices=["default", "reduce-overhead", "max-autotune"],
        default="reduce-overhead",
    )
    parser.add_argument("--hamer-export-vertices", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--hamer-export-mano-params", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--save-hamer-rendered-overlays",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save every HaMeR rendered overlay. Disabled by default to avoid full-sequence I/O; use selected debug/viewer outputs instead.",
    )
    parser.add_argument("--prompt-preset", choices=["bare", "gloved", "realtime", "custom"], default="bare")
    parser.add_argument("--prompt", action="append", dest="prompts")
    parser.add_argument(
        "--hand-track-backend",
        choices=["image", "posthoc", "sam3-native"],
        default="posthoc",
        help="Hand identity backend: image keeps old framewise SAM3; posthoc links boxes; sam3-native uses SAM3 video tracking.",
    )
    parser.add_argument("--track-overlap", type=int, default=10)
    parser.add_argument("--handedness-lock-score", type=float, default=0.85)
    parser.add_argument("--lock-votes", type=int, default=3)
    parser.add_argument("--unlock-votes", type=int, default=5)
    parser.add_argument("--max-missing", type=int, default=8)
    parser.add_argument(
        "--camera-handedness-override",
        default="none",
        help='Fusion-stage hard handedness overrides; use "none" to disable.',
    )
    parser.add_argument(
        "--camera-handedness-prior",
        default="C0:Left,C3:Right",
        help='Weak camera handedness priors used only for unknown labels; use "none" to disable.',
    )
    parser.add_argument("--temporal", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--temporal-selection-weight", type=float, default=0.45)
    parser.add_argument("--temporal-error-cap-m", type=float, default=0.08)
    parser.add_argument("--temporal-metric-alpha", type=float, default=0.75)
    parser.add_argument("--temporal-primary-alpha", type=float, default=0.45)
    parser.add_argument("--temporal-backup-alpha", type=float, default=0.40)
    parser.add_argument("--temporal-nonprimary-alpha", type=float, default=0.30)
    parser.add_argument("--temporal-quality-anchor-alpha", type=float, default=0.42)
    parser.add_argument("--quality-mask-weight", type=float, default=0.55)
    parser.add_argument("--quality-bbox-weight", type=float, default=0.15)
    parser.add_argument("--quality-edge-weight", type=float, default=0.12)
    parser.add_argument("--quality-source-bonus", type=float, default=0.06)
    parser.add_argument("--quality-known-bonus", type=float, default=0.05)
    parser.add_argument("--primary-prior-bonus", type=float, default=0.08)
    parser.add_argument("--backup-prior-bonus", type=float, default=0.04)
    parser.add_argument("--consensus-selection-weight", type=float, default=0.35)
    parser.add_argument("--consensus-error-cap-m", type=float, default=0.06)
    parser.add_argument("--anchor-switch-margin", type=float, default=0.04)
    parser.add_argument("--min-anchor-score", type=float, default=-1.0)
    parser.add_argument("--backup-primary-cameras", default="Left:C0,Right:C3")
    parser.add_argument("--backup-min-mask-score", type=float, default=0.25)
    parser.add_argument("--backup-min-view-quality", type=float, default=0.45)
    parser.add_argument("--backup-max-temporal-error-m", type=float, default=0.06)
    parser.add_argument("--backup-require-known", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-zero-shot-palm-fusion", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--zero-shot-primary-output",
        choices=["raw", "static-calibrated", "smoothed", "causal-smoothed", "adaptive-causal"],
        default="raw",
    )
    parser.add_argument("--zero-shot-bone-calibration-blend", type=float, default=0.0)
    parser.add_argument("--zero-shot-temporal-radius", type=int, default=0)
    parser.add_argument("--zero-shot-temporal-sigma", type=float, default=4.0)
    parser.add_argument("--zero-shot-causal-ema-alpha", type=float, default=0.0)
    parser.add_argument("--zero-shot-one-euro-min-cutoff", type=float, default=0.2)
    parser.add_argument("--zero-shot-one-euro-beta", type=float, default=5.0)
    parser.add_argument("--zero-shot-one-euro-derivative-cutoff", type=float, default=1.0)
    parser.add_argument("--frame-rate", type=float, help="Override FPS used by causal filters; inferred from frame timestamps by default.")
    parser.add_argument("--zero-shot-include-vertices", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--run-legacy-local-fusion",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Generate hamer_primary_local output. Not required by the default zero-shot fusion.",
    )
    parser.add_argument(
        "--run-mano-local-refine",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run the legacy MANO local refinement. Disabled by default because zero-shot palm fusion is faster and more accurate here.",
    )
    parser.add_argument("--run-mano-multiview-image-refine", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--run-mano-image-candidate-selection",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When image refinement is enabled, run baseline and PnP-gated candidates then select per hand by baseline reprojection diagnostics.",
    )
    parser.add_argument(
        "--image-global-initialization",
        choices=["hamer-virtual", "physical-pnp"],
        default="physical-pnp",
        help="Global initializer for image-space refinement. physical-pnp is the recommended rectified-camera path.",
    )
    parser.add_argument("--use-mediapipe-2d", choices=["never", "weak", "auto"], default="never")
    parser.add_argument("--window-size", type=int, default=7)
    parser.add_argument("--optimize-mano-pose", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--optimize-mano-betas", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-frame-beta-delta", action="store_true")
    parser.add_argument("--local-consistency-threshold-m", type=float, default=0.025)
    parser.add_argument("--temporal-reject-threshold-m", type=float, default=0.12)
    parser.add_argument("--pose-prior-weight", type=float, default=0.25)
    parser.add_argument("--beta-prior-weight", type=float, default=1.0)
    parser.add_argument("--temporal-pose-weight", type=float, default=0.35)
    parser.add_argument("--vertex-loss-weight", type=float, default=0.10)
    parser.add_argument("--mano-max-iters", type=int, default=80)
    parser.add_argument("--mano-beta-iters", type=int, default=120)
    parser.add_argument("--image-soft-reprojection-error-px", type=float, default=90.0)
    parser.add_argument("--image-soft-mask-distance-px", type=float, default=90.0)
    parser.add_argument("--image-min-view-weight", type=float, default=0.12)
    parser.add_argument("--image-anchor-view-weight", type=float, default=1.45)
    parser.add_argument("--image-min-soft-used-weight", type=float, default=0.08)
    parser.add_argument("--image-min-metric-used-weight", type=float, default=0.28)
    parser.add_argument("--image-primary-anchor-score-margin", type=float, default=0.08)
    parser.add_argument("--image-pnp-view-gate-m", type=float, default=0.04)
    parser.add_argument(
        "--image-selection-min-baseline-max-reprojection-px",
        type=float,
        default=0.0,
        help="Prefer every available PnP-gated image-refinement candidate by default; set a positive value for diagnostic-threshold selection.",
    )
    parser.add_argument("--image-min-readable-sam3-mask-ratio", type=float, default=0.50)
    parser.add_argument("--image-projection-correction", type=Path, help="Optional projection correction JSON for image-space MANO refine.")
    parser.add_argument(
        "--image-beta-estimation-space",
        choices=["hamer-local", "image-2d"],
        default="hamer-local",
        help="Sequence beta initializer for image refinement; image-2d is the calibrated multi-view experimental path.",
    )
    parser.add_argument("--image-beta-max-observations", type=int, default=240)
    parser.add_argument("--image-beta-prior-weight", type=float, default=2.0)
    parser.add_argument("--image-temporal-acceleration-weight", type=float, default=0.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-mediapipe", action="store_true")
    parser.add_argument("--skip-sam3", action="store_true")
    parser.add_argument("--skip-hamer", action="store_true")
    return parser.parse_args()


def script_path(name: str) -> str:
    return str((Path(__file__).resolve().parent / name).resolve())


def emit(message: str) -> None:
    global TTY_STREAM
    line = str(message)
    with LOG_LOCK:
        if TTY_STREAM is None:
            try:
                TTY_STREAM = open("/dev/tty", "w", encoding="utf-8", buffering=1)
            except OSError:
                TTY_STREAM = False
        if TTY_STREAM:
            TTY_STREAM.write(line + "\n")
            TTY_STREAM.flush()
        else:
            print(line, flush=True)


def command_env(progress_position: int = 0) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONWARNINGS", "ignore")
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    env.setdefault("GLOG_minloglevel", "2")
    env.setdefault("ABSL_MIN_LOG_LEVEL", "2")
    env["TQDM_POSITION"] = str(progress_position)
    return env


def register_proc(proc: subprocess.Popen) -> None:
    with PROC_LOCK:
        ACTIVE_PROCS.add(proc)


def unregister_proc(proc: subprocess.Popen) -> None:
    with PROC_LOCK:
        ACTIVE_PROCS.discard(proc)


def stop_process_group(proc: subprocess.Popen, sig: int) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, sig)
    except ProcessLookupError:
        return
    except OSError:
        try:
            proc.send_signal(sig)
        except OSError:
            return


def terminate_active_processes() -> None:
    with PROC_LOCK:
        procs = [proc for proc in ACTIVE_PROCS if proc.poll() is None]
    if not procs:
        return
    emit(f"[pipeline] stopping {len(procs)} active child process(es)...")
    for proc in procs:
        stop_process_group(proc, signal.SIGINT)
    for proc in procs:
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            stop_process_group(proc, signal.SIGTERM)
    for proc in procs:
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            stop_process_group(proc, signal.SIGKILL)


def run_command(label: str, command: list[str], dry_run: bool, progress_position: int = 0) -> None:
    emit(f"[pipeline] run {label}: {' '.join(command)}")
    if dry_run:
        return
    started = time.perf_counter()
    tty = None
    try:
        tty = open("/dev/tty", "w", encoding="utf-8", buffering=1)
    except OSError:
        tty = None
    if tty is not None:
        proc = subprocess.Popen(
            command,
            stdout=tty,
            stderr=tty,
            text=True,
            env=command_env(progress_position),
            start_new_session=True,
        )
        register_proc(proc)
        try:
            returncode = proc.wait()
        except KeyboardInterrupt:
            stop_process_group(proc, signal.SIGINT)
            raise
        finally:
            unregister_proc(proc)
            tty.flush()
            tty.close()
        if returncode != 0:
            raise subprocess.CalledProcessError(returncode, command)
        emit(f"[pipeline] done {label} elapsed={time.perf_counter() - started:.3f}s")
        return
    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=command_env(progress_position),
        start_new_session=True,
    )
    register_proc(proc)
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            line = line.rstrip("\n")
            if line:
                emit(f"[{label}] {line}")
        returncode = proc.wait()
    except KeyboardInterrupt:
        stop_process_group(proc, signal.SIGINT)
        raise
    finally:
        unregister_proc(proc)
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, command)
    emit(f"[pipeline] done {label} elapsed={time.perf_counter() - started:.3f}s")


def run_parallel_command(label: str, command: list[str], progress_position: int) -> None:
    run_command(label, command, False, progress_position)


def run_parallel(commands: list[tuple[str, list[str]]], max_workers: int, dry_run: bool, desc: str) -> None:
    if dry_run:
        for label, command in commands:
            emit(f"[pipeline] parallel {label}: {' '.join(command)}")
        return
    positions: Queue[int] = Queue()
    for position in range(1, max_workers + 1):
        positions.put(position)

    def wrapped(label: str, command: list[str]) -> None:
        position = positions.get()
        try:
            run_parallel_command(label, command, position)
        finally:
            positions.put(position)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(wrapped, label, command): (label, command) for label, command in commands}
        with tqdm(total=len(futures), desc=desc, unit="task", position=0) as progress:
            for future in as_completed(futures):
                label, command = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    raise RuntimeError(f"command failed ({label}): {' '.join(command)}") from exc
                progress.update(1)


def merge_jsonl(inputs: list[Path], output: Path, record_type: str | None = None) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8") as out:
        for path in inputs:
            if not path.exists():
                continue
            for record in iter_jsonl(path):
                if record_type and record.get("type") != record_type:
                    continue
                out.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                count += 1
    return count


def merge_jsonl_unique(inputs: list[Path], output: Path, record_type: str | None, key_fields: tuple[str, ...]) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    count = 0
    with output.open("w", encoding="utf-8") as out:
        for path in inputs:
            if not path.exists():
                continue
            for record in iter_jsonl(path):
                if record_type and record.get("type") != record_type:
                    continue
                key = tuple(record.get(field) for field in key_fields)
                if key in seen:
                    continue
                seen.add(key)
                out.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                count += 1
    return count


def conda_python(conda_bin: str, env_name: str) -> list[str]:
    return [conda_bin, "run", "--no-capture-output", "-n", env_name, "python", "-u", "-s"]


def group_ids_from_frames(frames_path: Path, cameras: set[str], requested: set[int] | None) -> list[int]:
    ids = set()
    for record in iter_jsonl(frames_path):
        if record.get("camera_id") not in cameras:
            continue
        group_id = int(record["group_id"])
        if requested is not None and group_id not in requested:
            continue
        ids.add(group_id)
    return sorted(ids)


def infer_frame_rate(frames_path: Path, camera_id: str) -> float:
    timestamps: list[tuple[int, int]] = []
    for record in iter_jsonl(frames_path):
        if record.get("camera_id") != camera_id or record.get("timestamp_monotonic_ns") is None:
            continue
        timestamps.append((int(record["group_id"]), int(record["timestamp_monotonic_ns"])))
    timestamps.sort()
    deltas = [
        (current_time - previous_time) / 1e9
        for (previous_group, previous_time), (current_group, current_time) in zip(timestamps, timestamps[1:])
        if current_group == previous_group + 1 and 0 < current_time - previous_time < 1_000_000_000
    ]
    return 1.0 / median(deltas) if deltas else 25.0


def frames_path_for_base_dir(base_dir: Path) -> Path:
    dataset_name = base_dir.name.removeprefix("sam3_hamer_")
    candidate = base_dir.parent / f"cameras_{dataset_name}" / "frames.jsonl"
    return candidate if candidate.exists() else DEFAULT_FRAMES


def chunk_group_ids(group_ids: list[int], chunk_size: int) -> list[list[int]]:
    if chunk_size < 1:
        raise ValueError("--chunk-size must be positive")
    return [group_ids[index:index + chunk_size] for index in range(0, len(group_ids), chunk_size)]


def chunk_suffix_and_args(ids: list[int]) -> tuple[str, list[str]]:
    chunk_set = set(ids)
    suffix = range_suffix(chunk_set)
    if ids == list(range(ids[0], ids[-1] + 1)):
        return suffix, ["--group-range", f"{ids[0]}-{ids[-1]}"]
    return suffix, ["--group-ids", ",".join(str(item) for item in ids)]


def expanded_chunk_ids(chunk_ids: list[int], selected_group_ids: list[int], overlap: int) -> list[int]:
    if overlap <= 0:
        return list(chunk_ids)
    selected = set(selected_group_ids)
    start = chunk_ids[0] - overlap
    end = chunk_ids[-1] + overlap
    return [group_id for group_id in selected_group_ids if group_id in selected and start <= group_id <= end]


def main() -> None:
    args = parse_args()
    if args.hamer_batch_size < 1 or args.hamer_job_batch_size < 1:
        raise SystemExit("--hamer-batch-size and --hamer-job-batch-size must be positive")
    profile_defaults = {
        "quality": ("1.0,1.1,1.2", "mask-adaptive", "float32", "all", "mesh", False),
        "balanced": ("1.0", "fixed", "float32", "selection-only", "mesh", False),
        "fast": ("1.0", "fixed", "float16", "selection-only", "mesh", False),
        "aggressive": ("1.0,1.1,1.2", "mask-adaptive", "float16", "all", "skeleton", True),
    }
    (
        profile_scales,
        profile_scale_policy,
        profile_precision,
        profile_mask_scoring,
        profile_mask_score_method,
        profile_compile_backbone,
    ) = profile_defaults[args.hamer_speed_profile]
    args.hamer_candidate_bbox_scales = args.hamer_candidate_bbox_scales or profile_scales
    args.hamer_candidate_scale_policy = args.hamer_candidate_scale_policy or profile_scale_policy
    args.hamer_precision = args.hamer_precision or profile_precision
    args.hamer_mask_scoring = args.hamer_mask_scoring or profile_mask_scoring
    args.hamer_mask_score_method = args.hamer_mask_score_method or profile_mask_score_method
    if args.hamer_compile_backbone is None:
        args.hamer_compile_backbone = profile_compile_backbone
    if not 0.0 <= args.zero_shot_bone_calibration_blend <= 1.0:
        raise SystemExit("--zero-shot-bone-calibration-blend must be in [0, 1]")
    if args.zero_shot_temporal_radius < 0:
        raise SystemExit("--zero-shot-temporal-radius must be non-negative")
    if args.zero_shot_temporal_radius > 0 and args.zero_shot_temporal_sigma <= 0.0:
        raise SystemExit("--zero-shot-temporal-sigma must be positive when smoothing is enabled")
    if not 0.0 <= args.zero_shot_causal_ema_alpha <= 1.0:
        raise SystemExit("--zero-shot-causal-ema-alpha must be in [0, 1]")
    if args.zero_shot_one_euro_min_cutoff < 0.0 or args.zero_shot_one_euro_beta < 0.0:
        raise SystemExit("One Euro cutoff and beta must be non-negative")
    if args.zero_shot_one_euro_derivative_cutoff <= 0.0:
        raise SystemExit("One Euro derivative cutoff must be positive")
    if args.zero_shot_primary_output == "static-calibrated" and args.zero_shot_bone_calibration_blend <= 0.0:
        raise SystemExit("--zero-shot-primary-output static-calibrated requires a positive bone calibration blend")
    if args.zero_shot_primary_output == "smoothed" and args.zero_shot_temporal_radius <= 0:
        raise SystemExit("--zero-shot-primary-output smoothed requires --zero-shot-temporal-radius > 0")
    if args.zero_shot_primary_output == "causal-smoothed" and args.zero_shot_causal_ema_alpha <= 0.0:
        raise SystemExit("--zero-shot-primary-output causal-smoothed requires --zero-shot-causal-ema-alpha > 0")
    if args.zero_shot_primary_output == "adaptive-causal" and args.zero_shot_one_euro_min_cutoff <= 0.0:
        raise SystemExit("--zero-shot-primary-output adaptive-causal requires --zero-shot-one-euro-min-cutoff > 0")
    if args.range_alias and not args.group_range:
        args.group_range = args.range_alias
    cameras = sorted(parse_cameras(args.cameras))
    if args.frames is None:
        args.frames = frames_path_for_base_dir(args.base_dir)
        emit(f"[pipeline] inferred frames metadata: {args.frames}")
    if args.frame_rate is None:
        args.frame_rate = infer_frame_rate(args.frames, cameras[0])
        emit(f"[pipeline] inferred frame rate: {args.frame_rate:.3f} FPS")
    if args.frame_rate <= 0.0:
        raise SystemExit("--frame-rate must be positive")
    camera_set = set(cameras)
    group_ids = parse_group_ids(args.group_range, args.group_ids)
    suffix = range_suffix(group_ids)
    max_workers = max(1, min(2, int(args.max_parallel_workers)))
    max_mediapipe_workers = max(1, min(len(cameras), int(args.max_mediapipe_workers)))
    max_hamer_workers = max(1, min(len(cameras), int(args.max_hamer_workers)))
    selected_group_ids = group_ids_from_frames(args.frames, camera_set, group_ids)
    chunks = chunk_group_ids(selected_group_ids, args.chunk_size)
    if not selected_group_ids:
        raise SystemExit("no group ids selected from frames")

    if not args.sam3_root.exists():
        raise SystemExit(f"SAM3 root not found: {args.sam3_root}. Reuse/setup wrist_cam first.")
    if not args.hamer_root.exists():
        raise SystemExit(f"HaMeR root not found: {args.hamer_root}. Reuse/setup wrist_cam first.")
    if args.sam3_no_hf and args.sam3_checkpoint is None and not args.skip_sam3:
        raise SystemExit("--sam3-no-hf requires --sam3-checkpoint, otherwise SAM3 has no checkpoint to load.")

    common_range = []
    if args.group_range:
        common_range.extend(["--group-range", args.group_range])
    if args.group_ids:
        common_range.extend(["--group-ids", args.group_ids])
    overwrite = ["--overwrite"] if args.overwrite else []
    mediapipe_path = args.mediapipe or (args.base_dir / "landmarks.jsonl")
    rectified_config_path = args.base_dir / "rectified_for_hamer" / f"rectified_config_{suffix}.json"

    rectified_cmd = [
        sys.executable,
        "-u",
        script_path("prepare_hamer_rectified.py"),
        "--frames",
        str(args.frames),
        "--output-dir",
        str(args.base_dir / "rectified_for_hamer"),
        "--cameras",
        ",".join(cameras),
        "--rectify-focal-scale",
        str(args.rectify_focal_scale),
        *common_range,
        *overwrite,
    ]
    run_command("rectify", rectified_cmd, args.dry_run, progress_position=0)

    if not args.skip_mediapipe:
        if mediapipe_path.name != "landmarks.jsonl":
            raise SystemExit("Auto MediaPipe generation requires --mediapipe to end with landmarks.jsonl")
        if args.overwrite or not mediapipe_path.exists():
            mediapipe_shards = []
            mediapipe_commands = []
            for camera_id in cameras:
                shard_dir = mediapipe_path.parent / "mediapipe_shards" / camera_id
                shard_path = shard_dir / "landmarks.jsonl"
                mediapipe_shards.append(shard_path)
                mediapipe_commands.append(
                    (
                        f"mediapipe:{camera_id}",
                        [
                            sys.executable,
                            "-u",
                            script_path("detect_mediapipe_hands.py"),
                            "--frames",
                            str(args.frames),
                            "--output",
                            str(shard_dir),
                            "--cameras",
                            camera_id,
                            "--rectify-focal-scale",
                            str(args.rectify_focal_scale),
                            *common_range,
                            *overwrite,
                        ],
                    )
                )
            run_parallel(mediapipe_commands, max_mediapipe_workers, args.dry_run, desc="MediaPipe cameras")
            if args.dry_run:
                emit(f"[pipeline] merge MediaPipe shards -> {mediapipe_path}")
            else:
                count = merge_jsonl(mediapipe_shards, mediapipe_path)
                emit(f"[pipeline] merged MediaPipe records: {count} -> {mediapipe_path}")
        else:
            emit(f"[pipeline] skip MediaPipe, found {mediapipe_path}")

    sam3_output = args.base_dir / "sam3_bboxes" / f"sam3_bboxes_{suffix}.jsonl"
    stabilized_tracks_output = args.base_dir / "sam3_tracks_stabilized" / f"sam3_tracks_stabilized_{suffix}.jsonl"
    jobs_path = args.base_dir / "hamer_jobs" / f"hamer_jobs_{suffix}.jsonl"
    hamer_output = args.base_dir / "hamer_per_view" / f"hamer_predictions_{suffix}.jsonl"
    sam3_shards = []
    stabilized_track_shards = []
    job_shards = []
    hamer_shards = []

    def sam3_command(output_dir: Path, camera_value: str, range_args: list[str]) -> list[str]:
        command = [
            *conda_python(args.conda_bin, args.sam3_conda_env),
            script_path("detect_sam3_hands_multiview.py"),
            "--frames",
            str(args.frames),
            "--rectified-dir",
            str(args.base_dir / "rectified_for_hamer"),
            "--output-dir",
            str(output_dir),
            "--sam3-root",
            str(args.sam3_root),
            "--hf-endpoint",
            args.sam3_hf_endpoint,
            "--cameras",
            camera_value,
            "--prompt-preset",
            args.prompt_preset,
            "--save-mask-debug" if args.save_sam3_debug else "--no-save-mask-debug",
            "--save-bbox-debug" if args.save_sam3_debug else "--no-save-bbox-debug",
            *range_args,
            *overwrite,
        ]
        if args.sam3_checkpoint:
            command.extend(["--checkpoint", str(args.sam3_checkpoint)])
        if args.sam3_no_hf:
            command.append("--no-hf")
        for prompt in args.prompts or []:
            command.extend(["--prompt", prompt])
        return command

    def hamer_command(jobs_source: Path, camera_id: str | None, range_args: list[str]) -> list[str]:
        export_vertices = bool(
            args.hamer_export_vertices or args.run_mano_local_refine or args.run_mano_multiview_image_refine
        )
        export_mano_params = bool(
            args.hamer_export_mano_params or args.run_mano_local_refine or args.run_mano_multiview_image_refine
        )
        command = [
            *conda_python(args.conda_bin, args.hamer_conda_env),
            script_path("hamer_multiview_worker.py"),
            "--jobs",
            str(jobs_source),
            "--output-dir",
            str(args.base_dir / "hamer_per_view"),
            "--hamer-root",
            str(args.hamer_root),
            "--rectified-config",
            str(rectified_config_path),
            "--batch-size",
            str(args.hamer_batch_size),
            "--job-batch-size",
            str(args.hamer_job_batch_size),
            "--candidate-bbox-scales",
            args.hamer_candidate_bbox_scales,
            "--candidate-scale-policy",
            args.hamer_candidate_scale_policy,
            "--precision",
            args.hamer_precision,
            "--mask-scoring",
            args.hamer_mask_scoring,
            "--mask-score-method",
            args.hamer_mask_score_method,
            "--compile-backbone" if args.hamer_compile_backbone else "--no-compile-backbone",
            "--compile-mode",
            args.hamer_compile_mode,
            "--allow-tf32" if args.hamer_allow_tf32 else "--no-allow-tf32",
            "--export-vertices" if export_vertices else "--no-export-vertices",
            "--export-mano-params" if export_mano_params else "--no-export-mano-params",
            "--save-rendered-overlay" if args.save_hamer_rendered_overlays else "--no-save-rendered-overlay",
            *range_args,
            *overwrite,
        ]
        if camera_id is not None:
            camera_insert = command.index("--batch-size")
            command[camera_insert:camera_insert] = ["--camera-id", camera_id]
        return command

    if not args.skip_sam3 and args.sam3_execution == "per-sequence":
        run_command(
            "sam3:sequence",
            sam3_command(args.base_dir / "sam3_bboxes", ",".join(cameras), common_range),
            args.dry_run,
        )

    def heavy_chunk_pipeline(camera_id: str, chunk_ids: list[int], progress_position: int) -> None:
        chunk_suffix, chunk_range = chunk_suffix_and_args(chunk_ids)
        track_ids = expanded_chunk_ids(chunk_ids, selected_group_ids, args.track_overlap if args.hand_track_backend == "sam3-native" else 0)
        track_suffix, track_range = chunk_suffix_and_args(track_ids)
        label_suffix = f"{camera_id}:{chunk_suffix}"
        sam3_detect_suffix = track_suffix if args.hand_track_backend == "sam3-native" else chunk_suffix
        sam3_detect_range = track_range if args.hand_track_backend == "sam3-native" else chunk_range
        sam3_dir = args.base_dir / "sam3_bboxes" / "chunks" / camera_id / sam3_detect_suffix
        sam3_shard = (
            sam3_output
            if args.sam3_execution == "per-sequence"
            else sam3_dir / f"sam3_bboxes_{sam3_detect_suffix}.jsonl"
        )
        sequence_dir = args.base_dir / "sam3_video_sequences"
        track_dir = args.base_dir / "sam3_tracks" / "chunks" / camera_id / track_suffix
        track_shard = track_dir / f"sam3_tracks_{track_suffix}.jsonl"
        stabilized_dir = args.base_dir / "sam3_tracks_stabilized" / "chunks" / camera_id / track_suffix
        stabilized_shard = stabilized_dir / f"sam3_tracks_stabilized_{track_suffix}.jsonl"
        jobs_dir = args.base_dir / "hamer_jobs" / "chunks" / camera_id / chunk_suffix
        jobs_shard = jobs_dir / f"hamer_jobs_{chunk_suffix}.jsonl"

        if not args.skip_sam3 and args.sam3_execution == "per-chunk":
            sam3_cmd = sam3_command(sam3_dir, camera_id, sam3_detect_range)
            run_command(f"sam3:{label_suffix}", sam3_cmd, False, progress_position)

        tracked_hands_args: list[str] = []
        if args.hand_track_backend in {"posthoc", "sam3-native"}:
            if args.hand_track_backend == "sam3-native":
                sequence_cmd = [
                    sys.executable,
                    "-u",
                    script_path("prepare_sam3_video_sequences.py"),
                    "--frames",
                    str(args.frames),
                    "--rectified-dir",
                    str(args.base_dir / "rectified_for_hamer"),
                    "--output-dir",
                    str(sequence_dir),
                    "--cameras",
                    camera_id,
                    *track_range,
                    *overwrite,
                ]
                run_command(f"sequence:{label_suffix}", sequence_cmd, False, progress_position)
                track_cmd = [
                    *conda_python(args.conda_bin, args.sam3_conda_env),
                    script_path("track_sam3_hands_video.py"),
                    "--sequence-dir",
                    str(sequence_dir),
                    "--seed-sam3",
                    str(sam3_shard),
                    "--output-dir",
                    str(track_dir),
                    "--sam3-root",
                    str(args.sam3_root),
                    "--hf-endpoint",
                    args.sam3_hf_endpoint,
                    "--version",
                    args.sam3_version,
                    "--cameras",
                    camera_id,
                    *track_range,
                    *overwrite,
                ]
                if args.sam3_checkpoint:
                    track_cmd.extend(["--checkpoint", str(args.sam3_checkpoint)])
                if args.sam3_no_hf:
                    track_cmd.append("--no-hf")
                run_command(f"sam3_track:{label_suffix}", track_cmd, False, progress_position)
                tracks_for_stabilizer = track_shard
            else:
                tracks_for_stabilizer = sam3_shard

            stabilize_cmd = [
                sys.executable,
                "-u",
                script_path("stabilize_hand_identity_tracks.py"),
                "--tracks",
                str(tracks_for_stabilizer),
                "--mediapipe",
                str(mediapipe_path),
                "--output-dir",
                str(stabilized_dir),
                "--cameras",
                camera_id,
                "--mediapipe-strong-score",
                str(args.handedness_lock_score),
                "--lock-votes",
                str(args.lock_votes),
                "--unlock-votes",
                str(args.unlock_votes),
                "--max-missing",
                str(args.max_missing),
                *track_range,
                *overwrite,
            ]
            run_command(f"stabilize:{label_suffix}", stabilize_cmd, False, progress_position)
            tracked_hands_args = ["--tracked-hands", str(stabilized_shard)]

        fusion_cmd = [
            sys.executable,
            "-u",
            script_path("fuse_hamer_jobs.py"),
            "--frames",
            str(args.frames),
            "--rectified-dir",
            str(args.base_dir / "rectified_for_hamer"),
            "--sam3",
            str(sam3_shard),
            "--mediapipe",
            str(mediapipe_path),
            "--output-dir",
            str(jobs_dir),
            "--cameras",
            camera_id,
            "--camera-handedness-override",
            args.camera_handedness_override,
            "--camera-handedness-prior",
            args.camera_handedness_prior,
            *tracked_hands_args,
            *chunk_range,
            *overwrite,
        ]
        run_command(f"fuse_jobs:{label_suffix}", fusion_cmd, False, progress_position)

        if not args.skip_hamer and args.hamer_execution == "per-chunk":
            hamer_cmd = hamer_command(jobs_shard, camera_id, chunk_range)
            run_command(f"hamer:{label_suffix}", hamer_cmd, False, progress_position)

    heavy_tasks: list[tuple[str, str, list[int]]] = []
    for chunk_ids in chunks:
        for camera_id in cameras:
            chunk_suffix, _chunk_range = chunk_suffix_and_args(chunk_ids)
            sam3_detect_ids = expanded_chunk_ids(chunk_ids, selected_group_ids, args.track_overlap if args.hand_track_backend == "sam3-native" else 0)
            sam3_detect_suffix, _sam3_detect_range = chunk_suffix_and_args(sam3_detect_ids)
            if args.sam3_execution == "per-chunk":
                sam3_shards.append(args.base_dir / "sam3_bboxes" / "chunks" / camera_id / sam3_detect_suffix / f"sam3_bboxes_{sam3_detect_suffix}.jsonl")
            if args.hand_track_backend in {"posthoc", "sam3-native"}:
                track_ids = expanded_chunk_ids(chunk_ids, selected_group_ids, args.track_overlap if args.hand_track_backend == "sam3-native" else 0)
                track_suffix, _track_range = chunk_suffix_and_args(track_ids)
                stabilized_track_shards.append(args.base_dir / "sam3_tracks_stabilized" / "chunks" / camera_id / track_suffix / f"sam3_tracks_stabilized_{track_suffix}.jsonl")
            job_shards.append(args.base_dir / "hamer_jobs" / "chunks" / camera_id / chunk_suffix / f"hamer_jobs_{chunk_suffix}.jsonl")
            if args.hamer_execution == "per-chunk":
                hamer_shards.append(args.base_dir / "hamer_per_view" / f"hamer_predictions_{chunk_suffix}_{camera_id}.jsonl")
            heavy_tasks.append((f"heavy:{camera_id}:{chunk_suffix}", camera_id, chunk_ids))

    if args.dry_run:
        for label, camera_id, chunk_ids in heavy_tasks:
            tail = " -> HaMeR" if args.hamer_execution == "per-chunk" and not args.skip_hamer else ""
            emit(f"[pipeline] parallel {label}: SAM3 -> {args.hand_track_backend} identity -> fuse jobs{tail} for {camera_id} groups={chunk_ids[0]}-{chunk_ids[-1]}")
    else:
        positions: Queue[int] = Queue()
        for position in range(1, max_workers + 1):
            positions.put(position)

        def wrapped_heavy(camera_id: str, chunk_ids: list[int]) -> None:
            position = positions.get()
            try:
                heavy_chunk_pipeline(camera_id, chunk_ids, position)
            finally:
                positions.put(position)

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(wrapped_heavy, camera_id, chunk_ids): label for label, camera_id, chunk_ids in heavy_tasks}
            with tqdm(total=len(futures), desc="SAM3+HaMeR chunks", unit="chunk", position=0) as progress:
                for future in as_completed(futures):
                    label = futures[future]
                    try:
                        future.result()
                    except Exception as exc:
                        raise RuntimeError(f"camera heavy chunk failed ({label})") from exc
                    progress.update(1)

    if args.dry_run:
        if args.sam3_execution == "per-chunk":
            emit(f"[pipeline] merge SAM3 shards -> {sam3_output}")
        if args.hand_track_backend in {"posthoc", "sam3-native"}:
            emit(f"[pipeline] merge stabilized tracks -> {stabilized_tracks_output}")
        emit(f"[pipeline] merge HaMeR job shards -> {jobs_path}")
    else:
        sam3_count = None
        if args.sam3_execution == "per-chunk":
            sam3_count = merge_jsonl_unique(sam3_shards, sam3_output, "sam3_multiview_bboxes", ("type", "group_id", "camera_id"))
        job_count = merge_jsonl(job_shards, jobs_path, "hamer_multiview_jobs")
        if sam3_count is not None:
            emit(f"[pipeline] merged SAM3 records: {sam3_count} -> {sam3_output}")
        if args.hand_track_backend in {"posthoc", "sam3-native"}:
            track_count = merge_jsonl_unique(stabilized_track_shards, stabilized_tracks_output, "sam3_stabilized_tracks", ("type", "group_id", "camera_id"))
            emit(f"[pipeline] merged stabilized tracks: {track_count} -> {stabilized_tracks_output}")
        emit(f"[pipeline] merged HaMeR jobs: {job_count} -> {jobs_path}")

    if not args.skip_hamer and args.hamer_execution == "per-camera":
        hamer_commands = []
        for camera_id in cameras:
            camera_output = args.base_dir / "hamer_per_view" / f"hamer_predictions_{suffix}_{camera_id}.jsonl"
            hamer_shards.append(camera_output)
            hamer_commands.append((f"hamer:{camera_id}", hamer_command(jobs_path, camera_id, common_range)))
        run_parallel(hamer_commands, max_hamer_workers, args.dry_run, desc="HaMeR cameras")

    if not args.skip_hamer and args.hamer_execution == "per-sequence":
        run_command("hamer:sequence", hamer_command(jobs_path, None, common_range), args.dry_run)

    if not args.skip_hamer and args.hamer_execution != "per-sequence":
        if args.dry_run:
            emit(f"[pipeline] merge HaMeR shards -> {hamer_output}")
        else:
            hamer_count = merge_jsonl(hamer_shards, hamer_output, "hamer_multiview_prediction")
            emit(f"[pipeline] merged HaMeR predictions: {hamer_count} -> {hamer_output}")

    if args.run_zero_shot_palm_fusion:
        zero_shot_cmd = [
            sys.executable,
            "-u",
            script_path("fuse_hamer_palm_local.py"),
            "--predictions",
            str(hamer_output),
            "--output-dir",
            str(args.base_dir / "hamer_palm_local_fused"),
            "--quality-mask-weight",
            str(args.quality_mask_weight),
            "--quality-bbox-weight",
            str(args.quality_bbox_weight),
            "--quality-edge-weight",
            str(args.quality_edge_weight),
            "--quality-source-bonus",
            str(args.quality_source_bonus),
            "--quality-known-bonus",
            str(args.quality_known_bonus),
            "--bone-calibration-blend",
            str(args.zero_shot_bone_calibration_blend),
            "--temporal-radius",
            str(args.zero_shot_temporal_radius),
            "--temporal-sigma",
            str(args.zero_shot_temporal_sigma),
            "--causal-ema-alpha",
            str(args.zero_shot_causal_ema_alpha),
            "--one-euro-min-cutoff",
            str(args.zero_shot_one_euro_min_cutoff),
            "--one-euro-beta",
            str(args.zero_shot_one_euro_beta),
            "--one-euro-derivative-cutoff",
            str(args.zero_shot_one_euro_derivative_cutoff),
            "--frame-rate",
            str(args.frame_rate),
            "--primary-output",
            args.zero_shot_primary_output,
            "--include-vertices" if args.zero_shot_include_vertices else "--no-include-vertices",
            *common_range,
            *overwrite,
        ]
        run_command("zero_shot_palm_fusion", zero_shot_cmd, args.dry_run)

    local_cmd = [
        sys.executable,
        "-u",
        script_path("fuse_hamer_primary_local.py"),
        "--predictions",
        str(hamer_output),
        "--output-dir",
        str(args.base_dir / "hamer_primary_local"),
        "--temporal" if args.temporal else "--no-temporal",
        "--temporal-selection-weight",
        str(args.temporal_selection_weight),
        "--temporal-error-cap-m",
        str(args.temporal_error_cap_m),
        "--temporal-metric-alpha",
        str(args.temporal_metric_alpha),
        "--temporal-primary-alpha",
        str(args.temporal_primary_alpha),
        "--temporal-backup-alpha",
        str(args.temporal_backup_alpha),
        "--temporal-nonprimary-alpha",
        str(args.temporal_nonprimary_alpha),
        "--temporal-quality-anchor-alpha",
        str(args.temporal_quality_anchor_alpha),
        "--quality-mask-weight",
        str(args.quality_mask_weight),
        "--quality-bbox-weight",
        str(args.quality_bbox_weight),
        "--quality-edge-weight",
        str(args.quality_edge_weight),
        "--quality-source-bonus",
        str(args.quality_source_bonus),
        "--quality-known-bonus",
        str(args.quality_known_bonus),
        "--primary-prior-bonus",
        str(args.primary_prior_bonus),
        "--backup-prior-bonus",
        str(args.backup_prior_bonus),
        "--consensus-selection-weight",
        str(args.consensus_selection_weight),
        "--consensus-error-cap-m",
        str(args.consensus_error_cap_m),
        "--anchor-switch-margin",
        str(args.anchor_switch_margin),
        "--min-anchor-score",
        str(args.min_anchor_score),
        "--backup-primary-cameras",
        args.backup_primary_cameras,
        "--backup-min-mask-score",
        str(args.backup_min_mask_score),
        "--backup-min-view-quality",
        str(args.backup_min_view_quality),
        "--backup-max-temporal-error-m",
        str(args.backup_max_temporal_error_m),
        "--backup-require-known" if args.backup_require_known else "--no-backup-require-known",
        *common_range,
        *overwrite,
    ]
    if args.run_legacy_local_fusion or args.run_mano_local_refine:
        run_command("local_fusion", local_cmd, args.dry_run)

    if args.run_mano_local_refine and not args.skip_hamer:
        local_hands_path = args.base_dir / "hamer_primary_local" / f"hamer_local_hands_{suffix}.jsonl"
        mano_refine_cmd = [
            *conda_python(args.conda_bin, args.hamer_conda_env),
            script_path("refine_hamer_mano_local.py"),
            "--predictions",
            str(hamer_output),
            "--local-hands",
            str(local_hands_path),
            "--output-dir",
            str(args.base_dir / "hamer_mano_local_refined"),
            "--hamer-root",
            str(args.hamer_root),
            "--optimize-mano-pose" if args.optimize_mano_pose else "--no-optimize-mano-pose",
            "--optimize-mano-betas" if args.optimize_mano_betas else "--no-optimize-mano-betas",
            "--local-consistency-threshold-m",
            str(args.local_consistency_threshold_m),
            "--temporal-error-cap-m",
            str(args.temporal_error_cap_m),
            "--temporal-reject-threshold-m",
            str(args.temporal_reject_threshold_m),
            "--anchor-switch-margin",
            str(args.anchor_switch_margin),
            "--pose-prior-weight",
            str(args.pose_prior_weight),
            "--beta-prior-weight",
            str(args.beta_prior_weight),
            "--temporal-pose-weight",
            str(args.temporal_pose_weight),
            "--vertex-loss-weight",
            str(args.vertex_loss_weight),
            "--max-iters",
            str(args.mano_max_iters),
            "--beta-iters",
            str(args.mano_beta_iters),
            *common_range,
            *overwrite,
        ]
        if args.allow_frame_beta_delta:
            mano_refine_cmd.append("--allow-frame-beta-delta")
        run_command("mano_local_refine", mano_refine_cmd, args.dry_run)

    if args.run_mano_multiview_image_refine and not args.skip_hamer:
        image_refine_output_dir = (
            args.base_dir / "hamer_mano_multiview_refined_gated"
            if args.run_mano_image_candidate_selection
            else args.base_dir / "hamer_mano_multiview_refined"
        )
        mano_image_refine_cmd = [
            *conda_python(args.conda_bin, args.hamer_conda_env),
            script_path("refine_hamer_mano_multiview_image.py"),
            "--predictions",
            str(hamer_output),
            "--output-dir",
            str(image_refine_output_dir),
            "--hamer-root",
            str(args.hamer_root),
            "--calib",
            str(args.frames.parent / "cameras.yaml"),
            "--rectified-config",
            str(rectified_config_path),
            *(["--projection-correction", str(args.image_projection_correction)] if args.image_projection_correction else []),
            "--rectify-focal-scale",
            str(args.rectify_focal_scale),
            "--min-readable-sam3-mask-ratio",
            str(args.image_min_readable_sam3_mask_ratio),
            "--use-mediapipe-2d",
            args.use_mediapipe_2d,
            "--global-initialization",
            args.image_global_initialization,
            "--window-size",
            str(args.window_size),
            "--optimize-mano-pose" if args.optimize_mano_pose else "--no-optimize-mano-pose",
            "--optimize-mano-betas" if args.optimize_mano_betas else "--no-optimize-mano-betas",
            "--temporal-error-cap-m",
            str(args.temporal_error_cap_m),
            "--temporal-reject-threshold-m",
            str(args.temporal_reject_threshold_m),
            "--pose-prior-weight",
            str(args.pose_prior_weight),
            "--beta-prior-weight",
            str(args.beta_prior_weight),
            "--temporal-pose-weight",
            str(args.temporal_pose_weight),
            "--temporal-acceleration-weight",
            str(args.image_temporal_acceleration_weight),
            "--soft-reprojection-error-px",
            str(args.image_soft_reprojection_error_px),
            "--soft-mask-distance-px",
            str(args.image_soft_mask_distance_px),
            "--min-view-weight",
            str(args.image_min_view_weight),
            "--anchor-view-weight",
            str(args.image_anchor_view_weight),
            "--min-soft-used-weight",
            str(args.image_min_soft_used_weight),
            "--min-metric-used-weight",
            str(args.image_min_metric_used_weight),
            "--primary-anchor-score-margin",
            str(args.image_primary_anchor_score_margin),
            "--pnp-view-gate-m",
            str(args.image_pnp_view_gate_m),
            "--max-iters",
            str(args.mano_max_iters),
            "--beta-iters",
            str(args.mano_beta_iters),
            "--beta-estimation-space",
            args.image_beta_estimation_space,
            "--image-beta-max-observations",
            str(args.image_beta_max_observations),
            "--image-beta-prior-weight",
            str(args.image_beta_prior_weight),
            *common_range,
            *overwrite,
        ]
        if args.allow_frame_beta_delta:
            mano_image_refine_cmd.append("--allow-frame-beta-delta")
        # Per-frame overlays are useful for targeted inspection but add heavy I/O
        # during a production reconstruction. Projection diagnostics remain in JSON.
        mano_image_refine_cmd.append("--no-save-debug-overlays")
        if args.use_mediapipe_2d != "never":
            mano_image_refine_cmd.extend(["--mediapipe", str(mediapipe_path)])
        if args.run_mano_image_candidate_selection:
            baseline_cmd = list(mano_image_refine_cmd)
            baseline_cmd[baseline_cmd.index("--output-dir") + 1] = str(args.base_dir / "hamer_mano_multiview_refined_baseline")
            baseline_cmd[baseline_cmd.index("--pnp-view-gate-m") + 1] = "0"
            run_command("mano_multiview_image_refine_baseline", baseline_cmd, args.dry_run)
            run_command("mano_multiview_image_refine_gated", mano_image_refine_cmd, args.dry_run)
            baseline_output = args.base_dir / "hamer_mano_multiview_refined_baseline" / f"mano_multiview_local_hands_{suffix}.jsonl"
            gated_output = args.base_dir / "hamer_mano_multiview_refined_gated" / f"mano_multiview_local_hands_{suffix}.jsonl"
            selection_cmd = [
                sys.executable,
                script_path("select_image_refinement_candidates.py"),
                "--baseline",
                str(baseline_output),
                "--gated",
                str(gated_output),
                "--output",
                str(args.base_dir / "hamer_mano_multiview_selected" / f"mano_multiview_local_hands_{suffix}.jsonl"),
                "--min-baseline-max-reprojection-px",
                str(args.image_selection_min_baseline_max_reprojection_px),
                *overwrite,
            ]
            run_command("mano_multiview_image_select", selection_cmd, args.dry_run)
        else:
            run_command("mano_multiview_image_refine", mano_image_refine_cmd, args.dry_run)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        emit("[pipeline] Ctrl+C received, cleaning up child processes...")
        terminate_active_processes()
        raise SystemExit(130)
