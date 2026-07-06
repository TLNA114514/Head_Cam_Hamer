#!/usr/bin/env python3
"""Orchestrate the SAM3 + MediaPipe-guided multi-view HaMeR pipeline."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
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
    parser.add_argument("--frames", type=Path, default=DEFAULT_FRAMES)
    parser.add_argument("--mediapipe", type=Path, help="MediaPipe landmarks JSONL. Defaults to base-dir/landmarks.jsonl.")
    parser.add_argument("--rectify-focal-scale", type=float, default=0.30)
    parser.add_argument("--group-range")
    parser.add_argument("--group-ids")
    parser.add_argument("--cameras", default=",".join(DEFAULT_CAMERAS))
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--max-parallel-workers", type=int, default=2)
    parser.add_argument("--max-mediapipe-workers", type=int, default=4)
    parser.add_argument("--conda-bin", default="/home/luojiangrui/miniconda3/bin/conda")
    parser.add_argument("--sam3-conda-env", default="sam3hand")
    parser.add_argument("--hamer-conda-env", default="hamer")
    parser.add_argument("--sam3-root", type=Path, default=WRIST_CAM_ROOT / "third_party" / "sam3")
    parser.add_argument("--hamer-root", type=Path, default=WRIST_CAM_ROOT / "third_party" / "hamer")
    parser.add_argument("--prompt-preset", choices=["bare", "gloved", "custom"], default="bare")
    parser.add_argument("--prompt", action="append", dest="prompts")
    parser.add_argument(
        "--camera-handedness-override",
        default="C0:Left,C3:Right",
        help='Fusion-stage camera handedness overrides, e.g. "C0:Left,C3:Right"; use "none" to disable.',
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
        emit(f"[pipeline] done {label}")
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
    emit(f"[pipeline] done {label}")


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


def main() -> None:
    args = parse_args()
    cameras = sorted(parse_cameras(args.cameras))
    camera_set = set(cameras)
    group_ids = parse_group_ids(args.group_range, args.group_ids)
    suffix = range_suffix(group_ids)
    max_workers = max(1, min(2, int(args.max_parallel_workers)))
    max_mediapipe_workers = max(1, min(len(cameras), int(args.max_mediapipe_workers)))
    selected_group_ids = group_ids_from_frames(args.frames, camera_set, group_ids)
    chunks = chunk_group_ids(selected_group_ids, args.chunk_size)
    if not selected_group_ids:
        raise SystemExit("no group ids selected from frames")

    if not args.sam3_root.exists():
        raise SystemExit(f"SAM3 root not found: {args.sam3_root}. Reuse/setup wrist_cam first.")
    if not args.hamer_root.exists():
        raise SystemExit(f"HaMeR root not found: {args.hamer_root}. Reuse/setup wrist_cam first.")

    common_range = []
    if args.group_range:
        common_range.extend(["--group-range", args.group_range])
    if args.group_ids:
        common_range.extend(["--group-ids", args.group_ids])
    overwrite = ["--overwrite"] if args.overwrite else []
    mediapipe_path = args.mediapipe or (args.base_dir / "landmarks.jsonl")

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
    jobs_path = args.base_dir / "hamer_jobs" / f"hamer_jobs_{suffix}.jsonl"
    hamer_output = args.base_dir / "hamer_per_view" / f"hamer_predictions_{suffix}.jsonl"
    sam3_shards = []
    job_shards = []
    hamer_shards = []

    def heavy_chunk_pipeline(camera_id: str, chunk_ids: list[int], progress_position: int) -> None:
        chunk_suffix, chunk_range = chunk_suffix_and_args(chunk_ids)
        label_suffix = f"{camera_id}:{chunk_suffix}"
        sam3_dir = args.base_dir / "sam3_bboxes" / "chunks" / camera_id / chunk_suffix
        sam3_shard = sam3_dir / f"sam3_bboxes_{chunk_suffix}.jsonl"
        jobs_dir = args.base_dir / "hamer_jobs" / "chunks" / camera_id / chunk_suffix
        jobs_shard = jobs_dir / f"hamer_jobs_{chunk_suffix}.jsonl"

        if not args.skip_sam3:
            sam3_cmd = [
                *conda_python(args.conda_bin, args.sam3_conda_env),
                script_path("detect_sam3_hands_multiview.py"),
                "--frames",
                str(args.frames),
                "--rectified-dir",
                str(args.base_dir / "rectified_for_hamer"),
                "--output-dir",
                str(sam3_dir),
                "--sam3-root",
                str(args.sam3_root),
                "--cameras",
                camera_id,
                "--prompt-preset",
                args.prompt_preset,
                *chunk_range,
                *overwrite,
            ]
            for prompt in args.prompts or []:
                sam3_cmd.extend(["--prompt", prompt])
            run_command(f"sam3:{label_suffix}", sam3_cmd, False, progress_position)

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
            *chunk_range,
            *overwrite,
        ]
        run_command(f"fuse_jobs:{label_suffix}", fusion_cmd, False, progress_position)

        if not args.skip_hamer:
            hamer_cmd = [
                *conda_python(args.conda_bin, args.hamer_conda_env),
                script_path("hamer_multiview_worker.py"),
                "--jobs",
                str(jobs_shard),
                "--output-dir",
                str(args.base_dir / "hamer_per_view"),
                "--hamer-root",
                str(args.hamer_root),
                "--camera-id",
                camera_id,
                *chunk_range,
                *overwrite,
            ]
            run_command(f"hamer:{label_suffix}", hamer_cmd, False, progress_position)

    heavy_tasks: list[tuple[str, str, list[int]]] = []
    for chunk_ids in chunks:
        for camera_id in cameras:
            chunk_suffix, _chunk_range = chunk_suffix_and_args(chunk_ids)
            sam3_shards.append(args.base_dir / "sam3_bboxes" / "chunks" / camera_id / chunk_suffix / f"sam3_bboxes_{chunk_suffix}.jsonl")
            job_shards.append(args.base_dir / "hamer_jobs" / "chunks" / camera_id / chunk_suffix / f"hamer_jobs_{chunk_suffix}.jsonl")
            hamer_shards.append(args.base_dir / "hamer_per_view" / f"hamer_predictions_{chunk_suffix}_{camera_id}.jsonl")
            heavy_tasks.append((f"heavy:{camera_id}:{chunk_suffix}", camera_id, chunk_ids))

    if args.dry_run:
        for label, camera_id, chunk_ids in heavy_tasks:
            emit(f"[pipeline] parallel {label}: SAM3 -> fuse jobs -> HaMeR for {camera_id} groups={chunk_ids[0]}-{chunk_ids[-1]}")
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
        emit(f"[pipeline] merge SAM3 shards -> {sam3_output}")
        emit(f"[pipeline] merge HaMeR job shards -> {jobs_path}")
        emit(f"[pipeline] merge HaMeR shards -> {hamer_output}")
    else:
        sam3_count = merge_jsonl(sam3_shards, sam3_output, "sam3_multiview_bboxes")
        job_count = merge_jsonl(job_shards, jobs_path, "hamer_multiview_jobs")
        emit(f"[pipeline] merged SAM3 records: {sam3_count} -> {sam3_output}")
        emit(f"[pipeline] merged HaMeR jobs: {job_count} -> {jobs_path}")
        if not args.skip_hamer:
            hamer_count = merge_jsonl(hamer_shards, hamer_output, "hamer_multiview_prediction")
            emit(f"[pipeline] merged HaMeR predictions: {hamer_count} -> {hamer_output}")

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
    run_command("local_fusion", local_cmd, args.dry_run)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        emit("[pipeline] Ctrl+C received, cleaning up child processes...")
        terminate_active_processes()
        raise SystemExit(130)
