#!/usr/bin/env python3
"""Run rectification, sparse SAM3 tracking, MobRecon, and palm-local fusion."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from hamer_multiview_utils import parse_cameras, parse_group_ids, range_suffix


REPO_ROOT = Path(__file__).resolve().parents[1]
from dependency_paths import DEFAULT_MOBRECON_ROOT, DEFAULT_SAM3_ROOT, default_conda_executable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument(
        "--image-root",
        type=Path,
        help="Root for image_path entries in frames.jsonl. Defaults to the frames file directory.",
    )
    parser.add_argument(
        "--calib",
        type=Path,
        help="Camera calibration YAML. Defaults to cameras.yaml beside frames.jsonl.",
    )
    parser.add_argument("--rectify-focal-scale", type=float, default=0.30)
    parser.add_argument(
        "--prepare-rectified",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prepare/reuse base-dir/rectified_for_hamer before inference.",
    )
    parser.add_argument("--cameras", default="C0,C2,C3")
    parser.add_argument("--group-range")
    parser.add_argument("--group-ids")
    parser.add_argument("--keyframe-stride", type=int, default=10)
    parser.add_argument("--flow-scale", type=float, default=0.25)
    parser.add_argument("--sam3-workers", type=int, choices=[1, 2], default=2)
    parser.add_argument("--execution-mode", choices=["streaming", "sequential"], default="streaming")
    parser.add_argument("--vram-budget-gib", type=float, default=24.0)
    parser.add_argument("--sam3-keyframes", type=Path, help="Reuse existing SAM3 JSONL instead of running sparse detection.")
    parser.add_argument("--reference-sam3", type=Path, help="Dense SAM3 JSONL used only for tracker evaluation.")
    parser.add_argument("--sam3-root", type=Path, default=DEFAULT_SAM3_ROOT)
    parser.add_argument("--sam3-checkpoint", type=Path)
    parser.add_argument("--sam3-no-hf", action="store_true")
    parser.add_argument("--sam3-amp-dtype", choices=["float32", "bfloat16", "float16"], default="float16")
    parser.add_argument("--sam3-torch-threads", type=int, default=2)
    parser.add_argument("--conda-bin", default=default_conda_executable())
    parser.add_argument("--sam3-conda-env", default="sam3hand")
    parser.add_argument("--sam3-python", type=Path, help="Explicit SAM3 Python; defaults to --sam3-conda-env.")
    parser.add_argument("--mobrecon-conda-env", default="hamer")
    parser.add_argument("--mobrecon-python", type=Path, help="Python with torch/OpenCV; defaults to current Python when available.")
    parser.add_argument("--mobrecon-root", type=Path, default=DEFAULT_MOBRECON_ROOT)
    parser.add_argument("--mobrecon-checkpoint", type=Path)
    parser.add_argument("--mobrecon-device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--mobrecon-precision", choices=["float32", "float16"], default="float32")
    parser.add_argument("--mobrecon-batch-size", type=int, default=8)
    parser.add_argument("--mobrecon-torch-threads", type=int, default=8)
    parser.add_argument("--one-euro-min-cutoff", type=float, default=0.25)
    parser.add_argument("--one-euro-beta", type=float, default=0.05)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def script_path(name: str) -> str:
    return str(REPO_ROOT / "scripts" / name)


def conda_python(conda_bin: str, environment: str) -> list[str]:
    return [conda_bin, "run", "--no-capture-output", "-n", environment, "python", "-u"]


def sam3_python(args: argparse.Namespace) -> list[str]:
    if args.sam3_python is not None and args.sam3_python.exists():
        return [str(args.sam3_python), "-u"]
    return conda_python(args.conda_bin, args.sam3_conda_env)


def mobrecon_python(args: argparse.Namespace) -> list[str]:
    if args.mobrecon_python is not None:
        return [str(args.mobrecon_python), "-u"]
    if importlib.util.find_spec("torch") is not None:
        return [sys.executable, "-u"]
    return conda_python(args.conda_bin, args.mobrecon_conda_env)


def range_args(args: argparse.Namespace) -> list[str]:
    output = []
    if args.group_range:
        output.extend(["--group-range", args.group_range])
    if args.group_ids:
        output.extend(["--group-ids", args.group_ids])
    return output


def run_command(label: str, command: list[str], dry_run: bool) -> float:
    print(f"[{label}] {' '.join(command)}", flush=True)
    if dry_run:
        return 0.0
    started = time.perf_counter()
    subprocess.run(command, check=True, cwd=REPO_ROOT)
    elapsed = time.perf_counter() - started
    print(f"[{label}] elapsed={elapsed:.3f}s", flush=True)
    return elapsed


def split_cameras(cameras: list[str], workers: int) -> list[list[str]]:
    groups = [[] for _ in range(min(workers, len(cameras)))]
    for index, camera_id in enumerate(cameras):
        groups[index % len(groups)].append(camera_id)
    return [group for group in groups if group]


def merge_jsonl(paths: list[Path], output_path: Path) -> int:
    records = []
    for path in paths:
        with path.open("r", encoding="utf-8") as source:
            records.extend(json.loads(line) for line in source if line.strip())
    records.sort(key=lambda record: (int(record["group_id"]), str(record["camera_id"])))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    return len(records)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as json_file:
        return json.load(json_file)


def sam3_memory_gate(config_paths: list[Path], budget_gib: float) -> tuple[list[float], float | None]:
    peaks = []
    for config_path in config_paths:
        value = read_json(config_path).get("peak_cuda_reserved_mib")
        if isinstance(value, (int, float)):
            peaks.append(float(value))
    if config_paths and len(peaks) != len(config_paths):
        raise RuntimeError("one or more SAM3 workers did not report peak CUDA reserved memory")
    peak_sum_mib = sum(peaks) if peaks else None
    budget_mib = budget_gib * 1024.0
    if peak_sum_mib is not None and peak_sum_mib > budget_mib:
        raise RuntimeError(f"SAM3 peak reserved {peak_sum_mib:.1f}MiB exceeds {budget_mib:.1f}MiB budget")
    return peaks, peak_sum_mib


def run_streaming_pipeline(
    sam3_commands: list[tuple[str, list[str]]], cpu_command: list[str]
) -> tuple[float, float]:
    started = time.perf_counter()
    sam3_processes: list[tuple[str, subprocess.Popen]] = []
    cpu_process: subprocess.Popen | None = None
    try:
        for label, command in sam3_commands:
            print(f"[{label}] {' '.join(command)}", flush=True)
            sam3_processes.append((label, subprocess.Popen(command, cwd=REPO_ROOT)))
        cpu_started = time.perf_counter()
        cpu_finished_at: float | None = None
        print(f"[realtime-cpu] {' '.join(cpu_command)}", flush=True)
        cpu_process = subprocess.Popen(cpu_command, cwd=REPO_ROOT)
        while True:
            for label, process in sam3_processes:
                return_code = process.poll()
                if return_code not in {None, 0}:
                    raise subprocess.CalledProcessError(return_code, process.args, output=f"{label} failed")
            cpu_return_code = cpu_process.poll()
            if cpu_return_code == 0 and cpu_finished_at is None:
                cpu_finished_at = time.perf_counter()
            if cpu_return_code not in {None, 0}:
                raise subprocess.CalledProcessError(cpu_return_code, cpu_process.args)
            if cpu_return_code == 0 and all(process.poll() == 0 for _label, process in sam3_processes):
                break
            time.sleep(0.05)
        cpu_seconds = (cpu_finished_at or time.perf_counter()) - cpu_started
        total_seconds = time.perf_counter() - started
        return total_seconds, cpu_seconds
    finally:
        processes = [process for _label, process in sam3_processes]
        if cpu_process is not None:
            processes.append(cpu_process)
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            if process.poll() is None:
                try:
                    process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    process.kill()


def main() -> None:
    args = parse_args()
    if (
        args.keyframe_stride < 1
        or args.vram_budget_gib <= 0.0
        or args.sam3_torch_threads < 1
        or args.mobrecon_torch_threads < 1
        or args.one_euro_min_cutoff < 0.0
        or args.one_euro_beta < 0.0
    ):
        raise SystemExit("stride, VRAM, and threads must be positive; One-Euro values must be non-negative")
    if args.rectify_focal_scale <= 0.0:
        raise SystemExit("--rectify-focal-scale must be positive")
    args.image_root = args.image_root or args.frames.parent
    args.calib = args.calib or (args.frames.parent / "cameras.yaml")
    if not args.frames.is_file():
        raise SystemExit(f"frames metadata not found: {args.frames}")
    if args.prepare_rectified and not args.image_root.is_dir():
        raise SystemExit(f"image root not found: {args.image_root}")
    if args.prepare_rectified and not args.calib.is_file():
        raise SystemExit(f"camera calibration not found: {args.calib}")
    if args.sam3_keyframes is not None and not args.sam3_keyframes.is_file():
        raise SystemExit(f"SAM3 keyframe JSONL not found: {args.sam3_keyframes}")
    if args.reference_sam3 is not None and not args.reference_sam3.is_file():
        raise SystemExit(f"reference SAM3 JSONL not found: {args.reference_sam3}")
    if args.sam3_keyframes is None and not (args.sam3_root / "sam3" / "__init__.py").is_file():
        raise SystemExit(f"SAM3 root not found: {args.sam3_root}")
    if args.sam3_checkpoint is not None and not args.sam3_checkpoint.is_file():
        raise SystemExit(f"SAM3 checkpoint not found: {args.sam3_checkpoint}")
    if not (args.mobrecon_root / "cmr" / "models" / "mobrecon_densestack.py").is_file():
        raise SystemExit(f"MobRecon root not found: {args.mobrecon_root}")
    if args.mobrecon_checkpoint is not None and not args.mobrecon_checkpoint.is_file():
        raise SystemExit(f"MobRecon checkpoint not found: {args.mobrecon_checkpoint}")
    cameras = sorted(parse_cameras(args.cameras))
    if len(cameras) < 2:
        raise SystemExit("realtime multi-view mode requires at least two cameras")
    group_ids = parse_group_ids(args.group_range, args.group_ids)
    suffix = range_suffix(group_ids)
    output_root = args.output_dir or (args.base_dir / "sam3_mobrecon_realtime")
    rectified_dir = args.base_dir / "rectified_for_hamer"
    overwrite = ["--overwrite"] if args.overwrite else []
    selected_range_args = range_args(args)
    stage_seconds: dict[str, float] = {}
    total_started = time.perf_counter()

    if args.prepare_rectified:
        rectify_command = [
            sys.executable,
            "-u",
            script_path("prepare_hamer_rectified.py"),
            "--image-root",
            str(args.image_root),
            "--frames",
            str(args.frames),
            "--calib",
            str(args.calib),
            "--output-dir",
            str(rectified_dir),
            "--cameras",
            ",".join(cameras),
            "--rectify-focal-scale",
            str(args.rectify_focal_scale),
            *range_args(args),
            *(["--overwrite"] if args.overwrite else []),
        ]
        stage_seconds["rectify"] = run_command("rectify", rectify_command, args.dry_run)
    elif not args.dry_run and not rectified_dir.is_dir():
        raise SystemExit(f"rectified image cache not found: {rectified_dir}")

    keyframe_reused = args.sam3_keyframes is not None
    sam3_configs: list[Path] = []
    commands: list[tuple[str, list[str]]] = []
    shard_paths: list[Path] = []
    if args.sam3_keyframes is not None:
        keyframe_path = args.sam3_keyframes
    else:
        camera_groups = split_cameras(cameras, args.sam3_workers)
        for worker_index, camera_group in enumerate(camera_groups):
            worker_dir = output_root / "sam3_keyframes" / f"worker_{worker_index}"
            shard_path = worker_dir / f"sam3_bboxes_{suffix}.jsonl"
            config_path = worker_dir / f"sam3_config_{suffix}.json"
            command = [
                *sam3_python(args),
                script_path("detect_sam3_hands_multiview.py"),
                "--frames",
                str(args.frames),
                "--rectified-dir",
                str(rectified_dir),
                "--output-dir",
                str(worker_dir),
                "--sam3-root",
                str(args.sam3_root),
                "--cameras",
                ",".join(camera_group),
                "--frame-stride",
                str(args.keyframe_stride),
                "--prompt-preset",
                "realtime",
                "--amp-dtype",
                args.sam3_amp_dtype,
                "--torch-threads",
                str(args.sam3_torch_threads),
                "--stream-output",
                "--no-save-mask-debug",
                "--no-save-bbox-debug",
                *selected_range_args,
                *overwrite,
            ]
            if args.sam3_checkpoint:
                command.extend(["--checkpoint", str(args.sam3_checkpoint)])
            if args.sam3_no_hf:
                command.append("--no-hf")
            commands.append((f"sam3:{worker_index}", command))
            shard_paths.append(shard_path)
            sam3_configs.append(config_path)
        if args.dry_run:
            for label, command in commands:
                run_command(label, command, True)
            stage_seconds["sam3_keyframes"] = 0.0
        elif args.execution_mode == "sequential":
            started = time.perf_counter()
            with ThreadPoolExecutor(max_workers=len(commands)) as executor:
                futures = [executor.submit(run_command, label, command, False) for label, command in commands]
                for future in futures:
                    future.result()
            stage_seconds["sam3_keyframes"] = time.perf_counter() - started
        keyframe_path = output_root / "sam3_keyframes" / f"sam3_keyframes_{suffix}.jsonl"
        if not args.dry_run and args.execution_mode == "sequential":
            merge_jsonl(shard_paths, keyframe_path)

    if args.execution_mode == "streaming":
        cpu_dir = output_root / "realtime_cpu"
        cpu_config_path = cpu_dir / f"mobrecon_realtime_config_{suffix}.json"
        cpu_command = [
            *mobrecon_python(args),
            script_path("mobrecon_realtime_cpu.py"),
            "--frames",
            str(args.frames),
            "--rectified-dir",
            str(rectified_dir),
            "--output-dir",
            str(cpu_dir),
            "--mobrecon-root",
            str(args.mobrecon_root),
            "--cameras",
            ",".join(cameras),
            "--keyframe-stride",
            str(args.keyframe_stride),
            "--flow-scale",
            str(args.flow_scale),
            "--batch-size",
            str(args.mobrecon_batch_size),
            "--torch-threads",
            str(args.mobrecon_torch_threads),
            "--one-euro-min-cutoff",
            str(args.one_euro_min_cutoff),
            "--one-euro-beta",
            str(args.one_euro_beta),
            "--microbatch-groups",
            "1",
            "--device",
            args.mobrecon_device,
            "--precision",
            args.mobrecon_precision,
            *selected_range_args,
            *overwrite,
        ]
        if args.mobrecon_checkpoint:
            cpu_command.extend(["--checkpoint", str(args.mobrecon_checkpoint)])
        if keyframe_reused:
            cpu_command.extend(["--keyframe-sam3", str(keyframe_path)])
        else:
            for shard_path in shard_paths:
                cpu_command.extend(["--keyframe-shard", str(shard_path)])
            cpu_command.append("--follow-keyframes")
        if args.dry_run:
            run_command("realtime-cpu", cpu_command, True)
            return
        if not keyframe_reused:
            for stale_path in [*shard_paths, *sam3_configs]:
                if not stale_path.exists():
                    continue
                if not args.overwrite:
                    raise SystemExit(f"{stale_path} exists; pass --overwrite to replace it")
                stale_path.unlink()
        streaming_seconds, cpu_seconds = run_streaming_pipeline(commands, cpu_command)
        stage_seconds["streaming_overlap"] = streaming_seconds
        stage_seconds["realtime_cpu"] = cpu_seconds
        if not keyframe_reused:
            merge_jsonl(shard_paths, keyframe_path)
        sam3_peak_reserved, peak_sum_mib = sam3_memory_gate(sam3_configs, args.vram_budget_gib)
        total_seconds = time.perf_counter() - total_started
        cpu_config = read_json(cpu_config_path)
        frame_count = float(cpu_config["stats"]["frames"])
        warm_processing_seconds = cpu_config["timing"].get("warm_processing_seconds")
        steady_state_fps = cpu_config["timing"].get("warm_processing_fps")
        startup_seconds = (
            total_seconds - float(warm_processing_seconds)
            if isinstance(warm_processing_seconds, (int, float))
            else None
        )
        config = {
            "base_dir": str(args.base_dir),
            "frames_metadata": str(args.frames),
            "image_root": str(args.image_root),
            "camera_calibration": str(args.calib),
            "rectify_focal_scale": args.rectify_focal_scale,
            "prepare_rectified": args.prepare_rectified,
            "cameras": cameras,
            "keyframe_stride": args.keyframe_stride,
            "sam3_torch_threads": args.sam3_torch_threads,
            "mobrecon_torch_threads": args.mobrecon_torch_threads,
            "mobrecon_root": str(args.mobrecon_root),
            "mobrecon_checkpoint": str(args.mobrecon_checkpoint) if args.mobrecon_checkpoint else None,
            "one_euro_min_cutoff": args.one_euro_min_cutoff,
            "one_euro_beta": args.one_euro_beta,
            "mobrecon_precision": args.mobrecon_precision,
            "execution_mode": "streaming",
            "sam3_workers": args.sam3_workers if not keyframe_reused else 0,
            "keyframe_inference_reused": keyframe_reused,
            "vram_budget_gib": args.vram_budget_gib,
            "sam3_worker_peak_reserved_mib": sam3_peak_reserved,
            "sam3_peak_reserved_sum_mib": peak_sum_mib,
            "stage_seconds": stage_seconds,
            "total_seconds": total_seconds,
            "frame_count": frame_count,
            "end_to_end_fps": frame_count / total_seconds if not keyframe_reused and total_seconds else None,
            "steady_state_fps": steady_state_fps,
            "startup_seconds": startup_seconds,
            "reused_keyframe_pipeline_fps": frame_count / total_seconds if keyframe_reused and total_seconds else None,
            "outputs": {
                "keyframes": str(keyframe_path),
                "predictions": cpu_config["outputs"]["predictions"],
                "fused": cpu_config["outputs"]["fused"],
                "cpu_config": str(cpu_config_path),
            },
        }
        config_path = output_root / f"realtime_config_{suffix}.json"
        with config_path.open("w", encoding="utf-8") as config_file:
            json.dump(config, config_file, ensure_ascii=False, indent=2)
            config_file.write("\n")
        print(json.dumps(config, ensure_ascii=False, indent=2))
        print(f"wrote: {config_path}")
        return

    if args.dry_run:
        sam3_peak_reserved, peak_sum_mib = [], None
    else:
        sam3_peak_reserved, peak_sum_mib = sam3_memory_gate(sam3_configs, args.vram_budget_gib)

    tracks_dir = output_root / "sam3_sparse_tracks"
    tracks_path = tracks_dir / f"sam3_sparse_tracks_{suffix}.jsonl"
    tracker_command = [
        sys.executable,
        "-u",
        script_path("track_sam3_sparse_keyframes.py"),
        "--frames",
        str(args.frames),
        "--rectified-dir",
        str(rectified_dir),
        "--keyframe-sam3",
        str(keyframe_path),
        "--output-dir",
        str(tracks_dir),
        "--cameras",
        ",".join(cameras),
        "--keyframe-stride",
        str(args.keyframe_stride),
        "--flow-scale",
        str(args.flow_scale),
        *selected_range_args,
        *overwrite,
    ]
    if args.reference_sam3:
        tracker_command.extend(["--reference-sam3", str(args.reference_sam3)])
    stage_seconds["tracking"] = run_command("tracking", tracker_command, args.dry_run)

    jobs_dir = output_root / "jobs"
    jobs_path = jobs_dir / f"hamer_jobs_{suffix}.jsonl"
    jobs_command = [
        sys.executable,
        "-u",
        script_path("fuse_hamer_jobs.py"),
        "--frames",
        str(args.frames),
        "--rectified-dir",
        str(rectified_dir),
        "--tracked-hands",
        str(tracks_path),
        "--output-dir",
        str(jobs_dir),
        "--cameras",
        ",".join(cameras),
        "--no-use-mediapipe",
        "--mask-frame-mode",
        "none",
        "--no-save-debug",
        "--camera-handedness-prior",
        "none",
        *selected_range_args,
        *overwrite,
    ]
    stage_seconds["jobs"] = run_command("jobs", jobs_command, args.dry_run)

    mobrecon_dir = output_root / "mobrecon_per_view"
    predictions_path = mobrecon_dir / f"mobrecon_predictions_{suffix}.jsonl"
    mobrecon_command = [
        *mobrecon_python(args),
        script_path("mobrecon_multiview_worker.py"),
        "--jobs",
        str(jobs_path),
        "--output-dir",
        str(mobrecon_dir),
        "--mobrecon-root",
        str(args.mobrecon_root),
        "--batch-size",
        str(args.mobrecon_batch_size),
        "--torch-threads",
        str(args.mobrecon_torch_threads),
        "--image-source",
        "rectified",
        "--device",
        args.mobrecon_device,
        "--precision",
        args.mobrecon_precision,
        *selected_range_args,
        *overwrite,
    ]
    if args.mobrecon_checkpoint:
        mobrecon_command.extend(["--checkpoint", str(args.mobrecon_checkpoint)])
    stage_seconds["mobrecon"] = run_command("mobrecon", mobrecon_command, args.dry_run)

    fused_dir = output_root / "palm_local_fused"
    fusion_command = [
        sys.executable,
        "-u",
        script_path("fuse_hamer_palm_local.py"),
        "--predictions",
        str(predictions_path),
        "--output-dir",
        str(fused_dir),
        *selected_range_args,
        *overwrite,
    ]
    if args.one_euro_min_cutoff > 0.0:
        fusion_command.extend(
            [
                "--one-euro-min-cutoff",
                str(args.one_euro_min_cutoff),
                "--one-euro-beta",
                str(args.one_euro_beta),
                "--primary-output",
                "adaptive-causal",
            ]
        )
    stage_seconds["fusion"] = run_command("fusion", fusion_command, args.dry_run)
    if args.dry_run:
        return

    total_seconds = time.perf_counter() - total_started
    tracker_config = read_json(tracks_dir / f"sam3_sparse_tracks_config_{suffix}.json")
    frame_count = int(tracker_config["stats"]["records"]) / len(cameras)
    config = {
        "base_dir": str(args.base_dir),
        "frames_metadata": str(args.frames),
        "image_root": str(args.image_root),
        "camera_calibration": str(args.calib),
        "rectify_focal_scale": args.rectify_focal_scale,
        "prepare_rectified": args.prepare_rectified,
        "cameras": cameras,
        "keyframe_stride": args.keyframe_stride,
        "sam3_torch_threads": args.sam3_torch_threads,
        "mobrecon_torch_threads": args.mobrecon_torch_threads,
        "mobrecon_root": str(args.mobrecon_root),
        "mobrecon_checkpoint": str(args.mobrecon_checkpoint) if args.mobrecon_checkpoint else None,
        "one_euro_min_cutoff": args.one_euro_min_cutoff,
        "one_euro_beta": args.one_euro_beta,
        "mobrecon_precision": args.mobrecon_precision,
        "execution_mode": "sequential",
        "sam3_workers": args.sam3_workers if not keyframe_reused else 0,
        "keyframe_inference_reused": keyframe_reused,
        "vram_budget_gib": args.vram_budget_gib,
        "sam3_worker_peak_reserved_mib": sam3_peak_reserved,
        "sam3_peak_reserved_sum_mib": peak_sum_mib,
        "stage_seconds": stage_seconds,
        "total_seconds": total_seconds,
        "frame_count": frame_count,
        "end_to_end_fps": frame_count / total_seconds if not keyframe_reused and total_seconds else None,
        "reused_keyframe_pipeline_fps": frame_count / total_seconds if keyframe_reused and total_seconds else None,
        "outputs": {
            "keyframes": str(keyframe_path),
            "tracks": str(tracks_path),
            "jobs": str(jobs_path),
            "predictions": str(predictions_path),
            "fused": str(fused_dir / f"palm_local_hands_{suffix}.jsonl"),
        },
    }
    config_path = output_root / f"realtime_config_{suffix}.json"
    with config_path.open("w", encoding="utf-8") as config_file:
        json.dump(config, config_file, ensure_ascii=False, indent=2)
        config_file.write("\n")
    print(json.dumps(config, ensure_ascii=False, indent=2))
    print(f"wrote: {config_path}")


if __name__ == "__main__":
    main()
