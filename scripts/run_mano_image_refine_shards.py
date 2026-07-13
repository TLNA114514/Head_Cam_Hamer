#!/usr/bin/env python3
"""Run baseline/gated MANO image refinement shards and select candidates."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
from pathlib import Path

from hamer_multiview_utils import DEFAULT_CAMERAS, parse_cameras, parse_group_ids, range_suffix
from progress_utils import tqdm


WRIST_CAM_ROOT = Path("/home/luojiangrui/ljr/wrist_cam")


ACTIVE_PROCS: set[subprocess.Popen] = set()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--group-range", required=True)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--cameras", default=",".join(DEFAULT_CAMERAS))
    parser.add_argument("--conda-bin", default="/home/luojiangrui/miniconda3/bin/conda")
    parser.add_argument("--hamer-conda-env", default="hamer")
    parser.add_argument("--hamer-root", type=Path, default=WRIST_CAM_ROOT / "third_party" / "hamer")
    parser.add_argument("--calib", type=Path, required=True)
    parser.add_argument("--rectified-config", type=Path, required=True)
    parser.add_argument("--rectify-focal-scale", type=float, default=0.30)
    parser.add_argument("--pnp-view-gate-m", type=float, default=0.04)
    parser.add_argument("--min-readable-sam3-mask-ratio", type=float, default=0.50)
    parser.add_argument("--window-size", type=int, default=7)
    parser.add_argument("--temporal-error-cap-m", type=float, default=0.08)
    parser.add_argument("--temporal-reject-threshold-m", type=float, default=0.12)
    parser.add_argument("--pose-prior-weight", type=float, default=0.25)
    parser.add_argument("--beta-prior-weight", type=float, default=1.0)
    parser.add_argument("--temporal-pose-weight", type=float, default=0.35)
    parser.add_argument("--temporal-acceleration-weight", type=float, default=0.0)
    parser.add_argument("--soft-reprojection-error-px", type=float, default=90.0)
    parser.add_argument("--soft-mask-distance-px", type=float, default=90.0)
    parser.add_argument("--min-view-weight", type=float, default=0.12)
    parser.add_argument("--anchor-view-weight", type=float, default=1.45)
    parser.add_argument("--min-soft-used-weight", type=float, default=0.08)
    parser.add_argument("--min-metric-used-weight", type=float, default=0.28)
    parser.add_argument("--primary-anchor-score-margin", type=float, default=0.08)
    parser.add_argument("--max-iters", type=int, default=80)
    parser.add_argument("--beta-iters", type=int, default=120)
    parser.add_argument("--beta-estimation-space", choices=["hamer-local", "image-2d"], default="hamer-local")
    parser.add_argument("--image-beta-max-observations", type=int, default=240)
    parser.add_argument("--image-beta-prior-weight", type=float, default=2.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def chunk_group_ids(group_ids: set[int], chunk_size: int) -> list[set[int]]:
    ordered = sorted(group_ids)
    chunks = []
    for start in range(0, len(ordered), chunk_size):
        chunk = ordered[start : start + chunk_size]
        chunks.append(set(chunk))
    return chunks


def script_path(name: str) -> str:
    return str((Path(__file__).resolve().parent / name).resolve())


def terminate_active_processes() -> None:
    for proc in list(ACTIVE_PROCS):
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    for proc in list(ACTIVE_PROCS):
        if proc.poll() is None:
            try:
                proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass


def run_command(label: str, command: list[str], dry_run: bool) -> None:
    print(f"[{label}] {' '.join(command)}", flush=True)
    if dry_run:
        return
    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=Path.cwd(),
        preexec_fn=os.setsid,
    )
    ACTIVE_PROCS.add(proc)
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line.rstrip(), flush=True)
        code = proc.wait()
    finally:
        ACTIVE_PROCS.discard(proc)
    if code != 0:
        raise subprocess.CalledProcessError(code, command)


def refine_command(
    args: argparse.Namespace,
    output_dir: Path,
    gate_m: float,
    prediction_paths: list[Path],
    range_text: str,
) -> list[str]:
    command = [
        args.conda_bin,
        "run",
        "-n",
        args.hamer_conda_env,
        "python",
        "-u",
        "-s",
        script_path("refine_hamer_mano_multiview_image.py"),
    ]
    for path in prediction_paths:
        command.extend(["--predictions", str(path)])
    command.extend(
        [
            "--output-dir",
            str(output_dir),
            "--hamer-root",
            str(args.hamer_root),
            "--calib",
            str(args.calib),
            "--rectified-config",
            str(args.rectified_config),
            "--rectify-focal-scale",
            str(args.rectify_focal_scale),
            "--min-readable-sam3-mask-ratio",
            str(args.min_readable_sam3_mask_ratio),
            "--use-mediapipe-2d",
            "never",
            "--global-initialization",
            "physical-pnp",
            "--window-size",
            str(args.window_size),
            "--optimize-mano-pose",
            "--optimize-mano-betas",
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
            str(args.temporal_acceleration_weight),
            "--soft-reprojection-error-px",
            str(args.soft_reprojection_error_px),
            "--soft-mask-distance-px",
            str(args.soft_mask_distance_px),
            "--min-view-weight",
            str(args.min_view_weight),
            "--anchor-view-weight",
            str(args.anchor_view_weight),
            "--min-soft-used-weight",
            str(args.min_soft_used_weight),
            "--min-metric-used-weight",
            str(args.min_metric_used_weight),
            "--primary-anchor-score-margin",
            str(args.primary_anchor_score_margin),
            "--pnp-view-gate-m",
            str(gate_m),
            "--max-iters",
            str(args.max_iters),
            "--beta-iters",
            str(args.beta_iters),
            "--beta-estimation-space",
            args.beta_estimation_space,
            "--image-beta-max-observations",
            str(args.image_beta_max_observations),
            "--image-beta-prior-weight",
            str(args.image_beta_prior_weight),
            "--group-range",
            range_text,
            "--no-save-debug-overlays",
        ]
    )
    if args.overwrite:
        command.append("--overwrite")
    return command


def main() -> None:
    args = parse_args()
    group_ids = parse_group_ids(args.group_range, None)
    if not group_ids:
        raise SystemExit("--group-range must select at least one group")
    cameras = sorted(parse_cameras(args.cameras))
    chunks = chunk_group_ids(group_ids, args.chunk_size)

    baseline_dir = args.base_dir / "hamer_mano_multiview_refined_baseline"
    gated_dir = args.base_dir / "hamer_mano_multiview_refined_gated"
    selected_dir = args.base_dir / "hamer_mano_multiview_selected"

    for chunk in tqdm(chunks, desc="image-refine shards", unit="shard"):
        suffix = range_suffix(chunk)
        range_text = f"{min(chunk)}-{max(chunk)}"
        prediction_paths = [
            args.base_dir / "hamer_per_view" / f"hamer_predictions_{suffix}_{camera_id}.jsonl"
            for camera_id in cameras
        ]
        missing = [path for path in prediction_paths if not path.exists()]
        if missing:
            raise SystemExit(f"Missing prediction shards for {suffix}: {', '.join(str(path) for path in missing)}")

        baseline_output = baseline_dir / f"mano_multiview_local_hands_{suffix}.jsonl"
        gated_output = gated_dir / f"mano_multiview_local_hands_{suffix}.jsonl"
        selected_output = selected_dir / f"mano_multiview_local_hands_{suffix}.jsonl"

        if not baseline_output.exists() or args.overwrite:
            cmd = refine_command(args, baseline_dir, 0.0, prediction_paths, range_text)
            run_command(f"baseline {suffix}", cmd, args.dry_run)
        else:
            print(f"[baseline {suffix}] exists, skipping", flush=True)

        if not gated_output.exists() or args.overwrite:
            cmd = refine_command(args, gated_dir, args.pnp_view_gate_m, prediction_paths, range_text)
            run_command(f"gated {suffix}", cmd, args.dry_run)
        else:
            print(f"[gated {suffix}] exists, skipping", flush=True)

        if not selected_output.exists() or args.overwrite:
            command = [
                sys.executable,
                script_path("select_image_refinement_candidates.py"),
                "--baseline",
                str(baseline_output),
                "--gated",
                str(gated_output),
                "--output",
                str(selected_output),
            ]
            if args.overwrite:
                command.append("--overwrite")
            run_command(f"select {suffix}", command, args.dry_run)
        else:
            print(f"[select {suffix}] exists, skipping", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("[image-refine shards] Ctrl+C received, cleaning up child processes...", flush=True)
        terminate_active_processes()
        raise SystemExit(130)
