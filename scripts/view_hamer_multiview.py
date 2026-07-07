#!/usr/bin/env python3
"""Convenience launcher for the multi-view HaMeR local-hand viewer."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from hamer_multiview_utils import DEFAULT_BASE_DIR, parse_group_ids, range_suffix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument("--triangulated", type=Path, help="Explicit hamer_local_hands JSONL path.")
    parser.add_argument("--detections", type=Path, help="MediaPipe landmarks JSONL. Defaults to base-dir/landmarks.jsonl.")
    parser.add_argument("--group-range", help='Inclusive group range, e.g. "1-100".')
    parser.add_argument("--group-ids", help='Comma-separated ids, e.g. "1,7,42"; only used for file suffix selection.')
    parser.add_argument("--frames", type=Path, default=Path("video/cameras/frames.jsonl"))
    parser.add_argument("--image-root", type=Path, default=Path("video/cameras"))
    parser.add_argument("--calib", type=Path, default=Path("video/cameras/cameras.yaml"))
    parser.add_argument("--space", choices=["world", "root-relative", "palm-local"], default="palm-local")
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--interval-ms", type=int, default=40)
    parser.add_argument("--fixed-range", type=float, default=0.35)
    parser.add_argument("--image-downscale", type=int, default=2)
    parser.add_argument("--elev", type=float, default=20.0)
    parser.add_argument("--azim", type=float, default=-65.0)
    parser.add_argument("--render-mode", choices=["skeleton", "mesh", "mesh+skeleton"], default="mesh+skeleton")
    parser.add_argument("--mesh-stride", type=int, default=8)
    parser.add_argument("--no-mediapipe-overlay", action="store_true")
    parser.add_argument("--no-camera-rig", action="store_true")
    return parser.parse_args()


def default_triangulated_path(base_dir: Path, group_range: str | None, group_ids: str | None) -> Path:
    suffix = range_suffix(parse_group_ids(group_range, group_ids))
    refined = base_dir / "hamer_mano_local_refined" / f"mano_local_hands_{suffix}.jsonl"
    if refined.exists():
        return refined
    ranged = base_dir / "hamer_primary_local" / f"hamer_local_hands_{suffix}.jsonl"
    if ranged.exists():
        return ranged
    return base_dir / "hamer_primary_local" / "hamer_local_hands.jsonl"


def main() -> None:
    args = parse_args()
    viewer = Path(__file__).resolve().parent / "view_triangulated_hands_rgb.py"
    triangulated = args.triangulated or default_triangulated_path(args.base_dir, args.group_range, args.group_ids)
    detections = args.detections or (args.base_dir / "landmarks.jsonl")

    command = [
        sys.executable,
        str(viewer),
        "--triangulated",
        str(triangulated),
        "--frames",
        str(args.frames),
        "--image-root",
        str(args.image_root),
        "--calib",
        str(args.calib),
        "--detections",
        str(detections),
        "--space",
        args.space,
        "--stride",
        str(args.stride),
        "--interval-ms",
        str(args.interval_ms),
        "--fixed-range",
        str(args.fixed_range),
        "--image-downscale",
        str(args.image_downscale),
        "--elev",
        str(args.elev),
        "--azim",
        str(args.azim),
        "--render-mode",
        args.render_mode,
        "--mesh-stride",
        str(args.mesh_stride),
    ]
    if args.group_range:
        command.extend(["--group-range", args.group_range])
    if args.max_frames is not None:
        command.extend(["--max-frames", str(args.max_frames)])
    if args.no_mediapipe_overlay:
        command.append("--no-mediapipe-overlay")
    if args.no_camera_rig:
        command.append("--no-show-camera-rig")

    print(" ".join(command), flush=True)
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
