#!/usr/bin/env python3
"""Convenience launcher for the multi-view HaMeR local-hand viewer."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from hamer_multiview_utils import DEFAULT_BASE_DIR, DEFAULT_CALIB, DEFAULT_FRAMES, DEFAULT_IMAGE_ROOT, parse_group_ids, range_suffix


DATASET_PRESETS = {
    "left_index": {
        "base_dir": Path("video/sam3_hamer_left_index"),
        "frames": Path("video/cameras_left_index/frames.jsonl"),
        "image_root": Path("video/cameras_left_index"),
        "calib": Path("video/cameras_left_index/cameras.yaml"),
    },
    "right_index": {
        "base_dir": Path("video/sam3_hamer_right_index"),
        "frames": Path("video/cameras_right_index/frames.jsonl"),
        "image_root": Path("video/cameras_right_index"),
        "calib": Path("video/cameras_right_index/cameras.yaml"),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="left_index", choices=sorted(DATASET_PRESETS), help="Named dataset preset.")
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument("--triangulated", type=Path, help="Explicit hamer_local_hands JSONL path.")
    parser.add_argument(
        "--glove",
        choices=["left_index", "right_index", "left_index_synced", "right_index_synced"],
        help="Quickly view a glove JSONL instead of HaMeR output.",
    )
    parser.add_argument("--detections", type=Path, help="MediaPipe landmarks JSONL. Defaults to base-dir/landmarks.jsonl.")
    parser.add_argument("--hamer-predictions", type=Path, help="HaMeR per-view predictions JSONL for 2D overlay.")
    parser.add_argument("--group-range", help='Inclusive group range, e.g. "1-100".')
    parser.add_argument("--range", dest="range_alias", help='Short alias for --group-range, e.g. "0-442".')
    parser.add_argument("--group-ids", help='Comma-separated ids, e.g. "1,7,42"; only used for file suffix selection.')
    parser.add_argument("--frames", type=Path, default=DEFAULT_FRAMES)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--calib", type=Path, default=DEFAULT_CALIB)
    parser.add_argument("--space", choices=["world", "root-relative", "palm-local"], default="palm-local")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--interval-ms", type=int, default=40)
    parser.add_argument("--fixed-range", type=float, default=0.35)
    parser.add_argument("--image-downscale", type=int, default=2)
    parser.add_argument("--elev", type=float, default=20.0)
    parser.add_argument("--azim", type=float, default=-65.0)
    parser.add_argument("--render-mode", choices=["skeleton", "mesh", "mesh+skeleton"], default="mesh+skeleton")
    parser.add_argument("--skeleton", action="store_true", help="Shortcut for --render-mode skeleton.")
    parser.add_argument("--mesh", action="store_true", help="Shortcut for --render-mode mesh.")
    parser.add_argument("--mesh-skeleton", action="store_true", help="Shortcut for --render-mode mesh+skeleton.")
    parser.add_argument("--mesh-stride", type=int, default=8)
    parser.add_argument("--clean", action="store_true", help="Hide RGB overlays and camera rig.")
    parser.add_argument("--no-mediapipe-overlay", action="store_true")
    parser.add_argument("--no-camera-rig", action="store_true")
    return parser.parse_args()


def default_triangulated_path(base_dir: Path, group_range: str | None, group_ids: str | None) -> Path:
    suffix = range_suffix(parse_group_ids(group_range, group_ids))
    selected = base_dir / "hamer_mano_multiview_selected" / f"mano_multiview_local_hands_{suffix}.jsonl"
    if selected.exists():
        return selected
    soft_refined = base_dir / "hamer_mano_multiview_soft_refined" / f"mano_multiview_local_hands_{suffix}.jsonl"
    if soft_refined.exists():
        return soft_refined
    image_refined = base_dir / "hamer_mano_multiview_refined" / f"mano_multiview_local_hands_{suffix}.jsonl"
    if image_refined.exists():
        return image_refined
    refined = base_dir / "hamer_mano_local_refined" / f"mano_local_hands_{suffix}.jsonl"
    if refined.exists():
        return refined
    ranged = base_dir / "hamer_primary_local" / f"hamer_local_hands_{suffix}.jsonl"
    if ranged.exists():
        return ranged
    return base_dir / "hamer_primary_local" / "hamer_local_hands.jsonl"


def glove_path(name: str) -> Path:
    paths = {
        "left_index": Path("gloves/glove_local/pn3_leftindex_glove_local.jsonl"),
        "right_index": Path("gloves/glove_local/pn3_rightindex_glove_local.jsonl"),
        "left_index_synced": Path("gloves/glove_local/pn3_leftindex_camera_sync_g414p000_c47p000_cut_000000_000442.jsonl"),
        "right_index_synced": Path("gloves/glove_local/pn3_rightindex_camera_sync_g356p000_c17p000_cut_000000_000477.jsonl"),
    }
    return paths[name]


def main() -> None:
    args = parse_args()
    preset = DATASET_PRESETS[args.dataset]
    if args.base_dir == DEFAULT_BASE_DIR:
        args.base_dir = preset["base_dir"]
    if args.frames == DEFAULT_FRAMES:
        args.frames = preset["frames"]
    if args.image_root == DEFAULT_IMAGE_ROOT:
        args.image_root = preset["image_root"]
    if args.calib == DEFAULT_CALIB:
        args.calib = preset["calib"]
    if args.range_alias and not args.group_range:
        args.group_range = args.range_alias
    if args.skeleton:
        args.render_mode = "skeleton"
    if args.mesh:
        args.render_mode = "mesh"
    if args.mesh_skeleton:
        args.render_mode = "mesh+skeleton"
    if args.clean:
        args.no_mediapipe_overlay = True
        args.no_camera_rig = True
    if args.glove:
        args.triangulated = glove_path(args.glove)
        args.render_mode = "skeleton"
        args.no_mediapipe_overlay = True
        args.no_camera_rig = True

    viewer = Path(__file__).resolve().parent / "view_triangulated_hands_rgb.py"
    triangulated = args.triangulated or default_triangulated_path(args.base_dir, args.group_range, args.group_ids)
    detections = args.detections or (args.base_dir / "landmarks.jsonl")
    hamer_predictions = args.hamer_predictions or (args.base_dir / "hamer_per_view" / f"hamer_predictions_{range_suffix(parse_group_ids(args.group_range, args.group_ids))}.jsonl")

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
    if hamer_predictions.exists():
        command.extend(["--hamer-predictions", str(hamer_predictions)])
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
