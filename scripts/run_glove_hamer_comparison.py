#!/usr/bin/env python3
"""One-command glove-to-camera sync and HaMeR/MANO local error evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path

from dependency_paths import default_conda_executable

from hamer_multiview_utils import DEFAULT_BASE_DIR, DEFAULT_FRAMES, parse_group_ids, range_suffix
from sync_glove_to_camera import (
    default_output_path,
    load_camera_group_ids,
    parse_group_ids as parse_sync_group_ids,
    parse_group_range,
)

DATASET_PRESETS = {
    "left_index": {
        "frames": Path("video/cameras_left_index/frames.jsonl"),
        "base_dir": Path("video/sam3_hamer_left_index"),
        "output_dir": Path("gloves/left_index_hamer_eval"),
        "anchor_glove_frame": 414.0,
        "anchor_camera_frame": 47.0,
    },
    "right_index": {
        "frames": Path("video/cameras_right_index/frames.jsonl"),
        "base_dir": Path("video/sam3_hamer_right_index"),
        "output_dir": Path("gloves/right_index_hamer_eval"),
        "anchor_glove_frame": 356.0,
        "anchor_camera_frame": 17.0,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("glove_csv", nargs="+", type=Path, help="PN glove CSV file(s).")
    parser.add_argument("--dataset", choices=sorted(DATASET_PRESETS), help="Named dataset preset.")
    parser.add_argument("--frames", type=Path)
    parser.add_argument("--base-dir", type=Path)
    parser.add_argument("--hamer", type=Path, help="Existing HaMeR/MANO local JSONL. Defaults from base-dir and cut range.")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--sync-output-dir", type=Path, default=Path("gloves/glove_local"))
    parser.add_argument("--hands", default="Left,Right")
    parser.add_argument("--fingers", default="thumb,index,middle")
    parser.add_argument("--space", choices=["palm-local", "root-relative"], default="palm-local")
    parser.add_argument("--glove-fps", type=float, default=60.0)
    parser.add_argument("--camera-fps", type=float, default=25.0)
    parser.add_argument("--anchor-glove-frame", type=float)
    parser.add_argument("--anchor-glove-is-one-based", action="store_true")
    parser.add_argument("--anchor-camera-frame", type=float)
    parser.add_argument("--group-range", help="Optional extra camera group range clamp, e.g. 0-441.")
    parser.add_argument("--range", dest="range_alias", help="Short alias for --group-range.")
    parser.add_argument("--group-ids", help="Optional extra camera group id clamp.")
    parser.add_argument("--overlap-mode", choices=["intersection", "per-csv"], default="intersection")
    parser.add_argument("--interpolation", choices=["linear", "nearest"], default="linear")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--run-pipeline", action="store_true", help="Run SAM3+HaMeR/MANO pipeline before evaluation.")
    parser.add_argument("--conda-bin", type=Path, default=Path(default_conda_executable()))
    parser.add_argument("--pipeline-env", default="headcam")
    parser.add_argument("--max-parallel-workers", type=int, default=2)
    parser.add_argument("--hand-track-backend", choices=["image", "posthoc", "sam3-native"], default="posthoc")
    parser.add_argument("--pipeline-overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def infer_dataset(args: argparse.Namespace) -> str:
    if args.dataset:
        return args.dataset
    stems = {path.stem.lower() for path in args.glove_csv}
    if stems and all("rightindex" in stem or "right_index" in stem for stem in stems):
        return "right_index"
    return "left_index"


def apply_dataset_defaults(args: argparse.Namespace) -> None:
    dataset = infer_dataset(args)
    preset = DATASET_PRESETS[dataset]
    args.dataset = dataset
    if args.frames is None:
        args.frames = preset["frames"]
    if args.base_dir is None:
        args.base_dir = preset["base_dir"]
    if args.output_dir is None:
        args.output_dir = preset["output_dir"]
    if args.anchor_glove_frame is None:
        args.anchor_glove_frame = preset["anchor_glove_frame"]
    if args.anchor_camera_frame is None:
        args.anchor_camera_frame = preset["anchor_camera_frame"]


def csv_frame_bounds(path: Path) -> tuple[int, int]:
    with path.open("r", encoding="utf-8", newline="") as f:
        first = f.readline()
        header = next(csv.reader(f)) if not first.startswith("Frame-No") else next(csv.reader([first]))
        reader = csv.DictReader(f, fieldnames=header)
        frame_numbers = [int(float(row.get("Frame-No", index))) for index, row in enumerate(reader)]
    if not frame_numbers:
        raise ValueError(f"No glove frames found: {path}")
    return min(frame_numbers), max(frame_numbers)


def valid_camera_range_for_glove(
    camera_group_min: int,
    camera_group_max: int,
    glove_frame_min: int,
    glove_frame_max: int,
    anchor_glove_frame: float,
    anchor_camera_frame: float,
    glove_fps: float,
    camera_fps: float,
) -> tuple[int, int]:
    ratio = glove_fps / camera_fps
    start_float = anchor_camera_frame + (glove_frame_min - anchor_glove_frame) / ratio
    end_float = anchor_camera_frame + (glove_frame_max - anchor_glove_frame) / ratio
    start = max(camera_group_min, int(math.ceil(start_float - 1e-9)))
    end = min(camera_group_max, int(math.floor(end_float + 1e-9)))
    if end < start:
        raise ValueError(
            "No camera/glove overlap: "
            f"camera=[{camera_group_min},{camera_group_max}], glove=[{glove_frame_min},{glove_frame_max}], "
            f"anchor_glove={anchor_glove_frame}, anchor_camera={anchor_camera_frame}"
        )
    return start, end


def clamp_range(base_range: tuple[int, int], group_ids: list[int]) -> tuple[int, int]:
    selected = [group_id for group_id in group_ids if base_range[0] <= group_id <= base_range[1]]
    if not selected:
        raise ValueError(f"No selected group ids inside {base_range}")
    return min(selected), max(selected)


def cut_sync_path(sync_output_dir: Path, csv_path: Path, anchor_glove_frame: float, anchor_camera_frame: float, cut_range: tuple[int, int]) -> Path:
    base = default_output_path(sync_output_dir, csv_path, anchor_glove_frame, anchor_camera_frame)
    return base.with_name(f"{base.stem}_cut_{cut_range[0]:06d}_{cut_range[1]:06d}{base.suffix}")


def run_command(label: str, command: list[str], dry_run: bool) -> None:
    print(f"[{label}] {' '.join(command)}", flush=True)
    if dry_run:
        return
    subprocess.run(command, check=True)


def default_hamer_path(base_dir: Path, cut_range: tuple[int, int]) -> Path:
    group_ids = set(range(cut_range[0], cut_range[1] + 1))
    suffix = range_suffix(group_ids)
    refined = base_dir / "hamer_mano_local_refined" / f"mano_local_hands_{suffix}.jsonl"
    if refined.exists():
        return refined
    primary = base_dir / "hamer_primary_local" / f"hamer_local_hands_{suffix}.jsonl"
    if primary.exists():
        return primary
    return refined


def run_sync(csv_path: Path, output_path: Path, cut_range: tuple[int, int], args: argparse.Namespace) -> None:
    script = Path(__file__).resolve().parent / "sync_glove_to_camera.py"
    command = [
        sys.executable,
        str(script),
        str(csv_path),
        "--frames",
        str(args.frames),
        "--output",
        str(output_path),
        "--hands",
        args.hands,
        "--glove-fps",
        str(args.glove_fps),
        "--camera-fps",
        str(args.camera_fps),
        "--anchor-glove-frame",
        str(args.anchor_glove_frame),
        "--anchor-camera-frame",
        str(args.anchor_camera_frame),
        "--group-range",
        f"{cut_range[0]}-{cut_range[1]}",
        "--interpolation",
        args.interpolation,
    ]
    if args.anchor_glove_is_one_based:
        command.append("--anchor-glove-is-one-based")
    if args.overwrite:
        command.append("--overwrite")
    run_command("sync_glove", command, args.dry_run)


def run_pipeline(cut_range: tuple[int, int], args: argparse.Namespace) -> None:
    script = Path(__file__).resolve().parent / "run_hamer_multiview_pipeline.py"
    command = [
        str(args.conda_bin),
        "run",
        "-n",
        args.pipeline_env,
        "python",
        "-s",
        str(script),
        "--base-dir",
        str(args.base_dir),
        "--frames",
        str(args.frames),
        "--group-range",
        f"{cut_range[0]}-{cut_range[1]}",
        "--max-parallel-workers",
        str(args.max_parallel_workers),
        "--hand-track-backend",
        args.hand_track_backend,
        "--run-mano-local-refine",
    ]
    if args.pipeline_overwrite:
        command.append("--overwrite")
    run_command("pipeline", command, args.dry_run)


def run_eval(hamer_path: Path, glove_path: Path, cut_range: tuple[int, int], args: argparse.Namespace) -> None:
    script = Path(__file__).resolve().parent / "evaluate_hamer_vs_glove.py"
    if glove_path.name.startswith("pn3_leftindex"):
        stem = "left_index"
    elif glove_path.name.startswith("pn3_rightindex"):
        stem = "right_index"
    else:
        stem = glove_path.stem.split("_camera_sync", 1)[0]
    json_path = args.output_dir / f"{stem}_vs_hamer_{cut_range[0]:06d}_{cut_range[1]:06d}.json"
    csv_path = args.output_dir / f"{stem}_vs_hamer_{cut_range[0]:06d}_{cut_range[1]:06d}.csv"
    command = [
        sys.executable,
        str(script),
        "--hamer",
        str(hamer_path),
        "--glove",
        str(glove_path),
        "--space",
        args.space,
        "--hands",
        args.hands,
        "--fingers",
        args.fingers,
        "--group-range",
        f"{cut_range[0]}-{cut_range[1]}",
        "--output-json",
        str(json_path),
        "--output-csv",
        str(csv_path),
    ]
    run_command("evaluate", command, args.dry_run)


def main() -> None:
    args = parse_args()
    apply_dataset_defaults(args)
    if args.range_alias and not args.group_range:
        args.group_range = args.range_alias
    if args.glove_fps <= 0 or args.camera_fps <= 0:
        raise SystemExit("FPS values must be positive")

    selected_groups = load_camera_group_ids(args.frames, parse_group_range(args.group_range), parse_sync_group_ids(args.group_ids))
    if not selected_groups:
        raise SystemExit("No camera groups selected")
    camera_min = min(selected_groups)
    camera_max = max(selected_groups)
    anchor_glove_frame = args.anchor_glove_frame - 1.0 if args.anchor_glove_is_one_based else args.anchor_glove_frame

    per_csv_ranges: dict[Path, tuple[int, int]] = {}
    for csv_path in args.glove_csv:
        glove_min, glove_max = csv_frame_bounds(csv_path)
        overlap = valid_camera_range_for_glove(
            camera_min,
            camera_max,
            glove_min,
            glove_max,
            anchor_glove_frame,
            args.anchor_camera_frame,
            args.glove_fps,
            args.camera_fps,
        )
        per_csv_ranges[csv_path] = clamp_range(overlap, selected_groups)

    if args.overlap_mode == "intersection":
        cut_start = max(item[0] for item in per_csv_ranges.values())
        cut_end = min(item[1] for item in per_csv_ranges.values())
        if cut_end < cut_start:
            raise SystemExit(f"No common overlap across glove CSVs: {per_csv_ranges}")
        per_csv_ranges = {path: (cut_start, cut_end) for path in per_csv_ranges}

    print("[cut]", flush=True)
    for csv_path, cut_range in per_csv_ranges.items():
        print(f"  {csv_path}: camera group {cut_range[0]}-{cut_range[1]}", flush=True)
    print(
        "[sync] "
        f"glove_frame = {anchor_glove_frame:.6f} + "
        f"(camera_group - {args.anchor_camera_frame:.6f}) * {args.glove_fps:.6f}/{args.camera_fps:.6f}",
        flush=True,
    )

    # Pipeline can only be run once when all CSVs share the same cut range. For per-csv mode,
    # run it on the union so each later evaluation has the needed camera groups.
    pipeline_range = (
        min(item[0] for item in per_csv_ranges.values()),
        max(item[1] for item in per_csv_ranges.values()),
    )
    if args.run_pipeline:
        run_pipeline(pipeline_range, args)

    for csv_path, cut_range in per_csv_ranges.items():
        sync_path = cut_sync_path(args.sync_output_dir, csv_path, anchor_glove_frame, args.anchor_camera_frame, cut_range)
        run_sync(csv_path, sync_path, cut_range, args)
        hamer_path = args.hamer or default_hamer_path(args.base_dir, pipeline_range)
        if not args.dry_run and not hamer_path.exists():
            raise SystemExit(
                f"HaMeR/MANO local output not found: {hamer_path}\n"
                f"Run with --run-pipeline, or pass --hamer explicitly."
            )
        run_eval(hamer_path, sync_path, cut_range, args)


if __name__ == "__main__":
    main()
