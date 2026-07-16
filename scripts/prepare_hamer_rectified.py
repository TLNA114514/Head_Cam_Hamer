#!/usr/bin/env python3
"""Create rectified image cache for SAM3 + HaMeR multi-view processing."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2

from hamer_multiview_utils import (
    DEFAULT_BASE_DIR,
    DEFAULT_CALIB,
    DEFAULT_CAMERAS,
    DEFAULT_FRAMES,
    DEFAULT_IMAGE_ROOT,
    DEFAULT_RECTIFY_FOCAL_SCALE,
    build_rectify_calibrations,
    filter_frame_records,
    parse_cameras,
    parse_group_ids,
    range_suffix,
    rectified_rel_path,
)
from progress_utils import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--frames", type=Path, default=DEFAULT_FRAMES)
    parser.add_argument("--calib", type=Path, default=DEFAULT_CALIB)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_BASE_DIR / "rectified_for_hamer")
    parser.add_argument("--cameras", default=",".join(DEFAULT_CAMERAS))
    parser.add_argument(
        "--rectify-focal-scale",
        type=float,
        default=DEFAULT_RECTIFY_FOCAL_SCALE,
        help=f"Rectification focal scale (default: {DEFAULT_RECTIFY_FOCAL_SCALE:g}).",
    )
    parser.add_argument("--group-range", help="Inclusive group range, e.g. 1-100. Commas allowed.")
    parser.add_argument("--group-ids", help="Comma-separated explicit group ids.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--progress-position", type=int, default=int(os.environ.get("TQDM_POSITION", "0")))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cameras = parse_cameras(args.cameras)
    group_ids = parse_group_ids(args.group_range, args.group_ids)
    records = filter_frame_records(args.frames, cameras, group_ids)
    suffix = range_suffix(group_ids)

    if args.dry_run:
        print(json.dumps({"records": len(records), "cameras": sorted(cameras), "suffix": suffix}, indent=2))
        return

    calibrations = build_rectify_calibrations(args.calib, cameras, args.rectify_focal_scale)
    written = 0
    skipped = 0
    missing = 0
    for record in tqdm(records, desc="rectify", unit="image", position=args.progress_position):
        camera_id = record["camera_id"]
        src = args.image_root / record["image_path"]
        dst = args.output_dir / rectified_rel_path(record)
        if dst.exists() and not args.overwrite:
            skipped += 1
            continue
        image = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if image is None:
            missing += 1
            print(f"missing image: {src}", flush=True)
            continue
        calib = calibrations[camera_id]
        rectified = cv2.remap(
            image,
            calib.map_x,
            calib.map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        dst.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dst), rectified)
        written += 1

    focal_scales = {
        camera_id: calib.rectify_focal_scale for camera_id, calib in calibrations.items()
    }
    unique_focal_scales = set(focal_scales.values())
    config = {
        "image_root": str(args.image_root),
        "frames": str(args.frames),
        "calib": str(args.calib),
        "output_dir": str(args.output_dir),
        "cameras": sorted(cameras),
        "rectify_focal_scale": next(iter(unique_focal_scales)) if len(unique_focal_scales) == 1 else None,
        "rectify_focal_scale_requested": args.rectify_focal_scale,
        "rectify_focal_scales": focal_scales,
        "camera_models": {
            camera_id: {
                "camera_model": calib.camera_model,
                "projection_model": calib.projection_model,
                "distortion_model": calib.distortion_model,
                "rectify_backend": calib.rectify_backend,
            }
            for camera_id, calib in calibrations.items()
        },
        "group_range": args.group_range,
        "group_ids": args.group_ids,
        "range_suffix": suffix,
        "records": len(records),
        "written": written,
        "skipped": skipped,
        "missing": missing,
        "new_intrinsics": {camera_id: calib.new_k.tolist() for camera_id, calib in calibrations.items()},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / f"rectified_config_{suffix}.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print("Summary")
    print(f"  records: {len(records)}")
    print(f"  written: {written}")
    print(f"  skipped: {skipped}")
    print(f"  missing: {missing}")
    for camera_id, calib in calibrations.items():
        print(
            f"  {camera_id}: backend={calib.rectify_backend} "
            f"focal_scale={calib.rectify_focal_scale:g}"
        )
    print(f"  output_dir: {args.output_dir}")


if __name__ == "__main__":
    main()
