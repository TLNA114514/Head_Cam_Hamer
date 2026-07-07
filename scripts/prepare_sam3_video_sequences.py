#!/usr/bin/env python3
"""Build per-camera JPEG image sequences for SAM3 video tracking."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from hamer_multiview_utils import (
    DEFAULT_BASE_DIR,
    DEFAULT_CAMERAS,
    DEFAULT_FRAMES,
    filter_frame_records,
    parse_cameras,
    parse_group_ids,
    range_suffix,
    rectified_rel_path,
)
from progress_utils import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=Path, default=DEFAULT_FRAMES)
    parser.add_argument("--rectified-dir", type=Path, default=DEFAULT_BASE_DIR / "rectified_for_hamer")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_BASE_DIR / "sam3_video_sequences")
    parser.add_argument("--cameras", default=",".join(DEFAULT_CAMERAS))
    parser.add_argument("--group-range")
    parser.add_argument("--group-ids")
    parser.add_argument("--copy", action="store_true", help="Copy images instead of creating symlinks.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--progress-position", type=int, default=int(os.environ.get("TQDM_POSITION", "0")))
    return parser.parse_args()


def link_or_copy(src: Path, dst: Path, copy: bool) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy:
        shutil.copy2(src, dst)
        return "copy"
    try:
        rel_src = os.path.relpath(src.resolve(), dst.parent.resolve())
        dst.symlink_to(rel_src)
        return "symlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def main() -> None:
    args = parse_args()
    cameras = parse_cameras(args.cameras)
    group_ids = parse_group_ids(args.group_range, args.group_ids)
    suffix = range_suffix(group_ids)
    records = filter_frame_records(args.frames, cameras, group_ids)
    by_camera: dict[str, list[dict]] = {camera_id: [] for camera_id in sorted(cameras)}
    for record in records:
        by_camera[record["camera_id"]].append(record)
    for camera_records in by_camera.values():
        camera_records.sort(key=lambda item: int(item["group_id"]))

    if args.dry_run:
        print(json.dumps({camera_id: len(items) for camera_id, items in by_camera.items()}, indent=2))
        return

    stats = {"cameras": len(by_camera), "frames": 0, "missing_images": 0, "symlink": 0, "copy": 0}
    for camera_id, camera_records in by_camera.items():
        camera_root = args.output_dir / suffix / camera_id
        images_dir = camera_root / "images"
        frame_map_path = camera_root / "frame_map.json"
        if images_dir.exists() and any(images_dir.iterdir()) and not args.overwrite:
            raise SystemExit(f"{images_dir} exists; pass --overwrite to replace it")
        if args.overwrite and camera_root.exists():
            shutil.rmtree(camera_root)
        images_dir.mkdir(parents=True, exist_ok=True)
        frame_map = []
        progress = tqdm(camera_records, desc=f"sequence {camera_id}", unit="image", position=args.progress_position)
        for video_frame_index, record in enumerate(progress):
            group_id = int(record["group_id"])
            src = args.rectified_dir / rectified_rel_path(record)
            if not src.exists():
                stats["missing_images"] += 1
                continue
            dst = images_dir / f"{video_frame_index:06d}.jpg"
            mode = link_or_copy(src, dst, args.copy)
            stats[mode] += 1
            stats["frames"] += 1
            frame_map.append(
                {
                    "video_frame_index": video_frame_index,
                    "group_id": group_id,
                    "camera_id": camera_id,
                    "rectified_image_path": str(src),
                    "sequence_image_path": str(dst),
                }
            )
        with frame_map_path.open("w", encoding="utf-8") as f:
            json.dump(frame_map, f, ensure_ascii=False, indent=2)
            f.write("\n")
        with (camera_root / "sequence_config.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "frames": str(args.frames),
                    "rectified_dir": str(args.rectified_dir),
                    "group_range": args.group_range,
                    "group_ids": args.group_ids,
                    "camera_id": camera_id,
                    "suffix": suffix,
                    "frame_count": len(frame_map),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
            f.write("\n")
    print("Summary")
    for key in sorted(stats):
        print(f"  {key}: {stats[key]}")
    print(f"wrote: {args.output_dir / suffix}")


if __name__ == "__main__":
    main()
