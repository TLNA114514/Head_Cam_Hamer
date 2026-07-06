#!/usr/bin/env python3
"""Fuse MediaPipe and SAM3 detections into per-view HaMeR jobs."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2

from hamer_multiview_utils import (
    DEFAULT_BASE_DIR,
    DEFAULT_CAMERAS,
    DEFAULT_FRAMES,
    bbox_area,
    bbox_iou,
    draw_bbox,
    expand_bbox,
    filter_frame_records,
    handedness_to_is_right,
    iter_jsonl,
    load_mask,
    masked_blur_frame,
    mediapipe_bbox,
    opposite_handedness,
    parse_cameras,
    parse_group_ids,
    range_suffix,
    rectified_rel_path,
    union_bbox,
)
from progress_utils import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=Path, default=DEFAULT_FRAMES)
    parser.add_argument("--rectified-dir", type=Path, default=DEFAULT_BASE_DIR / "rectified_for_hamer")
    parser.add_argument("--mediapipe", type=Path, default=DEFAULT_BASE_DIR / "landmarks.jsonl")
    parser.add_argument("--sam3", type=Path, help="SAM3 JSONL. Defaults to output-dir sibling sam3_bboxes by range.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_BASE_DIR / "hamer_jobs")
    parser.add_argument("--cameras", default=",".join(DEFAULT_CAMERAS))
    parser.add_argument("--group-range")
    parser.add_argument("--group-ids")
    parser.add_argument("--min-mediapipe-score", type=float, default=0.7)
    parser.add_argument("--match-iou", type=float, default=0.25)
    parser.add_argument("--bbox-pad", type=float, default=1.15)
    parser.add_argument("--max-hands", type=int, default=2)
    parser.add_argument("--mask-frame-mode", choices=["none", "gray", "blur"], default="blur")
    parser.add_argument("--mask-frame-dilate", type=int, default=9)
    parser.add_argument("--mask-feather", type=int, default=11)
    parser.add_argument(
        "--camera-handedness-override",
        default="C0:Left,C3:Right",
        help='Comma-separated camera handedness overrides, e.g. "C0:Left,C3:Right"; use "none" to disable.',
    )
    parser.add_argument("--progress-position", type=int, default=int(os.environ.get("TQDM_POSITION", "0")))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def parse_camera_handedness_override(value: str | None) -> dict[str, str]:
    if not value or value.strip().lower() in {"none", "off", "false", "0"}:
        return {}
    overrides = {}
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if ":" in part:
            camera_id, handedness = part.split(":", 1)
        elif "=" in part:
            camera_id, handedness = part.split("=", 1)
        else:
            raise ValueError(f"Invalid camera handedness override: {part}")
        camera_id = camera_id.strip()
        handedness = handedness.strip()
        if handedness not in {"Left", "Right"}:
            raise ValueError(f"Invalid handedness override for {camera_id}: {handedness}")
        overrides[camera_id] = handedness
    return overrides


def load_mediapipe(path: Path, cameras: set[str], group_ids: set[int] | None, min_score: float) -> dict[tuple[int, str], list[dict[str, Any]]]:
    data: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for record in iter_jsonl(path):
        group_id = int(record["group_id"])
        camera_id = record["camera_id"]
        if camera_id not in cameras:
            continue
        if group_ids is not None and group_id not in group_ids:
            continue
        for hand in record.get("hands") or []:
            score = hand.get("handedness_score")
            score = float(score) if isinstance(score, (int, float)) else -1.0
            handedness = hand.get("handedness")
            bbox = mediapipe_bbox(hand, pad=1.25)
            if score < min_score or handedness not in {"Left", "Right"} or bbox is None:
                continue
            data[(group_id, camera_id)].append(
                {
                    "bbox": bbox,
                    "handedness": handedness,
                    "handedness_score": score,
                    "source": "mediapipe",
                }
            )
    for hands in data.values():
        repair_duplicate_mediapipe_handedness(hands)
    return data


def repair_duplicate_mediapipe_handedness(hands: list[dict[str, Any]]) -> int:
    if len(hands) != 2:
        return 0
    labels = [hand.get("handedness") for hand in hands]
    if labels[0] not in {"Left", "Right"} or labels[0] != labels[1]:
        return 0
    scores = [
        float(hand.get("handedness_score")) if isinstance(hand.get("handedness_score"), (int, float)) else -1.0
        for hand in hands
    ]
    flip_index = 0 if scores[0] < scores[1] else 1
    original = hands[flip_index]["handedness"]
    hands[flip_index]["original_handedness"] = original
    hands[flip_index]["handedness"] = opposite_handedness(original)
    hands[flip_index]["source"] = "mediapipe_duplicate_repair"
    return 1


def load_sam3(path: Path, cameras: set[str], group_ids: set[int] | None) -> dict[tuple[int, str], list[dict[str, Any]]]:
    data: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for record in iter_jsonl(path):
        group_id = int(record["group_id"])
        camera_id = record["camera_id"]
        if camera_id not in cameras:
            continue
        if group_ids is not None and group_id not in group_ids:
            continue
        for hand in record.get("hands") or []:
            bbox = hand.get("bbox")
            if not bbox:
                continue
            item = dict(hand)
            item["bbox"] = [float(v) for v in bbox]
            data[(group_id, camera_id)].append(item)
    return data


def create_blur_frame(
    image_path: Path,
    mask_path: str | None,
    output_path: Path,
    args: argparse.Namespace,
) -> tuple[str, bool]:
    if args.mask_frame_mode == "none" or not mask_path:
        return str(image_path), False
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    mask = load_mask(mask_path)
    if image is None or mask is None:
        return str(image_path), False
    if args.mask_frame_mode == "gray":
        out = image.copy()
        mask_d = mask
        out[~mask_d] = 127
    else:
        out = masked_blur_frame(image, mask, args.mask_frame_dilate, args.mask_feather)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), out)
    return str(output_path), True


def draw_debug(
    image_path: Path,
    mp_hands: list[dict[str, Any]],
    sam_hands: list[dict[str, Any]],
    jobs: list[dict[str, Any]],
    output_path: Path,
) -> None:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return
    for hand in mp_hands:
        draw_bbox(image, hand["bbox"], (0, 220, 255), f"MP {hand['handedness']}")
    for hand in sam_hands:
        draw_bbox(image, hand["bbox"], (40, 220, 40), f"SAM3 {hand.get('score', 0):.2f}")
    for job in jobs:
        color = (220, 80, 220) if not job.get("debug_only") else (120, 120, 120)
        draw_bbox(image, job["bbox_rectified_px"], color, f"JOB {job['handedness']} {job['source_detector']}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)


def fuse_one(
    group_id: int,
    camera_id: str,
    image_path: Path,
    mp_hands: list[dict[str, Any]],
    sam_hands: list[dict[str, Any]],
    args: argparse.Namespace,
    handedness_overrides: dict[str, str],
) -> list[dict[str, Any]]:
    jobs = []
    used_sam = set()
    for mp_index, mp in enumerate(mp_hands):
        best_i = None
        best_iou = 0.0
        for sam_index, sam in enumerate(sam_hands):
            if sam_index in used_sam:
                continue
            iou = bbox_iou(mp["bbox"], sam["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_i = sam_index
        sam = sam_hands[best_i] if best_i is not None and best_iou >= args.match_iou else None
        if best_i is not None and sam is not None:
            used_sam.add(best_i)
            bbox = expand_bbox(union_bbox(mp["bbox"], sam["bbox"]), args.bbox_pad)
            source = "mediapipe+sam3"
            mask_path = sam.get("mask_path")
            sam_score = sam.get("score")
        else:
            bbox = expand_bbox(mp["bbox"], args.bbox_pad)
            source = "mediapipe"
            mask_path = None
            sam_score = None
        hamer_frame, used_blur = create_blur_frame(
            image_path,
            mask_path,
            args.output_dir / "blurred_frames" / camera_id / f"{group_id:08d}_{len(jobs):02d}.jpg",
            args,
        )
        jobs.append(
            {
                "job_index": len(jobs),
                "group_id": group_id,
                "camera_id": camera_id,
                "rectified_image_path": str(image_path),
                "hamer_frame_path": hamer_frame,
                "used_mask_blur": used_blur,
                "bbox_rectified_px": bbox,
                "handedness": mp["handedness"],
                "handedness_source": mp.get("source", "mediapipe"),
                "handedness_score": mp["handedness_score"],
                "is_right": handedness_to_is_right(mp["handedness"]),
                "source_detector": source,
                "sam3_mask_path": mask_path,
                "sam3_score": sam_score,
                "debug_only": False,
                "original_handedness": mp.get("original_handedness"),
            }
        )

    mp_sides = {hand["handedness"] for hand in mp_hands}
    leftover = [
        (index, hand)
        for index, hand in enumerate(sam_hands)
        if index not in used_sam
    ]
    leftover.sort(key=lambda item: (float(item[1].get("score", 0.0)), bbox_area(item[1]["bbox"])), reverse=True)

    for _sam_index, sam in leftover:
        if len([job for job in jobs if not job.get("debug_only")]) >= args.max_hands:
            break
        if len(mp_sides) == 1:
            handedness = opposite_handedness(next(iter(mp_sides)))
            handedness_source = "sam3_opposite_of_single_mediapipe"
            debug_only = False
        elif len(mp_sides) >= 2:
            handedness = "unknown"
            handedness_source = "sam3_extra_when_mediapipe_has_both"
            debug_only = True
        else:
            handedness = "unknown"
            handedness_source = "sam3_only_unknown"
            debug_only = False

        bbox = expand_bbox(sam["bbox"], args.bbox_pad)
        hamer_frame, used_blur = create_blur_frame(
            image_path,
            sam.get("mask_path"),
            args.output_dir / "blurred_frames" / camera_id / f"{group_id:08d}_{len(jobs):02d}.jpg",
            args,
        )
        jobs.append(
            {
                "job_index": len(jobs),
                "group_id": group_id,
                "camera_id": camera_id,
                "rectified_image_path": str(image_path),
                "hamer_frame_path": hamer_frame,
                "used_mask_blur": used_blur,
                "bbox_rectified_px": bbox,
                "handedness": handedness,
                "handedness_source": handedness_source,
                "handedness_score": None,
                "is_right": handedness_to_is_right(handedness),
                "source_detector": "sam3",
                "sam3_mask_path": sam.get("mask_path"),
                "sam3_score": sam.get("score"),
                "debug_only": debug_only,
            }
        )
    jobs = jobs[: max(args.max_hands, len(mp_hands))]
    override_handedness = handedness_overrides.get(camera_id)
    if override_handedness:
        for job in jobs:
            job["original_handedness"] = job.get("handedness")
            job["original_handedness_source"] = job.get("handedness_source")
            job["handedness"] = override_handedness
            job["handedness_source"] = f"camera_override:{camera_id}"
            job["is_right"] = handedness_to_is_right(override_handedness)
            job["camera_handedness_override"] = True
    return jobs


def main() -> None:
    args = parse_args()
    cameras = parse_cameras(args.cameras)
    group_ids = parse_group_ids(args.group_range, args.group_ids)
    handedness_overrides = parse_camera_handedness_override(args.camera_handedness_override)
    suffix = range_suffix(group_ids)
    sam3_path = args.sam3 or (DEFAULT_BASE_DIR / "sam3_bboxes" / f"sam3_bboxes_{suffix}.jsonl")
    output_path = args.output_dir / f"hamer_jobs_{suffix}.jsonl"

    records = filter_frame_records(args.frames, cameras, group_ids)
    if args.dry_run:
        print(json.dumps({"records": len(records), "sam3": str(sam3_path), "suffix": suffix}, indent=2))
        return
    if output_path.exists() and not args.overwrite:
        raise SystemExit(f"{output_path} exists; pass --overwrite to replace it")
    if not args.mediapipe.exists():
        raise SystemExit(f"MediaPipe detections not found: {args.mediapipe}")
    if not sam3_path.exists():
        raise SystemExit(f"SAM3 detections not found: {sam3_path}")

    mp = load_mediapipe(args.mediapipe, cameras, group_ids, args.min_mediapipe_score)
    sam3 = load_sam3(sam3_path, cameras, group_ids)
    stats = defaultdict(int)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out:
        for record in tqdm(records, desc=f"fuse jobs {','.join(sorted(cameras))}", unit="frame", position=args.progress_position):
            group_id = int(record["group_id"])
            camera_id = record["camera_id"]
            key = (group_id, camera_id)
            image_path = args.rectified_dir / rectified_rel_path(record)
            jobs = fuse_one(group_id, camera_id, image_path, mp.get(key, []), sam3.get(key, []), args, handedness_overrides)
            draw_debug(
                image_path,
                mp.get(key, []),
                sam3.get(key, []),
                jobs,
                args.output_dir / "debug" / camera_id / f"{group_id:08d}.jpg",
            )
            stats["frames"] += 1
            stats["jobs"] += len(jobs)
            for job in jobs:
                stats[f"source:{job['source_detector']}"] += 1
                if job.get("used_mask_blur"):
                    stats["used_mask_blur"] += 1
                if job.get("debug_only"):
                    stats["debug_only"] += 1
                if job.get("camera_handedness_override"):
                    stats[f"camera_handedness_override:{camera_id}->{job['handedness']}"] += 1
            out.write(
                json.dumps(
                    {
                        "type": "hamer_multiview_jobs",
                        "group_id": group_id,
                        "camera_id": camera_id,
                        "jobs": jobs,
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )

    with (args.output_dir / f"hamer_jobs_config_{suffix}.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "mediapipe": str(args.mediapipe),
                "sam3": str(sam3_path),
                "output_path": str(output_path),
                "group_range": args.group_range,
                "group_ids": args.group_ids,
                "camera_handedness_override": handedness_overrides,
                "stats": dict(stats),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
        f.write("\n")

    print("Summary")
    for key in sorted(stats):
        print(f"  {key}: {stats[key]}")
    print(f"wrote: {output_path}")


if __name__ == "__main__":
    main()
