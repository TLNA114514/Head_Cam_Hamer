#!/usr/bin/env python3
"""Run SAM3 hand bbox/mask detection on rectified multi-camera image sequences."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

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


from dependency_paths import DEFAULT_SAM3_ROOT, WRIST_SCRIPTS
if str(WRIST_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(WRIST_SCRIPTS))

from run_sam3_bbox_on_video import (  # type: ignore  # noqa: E402
    assign_track_ids,
    collect_detections,
    draw_debug,
    draw_mask_debug,
    ensure_sam3_importable,
    inference_context,
    json_safe_hands,
    merge_detections,
    resolve_is_right,
    torch_dtype_from_name,
)


PROMPT_PRESETS = {
    "bare": ["bare hand", "human hand", "hand", "left hand", "right hand"],
    "gloved": [
        "black fingerless glove",
        "fingerless glove",
        "gloved hand",
        "hand wearing black glove",
        "left gloved hand",
        "right gloved hand",
    ],
    "realtime": ["hand"],
    "custom": [],
}

TRUSTED_HANDEDNESS_SOURCES = {"manual", "prompt", "prompt_cluster"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=Path, default=DEFAULT_FRAMES)
    parser.add_argument("--rectified-dir", type=Path, default=DEFAULT_BASE_DIR / "rectified_for_hamer")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_BASE_DIR / "sam3_bboxes")
    parser.add_argument("--sam3-root", type=Path, default=DEFAULT_SAM3_ROOT)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--hf-endpoint", default="https://hf-mirror.com")
    parser.add_argument(
        "--no-hf",
        action="store_true",
        help="Disable Hugging Face loading; requires --checkpoint to avoid an uninitialized SAM3 model.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp-dtype", choices=["float32", "bfloat16", "float16"], default="float32")
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--cameras", default=",".join(DEFAULT_CAMERAS))
    parser.add_argument("--group-range")
    parser.add_argument("--group-ids")
    parser.add_argument("--frame-stride", type=int, default=1, help="Run SAM3 on every Nth frame per camera.")
    parser.add_argument("--frame-offset", type=int, default=0, help="Per-camera sequence offset used with --frame-stride.")
    parser.add_argument("--prompt-preset", choices=sorted(PROMPT_PRESETS), default="bare")
    parser.add_argument("--prompt", action="append", dest="prompts")
    parser.add_argument("--confidence-threshold", type=float, default=0.35)
    parser.add_argument("--max-hands", type=int, default=2)
    parser.add_argument("--hand-selection", choices=["area", "score", "score-area"], default="score-area")
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument(
        "--duplicate-mask-containment",
        type=float,
        default=0.9,
        help="Suppress nested SAM3 masks when intersection/min(area) reaches this value; 0 disables it.",
    )
    parser.add_argument("--bbox-pad", type=float, default=1.15)
    parser.add_argument("--merge-mode", choices=["largest", "all", "none"], default="none")
    parser.add_argument("--mask-component", choices=["largest", "all"], default="largest")
    parser.add_argument("--min-area-frac", type=float, default=0.002)
    parser.add_argument("--max-area-frac", type=float, default=0.85)
    parser.add_argument("--handedness", choices=["auto", "both", "left", "right"], default="auto")
    parser.add_argument("--is-right", type=int, choices=[0, 1])
    parser.add_argument("--save-mask-debug", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-bbox-debug", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--progress-every", type=int, default=10, help="Print line-based progress every N images; 0 disables.")
    parser.add_argument("--progress-position", type=int, default=int(os.environ.get("TQDM_POSITION", "0")), help="tqdm terminal row position.")
    parser.add_argument("--stream-output", action="store_true", help="Flush every keyframe record for downstream consumers.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_prompts(args: argparse.Namespace) -> list[str]:
    prompts = list(args.prompts or PROMPT_PRESETS[args.prompt_preset])
    if args.prompt_preset == "custom" and not prompts:
        raise SystemExit("--prompt-preset custom requires at least one --prompt")
    seen = set()
    unique = []
    for prompt in prompts:
        key = prompt.casefold()
        if key in seen:
            continue
        seen.add(key)
        unique.append(prompt)
    return unique


def select_stride_records(records: list[dict], stride: int, offset: int) -> list[dict]:
    by_camera: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_camera[str(record["camera_id"])].append(record)
    selected = []
    for camera_records in by_camera.values():
        camera_records.sort(key=lambda record: int(record["group_id"]))
        selected.extend(
            record
            for sequence_index, record in enumerate(camera_records)
            if sequence_index % stride == offset
        )
    return sorted(selected, key=lambda record: (int(record["group_id"]), str(record["camera_id"])))


def save_masks(hands: list[dict], mask_dir: Path, group_id: int, camera_id: str) -> list[dict]:
    out = []
    mask_dir.mkdir(parents=True, exist_ok=True)
    for hand_index, hand in enumerate(hands):
        item = {key: value for key, value in hand.items() if not key.startswith("_")}
        mask = hand.get("_mask")
        if mask is not None:
            mask_path = mask_dir / f"{camera_id}_{group_id:08d}_{hand_index:02d}.png"
            cv2.imwrite(str(mask_path), np.asarray(mask, dtype=np.uint8) * 255)
            item["mask_path"] = str(mask_path)
        out.append(item)
    return out


def mask_containment(first: dict, second: dict) -> float:
    first_mask = first.get("_mask")
    second_mask = second.get("_mask")
    if first_mask is None or second_mask is None:
        return 0.0
    first_array = np.asarray(first_mask, dtype=bool)
    second_array = np.asarray(second_mask, dtype=bool)
    if first_array.shape != second_array.shape:
        return 0.0
    first_area = int(first_array.sum())
    second_area = int(second_array.sum())
    minimum_area = min(first_area, second_area)
    if minimum_area <= 0:
        return 0.0
    intersection = int(np.logical_and(first_array, second_array).sum())
    return float(intersection / minimum_area)


def inherit_trusted_handedness(kept: dict, duplicate: dict) -> None:
    kept_source = str(kept.get("handedness_source") or "")
    duplicate_source = str(duplicate.get("handedness_source") or "")
    if kept_source in TRUSTED_HANDEDNESS_SOURCES or duplicate_source not in TRUSTED_HANDEDNESS_SOURCES:
        return
    kept["is_right"] = duplicate.get("is_right")
    kept["handedness_source"] = duplicate_source
    kept["handedness_prompt"] = duplicate.get("handedness_prompt")


def suppress_duplicate_hands(hands: list[dict], threshold: float) -> tuple[list[dict], int]:
    if threshold <= 0.0:
        return hands, 0
    selected: list[dict] = []
    suppressed = 0
    for hand in hands:
        duplicate = next(
            (
                kept
                for kept in selected
                if hand.get("handedness_source") != "hypothesis"
                and kept.get("handedness_source") != "hypothesis"
                and mask_containment(hand, kept) >= threshold
            ),
            None,
        )
        if duplicate is None:
            selected.append(hand)
            continue
        inherit_trusted_handedness(duplicate, hand)
        suppressed += 1
    return selected, suppressed


def main() -> None:
    args = parse_args()
    if args.torch_threads < 1:
        raise SystemExit("--torch-threads must be positive")
    torch.set_num_threads(args.torch_threads)
    cv2.setNumThreads(1)
    args.sam3_root = args.sam3_root.expanduser().resolve()
    if args.checkpoint:
        args.checkpoint = args.checkpoint.expanduser().resolve()
    if args.no_hf and args.checkpoint is None:
        raise SystemExit("--no-hf requires --checkpoint; otherwise SAM3 has no pretrained weights")
    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint
    if args.max_hands < 1:
        raise SystemExit("--max-hands must be >= 1")
    if not 0.0 <= args.duplicate_mask_containment <= 1.0:
        raise SystemExit("--duplicate-mask-containment must be in [0, 1]")
    if args.frame_stride < 1 or not 0 <= args.frame_offset < args.frame_stride:
        raise SystemExit("--frame-stride must be positive and --frame-offset must be in [0, stride)")
    if args.max_hands > 1 and args.merge_mode == "largest":
        args.merge_mode = "none"
    args.resolved_is_right = resolve_is_right(args.handedness, args.is_right)

    cameras = parse_cameras(args.cameras)
    group_ids = parse_group_ids(args.group_range, args.group_ids)
    suffix = range_suffix(group_ids)
    source_records = filter_frame_records(args.frames, cameras, group_ids)
    records = select_stride_records(source_records, args.frame_stride, args.frame_offset)
    output_path = args.output_dir / f"sam3_bboxes_{suffix}.jsonl"

    if args.dry_run:
        print(
            json.dumps(
                {
                    "source_records": len(source_records),
                    "records": len(records),
                    "cameras": sorted(cameras),
                    "frame_stride": args.frame_stride,
                    "frame_offset": args.frame_offset,
                    "suffix": suffix,
                },
                indent=2,
            )
        )
        return
    if output_path.exists() and not args.overwrite:
        raise SystemExit(f"{output_path} exists; pass --overwrite to replace it")

    ensure_sam3_importable(args.sam3_root)
    from sam3.model.sam3_image_processor import Sam3Processor  # type: ignore
    from sam3.model_builder import build_sam3_image_model  # type: ignore

    prompts = build_prompts(args)
    model_load_started = time.perf_counter()
    model = build_sam3_image_model(
        device=args.device,
        checkpoint_path=str(args.checkpoint) if args.checkpoint else None,
        load_from_HF=not args.no_hf,
    )
    model = model.float()
    processor = Sam3Processor(model, device=args.device, confidence_threshold=args.confidence_threshold)
    model_load_seconds = time.perf_counter() - model_load_started
    cuda_enabled = args.device.startswith("cuda") and torch.cuda.is_available()
    if cuda_enabled:
        torch.cuda.reset_peak_memory_stats()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = args.output_dir / "masks" / suffix
    bbox_debug_dir = args.output_dir / "debug" / suffix
    mask_debug_dir = args.output_dir / "mask_debug" / suffix

    stats = {
        "source_records": len(source_records),
        "records": len(records),
        "frames_with_hands": 0,
        "hands": 0,
        "suppressed_duplicate_hands": 0,
        "missing_images": 0,
    }
    inference_started = time.perf_counter()
    print(f"[SAM3] start cameras={','.join(sorted(cameras))} records={len(records)} output={output_path}", flush=True)
    with output_path.open("w", encoding="utf-8", buffering=1 if args.stream_output else -1) as out:
        progress = tqdm(records, desc=f"SAM3 {','.join(sorted(cameras))}", unit="image", position=args.progress_position)
        for index, record in enumerate(progress, start=1):
            group_id = int(record["group_id"])
            camera_id = record["camera_id"]
            image_path = args.rectified_dir / rectified_rel_path(record)
            image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image_bgr is None:
                stats["missing_images"] += 1
                out.write(
                    json.dumps(
                        {
                            "type": "sam3_multiview_bboxes",
                            "group_id": group_id,
                            "camera_id": camera_id,
                            "rectified_image_path": str(image_path),
                            "hands": [],
                            "missing_image": True,
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                if args.stream_output:
                    out.flush()
                continue
            image_rgb = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
            with inference_context(args):
                state = processor.set_image(image_rgb)
                detections = []
                for prompt in prompts:
                    output = processor.set_text_prompt(prompt=prompt, state=state)
                    detections.extend(collect_detections(output, prompt, image_bgr.shape, args))
            merge_args = copy.copy(args)
            if args.merge_mode == "none" and args.duplicate_mask_containment > 0.0:
                merge_args.max_hands = max(args.max_hands, min(8, args.max_hands * 3))
            hands = merge_detections(detections, image_bgr.shape, merge_args)
            hands, suppressed = suppress_duplicate_hands(hands, args.duplicate_mask_containment)
            hands = hands[: args.max_hands]
            stats["suppressed_duplicate_hands"] += suppressed
            assign_track_ids(hands)
            json_hands = save_masks(hands, mask_dir, group_id, camera_id)

            if args.save_bbox_debug:
                draw_debug(image_bgr, hands, bbox_debug_dir / camera_id / f"{group_id:08d}.jpg")
            if args.save_mask_debug:
                draw_mask_debug(image_bgr, detections, mask_debug_dir / camera_id / f"{group_id:08d}.jpg")

            stats["hands"] += len(json_hands)
            if json_hands:
                stats["frames_with_hands"] += 1
            out.write(
                json.dumps(
                    {
                        "type": "sam3_multiview_bboxes",
                        "group_id": group_id,
                        "camera_id": camera_id,
                        "rectified_image_path": str(image_path),
                        "hands": json_safe_hands(json_hands),
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )
            if args.stream_output:
                out.flush()
            if args.progress_every > 0 and (index % args.progress_every == 0 or index == len(records)):
                progress.set_postfix_str(f"group={group_id} hands={len(json_hands)} total={stats['hands']}")

    if cuda_enabled:
        torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - inference_started
    timing = {
        "model_load_seconds": model_load_seconds,
        "inference_total_seconds": inference_seconds,
        "keyframe_images_per_second": len(records) / inference_seconds if inference_seconds else None,
        "source_multiview_frames_per_second": (
            len(source_records) / max(1, len(cameras)) / inference_seconds if inference_seconds else None
        ),
    }
    peak_cuda_allocated_mib = torch.cuda.max_memory_allocated() / 2**20 if cuda_enabled else None
    peak_cuda_reserved_mib = torch.cuda.max_memory_reserved() / 2**20 if cuda_enabled else None

    config = {
        "sam3_root": str(args.sam3_root),
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "load_from_hf": not args.no_hf,
        "frames": str(args.frames),
        "rectified_dir": str(args.rectified_dir),
        "output_path": str(output_path),
        "cameras": sorted(cameras),
        "group_range": args.group_range,
        "group_ids": args.group_ids,
        "frame_stride": args.frame_stride,
        "frame_offset": args.frame_offset,
        "prompt_preset": args.prompt_preset,
        "prompts": prompts,
        "max_hands": args.max_hands,
        "duplicate_mask_containment": args.duplicate_mask_containment,
        "amp_dtype": args.amp_dtype,
        "torch_threads": args.torch_threads,
        "timing": timing,
        "peak_cuda_allocated_mib": peak_cuda_allocated_mib,
        "peak_cuda_reserved_mib": peak_cuda_reserved_mib,
        "stats": stats,
    }
    with (args.output_dir / f"sam3_config_{suffix}.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print("Summary")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print(f"wrote: {output_path}")


if __name__ == "__main__":
    main()
