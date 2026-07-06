#!/usr/bin/env python3
"""Run SAM3 hand bbox/mask detection on rectified multi-camera image sequences."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
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


WRIST_CAM_ROOT = Path("/home/luojiangrui/ljr/wrist_cam")
WRIST_SCRIPTS = WRIST_CAM_ROOT / "scripts"
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
    "gloved": ["black fingerless glove", "fingerless glove", "gloved hand", "hand wearing black glove"],
    "custom": [],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=Path, default=DEFAULT_FRAMES)
    parser.add_argument("--rectified-dir", type=Path, default=DEFAULT_BASE_DIR / "rectified_for_hamer")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_BASE_DIR / "sam3_bboxes")
    parser.add_argument("--sam3-root", type=Path, default=WRIST_CAM_ROOT / "third_party" / "sam3")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--hf-endpoint", default="https://hf-mirror.com")
    parser.add_argument("--no-hf", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp-dtype", choices=["float32", "bfloat16", "float16"], default="float32")
    parser.add_argument("--cameras", default=",".join(DEFAULT_CAMERAS))
    parser.add_argument("--group-range")
    parser.add_argument("--group-ids")
    parser.add_argument("--prompt-preset", choices=sorted(PROMPT_PRESETS), default="bare")
    parser.add_argument("--prompt", action="append", dest="prompts")
    parser.add_argument("--confidence-threshold", type=float, default=0.35)
    parser.add_argument("--max-hands", type=int, default=2)
    parser.add_argument("--hand-selection", choices=["area", "score", "score-area"], default="score-area")
    parser.add_argument("--nms-iou", type=float, default=0.5)
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


def main() -> None:
    args = parse_args()
    args.sam3_root = args.sam3_root.expanduser().resolve()
    if args.checkpoint:
        args.checkpoint = args.checkpoint.expanduser().resolve()
    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint
    if args.max_hands < 1:
        raise SystemExit("--max-hands must be >= 1")
    if args.max_hands > 1 and args.merge_mode == "largest":
        args.merge_mode = "none"
    args.resolved_is_right = resolve_is_right(args.handedness, args.is_right)

    cameras = parse_cameras(args.cameras)
    group_ids = parse_group_ids(args.group_range, args.group_ids)
    suffix = range_suffix(group_ids)
    records = filter_frame_records(args.frames, cameras, group_ids)
    output_path = args.output_dir / f"sam3_bboxes_{suffix}.jsonl"

    if args.dry_run:
        print(json.dumps({"records": len(records), "cameras": sorted(cameras), "suffix": suffix}, indent=2))
        return
    if output_path.exists() and not args.overwrite:
        raise SystemExit(f"{output_path} exists; pass --overwrite to replace it")

    ensure_sam3_importable(args.sam3_root)
    from sam3.model.sam3_image_processor import Sam3Processor  # type: ignore
    from sam3.model_builder import build_sam3_image_model  # type: ignore

    prompts = build_prompts(args)
    model = build_sam3_image_model(
        device=args.device,
        checkpoint_path=str(args.checkpoint) if args.checkpoint else None,
        load_from_HF=not args.no_hf,
    )
    if args.device == "cuda" and args.amp_dtype != "float32":
        model = model.to(dtype=torch_dtype_from_name(args.amp_dtype))
    else:
        model = model.float()
    processor = Sam3Processor(model, device=args.device, confidence_threshold=args.confidence_threshold)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = args.output_dir / "masks" / suffix
    bbox_debug_dir = args.output_dir / "debug" / suffix
    mask_debug_dir = args.output_dir / "mask_debug" / suffix

    stats = {"records": len(records), "frames_with_hands": 0, "hands": 0, "missing_images": 0}
    print(f"[SAM3] start cameras={','.join(sorted(cameras))} records={len(records)} output={output_path}", flush=True)
    with output_path.open("w", encoding="utf-8") as out:
        progress = tqdm(records, desc=f"SAM3 {','.join(sorted(cameras))}", unit="image", position=args.progress_position)
        for index, record in enumerate(progress, start=1):
            group_id = int(record["group_id"])
            camera_id = record["camera_id"]
            image_path = args.rectified_dir / rectified_rel_path(record)
            image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image_bgr is None:
                stats["missing_images"] += 1
                continue
            image_rgb = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
            with inference_context(args):
                state = processor.set_image(image_rgb)
                detections = []
                for prompt in prompts:
                    output = processor.set_text_prompt(prompt=prompt, state=state)
                    detections.extend(collect_detections(output, prompt, image_bgr.shape, args))
            hands = merge_detections(detections, image_bgr.shape, args)
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
            if args.progress_every > 0 and (index % args.progress_every == 0 or index == len(records)):
                progress.set_postfix_str(f"group={group_id} hands={len(json_hands)} total={stats['hands']}")

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
        "prompt_preset": args.prompt_preset,
        "prompts": prompts,
        "max_hands": args.max_hands,
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
