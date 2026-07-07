#!/usr/bin/env python3
"""Track SAM3 hand masks over a per-camera JPEG sequence."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from hamer_multiview_utils import (
    DEFAULT_BASE_DIR,
    DEFAULT_CAMERAS,
    bbox_area,
    bbox_iou,
    draw_bbox,
    iter_jsonl,
    parse_cameras,
    parse_group_ids,
    range_suffix,
)
from progress_utils import tqdm


WRIST_CAM_ROOT = Path("/home/luojiangrui/ljr/wrist_cam")
WRIST_SCRIPTS = WRIST_CAM_ROOT / "scripts"
if str(WRIST_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(WRIST_SCRIPTS))

try:
    from run_sam3_bbox_on_video import ensure_sam3_importable  # type: ignore
except Exception:  # pragma: no cover
    ensure_sam3_importable = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-dir", type=Path, default=DEFAULT_BASE_DIR / "sam3_video_sequences")
    parser.add_argument("--seed-sam3", type=Path, help="Per-frame SAM3 bbox JSONL used for prompts and fallback.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_BASE_DIR / "sam3_tracks")
    parser.add_argument("--sam3-root", type=Path, default=WRIST_CAM_ROOT / "third_party" / "sam3")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--hf-endpoint", default="https://hf-mirror.com")
    parser.add_argument("--no-hf", action="store_true")
    parser.add_argument("--version", choices=["sam3", "sam3.1"], default="sam3.1")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cameras", default=",".join(DEFAULT_CAMERAS))
    parser.add_argument("--group-range")
    parser.add_argument("--group-ids")
    parser.add_argument("--max-hands", type=int, default=2)
    parser.add_argument("--seed-search-frames", type=int, default=8)
    parser.add_argument("--min-seed-score", type=float, default=0.20)
    parser.add_argument("--mask-threshold", type=float, default=0.0)
    parser.add_argument("--min-mask-area-frac", type=float, default=0.001)
    parser.add_argument("--fallback-on-failure", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-debug", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--progress-position", type=int, default=int(os.environ.get("TQDM_POSITION", "0")))
    return parser.parse_args()


def load_frame_map(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return sorted(data, key=lambda item: int(item["video_frame_index"]))


def load_seed_sam3(path: Path | None, cameras: set[str], group_ids: set[int] | None) -> dict[tuple[int, str], list[dict[str, Any]]]:
    data: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    if path is None or not path.exists():
        return data
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
            item["score"] = float(item.get("score", 0.0) or 0.0)
            data[(group_id, camera_id)].append(item)
    for hands in data.values():
        hands.sort(key=lambda item: (item.get("score", 0.0), bbox_area(item["bbox"])), reverse=True)
    return data


def find_seed_hands(frame_map: list[dict[str, Any]], camera_id: str, seeds: dict[tuple[int, str], list[dict[str, Any]]], args: argparse.Namespace) -> tuple[int | None, list[dict[str, Any]]]:
    for item in frame_map[: max(1, args.seed_search_frames)]:
        group_id = int(item["group_id"])
        hands = [
            hand
            for hand in seeds.get((group_id, camera_id), [])
            if float(hand.get("score", 0.0) or 0.0) >= args.min_seed_score
        ][: args.max_hands]
        if hands:
            return int(item["video_frame_index"]), hands
    return None, []


def xyxy_to_xywh(bbox: list[float]) -> list[float]:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return [x1, y1, max(1.0, x2 - x1), max(1.0, y2 - y1)]


def mask_to_bbox(mask: np.ndarray) -> list[float] | None:
    ys, xs = np.where(mask.astype(bool))
    if len(xs) == 0:
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def score_to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if hasattr(value, "detach"):
            value = value.detach().float().cpu().reshape(-1)
            return float(value[0]) if len(value) else None
        arr = np.asarray(value).reshape(-1)
        return float(arr[0]) if len(arr) else None
    except Exception:
        return None


def tensor_to_numpy(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def save_debug(image_path: Path, hands: list[dict[str, Any]], output_path: Path) -> None:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return
    for hand in hands:
        label = f"T{hand.get('track_id')} {hand.get('sam3_object_score', hand.get('score', 0.0)):.2f}"
        draw_bbox(image, hand["bbox"], (80, 220, 80), label)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)


def native_track_camera(
    camera_id: str,
    camera_root: Path,
    frame_map: list[dict[str, Any]],
    seeds: dict[tuple[int, str], list[dict[str, Any]]],
    output_dir: Path,
    suffix: str,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    if ensure_sam3_importable is None:
        raise RuntimeError("Cannot import wrist_cam SAM3 helpers")
    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint
    ensure_sam3_importable(args.sam3_root.expanduser().resolve())
    from sam3 import build_sam3_predictor  # type: ignore

    seed_frame, seed_hands = find_seed_hands(frame_map, camera_id, seeds, args)
    if seed_frame is None or not seed_hands:
        raise RuntimeError(f"No seed hands found for {camera_id}")

    build_kwargs = {
        "version": args.version,
        "compile": False,
        "async_loading_frames": False,
        "max_num_objects": max(2, args.max_hands),
    }
    if args.checkpoint:
        build_kwargs["checkpoint_path"] = str(args.checkpoint.expanduser().resolve())
    if args.no_hf and not args.checkpoint:
        raise RuntimeError("--no-hf requires --checkpoint for SAM3 native video tracking")

    model = build_sam3_predictor(**build_kwargs)
    response = model.handle_request({"type": "start_session", "resource_path": str(camera_root / "images")})
    session_id = response["session_id"]
    seed_by_obj = {}
    for obj_id, hand in enumerate(seed_hands[: args.max_hands], start=1):
        model.handle_request(
            {
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": seed_frame,
                "bounding_boxes": [xyxy_to_xywh(hand["bbox"])],
                "bounding_box_labels": [1],
                "obj_id": obj_id,
                "rel_coordinates": False,
                "output_prob_thresh": args.mask_threshold,
            }
        )
        seed_by_obj[obj_id] = hand

    frame_lookup = {int(item["video_frame_index"]): item for item in frame_map}
    records_by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    stream = model.handle_stream_request(
        {
            "type": "propagate_in_video",
            "session_id": session_id,
            "start_frame_index": seed_frame,
            "propagation_direction": "both",
            "output_prob_thresh": args.mask_threshold,
        }
    )
    total = len(frame_map)
    for response in tqdm(stream, total=total, desc=f"SAM3 track {camera_id}", unit="frame", position=args.progress_position):
        frame_index = response.get("frame_index")
        if frame_index is None or int(frame_index) not in frame_lookup:
            continue
        outputs = response.get("outputs", response)
        obj_ids = tensor_to_numpy(outputs.get("out_obj_ids"))
        masks = tensor_to_numpy(outputs.get("out_binary_masks"))
        raw_scores = outputs.get("object_score_logits")
        if raw_scores is None:
            raw_scores = outputs.get("out_object_scores")
        scores = tensor_to_numpy(raw_scores)
        if scores is not None:
            scores = np.asarray(scores).reshape(-1)
        if obj_ids is None or masks is None:
            continue
        frame_info = frame_lookup[int(frame_index)]
        group_id = int(frame_info["group_id"])
        image_path = Path(frame_info["rectified_image_path"])
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        height, width = image.shape[:2]
        min_area = width * height * args.min_mask_area_frac
        for index, obj_id_value in enumerate(np.asarray(obj_ids).reshape(-1)):
            obj_id = int(obj_id_value)
            mask = np.asarray(masks[index])
            if mask.ndim == 3:
                mask = mask[0]
            mask_bool = mask > args.mask_threshold
            if float(mask_bool.sum()) < min_area:
                continue
            bbox = mask_to_bbox(mask_bool)
            if bbox is None:
                continue
            mask_path = output_dir / "masks" / suffix / camera_id / f"{group_id:08d}_track{obj_id:02d}.png"
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(mask_path), mask_bool.astype(np.uint8) * 255)
            score = score_to_float(scores[index] if scores is not None and index < len(scores) else None)
            seed = seed_by_obj.get(obj_id, {})
            records_by_frame[int(frame_index)].append(
                {
                    "track_id": f"{camera_id}_t{obj_id:02d}",
                    "track_numeric_id": obj_id,
                    "video_frame_index": int(frame_index),
                    "group_id": group_id,
                    "camera_id": camera_id,
                    "bbox": bbox,
                    "mask_path": str(mask_path),
                    "score": float(score) if score is not None else float(seed.get("score", 1.0) or 1.0),
                    "sam3_object_score": score,
                    "track_age": max(0, abs(int(frame_index) - int(seed_frame))),
                    "track_backend": "sam3_native",
                    "seed_frame_index": seed_frame,
                    "seed_bbox": seed.get("bbox"),
                    "seed_score": seed.get("score"),
                    "prompt": seed.get("prompt"),
                }
            )
    try:
        model.handle_request({"type": "close_session", "session_id": session_id})
    except Exception:
        pass
    return records_from_tracks(frame_map, records_by_frame, camera_id, output_dir, suffix, args)


def fallback_track_camera(
    camera_id: str,
    frame_map: list[dict[str, Any]],
    seeds: dict[tuple[int, str], list[dict[str, Any]]],
    output_dir: Path,
    suffix: str,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    next_id = 1
    active: dict[int, dict[str, Any]] = {}
    records_by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in tqdm(frame_map, desc=f"SAM3 posthoc {camera_id}", unit="frame", position=args.progress_position):
        frame_index = int(item["video_frame_index"])
        group_id = int(item["group_id"])
        hands = seeds.get((group_id, camera_id), [])[: args.max_hands]
        assigned_tracks: set[int] = set()
        for hand in hands:
            best_id = None
            best_iou = 0.0
            for track_id, state in active.items():
                if track_id in assigned_tracks:
                    continue
                if frame_index - int(state["last_frame"]) > 8:
                    continue
                iou = bbox_iou(hand["bbox"], state["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_id = track_id
            if best_id is None or best_iou < 0.20:
                best_id = next_id
                next_id += 1
            assigned_tracks.add(best_id)
            active[best_id] = {"bbox": hand["bbox"], "last_frame": frame_index}
            item_out = dict(hand)
            item_out.update(
                {
                    "track_id": f"{camera_id}_p{best_id:02d}",
                    "track_numeric_id": best_id,
                    "video_frame_index": frame_index,
                    "group_id": group_id,
                    "camera_id": camera_id,
                    "sam3_object_score": hand.get("score"),
                    "track_age": frame_index,
                    "track_backend": "posthoc_iou_fallback",
                }
            )
            records_by_frame[frame_index].append(item_out)
    return records_from_tracks(frame_map, records_by_frame, camera_id, output_dir, suffix, args)


def records_from_tracks(
    frame_map: list[dict[str, Any]],
    records_by_frame: dict[int, list[dict[str, Any]]],
    camera_id: str,
    output_dir: Path,
    suffix: str,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    records = []
    for item in frame_map:
        frame_index = int(item["video_frame_index"])
        group_id = int(item["group_id"])
        deduped: dict[str, dict[str, Any]] = {}
        for hand in records_by_frame.get(frame_index, []):
            track_id = str(hand.get("track_id"))
            previous = deduped.get(track_id)
            if previous is None or bbox_area(hand["bbox"]) > bbox_area(previous["bbox"]):
                deduped[track_id] = hand
        hands = list(deduped.values())
        hands.sort(key=lambda hand: (str(hand.get("track_id")), -bbox_area(hand["bbox"])))
        hands = hands[: args.max_hands]
        if args.save_debug:
            save_debug(
                Path(item["rectified_image_path"]),
                hands,
                output_dir / "debug" / suffix / camera_id / f"{group_id:08d}.jpg",
            )
        records.append(
            {
                "type": "sam3_native_tracks",
                "group_id": group_id,
                "camera_id": camera_id,
                "video_frame_index": frame_index,
                "rectified_image_path": item["rectified_image_path"],
                "hands": hands,
            }
        )
    return records


def main() -> None:
    args = parse_args()
    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint
    cameras = parse_cameras(args.cameras)
    group_ids = parse_group_ids(args.group_range, args.group_ids)
    suffix = range_suffix(group_ids)
    output_path = args.output_dir / f"sam3_tracks_{suffix}.jsonl"
    if args.dry_run:
        print(json.dumps({"cameras": sorted(cameras), "suffix": suffix, "output": str(output_path)}, indent=2))
        return
    if output_path.exists() and not args.overwrite:
        raise SystemExit(f"{output_path} exists; pass --overwrite to replace it")
    seeds = load_seed_sam3(args.seed_sam3, cameras, group_ids)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_records = []
    stats = {"frames": 0, "hands": 0, "native_failures": 0, "fallback_frames": 0}
    for camera_id in sorted(cameras):
        camera_root = args.sequence_dir / suffix / camera_id
        frame_map_path = camera_root / "frame_map.json"
        if not frame_map_path.exists():
            raise SystemExit(f"frame_map not found: {frame_map_path}")
        frame_map = load_frame_map(frame_map_path)
        try:
            records = native_track_camera(camera_id, camera_root, frame_map, seeds, args.output_dir, suffix, args)
        except Exception as exc:
            if not args.fallback_on_failure:
                raise
            print(f"[SAM3 video] native tracking failed for {camera_id}; using posthoc fallback: {exc}", flush=True)
            stats["native_failures"] += 1
            records = fallback_track_camera(camera_id, frame_map, seeds, args.output_dir, suffix, args)
            stats["fallback_frames"] += len(records)
        all_records.extend(records)
    with output_path.open("w", encoding="utf-8") as f:
        for record in all_records:
            stats["frames"] += 1
            stats["hands"] += len(record.get("hands") or [])
            f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    with (args.output_dir / f"sam3_tracks_config_{suffix}.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "sequence_dir": str(args.sequence_dir),
                "seed_sam3": str(args.seed_sam3) if args.seed_sam3 else None,
                "sam3_root": str(args.sam3_root),
                "checkpoint": str(args.checkpoint) if args.checkpoint else None,
                "version": args.version,
                "cameras": sorted(cameras),
                "group_range": args.group_range,
                "group_ids": args.group_ids,
                "stats": stats,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
        f.write("\n")
    print("Summary")
    for key in sorted(stats):
        print(f"  {key}: {stats[key]}")
    print(f"wrote: {output_path}")


if __name__ == "__main__":
    main()
