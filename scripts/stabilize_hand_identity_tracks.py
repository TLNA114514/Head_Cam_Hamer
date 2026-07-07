#!/usr/bin/env python3
"""Stabilize handedness as a persistent SAM3 track property."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hamer_multiview_utils import (
    DEFAULT_BASE_DIR,
    DEFAULT_CAMERAS,
    bbox_iou,
    iter_jsonl,
    mediapipe_bbox,
    opposite_handedness,
    parse_cameras,
    parse_group_ids,
    range_suffix,
)
from progress_utils import tqdm


@dataclass
class TrackState:
    locked_handedness: str = "unknown"
    confidence: float = 0.0
    source: str = "unlocked"
    votes: dict[str, float] = field(default_factory=lambda: {"Left": 0.0, "Right": 0.0})
    lock_votes: int = 0
    opposite_streak: int = 0
    seen_count: int = 0
    missing_count: int = 0
    last_group_id: int | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracks", type=Path, default=DEFAULT_BASE_DIR / "sam3_tracks" / "sam3_tracks_all.jsonl")
    parser.add_argument("--mediapipe", type=Path, default=DEFAULT_BASE_DIR / "landmarks.jsonl")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_BASE_DIR / "sam3_tracks_stabilized")
    parser.add_argument("--cameras", default=",".join(DEFAULT_CAMERAS))
    parser.add_argument("--group-range")
    parser.add_argument("--group-ids")
    parser.add_argument("--match-iou", type=float, default=0.25)
    parser.add_argument("--mediapipe-vote-score", type=float, default=0.70)
    parser.add_argument("--mediapipe-strong-score", type=float, default=0.85)
    parser.add_argument("--lock-votes", type=int, default=3)
    parser.add_argument("--unlock-votes", type=int, default=5)
    parser.add_argument("--max-missing", type=int, default=8)
    parser.add_argument("--max-hands", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--progress-position", type=int, default=int(os.environ.get("TQDM_POSITION", "0")))
    return parser.parse_args()


def load_mediapipe(path: Path, cameras: set[str], group_ids: set[int] | None, min_score: float) -> dict[tuple[int, str], list[dict[str, Any]]]:
    data: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    if not path.exists():
        return data
    for record in iter_jsonl(path):
        group_id = int(record["group_id"])
        camera_id = record["camera_id"]
        if camera_id not in cameras:
            continue
        if group_ids is not None and group_id not in group_ids:
            continue
        for hand in record.get("hands") or []:
            handedness = hand.get("handedness")
            score = hand.get("handedness_score")
            score = float(score) if isinstance(score, (int, float)) else -1.0
            bbox = mediapipe_bbox(hand, pad=1.25)
            if handedness not in {"Left", "Right"} or score < min_score or bbox is None:
                continue
            data[(group_id, camera_id)].append({"bbox": bbox, "handedness": handedness, "score": score})
    return data


def load_track_records(path: Path, cameras: set[str], group_ids: set[int] | None) -> list[dict[str, Any]]:
    records = []
    for record in iter_jsonl(path):
        group_id = int(record["group_id"])
        camera_id = record["camera_id"]
        if camera_id not in cameras:
            continue
        if group_ids is not None and group_id not in group_ids:
            continue
        records.append(record)
    records.sort(key=lambda item: (item["camera_id"], int(item["group_id"]), int(item.get("video_frame_index", 0))))
    return records


def best_mediapipe_vote(hand: dict[str, Any], mp_hands: list[dict[str, Any]], match_iou: float) -> tuple[str, float, float] | None:
    best = None
    best_iou = 0.0
    for mp in mp_hands:
        iou = bbox_iou(hand["bbox"], mp["bbox"])
        if iou > best_iou:
            best_iou = iou
            best = mp
    if best is None or best_iou < match_iou:
        return None
    return best["handedness"], float(best["score"]), best_iou


def update_state(state: TrackState, vote: tuple[str, float, str] | None, args: argparse.Namespace) -> None:
    state.seen_count += 1
    state.missing_count = 0
    if vote is None:
        return
    handedness, weight, source = vote
    if handedness not in {"Left", "Right"}:
        return
    state.votes[handedness] += weight
    winner = "Left" if state.votes["Left"] >= state.votes["Right"] else "Right"
    diff = abs(state.votes["Left"] - state.votes["Right"])
    total = max(1e-6, state.votes["Left"] + state.votes["Right"])
    if state.locked_handedness == "unknown":
        if diff >= args.lock_votes:
            state.locked_handedness = winner
            state.confidence = min(1.0, 0.5 + diff / max(total, args.lock_votes) * 0.5)
            state.source = source
            state.lock_votes = int(diff)
        return
    if handedness == state.locked_handedness:
        state.opposite_streak = 0
        state.confidence = min(1.0, state.confidence + 0.04 * weight)
        return
    state.opposite_streak += 1
    if state.opposite_streak >= args.unlock_votes and weight >= 2.0:
        state.locked_handedness = handedness
        state.confidence = 0.70
        state.source = f"flipped_by_{source}"
        state.opposite_streak = 0


def infer_posthoc_track_id(hand: dict[str, Any], camera_id: str, active: dict[str, dict[str, Any]], next_index: dict[str, int], frame_index: int) -> str:
    if hand.get("track_id") is not None:
        return str(hand["track_id"])
    best_id = None
    best_iou = 0.0
    for track_id, state in active.items():
        if frame_index - int(state["frame_index"]) > 8:
            continue
        iou = bbox_iou(hand["bbox"], state["bbox"])
        if iou > best_iou:
            best_iou = iou
            best_id = track_id
    if best_id is None or best_iou < 0.20:
        best_id = f"{camera_id}_s{next_index[camera_id]:02d}"
        next_index[camera_id] += 1
    active[best_id] = {"bbox": hand["bbox"], "frame_index": frame_index}
    return best_id


def enforce_two_hand_exclusivity(hands: list[dict[str, Any]]) -> None:
    valid = [hand for hand in hands if not hand.get("debug_only")]
    if len(valid) != 2:
        return
    labels = [hand.get("locked_handedness") or hand.get("handedness") for hand in valid]
    if labels[0] in {"Left", "Right"} and labels[1] == "unknown":
        valid[1]["locked_handedness"] = opposite_handedness(labels[0])
        valid[1]["handedness"] = valid[1]["locked_handedness"]
        valid[1]["handedness_source"] = "mutual_exclusion"
    elif labels[1] in {"Left", "Right"} and labels[0] == "unknown":
        valid[0]["locked_handedness"] = opposite_handedness(labels[1])
        valid[0]["handedness"] = valid[0]["locked_handedness"]
        valid[0]["handedness_source"] = "mutual_exclusion"
    elif labels[0] == labels[1] and labels[0] in {"Left", "Right"}:
        scores = [
            float(hand.get("handedness_confidence", 0.0) or 0.0) + 0.01 * float(hand.get("track_seen_count", 0) or 0)
            for hand in valid
        ]
        flip_index = 0 if scores[0] < scores[1] else 1
        valid[flip_index]["locked_handedness"] = opposite_handedness(labels[flip_index])
        valid[flip_index]["handedness"] = valid[flip_index]["locked_handedness"]
        valid[flip_index]["handedness_source"] = "mutual_exclusion_duplicate_repair"


def main() -> None:
    args = parse_args()
    cameras = parse_cameras(args.cameras)
    group_ids = parse_group_ids(args.group_range, args.group_ids)
    suffix = range_suffix(group_ids)
    output_path = args.output_dir / f"sam3_tracks_stabilized_{suffix}.jsonl"
    if args.dry_run:
        print(json.dumps({"tracks": str(args.tracks), "output": str(output_path), "suffix": suffix}, indent=2))
        return
    if output_path.exists() and not args.overwrite:
        raise SystemExit(f"{output_path} exists; pass --overwrite to replace it")
    if not args.tracks.exists():
        raise SystemExit(f"tracks not found: {args.tracks}")

    mp = load_mediapipe(args.mediapipe, cameras, group_ids, args.mediapipe_vote_score)
    records = load_track_records(args.tracks, cameras, group_ids)
    states: dict[tuple[str, str], TrackState] = defaultdict(TrackState)
    active_by_camera: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    next_index: dict[str, int] = defaultdict(lambda: 1)
    stats = defaultdict(int)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out:
        for record in tqdm(records, desc="stabilize tracks", unit="frame", position=args.progress_position):
            group_id = int(record["group_id"])
            camera_id = record["camera_id"]
            frame_index = int(record.get("video_frame_index", group_id))
            mp_hands = mp.get((group_id, camera_id), [])
            out_hands = []
            for raw_hand in (record.get("hands") or [])[: args.max_hands]:
                hand = dict(raw_hand)
                hand["bbox"] = [float(v) for v in hand["bbox"]]
                track_id = infer_posthoc_track_id(hand, camera_id, active_by_camera[camera_id], next_index, frame_index)
                hand["track_id"] = track_id
                state = states[(camera_id, track_id)]
                vote_info = best_mediapipe_vote(hand, mp_hands, args.match_iou)
                vote = None
                if vote_info is not None:
                    handedness, mp_score, iou = vote_info
                    weight = 2.0 if mp_score >= args.mediapipe_strong_score else 1.0
                    vote = (handedness, weight, "mediapipe_strong" if weight >= 2.0 else "mediapipe")
                    hand["mediapipe_handedness_vote"] = handedness
                    hand["mediapipe_handedness_score"] = mp_score
                    hand["mediapipe_match_iou"] = iou
                update_state(state, vote, args)
                hand["locked_handedness"] = state.locked_handedness
                hand["handedness"] = state.locked_handedness
                hand["handedness_confidence"] = state.confidence
                hand["handedness_source"] = state.source
                hand["lock_votes"] = state.lock_votes
                hand["track_seen_count"] = state.seen_count
                hand["missing_count"] = state.missing_count
                out_hands.append(hand)
                stats["hands"] += 1
                if vote is not None:
                    stats[f"mp_vote:{vote[0]}"] += 1
                if state.locked_handedness in {"Left", "Right"}:
                    stats[f"locked:{state.locked_handedness}"] += 1
            enforce_two_hand_exclusivity(out_hands)
            for hand in out_hands:
                if hand.get("handedness") in {"Left", "Right"}:
                    hand["is_right"] = 1 if hand["handedness"] == "Right" else 0
            stats["frames"] += 1
            out.write(
                json.dumps(
                    {
                        "type": "sam3_stabilized_tracks",
                        "group_id": group_id,
                        "camera_id": camera_id,
                        "video_frame_index": frame_index,
                        "rectified_image_path": record.get("rectified_image_path"),
                        "hands": out_hands,
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )
    with (args.output_dir / f"sam3_tracks_stabilized_config_{suffix}.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "tracks": str(args.tracks),
                "mediapipe": str(args.mediapipe),
                "output_path": str(output_path),
                "group_range": args.group_range,
                "group_ids": args.group_ids,
                "stats": dict(stats),
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
