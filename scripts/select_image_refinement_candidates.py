#!/usr/bin/env python3
"""Select between baseline and PnP-gated MANO image-refinement candidates.

The default prefers the PnP-gated candidate whenever it is available. It never
reads glove data; the default was selected from independent left/right offline
evaluations and is therefore usable on unlabeled recordings.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--gated", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--min-baseline-max-reprojection-px",
        type=float,
        default=0.0,
        help="Use the gated candidate only when baseline max per-view reprojection reaches this threshold; 0 prefers every available gated candidate.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_frames(path: Path) -> dict[int, dict[str, Any]]:
    frames = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if "group_id" in record:
                frames[int(record["group_id"])] = record
    return frames


def max_reprojection_error(hand: dict[str, Any]) -> float | None:
    values = []
    for item in (hand.get("projection_debug") or {}).values():
        if not isinstance(item, dict):
            continue
        value = item.get("mean_reprojection_error_px")
        if isinstance(value, (int, float)) and float(value) == float(value):
            values.append(float(value))
    return max(values) if values else None


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"{args.output} exists; pass --overwrite to replace it")
    baseline_frames = load_frames(args.baseline)
    gated_frames = load_frames(args.gated)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    stats = {"frames": 0, "baseline_hands": 0, "gated_hands": 0, "missing_gated_hand": 0}
    with args.output.open("w", encoding="utf-8") as out:
        for group_id in sorted(baseline_frames):
            baseline_frame = baseline_frames[group_id]
            gated_by_hand = {
                hand.get("handedness"): hand
                for hand in (gated_frames.get(group_id, {}).get("hands") or [])
                if hand.get("handedness") in {"Left", "Right"}
            }
            selected_hands = []
            for baseline_hand in baseline_frame.get("hands") or []:
                handedness = baseline_hand.get("handedness")
                gated_hand = gated_by_hand.get(handedness)
                max_reprojection = max_reprojection_error(baseline_hand)
                use_gated = (
                    gated_hand is not None
                    and (
                        args.min_baseline_max_reprojection_px <= 0.0
                        or (
                            max_reprojection is not None
                            and max_reprojection >= args.min_baseline_max_reprojection_px
                        )
                    )
                )
                if gated_hand is None:
                    stats["missing_gated_hand"] += 1
                chosen = dict(gated_hand if use_gated else baseline_hand)
                chosen["candidate_selection"] = {
                    "source": "pnp_gated" if use_gated else "baseline",
                    "baseline_max_reprojection_error_px": max_reprojection,
                    "min_baseline_max_reprojection_px": args.min_baseline_max_reprojection_px,
                }
                selected_hands.append(chosen)
                stats["gated_hands" if use_gated else "baseline_hands"] += 1
            frame = dict(baseline_frame)
            frame["hands"] = selected_hands
            frame["candidate_selection_rule"] = (
                "prefer_pnp_gated_when_available"
                if args.min_baseline_max_reprojection_px <= 0.0
                else "baseline_max_reprojection_threshold"
            )
            out.write(json.dumps(frame, ensure_ascii=False, separators=(",", ":")) + "\n")
            stats["frames"] += 1
    print(json.dumps({"output": str(args.output), "stats": stats}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
