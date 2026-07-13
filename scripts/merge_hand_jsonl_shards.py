#!/usr/bin/env python3
"""Merge non-overlapping hand-frame JSONL shards deterministically by group_id."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=Path, help="Input JSONL shard; may be passed repeatedly.")
    parser.add_argument("--glob", dest="input_glob", help="Optional glob expanded in lexical order.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def canonical(record: dict[str, Any]) -> str:
    return json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def main() -> None:
    args = parse_args()
    paths = list(args.input or [])
    if args.input_glob:
        paths.extend(Path(value) for value in sorted(glob.glob(args.input_glob)))
    paths = list(dict.fromkeys(paths))
    if not paths:
        raise SystemExit("Provide at least one --input or --glob")
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise SystemExit(f"Missing input shards: {', '.join(str(path) for path in missing)}")
    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"{args.output} exists; pass --overwrite to replace it")

    frames: dict[int, dict[str, Any]] = {}
    duplicates = 0
    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if "group_id" not in record:
                    raise SystemExit(f"{path}:{line_number}: missing group_id")
                group_id = int(record["group_id"])
                old = frames.get(group_id)
                if old is not None:
                    if canonical(old) != canonical(record):
                        raise SystemExit(f"Conflicting duplicate group_id {group_id}: {path}:{line_number}")
                    duplicates += 1
                    continue
                frames[group_id] = record

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for group_id in sorted(frames):
            f.write(json.dumps(frames[group_id], ensure_ascii=False, separators=(",", ":")) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "frames": len(frames),
                "duplicate_identical_frames": duplicates,
                "input_shards": [str(path) for path in paths],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
