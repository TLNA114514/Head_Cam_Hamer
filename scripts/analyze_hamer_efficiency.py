#!/usr/bin/env python3
"""Analyze HaMeR model-load, forward-sample, and output-I/O costs without running inference."""

from __future__ import annotations

import argparse
import glob
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


from dependency_paths import DEFAULT_HAMER_ROOT


DEFAULT_CHECKPOINT = DEFAULT_HAMER_ROOT / "_DATA" / "hamer_ckpts" / "checkpoints" / "hamer.ckpt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", action="append", type=Path)
    parser.add_argument("--jobs-glob")
    parser.add_argument("--predictions", action="append", type=Path)
    parser.add_argument("--predictions-glob")
    parser.add_argument("--rendered-dir", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--candidate-bbox-scales", default="1.0,1.1,1.2")
    parser.add_argument("--candidate-scale-policy", choices=["fixed", "mask-adaptive"], default="fixed")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def resolve_paths(explicit: list[Path] | None, pattern: str | None) -> list[Path]:
    paths = list(explicit or [])
    if pattern:
        paths.extend(Path(item) for item in glob.glob(pattern))
    return sorted(set(path.resolve() for path in paths))


def iter_jsonl(path: Path) -> Iterable[tuple[str, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as source:
        for line in source:
            if line.strip():
                yield line, json.loads(line)


def parse_scales(value: str) -> list[float]:
    scales = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not scales:
        raise SystemExit("--candidate-bbox-scales must contain at least one value")
    return scales


def readable_mask(job: dict[str, Any]) -> bool:
    mask_path = job.get("sam3_mask_path")
    return bool(mask_path and Path(mask_path).is_file())


def collect_jobs(
    paths: list[Path],
    scale_count: int,
    scale_policy: str,
    batch_size: int,
    chunk_size: int,
) -> dict[str, Any]:
    unique_jobs: dict[tuple[int, str, int], dict[str, Any]] = {}
    groups: set[int] = set()
    cameras: set[str] = set()
    for path in paths:
        for _line, record in iter_jsonl(path):
            if record.get("type") != "hamer_multiview_jobs":
                continue
            group_id = int(record["group_id"])
            camera_id = str(record["camera_id"])
            groups.add(group_id)
            cameras.add(camera_id)
            for fallback_index, job in enumerate(record.get("jobs") or []):
                if job.get("debug_only"):
                    continue
                job_index = int(job.get("job_index", fallback_index))
                unique_jobs[(group_id, camera_id, job_index)] = job

    handedness = Counter()
    candidate_samples = 0
    adaptive_jobs = 0
    adaptive_candidates_avoided = 0
    isolated_batches = 0
    samples_by_camera: Counter[str] = Counter()
    for (group_id, camera_id, _job_index), job in unique_jobs.items():
        _ = group_id
        side_count = 1 if job.get("handedness") in {"Left", "Right"} else 2
        label = str(job.get("handedness") or "unknown")
        handedness[label] += 1
        job_scale_count = scale_count
        if scale_policy == "mask-adaptive" and scale_count > 1 and not readable_mask(job):
            job_scale_count = 1
            adaptive_jobs += 1
            adaptive_candidates_avoided += side_count * (scale_count - job_scale_count)
        samples = side_count * job_scale_count
        candidate_samples += samples
        samples_by_camera[camera_id] += samples
        isolated_batches += math.ceil(samples / batch_size)

    packed_batches = sum(math.ceil(samples / batch_size) for samples in samples_by_camera.values())
    chunk_count = math.ceil(len(groups) / chunk_size) if groups else 0
    per_camera_loads = len(cameras)
    per_chunk_loads = len(cameras) * chunk_count
    per_sequence_loads = 1 if unique_jobs else 0
    return {
        "input_files": [str(path) for path in paths],
        "groups": len(groups),
        "cameras": sorted(cameras),
        "jobs": len(unique_jobs),
        "handedness": dict(sorted(handedness.items())),
        "candidate_forward_samples": candidate_samples,
        "mask_adaptive_jobs": adaptive_jobs,
        "mask_adaptive_candidates_avoided": adaptive_candidates_avoided,
        "forward_batches_job_isolated": isolated_batches,
        "forward_batches_camera_packed_lower_bound": packed_batches,
        "model_loads_per_chunk": per_chunk_loads,
        "model_loads_per_camera": per_camera_loads,
        "model_loads_per_sequence": per_sequence_loads,
        "model_load_reduction_factor": per_chunk_loads / per_camera_loads if per_camera_loads else None,
        "model_load_reduction_factor_per_sequence": per_chunk_loads / per_sequence_loads if per_sequence_loads else None,
    }


def collect_predictions(paths: list[Path]) -> dict[str, Any]:
    records = 0
    original_bytes = 0
    compact_bytes = 0
    vertices_records = 0
    mano_param_records = 0
    overlay_records = 0
    scales: Counter[str] = Counter()
    for path in paths:
        for line, record in iter_jsonl(path):
            if record.get("type") != "hamer_multiview_prediction":
                continue
            records += 1
            original_bytes += len(line.encode("utf-8"))
            vertices_records += isinstance(record.get("hamer_vertices_cam"), list)
            mano_param_records += isinstance(record.get("mano_params_rotmat"), dict)
            overlay_records += bool(record.get("rendered_overlay_path"))
            scales[str(record.get("bbox_scale"))] += 1
            record["hamer_vertices_cam"] = None
            record["mano_params_rotmat"] = None
            record["mano_param_source"] = None
            record["rendered_overlay_path"] = None
            compact_bytes += len((json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8"))
    return {
        "input_files": [str(path) for path in paths],
        "records": records,
        "selected_bbox_scales": dict(sorted(scales.items())),
        "records_with_vertices": vertices_records,
        "records_with_mano_params": mano_param_records,
        "records_with_overlays": overlay_records,
        "jsonl_bytes": original_bytes,
        "joint_only_jsonl_bytes": compact_bytes,
        "jsonl_reduction_fraction": 1.0 - compact_bytes / original_bytes if original_bytes else None,
    }


def directory_bytes(path: Path | None) -> int | None:
    if path is None or not path.exists():
        return None
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.chunk_size < 1:
        raise SystemExit("--batch-size and --chunk-size must be positive")
    scales = parse_scales(args.candidate_bbox_scales)
    job_paths = resolve_paths(args.jobs, args.jobs_glob)
    prediction_paths = resolve_paths(args.predictions, args.predictions_glob)
    jobs = collect_jobs(job_paths, len(scales), args.candidate_scale_policy, args.batch_size, args.chunk_size)
    predictions = collect_predictions(prediction_paths)
    checkpoint_bytes = args.checkpoint.stat().st_size if args.checkpoint.exists() else None
    loads_per_chunk = jobs["model_loads_per_chunk"]
    loads_per_camera = jobs["model_loads_per_camera"]
    loads_per_sequence = jobs["model_loads_per_sequence"]
    report = {
        "analysis_type": "static_workload_not_wall_clock_benchmark",
        "candidate_bbox_scales": scales,
        "candidate_scale_policy": args.candidate_scale_policy,
        "batch_size": args.batch_size,
        "chunk_size": args.chunk_size,
        "checkpoint": str(args.checkpoint),
        "checkpoint_bytes": checkpoint_bytes,
        "estimated_checkpoint_bytes_read_per_chunk": checkpoint_bytes * loads_per_chunk if checkpoint_bytes is not None else None,
        "estimated_checkpoint_bytes_read_per_camera": checkpoint_bytes * loads_per_camera if checkpoint_bytes is not None else None,
        "estimated_checkpoint_bytes_read_per_sequence": checkpoint_bytes * loads_per_sequence if checkpoint_bytes is not None else None,
        "jobs": jobs,
        "pipeline_model_loads": {
            "sam3_per_chunk": loads_per_chunk,
            "sam3_per_sequence": loads_per_sequence,
            "hamer_per_chunk": loads_per_chunk,
            "hamer_per_camera": loads_per_camera,
            "hamer_per_sequence": loads_per_sequence,
        },
        "predictions": predictions,
        "rendered_overlay_bytes": directory_bytes(args.rendered_dir),
    }
    rendered_bytes = report["rendered_overlay_bytes"] or 0
    saved_json_bytes = predictions["jsonl_bytes"] - predictions["joint_only_jsonl_bytes"]
    report["avoidable_output_bytes_for_joint_only_pipeline"] = saved_json_bytes + rendered_bytes
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
