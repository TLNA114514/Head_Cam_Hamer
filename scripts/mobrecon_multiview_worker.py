#!/usr/bin/env python3
"""Run MobRecon on existing multi-view hand jobs with one shared model instance."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from collections import Counter, defaultdict
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import torch

from hamer_multiview_utils import DEFAULT_BASE_DIR, iter_jsonl, parse_group_ids, range_suffix
from progress_utils import tqdm


from dependency_paths import DEFAULT_MOBRECON_ROOT
MANO_TO_MPII = np.asarray([0, 13, 14, 15, 20, 1, 2, 3, 16, 4, 5, 6, 17, 10, 11, 12, 19, 7, 8, 9, 18])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=Path, help="Existing HaMeR jobs JSONL.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_BASE_DIR / "mobrecon_per_view")
    parser.add_argument("--mobrecon-root", type=Path, default=DEFAULT_MOBRECON_ROOT)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--group-range")
    parser.add_argument("--group-ids")
    parser.add_argument("--camera-id")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--job-batch-size", type=int, default=64)
    parser.add_argument("--crop-scale", type=float, default=1.5)
    parser.add_argument("--input-size", type=int, default=128)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--precision", choices=["float32", "float16"], default="float32")
    parser.add_argument(
        "--image-source",
        choices=["hamer-frame", "rectified"],
        default="hamer-frame",
        help="Use the mask-blurred HaMeR frame or the original rectified frame.",
    )
    parser.add_argument("--torch-threads", type=int, default=12)
    parser.add_argument("--export-vertices", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--unknown-handedness",
        choices=["spatial", "skip", "left", "right"],
        default="spatial",
        help="MobRecon cannot infer handedness; spatial maps paired boxes left-to-right or uses a fixed fallback.",
    )
    parser.add_argument("--include-debug-only", action="store_true")
    parser.add_argument("--progress-position", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def iter_jobs(
    path: Path,
    camera_id: str | None,
    include_debug_only: bool,
    group_ids: set[int] | None,
) -> list[dict[str, Any]]:
    jobs = []
    for record in iter_jsonl(path):
        if record.get("type") != "hamer_multiview_jobs":
            continue
        if group_ids is not None and int(record["group_id"]) not in group_ids:
            continue
        if camera_id and record.get("camera_id") != camera_id:
            continue
        for job in record.get("jobs") or []:
            if job.get("debug_only") and not include_debug_only:
                continue
            jobs.append(job)
    return jobs


def job_key(job: dict[str, Any]) -> tuple[int, str, int]:
    return int(job["group_id"]), str(job["camera_id"]), int(job.get("job_index", 0))


def spatial_handedness(jobs: list[dict[str, Any]]) -> dict[tuple[int, str, int], str]:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = {}
    track_votes: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    result = {}
    for job in jobs:
        handedness = str(job.get("handedness") or "").title()
        track_id = job.get("track_id")
        if handedness in {"Left", "Right"}:
            if track_id is not None:
                track_votes[(str(job["camera_id"]), str(track_id))][handedness] += 1
            continue
        grouped.setdefault((int(job["group_id"]), str(job["camera_id"])), []).append(job)
    for unknown_jobs in grouped.values():
        if len(unknown_jobs) != 2:
            continue
        ordered = sorted(
            unknown_jobs,
            key=lambda job: 0.5 * (job["bbox_rectified_px"][0] + job["bbox_rectified_px"][2]),
        )
        result[job_key(ordered[0])] = "Left"
        result[job_key(ordered[1])] = "Right"
        for job, handedness in zip(ordered, ("Left", "Right")):
            track_id = job.get("track_id")
            if track_id is not None:
                track_votes[(str(job["camera_id"]), str(track_id))][handedness] += 1
    for unknown_jobs in grouped.values():
        for job in unknown_jobs:
            if job_key(job) in result or job.get("track_id") is None:
                continue
            votes = track_votes.get((str(job["camera_id"]), str(job["track_id"])))
            if votes:
                result[job_key(job)] = votes.most_common(1)[0][0]
    return result


def resolve_handedness(
    job: dict[str, Any], fallback: str, spatial_assignments: dict[tuple[int, str, int], str]
) -> str | None:
    handedness = str(job.get("handedness") or "").title()
    if handedness in {"Left", "Right"}:
        return handedness
    if fallback == "spatial":
        return spatial_assignments.get(job_key(job))
    if fallback == "left":
        return "Left"
    if fallback == "right":
        return "Right"
    return None


def square_crop(image: np.ndarray, bbox: list[float], scale: float, size: int) -> tuple[np.ndarray, np.ndarray]:
    x1, y1, x2, y2 = [float(value) for value in bbox]
    center_x = 0.5 * (x1 + x2)
    center_y = 0.5 * (y1 + y2)
    side = max(x2 - x1, y2 - y1, 2.0) * scale
    left = center_x - 0.5 * side
    top = center_y - 0.5 * side
    affine = np.asarray(
        [[(size - 1) / side, 0.0, -left * (size - 1) / side], [0.0, (size - 1) / side, -top * (size - 1) / side]],
        dtype=np.float32,
    )
    crop = cv2.warpAffine(
        image,
        affine,
        (size, size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    return crop, np.asarray([left, top, side], dtype=np.float32)


def prepare_sample(image: np.ndarray, job: dict[str, Any], handedness: str, args: argparse.Namespace) -> dict[str, Any]:
    crop, crop_geometry = square_crop(image, job["bbox_rectified_px"], args.crop_scale, args.input_size)
    if handedness == "Left":
        crop = np.ascontiguousarray(crop[:, ::-1])
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    tensor = rgb.astype(np.float32) / 127.5 - 1.0
    return {
        "job": job,
        "handedness": handedness,
        "input": torch.from_numpy(tensor.transpose(2, 0, 1)),
        "crop_geometry": crop_geometry,
    }


def next_ring(mesh: Any, last_ring: list[int], other: list[int], openmesh: Any) -> list[int]:
    result: list[int] = []

    def is_new(index: int) -> bool:
        return index not in last_ring and index not in other and index not in result

    for vertex_index in last_ring:
        vertex = openmesh.VertexHandle(vertex_index)
        after_last_ring = False
        for adjacent in mesh.vv(vertex):
            if after_last_ring and is_new(adjacent.idx()):
                result.append(adjacent.idx())
            if adjacent.idx() in last_ring:
                after_last_ring = True
        for adjacent in mesh.vv(vertex):
            if adjacent.idx() in last_ring:
                break
            if is_new(adjacent.idx()):
                result.append(adjacent.idx())
    return result


def spiral_indices(mesh: Any, length: int, openmesh: Any) -> torch.Tensor:
    spirals = []
    for center in mesh.vertices():
        one_ring = [vertex.idx() for vertex in mesh.vv(center)]
        spiral = [center.idx()]
        last_ring = one_ring
        following_ring = next_ring(mesh, last_ring, spiral, openmesh)
        spiral.extend(last_ring)
        while len(spiral) + len(following_ring) < length:
            if not following_ring:
                raise RuntimeError("MobRecon topology contains a disconnected mesh")
            last_ring = following_ring
            following_ring = next_ring(mesh, last_ring, spiral, openmesh)
            spiral.extend(last_ring)
        spiral.extend(following_ring)
        spirals.append(spiral[:length])
    return torch.tensor(spirals, dtype=torch.long)


def sparse_pool_like_input(features: torch.Tensor, transform: tuple[torch.Tensor, ...], dim: int = 1) -> torch.Tensor:
    row, col, value = transform
    row = row.to(features.device)
    col = col.to(features.device)
    value = value.to(device=features.device, dtype=features.dtype).unsqueeze(-1)
    selected = torch.index_select(features, dim, col) * value
    output = torch.zeros(
        features.size(0), row.size(0) // 3, features.size(-1), device=features.device, dtype=features.dtype
    )
    indices = row.unsqueeze(0).unsqueeze(-1).expand_as(selected)
    return torch.scatter_add(output, dim, indices, selected)


def load_model(root: Path, checkpoint: Path, device: torch.device) -> tuple[torch.nn.Module, np.ndarray, float]:
    try:
        import openmesh
    except ImportError as exc:
        raise SystemExit("OpenMesh is required once for MobRecon topology: pip install openmesh") from exc

    sys.path.insert(0, str(root))
    from cmr.models.mobrecon_densestack import MobRecon
    import mobrecon.models.modules as mobrecon_modules

    mobrecon_modules.Pool = sparse_pool_like_input
    with (root / "template" / "transform.pkl").open("rb") as transform_file:
        topology = pickle.load(transform_file, encoding="latin1")
    indices = []
    for faces, vertices in zip(topology["face"][:-1], topology["vertices"][:-1]):
        mesh = openmesh.TriMesh(np.asarray(vertices), np.asarray(faces))
        indices.append(spiral_indices(mesh, 9, openmesh))
    up_transforms = []
    for matrix in topology["up_transform"]:
        coo = matrix.tocoo()
        sparse = torch.sparse_coo_tensor(
            torch.tensor(np.vstack([coo.row, coo.col]), dtype=torch.long),
            torch.tensor(coo.data, dtype=torch.float32),
            coo.shape,
        ).coalesce()
        up_transforms.append((*sparse.indices(), sparse.values()))
    model = MobRecon(SimpleNamespace(out_channels=[32, 64, 128, 256], dsconv=False), indices, up_transforms)
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state.get("model_state_dict", state), strict=True)
    model = model.to(device).eval()
    joint_regressor = np.load(root / "template" / "j_reg.npy").astype(np.float32)
    model_mib = sum(parameter.numel() * parameter.element_size() for parameter in model.parameters()) / 2**20
    return model, joint_regressor, model_mib


def choose_device(value: str) -> torch.device:
    if value == "cpu":
        return torch.device("cpu")
    if value == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("--device cuda requested, but CUDA is unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def restore_outputs(
    mesh: np.ndarray,
    uv: np.ndarray,
    sample: dict[str, Any],
    joint_regressor: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mesh_m = mesh.astype(np.float32) * 0.20
    joints_m = (joint_regressor @ mesh_m)[MANO_TO_MPII]
    root = joints_m[0].copy()
    mesh_m -= root
    joints_m -= root
    uv = uv.astype(np.float32).copy()
    if sample["handedness"] == "Left":
        mesh_m[:, 0] *= -1.0
        joints_m[:, 0] *= -1.0
        uv[:, 0] = 1.0 - uv[:, 0]
    left, top, side = sample["crop_geometry"]
    points_2d = np.empty_like(uv)
    points_2d[:, 0] = left + uv[:, 0] * side
    points_2d[:, 1] = top + uv[:, 1] * side
    return joints_m, mesh_m, points_2d


def prediction_record(
    job: dict[str, Any],
    sample: dict[str, Any],
    joints_m: np.ndarray,
    mesh_m: np.ndarray,
    points_2d: np.ndarray,
    export_vertices: bool,
    crop_scale: float,
    unknown_handedness: str,
) -> dict[str, Any]:
    confidence = float(np.clip(job.get("sam3_score") or 0.5, 0.0, 1.0))
    return {
        "type": "hand_mesh_multiview_prediction",
        "model_name": "mobrecon_densestack",
        "group_id": int(job["group_id"]),
        "camera_id": str(job["camera_id"]),
        "job_index": int(job.get("job_index", 0)),
        "handedness": sample["handedness"],
        "handedness_source": job.get("handedness_source"),
        "is_right": int(sample["handedness"] == "Right"),
        "bbox_rectified_px": job["bbox_rectified_px"],
        "bbox_scale": 1.0,
        "mobrecon_crop_scale": crop_scale,
        "source_detector": job.get("source_detector"),
        "rectified_image_path": job.get("rectified_image_path"),
        "hamer_frame_path": job.get("hamer_frame_path"),
        "used_mask_blur": bool(job.get("used_mask_blur")),
        "sam3_mask_path": job.get("sam3_mask_path"),
        "sam3_score": job.get("sam3_score"),
        "track_id": job.get("track_id"),
        "locked_handedness": job.get("locked_handedness"),
        "handedness_confidence": job.get("handedness_confidence"),
        "ambiguous_handedness": bool(job.get("ambiguous_handedness")),
        "handedness_hypothesis_id": job.get("handedness_hypothesis_id"),
        "hand_mesh_joints_cam": joints_m.tolist(),
        "hand_mesh_vertices_cam": mesh_m.tolist() if export_vertices else None,
        "hand_mesh_joints_2d_rectified_px": points_2d.tolist(),
        "hand_mesh_joints_2d_conf": [confidence] * 21,
        "hypothesis_status": (
            "online_dual_hypothesis"
            if job.get("ambiguous_handedness")
            else (
                "known"
                if str(job.get("handedness") or "").title() in {"Left", "Right"}
                else f"{unknown_handedness}_fallback"
            )
        ),
    }


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.job_batch_size < 1 or args.input_size < 2:
        raise SystemExit("batch sizes must be positive and --input-size must be at least 2")
    if args.crop_scale <= 0.0:
        raise SystemExit("--crop-scale must be positive")
    torch.set_num_threads(max(1, args.torch_threads))
    original_cwd = Path.cwd()
    group_ids = parse_group_ids(args.group_range, args.group_ids)
    suffix = range_suffix(group_ids)
    jobs_path = args.jobs or (DEFAULT_BASE_DIR / "hamer_jobs" / f"hamer_jobs_{suffix}.jsonl")
    jobs_path = jobs_path if jobs_path.is_absolute() else original_cwd / jobs_path
    args.output_dir = args.output_dir if args.output_dir.is_absolute() else original_cwd / args.output_dir
    args.mobrecon_root = args.mobrecon_root.expanduser().resolve()
    checkpoint = args.checkpoint or (args.mobrecon_root / "pretrained" / "mobrecon_densestack.pt")
    checkpoint = checkpoint.expanduser().resolve()
    camera_suffix = f"_{args.camera_id}" if args.camera_id else ""
    output_path = args.output_dir / f"mobrecon_predictions_{suffix}{camera_suffix}.jsonl"
    config_path = args.output_dir / f"mobrecon_predictions_config_{suffix}{camera_suffix}.json"
    jobs = iter_jobs(jobs_path, args.camera_id, args.include_debug_only, group_ids) if jobs_path.exists() else []
    spatial_assignments = spatial_handedness(jobs) if args.unknown_handedness == "spatial" else {}
    if args.dry_run:
        print(json.dumps({"jobs": len(jobs), "checkpoint": str(checkpoint), "output": str(output_path)}, indent=2))
        return
    if not jobs_path.exists():
        raise SystemExit(f"jobs not found: {jobs_path}")
    if not args.mobrecon_root.exists():
        raise SystemExit(f"MobRecon root not found: {args.mobrecon_root}")
    if not checkpoint.exists():
        raise SystemExit(f"MobRecon checkpoint not found: {checkpoint}")
    if output_path.exists() and not args.overwrite:
        raise SystemExit(f"{output_path} exists; pass --overwrite to replace it")

    device = choose_device(args.device)
    if args.precision == "float16" and device.type != "cuda":
        raise SystemExit("float16 MobRecon inference requires CUDA")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    load_started = time.perf_counter()
    model, joint_regressor, model_mib = load_model(args.mobrecon_root, checkpoint, device)
    model_load_seconds = time.perf_counter() - load_started
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    stats = {
        "jobs": len(jobs),
        "predictions": 0,
        "failed_jobs": 0,
        "unknown_jobs_skipped": 0,
        "model_batches": 0,
        "image_reads": 0,
        "image_cache_hits": 0,
    }
    timing = {"model_load_seconds": model_load_seconds, "preprocess_seconds": 0.0, "model_seconds": 0.0, "serialize_seconds": 0.0}
    inference_started = time.perf_counter()
    with output_path.open("w", encoding="utf-8") as output_file:
        progress = tqdm(total=len(jobs), desc="MobRecon", unit="job", position=args.progress_position)
        for job_start in range(0, len(jobs), args.job_batch_size):
            samples = []
            image_cache: dict[str, np.ndarray] = {}
            preprocess_started = time.perf_counter()
            for job in jobs[job_start : job_start + args.job_batch_size]:
                handedness = resolve_handedness(job, args.unknown_handedness, spatial_assignments)
                if handedness is None:
                    stats["unknown_jobs_skipped"] += 1
                    progress.update(1)
                    continue
                image_path = job["rectified_image_path"] if args.image_source == "rectified" else job["hamer_frame_path"]
                image = image_cache.get(str(image_path))
                if image is None:
                    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                    stats["image_reads"] += 1
                    if image is not None:
                        image_cache[str(image_path)] = image
                else:
                    stats["image_cache_hits"] += 1
                if image is None:
                    stats["failed_jobs"] += 1
                    progress.update(1)
                    continue
                samples.append(prepare_sample(image, job, handedness, args))
            timing["preprocess_seconds"] += time.perf_counter() - preprocess_started

            for batch_start in range(0, len(samples), args.batch_size):
                batch_samples = samples[batch_start : batch_start + args.batch_size]
                inputs = torch.stack([sample["input"] for sample in batch_samples]).to(device, non_blocking=True)
                model_started = time.perf_counter()
                autocast_context = (
                    torch.autocast(device_type="cuda", dtype=torch.float16)
                    if args.precision == "float16"
                    else nullcontext()
                )
                with torch.inference_mode(), autocast_context:
                    prediction = model(inputs)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                timing["model_seconds"] += time.perf_counter() - model_started
                stats["model_batches"] += 1
                meshes = prediction["mesh_pred"].detach().float().cpu().numpy()
                points = prediction["uv_pred"].detach().float().cpu().numpy()
                for sample_index, sample in enumerate(batch_samples):
                    joints_m, mesh_m, points_2d = restore_outputs(
                        meshes[sample_index], points[sample_index], sample, joint_regressor
                    )
                    job = sample["job"]
                    record = prediction_record(
                        job,
                        sample,
                        joints_m,
                        mesh_m,
                        points_2d,
                        args.export_vertices,
                        args.crop_scale,
                        args.unknown_handedness,
                    )
                    serialize_started = time.perf_counter()
                    output_file.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                    timing["serialize_seconds"] += time.perf_counter() - serialize_started
                    stats["predictions"] += 1
                    progress.update(1)
        progress.close()

    timing["inference_total_seconds"] = time.perf_counter() - inference_started
    timing["jobs_per_second"] = len(jobs) / timing["inference_total_seconds"] if timing["inference_total_seconds"] else None
    timing["predictions_per_second"] = stats["predictions"] / timing["inference_total_seconds"] if timing["inference_total_seconds"] else None
    peak_allocated_mib = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None
    peak_reserved_mib = torch.cuda.max_memory_reserved(device) / 2**20 if device.type == "cuda" else None
    with config_path.open("w", encoding="utf-8") as config_file:
        json.dump(
            {
                "jobs": str(jobs_path),
                "output": str(output_path),
                "mobrecon_root": str(args.mobrecon_root),
                "checkpoint": str(checkpoint),
                "device": str(device),
                "precision": args.precision,
                "image_source": args.image_source,
                "batch_size": args.batch_size,
                "job_batch_size": args.job_batch_size,
                "crop_scale": args.crop_scale,
                "input_size": args.input_size,
                "model_mib": model_mib,
                "peak_cuda_allocated_mib": peak_allocated_mib,
                "peak_cuda_reserved_mib": peak_reserved_mib,
                "stats": stats,
                "timing": timing,
            },
            config_file,
            ensure_ascii=False,
            indent=2,
        )
        config_file.write("\n")
    print(json.dumps({"stats": stats, "timing": timing, "peak_cuda_reserved_mib": peak_reserved_mib}, indent=2))
    print(f"wrote: {output_path}")


if __name__ == "__main__":
    main()
