#!/usr/bin/env python3
"""Check source dependencies, Conda environments, and model assets."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from dependency_paths import (
    DEFAULT_HAMER_ROOT,
    DEFAULT_MOBRECON_ROOT,
    DEFAULT_SAM3_ROOT,
    REPO_ROOT,
    WRIST_CAM_ROOT,
    default_conda_executable,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conda-bin", default=default_conda_executable())
    parser.add_argument("--headcam-env", default=os.environ.get("HEADCAM_ENV", "headcam"))
    parser.add_argument("--hamer-env", default=os.environ.get("HAMER_ENV", "hamer"))
    parser.add_argument("--mobrecon-env", default=os.environ.get("MOBRECON_ENV", os.environ.get("HAMER_ENV", "hamer")))
    parser.add_argument("--sam3-env", default=os.environ.get("SAM3_ENV", "sam3hand"))
    parser.add_argument(
        "--pipeline",
        choices=["all", "hamer", "mobrecon"],
        default="all",
        help="Limit model/source checks to one runnable pipeline.",
    )
    parser.add_argument("--skip-environments", action="store_true")
    parser.add_argument("--skip-models", action="store_true")
    args = parser.parse_args()

    checks: list[tuple[str, bool, str]] = []

    def record(name: str, ok: bool, detail: str) -> None:
        checks.append((name, ok, detail))

    record("wrist_cam submodule", (WRIST_CAM_ROOT / "README.md").is_file(), str(WRIST_CAM_ROOT))
    record("SAM3 source", (DEFAULT_SAM3_ROOT / "sam3" / "__init__.py").is_file(), str(DEFAULT_SAM3_ROOT))
    if args.pipeline in {"all", "hamer"}:
        record("HaMeR source", (DEFAULT_HAMER_ROOT / "hamer" / "__init__.py").is_file(), str(DEFAULT_HAMER_ROOT))
    if args.pipeline in {"all", "mobrecon"}:
        record("MobRecon source", (DEFAULT_MOBRECON_ROOT / "cmr" / "models" / "mobrecon_densestack.py").is_file(), str(DEFAULT_MOBRECON_ROOT))

    conda = args.conda_bin
    if not args.skip_environments:
        env_checks = [
            (args.headcam_env, "import cv2, mediapipe, numpy, scipy, yaml"),
            (args.sam3_env, "import cv2, sam3, torch"),
        ]
        if args.pipeline in {"all", "hamer"}:
            env_checks.append((args.hamer_env, "import cv2, hamer, torch"))
        if args.pipeline in {"all", "mobrecon"}:
            env_checks.append((args.mobrecon_env, "import cv2, numpy, openmesh, torch"))
        seen_environment_checks: set[tuple[str, str]] = set()
        for environment, expression in env_checks:
            if (environment, expression) in seen_environment_checks:
                continue
            seen_environment_checks.add((environment, expression))
            try:
                result = subprocess.run(
                    [conda, "run", "--no-capture-output", "-n", environment, "python", "-c", expression],
                    cwd=REPO_ROOT,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            except FileNotFoundError:
                record(f"Conda environment {environment}", False, f"Conda executable not found: {conda}")
                continue
            detail = "imports OK" if result.returncode == 0 else result.stdout.strip().splitlines()[-1] if result.stdout.strip() else "unavailable"
            record(f"Conda environment {environment}", result.returncode == 0, detail)

    if not args.skip_models:
        if args.pipeline in {"all", "hamer"}:
            hamer_data = DEFAULT_HAMER_ROOT / "_DATA"
            model_files = {
                "HaMeR checkpoint": hamer_data / "hamer_ckpts" / "checkpoints" / "hamer.ckpt",
                "HaMeR model config": hamer_data / "hamer_ckpts" / "model_config.yaml",
                "ViTPose checkpoint": hamer_data / "vitpose_ckpts" / "vitpose+_huge" / "wholebody.pth",
                "MANO_RIGHT.pkl": hamer_data / "data" / "mano" / "MANO_RIGHT.pkl",
            }
            for name, path in model_files.items():
                record(name, path.is_file() and path.stat().st_size > 0, str(path))
        if args.pipeline in {"all", "mobrecon"}:
            mobrecon_checkpoint = DEFAULT_MOBRECON_ROOT / "pretrained" / "mobrecon_densestack.pt"
            record("MobRecon checkpoint", mobrecon_checkpoint.is_file() and mobrecon_checkpoint.stat().st_size > 0, str(mobrecon_checkpoint))
        if not args.skip_environments:
            try:
                sam_check = subprocess.run(
                    [
                        conda,
                        "run",
                        "--no-capture-output",
                        "-n",
                        args.sam3_env,
                        "python",
                        str(REPO_ROOT / "scripts" / "download_sam3_models.py"),
                        "--local-only",
                    ],
                    cwd=REPO_ROOT,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            except FileNotFoundError:
                record("SAM3/SAM3.1 checkpoints", False, f"Conda executable not found: {conda}")
                sam_check = None
            if sam_check is None:
                pass
            else:
                detail = "cache OK" if sam_check.returncode == 0 else sam_check.stdout.strip().splitlines()[-1] if sam_check.stdout.strip() else "not cached"
                record("SAM3/SAM3.1 checkpoints", sam_check.returncode == 0, detail)

    for name, ok, detail in checks:
        print(f"[{'OK' if ok else 'FAIL'}] {name}: {detail}")
    failures = [name for name, ok, _ in checks if not ok]
    if failures:
        print("\nFix the failed checks with ./scripts/setup.sh", file=sys.stderr)
        raise SystemExit(1)
    print("\nAll requested checks passed.")


if __name__ == "__main__":
    main()
