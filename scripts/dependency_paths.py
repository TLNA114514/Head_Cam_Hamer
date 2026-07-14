#!/usr/bin/env python3
"""Portable locations for source dependencies and external executables."""

from __future__ import annotations

import os
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _path_from_env(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else default.resolve()


WRIST_CAM_ROOT = _path_from_env("WRIST_CAM_ROOT", REPO_ROOT / "external" / "wrist_cam")
WRIST_SCRIPTS = WRIST_CAM_ROOT / "scripts"
DEFAULT_HAMER_ROOT = _path_from_env("HAMER_ROOT", WRIST_CAM_ROOT / "third_party" / "hamer")
DEFAULT_SAM3_ROOT = _path_from_env("SAM3_ROOT", WRIST_CAM_ROOT / "third_party" / "sam3")
DEFAULT_MOBRECON_ROOT = _path_from_env("MOBRECON_ROOT", REPO_ROOT / "external" / "HandMesh")


def default_conda_executable() -> str:
    """Prefer the active Conda executable without assuming a user home path."""
    return os.environ.get("CONDA_EXE") or shutil.which("conda") or "conda"
