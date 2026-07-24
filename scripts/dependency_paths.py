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


def _default_mobrecon_root() -> Path:
    value = os.environ.get("MOBRECON_ROOT")
    if value:
        return Path(value).expanduser().resolve()

    candidates = [
        REPO_ROOT / "external" / "HandMesh",
        REPO_ROOT / "third_party" / "HandMesh",
    ]
    source = Path("cmr/models/mobrecon_densestack.py")
    checkpoint = Path("pretrained/mobrecon_densestack.pt")
    for candidate in candidates:
        if (candidate / source).is_file() and (candidate / checkpoint).is_file():
            return candidate.resolve()
    for candidate in candidates:
        if (candidate / source).is_file():
            return candidate.resolve()
    return candidates[0].resolve()


WRIST_CAM_ROOT = _path_from_env("WRIST_CAM_ROOT", REPO_ROOT / "external" / "wrist_cam")
WRIST_SCRIPTS = WRIST_CAM_ROOT / "scripts"
DEFAULT_HAMER_ROOT = _path_from_env("HAMER_ROOT", WRIST_CAM_ROOT / "third_party" / "hamer")
DEFAULT_SAM3_ROOT = _path_from_env("SAM3_ROOT", WRIST_CAM_ROOT / "third_party" / "sam3")
DEFAULT_MOBRECON_ROOT = _default_mobrecon_root()


def default_conda_executable() -> str:
    """Prefer the active Conda executable without assuming a user home path."""
    return os.environ.get("CONDA_EXE") or shutil.which("conda") or "conda"
