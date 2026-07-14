#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCAL_CONDA="${ROOT_DIR}/.tools/conda/bin/conda"
CONDA_BIN="${CONDA_BIN:-${CONDA_EXE:-}}"
if [[ -z "${CONDA_BIN}" ]]; then
  if [[ -x "${LOCAL_CONDA}" ]]; then
    CONDA_BIN="${LOCAL_CONDA}"
  else
    CONDA_BIN="$(command -v conda || true)"
  fi
fi
if [[ -z "${CONDA_BIN}" || ! -x "${CONDA_BIN}" ]]; then
  echo "Conda is unavailable. Run ./scripts/setup.sh first." >&2
  exit 1
fi

HEADCAM_ENV="${HEADCAM_ENV:-headcam}"
HAMER_ENV="${HAMER_ENV:-hamer}"
SAM3_ENV="${SAM3_ENV:-sam3hand}"
cd "${ROOT_DIR}"

"${CONDA_BIN}" run --no-capture-output -n "${HEADCAM_ENV}" \
  python "${ROOT_DIR}/scripts/doctor.py" \
  --conda-bin "${CONDA_BIN}" \
  --headcam-env "${HEADCAM_ENV}" \
  --hamer-env "${HAMER_ENV}" \
  --sam3-env "${SAM3_ENV}"

exec "${CONDA_BIN}" run --no-capture-output -n "${HEADCAM_ENV}" \
  python -u -s "${ROOT_DIR}/scripts/run_hamer_multiview_pipeline.py" \
  --conda-bin "${CONDA_BIN}" \
  --hamer-conda-env "${HAMER_ENV}" \
  --sam3-conda-env "${SAM3_ENV}" \
  "$@"
