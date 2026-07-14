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
MOBRECON_ENV="${MOBRECON_ENV:-${HAMER_ENV}}"
SAM3_ENV="${SAM3_ENV:-sam3hand}"
cd "${ROOT_DIR}"

PIPELINE="${HEADCAM_PIPELINE:-hamer}"
FORWARD_ARGS=()
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --pipeline)
      if [[ "$#" -lt 2 ]]; then
        echo "--pipeline requires hamer or mobrecon" >&2
        exit 2
      fi
      PIPELINE="$2"
      shift 2
      ;;
    --pipeline=*)
      PIPELINE="${1#*=}"
      shift
      ;;
    --list-pipelines)
      printf '%s\n' 'hamer' 'mobrecon'
      exit 0
      ;;
    *)
      FORWARD_ARGS+=("$1")
      shift
      ;;
  esac
done
if [[ "${PIPELINE}" != "hamer" && "${PIPELINE}" != "mobrecon" ]]; then
  echo "Unknown pipeline: ${PIPELINE}. Expected hamer or mobrecon." >&2
  exit 2
fi

DOCTOR_EXTRA=()
SHOW_SELECTOR_HELP=0
SKIP_MODEL_CHECKS=0
for argument in "${FORWARD_ARGS[@]}"; do
  if [[ "${argument}" == "--dry-run" || "${argument}" == "-h" || "${argument}" == "--help" ]]; then
    SKIP_MODEL_CHECKS=1
  fi
  if [[ "${argument}" == "-h" || "${argument}" == "--help" ]]; then
    SHOW_SELECTOR_HELP=1
  fi
done
if [[ "${SKIP_MODEL_CHECKS}" -eq 1 ]]; then
  DOCTOR_EXTRA+=(--skip-models)
fi

if [[ "${SHOW_SELECTOR_HELP}" -eq 1 ]]; then
  cat <<EOF
HeadCam pipeline selector:
  --pipeline hamer|mobrecon  Select the full pipeline (default: hamer).
  --list-pipelines           List available pipeline names.
  HEADCAM_PIPELINE           Environment default for --pipeline.

Selected pipeline: ${PIPELINE}
Backend-specific options follow.

EOF
fi

"${CONDA_BIN}" run --no-capture-output -n "${HEADCAM_ENV}" \
  python "${ROOT_DIR}/scripts/doctor.py" \
  --conda-bin "${CONDA_BIN}" \
  --headcam-env "${HEADCAM_ENV}" \
  --hamer-env "${HAMER_ENV}" \
  --mobrecon-env "${MOBRECON_ENV}" \
  --sam3-env "${SAM3_ENV}" \
  --pipeline "${PIPELINE}" \
  "${DOCTOR_EXTRA[@]}"

if [[ "${PIPELINE}" == "hamer" ]]; then
  exec "${CONDA_BIN}" run --no-capture-output -n "${HEADCAM_ENV}" \
    python -u -s "${ROOT_DIR}/scripts/run_hamer_multiview_pipeline.py" \
    --conda-bin "${CONDA_BIN}" \
    --hamer-conda-env "${HAMER_ENV}" \
    --sam3-conda-env "${SAM3_ENV}" \
    "${FORWARD_ARGS[@]}"
fi

exec "${CONDA_BIN}" run --no-capture-output -n "${HEADCAM_ENV}" \
  python -u -s "${ROOT_DIR}/scripts/run_sparse_sam3_mobrecon.py" \
  --conda-bin "${CONDA_BIN}" \
  --mobrecon-conda-env "${MOBRECON_ENV}" \
  --sam3-conda-env "${SAM3_ENV}" \
  "${FORWARD_ARGS[@]}"
