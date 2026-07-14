#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOLS_DIR="${ROOT_DIR}/.tools"
LOCAL_CONDA="${TOOLS_DIR}/conda/bin/conda"
HEADCAM_ENV="${HEADCAM_ENV:-headcam}"
HAMER_ENV="${HAMER_ENV:-hamer}"
SAM3_ENV="${SAM3_ENV:-sam3hand}"
HAMER_TORCH_INDEX_URL="${HAMER_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu117}"
SAM3_TORCH_INDEX_URL="${SAM3_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
SKIP_MODELS=0
CHECK_ONLY=0

usage() {
  cat <<'EOF'
Usage: ./scripts/setup.sh [--skip-models] [--check-only]

Environment overrides:
  CONDA_BIN, HEADCAM_ENV, HAMER_ENV, SAM3_ENV
  HAMER_ROOT, SAM3_ROOT, MOBRECON_ROOT
  MANO_MODEL_PATH       Path to a licensed MANO_RIGHT.pkl
  HF_TOKEN              Hugging Face token with SAM3/SAM3.1 access
  HF_ENDPOINT           Optional Hugging Face mirror endpoint
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-models) SKIP_MODELS=1 ;;
    --check-only) CHECK_ONLY=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

find_conda() {
  if [[ -n "${CONDA_BIN:-}" && -x "${CONDA_BIN}" ]]; then
    printf '%s\n' "${CONDA_BIN}"
  elif [[ -n "${CONDA_EXE:-}" && -x "${CONDA_EXE}" ]]; then
    printf '%s\n' "${CONDA_EXE}"
  elif [[ -x "${LOCAL_CONDA}" ]]; then
    printf '%s\n' "${LOCAL_CONDA}"
  else
    command -v conda || true
  fi
}

install_conda() {
  local os arch platform url installer
  os="$(uname -s)"
  arch="$(uname -m)"
  case "${os}-${arch}" in
    Linux-x86_64) platform="Linux-x86_64" ;;
    *)
      echo "Automatic GPU environment setup currently supports Linux x86_64/aarch64; found ${os}-${arch}." >&2
      exit 1
      ;;
  esac
  mkdir -p "${TOOLS_DIR}"
  installer="${TOOLS_DIR}/miniforge.sh"
  url="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-${platform}.sh"
  echo "[setup] Conda not found; installing Miniforge under ${TOOLS_DIR}/conda"
  if command -v curl >/dev/null 2>&1; then
    curl -fL "${url}" -o "${installer}"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "${installer}" "${url}"
  else
    echo "curl or wget is required to bootstrap Miniforge." >&2
    exit 1
  fi
  bash "${installer}" -b -p "${TOOLS_DIR}/conda"
}

CONDA_BIN="$(find_conda)"
if [[ -z "${CONDA_BIN}" ]]; then
  if [[ "${CHECK_ONLY}" -eq 1 ]]; then
    echo "Conda is unavailable. Run ./scripts/setup.sh without --check-only." >&2
    exit 1
  fi
  install_conda
  CONDA_BIN="${LOCAL_CONDA}"
fi

if [[ "${CHECK_ONLY}" -eq 1 ]]; then
  exec "${CONDA_BIN}" run --no-capture-output -n "${HEADCAM_ENV}" \
    python "${ROOT_DIR}/scripts/doctor.py" \
    --conda-bin "${CONDA_BIN}" \
    --headcam-env "${HEADCAM_ENV}" \
    --hamer-env "${HAMER_ENV}" \
    --sam3-env "${SAM3_ENV}"
fi

echo "[setup] Initializing pinned Git dependencies"
git -C "${ROOT_DIR}" submodule sync --recursive
git -C "${ROOT_DIR}" submodule update --init --recursive

HAMER_ROOT="${HAMER_ROOT:-${ROOT_DIR}/external/wrist_cam/third_party/hamer}"
SAM3_ROOT="${SAM3_ROOT:-${ROOT_DIR}/external/wrist_cam/third_party/sam3}"
MOBRECON_ROOT="${MOBRECON_ROOT:-${ROOT_DIR}/external/HandMesh}"
if [[ ! -f "${HAMER_ROOT}/setup.py" || ! -f "${SAM3_ROOT}/pyproject.toml" || ! -f "${MOBRECON_ROOT}/cmr/models/mobrecon_densestack.py" ]]; then
  echo "Pinned HaMeR/SAM3/MobRecon source is missing after submodule initialization." >&2
  exit 1
fi

echo "[setup] Creating/updating ${HEADCAM_ENV}, ${HAMER_ENV}, and ${SAM3_ENV}"
export CONDA_ALWAYS_YES=true
"${CONDA_BIN}" env update -n "${HEADCAM_ENV}" -f "${ROOT_DIR}/environments/headcam.yml"
"${CONDA_BIN}" env update -n "${HAMER_ENV}" -f "${ROOT_DIR}/environments/hamer.yml"
"${CONDA_BIN}" env update -n "${SAM3_ENV}" -f "${ROOT_DIR}/environments/sam3.yml"

HEAD_RUN=("${CONDA_BIN}" run --no-capture-output -n "${HEADCAM_ENV}")
HAMER_RUN=("${CONDA_BIN}" run --no-capture-output -n "${HAMER_ENV}")
SAM3_RUN=("${CONDA_BIN}" run --no-capture-output -n "${SAM3_ENV}")

echo "[setup] Installing HaMeR runtime"
"${HAMER_RUN[@]}" python -m pip install --upgrade pip wheel "setuptools<70"
"${HAMER_RUN[@]}" python -m pip install torch==2.0.1 torchvision==0.15.2 --index-url "${HAMER_TORCH_INDEX_URL}"
"${HAMER_RUN[@]}" python -m pip install ninja "numpy<2" "opencv-python==4.10.0.84" tqdm
if ! "${HAMER_RUN[@]}" python -c "import detectron2" >/dev/null 2>&1; then
  env \
    CPATH=/usr/include:/usr/include/x86_64-linux-gnu \
    LIBRARY_PATH=/usr/lib/x86_64-linux-gnu \
    LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu \
    LDFLAGS=-L/usr/lib/x86_64-linux-gnu \
    "${HAMER_RUN[@]}" python -m pip install --no-build-isolation \
    "detectron2 @ git+https://github.com/facebookresearch/detectron2@v0.6"
fi
"${HAMER_RUN[@]}" python -m pip install --no-build-isolation \
  "chumpy @ git+https://github.com/mattloper/chumpy" mmcv==1.3.9
"${HAMER_RUN[@]}" python -m pip install \
  gdown pyrender pytorch-lightning==2.0.9 scikit-image smplx==0.1.28 \
  timm einops xtcocotools pandas hydra-submitit-launcher hydra-colorlog \
  pyrootutils rich webdataset json_tricks munkres
"${HAMER_RUN[@]}" python -m pip install openmesh yacs
"${HAMER_RUN[@]}" python -m pip install "numpy<2" "opencv-python==4.10.0.84"
"${HAMER_RUN[@]}" python -m pip install -e "${HAMER_ROOT}" --no-deps
"${HAMER_RUN[@]}" python -m pip install -v -e "${HAMER_ROOT}/third-party/ViTPose" --no-deps

echo "[setup] Installing SAM3 runtime"
"${SAM3_RUN[@]}" python -m pip install --upgrade pip "setuptools<81"
"${SAM3_RUN[@]}" python -m pip install torch==2.10.0 torchvision --index-url "${SAM3_TORCH_INDEX_URL}"
"${SAM3_RUN[@]}" python -m pip install -e "${SAM3_ROOT}"
"${SAM3_RUN[@]}" python -m pip install "numpy<2" "opencv-python<4.13" pillow tqdm einops pycocotools psutil

if [[ "${SKIP_MODELS}" -eq 0 ]]; then
  HAMER_CHECKPOINT="${HAMER_ROOT}/_DATA/hamer_ckpts/checkpoints/hamer.ckpt"
  VITPOSE_CHECKPOINT="${HAMER_ROOT}/_DATA/vitpose_ckpts/vitpose+_huge/wholebody.pth"
  if [[ ! -s "${HAMER_CHECKPOINT}" || ! -s "${VITPOSE_CHECKPOINT}" ]]; then
    echo "[setup] Downloading public HaMeR and ViTPose model data"
    (
      cd "${HAMER_ROOT}"
      "${HAMER_RUN[@]}" bash fetch_demo_data.sh
      if [[ -f hamer_demo_data.tar.gz ]]; then
        mkdir -p _DATA
        mv -f hamer_demo_data.tar.gz _DATA/hamer_demo_data.tar.gz
      fi
    )
  fi

  MOBRECON_CHECKPOINT="${MOBRECON_ROOT}/pretrained/mobrecon_densestack.pt"
  if [[ ! -s "${MOBRECON_CHECKPOINT}" ]]; then
    echo "[setup] Downloading the MobRecon checkpoint"
    mkdir -p "$(dirname "${MOBRECON_CHECKPOINT}")"
    "${HAMER_RUN[@]}" gdown 1QKtt5x-8Xe_afjpMTBIk2TI3G5QGk_iu -O "${MOBRECON_CHECKPOINT}"
  fi

  MANO_TARGET="${HAMER_ROOT}/_DATA/data/mano/MANO_RIGHT.pkl"
  if [[ -n "${MANO_MODEL_PATH:-}" ]]; then
    if [[ ! -s "${MANO_MODEL_PATH}" ]]; then
      echo "MANO_MODEL_PATH does not point to a readable file: ${MANO_MODEL_PATH}" >&2
      exit 1
    fi
    mkdir -p "$(dirname "${MANO_TARGET}")"
    install -m 0644 "${MANO_MODEL_PATH}" "${MANO_TARGET}"
  fi

  echo "[setup] Downloading gated SAM3 and SAM3.1 checkpoints into the Hugging Face cache"
  if ! "${SAM3_RUN[@]}" python "${ROOT_DIR}/scripts/download_sam3_models.py"; then
    cat >&2 <<'EOF'
SAM3 download failed. Accept access for facebook/sam3 and facebook/sam3.1,
then export HF_TOKEN or run `hf auth login` in the sam3hand environment and retry.
Use --skip-models only when checkpoints are intentionally managed elsewhere.
EOF
    exit 1
  fi

  if [[ ! -s "${MANO_TARGET}" ]]; then
    cat >&2 <<EOF

Environment and public weights are installed, but MANO_RIGHT.pkl cannot legally be
redistributed automatically. Register and download MANO from:
  https://mano.is.tue.mpg.de/

Then complete setup in one command:
  MANO_MODEL_PATH=/path/to/MANO_RIGHT.pkl ./scripts/setup.sh
EOF
    exit 2
  fi
fi

DOCTOR_ARGS=(
  --conda-bin "${CONDA_BIN}"
  --headcam-env "${HEADCAM_ENV}"
  --hamer-env "${HAMER_ENV}"
  --sam3-env "${SAM3_ENV}"
)
if [[ "${SKIP_MODELS}" -eq 1 ]]; then
  DOCTOR_ARGS+=(--skip-models)
fi

"${HEAD_RUN[@]}" python "${ROOT_DIR}/scripts/doctor.py" "${DOCTOR_ARGS[@]}"
cat <<EOF

Setup complete.
Run the pipeline with:
  ./scripts/run.sh --base-dir video/<dataset> --group-range 0-100
EOF
