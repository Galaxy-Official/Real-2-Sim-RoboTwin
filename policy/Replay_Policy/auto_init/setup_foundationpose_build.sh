#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash policy/Replay_Policy/auto_init/setup_foundationpose_build.sh [options]

Options:
  --foundationpose-root PATH   Path to third_party/FoundationPose.
  --max-jobs N                 Build parallelism. Default: 8.
  --skip-deps                  Do not install Conda/pip build dependencies.
  -h, --help                   Show this help.

Run this after activating the FoundationPose conda environment.
EOF
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
ROBOTWIN_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." >/dev/null 2>&1 && pwd)"
FOUNDATIONPOSE_ROOT="${ROBOTWIN_ROOT}/third_party/FoundationPose"
MAX_JOBS="${MAX_JOBS:-8}"
INSTALL_DEPS=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --foundationpose-root)
      FOUNDATIONPOSE_ROOT="$2"
      shift 2
      ;;
    --max-jobs)
      MAX_JOBS="$2"
      shift 2
      ;;
    --skip-deps)
      INSTALL_DEPS=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "CONDA_PREFIX is empty. Run: conda activate foundationpose" >&2
  exit 1
fi

if ! FOUNDATIONPOSE_ROOT="$(cd -- "${FOUNDATIONPOSE_ROOT}" >/dev/null 2>&1 && pwd)"; then
  echo "FoundationPose root does not exist: ${FOUNDATIONPOSE_ROOT}" >&2
  exit 1
fi

if [[ ! -d "${FOUNDATIONPOSE_ROOT}/mycpp" || ! -d "${FOUNDATIONPOSE_ROOT}/bundlesdf/mycuda" ]]; then
  echo "Invalid FoundationPose root: ${FOUNDATIONPOSE_ROOT}" >&2
  echo "Expected mycpp/ and bundlesdf/mycuda/ under that directory." >&2
  exit 1
fi

echo "[foundationpose-build] conda prefix: ${CONDA_PREFIX}"
echo "[foundationpose-build] FoundationPose root: ${FOUNDATIONPOSE_ROOT}"

python - <<'PY'
import sys
try:
    import torch
except Exception as exc:
    raise SystemExit(f"PyTorch is not importable in this environment: {exc}")
print("[foundationpose-build] python:", sys.executable)
print("[foundationpose-build] torch:", torch.__version__)
print("[foundationpose-build] torch cuda:", torch.version.cuda)
print("[foundationpose-build] cuda available:", torch.cuda.is_available())
PY

if [[ "${INSTALL_DEPS}" == "1" ]]; then
  conda install -c conda-forge \
    eigen=3.4.0 \
    "cmake<4" \
    make \
    pkg-config \
    boost-cpp \
    gcc_linux-64=11 \
    gxx_linux-64=11 \
    -y

  python -m pip install --upgrade pip
  python -m pip install "setuptools<70" wheel ninja pybind11 packaging
fi

export PATH="${CONDA_PREFIX}/bin:${PATH}"
hash -r

if [[ -x "${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-cc" ]]; then
  export CC="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-cc"
fi
if [[ -x "${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-c++" ]]; then
  export CXX="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-c++"
fi

echo "[foundationpose-build] cmake: $(command -v cmake)"
cmake --version
echo "[foundationpose-build] CC: ${CC:-unset}"
echo "[foundationpose-build] CXX: ${CXX:-unset}"

PYBIND11_CMAKE_DIR="$(
python - <<'PY'
import pybind11
print(pybind11.get_cmake_dir())
PY
)"

export CMAKE_PREFIX_PATH="${PYBIND11_CMAKE_DIR}:${CONDA_PREFIX}:${CMAKE_PREFIX_PATH:-}"
export PIP_NO_BUILD_ISOLATION=1
export CMAKE_BUILD_PARALLEL_LEVEL="${MAX_JOBS}"
export MAX_JOBS="${MAX_JOBS}"

echo "[foundationpose-build] Building mycpp"
cd "${FOUNDATIONPOSE_ROOT}/mycpp"
rm -rf build
mkdir -p build
cd build
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="${PYBIND11_CMAKE_DIR};${CONDA_PREFIX}" \
  -DBOOST_ROOT="${CONDA_PREFIX}" \
  -DBOOST_INCLUDEDIR="${CONDA_PREFIX}/include" \
  -DBOOST_LIBRARYDIR="${CONDA_PREFIX}/lib" \
  -DBoost_NO_SYSTEM_PATHS=ON
cmake --build . --parallel "${MAX_JOBS}"

echo "[foundationpose-build] Building bundlesdf/mycuda"
cd "${FOUNDATIONPOSE_ROOT}/bundlesdf/mycuda"
rm -rf build ./*.egg-info ./*.so
python -m pip install --no-build-isolation --no-use-pep517 -e .

echo "[foundationpose-build] Verifying imports"
cd "${FOUNDATIONPOSE_ROOT}"
python - <<'PY'
import torch
import nvdiffrast.torch as dr
from pytorch3d.transforms import so3_log_map
from estimater import FoundationPose, PoseRefinePredictor, ScorePredictor
import mycpp.build.mycpp as mycpp
from bundlesdf.mycuda import common
print("[foundationpose-build] torch:", torch.__version__, "cuda:", torch.version.cuda, "available:", torch.cuda.is_available())
print("[foundationpose-build] FoundationPose native extension imports ok")
PY

echo "[foundationpose-build] Done"
