#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash policy/Replay_Policy/auto_init/setup_foundationpose_weights.sh [options]

Options:
  --foundationpose-root PATH   Path to third_party/FoundationPose.
  --repo-id REPO_ID            Hugging Face repo id. Default: gpue/foundationpose-weights.
  --skip-install               Do not install/upgrade huggingface_hub.
  -h, --help                   Show this help.

The script downloads FoundationPose pretrained weights into:
  <foundationpose-root>/weights/2023-10-28-18-33-37
  <foundationpose-root>/weights/2024-01-11-20-02-45
EOF
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
ROBOTWIN_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." >/dev/null 2>&1 && pwd)"
FOUNDATIONPOSE_ROOT="${ROBOTWIN_ROOT}/third_party/FoundationPose"
REPO_ID="gpue/foundationpose-weights"
INSTALL_HF=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --foundationpose-root)
      FOUNDATIONPOSE_ROOT="$2"
      shift 2
      ;;
    --repo-id)
      REPO_ID="$2"
      shift 2
      ;;
    --skip-install)
      INSTALL_HF=0
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

if ! FOUNDATIONPOSE_ROOT="$(cd -- "${FOUNDATIONPOSE_ROOT}" >/dev/null 2>&1 && pwd)"; then
  echo "FoundationPose root does not exist: ${FOUNDATIONPOSE_ROOT}" >&2
  exit 1
fi

if [[ "${INSTALL_HF}" == "1" ]]; then
  python -m pip install --upgrade "huggingface_hub[hf_xet]"
fi

export FOUNDATIONPOSE_ROOT
export REPO_ID

python - <<'PY'
from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import snapshot_download

foundationpose_root = Path(os.environ["FOUNDATIONPOSE_ROOT"]).resolve()
repo_id = os.environ["REPO_ID"]
weights_dir = foundationpose_root / "weights"
weights_dir.mkdir(parents=True, exist_ok=True)

required = [
    "2023-10-28-18-33-37/config.yml",
    "2023-10-28-18-33-37/model_best.pth",
    "2024-01-11-20-02-45/config.yml",
    "2024-01-11-20-02-45/model_best.pth",
]

print(f"[foundationpose-weights] repo: {repo_id}")
print(f"[foundationpose-weights] target: {weights_dir}")

snapshot_download(
    repo_id=repo_id,
    repo_type="model",
    local_dir=str(weights_dir),
    allow_patterns=required,
)

missing = []
for rel in required:
    path = weights_dir / rel
    if not path.is_file():
        missing.append(f"{rel}: missing")
    elif path.suffix == ".pth" and path.stat().st_size < 1024 * 1024:
        missing.append(f"{rel}: too small ({path.stat().st_size} bytes), likely a Git LFS pointer")

if missing:
    raise SystemExit(
        "[foundationpose-weights] Download finished but required files are invalid:\n"
        + "\n".join(f"  - {item}" for item in missing)
    )

for rel in required:
    path = weights_dir / rel
    print(f"[foundationpose-weights] ok: {rel} ({path.stat().st_size} bytes)")

print("[foundationpose-weights] Done")
PY
