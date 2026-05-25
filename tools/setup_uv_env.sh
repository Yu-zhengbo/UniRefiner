#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

PYTHON_BIN=${PYTHON_BIN:-python3}
VENV_DIR=${VENV_DIR:-$PROJECT_ROOT/.venv}
UV_CACHE_DIR=${UV_CACHE_DIR:-$PROJECT_ROOT/.cache/uv}
TORCH_INDEX_URL=${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}

export UV_CACHE_DIR
mkdir -p "$UV_CACHE_DIR"

if ! "$PYTHON_BIN" -m uv --version >/dev/null 2>&1; then
    "$PYTHON_BIN" -m pip install --user uv
fi

"$PYTHON_BIN" -m uv venv "$VENV_DIR" --python 3.10
"$PYTHON_BIN" -m uv pip install \
    --python "$VENV_DIR/bin/python" \
    --index-url "$TORCH_INDEX_URL" \
    torch==2.6.0 \
    torchvision==0.21.0
"$PYTHON_BIN" -m uv pip install \
    --python "$VENV_DIR/bin/python" \
    -e "${PROJECT_ROOT}[dev]"

"$VENV_DIR/bin/python" - <<'PY'
import torch
import transformers
import peft

print("torch:", torch.__version__, "cuda:", torch.version.cuda, "cuda_available:", torch.cuda.is_available())
print("transformers:", transformers.__version__)
print("peft:", peft.__version__)
PY
