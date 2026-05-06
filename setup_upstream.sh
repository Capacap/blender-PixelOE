#!/usr/bin/env bash
# Clone upstream PixelOE pinned to the commit this port targets, and build
# an isolated venv for the harness to subprocess against. The upstream venv
# pulls torch/torchvision/cv2/kornia, which is exactly the dependency surface
# we are porting away from. Keeping it isolated from the project venv is the
# point of this whole setup.
set -euo pipefail

UPSTREAM_REPO="https://github.com/KohakuBlueleaf/PixelOE.git"
UPSTREAM_COMMIT="f7c0ae1"
UPSTREAM_DIR="tests/upstream"
UPSTREAM_VENV="$UPSTREAM_DIR/.venv"

cd "$(dirname "$0")"

if [ ! -d "$UPSTREAM_DIR/.git" ]; then
    git clone "$UPSTREAM_REPO" "$UPSTREAM_DIR"
fi

git -C "$UPSTREAM_DIR" fetch --quiet
git -C "$UPSTREAM_DIR" -c advice.detachedHead=false checkout "$UPSTREAM_COMMIT"

echo "Upstream pinned at $UPSTREAM_COMMIT in $UPSTREAM_DIR"

if [ ! -d "$UPSTREAM_VENV" ]; then
    echo "Creating upstream venv at $UPSTREAM_VENV (Python 3.11)"
    uv venv "$UPSTREAM_VENV" --python 3.11
fi

UPSTREAM_PY="$UPSTREAM_VENV/bin/python"

# --torch-backend cpu picks the CPU-only torch wheel (small) instead of the
# default CUDA build. The harness only needs upstream to run, not to run fast.
echo "Installing upstream (editable) with CPU-only torch into $UPSTREAM_VENV"
uv pip install \
    --python "$UPSTREAM_PY" \
    --torch-backend cpu \
    -e "$UPSTREAM_DIR"

echo "Upstream venv ready: $UPSTREAM_PY"
