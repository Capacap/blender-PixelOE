#!/usr/bin/env bash
# Clone upstream PixelOE pinned to the commit this port targets.
# Used for: reading reference source, generating reference outputs in the harness.
set -euo pipefail

UPSTREAM_REPO="https://github.com/KohakuBlueleaf/PixelOE.git"
UPSTREAM_COMMIT="f7c0ae1"
UPSTREAM_DIR="tests/upstream"

cd "$(dirname "$0")"

if [ ! -d "$UPSTREAM_DIR/.git" ]; then
    git clone "$UPSTREAM_REPO" "$UPSTREAM_DIR"
fi

git -C "$UPSTREAM_DIR" fetch --quiet
git -C "$UPSTREAM_DIR" -c advice.detachedHead=false checkout "$UPSTREAM_COMMIT"

echo "Upstream pinned at $UPSTREAM_COMMIT in $UPSTREAM_DIR"
