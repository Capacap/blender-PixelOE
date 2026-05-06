"""Contrast-based downscale: per-tile reduction in LAB space (find_pixel for
L, median for a/b) producing the target-resolution RGB output directly.

Functional port target: `pixeloe.legacy.downscale.contrast_based`. Upstream
ran apply_chunk with stride == kernel == patch_size, np.repeat'd the
per-tile scalar across the original grid, converted full-res LAB back to
full-res RGB, and finished with a NEAREST resize down to (target_w,
target_h). Every step after the per-tile reduction is degenerate when the
output is already tile-constant: the np.repeat fans one value into a
patch_size**2 block, lab_to_rgb runs on patch_size**2 redundant copies of
each value, and NEAREST sampling picks one back out. The whole tail
collapses to "compute reductions, convert small lab to small rgb, return
it" — which is what this rewrite does.

Reductions run via reshape-as-tiles + numpy axis reductions on the
(target_h, target_w, patch_size**2) view, no sliding window needed since
tiles are non-overlapping. lab_to_rgb runs on the (target_h, target_w, 3)
small image, which is target_size**2 / org_size**2 cheaper than upstream
(64x at the standard 8x downscale).

Boundary handling: if the input dimensions aren't multiples of patch_size,
target_h and target_w are floor-clamped and the trailing partial row/col
of input is dropped. The pipeline always feeds patch-divisible inputs by
construction in pixelize.py, so this only affects degenerate calls.
"""
from __future__ import annotations

import numpy as np

from .colorspace import lab_to_rgb, rgb_to_lab


def _find_pixel(tiles: np.ndarray) -> np.ndarray:
    """Per-tile pick: middle element nudged to min/max where the
    distribution skews dark/bright respectively.

    `tiles` is (..., K**2). The "middle" index `K**2 // 2` is the upstream
    convention — it lands on row K/2 col 0 of the 2D tile in C order, not
    the geometric centre. Replicated verbatim for parity.
    """
    K2 = tiles.shape[-1]
    mid = tiles[..., K2 // 2]
    med = np.median(tiles, axis=-1)
    mu = np.mean(tiles, axis=-1)
    maxi = np.max(tiles, axis=-1)
    mini = np.min(tiles, axis=-1)

    mini_loc = (med < mu) & ((maxi - med) > (med - mini))
    maxi_loc = (med > mu) & ((maxi - med) < (med - mini))

    out = np.where(mini_loc, mini, mid)
    out = np.where(maxi_loc, maxi, out)
    return out


def contrast_based_downscale(rgb: np.ndarray, target_size: int = 128) -> np.ndarray:
    h, w = rgb.shape[:2]
    ratio = w / h
    eff = (target_size**2 / ratio) ** 0.5
    target_w = int(eff * ratio)
    target_h = int(eff)
    patch_size = max(int(round(h / target_h)), int(round(w / target_w)))

    target_h = min(target_h, h // patch_size)
    target_w = min(target_w, w // patch_size)
    hs = target_h * patch_size
    ws = target_w * patch_size

    lab = rgb_to_lab(rgb)[:hs, :ws].astype(np.float32)
    tiles = (
        lab.reshape(target_h, patch_size, target_w, patch_size, 3)
        .transpose(0, 2, 1, 3, 4)
        .reshape(target_h, target_w, patch_size * patch_size, 3)
    )

    L = _find_pixel(tiles[..., 0])
    a = np.median(tiles[..., 1], axis=-1)
    b = np.median(tiles[..., 2], axis=-1)

    lab_small = np.stack([L, a, b], axis=-1)
    return lab_to_rgb(np.clip(lab_small, 0, 255).astype(np.uint8))
