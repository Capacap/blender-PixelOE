"""Contrast-based downscale: per-tile reduction in LAB space (find_pixel
for L, median for a/b) followed by nearest-neighbour pick at the target
size. Direct port of `pixeloe.legacy.downscale.contrast_based`.

The reduction operates on non-overlapping `patch_size x patch_size` tiles,
so `apply_chunk` runs with stride==kernel and produces a tile-constant
output. Because every pixel within a tile carries the same value, the
final NEAREST resize down to target dimensions picks one representative
from each tile regardless of which sub-pixel sampling cv2 vs Pillow
prefer; the only divergence risk is when the input dimensions aren't
exact multiples of `patch_size`, where the boundary tile is partial.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from .colorspace import lab_to_rgb, rgb_to_lab
from .sliding import apply_chunk


def _find_pixel(chunks: np.ndarray) -> np.ndarray:
    """Per-window pick: middle element, nudged to min/max where the
    distribution skews dark/bright respectively.

    `chunks` is (N, K*K). The "middle" index `K*K // 2` is the upstream
    convention — it lands on row K/2 col 0 of the 2D window for square
    kernels, not the geometric centre. Replicated verbatim for parity.
    """
    K2 = chunks.shape[-1]
    mid = chunks[..., K2 // 2 : K2 // 2 + 1].copy()
    med = np.median(chunks, axis=1, keepdims=True)
    mu = np.mean(chunks, axis=1, keepdims=True)
    maxi = np.max(chunks, axis=1, keepdims=True)
    mini = np.min(chunks, axis=1, keepdims=True)

    mini_loc = (med < mu) & ((maxi - med) > (med - mini))
    maxi_loc = (med > mu) & ((maxi - med) < (med - mini))

    mid[mini_loc] = mini[mini_loc]
    mid[maxi_loc] = maxi[maxi_loc]
    return mid


def contrast_based_downscale(rgb: np.ndarray, target_size: int = 128) -> np.ndarray:
    h, w = rgb.shape[:2]
    ratio = w / h
    eff = (target_size**2 / ratio) ** 0.5
    target_w = int(eff * ratio)
    target_h = int(eff)
    patch_size = max(int(round(h / target_h)), int(round(w / target_w)))

    lab = rgb_to_lab(rgb).astype(np.float32)
    lab[..., 0] = apply_chunk(lab[..., 0], patch_size, patch_size, _find_pixel)
    lab[..., 1] = apply_chunk(
        lab[..., 1],
        patch_size,
        patch_size,
        lambda x: np.median(x, axis=1, keepdims=True),
    )
    lab[..., 2] = apply_chunk(
        lab[..., 2],
        patch_size,
        patch_size,
        lambda x: np.median(x, axis=1, keepdims=True),
    )
    rgb_full = lab_to_rgb(np.clip(lab, 0, 255).astype(np.uint8))

    return np.array(
        Image.fromarray(rgb_full).resize((target_w, target_h), Image.NEAREST)
    )
