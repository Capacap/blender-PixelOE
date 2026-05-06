"""Contrast-based downscale: per-tile reduction in LAB space (find_pixel for
L, median for a/b) followed by nearest-neighbour pick at the target size.
Direct port target: upstream `pixeloe.legacy.downscale.contrast_based`.

Phase 2.6 scaffold: shape-correct nearest downsample, no LAB processing.
"""
from __future__ import annotations

import numpy as np
from PIL import Image


def contrast_based_downscale(rgb: np.ndarray, target_size: int = 128) -> np.ndarray:
    h, w = rgb.shape[:2]
    ratio = w / h
    eff = (target_size**2 / ratio) ** 0.5
    target_hw = (int(eff), int(eff * ratio))
    return np.array(
        Image.fromarray(rgb).resize((target_hw[1], target_hw[0]), Image.NEAREST)
    )
