"""K-centroid downscale: per-tile PIL.Image.quantize plus most-common-colour
selection. Algorithm originally from Astropulse's pixeldetector (MIT).
Direct port target: upstream `pixeloe.legacy.downscale.k_centroid`.

Phase 2.7 scaffold: shape-correct nearest downsample, no quantization.
"""
from __future__ import annotations

import numpy as np
from PIL import Image


def k_centroid_downscale(
    rgb: np.ndarray, target_size: int = 128, centroids: int = 2
) -> np.ndarray:
    h, w = rgb.shape[:2]
    ratio = w / h
    eff = (target_size**2 / ratio) ** 0.5
    target_hw = (int(eff), int(eff * ratio))
    return np.array(
        Image.fromarray(rgb).resize((target_hw[1], target_hw[0]), Image.NEAREST)
    )
