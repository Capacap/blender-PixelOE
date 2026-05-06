"""K-centroid downscale: per-tile PIL quantize plus most-common-colour
pick. Algorithm originally from Astropulse's pixeldetector (MIT).
Direct port of `pixeloe.legacy.downscale.k_centroid`.

Upstream is already a PIL-only routine wrapped in BGR/RGB conversions
for cv2 interop. The port is a near-identity translation: we take and
return RGB uint8, so the conversion wrappers vanish. Crop coordinates
remain float (PIL handles the rounding internally) to match upstream
byte-for-byte.
"""
from __future__ import annotations

from itertools import product

import numpy as np
from PIL import Image


def k_centroid_downscale(
    rgb: np.ndarray, target_size: int = 128, centroids: int = 2
) -> np.ndarray:
    h, w = rgb.shape[:2]
    ratio = w / h
    eff = (target_size**2 / ratio) ** 0.5
    target_h = int(eff)
    target_w = int(eff * ratio)

    image = Image.fromarray(rgb)
    w_factor = image.width / target_w
    h_factor = image.height / target_h

    out = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    for x, y in product(range(target_w), range(target_h)):
        tile = image.crop(
            (
                x * w_factor,
                y * h_factor,
                x * w_factor + w_factor,
                y * h_factor + h_factor,
            )
        )
        tile = tile.quantize(
            colors=centroids, method=Image.Quantize.MAXCOVERAGE, kmeans=centroids
        ).convert("RGB")
        most_common = max(tile.getcolors(), key=lambda c: c[0])[1]
        out[y, x] = most_common
    return out
