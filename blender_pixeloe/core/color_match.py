"""Wavelet color matching: replaces a source image's low-frequency colour
content with that of a target image, leaving high-frequency detail intact.
Direct port target: upstream `pixeloe.legacy.color.match_color`.

Phase 2.8 scaffold: returns source unchanged.
"""
from __future__ import annotations

import numpy as np


def match_color(
    source_rgb: np.ndarray, target_rgb: np.ndarray, level: int = 5
) -> np.ndarray:
    return source_rgb.copy()
