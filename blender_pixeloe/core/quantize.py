"""Colour-palette quantization: PIL non-weighted path, seeded numpy weighted
path, and a maxcover wrapper. Reduces an RGB image to a small palette.
Direct port target: upstream `pixeloe.legacy.color.color_quant` and friends.

Phase 2.9 scaffold: returns input unchanged.
"""
from __future__ import annotations

import numpy as np


def color_quant(
    rgb: np.ndarray,
    colors: int,
    weights: np.ndarray | None = None,
    repeats: int = 64,
    method: str = "kmeans",
) -> np.ndarray:
    return rgb.copy()
