"""Outline expansion: contrast-aware weight map plus an asymmetric
erode/dilate sequence that thickens dark and bright outlines before
downscaling. Direct port target: upstream `pixeloe.legacy.outline`.

Phase 2.5 scaffold: shape-correct no-ops. The real implementation comes
once `colorspace.rgb_to_lab` (2.3) and `sliding.apply_chunk` (2.4) exist.
"""
from __future__ import annotations

import numpy as np


def expansion_weight(
    rgb: np.ndarray,
    k: int = 8,
    stride: int = 2,
    avg_scale: int = 10,
    dist_scale: int = 3,
) -> np.ndarray:
    """Per-pixel weight in [0, 1] biasing erode (bright) vs dilate (dark)."""
    h, w = rgb.shape[:2]
    return np.full((h, w), 0.5, dtype=np.float32)


def outline_expansion(
    rgb: np.ndarray,
    erode: int = 2,
    dilate: int = 2,
    k: int = 16,
    avg_scale: int = 10,
    dist_scale: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Modified RGB plus the weight map for downstream color quantization."""
    h, w = rgb.shape[:2]
    weight = np.zeros((h, w), dtype=np.float32)
    return rgb.copy(), weight
