"""Top-level pixelize orchestrator.

Phase 1 stub: returns the input unchanged so the comparison harness produces a
visible diff against upstream. Replaced by the real implementation in Phase 2h.
"""
from __future__ import annotations

import numpy as np


def pixelize(
    img: np.ndarray,
    target_size: int = 256,
    patch_size: int = 8,
    thickness: int = 3,
    mode: str = "contrast",
    colors: int = 0,
) -> np.ndarray:
    return img.copy()
