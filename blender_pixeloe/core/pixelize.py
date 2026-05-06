"""Top-level pixelize orchestrator.

Mirrors upstream `pixeloe.legacy.pixelize.pixelize`, but operates in RGB
throughout, takes uint8 numpy arrays at the boundary, and depends only on
numpy + Pillow + scipy. Each pipeline stage delegates to a sibling module;
this file owns the pipeline order and parameter surface, nothing else.

The brief cuts upstream's `contrast`, `saturation`, and `no_downscale` knobs
along with the bicubic and nearest downscale modes; only `contrast` and
`k-centroid` modes are reachable here.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from .color_match import match_color
from .downscale_contrast import contrast_based_downscale
from .downscale_kcentroid import k_centroid_downscale
from .outline import expansion_weight, outline_expansion
from .quantize import color_quant


def _resize(rgb: np.ndarray, target_hw: tuple[int, int], resample: int) -> np.ndarray:
    h, w = target_hw
    return np.array(Image.fromarray(rgb).resize((w, h), resample))


def pixelize(
    rgb: np.ndarray,
    *,
    mode: str = "contrast",
    target_size: int = 128,
    patch_size: int = 16,
    pixel_size: int | None = None,
    thickness: int = 2,
    color_matching: bool = True,
    colors: int = 0,
    colors_with_weight: bool = False,
    color_quant_method: str = "kmeans",
    no_upscale: bool = False,
) -> np.ndarray:
    """Pixelize an HxWx3 uint8 RGB array. See BRIEF.md for parameter semantics."""
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(
            f"expected HxWx3 uint8 RGB; got dtype={rgb.dtype} shape={rgb.shape}"
        )
    if mode not in ("contrast", "k-centroid"):
        raise ValueError(f"unknown mode: {mode!r}")

    if pixel_size is None:
        pixel_size = patch_size
    weighted_color = colors > 0 and colors_with_weight

    h, w = rgb.shape[:2]
    ratio = w / h
    org_size = (target_size**2 * patch_size**2 / ratio) ** 0.5
    org_hw = (int(org_size), int(org_size * ratio))
    rgb = _resize(rgb, org_hw, Image.BILINEAR)
    org_rgb = rgb.copy()

    weight: np.ndarray | None = None
    if thickness:
        rgb, weight = outline_expansion(
            rgb, erode=thickness, dilate=thickness, k=patch_size
        )
    elif weighted_color:
        w_map = expansion_weight(rgb, k=patch_size, stride=(patch_size // 4) * 2)
        weight = np.abs(w_map * 2 - 1)

    if color_matching:
        rgb = match_color(rgb, org_rgb)

    if mode == "contrast":
        rgb_sm = contrast_based_downscale(rgb, target_size)
    else:
        rgb_sm = k_centroid_downscale(rgb, target_size)

    if colors > 0:
        weight_mat: np.ndarray | None = None
        if weighted_color and weight is not None:
            sm_h, sm_w = rgb_sm.shape[:2]
            wmat = np.array(
                Image.fromarray(
                    (weight.clip(0, 1) * 255).astype(np.uint8)
                ).resize((sm_w, sm_h), Image.BILINEAR)
            ).astype(np.float32) / 255.0
            weight_mat = wmat ** (target_size / 512.0)
        repeats = max(1, int((patch_size * colors) ** 0.5))
        rgb_quant = color_quant(
            rgb_sm, colors, weight_mat, repeats, color_quant_method
        )
        rgb_sm = match_color(rgb_quant, rgb_sm, level=3)

    if no_upscale:
        return rgb_sm

    sm_h, sm_w = rgb_sm.shape[:2]
    return _resize(rgb_sm, (sm_h * pixel_size, sm_w * pixel_size), Image.NEAREST)
