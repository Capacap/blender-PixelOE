"""Outline expansion: contrast-aware blend between erosion and dilation.

Direct port of `pixeloe.legacy.outline.expansion_weight` and
`outline_expansion`. Two cv2-only ops are substituted:

    cv2.erode/dilate    -> scipy.ndimage.grey_erosion/grey_dilation looped
                          per channel, per iteration. cv2's defaults pad
                          erosion with the maximum value (255) and dilation
                          with 0 so border pixels aren't biased; we match
                          via `mode='constant', cval=255` / `cval=0`.
    cv2.resize INTER_LINEAR
                        -> Pillow `Image.resize(BILINEAR)` on a single-
                          channel float32 array (mode 'F'). Sub-pixel
                          sampling differs by a fraction of a level vs cv2;
                          the bilinear round-trip in `expansion_weight` is
                          a smoothing pass, so the divergence is well below
                          the BRIEF's 2/255 acceptance threshold.

`expansion_weight` reproduces upstream's normalization line verbatim,
including the `(out - min) / max` form (a likely typo upstream — the
intended denominator is `(max - min)`). Replicating it preserves output
parity rather than silently fixing a behaviour the rest of the pipeline
has been calibrated against.
"""
from __future__ import annotations

import numpy as np
from PIL import Image
from scipy.ndimage import grey_dilation, grey_erosion

from .colorspace import rgb_to_lab
from .sliding import apply_chunk

KERNEL_EXPANSION = np.ones((3, 3), dtype=np.uint8)
KERNEL_SMOOTHING = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _morph(
    img: np.ndarray,
    footprint: np.ndarray,
    iterations: int,
    *,
    op,
    cval: int,
) -> np.ndarray:
    if iterations <= 0:
        return img.copy()
    if img.ndim == 2:
        out = img
        for _ in range(iterations):
            out = op(out, footprint=footprint, mode="constant", cval=cval)
        return out
    out = np.empty_like(img)
    for c in range(img.shape[-1]):
        ch = img[..., c]
        for _ in range(iterations):
            ch = op(ch, footprint=footprint, mode="constant", cval=cval)
        out[..., c] = ch
    return out


def _erode(img: np.ndarray, footprint: np.ndarray, iterations: int) -> np.ndarray:
    return _morph(img, footprint, iterations, op=grey_erosion, cval=255)


def _dilate(img: np.ndarray, footprint: np.ndarray, iterations: int) -> np.ndarray:
    return _morph(img, footprint, iterations, op=grey_dilation, cval=0)


def _resize_bilinear_2d(arr: np.ndarray, target_wh: tuple[int, int]) -> np.ndarray:
    return np.array(
        Image.fromarray(arr.astype(np.float32, copy=False), mode="F").resize(
            target_wh, Image.BILINEAR
        )
    )


def expansion_weight(
    rgb: np.ndarray,
    k: int = 8,
    stride: int = 2,
    avg_scale: float = 10,
    dist_scale: float = 3,
) -> np.ndarray:
    """Per-pixel weight in [0, 1] biasing erode (bright) vs dilate (dark)."""
    h, w = rgb.shape[:2]
    L = rgb_to_lab(rgb)[..., 0].astype(np.float32) / 255.0

    avg_y = apply_chunk(L, k * 2, stride, lambda x: np.median(x, axis=1, keepdims=True))
    max_y = apply_chunk(L, k, stride, lambda x: np.max(x, axis=1, keepdims=True))
    min_y = apply_chunk(L, k, stride, lambda x: np.min(x, axis=1, keepdims=True))

    bright_dist = max_y - avg_y
    dark_dist = avg_y - min_y

    weight = (avg_y - 0.5) * avg_scale - (bright_dist - dark_dist) * dist_scale
    out = _sigmoid(weight).astype(np.float32)

    out = _resize_bilinear_2d(out, (w // stride, h // stride))
    out = _resize_bilinear_2d(out, (w, h))

    return ((out - out.min()) / out.max()).astype(np.float32)


def outline_expansion(
    rgb: np.ndarray,
    erode: int = 2,
    dilate: int = 2,
    k: int = 16,
    avg_scale: float = 10,
    dist_scale: float = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Modified RGB plus the weight map for downstream color quantization."""
    weight = expansion_weight(rgb, k, (k // 4) * 2, avg_scale, dist_scale)[..., None]
    orig_weight = _sigmoid((weight - 0.5) * 5) * 0.25

    img_erode = _erode(rgb, KERNEL_EXPANSION, erode).astype(np.float32)
    img_dilate = _dilate(rgb, KERNEL_EXPANSION, dilate).astype(np.float32)

    output = img_erode * weight + img_dilate * (1 - weight)
    output = output * (1 - orig_weight) + rgb.astype(np.float32) * orig_weight
    output = output.astype(np.uint8)

    output = _erode(output, KERNEL_SMOOTHING, erode)
    output = _dilate(output, KERNEL_SMOOTHING, dilate * 2)
    output = _erode(output, KERNEL_SMOOTHING, erode)

    weight_out = (np.abs(weight * 2 - 1) * 255)[..., 0].astype(np.uint8)
    weight_out = _dilate(weight_out, KERNEL_EXPANSION, dilate)
    return output, weight_out.astype(np.float32) / 255.0
