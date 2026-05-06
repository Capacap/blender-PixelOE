"""Wavelet color matching.

Direct port of `pixeloe.legacy.color.match_color`. Two passes:

1. Global LAB statistics match. Standardize the source's flattened LAB
   distribution (single scalar mean/std across all pixels and channels)
   then re-scale to the target's mean/std. Round-trips through cv2's
   uint8 LAB byte layout to stay byte-compatible with upstream.

2. Per-channel wavelet colorfix on the LAB-matched source. The source
   contributes high-frequency content (input minus its progressively
   blurred copies, summed across `level` octaves), the target supplies
   the lowest-frequency low-pass at the deepest level. Adding the two
   yields the source's detail with the target's hue/luma envelope.

cv2.GaussianBlur with sigma=0 becomes a manually-built 1D Gaussian
kernel (cv2's auto-sigma formula `0.3 * ((ksize - 1) * 0.5 - 1) + 0.8`)
applied separably via scipy.ndimage.convolve1d with `mode='mirror'`,
which matches cv2's default BORDER_REFLECT_101.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import convolve1d

from .colorspace import lab_to_rgb, rgb_to_lab


def _gaussian_kernel_1d(radius: int) -> np.ndarray:
    ksize = 2 * radius + 1
    sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
    x = np.arange(ksize, dtype=np.float64) - radius
    k = np.exp(-(x**2) / (2.0 * sigma**2))
    return (k / k.sum()).astype(np.float32)


def _wavelet_blur(inp: np.ndarray, radius: int) -> np.ndarray:
    k = _gaussian_kernel_1d(radius)
    out = convolve1d(inp, k, axis=0, mode="mirror")
    out = convolve1d(out, k, axis=1, mode="mirror")
    return out


def _wavelet_decomposition(inp: np.ndarray, levels: int) -> tuple[np.ndarray, np.ndarray]:
    high_freq = np.zeros_like(inp)
    low_freq = inp
    for i in range(1, levels + 1):
        radius = 2**i
        low_freq = _wavelet_blur(inp, radius)
        high_freq = high_freq + (inp - low_freq)
        inp = low_freq
    return high_freq, low_freq


def _wavelet_colorfix(inp: np.ndarray, target: np.ndarray, level: int) -> np.ndarray:
    inp_high, _ = _wavelet_decomposition(inp, level)
    _, target_low = _wavelet_decomposition(target, level)
    return inp_high + target_low


def match_color(
    source_rgb: np.ndarray, target_rgb: np.ndarray, level: int = 5
) -> np.ndarray:
    src_lab = rgb_to_lab(source_rgb).astype(np.float32) / 255.0
    tgt_lab = rgb_to_lab(target_rgb).astype(np.float32) / 255.0

    standardized = (src_lab - src_lab.mean()) / src_lab.std()
    matched = standardized * tgt_lab.std() + tgt_lab.mean()
    src_rgb_matched = lab_to_rgb(np.clip(matched * 255.0, 0, 255).astype(np.uint8))

    src_f = src_rgb_matched.astype(np.float32)
    tgt_f = target_rgb.astype(np.float32)
    out = np.empty_like(src_f)
    for c in range(3):
        out[..., c] = _wavelet_colorfix(src_f[..., c], tgt_f[..., c], level)
    return np.clip(out, 0, 255).astype(np.uint8)
