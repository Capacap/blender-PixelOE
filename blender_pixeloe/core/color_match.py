"""Wavelet color matching.

Functional port target: `pixeloe.legacy.color.match_color`. Two passes:

1. Global LAB statistics match. Standardize the source's flattened LAB
   distribution (single scalar mean/std across all pixels and channels)
   then re-scale to the target's mean/std. Round-trips through cv2's
   uint8 LAB byte layout to stay byte-compatible with upstream.

2. Per-channel wavelet colorfix on the LAB-matched source.

   Algebraic note: upstream's loop accumulates `high_freq += inp - low`
   over `level` iterations while reassigning `inp = low`. This
   telescopes — after `level` iters, `high_freq = orig - L_deep` where
   `L_deep` is the cascade of `level` Gaussian blurs at radii 2, 4, 8,
   ..., 2^level applied to `orig`. So `_wavelet_colorfix` reduces to
   `inp - inp_L_deep + target_L_deep`: substitute the deep low-pass of
   `inp` with that of `target`, keep `inp`'s high-frequency detail.

   We compute `L_deep` with a Burt-Adelson Gaussian pyramid (small blur
   + 2x decimate per level) instead of upstream's growing-radius
   cascade at full resolution. Both produce a smooth deep low-pass; the
   pyramid is roughly an order of magnitude faster (radius-32 blur on
   720x720 vs radius-2 blur on 22x22 at the deepest level) but yields
   a visibly different low-pass image — the bilinear upsample from a
   ~22x22 base introduces interpolation patches that the cascade-based
   blur doesn't have. Within the project's quality tolerance because
   the high-frequency content (which carries the recognisable image
   structure) is unchanged; only the low-pass tint shifts slightly.

cv2.GaussianBlur with sigma=0 becomes a manually-built 1D Gaussian
kernel (cv2's auto-sigma formula `0.3 * ((ksize - 1) * 0.5 - 1) + 0.8`)
applied separably via scipy.ndimage.convolve1d with `mode='mirror'`,
which matches cv2's default BORDER_REFLECT_101.
"""
from __future__ import annotations

import numpy as np
from PIL import Image
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


def _resize_bilinear_2d(arr: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    th, tw = target_hw
    contig = np.ascontiguousarray(arr, dtype=np.float32)
    return np.array(Image.fromarray(contig, mode="F").resize((tw, th), Image.BILINEAR))


def _pyramid_low(inp: np.ndarray, levels: int) -> np.ndarray:
    """Deep low-pass via small-blur + 2x-decimate pyramid. Returns an
    array with the same shape as `inp`, upsampled bilinearly from the
    deepest pyramid level."""
    target_h, target_w = inp.shape[:2]
    current = inp
    for _ in range(levels):
        current = _wavelet_blur(current, radius=2)
        current = current[::2, ::2]
    return _resize_bilinear_2d(current, (target_h, target_w))


def _wavelet_colorfix(inp: np.ndarray, target: np.ndarray, level: int) -> np.ndarray:
    inp_low = _pyramid_low(inp, level)
    target_low = _pyramid_low(target, level)
    return inp - inp_low + target_low


def match_color(
    source_rgb: np.ndarray, target_rgb: np.ndarray, level: int = 5
) -> np.ndarray:
    src_lab = rgb_to_lab(source_rgb).astype(np.float32) / 255.0

    # The target enters only via its scalar LAB mean/std. With ~4M pixels
    # at full res, the sample size is wildly excessive for a global mean
    # estimate; subsampling by stride 4 (still 256k+ samples) agrees to
    # ~5 decimals and saves a full rgb_to_lab call. Only kick in when the
    # target is large enough that the downsample actually helps.
    if min(target_rgb.shape[:2]) >= 512:
        tgt_lab = rgb_to_lab(target_rgb[::4, ::4]).astype(np.float32) / 255.0
    else:
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
