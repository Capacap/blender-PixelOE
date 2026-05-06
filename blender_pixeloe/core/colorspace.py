"""sRGB <-> LAB conversion matching cv2's uint8 byte layout.

cv2's BGR2LAB on uint8 stores LAB as uint8 with a non-standard layout:

    L_byte = L * 255/100   (CIE L is normally 0-100)
    a_byte = a + 128       (CIE a is signed)
    b_byte = b + 128       (CIE b is signed)

The brief mandates parity within 1.0 LAB unit per channel against cv2's
output, so this module produces and consumes LAB in cv2's byte layout, not
in CIE standard ranges. Pinned to the D65 illuminant and the standard sRGB
transfer function (linear segment below 0.04045).

All arrays are uint8 HxWx3 at the boundary; intermediate math runs in
float32. RGB throughout (no BGR convention).
"""
from __future__ import annotations

import numpy as np

# sRGB to CIE XYZ (D65), the standard matrix used by skimage and cv2.
_RGB_TO_XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ],
    dtype=np.float32,
)
_XYZ_TO_RGB = np.array(
    [
        [3.2404542, -1.5371385, -0.4985314],
        [-0.9692660, 1.8760108, 0.0415560],
        [0.0556434, -0.2040259, 1.0572252],
    ],
    dtype=np.float32,
)
_XN, _YN, _ZN = np.float32(0.95047), np.float32(1.0), np.float32(1.08883)
_DELTA = np.float32(6.0 / 29.0)
_DELTA_CUBED = _DELTA**3
_INV_DELTA_FACTOR = np.float32(1.0 / (3.0 * float(_DELTA) ** 2))
_F_OFFSET = np.float32(4.0 / 29.0)

# 256-entry LUT for sRGB linearisation. The input to rgb_to_lab is uint8,
# so the float-space np.where + power evaluates the same function at only
# 256 distinct points per channel. Bake those into a LUT and index by the
# raw byte. Byte-identical to the np.where path; ~7x faster on 2k images.
def _build_srgb_lin_lut() -> np.ndarray:
    x = np.arange(256, dtype=np.float32) / 255.0
    return np.where(
        x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4
    ).astype(np.float32)


_SRGB_LIN_LUT = _build_srgb_lin_lut()


def _srgb_linearize(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _srgb_gamma(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1.0 / 2.4) - 0.055)


def _f(t: np.ndarray) -> np.ndarray:
    return np.where(t > _DELTA_CUBED, np.cbrt(t), t * _INV_DELTA_FACTOR + _F_OFFSET)


def _f_inverse(t: np.ndarray) -> np.ndarray:
    return np.where(t > _DELTA, t**3, 3 * _DELTA**2 * (t - _F_OFFSET))


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """RGB uint8 HxWx3 to LAB uint8 HxWx3 in cv2's byte layout."""
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(
            f"expected HxWx3 uint8 RGB; got dtype={rgb.dtype} shape={rgb.shape}"
        )

    rgb_lin = _SRGB_LIN_LUT[rgb]

    xyz = rgb_lin @ _RGB_TO_XYZ.T
    x_n = xyz[..., 0] / _XN
    y_n = xyz[..., 1] / _YN
    z_n = xyz[..., 2] / _ZN

    fx, fy, fz = _f(x_n), _f(y_n), _f(z_n)

    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)

    L_byte = np.clip(L * (255.0 / 100.0), 0, 255)
    a_byte = np.clip(a + 128.0, 0, 255)
    b_byte = np.clip(b + 128.0, 0, 255)

    return np.stack([L_byte, a_byte, b_byte], axis=-1).astype(np.uint8)


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """LAB uint8 HxWx3 in cv2's byte layout to RGB uint8 HxWx3."""
    if lab.dtype != np.uint8 or lab.ndim != 3 or lab.shape[2] != 3:
        raise ValueError(
            f"expected HxWx3 uint8 LAB; got dtype={lab.dtype} shape={lab.shape}"
        )

    lab_f = lab.astype(np.float32)
    L = lab_f[..., 0] * (100.0 / 255.0)
    a = lab_f[..., 1] - 128.0
    b = lab_f[..., 2] - 128.0

    fy = (L + 16.0) / 116.0
    fx = a / 500.0 + fy
    fz = fy - b / 200.0

    x = _f_inverse(fx) * _XN
    y = _f_inverse(fy) * _YN
    z = _f_inverse(fz) * _ZN

    xyz = np.stack([x, y, z], axis=-1)
    rgb_lin = xyz @ _XYZ_TO_RGB.T
    rgb_f = _srgb_gamma(rgb_lin)
    return np.clip(rgb_f * 255.0, 0, 255).astype(np.uint8)
