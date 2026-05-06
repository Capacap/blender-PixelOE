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
# so the analytic formula evaluates at only 256 distinct points per channel.
# Bake those into a LUT and index by the raw byte. ~7x faster on 2k images
# than computing the formula on the fly.
def _build_srgb_lin_lut() -> np.ndarray:
    x = np.arange(256, dtype=np.float32) / 255.0
    return np.where(
        x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4
    ).astype(np.float32)


_SRGB_LIN_LUT = _build_srgb_lin_lut()


# 65k-entry uint16 LUT for sRGB gamma encoding. The output of lab_to_rgb is
# uint8, so 16-bit precision on the input is well below output resolution.
# Quantising input to one of 65536 levels yields the same uint8 byte for
# every continuous input within 1/65535 of a level, so in practice the LUT
# matches a float-space evaluation of the gamma formula within max 1 byte /
# mean 0.01 byte on real images. ~6x faster on 2k images (the float power
# was the single dominant cost in lab_to_rgb).
def _build_srgb_gamma_lut() -> np.ndarray:
    x = np.arange(65536, dtype=np.float32) / 65535.0
    g = np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1.0 / 2.4) - 0.055)
    return np.clip(g * 255.0, 0, 255).astype(np.uint8)


_SRGB_GAMMA_LUT = _build_srgb_gamma_lut()


# 65k-entry LUTs for the LAB f() / f_inverse functions. f() is dominated by
# np.cbrt, which costs ~30ms per call on a 2k float32 array — three calls
# per rgb_to_lab. The LUT path is ~3x faster and within 1 byte of the float
# path on the final L channel. Domains chosen to cover all values that
# arise from in-gamut sRGB plus headroom for the slightly-out-of-range
# clipping that happens after the inverse XYZ matmul.
_F_LUT_DOMAIN_HI = np.float32(1.2)
_F_LUT_SCALE = np.float32(65535.0 / 1.2)
_F_INV_LUT_LO = np.float32(-0.5)
_F_INV_LUT_HI = np.float32(1.5)
_F_INV_LUT_SCALE = np.float32(65535.0 / 2.0)


def _build_f_lut() -> np.ndarray:
    x = np.linspace(0.0, float(_F_LUT_DOMAIN_HI), 65536, dtype=np.float64)
    return np.where(
        x > float(_DELTA_CUBED),
        np.cbrt(x),
        x * float(_INV_DELTA_FACTOR) + float(_F_OFFSET),
    ).astype(np.float32)


def _build_f_inverse_lut() -> np.ndarray:
    x = np.linspace(
        float(_F_INV_LUT_LO), float(_F_INV_LUT_HI), 65536, dtype=np.float64
    )
    return np.where(
        x > float(_DELTA),
        x**3,
        3 * float(_DELTA) ** 2 * (x - float(_F_OFFSET)),
    ).astype(np.float32)


_F_LUT = _build_f_lut()
_F_INV_LUT = _build_f_inverse_lut()


def _f(t: np.ndarray) -> np.ndarray:
    idx = np.clip(t * _F_LUT_SCALE, 0, 65535).astype(np.int32)
    return _F_LUT[idx]


def _f_inverse(t: np.ndarray) -> np.ndarray:
    idx = np.clip((t - _F_INV_LUT_LO) * _F_INV_LUT_SCALE, 0, 65535).astype(np.int32)
    return _F_INV_LUT[idx]


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


def rgb_to_lab_L(rgb: np.ndarray) -> np.ndarray:
    """Compute only the perceptual L* (lightness) channel of LAB from RGB,
    returned as float32 in [0, 1]. Callers that need only luminance (e.g.
    expansion_weight) avoid the XYZ matmul and the f() / packing for the
    a/b channels — about 3x faster than rgb_to_lab on a 2k input.

    Equivalent to `rgb_to_lab(rgb)[..., 0].astype(np.float32) / 255.0` up
    to the uint8 byte-quantization in the round-trip path; this routine
    skips that quantization, so values are slightly more precise.
    """
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(
            f"expected HxWx3 uint8 RGB; got dtype={rgb.dtype} shape={rgb.shape}"
        )
    rgb_lin = _SRGB_LIN_LUT[rgb]
    # D65 luminance: Y = 0.2127*R + 0.7152*G + 0.0722*B (second row of
    # _RGB_TO_XYZ). Y/Y_n with Y_n = 1.0 is just Y.
    Y = (
        rgb_lin[..., 0] * np.float32(0.2126729)
        + rgb_lin[..., 1] * np.float32(0.7151522)
        + rgb_lin[..., 2] * np.float32(0.0721750)
    )
    f_y = np.where(
        Y > _DELTA_CUBED, np.cbrt(Y), Y * _INV_DELTA_FACTOR + _F_OFFSET
    )
    return np.clip((116.0 * f_y - 16.0) * np.float32(0.01), 0.0, 1.0).astype(np.float32)


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
    idx = np.clip(rgb_lin * 65535.0, 0, 65535).astype(np.uint16)
    return _SRGB_GAMMA_LUT[idx]
