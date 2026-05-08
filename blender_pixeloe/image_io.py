"""Blender Image <-> numpy boundary helpers.

The two entry points are `image_to_array` and `array_to_image`. They handle:
  - vertical flip (Blender stores bottom-up, numpy works top-down)
  - alpha drop / restore
  - dtype scale (Blender float 0-1 <-> uint8 0-255)
  - linear <-> sRGB conversion

The colorspace point matters: Blender decodes sRGB-tagged files to scene-
linear floats at load time, so `image.pixels` always returns linear values
regardless of the image's colorspace tag. PixelOE expects display-space
(gamma-encoded) values for its contrast statistics, so we apply the sRGB
transfer in numpy on the way in and its inverse on the way out.

The pure-numpy helpers prefixed `_` are exposed for unit tests so a
checkerboard can be round-tripped without importing bpy.
"""
from __future__ import annotations

import numpy as np

try:
    import bpy
except ImportError:
    bpy = None


def _drop_alpha(rgba: np.ndarray) -> np.ndarray:
    return rgba[..., :3]


def _add_alpha(rgb: np.ndarray, fill: float = 1.0) -> np.ndarray:
    h, w = rgb.shape[:2]
    out = np.empty((h, w, 4), dtype=rgb.dtype)
    out[..., :3] = rgb
    out[..., 3] = fill
    return out


def _flip_vertical(arr: np.ndarray) -> np.ndarray:
    return arr[::-1, ...]


def _float_to_uint8(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8)


def _uint8_to_float(arr: np.ndarray) -> np.ndarray:
    return arr.astype(np.float32) / 255.0


def _linear_to_srgb(arr: np.ndarray) -> np.ndarray:
    """IEC 61966-2-1 sRGB transfer. Input/output float in [0, 1]."""
    arr = np.clip(arr, 0.0, 1.0)
    out = np.where(
        arr <= 0.0031308,
        arr * 12.92,
        1.055 * np.power(arr, 1.0 / 2.4) - 0.055,
    )
    return out.astype(np.float32, copy=False)


def _srgb_to_linear(arr: np.ndarray) -> np.ndarray:
    """Inverse sRGB transfer. Input/output float in [0, 1]."""
    arr = np.clip(arr, 0.0, 1.0)
    out = np.where(
        arr <= 0.04045,
        arr / 12.92,
        np.power((arr + 0.055) / 1.055, 2.4),
    )
    return out.astype(np.float32, copy=False)


# LUT-based fast paths. The analytic transfers above are correct but slow on
# 4k inputs because np.where evaluates both branches and np.power runs on
# every pixel. With ~10M pixels per call, image_to_array becomes the dominant
# cost in the operator wall-clock. The LUTs collapse each transfer to a
# single integer indexing op.
_LIN_TO_SRGB_U8_LUT_SIZE = 4096


def _build_srgb_u8_to_linear_f32_lut() -> np.ndarray:
    """One float32 entry per uint8 sRGB value. Exact, no precision loss."""
    return _srgb_to_linear(_uint8_to_float(np.arange(256, dtype=np.uint8)))


def _build_linear_f32_to_srgb_u8_lut(n: int) -> np.ndarray:
    """N uint8 entries indexed by quantized float32 linear-light input. n=4096
    gives at most 1 LSB difference from the analytic path; verified by test."""
    x = np.linspace(0.0, 1.0, n, dtype=np.float32)
    return _float_to_uint8(_linear_to_srgb(x))


_SRGB_U8_TO_LINEAR_F32_LUT = _build_srgb_u8_to_linear_f32_lut()
_LINEAR_F32_TO_SRGB_U8_LUT = _build_linear_f32_to_srgb_u8_lut(
    _LIN_TO_SRGB_U8_LUT_SIZE
)


def _srgb_u8_to_linear_f32(arr_u8: np.ndarray) -> np.ndarray:
    """Replaces `_uint8_to_float` + `_srgb_to_linear` with a 256-entry LUT lookup."""
    return _SRGB_U8_TO_LINEAR_F32_LUT[arr_u8]


def _linear_f32_to_srgb_u8(arr_f32: np.ndarray) -> np.ndarray:
    """Replaces `_linear_to_srgb` + `_float_to_uint8` with a 4096-entry LUT lookup.
    Clipping happens via the index range so out-of-[0,1] inputs are handled."""
    n_minus_1 = _LIN_TO_SRGB_U8_LUT_SIZE - 1
    idx = np.clip(arr_f32 * n_minus_1 + 0.5, 0.0, float(n_minus_1)).astype(np.intp)
    return _LINEAR_F32_TO_SRGB_U8_LUT[idx]


def _flat_linear_rgba_to_top_down_srgb_uint8(flat: np.ndarray, h: int, w: int) -> np.ndarray:
    rgba = flat.reshape(h, w, 4)
    rgb_linear = _drop_alpha(rgba)
    rgb_uint8 = _linear_f32_to_srgb_u8(rgb_linear)
    return _flip_vertical(rgb_uint8)


def _top_down_srgb_uint8_to_flat_linear_rgba(rgb_uint8: np.ndarray) -> np.ndarray:
    rgb_linear = _srgb_u8_to_linear_f32(rgb_uint8)
    rgba = _add_alpha(rgb_linear, fill=1.0)
    bottom_up = _flip_vertical(rgba)
    return np.ascontiguousarray(bottom_up).reshape(-1)


def image_to_array(image) -> np.ndarray:
    """Read a Blender Image as HxWx3 uint8 RGB, top-down, sRGB display-space."""
    if bpy is None:
        raise RuntimeError("image_to_array requires bpy")
    if image.type == 'RENDER_RESULT':
        raise ValueError(
            "Cannot read pixels from a Render Result. Save the render to a "
            "file first and load it as an Image."
        )
    w, h = image.size
    if w == 0 or h == 0:
        raise ValueError(f"Image {image.name!r} has zero size {(w, h)}.")

    flat = np.empty(w * h * 4, dtype=np.float32)
    image.pixels.foreach_get(flat)
    return _flat_linear_rgba_to_top_down_srgb_uint8(flat, h, w)


def array_to_image(arr: np.ndarray, name: str, overwrite: bool = True):
    """Create or update a Blender Image from an HxWx3 uint8 RGB array."""
    if bpy is None:
        raise RuntimeError("array_to_image requires bpy")
    if arr.dtype != np.uint8 or arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(
            f"Expected (H, W, 3) uint8 array, got dtype={arr.dtype} shape={arr.shape}"
        )

    h, w = arr.shape[:2]
    flat = _top_down_srgb_uint8_to_flat_linear_rgba(arr)

    image = bpy.data.images.get(name) if overwrite else None
    if image is not None:
        if image.size[0] != w or image.size[1] != h:
            image.scale(w, h)
    else:
        image = bpy.data.images.new(name, width=w, height=h, alpha=False)
        # Force the pixel buffer to materialize before the first foreach_set;
        # without this the initial write into a freshly-created image can be
        # swallowed and the editor shows a black texture until something else
        # touches the buffer.
        image.update()

    image.pixels.foreach_set(flat)
    image.update()
    return image
