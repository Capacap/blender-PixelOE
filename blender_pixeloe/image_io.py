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


def _flat_linear_rgba_to_top_down_srgb_uint8(flat: np.ndarray, h: int, w: int) -> np.ndarray:
    rgba = flat.reshape(h, w, 4)
    rgb_linear = _drop_alpha(rgba)
    rgb_srgb = _linear_to_srgb(rgb_linear)
    return _float_to_uint8(_flip_vertical(rgb_srgb))


def _top_down_srgb_uint8_to_flat_linear_rgba(rgb_uint8: np.ndarray) -> np.ndarray:
    rgb_srgb = _uint8_to_float(rgb_uint8)
    rgb_linear = _srgb_to_linear(rgb_srgb)
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
