"""Round-trip tests for the pure-numpy helpers in `blender_pixeloe.image_io`.

These exercise the orientation / alpha / dtype / sRGB pipeline without
importing bpy. The bpy boundary functions are covered by manual verification
inside Blender at install time.
"""
from __future__ import annotations

import numpy as np
import pytest

from blender_pixeloe.image_io import (
    _add_alpha,
    _drop_alpha,
    _flat_linear_rgba_to_top_down_srgb_uint8,
    _flip_vertical,
    _float_to_uint8,
    _linear_to_srgb,
    _srgb_to_linear,
    _top_down_srgb_uint8_to_flat_linear_rgba,
    _uint8_to_float,
)


def _checkerboard_rgb(h: int, w: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


def test_drop_alpha_strips_4th_channel():
    rgba = np.arange(2 * 3 * 4, dtype=np.uint8).reshape(2, 3, 4)
    rgb = _drop_alpha(rgba)
    assert rgb.shape == (2, 3, 3)
    np.testing.assert_array_equal(rgb, rgba[..., :3])


def test_add_alpha_appends_full_alpha():
    rgb = np.full((2, 3, 3), 0.5, dtype=np.float32)
    rgba = _add_alpha(rgb, fill=1.0)
    assert rgba.shape == (2, 3, 4)
    assert rgba.dtype == np.float32
    np.testing.assert_array_equal(rgba[..., :3], rgb)
    np.testing.assert_array_equal(rgba[..., 3], 1.0)


def test_flip_vertical_inverts_axis_0():
    arr = np.arange(12).reshape(3, 4)
    flipped = _flip_vertical(arr)
    np.testing.assert_array_equal(flipped, arr[::-1])
    np.testing.assert_array_equal(_flip_vertical(flipped), arr)


def test_float_to_uint8_clamps_out_of_range():
    arr = np.array([[-0.5, 0.0, 0.5, 1.0, 1.5]], dtype=np.float32)
    out = _float_to_uint8(arr)
    np.testing.assert_array_equal(out, np.array([[0, 0, 128, 255, 255]], dtype=np.uint8))


def test_uint8_to_float_normalizes_to_0_1():
    arr = np.array([[0, 128, 255]], dtype=np.uint8)
    out = _uint8_to_float(arr)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, [[0.0, 128 / 255.0, 1.0]])


def test_uint8_round_trip_through_float_is_lossless():
    arr = np.arange(256, dtype=np.uint8)
    recovered = _float_to_uint8(_uint8_to_float(arr))
    np.testing.assert_array_equal(recovered, arr)


def test_srgb_transfer_matches_known_values():
    # IEC 61966-2-1 reference points
    linear = np.array([0.0, 0.0031308, 0.5, 1.0], dtype=np.float32)
    srgb = _linear_to_srgb(linear)
    np.testing.assert_allclose(
        srgb, [0.0, 0.04045, 0.7353569, 1.0], atol=1e-5
    )


def test_srgb_round_trip_is_near_lossless():
    rng = np.random.default_rng(0)
    arr = rng.random((128, 128, 3), dtype=np.float32)
    recovered = _linear_to_srgb(_srgb_to_linear(arr))
    np.testing.assert_allclose(recovered, arr, atol=1e-5)


def test_uint8_round_trip_through_full_pipeline_is_lossless():
    """Top-down sRGB uint8 -> linear flat -> top-down sRGB uint8 must recover."""
    rgb = _checkerboard_rgb(32, 64)
    flat = _top_down_srgb_uint8_to_flat_linear_rgba(rgb)
    assert flat.dtype == np.float32
    recovered = _flat_linear_rgba_to_top_down_srgb_uint8(flat, 32, 64)
    # uint8 -> sRGB float -> linear -> sRGB float -> uint8 has at most 1
    # least-significant-bit drift from the float pipeline.
    diff = np.abs(recovered.astype(np.int16) - rgb.astype(np.int16))
    assert diff.max() <= 1, f"max LSB drift {diff.max()} exceeds 1"


@pytest.mark.parametrize("size", [(1, 1), (1, 8), (8, 1), (3, 5), (16, 16)])
def test_full_pipeline_preserves_shape_and_alpha(size):
    h, w = size
    rgb = _checkerboard_rgb(h, w)
    flat = _top_down_srgb_uint8_to_flat_linear_rgba(rgb)
    assert flat.shape == (h * w * 4,)
    rgba = flat.reshape(h, w, 4)
    np.testing.assert_array_equal(rgba[..., 3], 1.0)


def test_full_pipeline_flips_vertically():
    h, w = 3, 2
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[0] = (255, 0, 0)
    rgb[2] = (0, 0, 255)

    flat = _top_down_srgb_uint8_to_flat_linear_rgba(rgb)
    rgba = flat.reshape(h, w, 4)
    # Top row (red) should land at the bottom of the bottom-up buffer.
    assert rgba[2, 0, 0] > 0.99 and rgba[2, 0, 1] < 0.01
    assert rgba[0, 0, 2] > 0.99 and rgba[0, 0, 0] < 0.01
