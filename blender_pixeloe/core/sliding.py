"""Sliding-window helper replacing upstream's apply_chunk_torch.

Upstream `pixeloe.legacy.utils.apply_chunk_torch` does this in three steps:

    1. Pad the input asymmetrically by `k_shift = max(kernel - stride, 0)`
       so that windowing at stride steps yields exactly H/stride * W/stride
       windows (one window per stride-sized output tile).
    2. Apply `func` to all windows; func reduces along the kernel axis,
       returning one scalar per window.
    3. Tile each scalar into a stride x stride block, arranged so the output
       has the original H x W shape (tile-constant output).

The numpy port uses `numpy.lib.stride_tricks.sliding_window_view` for the
windowing and `np.repeat` for the tile-up, sidestepping the torch
unfold/fold round-trip. The func signature changes from torch to numpy:

    func(windows: ndarray (N, kernel*kernel)) -> ndarray (N, 1) or (N,)

where N is the number of stride-sized tiles. `keepdims=True`-style output
is accepted so torch reductions like

    lambda x: torch.median(x, dim=1, keepdims=True).values

translate directly to

    lambda x: np.median(x, axis=1, keepdims=True)

Boundary note: when input dims aren't multiples of stride, the last partial
row/column is filled by edge-replicating the nearest valid tile. Upstream's
F.fold zero-fills instead. The production pipeline always passes
stride-divisible dims (by construction in `pixelize.py`), so this divergence
only affects degenerate inputs.
"""
from __future__ import annotations

from typing import Callable

import numpy as np


def apply_chunk(
    data: np.ndarray,
    kernel: int,
    stride: int,
    func: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    """Apply `func` to every (kernel x kernel) window centered on a stride-tile.

    Output shape == input shape; each `stride x stride` tile is filled with
    the constant returned by `func` for that tile's neighborhood. Border
    tiles use edge-replicated padding for missing input.
    """
    if data.ndim != 2:
        raise ValueError(f"expected 2D array, got shape {data.shape}")
    org_h, org_w = data.shape

    k_shift = max(kernel - stride, 0)
    pad = (k_shift // 2, k_shift // 2 + k_shift % 2)
    padded = np.pad(data, (pad, pad), mode="edge")

    windows = np.lib.stride_tricks.sliding_window_view(padded, (kernel, kernel))
    windows = windows[::stride, ::stride]
    n_h, n_w = windows.shape[:2]
    flat = windows.reshape(n_h * n_w, kernel * kernel)

    reduced = func(flat)
    if reduced.ndim == 2:
        reduced = reduced[..., 0]
    if reduced.size != n_h * n_w:
        raise ValueError(
            f"func returned shape {reduced.shape}; expected ({n_h * n_w},) "
            f"or ({n_h * n_w}, 1)"
        )

    grid = reduced.reshape(n_h, n_w).astype(data.dtype, copy=False)
    out = np.repeat(np.repeat(grid, stride, axis=0), stride, axis=1)
    out_h, out_w = out.shape
    if out_h < org_h or out_w < org_w:
        out = np.pad(
            out,
            ((0, max(0, org_h - out_h)), (0, max(0, org_w - out_w))),
            mode="edge",
        )
    return out[:org_h, :org_w]
