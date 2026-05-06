"""Colour-palette quantization.

Three method branches:

  * `kmeans` (unweighted): scipy.cluster.vq.kmeans2 with k-means++
    initialisation. ++ samples seeds proportional to squared distance
    from the nearest already-chosen centroid, generally finding
    lower-distortion clusterings than upstream's cv2
    KMEANS_RANDOM_CENTERS + 4-attempt best-of-N selection without
    needing the attempts loop.

  * `kmeans` with a weight map (used by `colors_with_weight=True`):
    same k-means++ init on unweighted pixels, then a Lloyd loop using
    weighted means `sum(w*pixel) / sum(w)` per cluster. Empty-cluster
    handling preserved (donate farthest-from-centroid pixel of the
    largest cluster) — ++ init usually avoids empties but small
    palettes on concentrated colour distributions can still trigger
    them.

  * `maxcover`: PIL.Image.quantize with MAXCOVERAGE (byte-exact with
    upstream, no kmeans involved).

Output drift vs upstream is intentional: numpy RNG ≠ cv2 RNG (already
true before this change), and ++ produces different palettes than
cv2's bounding-box random + best-of-N. Visually similar or better on
real images per the regression harness.
"""
from __future__ import annotations

import numpy as np
from PIL import Image
from scipy.cluster.vq import kmeans2, vq

_KMEANS_SEED = 0
_KMEANS_MAX_ITER = 32
_KMEANS_EPS = 1.0


def _assign(pixels: np.ndarray, centers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns (labels, squared distance to assigned centroid).

    Backed by scipy.cluster.vq.vq, which evaluates the BLAS-friendly
    ``||p||^2 - 2 p.c + ||c||^2`` expansion at the C level. Cast to
    float32 for the fast path, square the returned Euclidean distance
    on the way out. Float32 is comfortably precise for 8-bit RGB.
    """
    obs = np.ascontiguousarray(pixels, dtype=np.float32)
    book = np.ascontiguousarray(centers, dtype=np.float32)
    labels, dist = vq(obs, book)
    chosen = dist.astype(np.float64) ** 2
    return labels.astype(np.int64), chosen


def _kmeanspp_init(
    pixels: np.ndarray, k: int, rng: np.random.Generator
) -> np.ndarray:
    """K-means++ seeding: first centroid uniform random, subsequent
    centroids sampled with probability proportional to squared distance
    from the nearest already-chosen centroid."""
    n, dims = pixels.shape
    centers = np.empty((k, dims), dtype=np.float64)
    centers[0] = pixels[rng.integers(n)]
    d2 = np.sum((pixels - centers[0]) ** 2, axis=1)
    for j in range(1, k):
        total = float(d2.sum())
        if total <= 0:
            idx = int(rng.integers(n))
        else:
            r = rng.random() * total
            idx = int(np.searchsorted(np.cumsum(d2), r))
            if idx >= n:
                idx = n - 1
        centers[j] = pixels[idx]
        new_d2 = np.sum((pixels - centers[j]) ** 2, axis=1)
        np.minimum(d2, new_d2, out=d2)
    return centers


def _split_empties(
    pixels: np.ndarray,
    sums: np.ndarray,
    counts: np.ndarray,
    labels: np.ndarray,
    chosen: np.ndarray,
) -> None:
    """For each empty cluster, donate the farthest-from-centroid pixel
    of the currently-largest cluster. Operates on accumulated sums/counts
    (pre-divide) so the transferred pixel cleanly migrates from one
    cluster's running mean to the empty cluster's running mean."""
    k = sums.shape[0]
    for c in range(k):
        if counts[c] != 0:
            continue
        max_k = int(np.argmax(counts))
        center = sums[max_k] / counts[max_k]
        cluster_mask = labels == max_k
        cluster_pixels = pixels[cluster_mask]
        cluster_indices = np.flatnonzero(cluster_mask)
        d2 = ((cluster_pixels - center) ** 2).sum(axis=1)
        farthest_local = int(np.argmax(d2))
        farthest_i = int(cluster_indices[farthest_local])
        sample = pixels[farthest_i]
        sums[max_k] -= sample
        sums[c] = sample.copy()
        counts[max_k] -= 1
        counts[c] = 1
        labels[farthest_i] = c
        chosen[farthest_i] = 0.0


def _kmeans_quant(rgb: np.ndarray, colors: int) -> np.ndarray:
    h, w, _ = rgb.shape
    pixels = rgb.reshape(-1, 3).astype(np.float32)
    centroids, labels = kmeans2(
        pixels,
        colors,
        iter=_KMEANS_MAX_ITER,
        minit="++",
        seed=_KMEANS_SEED,
        missing="warn",
    )
    out = centroids[labels].reshape(h, w, 3)
    return np.clip(out, 0, 255).astype(np.uint8)


def _weighted_kmeans_quant(
    rgb: np.ndarray, weights: np.ndarray, colors: int, repeats: int
) -> np.ndarray:
    h, w, _ = rgb.shape
    pixels = rgb.reshape(-1, 3).astype(np.float64)
    w_norm = weights / np.max(weights) * repeats
    sample_w = np.maximum(1.0, w_norm).reshape(-1).astype(np.float64)

    rng = np.random.default_rng(_KMEANS_SEED)
    centers = _kmeanspp_init(pixels, colors, rng)
    labels, chosen = _assign(pixels, centers)
    weighted_pixels = pixels * sample_w[:, None]
    dims = pixels.shape[1]
    for _ in range(_KMEANS_MAX_ITER):
        sums = np.zeros((colors, dims), dtype=np.float64)
        np.add.at(sums, labels, weighted_pixels)
        counts = np.bincount(labels, minlength=colors).astype(np.int64)
        _split_empties(pixels, sums, counts, labels, chosen)
        wsums = np.zeros(colors, dtype=np.float64)
        np.add.at(wsums, labels, sample_w)
        wsums_safe = np.where(wsums > 0, wsums, 1.0)
        new_centers = sums / wsums_safe[:, None]
        shift = np.linalg.norm(new_centers - centers, axis=1).max()
        centers = new_centers
        labels, chosen = _assign(pixels, centers)
        if shift < _KMEANS_EPS:
            break

    out = centers[labels].reshape(h, w, 3)
    return np.clip(out, 0, 255).astype(np.uint8)


def _maxcover_quant(rgb: np.ndarray, colors: int) -> np.ndarray:
    img = Image.fromarray(rgb)
    quant = img.quantize(
        colors=colors, method=Image.Quantize.MAXCOVERAGE, kmeans=colors
    ).convert("RGB")
    return np.array(quant)


def color_quant(
    rgb: np.ndarray,
    colors: int,
    weights: np.ndarray | None = None,
    repeats: int = 64,
    method: str = "kmeans",
) -> np.ndarray:
    if method == "kmeans":
        if weights is not None:
            return _weighted_kmeans_quant(rgb, weights, colors, repeats)
        return _kmeans_quant(rgb, colors)
    if method == "maxcover":
        return _maxcover_quant(rgb, colors)
    raise ValueError(f"unknown quantize method: {method!r}")
