"""Colour-palette quantization.

Direct port target: upstream `pixeloe.legacy.color.color_quant` plus its
three method branches (`kmeans`, `weighted_kmeans`, `maxcover`).

Upstream uses cv2.kmeans with `KMEANS_RANDOM_CENTERS` and termination
criteria `(EPS+MAX_ITER, 32, 1.0)`, run for 4 attempts with the lowest-
distortion attempt returned. The port replicates that algorithm with a
seeded numpy implementation:

  * Init: KMEANS_RANDOM_CENTERS in cv2 generates K uniformly random
    points in the data's per-dimension bounding box, expanded by a
    `1/dims` margin on each side. This is NOT random pixel sampling.
    The distinction matters: spreading initial centroids uniformly
    across colour space lets outlier regions (e.g. a small bright dot
    in a dark image) attract a nearby centroid early, instead of being
    permanently absorbed into a bulk cluster.

  * Loop structure mirrors cv2's:
        iter 0: random init -> assign labels
        iter k: means-by-label -> empty-cluster split -> shift check ->
                final-assign-and-exit OR re-assign-and-loop
    Convergence: max squared centroid shift <= eps^2 (cv2 squares its
    epsilon internally; passing eps=1.0 means stop when the largest L2
    centroid movement drops below 1.0).

  * Empty-cluster handling: when a cluster has no members after the
    label assignment, the largest cluster's farthest-from-centroid
    pixel is reassigned to the empty cluster (and the centroid placed
    at that pixel). Mirrors cv2's behaviour byte-for-byte.

  * 4 attempts seeded sequentially from a single base seed; return the
    attempt with the smallest sum-of-squared-distances distortion.

The port cannot bit-match cv2 because numpy and cv2 use different RNGs.
On real images (snow-leopard, dark-highlights) the resulting palettes
are visually equivalent and capture outliers correctly. On synthetic
gradients the local-minimum sensitivity of Lloyd's surfaces — both
upstream and port end up at chunky-but-different posterizations.

The 'maxcover' branch stays on `PIL.Image.quantize` (byte-exact with
upstream).
"""
from __future__ import annotations

import numpy as np
from PIL import Image

_KMEANS_MAX_ITER = 32
_KMEANS_EPS = 1.0
_KMEANS_ATTEMPTS = 4
_KMEANS_SEED = 0


def _assign(pixels: np.ndarray, centers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns (labels, squared distance to assigned centroid)."""
    d2 = ((pixels[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    labels = np.argmin(d2, axis=1).astype(np.int64)
    chosen = d2[np.arange(pixels.shape[0]), labels]
    return labels, chosen


def _generate_random_centers(
    pixels: np.ndarray, k: int, rng: np.random.Generator
) -> np.ndarray:
    """Mirror cv2's KMEANS_RANDOM_CENTERS: uniform random points in the
    data's per-dimension bounding box extended by `1/dims` margin on each
    side. NOT random pixel sampling."""
    dims = pixels.shape[1]
    margin = 1.0 / dims
    box_min = pixels.min(axis=0).astype(np.float64)
    box_max = pixels.max(axis=0).astype(np.float64)
    spread = box_max - box_min
    u = rng.random((k, dims))
    return u * (1.0 + 2.0 * margin) * spread - margin * spread + box_min


def _sums_and_counts(
    pixels: np.ndarray, labels: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized per-cluster sums and counts."""
    dims = pixels.shape[1]
    sums = np.zeros((k, dims), dtype=np.float64)
    np.add.at(sums, labels, pixels)
    counts = np.bincount(labels, minlength=k).astype(np.int64)
    return sums, counts


def _split_empties(
    pixels: np.ndarray,
    sums: np.ndarray,
    counts: np.ndarray,
    labels: np.ndarray,
    chosen: np.ndarray,
) -> None:
    """For each empty cluster, donate the farthest-from-centroid pixel of
    the currently-largest cluster to it. Mirrors cv2's empty-cluster
    handling: operates on accumulated sums/counts (pre-divide) so the
    transferred pixel cleanly migrates from one cluster's running mean to
    the empty cluster's running mean."""
    k = sums.shape[0]
    for c in range(k):
        if counts[c] != 0:
            continue
        max_k = int(np.argmax(counts))
        # Find pixel in max_k cluster with the largest distance to that
        # cluster's *current* centroid (sums[max_k] / counts[max_k]).
        center = sums[max_k] / counts[max_k]
        cluster_mask = labels == max_k
        cluster_pixels = pixels[cluster_mask]
        cluster_indices = np.flatnonzero(cluster_mask)
        d2 = ((cluster_pixels - center) ** 2).sum(axis=1)
        farthest_local = int(np.argmax(d2))
        farthest_i = int(cluster_indices[farthest_local])
        sample = pixels[farthest_i]
        sums[max_k] -= sample
        sums[c] = sums[c] + sample
        counts[max_k] -= 1
        counts[c] = 1
        labels[farthest_i] = c
        chosen[farthest_i] = 0.0


def _lloyd_attempt(
    pixels: np.ndarray, k: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, float]:
    centers = _generate_random_centers(pixels, k, rng)
    labels, chosen = _assign(pixels, centers)
    for iter_idx in range(_KMEANS_MAX_ITER):
        sums, counts = _sums_and_counts(pixels, labels, k)
        _split_empties(pixels, sums, counts, labels, chosen)
        new_centers = sums / counts[:, None]
        shift = np.linalg.norm(new_centers - centers, axis=1).max()
        is_last = iter_idx == _KMEANS_MAX_ITER - 1 or shift < _KMEANS_EPS
        centers = new_centers
        labels, chosen = _assign(pixels, centers)
        if is_last:
            break
    return centers, labels, float(chosen.sum())


def _kmeans_random_init(pixels: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(_KMEANS_SEED)
    pix = pixels.astype(np.float64)
    best = None
    for _ in range(_KMEANS_ATTEMPTS):
        centers, labels, distortion = _lloyd_attempt(pix, k, rng)
        if best is None or distortion < best[2]:
            best = (centers, labels, distortion)
    return best[0], best[1]


def _weighted_lloyd_attempt(
    pixels: np.ndarray, weights: np.ndarray, k: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, float]:
    """Sample-weighted variant. Init via the same bounding-box random
    points as the unweighted path; weighted means via per-cluster sum of
    (pixel * weight) divided by sum of weights."""
    centers = _generate_random_centers(pixels, k, rng)
    labels, chosen = _assign(pixels, centers)
    weighted_pixels = pixels * weights[:, None]
    for iter_idx in range(_KMEANS_MAX_ITER):
        dims = pixels.shape[1]
        sums = np.zeros((k, dims), dtype=np.float64)
        np.add.at(sums, labels, weighted_pixels)
        wsums = np.zeros(k, dtype=np.float64)
        np.add.at(wsums, labels, weights)
        counts = np.bincount(labels, minlength=k).astype(np.int64)
        _split_empties(pixels, sums, counts, labels, chosen)
        # rebuild wsums from labels after empty splitting (cluster membership changed)
        wsums = np.zeros(k, dtype=np.float64)
        np.add.at(wsums, labels, weights)
        wsums_safe = np.where(wsums > 0, wsums, 1.0)
        new_centers = sums / wsums_safe[:, None]
        shift = np.linalg.norm(new_centers - centers, axis=1).max()
        is_last = iter_idx == _KMEANS_MAX_ITER - 1 or shift < _KMEANS_EPS
        centers = new_centers
        labels, chosen = _assign(pixels, centers)
        if is_last:
            break
    return centers, labels, float((chosen * weights).sum())


def _kmeans_weighted(
    pixels: np.ndarray, weights: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(_KMEANS_SEED)
    pix = pixels.astype(np.float64)
    best = None
    for _ in range(_KMEANS_ATTEMPTS):
        centers, labels, distortion = _weighted_lloyd_attempt(pix, weights, k, rng)
        if best is None or distortion < best[2]:
            best = (centers, labels, distortion)
    return best[0], best[1]


def _kmeans_quant(rgb: np.ndarray, colors: int) -> np.ndarray:
    h, w, _ = rgb.shape
    pixels = rgb.reshape(-1, 3)
    centroids, labels = _kmeans_random_init(pixels, colors)
    out = centroids[labels].reshape(h, w, 3)
    return np.clip(out, 0, 255).astype(np.uint8)


def _weighted_kmeans_quant(
    rgb: np.ndarray, weights: np.ndarray, colors: int, repeats: int
) -> np.ndarray:
    h, w, _ = rgb.shape
    pixels = rgb.reshape(-1, 3)
    w_norm = weights / np.max(weights) * repeats
    sample_w = np.maximum(1.0, w_norm).reshape(-1).astype(np.float64)
    centroids, labels = _kmeans_weighted(pixels, sample_w, colors)
    out = centroids[labels].reshape(h, w, 3)
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
