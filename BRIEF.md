# Project Brief: Blender addon for PixelOE

- Project name `blender-pixeloe`
- pinned commit hash from `KohakuBlueleaf/PixelOE`: `f7c0ae1`
- Target blender version: `4.2+` current stable version is `5.1`

## Context

This project is a Blender addon that wraps the PixelOE pixelization algorithm by KohakuBlueleaf. Upstream is at https://github.com/KohakuBlueleaf/PixelOE, license Apache 2.0. PixelOE produces pixel art from regular images using a contrast-aware outline expansion technique that preserves fine details across downscaling. It's not an AI/ML technique despite the dependency surface; the math is classical image processing.

The addon exposes a single operator that takes any Blender Image datablock and produces a pixelized version as a new Image datablock. Same code path serves textures, rendered images saved to file, and reference images. No render handlers, no compositor nodes, no shader integration in v1.

## Why port the algorithm rather than wrap upstream

Upstream depends on `torch`, `torchvision`, `kornia`, `opencv-python`. Blender's bundled Python ships only numpy and PIL. Pip-installing torch into Blender's site-packages is fragile across distros and Blender versions, and would force a multi-hundred-megabyte first-run download. A pure-numpy port keeps the addon a clean zip install with no runtime side effects.

The legacy upstream implementation is ~640 lines of Python across 9 files. Most of that translates directly to numpy. The torch usage is only `torch.nn.functional.unfold/fold` plus `torch.median/max/min`, all of which have numpy equivalents. The cv2 usage is the substantive replacement work and is concentrated in: LAB color conversion, morphological erode/dilate, Gaussian blur, k-means clustering, image resize.

## Repository layout

```
<project-name>/
├── pyproject.toml              # for the core package (so harness can pip install it)
├── README.md
├── LICENSE                     # Apache 2.0
├── blender_pixeloe/            # the directory that becomes the addon zip
│   ├── __init__.py             # bl_info, register, unregister
│   ├── operators.py
│   ├── panels.py
│   ├── image_io.py             # the Blender ↔ numpy boundary helpers
│   └── core/                   # pure algorithm, no `import bpy`
│       ├── __init__.py
│       ├── pixelize.py         # top-level orchestrator
│       ├── outline.py
│       ├── colorspace.py
│       ├── sliding.py
│       ├── downscale_contrast.py
│       ├── downscale_kcentroid.py
│       ├── color_match.py
│       └── quantize.py
└── tests/
    ├── harness/
    │   └── compare.py          # diffs upstream vs port
    ├── images/                 # committed test images
    └── unit/                   # numpy-only unit tests for core
```

The `core` subpackage is the algorithm. It must remain importable outside Blender. The harness pip installs upstream pixeloe in a separate venv and imports `blender_pixeloe.core` directly. No `import bpy` anywhere in `core/`.

## Phase 0: Setup

Initialize the repo. Pin upstream to `<upstream-commit>` in a setup script that clones it into `tests/upstream/` for reference reading. Commit three test images:

1. A photograph for general visual quality (snow leopard from upstream `img/`)
2. A synthetic gradient with sharp edges, useful for algorithmic correctness checks
3. A dark scene with bright highlights for color space and weight map testing

The `pyproject.toml` declares `numpy` and `Pillow` as dependencies. Optionally `scipy` if morphological operations end up needing `scipy.ndimage`; the brief defaults to attempting custom numpy first.

## Phase 1: Comparison harness

This phase ships before any algorithm port code.

Deliverable: `tests/harness/compare.py`. The script takes an image path and a settings dictionary, runs both upstream pixeloe and the local `blender_pixeloe.core.pixelize`, and produces a 4-panel output PNG (input, upstream output, port output, per-pixel L1 diff heatmap) plus printed statistics (mean L1, max L1, percent of pixels with diff > threshold).

At this phase, `core.pixelize` is a stub that returns the input image unchanged. The harness should produce a meaningful, visible diff against this stub (not zero diff). This proves the harness works end-to-end before any port code is trusted by it.

Acceptance: `python tests/harness/compare.py tests/images/snow-leopard.webp --target_size 256 --patch_size 8` produces a PNG with all four panels and prints L1 statistics.

## Phase 2: Numpy algorithm port

Port the algorithm in this order. Each step is validated through the harness before moving to the next. Reference upstream code in `tests/upstream/src/pixeloe/legacy/`.

### 2a: sRGB ↔ LAB conversion

File: `blender_pixeloe/core/colorspace.py`

Implement `rgb_to_lab(arr: np.ndarray) -> np.ndarray` and inverse. Pin to D65 illuminant and the standard sRGB transfer function (with the linear segment below 0.04045). The reference algorithm is identical to `skimage.color.rgb2lab`; implement from scratch in numpy without taking the dependency.

Validation: output must match `cv2.cvtColor(img, cv2.COLOR_BGR2LAB)` within 1.0 LAB unit per channel on the three test images. Run upstream's cv2 conversion in the harness venv to generate the reference.

### 2b: Sliding window helper

File: `blender_pixeloe/core/sliding.py`

Replace upstream's `apply_chunk_torch` (`utils.py`) with a numpy implementation using `numpy.lib.stride_tricks.sliding_window_view`. Preserve the `np.pad(mode='edge')` padding step that upstream applies before unfolding. Output shape and dtype must match upstream's function on a known input array. The `func` callable contract changes from torch to numpy operations; document the new signature.

### 2c: Outline expansion

File: `blender_pixeloe/core/outline.py`

Direct port of upstream `legacy/outline.py`. Two functions: `expansion_weight` and `outline_expansion`.

`cv2.erode` / `cv2.dilate` translate to either `scipy.ndimage.grey_erosion`/`grey_dilation` or a custom numpy implementation using `maximum_filter`/`minimum_filter`. **The iteration count matters.** Upstream calls erode/dilate with `iterations=N`, which applies the 3x3 kernel N times, producing a different result on color images than a single `(2N+1)x(2N+1)` erode. Preserve the iterative semantics. Preserve the exact erode-dilate-erode sequence in the closing/opening cleanup with the asymmetric iteration counts (`erode N`, `dilate 2N`, `erode N`).

The `kernel_smoothing` and `kernel_expansion` arrays are different shapes; do not collapse them into a single kernel.

### 2d: Contrast-based downscale

File: `blender_pixeloe/core/downscale_contrast.py`

Direct port of `legacy/downscale/contrast_based.py`. Uses LAB conversion (from 2a) and sliding window (from 2b). The `find_pixel` function logic translates literally from torch to numpy.

### 2e: K-centroid downscale

File: `blender_pixeloe/core/downscale_kcentroid.py`

Direct port of `legacy/downscale/k_centroid.py`. Upstream already uses PIL for the per-tile quantization (`Image.quantize(method=1, kmeans=centroids)`), so this port is mostly removing the cv2 wrapping and operating on PIL Images directly. This is the simplest of the algorithm files.

### 2f: Color matching (wavelet colorfix)

File: `blender_pixeloe/core/color_match.py`

Direct port of `legacy/color.py`'s `match_color`, `wavelet_colorfix`, `wavelet_decomposition`, `wavelet_blur`. Replace `cv2.GaussianBlur` with `PIL.ImageFilter.GaussianBlur` applied to PIL Image conversions, or with a numpy convolution. The wavelet decomposition logic is purely arithmetic and translates without modification.

### 2g: K-means quantization

File: `blender_pixeloe/core/quantize.py`

Two paths from upstream:

- Non-weighted: `cv2.kmeans` → use `PIL.Image.quantize(colors=N, method=1, kmeans=N)`. Method=1 is fast octree, the `kmeans` parameter does an extra k-means refinement pass.
- Weighted: numpy port of upstream's repeat-based weighted k-means. Upstream creates a flattened pixel array with each pixel repeated `weight[i,j]` times, then runs cv2.kmeans on the expanded array. The repeat-array memory blowup is fine for typical sizes (a 256-pixel-wide output with weights up to 64 means ~4M pixels at most). Document the upper bound. Seed the random initialization for reproducibility.

The `maxcover` method also exists in upstream; port it as a thin wrapper around PIL's quantize for completeness.

### 2h: Top-level orchestrator

File: `blender_pixeloe/core/pixelize.py`

Direct port of `legacy/pixelize.py`'s `pixelize` function. Wires together outline expansion, downscaling, color matching, quantization, and final upscale. The control flow is straightforward; the value is in matching upstream's exact ordering and conditional branches.

After this file ships, the harness should show low diff against upstream on test images. Threshold TBD by running it; mean L1 of single digits per channel on uint8 images is realistic, perfect zero is not (cv2 vs numpy LAB has tiny rounding drift).

## Phase 3: Blender addon shell

### 3a: Addon structure

File: `blender_pixeloe/__init__.py`

Standard Blender addon entry point with `bl_info`, `register()`, `unregister()`. Target Blender `<blender-version>`+. If targeting 4.2+, also include the `blender_manifest.toml` for the extensions system.

### 3b: I/O helpers

File: `blender_pixeloe/image_io.py`

Two functions, the highest-fragility code in the addon:

```python
def image_to_array(image: bpy.types.Image) -> np.ndarray:
    """Returns HxWx3 uint8 array, RGB, top-down, sRGB display-space."""

def array_to_image(arr: np.ndarray, name: str, overwrite: bool = True) -> bpy.types.Image:
    """Creates or updates a Blender Image datablock from an HxWx3 uint8 array."""
```

These handle: vertical flip (Blender stores bottom-up, numpy works top-down), alpha drop and re-attach, scene-linear ↔ sRGB conversion, dtype conversion (Blender pixels are float 0-1, algorithm wants uint8 0-255).

Color space: in Blender 2.8+, `image.pixels` returns scene-linear floats regardless of the source image's color space tag. The PixelOE algorithm operates on display-space (gamma-encoded) values; reading linear values produces incorrect contrast statistics. The helpers must do the linear → sRGB → linear roundtrip.

`overwrite=True` means: if an Image with `name` exists, replace its pixels rather than create `name.001`. This is the iteration-friendly default.

Unit tests for these helpers are mandatory. Round-trip a known checkerboard image (created via numpy) through `array_to_image` then `image_to_array` and verify pixel-perfect recovery. Test with sRGB and Non-Color colorspace tags.

### 3c: Operator

File: `blender_pixeloe/operators.py`

Single operator class: `PIXELOE_OT_pixelize_image`.

Reads `context.space_data.image` when invoked from the Image Editor. Settings are operator properties: `target_size`, `patch_size`, `thickness`, `mode` (enum: `contrast`, `k-centroid`), `colors` (int, 0 = no quantization), `create_new` (bool).

Output naming: `f"{source.name}_pixel"`. Re-running overwrites unless `create_new=True`.

The operator runs synchronously in v1 (Blender freezes during execution). A typical 1080p render takes a few seconds in numpy; acceptable for v1, modal version is a v2 concern.

### 3d: Panel

File: `blender_pixeloe/panels.py`

N-panel in the Image Editor sidebar with a "PixelOE" tab. Layout: image picker (or label showing active image), settings sliders, run button. Standard Blender UI conventions.

## Phase 4: Polish

Out of scope for v1, listed here so they're explicitly deferred:

- Render Result handling (v1 documents the limitation; users save the render to file and load it)
- Animation/sequence pixelization
- Custom palette import (load a PNG palette, force quantization to those colors)
- Real-time viewport preview
- GPU/GLSL implementation
- Compositor node integration
- Render handler integration
- Modal operator with progress bar

## Fragility hotspots

The list of specific traps to watch for. Each has been identified during design analysis and should not need rediscovery.

1. **`image.pixels` is bottom-up RGBA scene-linear float.** The I/O helpers handle vertical flip, alpha drop/restore, sRGB/linear conversion, and dtype. Wrong vertical orientation is a visible bug. Wrong color space is a subtle bug that only shows on dark or saturated images. Wrong alpha handling looks fine in the editor and breaks downstream.

2. **LAB conversion drift from cv2 produces wrong weight maps.** cv2 uses specific D65 sRGB→XYZ→LAB constants. Pin numerics. Validate against `cv2.cvtColor` output in the harness. Drift here propagates into outline expansion which propagates into the entire pipeline.

3. **Sliding window without `np.pad(mode='edge')` silently changes output.** Either the output shape shrinks (if no pad) or edges appear darker (if zero pad). Upstream pads explicitly; the port must too.

4. **`cv2.erode` with iterations is not equivalent to a single larger kernel** for color images. Each iteration takes per-channel min independently. Iterate the same way upstream does. The temptation to "optimize" by using one larger kernel will silently change output.

5. **K-means non-determinism.** Seed every random source. Reproducible runs are mandatory because the harness diff is meaningless if the same input produces different outputs across runs.

6. **Render Result is a special datablock.** Its `pixels` access is unreliable across Blender versions. v1 does not handle it; the operator should detect and produce a clear user error message directing them to save the render to file first.

7. **Float precision and uint8 casts.** Upstream casts to uint8 between morphological steps. Mimicking this preserves the exact rounding behavior that downstream operations depend on. Don't keep everything in float for "precision" reasons; that diverges from upstream.

## Acceptance criteria for v1 ship

- Comparison harness shows mean per-pixel L1 diff under a threshold determined empirically (likely single digits on uint8) on the three test images at default settings, in `contrast` mode.
- Visual side-by-side indistinguishable to a reasonable observer on the three test images.
- I/O helpers round-trip a checkerboard image with zero loss in unit tests.
- Addon installs as a single zip file in Blender `<blender-version>`+ with no dependency installation step.
- Operator works on the active image in the Image Editor and produces correctly named output.
- README documents installation, attribution to KohakuBlueleaf and upstream PixelOE, and current limitations (no Render Result, no animation, no GPU).

## Out of scope

Do not expand into these areas without explicit discussion. Each is a deliberate v1 cut.

- GPU/GLSL implementation
- Multiple operators (only one for v1)
- Animation or sequence support
- Compositor node integration
- Render handler integration
- Custom UI for palette editing
- Modal/async operator execution
- Bicubic and nearest downscale modes (cut from upstream's mode list; only contrast and k-centroid in v1)
- Saturation and contrast adjustments (cut; users do this in Blender's color management)

## References

- Upstream PixelOE: https://github.com/KohakuBlueleaf/PixelOE
- Upstream legacy implementation: read from `tests/upstream/src/pixeloe/legacy/` after Phase 0 setup
- Astropulse k-centroid (origin of the k-centroid algorithm): https://github.com/Astropulse/pixeldetector
- Blender Image API: https://docs.blender.org/api/current/bpy.types.Image.html
- Blender extensions / manifest format: https://docs.blender.org/manual/en/latest/advanced/extensions/getting_started.html
- skimage.color.rgb2lab algorithm reference: https://scikit-image.org/docs/stable/api/skimage.color.html#skimage.color.rgb2lab

## Attribution

Algorithm by Shih-Ying Yeh (KohakuBlueleaf). This project is a port and integration; the README must credit upstream prominently and link the original repository. License must remain Apache 2.0 to match upstream.
