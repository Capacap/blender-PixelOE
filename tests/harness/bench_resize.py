"""Isolated comparison: PIL bilinear vs scipy.ndimage.zoom.

Times the two bilinear downscales that pixelize performs on snow-leopard at
default settings, and reports a quality delta (max abs uint8 difference)
between the two methods so we can judge whether scipy is a viable swap-in.

Run from repo root:
    uv run python tests/harness/bench_resize.py
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import PIL
import scipy
from PIL import Image
from scipy import ndimage

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPO_ROOT / "tests" / "images" / "snow-leopard.webp"


def pil_resize(rgb: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    h, w = target_hw
    return np.array(Image.fromarray(rgb).resize((w, h), Image.BILINEAR))


def scipy_zoom(rgb: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    h, w = target_hw
    src_h, src_w = rgb.shape[:2]
    return ndimage.zoom(rgb, (h / src_h, w / src_w, 1.0), order=1)


def time_runs(fn, src, target_hw, runs: int, warmup: int = 1) -> list[float]:
    for _ in range(warmup):
        fn(src, target_hw)
    out: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn(src, target_hw)
        out.append(time.perf_counter() - t0)
    return out


def fmt_ms(s: float) -> str:
    return f"{s * 1000:.1f}ms"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--runs", type=int, default=10)
    args = parser.parse_args()

    rgb = np.array(Image.open(args.input).convert("RGB"))
    print(f"input  : {args.input.name} {rgb.shape[1]}x{rgb.shape[0]}")
    print(f"Pillow : {PIL.__version__}")
    print(f"scipy  : {scipy.__version__}")
    print(f"runs   : {args.runs} (median reported)")
    print()

    # The two bilinear resizes pixelize performs at default settings on this
    # input: input -> org_hw, then org -> small (within match_color).
    org_hw = (1672, 2508)
    small_hw = (104, 156)
    org_rgb = pil_resize(rgb, org_hw)

    cases = [
        ("input -> org_hw", rgb, org_hw),
        ("org   -> small ", org_rgb, small_hw),
    ]

    print(f"{'case':<22}{'PIL bilinear':>14}{'scipy zoom':>14}{'speedup':>12}")
    print("-" * 62)
    for label, src, target_hw in cases:
        pil_med = statistics.median(time_runs(pil_resize, src, target_hw, args.runs))
        scipy_med = statistics.median(time_runs(scipy_zoom, src, target_hw, args.runs))
        speedup = pil_med / scipy_med if scipy_med > 0 else float("inf")
        marker = "PIL faster" if speedup < 1 else "scipy faster"
        print(
            f"  {label:<20}{fmt_ms(pil_med):>14}{fmt_ms(scipy_med):>14}"
            f"{f'{speedup:.2f}x ({marker})':>12}"
        )

    print()
    print("Quality check on the input -> org_hw case:")
    pil_out = pil_resize(rgb, org_hw)
    scipy_out = scipy_zoom(rgb, org_hw)
    diff = np.abs(pil_out.astype(np.int16) - scipy_out.astype(np.int16))
    print(f"  PIL output shape   : {pil_out.shape}")
    print(f"  scipy output shape : {scipy_out.shape}")
    print(f"  max  abs diff      : {diff.max()}")
    print(f"  mean abs diff      : {diff.mean():.2f}")
    print(f"  fraction > 5 LSB   : {(diff > 5).mean() * 100:.2f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
