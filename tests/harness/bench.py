"""End-to-end benchmark for the pixelize port.

Times `blender_pixeloe.core.pixelize` across N runs and reports median /
min / max wall-clock plus a peak-memory sample via tracemalloc. With
`--stages`, monkey-patches each stage function in pixelize's namespace
so per-stage durations and their share of the total are emitted.

Memory and timing are measured in separate phases so tracemalloc's
overhead doesn't pollute the wall-clock numbers.

Usage:
    uv run python tests/harness/bench.py tests/images/snow-leopard.webp \\
        --target_size 256 --patch_size 8 --thickness 3 --colors 32 --stages
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from _stage_tracer import STAGE_FUNCTIONS, install_stage_tracers  # noqa: E402, F401


def _fmt_seconds(s: float) -> str:
    return f"{s:.3f}s" if s >= 1.0 else f"{s * 1000:.1f}ms"


def _fmt_bytes(b: int) -> str:
    f = float(b)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if f < 1024 or unit == "GiB":
            return f"{f:.1f}{unit}"
        f /= 1024
    return f"{f:.1f}GiB"


def time_runs(rgb: np.ndarray, settings: dict, runs: int) -> list[float]:
    from blender_pixeloe.core import pixelize

    out: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        pixelize(rgb, **settings)
        out.append(time.perf_counter() - t0)
    return out


def measure_peak_memory(rgb: np.ndarray, settings: dict) -> int:
    from blender_pixeloe.core import pixelize

    tracemalloc.start()
    pixelize(rgb, **settings)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("input", type=Path)
    parser.add_argument("--target_size", type=int, default=256)
    parser.add_argument("--patch_size", type=int, default=8)
    parser.add_argument("--thickness", type=int, default=3)
    parser.add_argument("--mode", choices=["contrast", "k-centroid"], default="contrast")
    parser.add_argument("--colors", type=int, default=32)
    parser.add_argument(
        "--color_quant_method", choices=["kmeans", "maxcover"], default="kmeans"
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--stages", action="store_true")
    args = parser.parse_args()

    if not args.input.exists():
        sys.exit(f"Input not found: {args.input}")

    rgb = np.array(Image.open(args.input).convert("RGB"))
    settings = {
        "target_size": args.target_size,
        "patch_size": args.patch_size,
        "thickness": args.thickness,
        "mode": args.mode,
        "colors": args.colors,
        "color_quant_method": args.color_quant_method,
    }

    print(f"input    : {args.input} ({rgb.shape[1]}x{rgb.shape[0]})")
    print(f"settings : {settings}")
    print(f"warmup   : {args.warmup}, runs: {args.runs}, stages: {args.stages}")

    stage_timings = install_stage_tracers() if args.stages else None

    time_runs(rgb, settings, args.warmup)
    if stage_timings is not None:
        for k in stage_timings:
            stage_timings[k].clear()

    totals = time_runs(rgb, settings, args.runs)

    if stage_timings is not None:
        stage_snapshot = {k: list(v) for k, v in stage_timings.items()}
    peak = measure_peak_memory(rgb, settings)

    print("\nwall-clock:")
    print(f"  median = {_fmt_seconds(statistics.median(totals))}")
    print(f"  min    = {_fmt_seconds(min(totals))}")
    print(f"  max    = {_fmt_seconds(max(totals))}")

    print("\npeak memory (tracemalloc, single run):")
    print(f"  peak   = {_fmt_bytes(peak)}")

    if stage_timings is not None:
        median_total = statistics.median(totals)
        print(f"\nper-stage (sum across {args.runs} runs / {args.runs} = per-run):")
        rows = []
        for name, samples in stage_snapshot.items():
            if not samples:
                continue
            per_run = sum(samples) / args.runs
            calls_per_run = len(samples) // args.runs
            rows.append((per_run, name, calls_per_run))
        rows.sort(reverse=True)
        name_w = max(len(r[1]) for r in rows)
        accounted = 0.0
        for per_run, name, calls in rows:
            pct = (per_run / median_total) * 100 if median_total > 0 else 0.0
            accounted += per_run
            tag = f"{calls}x" if calls != 1 else "  "
            print(
                f"  {name:<{name_w}}  {_fmt_seconds(per_run):>8}  "
                f"({pct:5.1f}%)  {tag}"
            )
        remainder = median_total - accounted
        rem_pct = (remainder / median_total) * 100 if median_total > 0 else 0.0
        print(f"  {'(other)':<{name_w}}  {_fmt_seconds(remainder):>8}  ({rem_pct:5.1f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
