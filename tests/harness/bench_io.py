"""Compare per-phase pixelize timing inside Blender vs. outside Blender.

The Blender side runs in a `blender --background` subprocess, times
image_to_array + pixelize + array_to_image, and dumps the input numpy
buffer it produced. The local side runs pixelize on those exact bytes
in the current uv environment so we compare like-for-like.

Usage:
    uv run python tests/harness/bench_io.py [--input <path>] [--runs N]
"""
from __future__ import annotations

import argparse
import json
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
BLENDER_SIDE = REPO_ROOT / "tests" / "harness" / "_bench_blender_side.py"
DEFAULT_INPUT = REPO_ROOT / "tests" / "images" / "snow-leopard.webp"

sys.path.insert(0, str(Path(__file__).parent))
from _stage_tracer import install_stage_tracers  # noqa: E402


def _fmt_seconds(s: float) -> str:
    return f"{s:.3f}s" if s >= 1.0 else f"{s * 1000:.1f}ms"


def _run_blender_side(input_path: Path, out_npy: Path, runs: int) -> dict:
    blender = shutil.which("blender")
    if blender is None:
        raise RuntimeError("blender not on PATH")
    cmd = [
        blender, "--background",
        "--python", str(BLENDER_SIDE),
        "--",
        str(input_path), str(out_npy), str(runs),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise RuntimeError(f"blender side exited {proc.returncode}")

    for line in proc.stdout.splitlines():
        if line.startswith("BENCH_RESULT_JSON:"):
            return json.loads(line[len("BENCH_RESULT_JSON:"):].strip())
    sys.stderr.write(proc.stdout)
    raise RuntimeError("BENCH_RESULT_JSON not found in Blender stdout")


def _run_local(rgb: np.ndarray, settings: dict, runs: int) -> dict:
    import scipy

    from blender_pixeloe.core import pixelize

    pixelize(rgb, **settings)

    stage_timings = install_stage_tracers()
    times: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        pixelize(rgb, **settings)
        times.append(time.perf_counter() - t0)
    stage_snapshot = {k: list(v) for k, v in stage_timings.items()}

    return {
        "context": "local",
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pixelize": times,
        "stages": stage_snapshot,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()

    if not args.input.exists():
        sys.exit(f"input not found: {args.input}")

    print(f"input  : {args.input}")
    print(f"runs   : {args.runs}")
    print()

    with tempfile.TemporaryDirectory() as td:
        shared_npy = Path(td) / "input.npy"
        print("running Blender side (this includes Blender startup)...")
        b = _run_blender_side(args.input, shared_npy, args.runs)
        rgb = np.load(shared_npy)

    print(f"  python {b['python']}, numpy {b['numpy']}, scipy {b['scipy']}")
    print(f"  input shape {tuple(b['input_shape'])}")
    print(f"  settings    {b['settings']}")
    print()

    print("running local side...")
    local = _run_local(rgb, b["settings"], args.runs)
    print(f"  python {local['python']}, numpy {local['numpy']}, scipy {local['scipy']}")
    print()

    def med(xs: list[float]) -> float:
        return statistics.median(xs)

    in_b = med(b["image_to_array"])
    pix_b = med(b["pixelize"])
    out_b = med(b["array_to_image"])
    pix_l = med(local["pixelize"])
    total_b = in_b + pix_b + out_b

    print("=" * 60)
    print(f"{'phase':<22}{'blender':>14}{'local':>14}")
    print("-" * 60)
    print(f"{'image_to_array':<22}{_fmt_seconds(in_b):>14}{'-':>14}")
    print(f"{'pixelize':<22}{_fmt_seconds(pix_b):>14}{_fmt_seconds(pix_l):>14}")
    print(f"{'array_to_image':<22}{_fmt_seconds(out_b):>14}{'-':>14}")
    print("-" * 60)
    print(f"{'total':<22}{_fmt_seconds(total_b):>14}{_fmt_seconds(pix_l):>14}")
    print()

    delta = pix_b - pix_l
    delta_pct = (delta / pix_l * 100) if pix_l > 0 else 0.0
    direction = "blender slower" if delta > 0 else "blender faster"
    boundary = in_b + out_b
    bnd_pct = (boundary / total_b * 100) if total_b > 0 else 0.0

    print(f"pixelize delta   : {_fmt_seconds(delta)} ({delta_pct:+.1f}%, {direction})")
    print(f"I/O boundary cost: {_fmt_seconds(boundary)} ({bnd_pct:.1f}% of blender total)")

    stages_b = b.get("stages", {})
    stages_l = local.get("stages", {})
    stage_names = sorted(set(stages_b) | set(stages_l))
    if stage_names:
        print()
        print("=" * 60)
        print("pixelize stage breakdown (per-run, sum of all calls / runs)")
        print(f"{'stage':<28}{'blender':>12}{'local':>12}{'delta':>10}")
        print("-" * 60)
        for name in stage_names:
            sb = stages_b.get(name, [])
            sl = stages_l.get(name, [])
            per_b = (sum(sb) / args.runs) if sb else 0.0
            per_l = (sum(sl) / args.runs) if sl else 0.0
            calls_b = (len(sb) // args.runs) if sb else 0
            calls_l = (len(sl) // args.runs) if sl else 0
            calls_tag = f"{calls_b}x" if calls_b == calls_l else f"{calls_b}/{calls_l}x"
            d = per_b - per_l
            d_str = f"{d * 1000:+.0f}ms" if abs(d) >= 0.001 else "  ~0ms"
            label = f"{name} ({calls_tag})"
            print(
                f"  {label:<26}{_fmt_seconds(per_b):>12}{_fmt_seconds(per_l):>12}"
                f"{d_str:>10}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
