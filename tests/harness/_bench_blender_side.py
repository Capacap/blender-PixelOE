"""In-Blender half of the I/O benchmark. Run via:

    blender --background --python tests/harness/_bench_blender_side.py \\
        -- <input_image> <out_npy> [runs]

Times image_to_array, pixelize, and array_to_image on a fixed input. Saves
the post-image_to_array numpy buffer to <out_npy> so the orchestrator can
run pixelize on identical bytes outside Blender. Emits one JSON line
prefixed `BENCH_RESULT_JSON:` to stdout.

scipy is loaded from the addon's bundled wheel (matched by ABI tag) when
not already importable, mirroring what Blender's extension loader does
without requiring the addon to be installed.
"""
from __future__ import annotations

import json
import platform
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]


def _ensure_scipy_on_path() -> None:
    try:
        import scipy  # noqa: F401
        return
    except ImportError:
        pass

    abi = f"cp{sys.version_info.major}{sys.version_info.minor}"
    wheels_dir = REPO_ROOT / "blender_pixeloe" / "wheels"

    plat_filters: list[str] = []
    if sys.platform == "linux":
        plat_filters = ["manylinux", "x86_64"]
    elif sys.platform == "win32":
        plat_filters = ["win_amd64"]
    elif sys.platform == "darwin":
        plat_filters = ["macosx", platform.machine()]
    else:
        raise RuntimeError(f"unsupported platform: {sys.platform}")

    candidates = [
        p for p in wheels_dir.glob("scipy-*.whl")
        if abi in p.name and all(f in p.name for f in plat_filters)
    ]
    if not candidates:
        raise RuntimeError(
            f"no scipy wheel matching abi={abi} platform={sys.platform} in {wheels_dir}"
        )

    extracted = tempfile.mkdtemp(prefix="bench_scipy_")
    zipfile.ZipFile(candidates[0]).extractall(extracted)
    sys.path.insert(0, extracted)


_ensure_scipy_on_path()
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

import bpy  # noqa: E402
import scipy  # noqa: E402

from _stage_tracer import install_stage_tracers  # noqa: E402
from blender_pixeloe.core import pixelize  # noqa: E402
from blender_pixeloe.image_io import array_to_image, image_to_array  # noqa: E402


def main() -> int:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    if len(argv) < 2:
        print("usage: ... -- <input_image> <out_npy> [runs]", file=sys.stderr)
        return 2
    input_path = argv[0]
    out_npy = argv[1]
    runs = int(argv[2]) if len(argv) > 2 else 3

    settings = dict(
        target_size=128, patch_size=16, thickness=2, mode="contrast", colors=0
    )

    img = bpy.data.images.load(input_path)

    rgb_warm = image_to_array(img)
    out_warm = pixelize(rgb_warm, **settings)
    array_to_image(out_warm, "_bench_warm", overwrite=True)

    times_in: list[float] = []
    rgb = rgb_warm
    for _ in range(runs):
        t0 = time.perf_counter()
        rgb = image_to_array(img)
        times_in.append(time.perf_counter() - t0)

    stage_timings = install_stage_tracers()
    times_pix: list[float] = []
    out_rgb = out_warm
    for _ in range(runs):
        t0 = time.perf_counter()
        out_rgb = pixelize(rgb, **settings)
        times_pix.append(time.perf_counter() - t0)
    stage_snapshot = {k: list(v) for k, v in stage_timings.items()}

    times_out: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        array_to_image(out_rgb, "_bench_out", overwrite=True)
        times_out.append(time.perf_counter() - t0)

    np.save(out_npy, rgb)

    result = {
        "context": "blender",
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "input_shape": list(rgb.shape),
        "settings": settings,
        "image_to_array": times_in,
        "pixelize": times_pix,
        "array_to_image": times_out,
        "stages": stage_snapshot,
    }
    print("BENCH_RESULT_JSON:", json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
