"""Shared stage tracer for pixelize benchmarks.

Wraps the stage functions in `blender_pixeloe.core.pixelize`'s module
namespace so each call appends its duration to a shared dict. Pixelize
imports stage functions at module load, so we patch the references in
pixelize's namespace, not the source modules. The package re-exports
`pixelize` the function from `blender_pixeloe.core`, which shadows the
submodule attribute, so we pull the module object out of `sys.modules`
after import.
"""
from __future__ import annotations

import sys
import time

STAGE_FUNCTIONS = (
    "outline_expansion",
    "expansion_weight",
    "match_color",
    "contrast_based_downscale",
    "k_centroid_downscale",
    "color_quant",
    "_resize",
)


def install_stage_tracers() -> dict[str, list[float]]:
    import blender_pixeloe.core.pixelize  # noqa: F401
    pmod = sys.modules["blender_pixeloe.core.pixelize"]

    timings: dict[str, list[float]] = {}
    for name in STAGE_FUNCTIONS:
        original = getattr(pmod, name)

        def make_wrapped(label: str, fn):
            def wrapped(*a, **kw):
                t0 = time.perf_counter()
                try:
                    return fn(*a, **kw)
                finally:
                    timings.setdefault(label, []).append(time.perf_counter() - t0)
            return wrapped

        setattr(pmod, name, make_wrapped(name, original))
    return timings
