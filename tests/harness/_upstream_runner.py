"""Runs upstream PixelOE inside its own venv.

Two modes, controlled by which output flag is set:

    --output PATH
        End-to-end mode used by `compare.py`. Runs `pixeloe.legacy.pixelize`
        on the input and writes the final image as a PNG.

    --dump_dir DIR
        Per-stage mode used by `compare_stage.py`. Reproduces upstream's
        pipeline step-by-step and writes each intermediate result to DIR
        as `.npy` files. All arrays are stored in RGB byte order (regardless
        of upstream's BGR convention) so port-side comparison is direct.

Both flags can be set together; the runner does the dump and also writes the
final PNG for compare.py-style visualization. Settings come in as a JSON
string so the parent process never needs to know upstream's exact signature.

Files written by --dump_dir:
    post_resize.npy                  RGB uint8, after the initial cv2.resize
    lab.npy                          uint8 LAB byte layout (cv2 BGR2LAB of post_resize)
    expansion_weight.npy             float32 [0,1], upstream's expansion_weight()
    post_outline_rgb.npy             RGB uint8, after outline_expansion (or post_resize if thickness=0)
    post_outline_weight.npy          float32 [0,1] HxW (only if thickness>0)
    post_color_match.npy             RGB uint8, after match_color (or post_outline_rgb if color_matching=False)
    post_downscale.npy               RGB uint8, after downscale_mode[mode]
    post_quantize.npy                RGB uint8, after color_quant (or post_downscale if colors<=0)
    post_final.npy                   RGB uint8, end of pipeline
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from pixeloe.legacy.color import color_quant, match_color
from pixeloe.legacy.downscale import downscale_mode
from pixeloe.legacy.outline import expansion_weight, outline_expansion
from pixeloe.legacy.pixelize import pixelize


def _initial_resize_hw(h: int, w: int, target_size: int, patch_size: int) -> tuple[int, int]:
    """Reproduce upstream's pre-pipeline resize math. Returns (W, H) for cv2."""
    ratio = w / h
    org_size = (target_size**2 * patch_size**2 / ratio) ** 0.5
    return (int(org_size * ratio), int(org_size))


def run_with_dumps(rgb_input: np.ndarray, settings: dict, dump_dir: Path) -> np.ndarray:
    """Reproduce upstream's pipeline step-by-step. Returns the final RGB image."""
    mode = settings.get("mode", "contrast")
    target_size = settings.get("target_size", 128)
    patch_size = settings.get("patch_size", 16)
    thickness = settings.get("thickness", 2)
    color_matching = settings.get("color_matching", True)
    pixel_size = settings.get("pixel_size", patch_size)
    colors = settings.get("colors", 0)
    color_quant_method = settings.get("color_quant_method", "kmeans")

    bgr_input = cv2.cvtColor(rgb_input, cv2.COLOR_RGB2BGR)
    h, w = bgr_input.shape[:2]
    bgr_resized = cv2.resize(
        bgr_input, _initial_resize_hw(h, w, target_size, patch_size)
    )
    rgb_resized = cv2.cvtColor(bgr_resized, cv2.COLOR_BGR2RGB)
    np.save(dump_dir / "post_resize.npy", rgb_resized)

    np.save(dump_dir / "lab.npy", cv2.cvtColor(bgr_resized, cv2.COLOR_BGR2LAB))

    weight_only = expansion_weight(
        bgr_resized,
        k=patch_size,
        stride=(patch_size // 4) * 2,
        avg_scale=9,
        dist_scale=4,
    )
    np.save(dump_dir / "expansion_weight.npy", weight_only.astype(np.float32))

    if thickness:
        bgr_outlined, weight_full = outline_expansion(
            bgr_resized, thickness, thickness, patch_size, 9, 4
        )
        np.save(
            dump_dir / "post_outline_weight.npy", weight_full.astype(np.float32)
        )
    else:
        bgr_outlined = bgr_resized
    rgb_outlined = cv2.cvtColor(bgr_outlined, cv2.COLOR_BGR2RGB)
    np.save(dump_dir / "post_outline_rgb.npy", rgb_outlined)

    if color_matching:
        bgr_matched = match_color(bgr_outlined, bgr_resized)
    else:
        bgr_matched = bgr_outlined
    rgb_matched = cv2.cvtColor(bgr_matched, cv2.COLOR_BGR2RGB)
    np.save(dump_dir / "post_color_match.npy", rgb_matched)

    bgr_sm = downscale_mode[mode](bgr_matched, target_size)
    rgb_sm = cv2.cvtColor(bgr_sm, cv2.COLOR_BGR2RGB)
    np.save(dump_dir / "post_downscale.npy", rgb_sm)

    if colors > 0:
        bgr_quant = color_quant(bgr_sm, colors, method=color_quant_method)
        rgb_quant = cv2.cvtColor(bgr_quant, cv2.COLOR_BGR2RGB)
    else:
        rgb_quant = rgb_sm
    np.save(dump_dir / "post_quantize.npy", rgb_quant)

    sm_h, sm_w = rgb_quant.shape[:2]
    rgb_final = np.array(
        Image.fromarray(rgb_quant).resize(
            (sm_w * pixel_size, sm_h * pixel_size), Image.NEAREST
        )
    )
    np.save(dump_dir / "post_final.npy", rgb_final)

    return rgb_final


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dump_dir", type=Path)
    parser.add_argument("--settings", type=str, required=True, help="JSON dict")
    args = parser.parse_args()

    if args.output is None and args.dump_dir is None:
        sys.exit("at least one of --output / --dump_dir is required")

    settings = json.loads(args.settings)
    rgb = np.array(Image.open(args.input).convert("RGB"))

    if args.dump_dir is not None:
        args.dump_dir.mkdir(parents=True, exist_ok=True)
        rgb_final = run_with_dumps(rgb, settings, args.dump_dir)
        if args.output is not None:
            Image.fromarray(rgb_final).save(args.output)
    else:
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        out_bgr = pixelize(bgr, **settings)
        out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
        Image.fromarray(out_rgb).save(args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
