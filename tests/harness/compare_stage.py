"""Per-stage validation harness for the port vs upstream PixelOE.

Subprocesses upstream's per-stage runner once to dump every pipeline
intermediate as `.npy` files. Then runs the port's equivalent for the
requested stage on the matching upstream-derived input, diffs the result
against upstream's intermediate, and prints L1 statistics. Optionally
writes a per-stage visualization PNG.

Usage:
    uv run python tests/harness/compare_stage.py tests/images/snow-leopard.webp \\
        --stage outline --target_size 256 --patch_size 8

Stages:
    lab               cv2.cvtColor(BGR2LAB) of post_resize image
    expansion_weight  upstream.expansion_weight on post_resize
    outline           upstream.outline_expansion on post_resize (rgb + weight)
    color_match       upstream.match_color(outlined, resized)
    downscale         upstream.downscale_mode[mode] on post_color_match
    quantize          upstream.color_quant on post_downscale (requires --colors)
    final             full pixelize end-to-end (sanity check)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_PYTHON = PROJECT_ROOT / "tests" / "upstream" / ".venv" / "bin" / "python"
UPSTREAM_RUNNER = Path(__file__).parent / "_upstream_runner.py"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "tests" / "harness" / "output"

STAGES = [
    "lab",
    "expansion_weight",
    "outline",
    "color_match",
    "downscale",
    "quantize",
    "final",
]


def run_upstream_dumps(input_path: Path, settings: dict, dump_dir: Path) -> None:
    if not UPSTREAM_PYTHON.exists():
        sys.exit(
            f"Upstream venv not found at {UPSTREAM_PYTHON}.\n"
            f"Run ./setup_upstream.sh from the project root first."
        )
    cmd = [
        str(UPSTREAM_PYTHON),
        str(UPSTREAM_RUNNER),
        "--input", str(input_path),
        "--dump_dir", str(dump_dir),
        "--settings", json.dumps(settings),
    ]
    subprocess.run(cmd, check=True)


def run_port_stage(stage: str, dump_dir: Path, settings: dict, input_path: Path):
    """Run the port's equivalent of `stage` and return its output array(s)."""
    target_size = settings["target_size"]
    patch_size = settings["patch_size"]
    thickness = settings.get("thickness", 2)
    mode = settings.get("mode", "contrast")
    colors = settings.get("colors", 0)
    color_quant_method = settings.get("color_quant_method", "kmeans")

    if stage == "lab":
        from blender_pixeloe.core.colorspace import rgb_to_lab

        return rgb_to_lab(np.load(dump_dir / "post_resize.npy"))

    if stage == "expansion_weight":
        from blender_pixeloe.core.outline import expansion_weight

        return expansion_weight(
            np.load(dump_dir / "post_resize.npy"),
            k=patch_size,
            stride=(patch_size // 4) * 2,
            avg_scale=9,
            dist_scale=4,
        )

    if stage == "outline":
        from blender_pixeloe.core.outline import outline_expansion

        return outline_expansion(
            np.load(dump_dir / "post_resize.npy"),
            erode=thickness,
            dilate=thickness,
            k=patch_size,
            avg_scale=9,
            dist_scale=4,
        )

    if stage == "color_match":
        from blender_pixeloe.core.color_match import match_color

        source = np.load(dump_dir / "post_outline_rgb.npy")
        target = np.load(dump_dir / "post_resize.npy")
        return match_color(source, target, level=5)

    if stage == "downscale":
        pre = np.load(dump_dir / "post_color_match.npy")
        if mode == "contrast":
            from blender_pixeloe.core.downscale_contrast import contrast_based_downscale

            return contrast_based_downscale(pre, target_size)
        from blender_pixeloe.core.downscale_kcentroid import k_centroid_downscale

        return k_centroid_downscale(pre, target_size)

    if stage == "quantize":
        if colors <= 0:
            sys.exit("--stage quantize requires --colors > 0")
        from blender_pixeloe.core.quantize import color_quant

        repeats = max(1, int((patch_size * colors) ** 0.5))
        return color_quant(
            np.load(dump_dir / "post_downscale.npy"),
            colors,
            None,
            repeats,
            color_quant_method,
        )

    if stage == "final":
        from blender_pixeloe.core import pixelize

        rgb = np.array(Image.open(input_path).convert("RGB"))
        return pixelize(rgb, **settings)

    sys.exit(f"unknown stage: {stage!r}")


def load_upstream_post(stage: str, dump_dir: Path):
    if stage == "lab":
        return np.load(dump_dir / "lab.npy")
    if stage == "expansion_weight":
        return np.load(dump_dir / "expansion_weight.npy")
    if stage == "outline":
        return (
            np.load(dump_dir / "post_outline_rgb.npy"),
            np.load(dump_dir / "post_outline_weight.npy"),
        )
    if stage == "color_match":
        return np.load(dump_dir / "post_color_match.npy")
    if stage == "downscale":
        return np.load(dump_dir / "post_downscale.npy")
    if stage == "quantize":
        return np.load(dump_dir / "post_quantize.npy")
    if stage == "final":
        return np.load(dump_dir / "post_final.npy")
    sys.exit(f"unknown stage: {stage!r}")


def l1_stats(upstream: np.ndarray, port: np.ndarray, threshold: float) -> dict | None:
    u = upstream.astype(np.float32)
    p = port.astype(np.float32)
    if u.shape != p.shape:
        return None
    diff = np.abs(u - p)
    per_pixel = diff.mean(axis=-1) if diff.ndim == 3 else diff
    return {
        "mean": float(diff.mean()),
        "max": float(diff.max()),
        "over_pct": float((per_pixel > threshold).mean() * 100.0),
    }


def to_viz_uint8(arr: np.ndarray) -> np.ndarray:
    """Convert any array to HxWx3 uint8 for visualization."""
    if np.issubdtype(arr.dtype, np.floating):
        a = arr - arr.min()
        if a.max() > 0:
            a = a / a.max() * 255.0
        a = a.astype(np.uint8)
    else:
        a = arr.astype(np.uint8)
    if a.ndim == 2:
        return np.stack([a, a, a], axis=-1)
    return a


def diff_heat(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    diff = np.abs(a.astype(np.float32) - b.astype(np.float32))
    if diff.ndim == 3:
        diff = diff.mean(axis=-1)
    norm = np.clip(diff * 4.0, 0, 255).astype(np.uint8)
    heat = np.zeros((*norm.shape, 3), dtype=np.uint8)
    heat[..., 0] = norm
    heat[..., 1] = norm // 3
    return heat


def label_panel(arr: np.ndarray, label: str) -> np.ndarray:
    img = Image.fromarray(arr).convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    draw.text((9, 9), label, fill=(0, 0, 0))
    draw.text((8, 8), label, fill=(255, 255, 255))
    return np.array(img)


def compose_2x2(panels: list[np.ndarray], cell_width: int = 512) -> np.ndarray:
    resized = []
    for p in panels:
        h, w = p.shape[:2]
        new_h = max(1, int(round(h * (cell_width / w))))
        resized.append(
            np.array(Image.fromarray(p).resize((cell_width, new_h), Image.BILINEAR))
        )

    def pad_h(a: np.ndarray, target_h: int) -> np.ndarray:
        if a.shape[0] == target_h:
            return a
        pad = np.zeros((target_h - a.shape[0], a.shape[1], a.shape[2]), dtype=a.dtype)
        return np.concatenate([a, pad], axis=0)

    top_h = max(resized[0].shape[0], resized[1].shape[0])
    bot_h = max(resized[2].shape[0], resized[3].shape[0])
    top = np.concatenate([pad_h(resized[0], top_h), pad_h(resized[1], top_h)], axis=1)
    bot = np.concatenate([pad_h(resized[2], bot_h), pad_h(resized[3], bot_h)], axis=1)
    return np.concatenate([top, bot], axis=0)


def report_single(label: str, upstream: np.ndarray, port: np.ndarray, threshold: float):
    print(f"  [{label}] shape={upstream.shape} dtype={upstream.dtype}")
    stats = l1_stats(upstream, port, threshold)
    if stats is None:
        print(f"  [{label}] SHAPE MISMATCH: upstream={upstream.shape} port={port.shape}")
        return None
    print(
        f"  [{label}] mean L1 = {stats['mean']:.3f}  max = {stats['max']:.0f}  "
        f">{threshold} = {stats['over_pct']:.2f}%"
    )
    return stats


def write_viz(stage: str, upstream, port, viz_path: Path):
    """Write a 4-panel viz. For tuple outputs (outline) we show the rgb panel."""
    if isinstance(upstream, tuple):
        u_main, p_main = upstream[0], port[0]
        u_extra, p_extra = upstream[1], port[1]
        composite = compose_2x2(
            [
                label_panel(to_viz_uint8(u_main), f"upstream {stage} rgb"),
                label_panel(to_viz_uint8(p_main), f"port {stage} rgb"),
                label_panel(diff_heat(u_main, p_main), "|rgb diff|"),
                label_panel(diff_heat(u_extra, p_extra), "|weight diff|"),
            ]
        )
    else:
        composite = compose_2x2(
            [
                label_panel(to_viz_uint8(upstream), f"upstream {stage}"),
                label_panel(to_viz_uint8(port), f"port {stage}"),
                label_panel(diff_heat(upstream, port), "|diff|"),
                label_panel(np.zeros_like(to_viz_uint8(upstream)), ""),
            ]
        )
    Image.fromarray(composite).save(viz_path)
    print(f"\nwrote {viz_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("input", type=Path)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--target_size", type=int, default=256)
    parser.add_argument("--patch_size", type=int, default=8)
    parser.add_argument("--thickness", type=int, default=3)
    parser.add_argument("--mode", choices=["contrast", "k-centroid"], default="contrast")
    parser.add_argument("--colors", type=int, default=0)
    parser.add_argument(
        "--color_quant_method", choices=["kmeans", "maxcover"], default="kmeans"
    )
    parser.add_argument("--diff_threshold", type=float, default=2.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if not args.input.exists():
        sys.exit(f"Input image not found: {args.input}")

    settings = {
        "target_size": args.target_size,
        "patch_size": args.patch_size,
        "thickness": args.thickness,
        "mode": args.mode,
        "colors": args.colors,
        "color_quant_method": args.color_quant_method,
    }

    output_path = args.output or (
        DEFAULT_OUTPUT_DIR / f"{args.input.stem}_stage_{args.stage}.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"input    : {args.input}")
    print(f"stage    : {args.stage}")
    print(f"settings : {settings}")

    with tempfile.TemporaryDirectory() as tmp:
        dump_dir = Path(tmp)
        print("\nrunning upstream dumps...")
        run_upstream_dumps(args.input, settings, dump_dir)

        print(f"\nrunning port stage {args.stage!r}...")
        try:
            port_out = run_port_stage(args.stage, dump_dir, settings, args.input)
        except (ImportError, AttributeError, NotImplementedError) as e:
            sys.exit(f"port stage {args.stage!r} not yet implemented: {e}")

        upstream_out = load_upstream_post(args.stage, dump_dir)

        print("\nL1 statistics:")
        if isinstance(upstream_out, tuple):
            for sub_label, u, p in zip(("rgb", "weight"), upstream_out, port_out):
                report_single(sub_label, u, p, args.diff_threshold)
        else:
            report_single(args.stage, upstream_out, port_out, args.diff_threshold)

        write_viz(args.stage, upstream_out, port_out, output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
