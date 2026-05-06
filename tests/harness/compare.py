"""Side-by-side comparison harness for the port vs upstream PixelOE.

Runs both implementations on the same input image and produces a 4-panel PNG
(input, upstream, port, |diff| heatmap) plus L1 statistics. The port runs
in-process; upstream runs in `tests/upstream/.venv` via subprocess so its
torch/cv2/kornia stack stays isolated from the project venv.

Usage:
    uv run python tests/harness/compare.py tests/images/snow-leopard.webp \\
        --target_size 256 --patch_size 8
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


def run_upstream(input_path: Path, settings: dict) -> np.ndarray:
    if not UPSTREAM_PYTHON.exists():
        sys.exit(
            f"Upstream venv not found at {UPSTREAM_PYTHON}.\n"
            f"Run ./setup_upstream.sh from the project root first."
        )
    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "upstream.png"
        cmd = [
            str(UPSTREAM_PYTHON),
            str(UPSTREAM_RUNNER),
            "--input", str(input_path),
            "--output", str(out_path),
            "--settings", json.dumps(settings),
        ]
        subprocess.run(cmd, check=True)
        return np.array(Image.open(out_path).convert("RGB"))


def run_port(input_rgb: np.ndarray, settings: dict) -> np.ndarray:
    from blender_pixeloe.core import pixelize

    return pixelize(input_rgb, **settings)


def resize_rgb(arr: np.ndarray, hw: tuple[int, int], resample=Image.BILINEAR) -> np.ndarray:
    h, w = hw
    return np.array(Image.fromarray(arr).resize((w, h), resample))


def make_diff_heatmap(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    diff = np.abs(a.astype(np.int16) - b.astype(np.int16)).astype(np.uint8)
    mean_l1 = diff.mean(axis=-1)
    norm = np.clip(mean_l1 * 4.0, 0, 255).astype(np.uint8)
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


def compose_2x2(panels: list[np.ndarray], cell_width: int = 768) -> np.ndarray:
    resized = []
    for p in panels:
        h, w = p.shape[:2]
        new_h = max(1, int(round(h * (cell_width / w))))
        resized.append(resize_rgb(p, (new_h, cell_width)))

    def pad_to_height(a: np.ndarray, target_h: int) -> np.ndarray:
        if a.shape[0] == target_h:
            return a
        pad = np.zeros((target_h - a.shape[0], a.shape[1], a.shape[2]), dtype=a.dtype)
        return np.concatenate([a, pad], axis=0)

    top_h = max(resized[0].shape[0], resized[1].shape[0])
    bot_h = max(resized[2].shape[0], resized[3].shape[0])
    top = np.concatenate(
        [pad_to_height(resized[0], top_h), pad_to_height(resized[1], top_h)], axis=1
    )
    bot = np.concatenate(
        [pad_to_height(resized[2], bot_h), pad_to_height(resized[3], bot_h)], axis=1
    )
    return np.concatenate([top, bot], axis=0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("input", type=Path, help="path to input image")
    parser.add_argument("--target_size", type=int, default=256)
    parser.add_argument("--patch_size", type=int, default=8)
    parser.add_argument("--thickness", type=int, default=3)
    parser.add_argument(
        "--mode", choices=["contrast", "k-centroid"], default="contrast"
    )
    parser.add_argument(
        "--colors", type=int, default=0, help="quantization colors (0 = none)"
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--diff_threshold",
        type=int,
        default=10,
        help="per-channel mean L1 threshold for the % stat",
    )
    args = parser.parse_args()

    if not args.input.exists():
        sys.exit(f"Input image not found: {args.input}")

    output_path = args.output or (
        DEFAULT_OUTPUT_DIR / f"{args.input.stem}_compare.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    upstream_settings = {
        "target_size": args.target_size,
        "patch_size": args.patch_size,
        "thickness": args.thickness,
        "mode": args.mode,
    }
    if args.colors > 0:
        upstream_settings["colors"] = args.colors

    port_settings = {
        "target_size": args.target_size,
        "patch_size": args.patch_size,
        "thickness": args.thickness,
        "mode": args.mode,
        "colors": args.colors,
    }

    input_rgb = np.array(Image.open(args.input).convert("RGB"))
    print(f"input    : {args.input} ({input_rgb.shape[1]}x{input_rgb.shape[0]})")

    print("upstream : running (subprocess)")
    upstream_rgb = run_upstream(args.input, upstream_settings)
    print(f"upstream : {upstream_rgb.shape[1]}x{upstream_rgb.shape[0]}")

    print("port     : running (in-process)")
    port_rgb = run_port(input_rgb, port_settings)
    print(f"port     : {port_rgb.shape[1]}x{port_rgb.shape[0]}")

    if port_rgb.shape != upstream_rgb.shape:
        port_for_diff = resize_rgb(port_rgb, upstream_rgb.shape[:2])
    else:
        port_for_diff = port_rgb

    diff = np.abs(
        upstream_rgb.astype(np.int16) - port_for_diff.astype(np.int16)
    )
    mean_l1 = float(diff.mean())
    max_l1 = int(diff.max())
    over_threshold = float((diff.mean(axis=-1) > args.diff_threshold).mean() * 100.0)

    print()
    print("L1 statistics (upstream vs port, at upstream output shape):")
    print(f"  mean per-channel L1 : {mean_l1:.2f}")
    print(f"  max  per-channel L1 : {max_l1}")
    print(
        f"  pixels with mean L1 > {args.diff_threshold}: {over_threshold:.1f}%"
    )

    heat = make_diff_heatmap(upstream_rgb, port_for_diff)
    composite = compose_2x2(
        [
            label_panel(input_rgb, "input"),
            label_panel(upstream_rgb, "upstream"),
            label_panel(port_rgb, "port"),
            label_panel(heat, f"|diff| (mean L1 = {mean_l1:.1f})"),
        ]
    )
    Image.fromarray(composite).save(output_path)
    print(f"\nwrote {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
