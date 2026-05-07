"""Render comparison graphics for the README.

Each row is rendered in four panels:
  1. Original (resized to the panel size with Lanczos)
  2. Naive: Lanczos downscale to (t256, t256), then nearest upscale back to
     the panel size. The "if you just resize" baseline at the higher of the
     two pixel-art resolutions, so the naive result gets its best showing.
  3. PixelOE at target_size=256
  4. PixelOE at target_size=128

Outputs:
  docs/hero.png            - one 4-panel strip for the top of the README
  docs/comparison.png      - 3-row grid covering the headline test images
  docs/full_comparison.png - same layout across the full test suite (linked,
                             not inlined, in the README)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from blender_pixeloe.core import pixelize

REPO_ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = REPO_ROOT / "tests" / "images"
DOCS_DIR = REPO_ROOT / "docs"

PANEL_PX = 512
PADDING = 12
LABEL_HEIGHT = 32
BG = (24, 24, 24)
FG = (230, 230, 230)

NAIVE_TARGET = 256
PIXELOE_TARGETS = (256, 128)
PATCH_SIZE = 8
THICKNESS = 3
COLORS = 32


def _square_crop(im: Image.Image) -> Image.Image:
    w, h = im.size
    if w == h:
        return im
    s = min(w, h)
    left = (w - s) // 2
    top = (h - s) // 2
    return im.crop((left, top, left + s, top + s))


def _naive_pixelize(rgb: np.ndarray, target_size: int) -> np.ndarray:
    """Lanczos to small, nearest back up. The straw-man baseline."""
    src = Image.fromarray(rgb)
    w, h = src.size
    small = src.resize((target_size, target_size), Image.LANCZOS)
    big = small.resize((w, h), Image.NEAREST)
    return np.array(big)


def _render_panels(rgb: np.ndarray) -> tuple[Image.Image, ...]:
    """Return (original, naive, pixeloe@t256, pixeloe@t128) all at PANEL_PX square."""
    src_im = Image.fromarray(rgb)
    original = src_im.resize((PANEL_PX, PANEL_PX), Image.LANCZOS)

    naive = Image.fromarray(_naive_pixelize(rgb, NAIVE_TARGET))
    naive = naive.resize((PANEL_PX, PANEL_PX), Image.NEAREST)

    pixeloe_panels: list[Image.Image] = []
    for t in PIXELOE_TARGETS:
        out_rgb = pixelize(
            rgb,
            mode="contrast",
            target_size=t,
            patch_size=PATCH_SIZE,
            thickness=THICKNESS,
            colors=COLORS,
            no_upscale=False,
        )
        panel = Image.fromarray(out_rgb).resize((PANEL_PX, PANEL_PX), Image.NEAREST)
        pixeloe_panels.append(panel)

    return (original, naive, *pixeloe_panels)


def _font() -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, 18)
    return ImageFont.load_default()


def _draw_label(draw: ImageDraw.ImageDraw, x: int, y: int, label: str, font) -> None:
    bbox = draw.textbbox((0, 0), label, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    draw.text(
        (x + (PANEL_PX - tw) // 2, y + (LABEL_HEIGHT - th) // 2),
        label,
        fill=FG,
        font=font,
    )


def _compose_strip(panels: tuple[Image.Image, ...], labels: tuple[str, ...]) -> Image.Image:
    n = len(panels)
    total_w = n * PANEL_PX + (n + 1) * PADDING
    total_h = PANEL_PX + LABEL_HEIGHT + 2 * PADDING
    canvas = Image.new("RGB", (total_w, total_h), BG)
    draw = ImageDraw.Draw(canvas)
    font = _font()
    for i, (panel, label) in enumerate(zip(panels, labels)):
        x = PADDING + i * (PANEL_PX + PADDING)
        canvas.paste(panel, (x, LABEL_HEIGHT + PADDING))
        _draw_label(draw, x, 0, label, font)
    return canvas


def _compose_grid(
    rows: list[tuple[Image.Image, ...]], col_labels: tuple[str, ...]
) -> Image.Image:
    n_cols = len(col_labels)
    n_rows = len(rows)
    total_w = n_cols * PANEL_PX + (n_cols + 1) * PADDING
    total_h = LABEL_HEIGHT + (PANEL_PX + PADDING) * n_rows + PADDING
    canvas = Image.new("RGB", (total_w, total_h), BG)
    draw = ImageDraw.Draw(canvas)
    font = _font()

    for j, label in enumerate(col_labels):
        x = PADDING + j * (PANEL_PX + PADDING)
        _draw_label(draw, x, 0, label, font)

    for i, row in enumerate(rows):
        for j, panel in enumerate(row):
            x = PADDING + j * (PANEL_PX + PADDING)
            y = LABEL_HEIGHT + PADDING + i * (PANEL_PX + PADDING)
            canvas.paste(panel, (x, y))
    return canvas


def _load(name: str) -> np.ndarray:
    im = Image.open(IMAGES_DIR / name).convert("RGB")
    im = _square_crop(im)
    return np.array(im)


def main() -> None:
    DOCS_DIR.mkdir(exist_ok=True)
    col_labels = (
        "Original",
        f"Naive (Lanczos + Nearest, t={NAIVE_TARGET})",
        f"PixelOE (t={PIXELOE_TARGETS[0]}, c={COLORS})",
        f"PixelOE (t={PIXELOE_TARGETS[1]}, c={COLORS})",
    )

    hero_rgb = _load("snow-leopard.webp")
    strip = _compose_strip(_render_panels(hero_rgb), col_labels)
    strip.save(DOCS_DIR / "hero.png", optimize=True)
    print(f"wrote {DOCS_DIR / 'hero.png'} ({strip.size})")

    grid_images = [
        "snow-leopard.webp",
        "painterly_portrait.png",
        "stylized_man_portrait.png",
    ]
    rows = [_render_panels(_load(name)) for name in grid_images]
    grid = _compose_grid(rows, col_labels)
    grid.save(DOCS_DIR / "comparison.png", optimize=True)
    print(f"wrote {DOCS_DIR / 'comparison.png'} ({grid.size})")

    # Full suite: every committed test image except the synthetic gradient
    # (which is an algorithmic correctness check, not a visual demo).
    full_suite = [
        "snow-leopard.webp",
        "painterly_portrait.png",
        "stylized_man_portrait.png",
        "stylized_dark_portrait.png",
        "impressionism_lady.png",
        "toon_punk_girl.png",
        "realistic_surreal_moon_face.png",
        "dark-highlights.png",
    ]
    full_rows = [_render_panels(_load(name)) for name in full_suite]
    full_grid = _compose_grid(full_rows, col_labels)
    full_grid.save(DOCS_DIR / "full_comparison.png", optimize=True)
    print(f"wrote {DOCS_DIR / 'full_comparison.png'} ({full_grid.size})")


if __name__ == "__main__":
    main()
