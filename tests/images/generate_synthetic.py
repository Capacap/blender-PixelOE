"""Generate the two synthetic test images committed with the project.

- gradient.png: smooth gradient + sharp color blocks. Exercises outline
  expansion on hard edges and tile boundaries.
- dark-highlights.png: very dark background with small bright spots. Stresses
  LAB conversion drift and weight-map behavior on low-luminance images.
"""
from pathlib import Path

import numpy as np
from PIL import Image

OUT_DIR = Path(__file__).parent


def gradient(size: int = 512) -> np.ndarray:
    h = w = size
    arr = np.zeros((h, w, 3), dtype=np.uint8)

    t = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    arr[:, : w // 2, 0] = ((1.0 - t) * 255).astype(np.uint8)
    arr[:, : w // 2, 2] = (t * 255).astype(np.uint8)

    blocks = [
        (0, h // 4, [0, 255, 0]),
        (h // 4, h // 2, [255, 255, 0]),
        (h // 2, 3 * h // 4, [255, 0, 255]),
        (3 * h // 4, h, [0, 255, 255]),
    ]
    for y0, y1, color in blocks:
        arr[y0:y1, w // 2 :] = color
    return arr


def dark_highlights(size: int = 512) -> np.ndarray:
    h = w = size
    arr = np.full((h, w, 3), 12, dtype=np.uint8)
    yy, xx = np.mgrid[:h, :w]
    spots = [
        ((128, 128), 24, [255, 255, 255]),
        ((384, 128), 32, [255, 200, 50]),
        ((256, 384), 40, [80, 180, 255]),
        ((400, 400), 16, [255, 50, 50]),
    ]
    for (cx, cy), r, color in spots:
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
        arr[mask] = color
    return arr


def main() -> None:
    Image.fromarray(gradient()).save(OUT_DIR / "gradient.png", optimize=True)
    Image.fromarray(dark_highlights()).save(OUT_DIR / "dark-highlights.png", optimize=True)
    print(f"Wrote {OUT_DIR / 'gradient.png'}")
    print(f"Wrote {OUT_DIR / 'dark-highlights.png'}")


if __name__ == "__main__":
    main()
