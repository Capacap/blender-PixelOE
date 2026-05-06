"""Runs upstream PixelOE inside its own venv and writes the result to disk.

Invoked by `compare.py` via subprocess against `tests/upstream/.venv/bin/python`.
Stdin/stdout are not used; settings come in as a JSON string argument so the
parent process never needs to know upstream's exact signature shape.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from pixeloe.legacy.pixelize import pixelize


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--settings", type=str, required=True, help="JSON dict")
    args = parser.parse_args()

    settings = json.loads(args.settings)

    rgb = np.array(Image.open(args.input).convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    out_bgr = pixelize(bgr, **settings)

    out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
    Image.fromarray(out_rgb).save(args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
