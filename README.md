# blender-pixeloe

Blender addon that wraps the [PixelOE](https://github.com/KohakuBlueleaf/PixelOE) pixelization algorithm by Shih-Ying Yeh (KohakuBlueleaf). Operates on any Blender Image datablock and produces a pixelized version as a new datablock.

## Status

In development. See `BRIEF.md` for design and phasing.

## Development setup

Requires [uv](https://docs.astral.sh/uv/). The project pins Python 3.11 (matching Blender 4.2's bundled interpreter) via `.python-version`.

```sh
uv sync                              # create .venv, install deps
uv run python tests/images/generate_synthetic.py  # regenerate synthetic test images
./setup_upstream.sh                  # clone upstream PixelOE for reference reading
```

## Why a port

Upstream PixelOE depends on `torch`, `torchvision`, `kornia`, and `opencv-python`. Blender's bundled Python ships only `numpy` and `Pillow`. This project re-implements the algorithm in pure numpy + Pillow (+ scipy) so the addon installs as a single zip with no runtime download.

## Attribution

Algorithm by Shih-Ying Yeh (KohakuBlueleaf), upstream at https://github.com/KohakuBlueleaf/PixelOE, licensed Apache 2.0. The k-centroid downscale path originates from [Astropulse/pixeldetector](https://github.com/Astropulse/pixeldetector).

This project is a port and Blender integration; it carries the same Apache 2.0 license as upstream.

## License

Apache License 2.0. See `LICENSE`.
