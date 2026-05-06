"""Regression and quality harness for the pixelize port.

Each cell of the matrix below runs the port end-to-end and reports:
  - wall-clock and peak memory (perf)
  - RGB mean L1 vs upstream (legacy proxy)
  - LAB delta-E76 mean and p95 vs upstream (perceptual; JND ~= 2.3)
plus a 4-panel viz (input / upstream / port / dE76 heatmap).

The harness compares current numbers against `baseline.json`. After making a
change, run the harness, eyeball the per-cell viz to confirm any drift is
visually acceptable, then `--update-baseline` to commit the new numbers.

Upstream outputs are cached under tests/harness/cache/ keyed by sha256 of
(image bytes + settings json). The cache directory is gitignored so each
checkout rebuilds it on first run.

Optional `regression_local.py` (gitignored) may define `EXTRA_MATRIX` to
extend the matrix with cells referencing non-distributable local images.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGES_DIR = PROJECT_ROOT / "tests" / "images"
CACHE_DIR = PROJECT_ROOT / "tests" / "harness" / "cache"
OUTPUT_DIR = PROJECT_ROOT / "tests" / "harness" / "output" / "regression"
BASELINE_FILE = PROJECT_ROOT / "tests" / "harness" / "baseline.json"
UPSTREAM_PYTHON = PROJECT_ROOT / "tests" / "upstream" / ".venv" / "bin" / "python"
UPSTREAM_RUNNER = Path(__file__).parent / "_upstream_runner.py"
LOCAL_MATRIX_FILE = Path(__file__).parent / "regression_local.py"


def _cell(label: str, image: str, **settings) -> dict:
    base = {
        "target_size": 128,
        "patch_size": 8,
        "thickness": 3,
        "mode": "contrast",
        "colors": 32,
        "color_quant_method": "kmeans",
    }
    base.update(settings)
    return {"label": label, "image": image, "settings": base}


MATRIX: list[dict] = [
    _cell("dark_highlights_t128", "dark-highlights.png"),
    _cell("gradient_t128", "gradient.png"),
    _cell("snow_leopard_t128", "snow-leopard.webp"),
    _cell("painterly_t128", "painterly_portrait.png"),
    _cell("painterly_t256", "painterly_portrait.png", target_size=256),
    _cell("painterly_t128_kc", "painterly_portrait.png", mode="k-centroid"),
    _cell("painterly_t128_c64", "painterly_portrait.png", colors=64),
    _cell("painterly_t128_maxc", "painterly_portrait.png", color_quant_method="maxcover"),
    _cell("impressionism_lady_t128", "impressionism_lady.png"),
    _cell("moon_face_t128", "realistic_surreal_moon_face.png"),
    _cell("stylized_dark_t128", "stylized_dark_portrait.png"),
    _cell("stylized_man_t128", "stylized_man_portrait.png"),
    _cell("toon_punk_t128", "toon_punk_girl.png"),
]


def _load_local_matrix_extensions() -> list[dict]:
    if not LOCAL_MATRIX_FILE.exists():
        return []
    namespace: dict = {"_cell": _cell}
    exec(LOCAL_MATRIX_FILE.read_text(), namespace)
    extras = namespace.get("EXTRA_MATRIX", [])
    return list(extras)


def cache_key(image_path: Path, settings: dict) -> str:
    h = hashlib.sha256()
    h.update(image_path.read_bytes())
    h.update(json.dumps(settings, sort_keys=True).encode())
    return h.hexdigest()[:16]


def get_upstream_post_final(image_path: Path, settings: dict) -> np.ndarray:
    """Cached upstream end-to-end output."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = cache_key(image_path, settings)
    cache_file = CACHE_DIR / f"{key}.npy"
    if cache_file.exists():
        return np.load(cache_file)
    if not UPSTREAM_PYTHON.exists():
        sys.exit(f"Upstream venv missing at {UPSTREAM_PYTHON}; run setup_upstream.sh")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        subprocess.run(
            [
                str(UPSTREAM_PYTHON),
                str(UPSTREAM_RUNNER),
                "--input",
                str(image_path),
                "--dump_dir",
                str(tmp_path),
                "--settings",
                json.dumps(settings),
            ],
            check=True,
        )
        shutil.copy(tmp_path / "post_final.npy", cache_file)
    return np.load(cache_file)


def lab_delta_e76(rgb_a: np.ndarray, rgb_b: np.ndarray) -> np.ndarray:
    """Per-pixel CIE76 delta-E. Inputs are uint8 RGB arrays of equal shape."""
    from blender_pixeloe.core.colorspace import rgb_to_lab

    lab_a = rgb_to_lab(rgb_a).astype(np.float32)
    lab_b = rgb_to_lab(rgb_b).astype(np.float32)
    L_scale = 100.0 / 255.0
    L_diff = (lab_a[..., 0] - lab_b[..., 0]) * L_scale
    a_diff = lab_a[..., 1] - lab_b[..., 1]
    b_diff = lab_a[..., 2] - lab_b[..., 2]
    return np.sqrt(L_diff * L_diff + a_diff * a_diff + b_diff * b_diff)


def measure_port(image_path: Path, settings: dict) -> tuple[np.ndarray, float, int]:
    from blender_pixeloe.core import pixelize

    rgb = np.array(Image.open(image_path).convert("RGB"))
    tracemalloc.start()
    t0 = time.perf_counter()
    out = pixelize(rgb, **settings)
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return out, elapsed, peak


def de_heatmap(de: np.ndarray) -> np.ndarray:
    """ΔE76 → 3-channel uint8. Black at 0, red at JND (~2.3), orange at 10+."""
    r = np.clip(de / 10.0, 0, 1)
    g = np.clip(de / 20.0, 0, 1)
    heat = np.zeros((*de.shape, 3), dtype=np.uint8)
    heat[..., 0] = (r * 255).astype(np.uint8)
    heat[..., 1] = (g * 255).astype(np.uint8)
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


def write_viz(
    label: str,
    original: np.ndarray,
    upstream: np.ndarray,
    port: np.ndarray,
    de: np.ndarray,
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    composite = compose_2x2(
        [
            label_panel(original, "input"),
            label_panel(upstream, "upstream"),
            label_panel(port, "port"),
            label_panel(de_heatmap(de), "ΔE76 (port vs upstream)"),
        ]
    )
    path = output_dir / f"{label}.png"
    Image.fromarray(composite).save(path)
    return path


def measure_cell(cell: dict) -> dict:
    label = cell["label"]
    image_path = IMAGES_DIR / cell["image"]
    if not image_path.exists():
        return {"label": label, "status": "skipped", "reason": f"missing image: {cell['image']}"}

    upstream_out = get_upstream_post_final(image_path, cell["settings"])
    port_out, elapsed, peak = measure_port(image_path, cell["settings"])

    if upstream_out.shape != port_out.shape:
        return {
            "label": label,
            "status": "shape_mismatch",
            "reason": f"upstream {upstream_out.shape} vs port {port_out.shape}",
        }

    de = lab_delta_e76(port_out, upstream_out)
    rgb_diff = np.abs(port_out.astype(np.float32) - upstream_out.astype(np.float32))
    metrics = {
        "wall_clock_s": round(elapsed, 4),
        "peak_mem_mib": round(peak / (1024 * 1024), 2),
        "rgb_l1_mean": round(float(rgb_diff.mean()), 4),
        "lab_de76_mean": round(float(de.mean()), 4),
        "lab_de76_p95": round(float(np.percentile(de, 95)), 4),
    }
    original = np.array(Image.open(image_path).convert("RGB"))
    viz_path = write_viz(label, original, upstream_out, port_out, de, OUTPUT_DIR)
    return {
        "label": label,
        "status": "ok",
        "metrics": metrics,
        "viz": str(viz_path.relative_to(PROJECT_ROOT)),
    }


def load_baseline() -> dict:
    if not BASELINE_FILE.exists():
        return {}
    return json.loads(BASELINE_FILE.read_text()).get("cells", {})


def save_baseline(cells: dict) -> None:
    payload = {
        "_metadata": {
            "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "comment": "Regenerate via tests/harness/regression.py --update-baseline.",
        },
        "cells": cells,
    }
    BASELINE_FILE.write_text(json.dumps(payload, indent=2) + "\n")


METRIC_ORDER = ("wall_clock_s", "peak_mem_mib", "rgb_l1_mean", "lab_de76_mean", "lab_de76_p95")
METRIC_LABEL = {
    "wall_clock_s": "wall",
    "peak_mem_mib": "mem",
    "rgb_l1_mean": "L1",
    "lab_de76_mean": "ΔE",
    "lab_de76_p95": "ΔE95",
}
METRIC_FMT = {
    "wall_clock_s": lambda v: f"{v:.2f}s",
    "peak_mem_mib": lambda v: f"{v:.0f}M",
    "rgb_l1_mean": lambda v: f"{v:.2f}",
    "lab_de76_mean": lambda v: f"{v:.2f}",
    "lab_de76_p95": lambda v: f"{v:.2f}",
}


def format_delta(metric: str, current: float, baseline: float | None) -> str:
    if baseline is None:
        return ""
    delta = current - baseline
    if abs(delta) < 1e-3:
        return ""
    sign = "+" if delta >= 0 else ""
    fmt = METRIC_FMT[metric]
    return f" ({sign}{fmt(delta).lstrip('+')})" if delta < 0 else f" ({sign}{fmt(delta)})"


def print_row(result: dict, baseline_metrics: dict | None) -> None:
    label = result["label"]
    if result["status"] != "ok":
        print(f"  {label:<28} [{result['status']}] {result.get('reason', '')}")
        return
    m = result["metrics"]
    parts = []
    for key in METRIC_ORDER:
        cur = m[key]
        base = baseline_metrics.get(key) if baseline_metrics else None
        parts.append(f"{METRIC_LABEL[key]}={METRIC_FMT[key](cur)}{format_delta(key, cur, base)}")
    print(f"  {label:<28} {' '.join(parts)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--cells",
        type=str,
        help="Comma-separated cell labels to run (default: all)",
    )
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="Write current metrics to baseline.json after run",
    )
    args = parser.parse_args()

    matrix = list(MATRIX) + _load_local_matrix_extensions()
    if args.cells:
        wanted = {s.strip() for s in args.cells.split(",")}
        matrix = [c for c in matrix if c["label"] in wanted]
        missing = wanted - {c["label"] for c in matrix}
        if missing:
            sys.exit(f"unknown cells: {sorted(missing)}")

    if not matrix:
        sys.exit("no cells to run")

    baseline = load_baseline()
    print(f"Running {len(matrix)} cell(s); baseline {'present' if baseline else 'EMPTY (bootstrap)'}\n")

    results: list[dict] = []
    for cell in matrix:
        result = measure_cell(cell)
        results.append(result)
        print_row(result, baseline.get(result["label"]) if baseline else None)

    ok = [r for r in results if r["status"] == "ok"]
    skipped = [r for r in results if r["status"] != "ok"]
    print(f"\n{len(ok)} ok, {len(skipped)} skipped/errored")

    if args.update_baseline:
        new_baseline = {r["label"]: r["metrics"] for r in ok}
        save_baseline(new_baseline)
        print(f"baseline written to {BASELINE_FILE.relative_to(PROJECT_ROOT)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
