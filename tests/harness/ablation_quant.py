"""Quantization ablation: re-run the regression matrix with colors=0 to
isolate algorithmic drift from k-means palette-choice variance.

The headline ΔE76 vs upstream is confounded by two effects: the port's actual
algorithmic differences from upstream, and the fact that upstream's
`cv2.kmeans(..., KMEANS_RANDOM_CENTERS, attempts=4)` and the port's k-means++
single-run produce different but equally-optimal 32-way LAB partitions.
Borderline pixels get assigned to different cluster centers, inflating ΔE
without either output being visibly worse.

Setting colors=0 disables quantization in both pipelines, leaving only
algorithmic drift. The gap between the two runs is the palette-choice
contribution.

Output is written to `tests/harness/ablation_quant.json` next to baseline.json.
Re-run after harness changes via `uv run python tests/harness/ablation_quant.py`.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

# Reuse the regression harness machinery.
sys.path.insert(0, str(Path(__file__).parent))
from regression import (  # noqa: E402
    BASELINE_FILE,
    MATRIX,
    PROJECT_ROOT,
    measure_cell,
)


OUT_FILE = Path(__file__).parent / "ablation_quant.json"


def make_no_quant_cell(cell: dict) -> dict | None:
    """Return a copy of `cell` with quantization disabled, or None if not
    applicable (k-centroid mode does its own clustering; cells with colors=0
    already are the ablation)."""
    settings = dict(cell["settings"])
    if settings.get("mode") != "contrast":
        return None
    if settings.get("colors", 0) <= 0:
        return None
    settings["colors"] = 0
    label = f"{cell['label']}__c0"
    return {"label": label, "image": cell["image"], "settings": settings}


def main() -> int:
    baseline = json.loads(BASELINE_FILE.read_text())["cells"]

    rows = []
    for cell in MATRIX:
        no_quant = make_no_quant_cell(cell)
        if no_quant is None:
            continue
        baseline_metrics = baseline.get(cell["label"])
        if baseline_metrics is None:
            print(f"  {cell['label']:<28} [skipped] no baseline entry")
            continue

        result = measure_cell(no_quant)
        if result["status"] != "ok":
            print(f"  {cell['label']:<28} [{result['status']}]")
            continue

        m = result["metrics"]
        b_l1 = baseline_metrics["rgb_l1_mean"]
        b_de = baseline_metrics["lab_de76_mean"]
        n_l1 = m["rgb_l1_mean"]
        n_de = m["lab_de76_mean"]

        rows.append(
            {
                "cell": cell["label"],
                "colors_baseline": cell["settings"]["colors"],
                "rgb_l1_with_quant": b_l1,
                "rgb_l1_no_quant": n_l1,
                "rgb_l1_palette_share": round(b_l1 - n_l1, 4),
                "lab_de76_with_quant": b_de,
                "lab_de76_no_quant": n_de,
                "lab_de76_palette_share": round(b_de - n_de, 4),
            }
        )
        print(
            f"  {cell['label']:<28} "
            f"L1 {b_l1:>5.2f} -> {n_l1:>5.2f}  "
            f"ΔE {b_de:>5.2f} -> {n_de:>5.2f}"
        )

    if not rows:
        print("\nNo cells ran. Check baseline.json is current.")
        return 1

    de_share = [r["lab_de76_palette_share"] for r in rows]
    l1_share = [r["rgb_l1_palette_share"] for r in rows]
    summary = {
        "lab_de76_palette_share_mean": round(sum(de_share) / len(de_share), 4),
        "lab_de76_palette_share_max": round(max(de_share), 4),
        "rgb_l1_palette_share_mean": round(sum(l1_share) / len(l1_share), 4),
        "rgb_l1_palette_share_max": round(max(l1_share), 4),
    }

    payload = {
        "_metadata": {
            "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "comment": (
                "Per-cell deltas between baseline (with quantization) and "
                "ablation (colors=0). The 'palette_share' columns are the "
                "amount of the headline metric attributable to k-means "
                "palette-choice variance vs upstream."
            ),
        },
        "summary": summary,
        "rows": rows,
    }
    OUT_FILE.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nWrote {OUT_FILE.relative_to(PROJECT_ROOT)}")
    print(
        f"Mean palette share: ΔE {summary['lab_de76_palette_share_mean']:.2f}, "
        f"L1 {summary['rgb_l1_palette_share_mean']:.2f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
