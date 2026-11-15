"""Draw the CARDION pipeline diagram (paper/figures/fig_pipeline.png)."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent


def box(ax, x: float, y: float, w: float, h: float, text: str, color: str) -> None:
    """Rounded labelled box."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", fc=color, ec="k", lw=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7)


def arrow(ax, x0: float, y0: float, x1: float, y1: float) -> None:
    """Arrow between two points."""
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=9, lw=0.9, color="k"))


def main() -> None:
    """CLI entry point."""
    argparse.ArgumentParser(description=__doc__).parse_args()
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    ax.set_xlim(0, 10); ax.set_ylim(0, 4); ax.axis("off")
    box(ax, 0.1, 1.55, 1.3, 0.9, "RGB video\n(30 fps)", "#e8e8e8")
    box(ax, 1.8, 1.55, 1.3, 0.9, "Face ROI\n(landmark skin\nmask)", "#e8e8e8")
    arrow(ax, 1.4, 2.0, 1.8, 2.0)
    box(ax, 3.6, 3.0, 1.7, 0.8, "ROI mean RGB\n-> green / POS / CHROM", "#cfe3f7")
    box(ax, 3.6, 0.15, 1.7, 0.8, "DVS emulation\nper colour channel\n(analytic / v2e)", "#f7dccf")
    arrow(ax, 3.1, 2.2, 3.6, 3.3); arrow(ax, 3.1, 1.8, 3.6, 0.6)
    box(ax, 5.8, 3.0, 1.5, 0.8, "10 s FFT HR\n(RGB baselines)", "#cfe3f7")
    arrow(ax, 5.3, 3.4, 5.8, 3.4)
    box(ax, 5.8, 1.55, 1.7, 0.8, "Event POS / green\n(integrated net events)", "#f7dccf")
    box(ax, 5.8, 0.15, 1.7, 0.8, "Event frames ->\nSNN (LIF) / ANN (ReLU)", "#f7dccf")
    arrow(ax, 5.3, 0.7, 5.8, 0.6); arrow(ax, 5.3, 0.9, 5.8, 1.8)
    box(ax, 8.0, 0.9, 1.8, 1.2, "10 s FFT HR\n+ metrics, SOPs,\nenergy proxy", "#d6f0d6")
    arrow(ax, 7.5, 1.9, 8.0, 1.6); arrow(ax, 7.5, 0.55, 8.0, 1.2); arrow(ax, 7.3, 3.2, 8.4, 2.1)
    ax.text(5.0, 3.95, "RGB path (control)", fontsize=7, ha="center", color="#1f5c99")
    ax.text(5.0, 1.05, "event path (synthetic events)", fontsize=7, ha="center", color="#a04a1f")
    out = ROOT / "paper" / "figures" / "fig_pipeline.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    print("wrote", out)


if __name__ == "__main__":
    main()
