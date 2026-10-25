"""Re-draw the controlled noise/flicker figure from its saved results (no simulation is re-run).

    python scripts/replot_noise.py --amp 0.2

Reads the newest ``out/noise_study_amp<amp>_*.json`` and writes ``out/fig_noise_study_amp<amp>.png`` (overwriting the older, less legible drawing) with
labelled tick values, a legend that does not cover data, and consistent colours.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

LABELS = {"rgb_green": "RGB green", "rgb_pos": "RGB POS", "event_green@0.02": "event green θ=0.02", "event_green@0.1": "event green θ=0.1",
          "event_pos@0.02": "event POS θ=0.02", "event_pos@0.1": "event POS θ=0.1"}
COLORS = {"rgb_green": "#7aa6c9", "rgb_pos": "#1b6ca8", "event_green@0.02": "#f0b36a", "event_green@0.1": "#c06a10", "event_pos@0.02": "#c9a0dc", "event_pos@0.1": "#7b4fa0"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--amp", default="0.2")
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    p = sorted(glob.glob(str(ROOT / "out" / f"noise_study_amp{a.amp}_*.json")))[-1]
    d = json.loads(Path(p).read_text(encoding="utf-8-sig"))
    sig, fl = d["sigmas"], d["flickers"]
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, len(fl), figsize=(7.1, 2.9), sharey=True)
    for ax, f in zip(axes, fl):
        for m in LABELS:
            ax.plot(sig, [d["summary"][f"sigma{s}_flicker{f}"][m] for s in sig], marker="o", ms=3.5, lw=1.2, color=COLORS[m], ls="--" if m.startswith("rgb") else "-", label=LABELS[m])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(sig)
        ax.set_xticklabels([f"{s:g}" for s in sig])
        ax.minorticks_off()
        ax.set_xlabel("sensor noise σ (8-bit counts)")
        ax.set_title(f"aliased illumination flicker {f:.0%}", fontsize=8.5)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("HR MAE (BPM)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=6.6, frameon=False, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.01), columnspacing=1.0, handlelength=1.8)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    out = ROOT / "out" / f"fig_noise_study_amp{a.amp}.png"
    fig.savefig(out, dpi=220)
    print("wrote", out, "from", Path(p).name)


if __name__ == "__main__":
    main()
