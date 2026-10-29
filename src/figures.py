"""Make the README chart from results/results.json.

Usage:
    python -m src.figures
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

MODEL = "#2a78d6"      # categorical slot 1 (blue): learned models
BASELINE = "#8a8984"   # neutral grey: naive baselines
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"

LABELS = {  # results key -> (chart label, is_baseline)
    "Momentum (repeat last move)": ("Repeat the last move", True),
    "Majority class from training": ("Always predict the majority class", True),
    "Reversal (opposite of last move)": ("Reverse the last move", True),
    "ARIMA(5,1,0) one-step": ("ARIMA(5,1,0), one-step", False),
    "Ridge regression (same features)": ("Ridge regression", False),
    "LSTM (average of seeds)": ("LSTM (average of 3 seeds)", False),
}


def main(path="results/results.json", out="results/figures/directional_accuracy.png"):
    models = json.loads(Path(path).read_text())["models"]
    rows = [(lbl, base, models[k]) for k, (lbl, base) in LABELS.items() if k in models]
    rows.sort(key=lambda r: r[2]["directional_accuracy"])

    fig, ax = plt.subplots(figsize=(8.5, 0.5 * len(rows) + 1.7), dpi=200)
    for y, (lbl, base, m) in enumerate(rows):
        color = BASELINE if base else MODEL
        acc, (lo, hi) = m["directional_accuracy"] * 100, m["accuracy_ci95"]
        ax.plot([lo * 100, hi * 100], [y, y], color=color, lw=2.5, solid_capstyle="round")
        ax.plot(acc, y, "o", ms=8, color=color, mec="white", mew=2, zorder=3)
        left = acc < 50
        ax.text(hi * 100 + 0.12 if not left else lo * 100 - 0.12, y, f"{acc:.1f}%",
                va="center", ha="left" if not left else "right", fontsize=9, color=INK)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=9, color=INK)
    ax.axvline(50, color=MUTED, lw=1, ls=(0, (4, 3)))
    ax.text(50.04, len(rows) - 0.45, "50% = coin flip", fontsize=8, color=MUTED, va="bottom")
    ax.set_xlim(47.5, 52.5)
    ax.set_ylim(-0.6, len(rows) - 0.1)
    ax.set_xlabel("Directional accuracy on the next 1-minute move, test period (%)",
                  fontsize=9, color=MUTED)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="x", colors=MUTED, labelsize=8)
    ax.tick_params(axis="y", length=0)
    handles = [plt.Line2D([], [], marker="o", ls="-", lw=2.5, color=MODEL, ms=7,
                          label="Learned model (95% interval)"),
               plt.Line2D([], [], marker="o", ls="-", lw=2.5, color=BASELINE, ms=7,
                          label="Naive baseline (95% interval)")]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(-0.01, 1.0), ncol=2,
              frameon=False, fontsize=8.5, labelcolor=INK)
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor="white")
    print("saved", out)


if __name__ == "__main__":
    main()
