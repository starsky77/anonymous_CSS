"""Shared matplotlib rcParams for Kendall bar figures (single source of truth)."""

from __future__ import annotations

import matplotlib as mpl

# Keep Top1-Acc and mean-τ figures visually aligned (fonts, ticks, legend).
NATURE_RCPARAMS: dict[str, object] = {
    "font.family": "sans-serif",
    "font.sans-serif": [
        "Arial",
        "Helvetica",
        "Helvetica Neue",
        "DejaVu Sans",
    ],
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "axes.grid": False,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
    "savefig.facecolor": "white",
    "savefig.edgecolor": "white",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def apply_nature_style() -> None:
    """Sans-serif defaults aligned with common journal figure guidance."""
    mpl.rcParams.update(NATURE_RCPARAMS)
