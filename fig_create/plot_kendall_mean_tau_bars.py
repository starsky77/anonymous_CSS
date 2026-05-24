#!/usr/bin/env python3
"""Horizontal bar plot: model (from model_b) vs mean_kendall_tau (human vs model summary)."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

from figure_style import apply_nature_style

# Strip leading run timestamp: YYYYMMDD_HHMMSS
HUMAN_HUMAN_CONSISTENCY_X = 0.57524429
_TS_PREFIX = re.compile(r"^\d{8}_\d{6}")

# Paul Tol–style qualitative colours (colour-blind friendly; prints cleanly)
FAMILY_COLORS: dict[str, str] = {
    "claude": "#E69F00",
    "gemini": "#0072B2",
    "google_gemma": "#009E73",
    "gpt": "#CC79A7",
    "OpenGVLab_InternVL3": "#D55E00",
    "Qwen3-VL": "#56B4E9",
    "other": "#999999",
}

REFERENCE_LINE_COLOR = "#BC3C29"
REFERENCE_LINE_DASHES = (0, (10, 5))
REFERENCE_LINE_WIDTH = 2.0

# Bars span ~0.37–0.51; reference ~0.575 — pad so line and bars stay in frame
TAU_XLIM = (0.34, 0.62)


def strip_run_timestamp(model_b: str) -> str:
    return _TS_PREFIX.sub("", model_b.strip())


def format_display_name(stripped_name: str) -> str:
    return stripped_name.replace("Qwen_Qwen3-VL", "Qwen3-VL")


def model_family(stripped_name: str) -> str:
    if stripped_name.startswith("OpenGVLab_InternVL3"):
        return "OpenGVLab_InternVL3"
    if stripped_name.startswith("Qwen_Qwen3-VL"):
        return "Qwen3-VL"
    if stripped_name.startswith("google_gemma"):
        return "google_gemma"
    if stripped_name.startswith("gemini"):
        return "gemini"
    if stripped_name.startswith("claude"):
        return "claude"
    if stripped_name.startswith("gpt"):
        return "gpt"
    return "other"


def load_rows(csv_path: Path) -> list[tuple[str, str, float, str]]:
    """Returns list of (display_name, model_b_raw, mean_kendall_tau, family)."""
    out: list[tuple[str, str, float, str]] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw = row.get("model_b", "").strip()
            if not raw:
                continue
            stripped = strip_run_timestamp(raw)
            tau_s = row.get("mean_kendall_tau", "").strip()
            if not tau_s:
                continue
            tau = float(tau_s)
            fam = model_family(stripped)
            out.append((stripped, raw, tau, fam))
    return out


def save_figure(fig: mpl.figure.Figure, base_path: Path) -> list[Path]:
    stem = base_path
    if stem.suffix.lower() in {".png", ".pdf", ".svg"}:
        stem = stem.with_suffix("")
    written: list[Path] = []
    png_path = stem.with_suffix(".png")
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    written.append(png_path)
    pdf_path = stem.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white", edgecolor="none")
    written.append(pdf_path)
    return written


def main() -> None:
    apply_nature_style()

    default_csv = (
        Path(__file__).resolve().parents[1]
        / "Src"
        / "eval_output_tau"
        / "human_vs_model"
        / "kendall_vs_human_summary.csv"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csv_path",
        nargs="?",
        type=Path,
        default=default_csv,
        help=f"Input summary CSV (default: {default_csv})",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output base path (writes .png at 300 dpi and .pdf; default: .../kendall_mean_tau_bars)",
    )
    args = parser.parse_args()
    csv_path: Path = args.csv_path
    if not csv_path.is_file():
        raise SystemExit(f"CSV not found: {csv_path}")

    rows = load_rows(csv_path)
    if not rows:
        raise SystemExit("No rows to plot.")

    rows.sort(key=lambda r: r[2], reverse=True)
    labels = [format_display_name(r[0]) for r in rows]
    values = [r[2] for r in rows]
    colors = [FAMILY_COLORS.get(r[3], FAMILY_COLORS["other"]) for r in rows]

    fig_w_in = 7.2
    fig_h_in = max(3.2, 0.21 * len(labels) + 1.1)
    fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in), layout="constrained")

    y_pos = range(len(labels))
    ax.barh(
        y_pos,
        values,
        color=colors,
        height=0.68,
        edgecolor="white",
        linewidth=0.9,
        zorder=2,
    )
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels)
    # Unicode τ so axis label uses the same sans-serif stack as Top1-Acc (no mathtext size drift)
    ax.set_xlabel("Mean Kendall's τ")
    ax.set_ylabel("Model")
    ax.set_xlim(*TAU_XLIM)
    ax.xaxis.set_major_locator(MultipleLocator(0.05))
    ax.invert_yaxis()

    ax.grid(
        axis="x",
        linestyle="-",
        linewidth=0.4,
        color="#C8C8C8",
        zorder=0,
    )
    ax.set_axisbelow(True)

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    ax.axvline(
        HUMAN_HUMAN_CONSISTENCY_X,
        color=REFERENCE_LINE_COLOR,
        linestyle=REFERENCE_LINE_DASHES,
        linewidth=REFERENCE_LINE_WIDTH,
        zorder=5,
        dash_capstyle="butt",
    )

    families_in_data: list[str] = []
    seen: set[str] = set()
    for _disp, _raw, _v, fam in rows:
        if fam not in seen:
            seen.add(fam)
            families_in_data.append(fam)

    from matplotlib.patches import Patch

    legend_handles = [
        Patch(
            facecolor=FAMILY_COLORS.get(f, FAMILY_COLORS["other"]),
            edgecolor="white",
            linewidth=0.6,
            label=f,
        )
        for f in families_in_data
    ]
    legend_handles.append(
        Line2D(
            [0],
            [0],
            color=REFERENCE_LINE_COLOR,
            linestyle=REFERENCE_LINE_DASHES,
            linewidth=REFERENCE_LINE_WIDTH,
            label="Human–human consistency",
        )
    )
    ax.legend(
        handles=legend_handles,
        loc="lower right",
        frameon=True,
        fancybox=True,
        facecolor="#E8E8E8",
        edgecolor="#B0B0B0",
        framealpha=0.72,
        borderpad=0.55,
        handlelength=2.2,
        handletextpad=0.6,
        borderaxespad=0.4,
    )

    out_base = args.output
    if out_base is None:
        out_base = (
            Path(__file__).resolve().parents[1]
            / "results"
            / "kendall_mean_tau"
            / "kendall_mean_tau_bars.png"
        )
    out_base.parent.mkdir(parents=True, exist_ok=True)
    written = save_figure(fig, out_base)
    plt.close(fig)
    for p in written:
        print(f"Wrote {p}")


if __name__ == "__main__":
    main()
