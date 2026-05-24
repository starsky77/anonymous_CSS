#!/usr/bin/env python3
"""Bar-style figures: human vs per-model correlation for each geometry predictor."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FIG_CREATE = _REPO_ROOT / "fig_create"
if str(_FIG_CREATE) not in sys.path:
    sys.path.insert(0, str(_FIG_CREATE))

from figure_style import apply_nature_style

_DIR = Path(__file__).resolve().parent
_RESULTS_DIR = _REPO_ROOT / "results" / "correlation_bars"
CACHE_CSV = _RESULTS_DIR / "cache_correlation_by_csv.csv"
SUMMARY_CSV = _RESULTS_DIR / "correlation_summary.csv"
BOOTSTRAP_CSV = _RESULTS_DIR / "bootstrap_human_vs_cache_correlation_smaller_pct.csv"
OUT_DIR = _RESULTS_DIR

_TS_PREFIX = re.compile(r"^\d{8}_\d{6}")

# Match fig_create/plot_kendall_mean_tau_bars.py (Paul Tol–style families)
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

# Shared x-axis limits across all predictor figures so the bar lengths are
# directly comparable between panels.
FIXED_XLIM: tuple[float, float] = (0.16, 0.5)


def stars_for_bootstrap_p(p: float) -> str:
    if p < 0.0001:
        return "****"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


# (predictor_column, cache_col, human_summary_col, bootstrap_metric)
# Animacy is intentionally ignored.
PREDICTOR_SPECS: list[tuple[str, str, str, str]] = [
    ("mask_area_pixels", "spearman_rho_area", "spearman_rho", "spearman_rho_area"),
    ("center_distance_pixels", "spearman_rho_center_dist", "spearman_rho", "spearman_rho_center_dist"),
    ("person_bin", "pearson_r_person", "pearson_r", "pearson_r_person"),
    ("max_gbvs", "spearman_rho_gbvs", "spearman_rho", "spearman_rho_gbvs"),
]


def strip_cache_timestamp(cache_subfolder: str) -> str:
    return _TS_PREFIX.sub("", str(cache_subfolder).strip())


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


def bootstrap_p_value(pct_human_correlation_smaller: float) -> float:
    p = float(pct_human_correlation_smaller) / 100.0
    if p > 0.5:
        return 1.0 - p
    return p


def load_bootstrap_p_map(bootstrap_df: pd.DataFrame) -> pd.DataFrame:
    wanted = {(p, m) for p, _c, _h, m in PREDICTOR_SPECS}
    rows = []
    for _, r in bootstrap_df.iterrows():
        key = (r["predictor_column"], r["metric"])
        if key not in wanted:
            continue
        pct = float(r["pct_human_correlation_smaller"])
        p = bootstrap_p_value(pct)
        rows.append(
            {
                "cache_subfolder": r["cache_subfolder"],
                "predictor_column": r["predictor_column"],
                "metric": r["metric"],
                "p_bootstrap": p,
            }
        )
    return pd.DataFrame(rows)


def _xlim_with_padding(vals: np.ndarray, human_r: float, pad_frac: float = 0.06) -> tuple[float, float]:
    all_v = np.concatenate([[human_r], vals.astype(float)])
    lo = float(np.nanmin(all_v))
    hi = float(np.nanmax(all_v))
    span = hi - lo if hi > lo else max(abs(lo), abs(hi), 0.1) * 0.1
    pad = max(span * pad_frac, 0.02)
    return lo - pad, hi + pad


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


def plot_one_predictor(
    cache_df: pd.DataFrame,
    human_row: pd.Series,
    bootstrap_sub: pd.DataFrame,
    cache_col: str,
    human_col: str,
    out_base: Path,
    *,
    show_legend: bool,
    xlabel: str | None = None,
    predictor: str | None = None,
) -> list[Path]:
    # The human summary CSV uses a different sign convention than the cache CSV for
    # center_distance_pixels (already stored with the desired sign), so skip the flip.
    if predictor == "center_distance_pixels":
        human_r = float(human_row[human_col])
    else:
        human_r = -float(human_row[human_col])
    models = cache_df[["cache_subfolder", cache_col]].copy()
    models = models.rename(columns={cache_col: "r"})
    models["r"] = pd.to_numeric(models["r"], errors="coerce")
    models["stripped"] = models["cache_subfolder"].map(strip_cache_timestamp)
    models["family"] = models["stripped"].map(model_family)
    models["display_name"] = models["stripped"].map(format_display_name)
    models = models.sort_values("r", ascending=False).reset_index(drop=True)
    if predictor != "center_distance_pixels":
        models["r"] = -models["r"]

    labels = models["display_name"].tolist()
    values = models["r"].to_numpy(dtype=float)
    colors = [FAMILY_COLORS.get(f, FAMILY_COLORS["other"]) for f in models["family"]]

    fig_w_in = 7.2
    fig_h_in = max(3.2, 0.21 * len(labels) + 1.1)
    fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in), layout="constrained")

    p_by_sub: dict[str, float] = {}
    if not bootstrap_sub.empty:
        for _, br in bootstrap_sub.iterrows():
            p_by_sub[str(br["cache_subfolder"])] = float(br["p_bootstrap"])

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
    corr_name = "Pearson r" if human_col == "pearson_r" else "Spearman ρ"
    ax.set_xlabel(xlabel if xlabel is not None else corr_name)
    ax.set_ylabel("Model")
    x0, x1 = FIXED_XLIM
    ax.set_xlim(x0, x1)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
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
        human_r,
        color=REFERENCE_LINE_COLOR,
        linestyle=REFERENCE_LINE_DASHES,
        linewidth=REFERENCE_LINE_WIDTH,
        zorder=5,
        dash_capstyle="butt",
    )

    for yi, (_, row) in enumerate(models.iterrows()):
        sub = str(row["cache_subfolder"])
        p = p_by_sub.get(sub)
        if p is None:
            continue
        star_s = stars_for_bootstrap_p(p)
        if not star_s:
            continue
        x_text = float(row["r"])
        span = x1 - x0
        off = (0.012 + 0.0035 * len(star_s)) * span * (1 if x_text >= 0 else -1)
        ax.text(
            x_text + off,
            yi,
            star_s,
            va="center",
            ha="left" if x_text >= 0 else "right",
            fontsize=11,
            fontweight="bold",
            color="#222222",
            zorder=6,
        )

    if show_legend:
        families_in_data: list[str] = []
        seen: set[str] = set()
        for fam in models["family"]:
            if fam not in seen:
                seen.add(fam)
                families_in_data.append(fam)

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
                label="Human consensus",
            )
        )
        ax.legend(
            handles=legend_handles,
            loc="upper right",
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

    out_base.parent.mkdir(parents=True, exist_ok=True)
    written = save_figure(fig, out_base)
    plt.close(fig)
    return written


def main() -> None:
    apply_nature_style()
    # Bump every font size by +1 over the shared journal style for this figure family.
    mpl.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
        }
    )

    cache_df = pd.read_csv(CACHE_CSV)
    summary_df = pd.read_csv(SUMMARY_CSV)
    boot_df = pd.read_csv(BOOTSTRAP_CSV)
    boot_map = load_bootstrap_p_map(boot_df)

    for predictor, cache_col, human_col, bmetric in PREDICTOR_SPECS:
        human_row = summary_df.loc[summary_df["predictor"] == predictor].iloc[0]
        bsub = boot_map[
            (boot_map["predictor_column"] == predictor) & (boot_map["metric"] == bmetric)
        ]
        out_base = OUT_DIR / f"correlation_{predictor}.png"
        xlabel_pb = (
            r"point biserial correlation $r_{pb}$"
            if predictor == "person_bin"
            else None
        )
        for p in plot_one_predictor(
            cache_df,
            human_row,
            bsub,
            cache_col,
            human_col,
            out_base,
            show_legend=(predictor == "center_distance_pixels"),
            xlabel=xlabel_pb,
            predictor=predictor,
        ):
            print(f"Wrote {p}")


if __name__ == "__main__":
    main()
