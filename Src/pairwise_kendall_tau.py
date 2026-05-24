#!/usr/bin/env python3
"""
Per-image Kendall's τ between mask-level scores (same merge as ``mask_rank_correlation``).

Modes:
  * ``pairwise`` — each unordered pair of LLM runs from ``eval_outputs/*_rank_correlation.csv``.
  * ``human`` — each LLM run vs a fixed human ``scored_descriptions`` CSV (Kendall τ per image).
  * ``both`` — run pairwise then human-vs-model.

Writes one CSV per comparison (under ``--out-dir``). Pairwise mode also
writes one histogram PNG per pair. Human-vs-model mode writes per-pair
CSVs into a ``human_vs_model/`` subfolder plus a single combined figure
``human_vs_models_kendall_hist_grid.png`` arranged as an N×M grid of
subplots (default 7×3), each titled ``Human v.s. {Model Name}``. If a
Human-vs-Human reference CSV is provided via ``--human-vs-human-csv``,
its per-image τ distribution is appended as the final ``Human v.s. Human``
panel.

Each per-image row also includes ``lowest_score_mask_same`` (whether the
lowest-``mean_similarity_score`` mask matches between the two runs for that
image), matching ``mask_rank_correlation``. Summary CSVs add
``lowest_score_mask_same_count`` and ``lowest_score_mask_same_accuracy``.
"""

from __future__ import annotations

import argparse
import re
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from mask_rank_correlation import load_scores

try:
    from scipy.stats import kendalltau
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "scipy is required (pip install scipy). " + str(e)
    ) from e

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


_REPO_ROOT = _SRC.parent
DEFAULT_OUT = _SRC / "eval_output_tau"
DEFAULT_HUMAN_SCORED = (
    _REPO_ROOT
    / "CSS_result"
    / "Cache"
    / "human_filtered_10"
    / "scene_description_human_filtered_all_10_with_mean_similarity_score.csv"
)
DEFAULT_HUMAN_VS_HUMAN_CSV = (
    _SRC
    / "eval_output_tau"
    / "first5_vs_rest"
    / "first5_vs_rest_kendall_per_image.csv"
)


def model_run_from_rank_path(path: Path) -> str:
    return path.stem.replace("_rank_correlation", "")


def discover_model_runs(eval_dir: Path) -> list[str]:
    paths = sorted(eval_dir.glob("*_rank_correlation.csv"))
    if not paths:
        raise SystemExit(f"No *_rank_correlation.csv under {eval_dir}")
    return [model_run_from_rank_path(p) for p in paths]


def is_human_run_name(name: str) -> bool:
    return "human" in name.lower()


def resolve_scored_descriptions(cache_root: Path, model_run: str) -> Path | None:
    d = cache_root / model_run
    if not d.is_dir():
        return None
    matches = sorted(d.glob("scored_descriptions*.csv"))
    return matches[0] if matches else None


def safe_pair_slug(a: str, b: str) -> str:
    """Filesystem-safe slug for a model pair (order preserved: a vs b)."""
    raw = f"{a}__vs__{b}"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", raw)[:240]


def display_model_name(name: str) -> str:
    """Pretty model name for figure titles.

    * Strip a leading run of digits/underscores (e.g. timestamp prefixes
      like ``20260428_144419gpt-5.4`` -> ``gpt-5.4``).
    * Collapse the redundant HuggingFace org prefix ``Qwen_Qwen3`` to ``Qwen3``.
    """
    s = re.sub(r"^[\d_]+", "", name)
    s = s.replace("Qwen_Qwen3", "Qwen3")
    return s


def per_image_kendall_tau(path_a: Path, path_b: Path) -> pd.DataFrame:
    a = load_scores(str(path_a)).rename(columns={"mean_similarity_score": "score_a"})
    b = load_scores(str(path_b)).rename(columns={"mean_similarity_score": "score_b"})
    merged = a.merge(b[["img_fn", "score_b"]], on="img_fn", how="inner")
    if merged.empty:
        return pd.DataFrame(
            columns=[
                "image_name",
                "num_common_masks",
                "kendall_tau",
                "kendall_pvalue",
                "lowest_score_mask_same",
            ]
        )

    rows: list[dict[str, object]] = []
    for image_name, group in merged.groupby("image_name", sort=True):
        n = len(group)
        if n < 2:
            tau = np.nan
            pval = np.nan
        else:
            tau, pval = kendalltau(
                group["score_a"].to_numpy(dtype=float),
                group["score_b"].to_numpy(dtype=float),
            )
        # Same lowest-mask check as ``mask_rank_correlation.compute_per_image_spearman``.
        i_min_a = int(np.nanargmin(group["score_a"].to_numpy()))
        i_min_b = int(np.nanargmin(group["score_b"].to_numpy()))
        img_fn_min_a = str(group["img_fn"].iloc[i_min_a])
        img_fn_min_b = str(group["img_fn"].iloc[i_min_b])
        lowest_same = bool(img_fn_min_a == img_fn_min_b)

        rows.append(
            {
                "image_name": image_name,
                "num_common_masks": n,
                "kendall_tau": float(tau) if tau == tau else np.nan,
                "kendall_pvalue": float(pval) if pval == pval else np.nan,
                "lowest_score_mask_same": lowest_same,
            }
        )

    return pd.DataFrame(rows).sort_values("image_name").reset_index(drop=True)


def _write_pair_result(
    df: pd.DataFrame,
    label_a: str,
    label_b: str,
    path_a: Path,
    path_b: Path,
    slug: str,
    out_dir: Path,
    artifact_root: Path,
    hist_bins: int,
    summary_rows: list[dict[str, object]],
    make_histogram: bool = True,
    histogram_rel_override: str | None = None,
) -> np.ndarray:
    """Write per-pair CSV + (optionally) histogram; return per-image τ array."""
    df = df.copy()
    df["model_a"] = label_a
    df["model_b"] = label_b
    df["scored_descriptions_a"] = path_a.name
    df["scored_descriptions_b"] = path_b.name
    front = [
        "model_a",
        "model_b",
        "image_name",
        "num_common_masks",
        "kendall_tau",
        "kendall_pvalue",
        "lowest_score_mask_same",
    ]
    rest = [c for c in df.columns if c not in front]
    df = df[front + rest]

    csv_path = out_dir / f"{slug}_per_image_kendall.csv"
    hist_path = out_dir / f"{slug}_kendall_hist.png"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)

    taus = pd.to_numeric(df["kendall_tau"], errors="coerce").to_numpy()
    valid = taus[np.isfinite(taus)]
    low = df["lowest_score_mask_same"]
    if low.dtype != bool:
        low = low.astype(str).str.strip().str.lower().isin(("true", "1", "yes"))
    n_low_true = int(low.sum())
    n_img = int(len(df))
    low_acc = n_low_true / n_img if n_img else float("nan")
    try:
        rel_csv = str(csv_path.relative_to(artifact_root))
    except ValueError:
        rel_csv = csv_path.name
    if histogram_rel_override is not None:
        rel_hist = histogram_rel_override
    else:
        try:
            rel_hist = str(hist_path.relative_to(artifact_root))
        except ValueError:
            rel_hist = hist_path.name

    summary_rows.append(
        {
            "model_a": label_a,
            "model_b": label_b,
            "n_images": n_img,
            "n_valid_tau": int(np.isfinite(taus).sum()),
            "mean_kendall_tau": float(np.nanmean(taus)) if valid.size else np.nan,
            "std_kendall_tau": float(np.nanstd(taus, ddof=1))
            if valid.size > 1
            else (0.0 if valid.size == 1 else np.nan),
            "lowest_score_mask_same_count": n_low_true,
            "lowest_score_mask_same_accuracy": low_acc,
            "per_image_csv": rel_csv,
            "histogram_png": rel_hist,
            "skipped": False,
        }
    )

    if make_histogram:
        title = f"Kendall τ per image\n{slug[:100]}"
        plot_kendall_histogram(taus, hist_path, title=title, bins=hist_bins)
        print(f"Wrote {csv_path} + {hist_path.name}")
    else:
        print(f"Wrote {csv_path}")
    return taus


def plot_kendall_histogram(
    taus: np.ndarray,
    out_path: Path,
    title: str,
    bins: int,
) -> None:
    if not _HAS_MPL:
        print(f"matplotlib not installed; skip histogram: {out_path.name}")
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    t = taus[np.isfinite(taus)]
    if t.size:
        plt.hist(t, bins=bins, range=(-1.0, 1.0), edgecolor="black", alpha=0.85)
    plt.xlabel("Kendall's τ (per image)")
    plt.ylabel("count")
    plt.title(title)
    plt.xlim(-1.05, 1.05)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_kendall_histograms_grid(
    items: list[tuple[str, np.ndarray]],
    out_path: Path,
    bins: int,
    nrows: int,
    ncols: int,
    suptitle: str | None = None,
) -> None:
    """Render a grid of per-model Kendall τ histograms in a single figure.

    Each entry of ``items`` is ``(model_name, taus)``; the subplot title is
    ``"Human v.s. {model_name}"``. Empty cells (when ``len(items) < nrows*ncols``)
    are turned off.
    """
    if not _HAS_MPL:
        print(f"matplotlib not installed; skip combined histogram: {out_path.name}")
        return
    if not items:
        print(f"No items to plot; skip combined histogram: {out_path.name}")
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(ncols * 6.0, nrows * 4.5),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    try:
        fig.get_layout_engine().set(w_pad=0.25, h_pad=0.3, wspace=0.08, hspace=0.12)
    except Exception:
        pass
    axes_flat = np.atleast_1d(axes).flatten()

    n_cells = len(axes_flat)
    if len(items) > n_cells:
        print(
            f"Warning: {len(items)} items but only {n_cells} subfigures "
            f"({nrows}x{ncols}); extras will be dropped.",
            file=sys.stderr,
        )

    n_used = 0
    for idx, (model_name, taus) in enumerate(items[:n_cells]):
        ax = axes_flat[idx]
        t = taus[np.isfinite(taus)]
        if t.size:
            ax.hist(t, bins=bins, range=(-1.0, 1.0), edgecolor="black", alpha=0.85)
        ax.set_title(f"Human v.s. {display_model_name(model_name)}", fontsize=18)
        ax.set_xlim(-1.05, 1.05)
        ax.set_xticks(np.linspace(-1.0, 1.0, 5))
        ax.set_xlabel("Kendall's τ", fontsize=18)
        ax.set_ylabel("count", fontsize=18)
        ax.tick_params(
            axis="both",
            labelsize=14,
            labelbottom=True,
            labelleft=True,
        )
        n_used += 1

    for j in range(n_used, n_cells):
        axes_flat[j].axis("off")

    if suptitle:
        fig.suptitle(suptitle, fontsize=20)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    pdf_path = out_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Wrote vector copy: {pdf_path}")
    plt.close(fig)


def run_pairwise(
    model_runs: list[str],
    paths_map: dict[str, Path | None],
    out_dir: Path,
    artifact_root: Path,
    hist_bins: int,
) -> None:
    summary_rows: list[dict[str, object]] = []
    for a, b in combinations(model_runs, 2):
        pa, pb = paths_map[a], paths_map[b]
        slug = safe_pair_slug(a, b)

        if pa is None or pb is None:
            print(f"Skip pair {a} vs {b}: missing scored CSV", file=sys.stderr)
            summary_rows.append(
                {
                    "model_a": a,
                    "model_b": b,
                    "n_images": np.nan,
                    "n_valid_tau": 0,
                    "mean_kendall_tau": np.nan,
                    "std_kendall_tau": np.nan,
                    "lowest_score_mask_same_count": np.nan,
                    "lowest_score_mask_same_accuracy": np.nan,
                    "per_image_csv": f"{slug}_per_image_kendall.csv",
                    "histogram_png": f"{slug}_kendall_hist.png",
                    "skipped": True,
                }
            )
            continue

        df = per_image_kendall_tau(pa, pb)
        _write_pair_result(
            df, a, b, pa, pb, slug, out_dir, artifact_root, hist_bins, summary_rows
        )

    summary_path = out_dir / "pairwise_kendall_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"Wrote summary: {summary_path}")


def _load_taus_csv(path: Path) -> np.ndarray:
    """Load a per-image Kendall τ array from a CSV with a ``kendall_tau`` column."""
    df = pd.read_csv(path)
    if "kendall_tau" not in df.columns:
        raise SystemExit(
            f"{path} is missing required column 'kendall_tau' "
            f"(have: {list(df.columns)})"
        )
    return pd.to_numeric(df["kendall_tau"], errors="coerce").to_numpy()


def run_human_vs_models(
    model_runs: list[str],
    paths_map: dict[str, Path | None],
    human_scored: Path,
    out_dir: Path,
    artifact_root: Path,
    hist_bins: int,
    grid_rows: int = 7,
    grid_cols: int = 3,
    human_vs_human_csv: Path | None = None,
) -> None:
    if not human_scored.is_file():
        raise SystemExit(f"Human scored_descriptions not found: {human_scored}")

    human_label = human_scored.parent.name
    human_path = human_scored.resolve()
    summary_rows: list[dict[str, object]] = []
    out_sub = out_dir / "human_vs_model"
    out_sub.mkdir(parents=True, exist_ok=True)

    combined_hist_path = out_sub / "human_vs_models_kendall_hist_grid.png"
    try:
        combined_hist_rel = str(combined_hist_path.relative_to(artifact_root))
    except ValueError:
        combined_hist_rel = combined_hist_path.name

    hist_items: list[tuple[str, np.ndarray]] = []

    for m in model_runs:
        if is_human_run_name(m):
            print(f"Skip LLM slot {m}: looks like human baseline folder", file=sys.stderr)
            continue
        pm = paths_map.get(m)
        if pm is None:
            print(f"Skip human vs {m}: missing model scored CSV", file=sys.stderr)
            slug = safe_pair_slug(human_label, m)
            summary_rows.append(
                {
                    "model_a": human_label,
                    "model_b": m,
                    "n_images": np.nan,
                    "n_valid_tau": 0,
                    "mean_kendall_tau": np.nan,
                    "std_kendall_tau": np.nan,
                    "lowest_score_mask_same_count": np.nan,
                    "lowest_score_mask_same_accuracy": np.nan,
                    "per_image_csv": f"human_vs_model/{slug}_per_image_kendall.csv",
                    "histogram_png": combined_hist_rel,
                    "skipped": True,
                }
            )
            continue

        # Human scores as score_a, model as score_b (same merge as pairwise).
        df = per_image_kendall_tau(human_path, pm)
        slug = safe_pair_slug(human_label, m)
        taus = _write_pair_result(
            df,
            human_label,
            m,
            human_path,
            pm,
            slug,
            out_sub,
            artifact_root,
            hist_bins,
            summary_rows,
            make_histogram=False,
            histogram_rel_override=combined_hist_rel,
        )
        hist_items.append((m, taus))

    if human_vs_human_csv is not None:
        if human_vs_human_csv.is_file():
            try:
                hh_taus = _load_taus_csv(human_vs_human_csv)
                hist_items.append(("Human", hh_taus))
                print(f"Appended Human-vs-Human panel from {human_vs_human_csv}")
            except SystemExit as e:
                print(f"Skip Human-vs-Human panel: {e}", file=sys.stderr)
        else:
            print(
                f"Skip Human-vs-Human panel: file not found {human_vs_human_csv}",
                file=sys.stderr,
            )

    plot_kendall_histograms_grid(
        hist_items,
        combined_hist_path,
        bins=hist_bins,
        nrows=grid_rows,
        ncols=grid_cols,
        suptitle=None,
    )
    print(f"Wrote combined histogram: {combined_hist_path}")

    summary_path = out_sub / "kendall_vs_human_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"Wrote summary: {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Per-image Kendall τ for model pairs and/or each model vs human."
    )
    parser.add_argument(
        "--mode",
        choices=("pairwise", "human", "both"),
        default="pairwise",
        help="pairwise: LLM–LLM; human: human scored CSV vs each LLM; both.",
    )
    parser.add_argument(
        "--human-scored",
        type=Path,
        default=DEFAULT_HUMAN_SCORED,
        help="Human scored_descriptions CSV (used when --mode is human or both).",
    )
    parser.add_argument(
        "--eval-dir",
        type=Path,
        default=_SRC / "eval_outputs",
        help="Directory with *_rank_correlation.csv (used only to list model_run names).",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=_SRC.parent / "CSS_result" / "Cache",
        help="Cache root with <model_run>/scored_descriptions*.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT,
        help="Output directory (default: Src/eval_output_tau)",
    )
    parser.add_argument(
        "--hist-bins",
        type=int,
        default=30,
        help="Histogram bin count (fixed range [-1, 1]).",
    )
    parser.add_argument(
        "--human-grid-rows",
        type=int,
        default=7,
        help="Rows in the combined human-vs-model histogram grid.",
    )
    parser.add_argument(
        "--human-grid-cols",
        type=int,
        default=3,
        help="Columns in the combined human-vs-model histogram grid.",
    )
    parser.add_argument(
        "--human-vs-human-csv",
        type=Path,
        default=DEFAULT_HUMAN_VS_HUMAN_CSV,
        help=(
            "Per-image Kendall τ CSV for a Human-vs-Human reference run; "
            "appended as the final panel (titled 'Human v.s. Human') in the grid. "
            "Pass an empty/non-existent path to skip."
        ),
    )
    args = parser.parse_args()

    if not args.eval_dir.is_dir():
        raise SystemExit(f"Not a directory: {args.eval_dir}")
    if not args.cache_root.is_dir():
        raise SystemExit(f"Not a directory: {args.cache_root}")

    model_runs = discover_model_runs(args.eval_dir)
    paths_map: dict[str, Path | None] = {
        m: resolve_scored_descriptions(args.cache_root, m) for m in model_runs
    }
    for m, p in paths_map.items():
        if p is None:
            print(f"Warning: missing scored_descriptions under {args.cache_root / m}", file=sys.stderr)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode in ("pairwise", "both"):
        run_pairwise(model_runs, paths_map, args.out_dir, args.out_dir, args.hist_bins)
    if args.mode in ("human", "both"):
        run_human_vs_models(
            model_runs,
            paths_map,
            args.human_scored,
            args.out_dir,
            args.out_dir,
            args.hist_bins,
            grid_rows=args.human_grid_rows,
            grid_cols=args.human_grid_cols,
            human_vs_human_csv=args.human_vs_human_csv,
        )


if __name__ == "__main__":
    main()
