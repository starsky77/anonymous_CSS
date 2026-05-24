#!/usr/bin/env python3
"""
Kendall's τ agreement between first5 and rest semantic_distance slices (same CSV as
``analyze_human_human_lowest_mask_agreement.py``).

Per mask row, mean scores are computed from the first 5 vs remaining semantic_distance
values. Per base image id, τ is computed between those two score vectors over masks
present in both (aligned by ``img_fn``), matching ``pairwise_kendall_tau.per_image_kendall_tau``.

The headline metric is **mean Kendall's τ across images** (and std), not lowest-mask match rate.
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy.stats import kendalltau
except ImportError as e:  # pragma: no cover
    raise SystemExit("scipy is required (pip install scipy). " + str(e)) from e

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


_SRC = Path(__file__).resolve().parent
_REPO_ROOT = _SRC.parent


def default_input_csv() -> Path:
    return (
        _REPO_ROOT
        / "CSS_result"
        / "Cache"
        / "human_filtered_10"
        / "scene_description_human_filtered_all_10_with_mean_similarity_score.csv"
    )


def parse_semantic_distance(cell: object) -> list[float]:
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return []
    s = str(cell).strip()
    if not s:
        return []
    try:
        parsed = ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return []
    if not isinstance(parsed, (list, tuple)):
        return []
    out: list[float] = []
    for v in parsed:
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if np.isfinite(fv):
            out.append(fv)
    return out


def base_image_id_from_img_fn(img_fn: str) -> str:
    return str(img_fn).split("_", 1)[0]


def build_mask_score_table(csv_path: Path) -> pd.DataFrame:
    """One row per mask with mean first5 vs mean rest scores."""
    df = pd.read_csv(csv_path)
    needed = {"img_fn", "semantic_distance"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} missing columns: {sorted(missing)}")

    rows: list[dict[str, object]] = []
    for _, r in df.loc[:, ["img_fn", "semantic_distance"]].iterrows():
        img_fn = str(r["img_fn"]).strip()
        if not img_fn:
            continue
        vals = parse_semantic_distance(r["semantic_distance"])
        first5 = vals[:5]
        rest = vals[5:]
        if not first5 or not rest:
            continue
        rows.append(
            {
                "img_fn": img_fn,
                "base_image_id": base_image_id_from_img_fn(img_fn),
                "score_first5": float(np.mean(first5)),
                "score_rest": float(np.mean(rest)),
            }
        )
    return pd.DataFrame(rows)


def per_image_kendall_first5_rest(mask_scores: pd.DataFrame) -> pd.DataFrame:
    """Same τ / lowest-mask logic as ``pairwise_kendall_tau.per_image_kendall_tau``."""
    if mask_scores.empty:
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
    for image_name, group in mask_scores.groupby("base_image_id", sort=True):
        n = len(group)
        if n < 2:
            tau = np.nan
            pval = np.nan
        else:
            tau, pval = kendalltau(
                group["score_first5"].to_numpy(dtype=float),
                group["score_rest"].to_numpy(dtype=float),
            )
        i_min_a = int(np.nanargmin(group["score_first5"].to_numpy()))
        i_min_b = int(np.nanargmin(group["score_rest"].to_numpy()))
        img_fn_min_a = str(group["img_fn"].iloc[i_min_a])
        img_fn_min_b = str(group["img_fn"].iloc[i_min_b])
        lowest_same = bool(img_fn_min_a == img_fn_min_b)

        rows.append(
            {
                "image_name": str(image_name),
                "num_common_masks": n,
                "kendall_tau": float(tau) if tau == tau else np.nan,
                "kendall_pvalue": float(pval) if pval == pval else np.nan,
                "lowest_score_mask_same": lowest_same,
            }
        )

    return pd.DataFrame(rows).sort_values("image_name").reset_index(drop=True)


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Mean Kendall's τ across images between first5 and rest semantic_distance "
            "means (per mask), analogous to pairwise_kendall_tau."
        )
    )
    parser.add_argument("--input-csv", type=Path, default=default_input_csv())
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_SRC / "eval_output_tau" / "first5_vs_rest",
        help="Directory for output CSV files and histogram.",
    )
    parser.add_argument(
        "--hist-bins",
        type=int,
        default=30,
        help="Histogram bins (range [-1, 1]).",
    )
    args = parser.parse_args()

    if not args.input_csv.is_file():
        raise SystemExit(f"File not found: {args.input_csv}")

    mask_scores = build_mask_score_table(args.input_csv)
    per_img = per_image_kendall_first5_rest(mask_scores)

    taus = pd.to_numeric(per_img["kendall_tau"], errors="coerce").to_numpy()
    valid = taus[np.isfinite(taus)]
    low = per_img["lowest_score_mask_same"]
    if low.dtype != bool:
        low = low.astype(str).str.strip().str.lower().isin(("true", "1", "yes"))
    n_img = int(len(per_img))
    n_valid = int(np.isfinite(taus).sum())
    n_low_true = int(low.sum())
    low_acc = n_low_true / n_img if n_img else float("nan")

    _SCRIPT_DIR = Path(__file__).resolve().parent
    _ROOT = _SCRIPT_DIR.parent
    try:
        input_csv_str = str(args.input_csv.resolve().relative_to(_ROOT))
    except ValueError:
        input_csv_str = Path(args.input_csv).name
    summary = pd.DataFrame(
        [
            {
                "comparison": "first5_vs_rest",
                "input_csv": input_csv_str,
                "n_images": n_img,
                "n_valid_tau": n_valid,
                "mean_kendall_tau": float(np.nanmean(taus)) if valid.size else np.nan,
                "std_kendall_tau": float(np.nanstd(taus, ddof=1))
                if valid.size > 1
                else (0.0 if valid.size == 1 else np.nan),
                "lowest_score_mask_same_count": n_low_true,
                "lowest_score_mask_same_accuracy": low_acc,
            }
        ]
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    per_image_out = args.out_dir / "first5_vs_rest_kendall_per_image.csv"
    summary_out = args.out_dir / "first5_vs_rest_kendall_summary.csv"
    hist_out = args.out_dir / "first5_vs_rest_kendall_hist.png"

    per_img.to_csv(per_image_out, index=False)
    summary.to_csv(summary_out, index=False)
    plot_kendall_histogram(
        taus,
        hist_out,
        title="Kendall τ (first5 vs rest)\nper image",
        bins=args.hist_bins,
    )

    mean_tau = summary.loc[0, "mean_kendall_tau"]
    print(f"Images:           {n_img}")
    print(f"Valid τ:          {n_valid}")
    print(f"Mean Kendall τ:   {mean_tau}")
    print(f"Std Kendall τ:    {summary.loc[0, 'std_kendall_tau']}")
    print(f"Lowest-mask acc:  {low_acc:.4f} ({n_low_true}/{n_img})")
    print(f"Wrote: {per_image_out}")
    print(f"Wrote: {summary_out}")
    if _HAS_MPL:
        print(f"Wrote: {hist_out}")


if __name__ == "__main__":
    main()
