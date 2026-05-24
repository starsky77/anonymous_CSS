#!/usr/bin/env python3
"""
Compute per-image rank-order correlation between two mask score CSV files.

Also records whether the mask with the lowest mean_similarity_score in file A
is the same mask (same img_fn) as the mask with the lowest score in file B,
within the set of masks present in both files for that image.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def extract_image_name(img_fn: str) -> str:
    """Extract image_name from {image_name}_{mask_name}_mask_{maskid}.png."""
    # Split from the right to avoid underscores inside image_name/mask_name.
    # Expected tail is "..._mask_<id>.png".
    stem = Path(str(img_fn)).name
    parts = stem.rsplit("_mask_", 1)
    if len(parts) != 2:
        raise ValueError(f"Unexpected img_fn format: {img_fn}")
    left = parts[0]
    image_name, _, _mask_name = left.partition("_")
    if not image_name:
        raise ValueError(f"Could not parse image_name from: {img_fn}")
    return image_name


def load_scores(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required_cols = {"img_fn", "mean_similarity_score"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")

    out = df[["img_fn", "mean_similarity_score"]].copy()
    out["img_fn"] = out["img_fn"].astype(str)
    out["mean_similarity_score"] = pd.to_numeric(out["mean_similarity_score"], errors="coerce")
    out = out.dropna(subset=["mean_similarity_score"])
    out["image_name"] = out["img_fn"].map(extract_image_name)
    return out


def compute_per_image_spearman(file_a: str, file_b: str) -> pd.DataFrame:
    a = load_scores(file_a).rename(columns={"mean_similarity_score": "score_a"})
    b = load_scores(file_b).rename(columns={"mean_similarity_score": "score_b"})

    # Keep only masks that exist in both files.
    merged = a.merge(b[["img_fn", "score_b"]], on="img_fn", how="inner")
    if merged.empty:
        return pd.DataFrame(
            columns=[
                "image_name",
                "num_common_masks",
                "spearman_rank_correlation",
                "lowest_score_mask_same",
                "file_a",
                "file_b",
            ]
        )

    rows = []
    for image_name, group in merged.groupby("image_name", sort=True):
        n = len(group)
        # Spearman is Pearson correlation on average ranks (handles ties).
        if n < 2:
            corr = float("nan")
        else:
            ra = group["score_a"].rank(method="average")
            rb = group["score_b"].rank(method="average")
            std_a = float(ra.std(ddof=1))
            std_b = float(rb.std(ddof=1))
            if std_a == 0.0 or std_b == 0.0:
                corr = float("nan")
            else:
                corr = float(np.corrcoef(ra.to_numpy(), rb.to_numpy())[0, 1])

        # Lowest-score mask per file (first index if tied); compare img_fn identity.
        i_min_a = int(np.nanargmin(group["score_a"].to_numpy()))
        i_min_b = int(np.nanargmin(group["score_b"].to_numpy()))
        img_fn_min_a = str(group["img_fn"].iloc[i_min_a])
        img_fn_min_b = str(group["img_fn"].iloc[i_min_b])
        lowest_same = bool(img_fn_min_a == img_fn_min_b)

        rows.append(
            {
                "image_name": image_name,
                "num_common_masks": n,
                "spearman_rank_correlation": corr,
                "lowest_score_mask_same": lowest_same,
            }
        )

    result = pd.DataFrame(rows).sort_values("image_name").reset_index(drop=True)
    result["file_a"] = Path(file_a).name
    result["file_b"] = Path(file_b).name
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute per-image Spearman rank correlation between two scored_descriptions "
            "CSVs, and whether the lowest-scoring mask in A matches the lowest in B."
        )
    )
    parser.add_argument("file_a", help="First CSV path")
    parser.add_argument("file_b", help="Second CSV path")
    parser.add_argument(
        "-o",
        "--output",
        default="per_image_rank_correlation.csv",
        help="Output CSV path (default: per_image_rank_correlation.csv)",
    )
    args = parser.parse_args()

    result = compute_per_image_spearman(args.file_a, args.file_b)
    result.to_csv(args.output, index=False)
    print(f"Wrote {len(result)} rows to {args.output}")


if __name__ == "__main__":
    main()
