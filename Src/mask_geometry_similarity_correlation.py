#!/usr/bin/env python3
"""
Join mean_similarity_score from scene-description CSVs with mask geometry:
foreground pixel area and Euclidean distance from mask centroid to image center.

Optionally merges mask_animacy_person_labels.csv (Animacy/Person as Y/N) and
reports Pearson/Spearman vs mean_similarity_score with Y=1, N=0.

Optionally merges mask_max_gbvs.csv (max and mean GBVS salience inside each mask) and
reports Pearson/Spearman vs mean_similarity_score.

Single CSV mode: writes mask_geometry_and_similarity.csv and correlation_summary.*
Batch mode (--cache-root): scans Cache subfolders, skips names containing "human",
one output row per CSV under factor_analysis/cache_correlation_by_csv.csv.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

_SRC = Path(__file__).resolve().parent
_REPO_ROOT = _SRC.parent

# Human reference uses the mean of 10 ratings per mask (not 5).
DEFAULT_CSV = (
    _REPO_ROOT
    / "CSS_result"
    / "Cache"
    / "human_filtered_10"
    / "scene_description_human_filtered_all_10_mean_distance_mean_of_10.csv"
)
DEFAULT_MASK_DIR = _REPO_ROOT / "DATA" / "inpaint_image_processed" / "for_vis"
DEFAULT_CACHE_ROOT = _REPO_ROOT / "CSS_result" / "Cache"

# Static input lookup tables live next to this script (under ``Src/factor_analysis/``).
# Output artefacts (CSVs, figures) go under ``results/correlation_bars/`` so that the
# whole repo's machine-generated outputs stay grouped under ``results/``.
_FACTOR_ANALYSIS_DIR = _SRC / "factor_analysis"
DEFAULT_OUT_DIR = _REPO_ROOT / "results" / "correlation_bars"


def _path_relative_to_repo(p: Path) -> str:
    """Return ``p`` as a string relative to the repo root, falling back to
    the file's basename if ``p`` lives outside the repo. Keeps logged paths
    portable across machines and avoids leaking absolute filesystem paths.
    """
    try:
        return str(Path(p).resolve().relative_to(_REPO_ROOT))
    except ValueError:
        return Path(p).name
DEFAULT_LABELS_CSV = _FACTOR_ANALYSIS_DIR / "mask_animacy_person_labels.csv"
DEFAULT_MAX_GBVS_CSV = _FACTOR_ANALYSIS_DIR / "mask_max_gbvs.csv"


def mask_geometry(path: Path, threshold: int = 127) -> tuple[int, float, float, float, int, int]:
    """
    Returns (area_pixels, area_fraction, centroid_x, centroid_y, height, width).
    Foreground: grayscale value > threshold.
    """
    arr = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
    h, w = arr.shape
    fg = arr > threshold
    area = int(fg.sum())
    if area == 0:
        return 0, 0.0, float("nan"), float("nan"), h, w
    ys, xs = np.nonzero(fg)
    cx = float(xs.mean())
    cy = float(ys.mean())
    frac = area / float(h * w)
    return area, frac, cx, cy, h, w


def center_distance(cx: float, cy: float, w: int, h: int) -> float:
    icx = (w - 1) / 2.0
    icy = (h - 1) / 2.0
    return float(np.hypot(cx - icx, cy - icy))


def spearman_rho(x: pd.Series, y: pd.Series) -> float:
    """Spearman correlation without scipy (average ranks for ties)."""
    xr = x.rank(method="average")
    yr = y.rank(method="average")
    return float(xr.corr(yr, method="pearson"))


GeometryCache = dict[str, tuple[int, float, float, float, int, int]]


def yn_to_bin(val: object) -> float:
    """Map Animacy/Person Y/N to 1.0/0.0; unknown -> nan."""
    s = str(val).strip().upper()
    if s == "Y":
        return 1.0
    if s == "N":
        return 0.0
    return float("nan")


def load_animacy_person_labels(path: Path) -> pd.DataFrame | None:
    """
    Columns: mask_basename, animacy_bin, person_bin (0/1).
    Returns None if path is missing.
    """
    if not path.is_file():
        return None
    df = pd.read_csv(path)
    if "image_fn" not in df.columns:
        raise ValueError(f"{path} must contain column image_fn")
    for c in ("Animacy", "Person"):
        if c not in df.columns:
            raise ValueError(f"{path} must contain column {c}")
    out = pd.DataFrame(
        {
            "mask_basename": df["image_fn"].astype(str).map(lambda x: Path(x).name),
            "animacy_bin": df["Animacy"].map(yn_to_bin),
            "person_bin": df["Person"].map(yn_to_bin),
        }
    )
    return out.drop_duplicates(subset=["mask_basename"], keep="first")


def merge_animacy_person_labels(
    feats: pd.DataFrame, labels: pd.DataFrame | None
) -> pd.DataFrame:
    if labels is None or labels.empty:
        return feats
    return feats.merge(labels, on="mask_basename", how="left")


def load_max_gbvs(path: Path) -> pd.DataFrame | None:
    """
    Columns: mask_basename, max_gbvs; optional mean_gbvs (numeric).
    Returns None if path is missing.
    """
    if not path.is_file():
        return None
    df = pd.read_csv(path)
    if "img_fn" not in df.columns or "max_gbvs" not in df.columns:
        raise ValueError(f"{path} must contain columns img_fn and max_gbvs")
    cols: dict[str, pd.Series] = {
        "mask_basename": df["img_fn"].astype(str).map(lambda x: Path(x).name),
        "max_gbvs": pd.to_numeric(df["max_gbvs"], errors="coerce"),
    }
    if "mean_gbvs" in df.columns:
        cols["mean_gbvs"] = pd.to_numeric(df["mean_gbvs"], errors="coerce")
    out = pd.DataFrame(cols)
    return out.drop_duplicates(subset=["mask_basename"], keep="first")


def merge_max_gbvs(feats: pd.DataFrame, gbvs: pd.DataFrame | None) -> pd.DataFrame:
    if gbvs is None or gbvs.empty:
        return feats
    return feats.merge(gbvs, on="mask_basename", how="left")


def feature_rows_from_scores(
    df: pd.DataFrame,
    mask_dir: Path,
    threshold: int,
    geom_cache: GeometryCache | None,
) -> tuple[pd.DataFrame, list[str], GeometryCache]:
    """Build per-mask feature rows; reuse geom_cache across CSVs when provided."""
    if geom_cache is None:
        geom_cache = {}

    rows: list[dict] = []
    missing: list[str] = []

    for _, rec in df.iterrows():
        img_fn = str(rec["img_fn"])
        base = Path(img_fn).name
        mask_path = mask_dir / base
        score = pd.to_numeric(rec["mean_similarity_score"], errors="coerce")

        if not mask_path.is_file():
            missing.append(base)
            continue

        if base not in geom_cache:
            geom_cache[base] = mask_geometry(mask_path, threshold)
        area, frac, cx, cy, h, w = geom_cache[base]

        dist = (
            center_distance(cx, cy, w, h)
            if area > 0 and np.isfinite(cx) and np.isfinite(cy)
            else float("nan")
        )
        rows.append(
            {
                "img_fn": img_fn,
                "mask_basename": base,
                "mean_similarity_score": score,
                "mask_area_pixels": area,
                "mask_area_fraction": frac,
                "mask_centroid_x": cx,
                "mask_centroid_y": cy,
                "image_width": w,
                "image_height": h,
                "center_distance_pixels": dist,
            }
        )

    return pd.DataFrame(rows), missing, geom_cache


def correlation_metrics(valid: pd.DataFrame) -> dict[str, float]:
    """Pearson/Spearman for score vs area, center distance, optional animacy/person, optional GBVS stats."""
    pairs = [
        ("mask_area_pixels", "area"),
        ("center_distance_pixels", "center_dist"),
    ]
    # Animacy is intentionally ignored.
    if "person_bin" in valid.columns:
        pairs.append(("person_bin", "person"))
    if "max_gbvs" in valid.columns:
        pairs.append(("max_gbvs", "gbvs"))
    if "mean_gbvs" in valid.columns:
        pairs.append(("mean_gbvs", "gbvs_mean"))
    out: dict[str, float] = {}
    for col, short in pairs:
        sub = valid[["mean_similarity_score", col]].dropna()
        n = len(sub)
        if n < 3:
            out[f"pearson_r_{short}"] = float("nan")
            out[f"spearman_rho_{short}"] = float("nan")
        else:
            sx = sub["mean_similarity_score"]
            sy = sub[col]
            out[f"pearson_r_{short}"] = float(sx.corr(sy, method="pearson"))
            out[f"spearman_rho_{short}"] = float(spearman_rho(sx, sy))
    return out


def read_scores_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"img_fn", "mean_similarity_score"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(f"{csv_path} missing columns: {sorted(missing_cols)}")
    return df


def eligible_cache_csv_paths(cache_root: Path) -> list[tuple[Path, Path]]:
    """
    Direct child folders of cache_root; skip folder names containing 'human' (any case).
    Returns sorted list of (subfolder, csv_path) for every *.csv in each folder.
    """
    rows: list[tuple[Path, Path]] = []
    if not cache_root.is_dir():
        raise FileNotFoundError(str(cache_root))

    for sub in sorted(cache_root.iterdir()):
        if not sub.is_dir():
            continue
        if "human" in sub.name.lower():
            continue
        for csv_path in sorted(sub.glob("*.csv")):
            rows.append((sub, csv_path))
    return rows


def write_single_run_outputs(
    out: pd.DataFrame,
    missing: list[str],
    valid: pd.DataFrame,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / "mask_geometry_and_similarity.csv", index=False)

    n_lbl = 0
    if "animacy_bin" in valid.columns:
        n_lbl = int(valid["animacy_bin"].notna().sum())

    summary_lines = [
        "Correlation with mean_similarity_score",
        f"n_masks (with score, nonzero mask): {len(valid)}",
        f"n_masks missing on disk: {len(missing)}",
    ]
    if n_lbl:
        summary_lines.append(f"n_masks with Animacy/Person labels: {n_lbl}")
    if "max_gbvs" in valid.columns:
        n_g = int(valid["max_gbvs"].notna().sum())
        summary_lines.append(f"n_masks with max_gbvs: {n_g}")
    if "mean_gbvs" in valid.columns:
        n_m = int(valid["mean_gbvs"].notna().sum())
        summary_lines.append(f"n_masks with mean_gbvs: {n_m}")
    summary_lines.append("")

    pairs = [
        ("mask_area_pixels", "mask area (foreground pixels)"),
        ("center_distance_pixels", "distance mask centroid to image center (pixels)"),
    ]
    # Animacy is intentionally ignored.
    if "person_bin" in valid.columns:
        pairs.append(("person_bin", "Person (Y=1, N=0)"))
    if "max_gbvs" in valid.columns:
        pairs.append(("max_gbvs", "max GBVS inside mask (0–255)"))
    if "mean_gbvs" in valid.columns:
        pairs.append(("mean_gbvs", "mean GBVS inside mask"))

    corr_rows = []
    for col, label in pairs:
        sub = valid[["mean_similarity_score", col]].dropna()
        n = len(sub)
        if n < 3:
            pearson = spearman = float("nan")
        else:
            sx = sub["mean_similarity_score"]
            sy = sub[col]
            pearson = sx.corr(sy, method="pearson")
            spearman = spearman_rho(sx, sy)
        corr_rows.append(
            {
                "predictor": col,
                "predictor_label": label,
                "n": n,
                "pearson_r": pearson,
                "spearman_rho": spearman,
            }
        )
        summary_lines.append(f"{label}")
        summary_lines.append(f"  Pearson r  = {pearson}")
        summary_lines.append(f"  Spearman ρ = {spearman}")
        summary_lines.append("")

    pd.DataFrame(corr_rows).to_csv(out_dir / "correlation_summary.csv", index=False)
    (out_dir / "correlation_summary.txt").write_text(
        "\n".join(summary_lines) + "\n", encoding="utf-8"
    )

    if missing:
        (out_dir / "missing_masks.txt").write_text(
            "\n".join(sorted(missing)) + "\n", encoding="utf-8"
        )


def run_batch_cache(
    cache_root: Path,
    mask_dir: Path,
    out_dir: Path,
    threshold: int,
    labels: pd.DataFrame | None,
    max_gbvs: pd.DataFrame | None,
) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    geom_cache: GeometryCache = {}
    batch_rows: list[dict] = []

    for subfolder, csv_path in eligible_cache_csv_paths(cache_root):
        try:
            df_scores = read_scores_csv(csv_path)
        except ValueError:
            continue

        feats, missing, geom_cache = feature_rows_from_scores(
            df_scores, mask_dir, threshold, geom_cache
        )
        feats = merge_animacy_person_labels(feats, labels)
        feats = merge_max_gbvs(feats, max_gbvs)

        valid = feats.dropna(
            subset=["mean_similarity_score", "mask_area_pixels", "center_distance_pixels"]
        )
        valid = valid[valid["mask_area_pixels"] > 0]

        rel_csv = csv_path.relative_to(cache_root)
        m = correlation_metrics(valid)
        batch_rows.append(
            {
                "cache_subfolder": subfolder.name,
                "csv_relpath": str(rel_csv),
                "csv_path": _path_relative_to_repo(csv_path),
                "n_rows_in_csv": len(df_scores),
                "n_masks_joined": len(feats),
                "n_masks_missing_on_disk": len(missing),
                "n_masks_valid_correlation": len(valid),
                **m,
            }
        )

    batch_df = pd.DataFrame(batch_rows)
    batch_path = out_dir / "cache_correlation_by_csv.csv"
    batch_df.to_csv(batch_path, index=False)
    return batch_df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--mask-dir", type=Path, default=DEFAULT_MASK_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--threshold",
        type=int,
        default=127,
        help="Pixels above this (grayscale) count as foreground.",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=None,
        help="Scan this Cache folder (non-human subfolders); write cache_correlation_by_csv.csv.",
    )
    parser.add_argument(
        "--batch-cache",
        action="store_true",
        help=f"Same as --cache-root {DEFAULT_CACHE_ROOT}",
    )
    parser.add_argument(
        "--labels-csv",
        type=Path,
        default=DEFAULT_LABELS_CSV,
        help="mask_animacy_person_labels.csv (image_fn, Animacy, Person). "
        "Pass a nonexistent path to skip Animacy/Person correlations.",
    )
    parser.add_argument(
        "--max-gbvs-csv",
        type=Path,
        default=DEFAULT_MAX_GBVS_CSV,
        help="mask_max_gbvs.csv (img_fn, max_gbvs; optional mean_gbvs). "
        "Pass a nonexistent path to skip GBVS correlations.",
    )
    args = parser.parse_args()

    labels = load_animacy_person_labels(args.labels_csv)
    max_gbvs_df = load_max_gbvs(args.max_gbvs_csv)

    cache_root = args.cache_root
    if args.batch_cache:
        cache_root = DEFAULT_CACHE_ROOT

    if cache_root is not None:
        run_batch_cache(
            cache_root,
            args.mask_dir,
            args.out_dir,
            args.threshold,
            labels,
            max_gbvs_df,
        )
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = read_scores_csv(args.csv)
    feats, missing, _ = feature_rows_from_scores(df, args.mask_dir, args.threshold, {})
    feats = merge_animacy_person_labels(feats, labels)
    feats = merge_max_gbvs(feats, max_gbvs_df)

    valid = feats.dropna(
        subset=["mean_similarity_score", "mask_area_pixels", "center_distance_pixels"]
    )
    valid = valid[valid["mask_area_pixels"] > 0]

    write_single_run_outputs(feats, missing, valid, args.out_dir)


if __name__ == "__main__":
    main()
