#!/usr/bin/env python3
"""
Bootstrap (n=1306 with replacement) comparison of correlation coefficients
between DEFAULT_CSV (human) and each scored CSV under DEFAULT_CACHE_ROOT
(non-human subfolders), matching mask_geometry_similarity_correlation.py.

For each bootstrap draw, Pearson and Spearman are computed between
mean_similarity_score and each predictor (area, center distance, animacy,
person, max_gbvs, mean_gbvs when available). Count iterations where human r
is strictly smaller than the model r; report percentage over --n-bootstrap.

Output: long CSV (model, metric, counts, pct).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore[misc, assignment]

import mask_geometry_similarity_correlation as mgsc


def pearson_numpy(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return float("nan")
    x = x[m]
    y = y[m]
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman_numpy(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return float("nan")
    rx = pd.Series(x[m]).rank(method="average").to_numpy(dtype=np.float64)
    ry = pd.Series(y[m]).rank(method="average").to_numpy(dtype=np.float64)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def build_aligned_table(
    human_csv: Path,
    cache_root: Path,
    mask_dir: Path,
    threshold: int,
    labels: pd.DataFrame | None,
    max_gbvs: pd.DataFrame | None,
) -> tuple[
    list[tuple[Path, Path]],
    np.ndarray,
    np.ndarray,
    dict[Path, np.ndarray],
    list[str],
    list[tuple[str, str, str]],
]:
    """
    Returns:
      eligible: list of (subfolder, csv_path) for models
      P: (n, n_pred) predictor columns in fixed order
      score_human: (n,)
      model_scores: path -> (n,) aligned to same row order as human valid rows
      pred_cols: predictor column names (same order as P columns)
      metric_specs: list of (metric_key, method, predictor_col) for reporting
    """
    df_h = mgsc.read_scores_csv(human_csv)
    feats_h, _, _ = mgsc.feature_rows_from_scores(
        df_h, mask_dir, threshold, None
    )
    feats_h = mgsc.merge_animacy_person_labels(feats_h, labels)
    feats_h = mgsc.merge_max_gbvs(feats_h, max_gbvs)

    valid = feats_h.dropna(
        subset=[
            "mean_similarity_score",
            "mask_area_pixels",
            "center_distance_pixels",
        ]
    )
    valid = valid[valid["mask_area_pixels"] > 0]
    valid = valid.copy()
    valid["mask_basename"] = valid["mask_basename"].astype(str)

    pred_cols: list[str] = [
        "mask_area_pixels",
        "center_distance_pixels",
    ]
    # Animacy is intentionally ignored.
    for c in ("person_bin", "max_gbvs", "mean_gbvs"):
        if c in valid.columns:
            pred_cols.append(c)

    eligible = mgsc.eligible_cache_csv_paths(cache_root)
    if not eligible:
        raise RuntimeError(f"No eligible model CSVs under {cache_root}")

    basename_order = valid["mask_basename"].to_numpy()
    n = len(valid)
    P = np.column_stack(
        [pd.to_numeric(valid[c], errors="coerce").to_numpy(dtype=np.float64) for c in pred_cols]
    )
    score_human = pd.to_numeric(valid["mean_similarity_score"], errors="coerce").to_numpy(
        dtype=np.float64
    )

    model_scores: dict[Path, np.ndarray] = {}
    for sub, path in eligible:
        df_m = mgsc.read_scores_csv(path)
        mp = {
            Path(str(k)).name: float(v)
            for k, v in zip(
                df_m["img_fn"].astype(str),
                pd.to_numeric(df_m["mean_similarity_score"], errors="coerce"),
            )
        }
        arr = np.array([mp.get(b, np.nan) for b in basename_order], dtype=np.float64)
        model_scores[path] = arr

    row_ok = np.isfinite(score_human) & np.all(np.isfinite(P), axis=1)
    for path in model_scores:
        row_ok &= np.isfinite(model_scores[path])

    if not row_ok.any():
        raise RuntimeError("No rows with complete human, model scores and predictors.")

    P = P[row_ok]
    score_human = score_human[row_ok]
    for path in list(model_scores.keys()):
        model_scores[path] = model_scores[path][row_ok]

    metric_specs: list[tuple[str, str, str]] = []
    for col in pred_cols:
        short = {
            "mask_area_pixels": "area",
            "center_distance_pixels": "center_dist",
            "animacy_bin": "animacy",
            "person_bin": "person",
            "max_gbvs": "gbvs",
            "mean_gbvs": "gbvs_mean",
        }[col]
        metric_specs.append((f"pearson_r_{short}", "pearson", col))
        metric_specs.append((f"spearman_rho_{short}", "spearman", col))

    return eligible, P, score_human, model_scores, pred_cols, metric_specs


def run_bootstrap(
    eligible: list[tuple[Path, Path]],
    P: np.ndarray,
    score_human: np.ndarray,
    model_scores: dict[Path, np.ndarray],
    pred_cols: list[str],
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[dict[Path, dict[str, int]], int]:
    n = P.shape[0]
    n_models = len(eligible)
    n_metrics = 2 * len(pred_cols)
    counts = np.zeros((n_models, n_metrics), dtype=np.int64)
    metric_keys = []
    for col in pred_cols:
        short = {
            "mask_area_pixels": "area",
            "center_distance_pixels": "center_dist",
            "animacy_bin": "animacy",
            "person_bin": "person",
            "max_gbvs": "gbvs",
            "mean_gbvs": "gbvs_mean",
        }[col]
        metric_keys.append(f"pearson_r_{short}")
        metric_keys.append(f"spearman_rho_{short}")

    corr_fns = (pearson_numpy, spearman_numpy)

    iterator = range(n_bootstrap)
    if tqdm is not None:
        iterator = tqdm(iterator, desc="Bootstrap", unit="iter")

    paths = [p for _, p in eligible]

    for _ in iterator:
        idx = rng.integers(0, n, size=n, endpoint=False)
        P_b = P[idx]
        h_b = score_human[idx]

        for mi, path in enumerate(paths):
            m_b = model_scores[path][idx]
            for pj, _pred_col in enumerate(pred_cols):
                y = P_b[:, pj]
                for k, fn in enumerate(corr_fns):
                    rh = fn(h_b, y)
                    rm = fn(m_b, y)
                    if np.isfinite(rh) and np.isfinite(rm) and rh < rm:
                        counts[mi, 2 * pj + k] += 1

    out_dict: dict[Path, dict[str, int]] = {}
    for mi, path in enumerate(paths):
        out_dict[path] = {metric_keys[j]: int(counts[mi, j]) for j in range(n_metrics)}

    return out_dict, n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--human-csv", type=Path, default=mgsc.DEFAULT_CSV)
    parser.add_argument("--cache-root", type=Path, default=mgsc.DEFAULT_CACHE_ROOT)
    parser.add_argument("--mask-dir", type=Path, default=mgsc.DEFAULT_MASK_DIR)
    parser.add_argument("--labels-csv", type=Path, default=mgsc.DEFAULT_LABELS_CSV)
    parser.add_argument("--max-gbvs-csv", type=Path, default=mgsc.DEFAULT_MAX_GBVS_CSV)
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=mgsc.DEFAULT_OUT_DIR / "bootstrap_human_vs_cache_correlation_smaller_pct.csv",
    )
    parser.add_argument("--threshold", type=int, default=127)
    parser.add_argument("--n-bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    labels = mgsc.load_animacy_person_labels(args.labels_csv)
    max_gbvs_df = mgsc.load_max_gbvs(args.max_gbvs_csv)

    eligible, P, score_human, model_scores, pred_cols, metric_specs = build_aligned_table(
        args.human_csv,
        args.cache_root,
        args.mask_dir,
        args.threshold,
        labels,
        max_gbvs_df,
    )
    rng = np.random.default_rng(args.seed)
    counts_by_path, n_rows = run_bootstrap(
        eligible,
        P,
        score_human,
        model_scores,
        pred_cols,
        args.n_bootstrap,
        rng,
    )

    rows_out: list[dict] = []
    human_csv_str = mgsc._path_relative_to_repo(args.human_csv)
    for (subfolder, csv_path) in eligible:
        rel = csv_path.relative_to(args.cache_root)
        for metric_key, method, pred_col in metric_specs:
            cnt = counts_by_path[csv_path][metric_key]
            pct = 100.0 * cnt / args.n_bootstrap
            rows_out.append(
                {
                    "human_csv": human_csv_str,
                    "cache_subfolder": subfolder.name,
                    "model_csv_relpath": str(rel),
                    "model_csv_path": mgsc._path_relative_to_repo(csv_path),
                    "metric": metric_key,
                    "method": method,
                    "predictor_column": pred_col,
                    "n_bootstrap": args.n_bootstrap,
                    "n_rows_aligned": n_rows,
                    "n_human_correlation_smaller": cnt,
                    "pct_human_correlation_smaller": pct,
                }
            )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows_out).to_csv(args.out_csv, index=False)


if __name__ == "__main__":
    main()
