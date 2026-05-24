#!/usr/bin/env python3
"""
Permutation test for the per-model univariate correlation reported in
``regression_acc_gap_on_predictor_gaps.py``.

Algorithm (per predictor P, repeated n_permutations times)
----------------------------------------------------------
1. Randomly shuffle the per-mask values of P across the aligned set of
   masks (i.e., the same valid sample used by the regression).
2. With the *permuted* predictor, recompute the per-mask correlation
   versus human ``mean_similarity_score`` and versus each model's
   ``mean_similarity_score``, using the metric configured for P in the
   regression script (e.g. spearman_rho for mask_area_pixels) and the
   same sign convention (``flip_sign``).
3. Form the permuted per-model gap vector
       gap_perm[model] = signed(human_r_perm) - signed(model_r_perm)
4. Compute the univariate Pearson r and Spearman rho between
   ``gap_perm`` and ``acc_gap`` across models.

The 10 000 univariate correlations form the null distribution: how often
a *random* re-labeling of the predictor would by-chance produce a
correlation between gap_<P> and acc_gap that is as large (or larger)
than the observed one.

Predictors match the regression script (``regression_acc_gap_on_predictor_gaps.py``):
mask_area_pixels, center_distance_pixels, person_bin, max_gbvs.
animacy_bin and mean_gbvs are intentionally excluded.

Outputs (under ``results/regression_permutation/``):
  permutation_summary.csv           one row per (predictor, uni_metric)
  permutation_summary.txt           human-readable summary
  observed_gaps_per_model.csv       per-model observed gap_<P> and acc_gap
  permutation_null_distribution.csv long form, all 10 000 values per cell (--save-null)
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _DIR.parents[1]
_SRC = _REPO_ROOT / "Src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import mask_geometry_similarity_correlation as mgsc

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore[misc, assignment]


DEFAULT_KENDALL_CSV = (
    _REPO_ROOT / "Src" / "eval_output_tau" / "human_vs_model" / "kendall_vs_human_summary.csv"
)
HUMAN_HUMAN_CONSISTENCY_X = 0.7296
OUT_DIR = _REPO_ROOT / "results" / "regression_permutation"

_TS_PREFIX = re.compile(r"^\d{8}_\d{6}")


@dataclass(frozen=True)
class PredictorSpec:
    name: str        # column name in the per-mask feature table
    metric: str      # "pearson" or "spearman"; matches the regression's human_col
    flip_sign: bool  # matches plot_cache_vs_human_correlations.py / regression sign
    label: str


PREDICTORS: list[PredictorSpec] = [
    PredictorSpec("mask_area_pixels", "spearman", True, "mask area (foreground pixels)"),
    PredictorSpec("center_distance_pixels", "spearman", False, "centroid distance to center"),
    PredictorSpec("person_bin", "pearson", True, "Person (Y/N)"),
    PredictorSpec("max_gbvs", "spearman", True, "max GBVS in mask"),
]


def strip_run_timestamp(name: str) -> str:
    return _TS_PREFIX.sub("", str(name).strip())


def standardize(y: np.ndarray) -> np.ndarray:
    """Return (y - mean(y)) / std(y) (population std). Zeros if std==0."""
    y = np.asarray(y, dtype=np.float64)
    sd = float(np.std(y))
    if sd == 0:
        return np.zeros_like(y)
    return (y - float(np.mean(y))) / sd


def rank_avg(x: np.ndarray) -> np.ndarray:
    return pd.Series(x).rank(method="average").to_numpy(dtype=np.float64)


def load_acc_gap_per_model(kendall_csv: Path) -> dict[str, float]:
    df = pd.read_csv(kendall_csv).copy()
    df["model"] = df["model_b"].map(strip_run_timestamp)
    df["acc_gap"] = df["lowest_score_mask_same_accuracy"] - HUMAN_HUMAN_CONSISTENCY_X
    return dict(zip(df["model"], df["acc_gap"].astype(float)))


def build_aligned_data(
    human_csv: Path,
    cache_root: Path,
    mask_dir: Path,
    threshold: int,
    labels: pd.DataFrame | None,
    max_gbvs: pd.DataFrame | None,
) -> tuple[
    np.ndarray,        # human_sim, shape (n,)
    np.ndarray,        # M, shape (n, n_models)
    list[str],         # model_names (stripped, aligned to columns of M)
    list[str],         # cache_subfolders (raw, aligned to columns of M)
    dict[str, np.ndarray],  # predictor name -> shape (n,)
    int,               # n
]:
    df_h = mgsc.read_scores_csv(human_csv)
    feats_h, _, _ = mgsc.feature_rows_from_scores(df_h, mask_dir, threshold, None)
    feats_h = mgsc.merge_animacy_person_labels(feats_h, labels)
    feats_h = mgsc.merge_max_gbvs(feats_h, max_gbvs)
    valid = feats_h.dropna(
        subset=["mean_similarity_score", "mask_area_pixels", "center_distance_pixels"]
    )
    valid = valid[valid["mask_area_pixels"] > 0].copy()
    valid["mask_basename"] = valid["mask_basename"].astype(str)
    basenames = valid["mask_basename"].to_numpy()

    human_sim = pd.to_numeric(valid["mean_similarity_score"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    pred_data: dict[str, np.ndarray] = {}
    for spec in PREDICTORS:
        if spec.name not in valid.columns:
            raise SystemExit(f"Predictor {spec.name!r} missing from valid table.")
        pred_data[spec.name] = pd.to_numeric(valid[spec.name], errors="coerce").to_numpy(
            dtype=np.float64
        )

    eligible = mgsc.eligible_cache_csv_paths(cache_root)
    if not eligible:
        raise SystemExit(f"No eligible cache CSVs under {cache_root}")

    cache_subfolders: list[str] = []
    model_names: list[str] = []
    cols: list[np.ndarray] = []
    for sub, csv_path in eligible:
        df_m = mgsc.read_scores_csv(csv_path)
        mp = {
            Path(str(k)).name: float(v)
            for k, v in zip(
                df_m["img_fn"].astype(str),
                pd.to_numeric(df_m["mean_similarity_score"], errors="coerce"),
            )
        }
        arr = np.array([mp.get(b, np.nan) for b in basenames], dtype=np.float64)
        cols.append(arr)
        cache_subfolders.append(sub.name)
        model_names.append(strip_run_timestamp(sub.name))
    M = np.column_stack(cols)

    base_ok = np.isfinite(human_sim) & np.all(np.isfinite(M), axis=1)
    for spec in PREDICTORS:
        base_ok &= np.isfinite(pred_data[spec.name])
    n = int(base_ok.sum())
    if n < 3:
        raise SystemExit(f"Too few aligned masks: {n}")

    human_sim = human_sim[base_ok]
    M = M[base_ok]
    for k in list(pred_data.keys()):
        pred_data[k] = pred_data[k][base_ok]

    return human_sim, M, model_names, cache_subfolders, pred_data, n


def describe_null(observed: float, null: np.ndarray) -> dict[str, float]:
    finite = null[np.isfinite(null)]
    n_perm = int(finite.size)
    if not np.isfinite(observed) or n_perm == 0:
        return {
            "observed": float(observed) if np.isfinite(observed) else float("nan"),
            "null_mean": float("nan"),
            "null_std": float("nan"),
            "null_min": float("nan"),
            "null_p2_5": float("nan"),
            "null_p97_5": float("nan"),
            "null_max": float("nan"),
            "z_score": float("nan"),
            "p_two_sided": float("nan"),
            "p_one_sided_greater": float("nan"),
            "p_one_sided_less": float("nan"),
            "n_permutations_finite": n_perm,
        }
    mu = float(finite.mean())
    sd = float(finite.std(ddof=0))
    z = (observed - mu) / sd if sd > 0 else float("nan")
    return {
        "observed": float(observed),
        "null_mean": mu,
        "null_std": sd,
        "null_min": float(finite.min()),
        "null_p2_5": float(np.percentile(finite, 2.5)),
        "null_p97_5": float(np.percentile(finite, 97.5)),
        "null_max": float(finite.max()),
        "z_score": float(z),
        "p_two_sided": float((np.sum(np.abs(finite) >= abs(observed)) + 1) / (n_perm + 1)),
        "p_one_sided_greater": float((np.sum(finite >= observed) + 1) / (n_perm + 1)),
        "p_one_sided_less": float((np.sum(finite <= observed) + 1) / (n_perm + 1)),
        "n_permutations_finite": n_perm,
    }


def run_predictor(
    spec: PredictorSpec,
    pred: np.ndarray,
    human_z_pearson: np.ndarray,
    human_z_spearman: np.ndarray,
    Z_models_pearson: np.ndarray,
    Z_models_spearman: np.ndarray,
    acc_gap_z: np.ndarray,
    rank_acc_z: np.ndarray,
    n: int,
    n_models: int,
    n_permutations: int,
    rng: np.random.Generator,
) -> tuple[dict, dict, np.ndarray, np.ndarray, np.ndarray]:
    """
    For one predictor:
      - Build the permuted predictor vector once per permutation.
      - Compute permuted gap vector across models.
      - Compute univariate Pearson and Spearman of gap with acc_gap.
    Returns:
      (summary_pearson, summary_spearman, null_pearson, null_spearman, gap_observed)
    """
    if spec.metric == "pearson":
        pred_centered = standardize(pred)
        human_z = human_z_pearson
        Z_models = Z_models_pearson
    elif spec.metric == "spearman":
        pred_centered = standardize(rank_avg(pred))
        human_z = human_z_spearman
        Z_models = Z_models_spearman
    else:
        raise ValueError(f"unknown metric {spec.metric!r}")

    sign = -1.0 if spec.flip_sign else 1.0

    def gap_from_predvec(pv: np.ndarray) -> np.ndarray:
        corr_h = float(np.dot(human_z, pv) / n)
        corr_m = (Z_models.T @ pv) / n
        return sign * corr_h - sign * corr_m

    def uni_corrs(gap: np.ndarray) -> tuple[float, float]:
        gz = standardize(gap)
        rz = standardize(rank_avg(gap))
        if np.all(gz == 0):
            pr = float("nan")
        else:
            pr = float(np.dot(gz, acc_gap_z) / n_models)
        if np.all(rz == 0):
            sp = float("nan")
        else:
            sp = float(np.dot(rz, rank_acc_z) / n_models)
        return pr, sp

    gap_obs = gap_from_predvec(pred_centered)
    obs_pr, obs_sp = uni_corrs(gap_obs)

    null_pearson = np.empty(n_permutations, dtype=np.float64)
    null_spearman = np.empty(n_permutations, dtype=np.float64)

    iterator = range(n_permutations)
    if tqdm is not None:
        iterator = tqdm(iterator, desc=f"perm[{spec.name}]", unit="iter", leave=False)

    for i in iterator:
        perm = rng.permutation(n)
        pv = pred_centered[perm]
        gap = gap_from_predvec(pv)
        pr, sp = uni_corrs(gap)
        null_pearson[i] = pr
        null_spearman[i] = sp

    summary_pearson = describe_null(obs_pr, null_pearson)
    summary_spearman = describe_null(obs_sp, null_spearman)
    return summary_pearson, summary_spearman, null_pearson, null_spearman, gap_obs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--human-csv", type=Path, default=mgsc.DEFAULT_CSV)
    parser.add_argument("--cache-root", type=Path, default=mgsc.DEFAULT_CACHE_ROOT)
    parser.add_argument("--mask-dir", type=Path, default=mgsc.DEFAULT_MASK_DIR)
    parser.add_argument("--labels-csv", type=Path, default=mgsc.DEFAULT_LABELS_CSV)
    parser.add_argument("--max-gbvs-csv", type=Path, default=mgsc.DEFAULT_MAX_GBVS_CSV)
    parser.add_argument("--kendall-csv", type=Path, default=DEFAULT_KENDALL_CSV)
    parser.add_argument("--threshold", type=int, default=127)
    parser.add_argument("--n-permutations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument(
        "--save-null",
        action="store_true",
        help="Save the full per-permutation null distribution to CSV (long form).",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    labels = mgsc.load_animacy_person_labels(args.labels_csv)
    max_gbvs_df = mgsc.load_max_gbvs(args.max_gbvs_csv)
    human_sim, M, model_names, cache_subfolders, pred_data, n = build_aligned_data(
        args.human_csv,
        args.cache_root,
        args.mask_dir,
        args.threshold,
        labels,
        max_gbvs_df,
    )
    n_models = M.shape[1]

    acc_map = load_acc_gap_per_model(args.kendall_csv)
    missing_models = [m for m in model_names if m not in acc_map]
    if missing_models:
        print(
            f"[warn] {len(missing_models)} cache model(s) missing from Kendall CSV; dropping:"
        )
        for m in missing_models:
            print(f"  - {m}")
        keep_idx = [i for i, m in enumerate(model_names) if m in acc_map]
        if len(keep_idx) < 3:
            raise SystemExit(
                f"After dropping unmatched models only {len(keep_idx)} remain; need >= 3."
            )
        M = M[:, keep_idx]
        model_names = [model_names[i] for i in keep_idx]
        cache_subfolders = [cache_subfolders[i] for i in keep_idx]
        n_models = M.shape[1]
    acc_gap = np.array([acc_map[m] for m in model_names], dtype=np.float64)

    human_z_pearson = standardize(human_sim)
    human_z_spearman = standardize(rank_avg(human_sim))
    Z_models_pearson = np.column_stack([standardize(M[:, j]) for j in range(n_models)])
    Z_models_spearman = np.column_stack(
        [standardize(rank_avg(M[:, j])) for j in range(n_models)]
    )
    acc_gap_z = standardize(acc_gap)
    rank_acc_z = standardize(rank_avg(acc_gap))

    rng = np.random.default_rng(args.seed)

    summary_rows: list[dict] = []
    null_long_records: list[dict] = []
    observed_gaps: dict[str, np.ndarray] = {}

    for spec in PREDICTORS:
        sp_pr, sp_sp, null_pearson, null_spearman, gap_obs = run_predictor(
            spec=spec,
            pred=pred_data[spec.name],
            human_z_pearson=human_z_pearson,
            human_z_spearman=human_z_spearman,
            Z_models_pearson=Z_models_pearson,
            Z_models_spearman=Z_models_spearman,
            acc_gap_z=acc_gap_z,
            rank_acc_z=rank_acc_z,
            n=n,
            n_models=n_models,
            n_permutations=args.n_permutations,
            rng=rng,
        )
        observed_gaps[spec.name] = gap_obs
        summary_rows.append(
            {
                "predictor": spec.name,
                "predictor_label": spec.label,
                "metric_used_for_gap": spec.metric,
                "flip_sign": spec.flip_sign,
                "uni_metric": "pearson",
                "n_models": n_models,
                "n_masks": n,
                **sp_pr,
            }
        )
        summary_rows.append(
            {
                "predictor": spec.name,
                "predictor_label": spec.label,
                "metric_used_for_gap": spec.metric,
                "flip_sign": spec.flip_sign,
                "uni_metric": "spearman",
                "n_models": n_models,
                "n_masks": n,
                **sp_sp,
            }
        )

        for uni_metric, arr in (("pearson", null_pearson), ("spearman", null_spearman)):
            for i, v in enumerate(arr):
                null_long_records.append(
                    {
                        "predictor": spec.name,
                        "uni_metric": uni_metric,
                        "permutation_index": i,
                        "value": v,
                    }
                )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(args.out_dir / "permutation_summary.csv", index=False)

    obs_gap_df = pd.DataFrame({"model": model_names, "cache_subfolder": cache_subfolders,
                               "acc_gap": acc_gap})
    for k, v in observed_gaps.items():
        obs_gap_df[f"gap_{k}_observed"] = v
    obs_gap_df.to_csv(args.out_dir / "observed_gaps_per_model.csv", index=False)

    null_long_df = pd.DataFrame(null_long_records)
    if args.save_null:
        null_long_df.to_csv(args.out_dir / "permutation_null_distribution.csv", index=False)

    text_lines = [
        "Permutation test of univariate corr(gap_perm<P>, acc_gap)",
        f"acc_gap = lowest_score_mask_same_accuracy - {HUMAN_HUMAN_CONSISTENCY_X}",
        f"n_masks aligned: {n}",
        f"n_models: {n_models}",
        f"n_permutations: {args.n_permutations}",
        "",
    ]
    by_pred: dict[str, list[dict]] = {}
    for r in summary_rows:
        by_pred.setdefault(r["predictor"], []).append(r)
    for spec in PREDICTORS:
        if spec.name not in by_pred:
            continue
        text_lines.append(
            f"{spec.name}  ({spec.label})  metric={spec.metric}, flip_sign={spec.flip_sign}"
        )
        for r in by_pred[spec.name]:
            text_lines.append(
                f"  uni_{r['uni_metric']:<8s} observed={r['observed']: .4f}  "
                f"null_mean={r['null_mean']: .4f}  null_std={r['null_std']: .4f}  "
                f"z={r['z_score']: .3f}  p_two_sided={r['p_two_sided']:.4g}  "
                f"p_one_sided_greater={r['p_one_sided_greater']:.4g}  "
                f"p_one_sided_less={r['p_one_sided_less']:.4g}  "
                f"95%_null=[{r['null_p2_5']: .4f}, {r['null_p97_5']: .4f}]"
            )
        text_lines.append("")
    (args.out_dir / "permutation_summary.txt").write_text(
        "\n".join(text_lines) + "\n", encoding="utf-8"
    )

    print(f"\nWrote outputs to {args.out_dir}")
    print()
    print("=== Permutation summary (univariate corr(gap_perm, acc_gap)) ===")
    cols = [
        "predictor",
        "uni_metric",
        "observed",
        "null_mean",
        "null_std",
        "null_p2_5",
        "null_p97_5",
        "z_score",
        "p_two_sided",
    ]
    print(summary_df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
