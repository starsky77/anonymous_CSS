#!/usr/bin/env bash
# ----------------------------------------------------------------------------
# reproduce.sh - Single-entry reproduction script.
#
# Runs every analysis stage in dependency order. All final results are
# written under ``results/`` at the repo root, organized into:
#   results/correlation_bars/        factor-analysis CSVs and 4 bar panels
#   results/regression_permutation/  regression permutation-test CSVs
#   results/kendall_top1_acc/        kendall_top1_acc_bars.{png,pdf}
#   results/kendall_mean_tau/        kendall_mean_tau_bars.{png,pdf}
#
# Stage-internal intermediates (rank correlations, per-pair Kendall tau)
# stay under Src/eval_outputs/ and Src/eval_output_tau/ since they are
# checkpoints consumed by downstream stages.
#
# Usage (from any working directory):
#     bash reproduce.sh                  # full pipeline (~10 min)
#     SKIP_BOOTSTRAP=1 bash reproduce.sh # skip the slow bootstrap stage
#     N_BOOT=1000 bash reproduce.sh      # smaller bootstrap (default 10000)
#     PY=python3.11 bash reproduce.sh    # pick a specific Python interpreter
#
# Environment overrides:
#     PY              python interpreter to use (default: python3)
#     N_BOOT          number of bootstrap resamples (default: 10000)
#     SKIP_BOOTSTRAP  if set to 1, skip stage 5 (bootstrap)
#     SKIP_PERMUTE    if set to 1, skip stage 6 (permutation test)
#
# Required Python packages: numpy, pandas, scipy, matplotlib, Pillow.
# ----------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

PY="${PY:-python3}"
N_BOOT="${N_BOOT:-10000}"
RESULTS_DIR="${REPO_ROOT}/results"

stage() {
  printf '\n=========================================================\n'
  printf '  %s\n' "$1"
  printf '=========================================================\n'
}

mkdir -p "${RESULTS_DIR}"

stage "Stage 1/7  Per-model rank correlation  (eval.sh)"
bash Src/eval.sh

stage "Stage 2/7  Human-vs-human reference Kendall tau"
"${PY}" Src/analyze_first5_rest_kendall_tau.py

stage "Stage 3/7  Human-vs-model Kendall tau summary"
"${PY}" Src/pairwise_kendall_tau.py --mode human

stage "Stage 4/7  Mask-geometry correlations (single human + per-model batch)"
"${PY}" Src/mask_geometry_similarity_correlation.py
"${PY}" Src/mask_geometry_similarity_correlation.py --batch-cache

if [[ "${SKIP_BOOTSTRAP:-0}" == "1" ]]; then
  stage "Stage 5/7  Bootstrap (SKIPPED via SKIP_BOOTSTRAP=1)"
else
  stage "Stage 5/7  Bootstrap human-vs-cache correlation  (N=${N_BOOT}, ~8 min at 10k)"
  "${PY}" Src/bootstrap_correlation_human_vs_cache.py --n-bootstrap "${N_BOOT}"
fi

if [[ "${SKIP_PERMUTE:-0}" == "1" ]]; then
  stage "Stage 6/7  Permutation test (SKIPPED via SKIP_PERMUTE=1)"
else
  stage "Stage 6/7  Permutation test for acc-gap regression"
  "${PY}" Src/factor_analysis/regression_acc_gap_permutation_test.py --save-null
fi

stage "Stage 7/7  Bar plots (Top1-Acc, mean Kendall tau, factor-analysis)"
"${PY}" fig_create/plot_kendall_top1_acc_bars.py
"${PY}" fig_create/plot_kendall_mean_tau_bars.py
"${PY}" Src/factor_analysis/plot_cache_vs_human_correlations.py

printf '\n=========================================================\n'
printf '  All done. Results tree under: %s\n' "${RESULTS_DIR}"
printf '=========================================================\n'
if command -v tree >/dev/null 2>&1; then
  tree -L 2 "${RESULTS_DIR}"
else
  ( cd "${RESULTS_DIR}" && find . -maxdepth 2 -mindepth 1 | sort )
fi
