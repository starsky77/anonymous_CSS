#!/usr/bin/env bash
# Run mask_rank_correlation.py for every LLM model run under CACHE_ROOT:
# each immediate subdirectory that (1) is not a human baseline run and (2)
# contains scored_descriptions*.csv with mean_similarity_score (see Python script).
#
# Override paths with environment variables, e.g.:
#   BASELINE=/path/to/human/scored_descriptions.csv CACHE_ROOT=/path/to/Cache ./eval.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

BASELINE="${BASELINE:-${REPO_ROOT}/CSS_result/Cache/human_filtered_10/scene_description_human_filtered_all_10_with_mean_similarity_score.csv}"
CACHE_ROOT="${CACHE_ROOT:-${REPO_ROOT}/CSS_result/Cache}"
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/eval_outputs}"

mkdir -p "${OUT_DIR}"

if [[ ! -f "${BASELINE}" ]]; then
  echo "Baseline CSV not found: ${BASELINE}" >&2
  exit 1
fi
if [[ ! -d "${CACHE_ROOT}" ]]; then
  echo "Cache directory not found: ${CACHE_ROOT}" >&2
  exit 1
fi

shopt -s nullglob

count=0
for run_dir in "${CACHE_ROOT}"/*/; do
  [[ -d "${run_dir}" ]] || continue
  name="$(basename "${run_dir}")"
  # Skip human reference runs (directory name contains "human", any case).
  if [[ "${name}" == *[Hh][Uu][Mm][Aa][Nn]* ]]; then
    continue
  fi
  matches=( "${run_dir}"scored_descriptions*.csv )
  if [[ ${#matches[@]} -eq 0 ]]; then
    continue
  fi
  model_csv="${matches[0]}"
  if [[ ${#matches[@]} -gt 1 ]]; then
    echo "Warning: multiple scored_descriptions in ${run_dir}; using ${model_csv}" >&2
  fi
  # Sanitize run folder name for output filename.
  out_safe="${name//[^A-Za-z0-9._-]/_}"
  out_csv="${OUT_DIR}/${out_safe}_rank_correlation.csv"

  echo "Eval: ${name}"
  python3 "${SCRIPT_DIR}/mask_rank_correlation.py" \
    "${BASELINE}" \
    "${model_csv}" \
    -o "${out_csv}"
  count=$((count + 1))
done

if [[ "${count}" -eq 0 ]]; then
  echo "No LLM scored_descriptions runs found under ${CACHE_ROOT} (skipped *human* dirs)." >&2
  exit 1
fi

echo "Done. ${count} model(s); outputs in ${OUT_DIR}"
