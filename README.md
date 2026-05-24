# Revealing the Gap in Human and VLM Scene Perception through Counterfactual Semantic Saliency

This repository contains the code and processed data needed to reproduce
the figure and table in the submission.

Counterfactual Image dataset is not needed to reproduce the results, if you want to check the image data we used for the paper, you can download the dataset from the following link:

[Counterfactual Image dataset (Zenodo, anonymized preview link)](https://zenodo.org/records/20151292?preview=1&token=eyJhbGciOiJIUzUxMiIsImlhdCI6MTc3ODY0OTg3MCwiZXhwIjoxNzk2ODYwNzk5fQ.eyJpZCI6Ijg5NjQwMGEzLTY3YzMtNDI1ZC1iZDA2LWVlYmI3MzkyMTUwZCIsImRhdGEiOnt9LCJyYW5kb20iOiIzOGQwNjJiZThmZjczMGViYmY1ZDRkNDYxMjMzYjZlNiJ9.555ovV_fM1c0G57Ub3V_4AYu6XWRvQu2a8QDvOWHKtGWnLPDcKxjA0mb7ODZN1VbZJjYgfqaksmXcGKPymwZKg)

## Quick start

```bash
pip install numpy pandas scipy matplotlib pillow
# Reproduce all results and figures with a single command.
bash reproduce.sh
```

The full pipeline takes roughly **10 minutes** on a modern workstation (CPU model: AMD Ryzen 9 7900X3D ).
When it finishes, every generated results (final CSVs and
figures) is grouped under `./results/`.

## Mapping to the paper

Everything inside `results/` is organized so that each subfolder maps
to a single result in the paper:

| Path                                | Paper result | Contents                                                                              |
|-------------------------------------|----------------|---------------------------------------------------------------------------------------|
| `results/kendall_top1_acc/`         | **Figure 4**   | `kendall_top1_acc_bars.{png,pdf}` — Top1-Acc bar plot (one of the two Figure 4 panels). |
| `results/kendall_mean_tau/`         | **Figure 4**   | `kendall_mean_tau_bars.{png,pdf}` — mean Kendall-τ bar plot (the other Figure 4 panel). |
| `results/correlation_bars/`         | **Figure 5**   | The four `correlation_*.{png,pdf}` panels (mask area, centroid distance, person, max-GBVS), plus the CSVs that drive them. |
| `results/regression_permutation/`   | **Table 1**    | `permutation_summary.{csv,txt}` — rows in this CSV are the entries reported in Table 1 of the paper. |

## Structure

```
.
├── README.md                                          
├── reproduce.sh                                       # one-shot reproduction entry point
│
├── CSS_result/                                        # raw scoring runs (input data)
│   └── Cache/
│       ├── human_filtered_10/
│       │   ├── scene_description_human_filtered_all_10_with_mean_similarity_score.csv  # mean_similarity_score here is the mean of first 5 subjects (used for human-human consistency)
│       │   └── scene_description_human_filtered_all_10_mean_distance_mean_of_10.csv  # mean_similarity_score here is the mean of 10 subjects
│       └── <model_run>/scored_descriptions_*.csv      # 19 VLM runs, one CSV each
│
├── DATA/
│   └── inpaint_image_processed/
│       └── for_vis/                                   # binary-mask PNGs (used for calculating mask size & location)
│
├── Src/                                               # source code + script-internal checkpoints
│   ├── eval.sh                                        # stage 1 driver
│   ├── mask_rank_correlation.py                       # used by eval.sh
│   ├── analyze_first5_rest_kendall_tau.py             # stage 2
│   ├── pairwise_kendall_tau.py                        # stage 3
│   ├── mask_geometry_similarity_correlation.py        # stage 4
│   ├── bootstrap_correlation_human_vs_cache.py        # stage 5
│   ├── factor_analysis/
│   │   ├── regression_acc_gap_permutation_test.py     # stage 6  (Table 1)
│   │   ├── plot_cache_vs_human_correlations.py        # stage 7c (Figure 5)
│   │   ├── mask_animacy_person_labels.csv             # static input (hand labels)
│   │   └── mask_max_gbvs.csv                          # static input (precomputed GBVS)
│   ├── eval_outputs/                                  # stage 1 outputs   (intermediate, regenerated)
│   └── eval_output_tau/                               # stages 2-3 outputs (intermediate, regenerated)
│
├── fig_create/
│   ├── figure_style.py
│   ├── plot_kendall_top1_acc_bars.py                  # stage 7a (Figure 4 left)
│   └── plot_kendall_mean_tau_bars.py                  # stage 7b (Figure 4 right)
│
└── results/                                           # produced by reproduce.sh
    ├── kendall_top1_acc/                              # → Figure 4
    ├── kendall_mean_tau/                              # → Figure 4
    ├── correlation_bars/                              # → Figure 5
    └── regression_permutation/                        # → Table 1
```

## What `reproduce.sh` runs

| Stage | Command                                                                  | Outputs                                                                                                                                                  | Paper artefact |
|------:|--------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|----------------|
| 1     | `bash Src/eval.sh`                                                        | `Src/eval_outputs/<model>_rank_correlation.csv` for every model run under `CSS_result/Cache/`. (Intermediate.)                                          | —              |
| 2     | `Src/analyze_first5_rest_kendall_tau.py`                                  | `Src/eval_output_tau/first5_vs_rest/first5_vs_rest_kendall_per_image.csv` plus histogram. (Intermediate.)                                                | —              |
| 3     | `Src/pairwise_kendall_tau.py --mode human`                                | `Src/eval_output_tau/human_vs_model/kendall_vs_human_summary.csv` and the Kendall-τ histogram grid figure. (Intermediate; feeds stages 6 and 7a–7b.)     | —              |
| 4     | `Src/mask_geometry_similarity_correlation.py` (default + `--batch-cache`) | `results/correlation_bars/{correlation_summary,cache_correlation_by_csv,mask_geometry_and_similarity}.csv`.                                              | Figure 5 (data) |
| 5     | `Src/bootstrap_correlation_human_vs_cache.py`                             | `results/correlation_bars/bootstrap_human_vs_cache_correlation_smaller_pct.csv` (N=10000 by default).                                                    | Figure 5 (significance markers) |
| 6     | `Src/factor_analysis/regression_acc_gap_permutation_test.py --save-null`  | `results/regression_permutation/{permutation_summary.csv, permutation_summary.txt, observed_gaps_per_model.csv, permutation_null_distribution.csv}`.    | **Table 1**    |
| 7a    | `fig_create/plot_kendall_top1_acc_bars.py`                                | `results/kendall_top1_acc/kendall_top1_acc_bars.{png,pdf}`.                                                                                              | **Figure 4**   |
| 7b    | `fig_create/plot_kendall_mean_tau_bars.py`                                | `results/kendall_mean_tau/kendall_mean_tau_bars.{png,pdf}`.                                                                                              | **Figure 4**   |
| 7c    | `Src/factor_analysis/plot_cache_vs_human_correlations.py`                 | `results/correlation_bars/correlation_{mask_area_pixels, center_distance_pixels, person_bin, max_gbvs}.{png,pdf}`.                                       | **Figure 5**   |

