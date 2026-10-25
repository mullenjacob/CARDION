> Note: this spec dates from the original plan, in which UBFC-rPPG was the primary dataset. The final study uses **MCD-rPPG** (36/12/12 subject-disjoint split; see `data/README.md` and the paper); UBFC-rPPG and PURE loaders remain but were not used for any reported result.

# Spec 07 — evaluation (`src/evaluate.py`, `src/stats.py`)

- `python -m src.evaluate --dataset synthetic|ubfc --split test [--ckpt-snn GLOB --ckpt-ann GLOB]` runs RGB green/POS/CHROM, event-POS (33.3 ms, 10 ms) and the models on identical clips/windows; writes `out/results_<dataset>-<split>_*.{json,csv}` and registers claims.
- `stats.py`: `hr_metrics` (MAE, RMSE, MAPE, Pearson, Lin's CCC, Bland-Altman bias/LoA), `snr_db`, `macc`, `bootstrap_ci` (over subjects), `paired_wilcoxon`.
- Figures (Bland-Altman, Pareto, ablations, failure cases) are generated in Phase 7 from `out/` artifacts only.
