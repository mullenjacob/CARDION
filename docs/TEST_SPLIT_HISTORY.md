# History of the protected test split

Model checkpoints, event thresholds, heart-rate picking and the reliability gate were frozen using training and validation subjects before the first test evaluation.

The test split was evaluated twice. The first summary incorrectly used the window-pooled MAE of the first training seed in some network rows instead of the seed-averaged subject-level MAE. The second evaluation corrected summary and reporting code; no model, threshold or gate rule was changed.

The window-level estimates were identical across both runs for 696 method-clip pairs. The largest change in a per-family summary MAE was 5.20 BPM. The reported results use the corrected evaluation.

Later gate audits, paired effect sizes, per-subject views and the real-time pipeline check reused saved test outputs and are descriptive or exploratory. One test clip was also opened during frame-rate debugging. The test subjects were therefore no longer unseen after the first evaluation.

The original and corrected saved outputs and additional provenance are preserved in the published v1.0.0 archive: https://doi.org/10.5281/zenodo.23647425 . The manuscript distinguishes confirmatory comparisons from later exploratory analyses. Internal development commit references are omitted from this GitHub summary.
