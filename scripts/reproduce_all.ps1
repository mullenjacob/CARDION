# Full regeneration of the CARDION research results and paper (CPU; roughly 8-12 hours on a laptop, dominated by network training).
# Run from the repo root in PowerShell after creating the research environment (see REPRODUCE.md).
$ErrorActionPreference = "Stop"
$py = ".venv\Scripts\python.exe"
function Step($m) { Write-Host "`n=== $m ===" -ForegroundColor Cyan }

Step "unit + integration tests"
& $py -m pytest -q tests
Step "third-party reference code (pinned) and data"
& $py scripts\fetch_third_party.py
& $py scripts\download_mcd.py --n-subjects 60 --seed 1337          # real humans, CC BY 4.0 (~7 GB)
# SCAMPS 10-video example (diagnostic only): see data\README.md
& $py -m src.datasets manifest --dataset mcd
& $py -m src.datasets preprocess --dataset mcd --workers 6
& $py scripts\dataset_stats.py
Step "correctness anchor, development analyses (train/val subjects only)"
& $py scripts\anchor_toolbox.py --clips scamps:P000001 scamps:P000002 scamps:P000003 scamps:P000004 scamps:P000005
& $py scripts\dev_baselines.py --split train
& $py scripts\dev_baselines.py --split val
& $py -m src.dvs_emulate smoke
& $py -m src.efficiency --steps 100
Step "event tensors for every clip and threshold"
& $py -m src.train precompute --sets mcd:train mcd:val mcd:test --thresholds 0.02 0.05 0.1 --workers 6
Step "train SNN and matched ANN (3 seeds x 3 thresholds) + un-quantised control"
& .\scripts\run_mcd_training.ps1 -Thresholds 0.02 -Seeds 0,1,2 -Workers 3
Wait-Process -Name python -ErrorAction SilentlyContinue   # (the workers run in the background; wait for them to finish)
& .\scripts\run_mcd_training.ps1 -Thresholds 0.05 -Seeds 0,1,2 -Workers 3
& .\scripts\run_mcd_training.ps1 -Thresholds 0.1 -Seeds 0,1,2 -Workers 3
foreach ($m in "snn", "ann") { & $py -m src.train --model $m --input logdiff --threshold 0.02 --train-seed 0 --tag mcdlog_th0.02_s0 --train-sets mcd:train --val-sets mcd:val --diag-sets scamps:all --epochs 24 --windows-per-epoch 128 --threads 2 }
Step "validation evaluation, quality-rule calibration, app model export (validation only)"
& .\scripts\eval_mcd.ps1 -Split val -Tag val_all
& $py scripts\calibrate_quality.py --mode calibrate --results "out/results_mcd-val_val_all_*.json"
& $py scripts\export_app_models.py
& $py scripts\app_real_video_check.py --split val --max-clips 8
Step "FINAL test evaluation (protected split; already scored in the original study, see docs/TEST_SPLIT_HISTORY.md)"
& .\scripts\eval_mcd.ps1 -Split test -Tag final -Final
& $py scripts\app_real_video_check.py --split test --final --max-clips 8
& $py scripts\calibrate_quality.py --mode apply --results "out/results_mcd-test_final_*.json"
Step "diagnostics and controlled studies"
& $py scripts\summarize_diag.py
& $py scripts\event_vs_intensity.py --split val
& $py scripts\noise_study.py --clips 6 --amp-scale 1.0
& $py scripts\noise_study.py --clips 6 --amp-scale 0.2
& $py scripts\compression_study.py --clips 6
Step "report, references, paper"
& $py -m src.evaluate --report "out/results_mcd-test_final_*.json"
& $py scripts\verify_refs.py
& $py scripts\build_paper.py
Step "app self-test"
& $py -m src.cardion_app --selftest

