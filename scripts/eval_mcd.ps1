# Evaluate every method and every trained model family on an MCD-rPPG split (all checkpoints = all seeds).
#   powershell -File scripts\eval_mcd.ps1 -Split val -Tag val_all            # validation (model selection, quality calibration)
#   powershell -File scripts\eval_mcd.ps1 -Split test -Tag final -Final       # protected test split: run ONCE after everything is frozen
param([ValidateSet("val", "test")][string]$Split = "val", [string]$Tag = "val_all", [switch]$Final)
$py = ".venv\Scripts\python.exe"
$models = @()
foreach ($th in "0.02", "0.05", "0.1") { foreach ($m in "snn", "ann") { $models += "${m}_th$th=out\ckpt_${m}_mcd_th${th}_s*.pt" } }
foreach ($m in "snn", "ann") { $models += "${m}_log_th0.02=out\ckpt_${m}_mcdlog_th0.02_s*.pt" }   # un-quantised log-difference control
$a = @("-m", "src.evaluate", "--dataset", "mcd", "--split", $Split, "--tag", $Tag, "--thresholds", "0.02", "0.05", "0.1", "--set", "eval.bootstrap_n=2000")
if ($Final) { $a += "--final" }
& $py @a --models @models
