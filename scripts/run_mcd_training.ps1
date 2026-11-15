# Train SNN and matched ANN on MCD-rPPG (real humans) for the given thresholds and seeds, in parallel worker processes.
# Model selection uses ONLY mcd:val; scamps:all is monitored as a diagnostic (never used for selection).
#   powershell -File scripts\run_mcd_training.ps1 -Thresholds 0.02 -Seeds 0,1,2 -Workers 4
param([double[]]$Thresholds = @(0.02), [int[]]$Seeds = @(0, 1, 2), [int]$Workers = 4, [int]$Epochs = 24, [int]$Windows = 128, [int]$Threads = 2)
$jobs = @()
foreach ($th in $Thresholds) { foreach ($s in $Seeds) { foreach ($m in "snn", "ann") { $jobs += ,@($m, $th, $s) } } }
for ($w = 0; $w -lt $Workers; $w++) {
  $lines = @('$py = ".venv\Scripts\python.exe"')
  for ($i = $w; $i -lt $jobs.Count; $i += $Workers) {
    $m, $th, $s = $jobs[$i]
    $lines += "& `$py -m src.train --model $m --train-seed $s --train-sets mcd:train --val-sets mcd:val --diag-sets scamps:all --epochs $Epochs --windows-per-epoch $Windows --threshold $th --threads $Threads --tag mcd_th${th}_s$s *> out\train_${m}_mcd_th${th}_s$s.log"
  }
  $f = "scripts\_mcd_worker_$($Thresholds -join '_')_$w.ps1"
  $lines | Set-Content $f -Encoding ascii
  Start-Process powershell -ArgumentList "-NoProfile", "-File", $f -WindowStyle Hidden
}
"started $($jobs.Count) training runs on $Workers workers"

