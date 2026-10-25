# Build the paper: fill claims (Python), then compile with the first LaTeX toolchain found (portable MiKTeX in tools/, then pdflatex on PATH).
# (On some Windows application-control setups LaTeX binaries only start when launched from PowerShell, hence this wrapper.)
$ErrorActionPreference = "Continue"
$py = ".venv\Scripts\python.exe"
& $py scripts\build_paper.py --no-compile
if ($LASTEXITCODE -ne 0) { Write-Host "claim check failed; fix the missing claims above" -ForegroundColor Red; exit 1 }
$mik = Join-Path (Get-Location) "tools\miktex\texmfs\install\miktex\bin\x64"
$bin = if (Test-Path "$mik\pdflatex.exe") { $mik } else { $null }
$pdflatex = if ($bin) { "$bin\pdflatex.exe" } else { "pdflatex" }
$bibtex = if ($bin) { "$bin\bibtex.exe" } else { "bibtex" }
Push-Location paper
try {
  & $pdflatex -interaction=nonstopmode main.tex *> $null
  & $bibtex main *> $null
  & $pdflatex -interaction=nonstopmode main.tex *> $null
  & $pdflatex -interaction=nonstopmode main.tex *> $null
} finally { Pop-Location }
$log = Get-Content paper\main.log -ErrorAction SilentlyContinue
$problems = $log | Select-String -Pattern "undefined|Fatal error|Emergency stop|LaTeX Error"
$over = ($log | Select-String -Pattern "Overfull \\hbox").Count
if (Test-Path paper\main.pdf) { "main.pdf: {0:N0} KiB, written {1}" -f ((Get-Item paper\main.pdf).Length / 1KB), (Get-Item paper\main.pdf).LastWriteTime }
if ($problems) { $problems | Select-Object -First 10 | ForEach-Object { $_.Line }; "PROBLEMS FOUND"; exit 1 } else { "no undefined references or errors; overfull boxes: $over" }
