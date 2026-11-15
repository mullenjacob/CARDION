<#
  Copies this CARDION folder to a short path (default %USERPROFILE%\CARDION), without the Python environment (it is rebuilt there),
  leaves the original untouched, then offers to start the copy. ASCII only.
#>
param([string]$Dest = (Join-Path $env:USERPROFILE "CARDION"))
$src = Split-Path -Parent $PSScriptRoot
Write-Host "[CARDION] Copying $src"
Write-Host "[CARDION]      to $Dest  (the original is NOT deleted)"
if ((Resolve-Path -LiteralPath $src).Path.TrimEnd('\') -ieq $Dest.TrimEnd('\')) { Write-Host "[CARDION] Already in the short folder."; exit 0 }
$p = Start-Process robocopy -ArgumentList ('"' + $src + '" "' + $Dest + '" /E /XD .venv __pycache__ .pytest_cache logs /XF *.pyc /NFL /NDL /NP /R:1 /W:1') -NoNewWindow -Wait -PassThru
if ($p.ExitCode -ge 8) { Write-Host "[CARDION] Copy failed (robocopy code $($p.ExitCode))." -ForegroundColor Red; exit 1 }
Write-Host "[CARDION] Done. Start CARDION from: $Dest\Launch CARDION.bat"
Start-Process -FilePath (Join-Path $Dest "Launch CARDION.bat") -WorkingDirectory $Dest
exit 0
