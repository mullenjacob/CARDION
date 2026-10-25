<#
  CARDION launcher / setup / repair (Windows PowerShell 5.1+). Used by the .bat files in the CARDION folder.

  Normal launch : quick health check (a few seconds) -> start the app. First run (or damaged environment): builds a private
                  Python 3.11 environment in .venv (uv, checksum-verified; falls back to a system Python) and installs the pinned
                  packages (about 1.5 GB, internet needed ONCE), then runs the doctor (scripts\doctor.py) which diagnoses and repairs.
  -SelfTest       : environment check/repair, then the automated self-test with a PASS/FAIL report.
  -TestAndRepair  : full doctor + self-test + camera tests, writes out\test_and_repair_report.md.
  -CheckOnly      : only verify/repair the environment.   -Reinstall : rebuild .venv from scratch.   -NoBrowser / -NoPause : for automated tests.

  ERROR HANDLING: native programs are never run through PowerShell's stderr redirection (in Windows PowerShell 5.1 that turns the first
  line of stderr - e.g. "Traceback (most recent call last):" - into a terminating error and hides the rest). Programs write to the
  console directly or into files; everything is logged to logs\launcher_<time>.log, and a failure writes the complete error to
  logs\launcher_error_<time>.log and opens it.
  Keep this file ASCII-only (Windows PowerShell reads BOM-less files as ANSI).
#>
param([switch]$SelfTest, [switch]$TestAndRepair, [switch]$CheckOnly, [switch]$Reinstall, [switch]$NoBrowser, [switch]$NoPause)
$ErrorActionPreference = "Continue"
$ProgressPreference = "SilentlyContinue"
$root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $root
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$logs = Join-Path $root "logs"
try { New-Item -ItemType Directory -Force -Path $logs -ErrorAction Stop | Out-Null } catch { $logs = Join-Path $env:TEMP "cardion_logs"; New-Item -ItemType Directory -Force -Path $logs | Out-Null }
$log = Join-Path $logs "launcher_$stamp.log"
# Windows cannot load DLLs from paths longer than about 260 characters (WinError 206), even with "long paths" enabled. torch/mediapipe/Pillow
# have very deep files, so when the project folder is long (or a previous run hit this limit) the private Python environment is kept
# in a short folder under the user profile instead of inside the project. Models, data and results always stay in the project folder.
$envMarker = Join-Path $root "logs\env_location.txt"
$shortEnv = Join-Path $env:USERPROFILE (".cardion\env_" + ([BitConverter]::ToString((New-Object Security.Cryptography.SHA1Managed).ComputeHash([Text.Encoding]::UTF8.GetBytes($root.ToLower()))).Replace("-", "").Substring(0, 8)))
$maxRoot = 80
if ($env:CARDION_MAX_ROOT_LEN) { $maxRoot = [int]$env:CARDION_MAX_ROOT_LEN }      # test switch: pretend long folders are fine, to exercise the automatic relocation
if (($root.Length -gt $maxRoot) -or (Test-Path -LiteralPath $envMarker)) { $venv = $shortEnv } else { $venv = Join-Path $root ".venv" }
$py = Join-Path $venv "Scripts\python.exe"
$doctor = Join-Path $root "scripts\doctor.py"
$uvVersion = "0.13.5"
$uvUrl = "https://github.com/astral-sh/uv/releases/download/$uvVersion/uv-x86_64-pc-windows-msvc.zip"
$uvSha = "a38b8d1902b7cfbc76b423ba1bb456f623f42064cdc72628b6075b89afd1019a"
$noPause = $NoPause -or ($env:CARDION_NO_PAUSE -eq "1")

function Log([string]$m) { try { Add-Content -LiteralPath $log -Value ("{0} {1}" -f (Get-Date -Format "HH:mm:ss"), $m) -Encoding UTF8 } catch { } }
function Say([string]$m) { Write-Host "[CARDION] $m"; Log $m }
function Q([string]$s) { return '"' + $s + '"' }

# Run a program with its output going straight to the console (no PowerShell redirection involved). Returns the exit code.
function Invoke-Tool([string]$exe, [string]$argline) {
    Log "RUN: $exe $argline"
    $p = Start-Process -FilePath $exe -ArgumentList $argline -NoNewWindow -Wait -PassThru
    $p.WaitForExit()
    Log "exit code $($p.ExitCode)"
    return [int]$p.ExitCode
}

# Run a short program, capture stdout+stderr into files, log everything. Returns @{Code; Text}.
function Invoke-Captured([string]$exe, [string]$argline) {
    $o = Join-Path $env:TEMP ("cardion_" + [guid]::NewGuid().ToString("N") + ".txt")
    $e = $o + ".err"
    $code = -1
    $text = ""
    try {
        $p = Start-Process -FilePath $exe -ArgumentList $argline -NoNewWindow -Wait -PassThru -RedirectStandardOutput $o -RedirectStandardError $e -ErrorAction Stop
        $p.WaitForExit()
        $code = [int]$p.ExitCode
    } catch {
        $text = "Could not start $exe : $($_.Exception.Message)`n"
    }
    foreach ($f in @($o, $e)) {
        if (Test-Path -LiteralPath $f) { $text += [string](Get-Content -LiteralPath $f -Raw -ErrorAction SilentlyContinue); Remove-Item -LiteralPath $f -Force -ErrorAction SilentlyContinue }
    }
    Log ("RUN(captured): $exe $argline`nexit code $code`n$text")
    return [pscustomobject]@{ Code = $code; Text = $text }
}

function Show-File([string]$path) {
    if (-not $noPause -and (Test-Path -LiteralPath $path)) { try { Start-Process -FilePath "notepad.exe" -ArgumentList (Q $path) } catch { } }
}

function Get-Uv {
    if ($env:CARDION_NO_UV -eq "1") { return $null }      # test/troubleshooting switch: behave as if uv cannot be used
    $uv = Join-Path $root "tools\uv\uv.exe"
    if (Test-Path -LiteralPath $uv) { return $uv }
    try {
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        New-Item -ItemType Directory -Force -Path (Join-Path $root "tools\uv"), (Join-Path $root "tools\_dl") | Out-Null
        $zip = Join-Path $root "tools\_dl\uv.zip"
        Say "Downloading the uv installer tool $uvVersion from github.com/astral-sh/uv ..."
        Invoke-WebRequest -Uri $uvUrl -OutFile $zip -UseBasicParsing -ErrorAction Stop
        $h = (Get-FileHash -LiteralPath $zip -Algorithm SHA256).Hash.ToLower()
        if ($h -ne $uvSha) { Remove-Item -LiteralPath $zip -Force; throw "the uv download failed its checksum ($h)" }
        Expand-Archive -LiteralPath $zip -DestinationPath (Join-Path $root "tools\uv") -Force
        if (Test-Path -LiteralPath $uv) { return $uv }
    } catch { Say "Could not get uv ($($_.Exception.Message)); will try a Python already installed on this PC."; Log ($_ | Out-String) }
    return $null
}

function Find-SystemPython {
    foreach ($c in @(@("py", "-3.11"), @("py", "-3.12"), @("py", "-3.10"), @("python", ""))) {
        $cmd = Get-Command $c[0] -ErrorAction SilentlyContinue
        if (-not $cmd) { continue }
        $r = Invoke-Captured $cmd.Source (($c[1] + ' -c "import sys; assert (3,10) <= sys.version_info[:2] <= (3,12); print(sys.version_info[:2])"').Trim())
        if ($r.Code -eq 0) { return @{ exe = $cmd.Source; pre = $c[1] } }
    }
    return $null
}

function New-Venv {
    if (Test-Path -LiteralPath $venv) {
        Say "Removing the damaged private Python environment (.venv) so it can be rebuilt. Your data, models and results are not touched."
        Remove-Item -LiteralPath $venv -Recurse -Force -ErrorAction SilentlyContinue
        if (Test-Path -LiteralPath $venv) {
            $old = $venv + "_old_" + $stamp
            Rename-Item -LiteralPath $venv -NewName (Split-Path -Leaf $old) -ErrorAction SilentlyContinue
            Say "Some files in .venv were locked; the old copy was renamed to $old (safe to delete later)."
        }
    }
    $uv = Get-Uv
    if ($uv) {
        Say "Creating the private Python 3.11 environment (downloads Python the first time) ..."
        $r = Invoke-Captured $uv ("venv --python 3.11 --seed " + (Q $venv))
        if ($r.Code -eq 0 -and (Test-Path -LiteralPath $py)) { return }
        Say "uv could not create the environment: $(($r.Text -split "`n" | Select-Object -Last 2) -join ' ') - trying Python installed on this PC."
    }
    $sys = Find-SystemPython
    if ($sys) {
        Say "Creating the environment with the Python found on this PC ($($sys.exe) $($sys.pre)) ..."
        $r = Invoke-Captured $sys.exe (($sys.pre + " -m venv " + (Q $venv)).Trim())
        if ($r.Code -eq 0 -and (Test-Path -LiteralPath $py)) { return }
    }
    throw "Could not create a Python environment: the uv tool could not download/run (no internet, or security software blocked it) and no suitable Python 3.10-3.12 is installed. Details are in $log"
}

function Use-ShortEnv {
    $script:venv = $shortEnv
    $script:py = Join-Path $script:venv "Scripts\python.exe"
    try { Set-Content -LiteralPath $envMarker -Value $shortEnv -Encoding ascii } catch { Log "could not write $envMarker" }
}

function Get-DoctorFlags([bool]$full) {
    $f = "--repair --deep"
    $uv = Join-Path $root "tools\uv\uv.exe"
    if (-not (Test-Path -LiteralPath $uv)) { $uv = Get-Uv }
    if ($uv) { $f += " --uv " + (Q $uv) }
    if ($full) { $f += " --full" }
    return $f
}

function Invoke-Doctor([bool]$full) {
    $code = Invoke-Tool $py ((Q $doctor) + " --root " + (Q $root) + " " + (Get-DoctorFlags $full))
    if ($code -eq 11) {
        if ($venv -eq $shortEnv) { throw "The environment is already in a short folder ($venv) but Windows still reports path-length errors. See out\doctor_report.md and the newest logs\doctor_*.log." }
        Say "Windows cannot load the Python packages from this deep folder (path-length limit, WinError 206). Reinstalling in the same place cannot help, so the environment is moved to a short folder: $shortEnv"
        Use-ShortEnv
        New-Venv
        $code = Invoke-Tool $py ((Q $doctor) + " --root " + (Q $root) + " " + (Get-DoctorFlags $full) + " --no-recreate")
    }
    if ($code -eq 10) {
        Say "The Python environment cannot be repaired in place. Rebuilding it from scratch (one time) ..."
        New-Venv
        $code = Invoke-Tool $py ((Q $doctor) + " --root " + (Q $root) + " " + (Get-DoctorFlags $full) + " --no-recreate")
    }
    return $code
}

$exitCode = 0
try {
    Say "CARDION launcher. Folder: $root"
    Say "Python environment: $venv"
    Log ("PowerShell {0}; {1}; path length {2}" -f $PSVersionTable.PSVersion, [Environment]::OSVersion.VersionString, $root.Length)
    $env:PYTHONPATH = $root
    $env:PYTHONUTF8 = "1"
    $env:PYTHONNOUSERSITE = "1"

    if ($Reinstall) { Say "Reinstall requested."; New-Venv }
    $needRepair = [bool]$Reinstall
    $pyOk = $false
    if (Test-Path -LiteralPath $py) {
        $r = Invoke-Captured $py '-c "import sys; print(sys.version_info[:2])"'
        $pyOk = ($r.Code -eq 0)
    }
    if (-not $pyOk) {
        if (Test-Path -LiteralPath $venv) { Say "The Python environment is damaged; rebuilding it automatically." } else { Say "First-time setup: about 1.5 GB will be downloaded (5-15 minutes, internet required once)." }
        New-Venv
        $needRepair = $true
    } elseif (-not $needRepair) {
        $q = Invoke-Captured $py ((Q $doctor) + " --quick --root " + (Q $root))
        if ($q.Code -eq 11 -and $venv -ne $shortEnv) {
            Say "The Python packages cannot be loaded from this deep folder (Windows path-length limit). Moving the environment to a short folder: $shortEnv"
            Use-ShortEnv
            New-Venv
            $needRepair = $true
        } elseif ($q.Code -ne 0) {
            Say "Quick check found a problem: $((($q.Text -split "`n" | Where-Object { $_.Trim() } | Select-Object -Last 1)).Trim()) - repairing automatically ..."
            $needRepair = $true
        }
    }

    if ($needRepair -or $TestAndRepair -or $SelfTest) {
        if ($TestAndRepair) { Say "Running the full test and repair (about 5 minutes) ..." }
        $code = Invoke-Doctor ([bool]$TestAndRepair)
        if ($code -ne 0 -and $TestAndRepair -and (Test-Path -LiteralPath (Join-Path $root "out\test_and_repair_report.md"))) {
            $rep = Join-Path $root "out\test_and_repair_report.md"
            Say "Some checks failed. The report lists what and why: $rep" 
            Show-File $rep
            exit 1
        }
        if ($code -ne 0) {
            $rep = Join-Path $root "out\doctor_report.md"
            $why = ""
            try {
                $j = Get-Content -LiteralPath (Join-Path $root "out\doctor_report.json") -Raw | ConvertFrom-Json
                foreach ($r in $j.results) { if ($r.status -eq "FAIL") { $why += "`r`n  - " + $r.check + ": " + (($r.detail -split "`n")[0]) } }
            } catch { }
            throw ("CARDION could not fix everything automatically, or a test failed." + $why + "`r`nPlain-language report: $rep`r`nTechnical log: the newest file in $logs named doctor_*.log (or selftest_/cameratest_ for those tests).")
        }
    }
    Set-Content -LiteralPath (Join-Path $venv "cardion_ready.txt") -Value $stamp -Encoding ascii -ErrorAction SilentlyContinue

    if ($CheckOnly) { Say "Environment OK."; exit 0 }
    if ($TestAndRepair) {
        $rep = Join-Path $root "out\test_and_repair_report.md"
        Say "Done. Report: $rep"
        Show-File $rep
        exit 0
    }
    if ($SelfTest) {
        Say "Running the self-test (about 2-4 minutes) ..."
        $exitCode = Invoke-Tool $py "-m src.cardion_app --selftest"
        $rep = Join-Path $root "out\selftest_report.md"
        if (Test-Path -LiteralPath $rep) { Write-Host ""; Get-Content -LiteralPath $rep | ForEach-Object { Write-Host $_ } }
        Say ("Self-test " + $(if ($exitCode -eq 0) { "PASSED" } else { "FAILED - see the report above and the newest logs\selftest_*.log" }))
        exit $exitCode
    }

    Say "Starting CARDION - your browser will open. Close this window to quit."
    $appArgs = "-m src.cardion_app"
    if ($NoBrowser) { $appArgs += " --no-browser" }
    $exitCode = Invoke-Tool $py $appArgs
    if ($exitCode -ne 0) {
        $latest = Get-ChildItem -LiteralPath $logs -Filter "app_*.log" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
        $where = if ($latest) { $latest.FullName } else { $log }
        Write-Host ""
        Write-Host "[CARDION] CARDION stopped with an error (exit code $exitCode). The complete error report is saved here:" -ForegroundColor Red
        Write-Host "          $where" -ForegroundColor Red
        Show-File $where
    }
    exit $exitCode
} catch {
    $errLog = Join-Path $logs "launcher_error_$stamp.log"
    $detail = "CARDION launcher error $(Get-Date -Format s)`r`nMessage: $($_.Exception.Message)`r`n`r`nFull error record:`r`n$($_ | Out-String)`r`nScript stack trace:`r`n$($_.ScriptStackTrace)`r`n`r`nLauncher log: $log"
    try { Set-Content -LiteralPath $errLog -Value $detail -Encoding UTF8 } catch { }
    Log $detail
    Write-Host ""
    Write-Host "[CARDION] Problem: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "[CARDION] The complete error report (full details, nothing cut off) is saved here:" -ForegroundColor Red
    Write-Host "          $errLog" -ForegroundColor Red
    Show-File $errLog
    exit 1
}
