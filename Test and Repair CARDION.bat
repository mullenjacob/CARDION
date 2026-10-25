@echo off
rem Double-click to diagnose and automatically repair CARDION, then run every test (self-test, camera tests) and show one report.
setlocal
cd /d "%~dp0"
title CARDION test and repair
echo CARDION will now check the installation, repair routine problems, run the self-test and test your webcam (the camera light turns on for about 20 seconds).
echo.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\launch.ps1" -TestAndRepair
set RC=%ERRORLEVEL%
echo.
if "%RC%"=="0" (echo Finished. The report is out\test_and_repair_report.md ^(opened for you^).) else (echo Problems were found. See out\test_and_repair_report.md and the "logs" folder.)
if not "%CARDION_NO_PAUSE%"=="1" pause
exit /b %RC%