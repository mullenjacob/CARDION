@echo off
rem Double-click to run CARDION's automated self-test and see a PASS/FAIL report.
setlocal
cd /d "%~dp0"
title CARDION self-test
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\launch.ps1" -SelfTest
set RC=%ERRORLEVEL%
echo.
if not "%RC%"=="0" echo The self-test did not pass. Details: out\selftest_report.md and the newest file in the "logs" folder.
if not "%CARDION_NO_PAUSE%"=="1" pause
exit /b %RC%