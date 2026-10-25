@echo off
rem Double-click to start CARDION. First run installs what it needs (internet required once); every launch checks and repairs itself.
setlocal
cd /d "%~dp0"
title CARDION
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\launch.ps1" %*
set RC=%ERRORLEVEL%
if not "%RC%"=="0" (
  echo.
  echo CARDION could not start or stopped with an error ^(code %RC%^).
  echo The complete error report was saved in the "logs" folder next to this file, and the newest one was opened for you.
  echo You can also double-click "Test and Repair CARDION.bat" to diagnose and fix the installation automatically.
  echo.
  if not "%CARDION_NO_PAUSE%"=="1" pause
)
exit /b %RC%