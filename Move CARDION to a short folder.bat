@echo off
title CARDION - move to a short folder
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\move_to_short_folder.ps1"
if errorlevel 1 pause
