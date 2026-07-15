@echo off
cd /d "%~dp0\.."
if exist "Run_Log.txt" del "Run_Log.txt"
"HGS Sediment-Nutrient Module.exe" --nogui
