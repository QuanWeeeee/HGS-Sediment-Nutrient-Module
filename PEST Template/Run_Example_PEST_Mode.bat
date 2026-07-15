@echo off
cd /d "%~dp0\.."
if exist "Run_Log.txt" del "Run_Log.txt"
set GWSWI_CONFIG_PATH=PEST Template\Model_Config_PEST_Example.txt
"HGS Sediment-Nutrient Module.exe" --nogui
echo.
echo PEST-mode example finished. Check Outputs\PEST_Run\pest_values.txt
pause
