@echo off
cd /d "%~dp0"
python -m pip install pyinstaller
python -m PyInstaller --noconfirm --clean --onefile --console --hidden-import matplotlib.backends.backend_pdf --name "HGS Sediment-Nutrient Module" "Source\HGS Sediment-Nutrient Module.py"
copy /Y "dist\HGS Sediment-Nutrient Module.exe" ".\HGS Sediment-Nutrient Module.exe"
echo.
echo Built HGS Sediment-Nutrient Module.exe
pause
