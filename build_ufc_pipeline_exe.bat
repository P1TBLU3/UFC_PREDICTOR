@echo off
setlocal
cd /d "%~dp0"

echo ============================================================
echo Build UFC Predictor Pipeline EXE
echo ============================================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo ERROR: .venv\Scripts\python.exe was not found.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" -m pip install pyinstaller
if errorlevel 1 goto :error

if not exist "src\pipeline\run_ufc_pipeline.py" (
    echo ERROR: src\pipeline\run_ufc_pipeline.py is missing.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" -m PyInstaller ^
    --noconfirm ^
    --clean ^
    --onefile ^
    --console ^
    --name UFC_Predictor_Pipeline ^
    src\pipeline\run_ufc_pipeline.py

if errorlevel 1 goto :error

echo.
echo ============================================================
echo Build completed
echo ============================================================
echo EXE:
echo   dist\UFC_Predictor_Pipeline.exe
echo.
echo Copy the EXE to the UFC_PREDICTOR project root before using it.
pause
exit /b 0

:error
echo.
echo Build failed.
pause
exit /b 1
