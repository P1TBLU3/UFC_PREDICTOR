@echo off
setlocal
cd /d "%~dp0"

echo ============================================================
echo UFC Fight Predictor - Full Pipeline
echo ============================================================
echo.

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -m src.pipeline.run_ufc_pipeline %*
) else (
    echo ERROR: .venv\Scripts\python.exe was not found.
    echo Run this file from the UFC_PREDICTOR project root.
    pause
    exit /b 1
)

if errorlevel 1 (
    echo.
    echo Pipeline failed.
    pause
    exit /b 1
)

echo.
echo Pipeline completed successfully.
pause
