@echo off
setlocal

cd /d "%~dp0"

echo ============================================================
echo REQTRACE AI
echo Production Startup
echo ============================================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo ERROR: Python virtual environment was not found.
    echo.
    echo Expected:
    echo   .venv\Scripts\python.exe
    echo.
    echo Run the installation steps in README.md first.
    exit /b 1
)

echo Checking REQTRACE AI health...
echo.

".venv\Scripts\python.exe" health_check.py --skip-smoke-test

if errorlevel 1 (
    echo.
    echo ERROR: A required REQTRACE AI component is unavailable.
    echo Review the health-check output above.
    exit /b 1
)

echo.
echo Starting REQTRACE AI Streamlit application...
echo.

".venv\Scripts\python.exe" -m streamlit run app.py

endlocal