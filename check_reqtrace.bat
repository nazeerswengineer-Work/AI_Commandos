@echo off
setlocal

cd /d "%~dp0"

echo ============================================================
echo REQTRACE AI
echo Production Verification
echo ============================================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo ERROR: .venv was not found.
    exit /b 1
)

echo [1/3] Running health check...
".venv\Scripts\python.exe" health_check.py

if errorlevel 1 (
    echo.
    echo Health check FAILED.
    exit /b 1
)

echo.
echo [2/3] Running regression suite...
".venv\Scripts\python.exe" -m pytest -q

if errorlevel 1 (
    echo.
    echo Regression suite FAILED.
    exit /b 1
)

echo.
echo [3/3] Running production benchmark...
".venv\Scripts\python.exe" evaluate_production_app.py

if errorlevel 1 (
    echo.
    echo Production benchmark FAILED.
    exit /b 1
)

echo.
echo ============================================================
echo REQTRACE AI VERIFICATION PASSED
echo ============================================================

endlocal