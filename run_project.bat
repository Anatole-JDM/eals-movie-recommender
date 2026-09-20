@echo off
setlocal

cd /d "%~dp0"

set "PY=.\.venv\Scripts\python.exe"
set "LOG_DIR=.\logs"

if not exist "%PY%" (
  echo [ERROR] Python not found at %PY%
  echo Create the virtual environment first, then try again.
  pause
  exit /b 1
)

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "TS=%%i"
set "LOG_FILE=%LOG_DIR%\run_%TS%.log"

echo Logging output to: %LOG_FILE%
echo.

set "PY_ABS=%CD%\%PY%"
set "LOG_ABS=%CD%\%LOG_FILE%"

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "\"==== Run started: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ====\" | Tee-Object -FilePath '%LOG_ABS%' -Append"

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "\"[1/2] Installing dependencies...\" | Tee-Object -FilePath '%LOG_ABS%' -Append; & '%PY_ABS%' -m pip install -r requirements.txt *>&1 | Tee-Object -FilePath '%LOG_ABS%' -Append; exit $LASTEXITCODE"

if errorlevel 1 (
  echo.
  echo [ERROR] Dependency installation failed. See log: %LOG_FILE%
  pause
  exit /b 1
)

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "\"[2/2] Running main.py...\" | Tee-Object -FilePath '%LOG_ABS%' -Append; & '%PY_ABS%' main.py *>&1 | Tee-Object -FilePath '%LOG_ABS%' -Append; exit $LASTEXITCODE"

if errorlevel 1 (
  echo.
  echo [ERROR] Run failed. See log: %LOG_FILE%
  pause
  exit /b 1
)

echo.
echo Done.
echo Log saved to: %LOG_FILE%
pause
