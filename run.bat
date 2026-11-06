@echo off
setlocal
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  where py >nul 2>nul && (py -3 -m venv .venv) || (python -m venv .venv)
)
if not exist .venv\Scripts\python.exe (
  echo Python 3.9 or newer is required. Install it from https://www.python.org/downloads/ and run this file again.
  pause
  exit /b 1
)
call .venv\Scripts\activate.bat
fc /b requirements.txt .venv\requirements.installed >nul 2>nul && fc /b engine\requirements.txt .venv\engine-requirements.installed >nul 2>nul
if errorlevel 1 (
  echo Installing packages ^(first run downloads PyTorch, a few hundred MB^)...
  python -m pip install --upgrade pip
  pip install -r requirements.txt || (pause ^& exit /b 1)
  copy /y requirements.txt .venv\requirements.installed >nul
  copy /y engine\requirements.txt .venv\engine-requirements.installed >nul
)
python server\app.py --open %*
pause
