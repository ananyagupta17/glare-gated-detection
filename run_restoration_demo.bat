@echo off
setlocal
cd /d "%~dp0"
set "DEMO_PYTHON=.venv\Scripts\python.exe"
if not exist "%DEMO_PYTHON%" set "DEMO_PYTHON=venv\Scripts\python.exe"
if not exist "%DEMO_PYTHON%" set "DEMO_PYTHON=%USERPROFILE%\anaconda3\envs\venv\python.exe"
if not exist "%DEMO_PYTHON%" set "DEMO_PYTHON=python"
"%DEMO_PYTHON%" src\demo_restoration.py %*
endlocal
