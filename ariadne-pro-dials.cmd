@echo off
setlocal
chcp 65001 >nul
set "PYTHONNOUSERSITE=1"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PATH=%~dp0dials-env;%~dp0dials-env\Scripts;%~dp0dials-env\Library\bin;%PATH%"
if not exist "%~dp0dials-env\python.exe" (
  echo DIALS runtime is missing. Run install_dials_windows.ps1 first.
  exit /b 1
)
"%~dp0dials-env\python.exe" "%~dp0tools\dials_cli.py" %*
exit /b %errorlevel%
