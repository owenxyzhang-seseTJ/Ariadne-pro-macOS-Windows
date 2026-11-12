@echo off
setlocal
chcp 65001 >nul
set "ARIADNE_PRO_ROOT=%~dp0"
set "PYTHONNOUSERSITE=1"
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"
set "PYTHONPATH=%~dp0source\python\src;%~dp0source\native\src;%~dp0overlay"
set "PATH=%~dp0python312\tools;%~dp0python312\tools\DLLs;%PATH%"
if exist "%~dp0dials-env.cmd" call "%~dp0dials-env.cmd"
"%~dp0python312\tools\python.exe" "%~dp0tools\windows_cli.py" %*
exit /b %errorlevel%
