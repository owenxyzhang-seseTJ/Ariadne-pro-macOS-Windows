$ErrorActionPreference='Stop'
$root=$PSScriptRoot
$python=Join-Path $root 'python312\tools\python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw '请先运行 install_windows.ps1 安装计算版。' }
$env:PYTHONUTF8='1'
$env:PYTHONIOENCODING='utf-8'
Set-Location -LiteralPath $root
& $python -I "$root\tools\setup_dials_windows.py"
exit $LASTEXITCODE
