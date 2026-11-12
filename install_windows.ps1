param([switch]$BuildKernel)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$root = $PSScriptRoot
$python = Join-Path $root 'python312\tools\python.exe'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'
$env:PYTHONNOUSERSITE = '1'
New-Item -ItemType Directory -Path "$root\downloads", "$root\results" -Force | Out-Null
if (-not (Test-Path -LiteralPath $python)) {
    $package = Join-Path $root 'downloads\python.3.12.10.nupkg'
    Invoke-WebRequest -UseBasicParsing -Uri 'https://www.nuget.org/api/v2/package/python/3.12.10' -OutFile $package
    if (Test-Path "$root\python312") { throw 'python312 目录已存在但解释器缺失，停止以保护已有内容。' }
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    [IO.Compression.ZipFile]::ExtractToDirectory($package, "$root\python312")
}
$env:PYTHONPATH = "$root\source\python\src;$root\source\native\src;$root\overlay"
$ErrorActionPreference = 'Continue'
& $python -m pip install --index-url https://pypi.tuna.tsinghua.edu.cn/simple --disable-pip-version-check --no-warn-script-location --only-binary=:all: --target "$root\overlay" -r "$root\requirements-win-tested.txt"
$code = $LASTEXITCODE
$ErrorActionPreference = 'Stop'
if ($code -ne 0) { exit $code }
$env:ARIADNE_BUILD_CPP = '0'
if (-not (Test-Path "$root\overlay\ariadne-0.8.0.dist-info")) {
    $ErrorActionPreference = 'Continue'
    & $python -m pip install --index-url https://pypi.tuna.tsinghua.edu.cn/simple --disable-pip-version-check --no-warn-script-location --no-deps --target "$root\overlay" "$root\source\python" "$root\source\native"
    $code = $LASTEXITCODE
    $ErrorActionPreference = 'Stop'
    if ($code -ne 0) { exit $code }
}
$extension = @(Get-ChildItem "$root\source\native\src\ariadne_xtal" -Filter '_intensity_cpp*.pyd')
if ($extension.Count -eq 0 -or $BuildKernel) {
    if ($extension.Count -gt 0) { throw '已有 C++ 扩展；请先备份并移走，避免覆盖。' }
    if (-not (Test-Path "$root\toolchain-zig\ziglang\zig.exe")) {
        & $python -m pip install --index-url https://pypi.tuna.tsinghua.edu.cn/simple --disable-pip-version-check --no-warn-script-location --only-binary=:all: --target "$root\toolchain-zig" 'ziglang==0.15.2'
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
    & $python "$root\tools\build_windows_kernel.py" --source "$root\source" --compiler "$root\toolchain-zig\ziglang\zig.exe"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
& "$root\ariadne-pro-win.cmd" capabilities
exit $LASTEXITCODE
