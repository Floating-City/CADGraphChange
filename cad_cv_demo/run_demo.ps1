param(
    [string]$Python = ""
)

$ErrorActionPreference = "Stop"
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root = Split-Path -Parent $Here
$env:PYTHONPATH = "$Root\vendor;$Root"
$env:MPLCONFIGDIR = "$Root\.mplcache"
$env:XDG_CACHE_HOME = "$Root\.cache"

if (-not $Python) {
    $BundledPython = "C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
    if (Test-Path -LiteralPath $BundledPython) {
        $Python = $BundledPython
    }
    else {
        $Python = "python"
    }
}

& $Python "$Here\cad_cv_pipeline.py" --source "$Root\test.dwg" --reference-dir "$Root" --output-dir "$Here\output"
& $Python "$Here\validate_outputs.py" "$Here\output\dxf"

# 深度学习检测/匹配（--matcher dl，默认开启）：内置 3.12 运行时无 torch 会自动回退启发式。
# 使用系统 Python 3.13（torch + vendor313）的完整 DL 流程：
#   python "$Here\train_dl_models.py" --source "$Root\test.dwg" --reference-dir "$Root"
#   python "$Here\cad_cv_pipeline.py" --matcher dl --source "$Root\test.dwg" --reference-dir "$Root" --output-dir "$Here\output"
