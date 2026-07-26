$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$localRoot = Join-Path $repoRoot ".local"
$llamaRoot = Join-Path $localRoot "llama.cpp"
$modelRoot = Join-Path $localRoot "models"
$modelPath = Join-Path $modelRoot "Nanbeige4.2-3B-Q8_0.gguf"
$modelUrl = "https://huggingface.co/Abiray/Nanbeige4.2-3B-GGUF/resolve/774a61f8217ad18e7e102107fb7abcfecfae6a99/Nanbeige4.2-3B-Q8_0.gguf"
$modelSha256 = "76627e550979d8ea5746cb11922ad10352d91546aa540c0e8292522f8dd9c2b5"
$llamaCommit = "d28da865bf284acaecc98ad18a3c1f607c0fd754"
$safeDirectory = ($llamaRoot -replace "\\", "/")

New-Item -ItemType Directory -Path $localRoot, $modelRoot -Force | Out-Null

if (-not (Test-Path (Join-Path $llamaRoot ".git"))) {
    git clone --branch nanbeige42 --single-branch https://github.com/Nanbeige/llama.cpp.git $llamaRoot
}
git -c "safe.directory=$safeDirectory" -C $llamaRoot checkout --detach $llamaCommit
if ((git -c "safe.directory=$safeDirectory" -C $llamaRoot rev-parse HEAD) -ne $llamaCommit) {
    throw "Nanbeige llama.cpp commit verification failed"
}

if (-not (Test-Path $modelPath)) {
    Start-BitsTransfer -Source $modelUrl -Destination $modelPath
}
if ((Get-FileHash -Algorithm SHA256 $modelPath).Hash.ToLowerInvariant() -ne $modelSha256) {
    throw "Nanbeige Q8_0 GGUF hash verification failed"
}

$vsdev = "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"
if (-not (Test-Path $vsdev)) {
    throw "Visual Studio 2022 C++ build tools are required"
}
$environment = & cmd.exe /d /s /c "`"$vsdev`" -arch=amd64 -host_arch=amd64 >nul && set"
foreach ($line in $environment) {
    $parts = $line -split "=", 2
    if ($parts.Count -eq 2) {
        [Environment]::SetEnvironmentVariable($parts[0], $parts[1], "Process")
    }
}
$env:CC = "cl"
$env:CXX = "cl"
$env:GIT_CONFIG_COUNT = "1"
$env:GIT_CONFIG_KEY_0 = "safe.directory"
$env:GIT_CONFIG_VALUE_0 = $safeDirectory

$buildRoot = Join-Path $llamaRoot "build-cuda"
cmake -G Ninja -S $llamaRoot -B $buildRoot `
    -DGGML_CUDA=ON `
    -DLLAMA_CURL=OFF `
    -DLLAMA_BUILD_UI=OFF `
    -DGGML_CCACHE=OFF `
    -DCMAKE_BUILD_TYPE=Release
cmake --build $buildRoot --target llama-server --parallel 8

$serverPath = Join-Path $buildRoot "bin\llama-server.exe"
[pscustomobject]@{
    llama_cpp_commit = $llamaCommit
    model_path = $modelPath
    model_sha256 = $modelSha256
    server_path = $serverPath
    server_sha256 = (Get-FileHash -Algorithm SHA256 $serverPath).Hash.ToLowerInvariant()
} | ConvertTo-Json
