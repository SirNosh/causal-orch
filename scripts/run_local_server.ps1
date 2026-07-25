$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$manifest = Get-Content (Join-Path $repoRoot "configs\local_model_manifest.json") -Raw | ConvertFrom-Json
$modelPath = Join-Path $repoRoot $manifest.runtime.model_path
$serverPath = Join-Path $repoRoot $manifest.runtime.server_binary_path

if ((Get-FileHash -Algorithm SHA256 $modelPath).Hash.ToLowerInvariant() -ne $manifest.model_manifest.gguf_sha256) {
    throw "GGUF hash does not match configs/local_model_manifest.json"
}
if ((Get-FileHash -Algorithm SHA256 $serverPath).Hash.ToLowerInvariant() -ne $manifest.model_manifest.server_binary_sha256) {
    throw "llama-server hash does not match configs/local_model_manifest.json"
}

& $serverPath `
    --model $modelPath `
    --alias $manifest.model_manifest.requested_model_slug `
    --host $manifest.runtime.host `
    --port $manifest.runtime.port `
    --ctx-size $manifest.runtime.context_length `
    --parallel $manifest.runtime.parallel_slots `
    --gpu-layers $manifest.runtime.gpu_layers `
    --cache-type-k $manifest.runtime.kv_cache_type_k `
    --cache-type-v $manifest.runtime.kv_cache_type_v `
    --flash-attn on `
    --reasoning-preserve `
    --jinja
