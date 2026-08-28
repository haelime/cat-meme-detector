$ErrorActionPreference = 'Stop'

$projectRoot = Split-Path -Parent $PSScriptRoot
$dependencyRoot = Join-Path $projectRoot '.deps'
$executable = Join-Path $projectRoot 'build-opencv5-cuda\cat_meme_detector.exe'
if (-not (Test-Path -LiteralPath $executable -PathType Leaf)) {
    throw 'The CUDA executable was not found. Run scripts\build.ps1 -EnableCuda first.'
}

$nvccPath = (Get-Command nvcc.exe -ErrorAction SilentlyContinue).Source
if (-not $nvccPath) {
    $machineCudaRoot = [Environment]::GetEnvironmentVariable('CUDA_PATH', 'Machine')
    if ($machineCudaRoot) {
        $candidateNvcc = Join-Path $machineCudaRoot 'bin\nvcc.exe'
        if (Test-Path -LiteralPath $candidateNvcc) { $nvccPath = $candidateNvcc }
    }
}
if (-not $nvccPath) { throw 'CUDA Toolkit was not found.' }
$cudaRoot = Split-Path -Parent (Split-Path -Parent $nvccPath)

$cudnnRoots = @($cudaRoot, $dependencyRoot)
if ($env:CUDNN_ROOT) { $cudnnRoots += $env:CUDNN_ROOT }
$cudnnRuntime = $cudnnRoots | ForEach-Object {
    Get-ChildItem -LiteralPath $_ -Filter cudnn64_9.dll -File -Recurse `
        -ErrorAction SilentlyContinue
} | Select-Object -First 1
if (-not $cudnnRuntime) { throw 'The cuDNN 9 runtime was not found.' }

$cufftRuntime = Get-ChildItem -LiteralPath $dependencyRoot -Filter cufft64_*.dll `
    -File -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $cufftRuntime) { throw 'The cuFFT runtime was not found under .deps.' }

$runtimeDirectories = @(
    (Join-Path $cudaRoot 'bin'),
    (Join-Path $cudaRoot 'bin\x64'),
    $cudnnRuntime.DirectoryName,
    $cufftRuntime.DirectoryName
) | Select-Object -Unique
$env:Path = ($runtimeDirectories -join ';') + ';' + $env:Path

& $executable @args
exit $LASTEXITCODE
