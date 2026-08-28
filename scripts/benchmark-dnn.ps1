[CmdletBinding()]
param(
    [ValidateRange(1, 20)]
    [int]$Runs = 5,
    [ValidateRange(0, 1000)]
    [int]$Warmup = 10,
    [ValidateRange(1, 10000)]
    [int]$Iterations = 100,
    [string]$Output = 'benchmarks\results\dnn-cpu-vs-cuda.csv'
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$dependencyRoot = Join-Path $projectRoot '.deps'
$benchmark = Join-Path $projectRoot 'build-opencv5-cuda\cat_meme_dnn_benchmark.exe'
if (-not (Test-Path -LiteralPath $benchmark -PathType Leaf)) {
    throw 'The CUDA benchmark was not found. Run scripts\build.ps1 -EnableCuda first.'
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
$cufftRuntime = Get-ChildItem -LiteralPath $dependencyRoot -Filter cufft64_*.dll `
    -File -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $cudnnRuntime -or -not $cufftRuntime) {
    throw 'The project-local cuDNN/cuFFT runtime was not found.'
}

$runtimeDirectories = @(
    (Join-Path $cudaRoot 'bin'),
    (Join-Path $cudaRoot 'bin\x64'),
    $cudnnRuntime.DirectoryName,
    $cufftRuntime.DirectoryName
) | Select-Object -Unique
$combinedPath = ($runtimeDirectories -join ';') + ';' + $env:Path
# Some agent shells supply both Path and PATH. Start-Process treats those as a
# duplicate key on Windows, so normalize the process environment before sampling.
[Environment]::SetEnvironmentVariable('PATH', $null, 'Process')
[Environment]::SetEnvironmentVariable('Path', $combinedPath, 'Process')

if ([System.IO.Path]::IsPathRooted($Output)) {
    $outputPath = [System.IO.Path]::GetFullPath($Output)
} else {
    $outputPath = [System.IO.Path]::GetFullPath((Join-Path $projectRoot $Output))
}
New-Item -ItemType Directory -Path (Split-Path -Parent $outputPath) -Force | Out-Null

$rows = [System.Collections.Generic.List[object]]::new()
for ($run = 1; $run -le $Runs; ++$run) {
    foreach ($backend in @('cpu', 'cuda')) {
        Write-Host "Run $run/$Runs - $($backend.ToUpperInvariant())"
        $sampleOutput = Join-Path $env:TEMP ("cat-meme-gpu-{0}.csv" -f [guid]::NewGuid())
        $sampleError = Join-Path $env:TEMP ("cat-meme-gpu-{0}.err" -f [guid]::NewGuid())
        $sampler = Start-Process -FilePath 'nvidia-smi.exe' `
            -ArgumentList @(
                '--query-gpu=utilization.gpu,power.draw,memory.used',
                '--format=csv,noheader,nounits',
                '-lms', '100'
            ) `
            -RedirectStandardOutput $sampleOutput `
            -RedirectStandardError $sampleError `
            -WindowStyle Hidden -PassThru
        try {
            $json = & $benchmark --backend $backend --warmup $Warmup `
                --iterations $Iterations | Out-String
            if ($LASTEXITCODE -ne 0) {
                throw "The $backend benchmark failed with exit code $LASTEXITCODE"
            }
            $measurement = $json | ConvertFrom-Json
        } finally {
            if (-not $sampler.HasExited) {
                Stop-Process -Id $sampler.Id -Force
                Wait-Process -Id $sampler.Id -ErrorAction SilentlyContinue
            }
        }

        $gpuSamples = @(Get-Content -LiteralPath $sampleOutput -ErrorAction SilentlyContinue |
            ForEach-Object {
                $columns = $_ -split ','
                if ($columns.Count -eq 3) {
                    [pscustomobject]@{
                        Utilization = [double]$columns[0].Trim()
                        PowerWatts = [double]$columns[1].Trim()
                        MemoryMiB = [double]$columns[2].Trim()
                    }
                }
            })
        Remove-Item -LiteralPath $sampleOutput, $sampleError -Force -ErrorAction SilentlyContinue

        $gpuUtilizationMean = if ($gpuSamples) {
            ($gpuSamples | Measure-Object Utilization -Average).Average
        } else { 0.0 }
        $gpuUtilizationMax = if ($gpuSamples) {
            ($gpuSamples | Measure-Object Utilization -Maximum).Maximum
        } else { 0.0 }
        $gpuPowerMean = if ($gpuSamples) {
            ($gpuSamples | Measure-Object PowerWatts -Average).Average
        } else { 0.0 }
        $gpuMemoryDelta = if ($gpuSamples) {
            $memory = $gpuSamples | Measure-Object MemoryMiB -Minimum -Maximum
            $memory.Maximum - $memory.Minimum
        } else { 0.0 }

        $rows.Add([pscustomobject]@{
            run = $run
            requested_backend = $measurement.requested_backend
            actual_backend = $measurement.actual_backend
            iterations = $measurement.iterations
            warmup_cycles = $measurement.warmup_cycles
            opencv_threads = $measurement.opencv_threads
            model_load_ms = $measurement.model_load_ms
            first_cycle_ms = $measurement.first_cycle_ms
            face_mean_ms = $measurement.face_mean_ms
            face_p50_ms = $measurement.face_p50_ms
            face_p95_ms = $measurement.face_p95_ms
            hand_mean_ms = $measurement.hand_mean_ms
            hand_p50_ms = $measurement.hand_p50_ms
            hand_p95_ms = $measurement.hand_p95_ms
            cycle_mean_ms = $measurement.cycle_mean_ms
            cycle_p50_ms = $measurement.cycle_p50_ms
            cycle_p95_ms = $measurement.cycle_p95_ms
            cycles_per_second = $measurement.cycles_per_second
            images_per_second = $measurement.images_per_second
            process_cpu_ms = $measurement.process_cpu_ms
            cpu_core_equivalents = $measurement.cpu_core_equivalents
            peak_working_set_mib = $measurement.peak_working_set_mib
            gpu_samples = $gpuSamples.Count
            gpu_utilization_mean_percent = [math]::Round($gpuUtilizationMean, 3)
            gpu_utilization_max_percent = [math]::Round($gpuUtilizationMax, 3)
            gpu_power_mean_watts = [math]::Round($gpuPowerMean, 3)
            gpu_memory_delta_mib = [math]::Round($gpuMemoryDelta, 3)
            checksum = $measurement.checksum
        })
    }
}

$rows | Export-Csv -LiteralPath $outputPath -NoTypeInformation -Encoding utf8
$rows | Format-Table run, requested_backend, actual_backend, cycle_p50_ms, `
    cycle_p95_ms, images_per_second, gpu_utilization_mean_percent -AutoSize
Write-Host "Raw benchmark results: $outputPath"
