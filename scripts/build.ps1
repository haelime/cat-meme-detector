[CmdletBinding()]
param(
    [ValidateSet('Release', 'Debug')]
    [string]$Configuration = 'Release',
    [switch]$SkipOpenCVBuild,
    [switch]$ReconfigureOpenCV,
    [switch]$EnableCuda
)

$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.IO.Compression.FileSystem
$projectRoot = Split-Path -Parent $PSScriptRoot
$dependencyRoot = Join-Path $projectRoot '.deps'
$opencvSource = Join-Path $dependencyRoot 'opencv-5.0.0'
$contribSource = Join-Path $dependencyRoot 'opencv_contrib-5.0.0'
$buildSuffix = if ($EnableCuda) { '-cuda' } else { '' }
$opencvBuild = Join-Path $dependencyRoot ("opencv-build$buildSuffix")
$opencvInstall = Join-Path $dependencyRoot ("opencv-install$buildSuffix")
$appBuild = Join-Path $projectRoot ("build-opencv5$buildSuffix")

if ($EnableCuda) {
    $nvcc = Get-Command nvcc.exe -ErrorAction SilentlyContinue
    if (-not $nvcc) {
        throw 'CUDA Toolkit was not found. Install CUDA Toolkit 13.x, reopen PowerShell, then rerun with -EnableCuda.'
    }
    $cudaRoot = Split-Path -Parent (Split-Path -Parent $nvcc.Source)
    $cudnnRoots = @($cudaRoot)
    if ($env:CUDNN_ROOT) { $cudnnRoots += $env:CUDNN_ROOT }
    $standardCudnnRoot = 'C:\Program Files\NVIDIA\CUDNN'
    if (Test-Path -LiteralPath $standardCudnnRoot) { $cudnnRoots += $standardCudnnRoot }
    $cudnnHeader = $cudnnRoots | ForEach-Object {
        Get-ChildItem -LiteralPath $_ -Filter cudnn_version.h -Recurse `
            -ErrorAction SilentlyContinue
    } | Select-Object -First 1
    if (-not $cudnnHeader) {
        throw 'cuDNN was not found. Install cuDNN 9 and set CUDNN_ROOT if it is in a custom directory.'
    }
    $cudnnLibrary = Get-ChildItem -LiteralPath (Split-Path -Parent (Split-Path -Parent $cudnnHeader.FullName)) `
        -Filter cudnn.lib -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $cudnnLibrary) { throw 'cudnn.lib was not found near the cuDNN headers.' }
}

$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
if (-not (Test-Path -LiteralPath $vswhere)) {
    throw 'Visual Studio Installer (vswhere.exe) was not found.'
}
$visualStudio = & $vswhere -latest -products * `
    -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
    -property installationPath
if (-not $visualStudio) {
    throw 'Visual Studio C++ Build Tools were not found.'
}
$developerPrompt = Join-Path $visualStudio 'Common7\Tools\VsDevCmd.bat'
$assembler = Get-ChildItem -LiteralPath (Join-Path $visualStudio 'VC\Tools\MSVC') `
    -Filter ml64.exe -Recurse |
    Where-Object { $_.FullName -like '*Hostx64*x64*' } |
    Select-Object -First 1
if (-not $assembler) {
    throw 'The Visual Studio x64 assembler (ml64.exe) was not found.'
}

$ninjaCommand = Get-Command ninja.exe -ErrorAction SilentlyContinue
if ($ninjaCommand) {
    $ninja = $ninjaCommand.Source
} else {
    $ninja = Join-Path $visualStudio 'Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe'
}
if (-not (Test-Path -LiteralPath $ninja)) {
    throw 'Ninja was not found. Install it with: winget install Ninja-build.Ninja'
}

function Invoke-DeveloperCommand([string]$command) {
    $fullCommand = '"' + $developerPrompt + '" -arch=x64 -host_arch=x64 >nul && ' + $command
    & cmd.exe /d /s /c $fullCommand
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code $LASTEXITCODE"
    }
}

function Get-AndExpand([string]$url, [string]$archive, [string]$destination) {
    if (Test-Path -LiteralPath $destination) {
        return
    }
    New-Item -ItemType Directory -Path $dependencyRoot -Force | Out-Null
    if (Test-Path -LiteralPath $archive) {
        try {
            $zip = [System.IO.Compression.ZipFile]::OpenRead($archive)
            $zip.Dispose()
        } catch {
            Write-Host "Removing incomplete archive $archive"
            Remove-Item -LiteralPath $archive -Force
        }
    }
    if (-not (Test-Path -LiteralPath $archive)) {
        Write-Host "Downloading $url"
        & curl.exe --fail --location --retry 3 --output $archive $url
        if ($LASTEXITCODE -ne 0) {
            throw "Download failed with exit code $LASTEXITCODE"
        }
    }
    Expand-Archive -LiteralPath $archive -DestinationPath $dependencyRoot -Force
}

$installedConfig = Get-ChildItem -LiteralPath $opencvInstall -Filter OpenCVConfig.cmake `
    -Recurse -ErrorAction SilentlyContinue |
    Where-Object { $_.Directory.Name -eq 'staticlib' } |
    Select-Object -First 1
if (-not $SkipOpenCVBuild -and ($ReconfigureOpenCV -or -not $installedConfig)) {
    Get-AndExpand `
        'https://github.com/opencv/opencv/archive/refs/tags/5.0.0.zip' `
        (Join-Path $dependencyRoot 'opencv-5.0.0.zip') `
        $opencvSource
    Get-AndExpand `
        'https://github.com/opencv/opencv_contrib/archive/refs/tags/5.0.0.zip' `
        (Join-Path $dependencyRoot 'opencv_contrib-5.0.0.zip') `
        $contribSource

    $mlasPatch = Join-Path $projectRoot 'patches\opencv-5.0-disable-mlas-option.patch'
    & git.exe -C $opencvSource apply --check $mlasPatch 2>$null
    if ($LASTEXITCODE -eq 0) {
        & git.exe -C $opencvSource apply $mlasPatch
        if ($LASTEXITCODE -ne 0) { throw 'Could not apply the OpenCV MLAS compatibility patch.' }
    } else {
        & git.exe -C $opencvSource apply --reverse --check $mlasPatch 2>$null
        if ($LASTEXITCODE -ne 0) { throw 'The OpenCV MLAS compatibility patch no longer applies.' }
    }

    $opencvConfigure = @(
        'cmake', '-S', ('"' + $opencvSource + '"'), '-B', ('"' + $opencvBuild + '"'),
        '-G', 'Ninja',
        ('-DCMAKE_MAKE_PROGRAM="' + $ninja + '"'),
        ('-DCMAKE_ASM_COMPILER="' + $assembler.FullName + '"'),
        ('-DCMAKE_BUILD_TYPE=' + $Configuration),
        '-UCMAKE_MSVC_RUNTIME_LIBRARY',
        '-DCMAKE_POLICY_DEFAULT_CMP0091=OLD',
        ('-DCMAKE_INSTALL_PREFIX="' + $opencvInstall + '"'),
        ('-DOPENCV_EXTRA_MODULES_PATH="' + (Join-Path $contribSource 'modules') + '"'),
        '-DBUILD_LIST=core,dnn,imgproc,imgcodecs,features,highgui,videoio,xobjdetect',
        '-DBUILD_SHARED_LIBS=OFF',
        '-DBUILD_WITH_STATIC_CRT=OFF',
        '-DBUILD_TESTS=OFF', '-DBUILD_PERF_TESTS=OFF', '-DBUILD_EXAMPLES=OFF',
        '-DBUILD_opencv_apps=OFF', '-DBUILD_JAVA=OFF', '-DBUILD_opencv_python_bindings_generator=OFF',
        '-DWITH_MSMF=ON', '-DWITH_DSHOW=ON', '-DWITH_OPENCL=OFF',
        '-DWITH_IPP=OFF', '-DWITH_ITT=OFF',
        '-DWITH_PROTOBUF=ON', '-DBUILD_PROTOBUF=ON', '-DWITH_FLATBUFFERS=OFF',
        '-DOPENCV_DNN_DISABLE_MLAS=ON',
        '-DWITH_ADE=OFF', '-DWITH_PTHREADS_PF=OFF'
    )
    if ($EnableCuda) {
        $opencvConfigure += @(
            '-DWITH_CUDA=ON', '-DWITH_CUBLAS=ON', '-DWITH_CUDNN=ON',
            '-DOPENCV_DNN_CUDA=ON', '-DCUDA_ARCH_BIN=12.0',
            ('-DCUDNN_INCLUDE_DIR="' + $cudnnHeader.DirectoryName + '"'),
            ('-DCUDNN_LIBRARY="' + $cudnnLibrary.FullName + '"'),
            '-DENABLE_FAST_MATH=ON', '-DCUDA_FAST_MATH=ON'
        )
    } else {
        $opencvConfigure += '-DWITH_CUDA=OFF'
    }
    $opencvConfigure = $opencvConfigure -join ' '
    Invoke-DeveloperCommand $opencvConfigure
    Invoke-DeveloperCommand ('cmake --build "' + $opencvBuild + '" --config ' + $Configuration)
    Invoke-DeveloperCommand ('cmake --install "' + $opencvBuild + '" --config ' + $Configuration)
}

$opencvConfig = Get-ChildItem -LiteralPath $opencvInstall -Filter OpenCVConfig.cmake -Recurse |
    Where-Object { $_.Directory.Name -eq 'staticlib' } |
    Select-Object -First 1
if (-not $opencvConfig) {
    throw 'OpenCV 5 installation was not found. Run this script without -SkipOpenCVBuild first.'
}

$appConfigure = @(
    'cmake', '-S', ('"' + $projectRoot + '"'), '-B', ('"' + $appBuild + '"'),
    '-G', 'Ninja',
    ('-DCMAKE_MAKE_PROGRAM="' + $ninja + '"'),
    ('-DCMAKE_BUILD_TYPE=' + $Configuration),
    '-DOpenCV_STATIC=ON',
    ('-DOpenCV_DIR="' + $opencvConfig.DirectoryName + '"')
) -join ' '
Invoke-DeveloperCommand $appConfigure
Invoke-DeveloperCommand ('cmake --build "' + $appBuild + '" --config ' + $Configuration)

$ffmpegPlugin = Get-ChildItem -LiteralPath $opencvInstall `
    -Filter 'opencv_videoio_ffmpeg*_64.dll' -Recurse -ErrorAction SilentlyContinue |
    Select-Object -First 1
if ($ffmpegPlugin) {
    Copy-Item -LiteralPath $ffmpegPlugin.FullName -Destination $appBuild -Force
}
Invoke-DeveloperCommand ('ctest --test-dir "' + $appBuild + '" -C ' + $Configuration + ' --output-on-failure')

Write-Host "Built: $(Join-Path $appBuild 'cat_meme_detector.exe')"
