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
    $nvccPath = (Get-Command nvcc.exe -ErrorAction SilentlyContinue).Source
    if (-not $nvccPath) {
        $machineCudaRoot = [Environment]::GetEnvironmentVariable('CUDA_PATH', 'Machine')
        if ($machineCudaRoot) {
            $candidateNvcc = Join-Path $machineCudaRoot 'bin\nvcc.exe'
            if (Test-Path -LiteralPath $candidateNvcc) { $nvccPath = $candidateNvcc }
        }
    }
    if (-not $nvccPath) {
        throw 'CUDA Toolkit was not found. Install CUDA Toolkit 13.x, reopen PowerShell, then rerun with -EnableCuda.'
    }
    $cudaRoot = Split-Path -Parent (Split-Path -Parent $nvccPath)
    $env:CUDA_PATH = $cudaRoot
    $cudnnRoots = @($cudaRoot, $dependencyRoot)
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
    $cudnnRuntime = Get-ChildItem -LiteralPath (Split-Path -Parent (Split-Path -Parent $cudnnHeader.FullName)) `
        -Filter cudnn64_9.dll -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $cudnnRuntime) { throw 'cudnn64_9.dll was not found near the cuDNN headers.' }

    # A selective CUDA Toolkit install may omit cuFFT. Allow an official
    # project-local redistribution to provide the import library instead.
    $cufftLibrary = Get-ChildItem -LiteralPath $dependencyRoot -Filter cufft.lib `
        -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
    $cufftRuntime = Get-ChildItem -LiteralPath $dependencyRoot -Filter cufft64_*.dll `
        -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
    $runtimeDirectories = @(
        (Join-Path $cudaRoot 'bin'),
        (Join-Path $cudaRoot 'bin\x64'),
        $cudnnRuntime.DirectoryName
    )
    if ($cufftRuntime) { $runtimeDirectories += $cufftRuntime.DirectoryName }
    $env:Path = (($runtimeDirectories | Select-Object -Unique) -join ';') + ';' + $env:Path
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

function Apply-SourcePatch([string]$source, [string]$patch, [string]$description) {
    $rootPath = [System.IO.Path]::GetFullPath($projectRoot).TrimEnd('\')
    $sourcePath = [System.IO.Path]::GetFullPath($source)
    if (-not $sourcePath.StartsWith($rootPath + '\', [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Patch source is outside the project: $sourcePath"
    }
    $sourcePrefix = $sourcePath.Substring($rootPath.Length + 1).Replace('\', '/')
    $directoryArgument = '--directory=' + $sourcePrefix

    $previousErrorAction = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    & git.exe -C $projectRoot apply --check $directoryArgument $patch 2>$null
    $canApply = $LASTEXITCODE -eq 0
    $ErrorActionPreference = $previousErrorAction
    if ($canApply) {
        & git.exe -C $projectRoot apply $directoryArgument $patch
        if ($LASTEXITCODE -ne 0) { throw "Could not apply $description." }
        return
    }

    $ErrorActionPreference = 'Continue'
    & git.exe -C $projectRoot apply --reverse --check $directoryArgument $patch 2>$null
    $isApplied = $LASTEXITCODE -eq 0
    $ErrorActionPreference = $previousErrorAction
    if (-not $isApplied) { throw "$description no longer applies." }
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

    Apply-SourcePatch `
        $opencvSource `
        (Join-Path $projectRoot 'patches\opencv-5.0-disable-mlas-option.patch') `
        'the OpenCV MLAS compatibility patch'
    if ($EnableCuda) {
        Apply-SourcePatch `
            $contribSource `
            (Join-Path $projectRoot 'patches\opencv-contrib-5.0-windows-ulong.patch') `
            'the OpenCV contrib Windows CUDA compatibility patch'
    }

    $opencvModules = @('core', 'dnn', 'imgproc', 'imgcodecs', 'features',
                       'highgui', 'videoio', 'xobjdetect')
    if ($EnableCuda) {
        # OpenCV core requires the opencv_contrib cudev module whenever CUDA is enabled.
        $opencvModules += 'cudev'
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
        ('-DBUILD_LIST=' + ($opencvModules -join ',')),
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
        if ($cufftLibrary) {
            $opencvConfigure += ('-DCUDA_cufft_LIBRARY="' + $cufftLibrary.FullName + '"')
        }
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
$ctestCommand = 'ctest --test-dir "' + $appBuild + '" -C ' + $Configuration
if ($EnableCuda) {
    # Keep separately linked CUDA processes in separate CTest hosts on Windows.
    Invoke-DeveloperCommand ($ctestCommand + ' -R cat_meme_bundled_assets --output-on-failure')
    Invoke-DeveloperCommand ($ctestCommand + ' -R "^cat_meme_tests$" --output-on-failure')
} else {
    Invoke-DeveloperCommand ($ctestCommand + ' --output-on-failure')
}

Write-Host "Built: $(Join-Path $appBuild 'cat_meme_detector.exe')"
