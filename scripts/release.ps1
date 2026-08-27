[CmdletBinding()]
param(
    [string]$Version,
    [string]$OutputDirectory = 'out',
    [switch]$SkipOpenCVBuild,
    [switch]$ReconfigureOpenCV
)

$ErrorActionPreference = 'Stop'

$projectRoot = Split-Path -Parent $PSScriptRoot
$cmakeFile = Join-Path $projectRoot 'CMakeLists.txt'
$cmakeContents = Get-Content -LiteralPath $cmakeFile -Raw

if (-not $Version) {
    $versionMatch = [regex]::Match(
        $cmakeContents,
        'project\s*\([^\)]*?\bVERSION\s+([0-9]+(?:\.[0-9]+){1,3})',
        [System.Text.RegularExpressions.RegexOptions]::IgnoreCase
    )
    if (-not $versionMatch.Success) {
        throw 'Could not read the project version from CMakeLists.txt. Pass -Version explicitly.'
    }
    $Version = $versionMatch.Groups[1].Value
}

if ($Version -notmatch '^[0-9A-Za-z][0-9A-Za-z.-]*$') {
    throw "Invalid release version '$Version'. Use only letters, numbers, dots, and hyphens."
}
if ($SkipOpenCVBuild -and $ReconfigureOpenCV) {
    throw '-SkipOpenCVBuild and -ReconfigureOpenCV cannot be used together.'
}

$buildParameters = @{ Configuration = 'Release' }
if ($SkipOpenCVBuild) { $buildParameters.SkipOpenCVBuild = $true }
if ($ReconfigureOpenCV) { $buildParameters.ReconfigureOpenCV = $true }

Write-Host "Building Cat Meme Detector $Version (Release)..."
& (Join-Path $PSScriptRoot 'build.ps1') @buildParameters

$buildDirectory = Join-Path $projectRoot 'build-opencv5'
$executable = Join-Path $buildDirectory 'cat_meme_detector.exe'
if (-not (Test-Path -LiteralPath $executable -PathType Leaf)) {
    throw "Release executable was not created: $executable"
}

if ([System.IO.Path]::IsPathRooted($OutputDirectory)) {
    $outputRoot = [System.IO.Path]::GetFullPath($OutputDirectory)
} else {
    $outputRoot = [System.IO.Path]::GetFullPath((Join-Path $projectRoot $OutputDirectory))
}
$packageName = "cat-meme-detector-v$Version-windows-x64"
$packageDirectory = Join-Path $outputRoot $packageName
$archive = Join-Path $outputRoot "$packageName.zip"
$checksum = "$archive.sha256"

New-Item -ItemType Directory -Path $outputRoot -Force | Out-Null
if (Test-Path -LiteralPath $packageDirectory) {
    Remove-Item -LiteralPath $packageDirectory -Recurse -Force
}
if (Test-Path -LiteralPath $archive) {
    Remove-Item -LiteralPath $archive -Force
}
if (Test-Path -LiteralPath $checksum) {
    Remove-Item -LiteralPath $checksum -Force
}
New-Item -ItemType Directory -Path $packageDirectory | Out-Null

Copy-Item -LiteralPath $executable -Destination $packageDirectory
Copy-Item -LiteralPath (Join-Path $projectRoot 'assets') `
    -Destination $packageDirectory -Recurse
Copy-Item -LiteralPath (Join-Path $projectRoot 'README.md') -Destination $packageDirectory
Copy-Item -LiteralPath (Join-Path $projectRoot 'THIRD_PARTY_NOTICES.md') `
    -Destination $packageDirectory

$ffmpegPlugins = @(Get-ChildItem -LiteralPath $buildDirectory `
    -Filter 'opencv_videoio_ffmpeg*_64.dll' -File)
if (-not $ffmpegPlugins) {
    throw 'The OpenCV FFmpeg plug-in was not found in the build directory.'
}
$ffmpegPlugins | Copy-Item -Destination $packageDirectory

# The executable uses the DLL form of the MSVC runtime. Bundle the redistributable
# x64 DLLs next to it so the ZIP also works on machines without Visual Studio.
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
if (-not (Test-Path -LiteralPath $vswhere -PathType Leaf)) {
    throw 'Visual Studio Installer (vswhere.exe) was not found.'
}
$visualStudio = & $vswhere -latest -products * `
    -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
    -property installationPath
if (-not $visualStudio) {
    throw 'Visual Studio C++ Build Tools were not found.'
}
$crtDirectory = Get-ChildItem -LiteralPath (Join-Path $visualStudio 'VC\Redist\MSVC') `
    -Filter 'msvcp140.dll' -File -Recurse |
    Where-Object { $_.FullName -match '\\x64\\Microsoft\.VC[0-9]+\.CRT\\msvcp140\.dll$' } |
    Sort-Object FullName -Descending |
    Select-Object -First 1 -ExpandProperty Directory
if (-not $crtDirectory) {
    throw 'The x64 Visual C++ redistributable runtime was not found.'
}

$runtimeLibraries = @(
    'msvcp140.dll',
    'concrt140.dll',
    'vcruntime140.dll',
    'vcruntime140_1.dll'
)
foreach ($library in $runtimeLibraries) {
    $source = Join-Path $crtDirectory.FullName $library
    if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
        throw "Required Visual C++ runtime library was not found: $source"
    }
    Copy-Item -LiteralPath $source -Destination $packageDirectory
}

# Check that every authored asset reached the package before testing the binary.
$sourceAssets = Join-Path $projectRoot 'assets'
$packagedAssets = Join-Path $packageDirectory 'assets'
foreach ($sourceFile in Get-ChildItem -LiteralPath $sourceAssets -File -Recurse) {
    $relativePath = $sourceFile.FullName.Substring($sourceAssets.Length).TrimStart('\', '/')
    $packagedFile = Join-Path $packagedAssets $relativePath
    if (-not (Test-Path -LiteralPath $packagedFile -PathType Leaf)) {
        throw "Release package is missing an authored asset: $relativePath"
    }
    if ((Get-Item -LiteralPath $packagedFile).Length -ne $sourceFile.Length) {
        throw "Release asset size does not match the source: $relativePath"
    }
}

$packagedExecutable = Join-Path $packageDirectory 'cat_meme_detector.exe'
Write-Host 'Checking the packaged executable and authored assets...'
& $packagedExecutable --check-assets
if ($LASTEXITCODE -ne 0) {
    throw "Packaged asset check failed with exit code $LASTEXITCODE"
}
$generatedLog = Join-Path $packageDirectory 'cat_meme_detector.log'
if (Test-Path -LiteralPath $generatedLog) {
    Remove-Item -LiteralPath $generatedLog -Force
}

Compress-Archive -LiteralPath $packageDirectory -DestinationPath $archive `
    -CompressionLevel Optimal
$archiveHash = (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant()
Set-Content -LiteralPath $checksum -Encoding ascii `
    -Value "$archiveHash  $([System.IO.Path]::GetFileName($archive))"

Write-Host "Release directory: $packageDirectory"
Write-Host "Release archive:   $archive"
Write-Host "SHA-256:           $archiveHash"
