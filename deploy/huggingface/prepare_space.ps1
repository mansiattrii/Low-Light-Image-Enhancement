<#
.SYNOPSIS
    Copy the files the Hugging Face Space needs into a local clone of the Space repository.

.DESCRIPTION
    Files come from the last commit (git archive), not the working folder, so only committed
    code is deployed. The Space runs the same Dockerfile as everywhere else, with two lines
    appended that switch it to the Streamlit demo.

.EXAMPLE
    .\deploy\huggingface\prepare_space.ps1 -Destination ..\low-light-space
    .\deploy\huggingface\prepare_space.ps1 -Destination ..\low-light-space -RepoUrl https://github.com/<user>/<repo>
#>
param(
    [Parameter(Mandatory = $true)][string]$Destination,
    [string]$RepoUrl = "",
    [string]$Ref = "HEAD"
)
$ErrorActionPreference = "Stop"

$root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$paths = @(
    "Dockerfile", ".dockerignore", "requirements.txt", "config.yaml", "app.py", "src",
    "weights/zerodce.pth", "results/tuned_params.yaml"
)

New-Item -ItemType Directory -Force $Destination | Out-Null
$Destination = (Resolve-Path $Destination).Path
# Remove code from an earlier run so deleted modules do not linger in the Space.
if (Test-Path (Join-Path $Destination "src")) {
    Remove-Item -Recurse -Force (Join-Path $Destination "src")
}

# Windows PowerShell pipes native output as text, which would corrupt a tar stream,
# so the archive goes through a temporary file.
$archive = Join-Path ([System.IO.Path]::GetTempPath()) "llie-space-$PID.tar"
try {
    git -C $root archive --format=tar -o $archive $Ref @paths
    if ($LASTEXITCODE -ne 0) { throw "git archive failed" }
    tar -xf $archive -C $Destination
    if ($LASTEXITCODE -ne 0) { throw "tar failed" }
} finally {
    Remove-Item -Force -ErrorAction SilentlyContinue $archive
}

$readme = Get-Content -Raw -Encoding UTF8 (Join-Path $PSScriptRoot "README.md")
if ($RepoUrl) {
    $readme = $readme.TrimEnd() + "`n`nSource code, results and write-up: $RepoUrl`n"
}
$utf8 = New-Object System.Text.UTF8Encoding($false)  # UTF-8 without BOM, LF line endings
[System.IO.File]::WriteAllText((Join-Path $Destination "README.md"), $readme.Replace("`r`n", "`n"), $utf8)

$spaceLines = @(
    "",
    "# Hugging Face Space: serve the Streamlit demo on app_port 8501 (see README.md).",
    "# Streamlit's XSRF check rejects file uploads that arrive through the Spaces proxy,",
    "# so it is turned off; the demo keeps no user state that XSRF would protect.",
    "ENV APP=demo \",
    "    STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false",
    ""
) -join "`n"
[System.IO.File]::AppendAllText((Join-Path $Destination "Dockerfile"), $spaceLines, $utf8)

# The Hub rejects binary files pushed with plain git, so the weights must go through Git LFS.
# New Spaces usually include this rule already; add it if it is missing.
$attributes = Join-Path $Destination ".gitattributes"
$hasRule = (Test-Path $attributes) -and (Select-String -Quiet -SimpleMatch "*.pth filter=lfs" $attributes)
if (-not $hasRule) {
    [System.IO.File]::AppendAllText($attributes, "*.pth filter=lfs diff=lfs merge=lfs -text`n", $utf8)
}

$commit = git -C $root rev-parse --short $Ref
Write-Host "Space files from commit $commit written to $Destination"
