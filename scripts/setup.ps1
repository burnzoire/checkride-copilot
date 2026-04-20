$ErrorActionPreference = "Stop"

param(
    [switch]$SkipOllama,
    [switch]$SkipModelPull,
    [switch]$SkipDcsHook,
    [switch]$NonInteractive
)

function Get-PythonExe {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        return "py"
    }
    if (Get-Command python -ErrorAction SilentlyContinue) {
        return "python"
    }
    throw "Python 3.11+ is required. Install Python, then run setup.ps1 again."
}

function Assert-PythonVersion([string]$pythonExe) {
    if ($pythonExe -eq "py") {
        $version = (& py -3 -c "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')").Trim()
    } else {
        $version = (& python -c "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')").Trim()
    }

    $parts = $version.Split(".")
    if ($parts.Length -lt 2) {
        throw "Unable to determine Python version."
    }
    $major = [int]$parts[0]
    $minor = [int]$parts[1]
    if ($major -ne 3 -or $minor -lt 11) {
        throw "Python 3.11+ is required (detected $version)."
    }
}

function Refresh-Path {
    $env:PATH = [System.Environment]::GetEnvironmentVariable("PATH", "Machine") + ";" +
                [System.Environment]::GetEnvironmentVariable("PATH", "User")
}

function Assert-TrustedInstaller([string]$path, [string]$expectedPublisherContains) {
    $sig = Get-AuthenticodeSignature -FilePath $path
    if ($sig.Status -ne "Valid") {
        throw "Installer signature validation failed with status: $($sig.Status)"
    }
    if (-not ($sig.SignerCertificate.Subject -match "^CN=$expectedPublisherContains(\b|,)")) {
        throw "Unexpected installer signer: $($sig.SignerCertificate.Subject)"
    }
}

function Ensure-Uv {
    if (Get-Command uv -ErrorAction SilentlyContinue) {
        return
    }

    Write-Host "Installing uv..." -ForegroundColor Yellow
    $py = Get-PythonExe
    Assert-PythonVersion -pythonExe $py
    if ($py -eq "py") {
        & py -3 -m pip install --user uv
    } else {
        & python -m pip install --user uv
    }
    Refresh-Path

    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        throw "uv installation completed but uv was not found in PATH. Open a new terminal and retry."
    }
}

function Get-GpuInfo {
    $gpus = @()
    try {
        $gpus = @(Get-CimInstance Win32_VideoController -ErrorAction Stop)
    } catch {
        return [pscustomobject]@{
            HasNvidia = $false
            IsRtx50 = $false
            Names = @()
        }
    }

    $nvidia = @($gpus | Where-Object { $_.Name -match "NVIDIA" })
    # RTX 50-series consumer cards currently use 4-digit model naming (e.g., RTX 5090).
    $rtx50 = @($nvidia | Where-Object { $_.Name -match "RTX\s*50\d{2}" })
    return [pscustomobject]@{
        HasNvidia = $nvidia.Count -gt 0
        IsRtx50 = $rtx50.Count -gt 0
        Names = @($gpus | ForEach-Object { $_.Name })
    }
}

function Get-TorchRecommendation([pscustomobject]$gpu) {
    if (-not $gpu.HasNvidia) {
        return [pscustomobject]@{
            Label = "CPU build"
            Why = "No NVIDIA GPU detected."
            InstallArgs = @("pip", "install", "--upgrade", "--force-reinstall", "torch", "--index-url", "https://download.pytorch.org/whl/cpu")
        }
    }

    if ($gpu.IsRtx50) {
        return [pscustomobject]@{
            Label = "CUDA 12.8 nightly (RTX 50xx)"
            Why = "RTX 50-series GPU detected."
            InstallArgs = @("pip", "install", "--pre", "--upgrade", "--force-reinstall", "torch", "--index-url", "https://download.pytorch.org/whl/nightly/cu128")
        }
    }

    return [pscustomobject]@{
        Label = "CUDA 12.1 stable (RTX 10xx-40xx)"
        Why = "NVIDIA GPU detected."
        InstallArgs = @("pip", "install", "--upgrade", "--force-reinstall", "torch", "--index-url", "https://download.pytorch.org/whl/cu121")
    }
}

function Confirm-Step([string]$message) {
    if ($NonInteractive) {
        return $true
    }
    $answer = Read-Host "$message [Y/n]"
    return ($answer -eq "" -or $answer -match "^[Yy]")
}

$runningOnWindows = $env:OS -eq "Windows_NT"
if (-not $runningOnWindows) {
    throw "This installer currently supports Windows only."
}

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

Write-Host ""
Write-Host "=== Checkride Copilot Source Setup ===" -ForegroundColor Cyan
Write-Host "Repo: $root" -ForegroundColor DarkGray
Write-Host ""

$pythonExe = Get-PythonExe
Assert-PythonVersion -pythonExe $pythonExe

$gpu = Get-GpuInfo
if ($gpu.Names.Count -gt 0) {
    Write-Host "Detected GPU(s):" -ForegroundColor DarkGray
    foreach ($name in $gpu.Names) {
        Write-Host "  - $name" -ForegroundColor DarkGray
    }
} else {
    Write-Host "Detected GPU(s): unavailable (WMI query failed)." -ForegroundColor DarkGray
}

$torch = Get-TorchRecommendation -gpu $gpu
Write-Host ""
Write-Host "Recommended PyTorch: $($torch.Label)" -ForegroundColor Green
Write-Host "Reason: $($torch.Why)" -ForegroundColor DarkGray

if (-not (Confirm-Step "Proceed with this PyTorch selection and full install?")) {
    Write-Host "Cancelled." -ForegroundColor Yellow
    exit 0
}

Ensure-Uv

Write-Host ""
Write-Host "Installing project dependencies with uv sync..." -ForegroundColor Yellow
& uv sync

Write-Host ""
Write-Host "Installing recommended PyTorch variant..." -ForegroundColor Yellow
$torchArgs = @($torch.InstallArgs)
& uv $torchArgs

Write-Host ""
Write-Host "Verifying torch install..." -ForegroundColor Yellow
& uv run python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

if (-not $SkipOllama) {
    if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
        Write-Host ""
        Write-Host "Installing Ollama..." -ForegroundColor Yellow
        $tmp = Join-Path $env:TEMP "OllamaSetup.exe"
        Invoke-WebRequest -Uri "https://ollama.com/download/OllamaSetup.exe" -OutFile $tmp
        Assert-TrustedInstaller -path $tmp -expectedPublisherContains "Ollama"
        Start-Process $tmp -ArgumentList "/S" -Wait
        Remove-Item $tmp -ErrorAction SilentlyContinue
        Refresh-Path
    } else {
        Write-Host ""
        Write-Host "Ollama already installed." -ForegroundColor Green
    }

    if (-not $SkipModelPull) {
        Write-Host "Pulling Ollama model qwen2.5:7b..." -ForegroundColor Yellow
        & ollama pull qwen2.5:7b
    }
}

if (-not $SkipDcsHook) {
    Write-Host ""
    Write-Host "Installing DCS Export.lua hook..." -ForegroundColor Yellow
    & uv run python scripts/install.py
}

Write-Host ""
Write-Host "Setup complete." -ForegroundColor Green
Write-Host "Run: uv run start-demo" -ForegroundColor Cyan
