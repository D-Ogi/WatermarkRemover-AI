# WatermarkRemover-AI Setup Script
$Host.UI.RawUI.WindowTitle = "WatermarkRemover-AI Setup"

Set-Location -LiteralPath $PSScriptRoot

$PYTHON_VERSION = "3.12.7"
$PYTHON_DIR = "python"
$PYTHON_EXE = "$PYTHON_DIR\python.exe"

# China mirror configuration
$CHINA_MODE = $false
$PIP_INDEX_URL = ""
$PIP_EXTRA_ARGS = @()
$HF_ENDPOINT = "https://huggingface.co"

# Fun facts and tips to show during installation
$tips = @(
    @{icon="[i]"; color="Cyan"; text="Florence-2 can detect watermarks in any language - even emojis!"},
    @{icon="[?]"; color="Yellow"; text="Tip: Use 'Transparent mode' to keep the original background visible"},
    @{icon="[i]"; color="Cyan"; text="The AI model was trained on millions of images to understand context"},
    @{icon="[?]"; color="Yellow"; text="Tip: Lower 'Max detection size' if the AI removes too much"},
    @{icon="[i]"; color="Cyan"; text="LaMA stands for 'Large Mask inpainting' - it fills gaps naturally"},
    @{icon="[?]"; color="Yellow"; text="Tip: GPU processing is 10-50x faster than CPU"},
    @{icon="[i]"; color="Cyan"; text="This tool works on both images AND videos!"},
    @{icon="[?]"; color="Yellow"; text="Tip: Batch mode can process entire folders at once"},
    @{icon="[i]"; color="Cyan"; text="The AI analyzes each frame independently for best results"},
    @{icon="[?]"; color="Yellow"; text="Tip: PNG format preserves quality, JPG saves space"},
    @{icon="[i]"; color="Cyan"; text="Florence-2 is Microsoft's latest vision AI model"},
    @{icon="[?]"; color="Yellow"; text="Tip: Install FFmpeg to keep audio in processed videos"},
    @{icon="[i]"; color="Cyan"; text="The inpainting AI 'imagines' what should be behind the watermark"},
    @{icon="[?]"; color="Yellow"; text="Tip: Works best on watermarks that cover less than 10% of image"},
    @{icon="[i]"; color="Cyan"; text="Processing 4K video? Get some snacks, it takes a while"},
    @{icon="[?]"; color="Yellow"; text="Tip: Check the logs if something goes wrong"},
    @{icon="[i]"; color="Cyan"; text="The AI can handle semi-transparent watermarks too!"},
    @{icon="[?]"; color="Yellow"; text="Tip: Your settings are saved automatically between sessions"}
)

Write-Host ""
Write-Host "  =============================================" -ForegroundColor Cyan
Write-Host "     WatermarkRemover-AI Setup                 " -ForegroundColor Cyan
Write-Host "  =============================================" -ForegroundColor Cyan
Write-Host ""

# Check if user is in China (for mirror selection)
Write-Host "  [?] Are you in China? (y/n)" -ForegroundColor Yellow
Write-Host "      This will use faster mirrors for downloads" -ForegroundColor DarkGray
$chinaChoice = Read-Host "      "
if ($chinaChoice -eq "y" -or $chinaChoice -eq "Y") {
    $CHINA_MODE = $true
    $PIP_INDEX_URL = "https://pypi.tuna.tsinghua.edu.cn/simple"
    $PIP_EXTRA_ARGS = @("-i", $PIP_INDEX_URL, "--trusted-host", "pypi.tuna.tsinghua.edu.cn")
    $HF_ENDPOINT = "https://hf-mirror.com"
    Write-Host "  [OK] Using China mirrors (Tsinghua PyPI + HF-Mirror)" -ForegroundColor Green
} else {
    Write-Host "  [OK] Using default mirrors" -ForegroundColor Green
}
Write-Host ""

# Check if embedded Python exists
if (-not (Test-Path $PYTHON_EXE)) {
    Write-Host "  [*] Downloading Python $PYTHON_VERSION..." -ForegroundColor Cyan

    $arch = if ([Environment]::Is64BitOperatingSystem) { "amd64" } else { "win32" }
    $pythonZip = "python-$PYTHON_VERSION-embed-$arch.zip"
    $pythonUrl = "https://www.python.org/ftp/python/$PYTHON_VERSION/$pythonZip"

    try {
        # Download Python
        Invoke-WebRequest -Uri $pythonUrl -OutFile $pythonZip -UseBasicParsing
        Write-Host "  [OK] Downloaded Python" -ForegroundColor Green

        # Extract
        Write-Host "  [*] Extracting..." -ForegroundColor Cyan
        Expand-Archive -Path $pythonZip -DestinationPath $PYTHON_DIR -Force
        Remove-Item $pythonZip

        # Enable pip by modifying python312._pth
        $pthFile = Join-Path $PYTHON_DIR "python312._pth"
        if (Test-Path $pthFile) {
            $pthContent = Get-Content $pthFile -Raw
            $pthContent = $pthContent -replace "#import site", "import site"
            $pthContent = $pthContent + "`nLib\site-packages"
            Set-Content -Path $pthFile -Value $pthContent -NoNewline
        }

        # Create Lib\site-packages directory
        $sitePackages = Join-Path $PYTHON_DIR "Lib\site-packages"
        New-Item -ItemType Directory -Path $sitePackages -Force | Out-Null

        # Download and install pip
        Write-Host "  [*] Installing pip..." -ForegroundColor Cyan
        $getPipUrl = "https://bootstrap.pypa.io/get-pip.py"
        Invoke-WebRequest -Uri $getPipUrl -OutFile "get-pip.py" -UseBasicParsing
        & $PYTHON_EXE get-pip.py --no-warn-script-location 2>&1 | Out-Null
        Remove-Item "get-pip.py"

        Write-Host "  [OK] Python $PYTHON_VERSION ready" -ForegroundColor Green
    }
    catch {
        Write-Host "  [X] Failed to download Python: $_" -ForegroundColor Red
        Read-Host "  Press Enter to exit"
        exit 1
    }
}
else {
    Write-Host "  [OK] Python found" -ForegroundColor Green
}

# Make local packages importable in fresh and existing embedded runtimes.
$pthFile = Join-Path $PYTHON_DIR "python312._pth"
if (Test-Path -LiteralPath $pthFile) {
    $pthLines = @(Get-Content -LiteralPath $pthFile)
    if ($pthLines -notcontains "..") {
        Add-Content -LiteralPath $pthFile -Value "`n.."
    }
}

Write-Host ""
Write-Host "  [*] Installing dependencies..." -ForegroundColor Cyan
Write-Host "      This takes 5-10 minutes. Chill and learn something!" -ForegroundColor Magenta
Write-Host ""
Write-Host "      Did you know?" -ForegroundColor DarkGray
Write-Host ""

# Upgrade pip and ensure build tooling is available for sdists
if ($CHINA_MODE) {
    & $PYTHON_EXE -m pip install --upgrade pip setuptools wheel -i $PIP_INDEX_URL --trusted-host pypi.tuna.tsinghua.edu.cn 2>&1 | Out-Null
} else {
    & $PYTHON_EXE -m pip install --upgrade pip setuptools wheel 2>&1 | Out-Null
}

# Install dependencies with the normal resolver and retain the installation tips.
if ($CHINA_MODE) {
    # Use the selected package mirror.
    $process = Start-Process -FilePath $PYTHON_EXE -ArgumentList "-m", "pip", "install", "--upgrade", "-r", "requirements.txt", "--no-cache-dir", "-i", $PIP_INDEX_URL, "--trusted-host", "pypi.tuna.tsinghua.edu.cn" -NoNewWindow -PassThru
} else {
    $process = Start-Process -FilePath $PYTHON_EXE -ArgumentList "-m", "pip", "install", "--upgrade", "-r", "requirements.txt", "--no-cache-dir" -NoNewWindow -PassThru
}

# Retain the native process handle so ExitCode remains available after exit.
$processHandle = $process.Handle
$lastTipTime = Get-Date
$currentTip = Get-Random -Maximum $tips.Count

while (-not $process.HasExited) {
    $now = Get-Date
    if (($now - $lastTipTime).TotalSeconds -ge 5) {
        $tip = $tips[$currentTip]
        $line = "      $($tip.icon) $($tip.text)"
        $line = $line.PadRight(90)
        Write-Host "`r$line" -ForegroundColor $tip.color -NoNewline

        $currentTip = ($currentTip + 1) % $tips.Count
        $lastTipTime = $now
    }
    Start-Sleep -Milliseconds 300
}

Write-Host "`r                                                                                              "

$process.WaitForExit()
if ($process.ExitCode -ne 0) {
    Write-Host "  [X] Failed to install dependencies" -ForegroundColor Red
    exit 1
}
& $PYTHON_EXE -m pip check
if ($LASTEXITCODE -ne 0) {
    Write-Host "  [X] Dependency verification failed. Use a fresh application environment." -ForegroundColor Red
    exit 1
}
& $PYTHON_EXE -c "import remwm; import webview; import yaml; import psutil"
if ($LASTEXITCODE -ne 0) {
    Write-Host "  [X] Failed to import application packages" -ForegroundColor Red
    exit 1
}
Write-Host "  [OK] Dependencies installed and verified" -ForegroundColor Green

# All installers use the same checksum-verified, atomic model download.
Write-Host "  [*] Preparing LaMA model (196MB)..." -ForegroundColor Cyan
if ($CHINA_MODE) {
    Write-Host "      If GitHub is blocked, preseed the verified cache: docs/lama-runtime.md#restricted-networks" -ForegroundColor DarkGray
}
& $PYTHON_EXE -m lama_inpaint download --verbose
if ($LASTEXITCODE -ne 0) {
    Write-Host "  [X] Could not prepare verified LaMA weights. Fix the error above and retry." -ForegroundColor Red
    exit 1
}

# Download Florence-2 model for watermark detection
Write-Host ""
Write-Host "  [*] Downloading Florence-2 detection model (~1.5GB)..." -ForegroundColor Cyan
Write-Host ""
Write-Host "      Did you know?" -ForegroundColor DarkGray
Write-Host ""

if ($CHINA_MODE) {
    Write-Host "      Using HF-Mirror for faster download in China" -ForegroundColor DarkGray
}
$florenceProcess = Start-Process -FilePath $PYTHON_EXE -ArgumentList "-m", "model_assets", "--florence-only", "--endpoint", $HF_ENDPOINT -NoNewWindow -PassThru
$florenceHandle = $florenceProcess.Handle

$lastTipTime = Get-Date
while (-not $florenceProcess.HasExited) {
    $now = Get-Date
    if (($now - $lastTipTime).TotalSeconds -ge 5) {
        $tip = $tips[$currentTip]
        $line = "      $($tip.icon) $($tip.text)"
        $line = $line.PadRight(90)
        Write-Host "`r$line" -ForegroundColor $tip.color -NoNewline

        $currentTip = ($currentTip + 1) % $tips.Count
        $lastTipTime = $now
    }
    Start-Sleep -Milliseconds 300
}

Write-Host "`r                                                                                              "

$florenceProcess.WaitForExit()
if ($florenceProcess.ExitCode -ne 0) {
    Write-Host "  [!] Warning: Could not download Florence-2 model" -ForegroundColor Yellow
    Write-Host "      Open Models in the application and choose Download / Retry" -ForegroundColor Yellow
}
else {
    Write-Host "  [OK] Florence-2 model ready" -ForegroundColor Green
}

Write-Host ""
Write-Host "  =============================================" -ForegroundColor Green
Write-Host "     Setup complete! Ready to go!              " -ForegroundColor Green
Write-Host "  =============================================" -ForegroundColor Green
Write-Host ""
Write-Host "  To run the app: Double-click " -ForegroundColor Cyan -NoNewline
Write-Host "run.bat" -ForegroundColor White
Write-Host ""

$launch = Read-Host "  Launch now? (y/n)"
if ($launch -eq "y" -or $launch -eq "Y") {
    Write-Host ""
    Write-Host "  Starting WatermarkRemover-AI..." -ForegroundColor Green
    Start-Process -FilePath $PYTHON_EXE -ArgumentList "remwmgui.py" -NoNewWindow
}

Write-Host ""
Write-Host "  Have fun yeeting watermarks!" -ForegroundColor Magenta
Write-Host ""
Read-Host "  Press Enter to exit"
