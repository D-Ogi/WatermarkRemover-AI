# WatermarkRemover-AI Setup Script
$Host.UI.RawUI.WindowTitle = "WatermarkRemover-AI Setup"

$PYTHON_VERSION = "3.12.7"
$PYTHON_DIR = "python"
$PYTHON_EXE = "$PYTHON_DIR\python.exe"

# China mirror configuration
$CHINA_MODE = $false
$PIP_INDEX_URL = ""
$PIP_EXTRA_ARGS = @()
$HF_ENDPOINT = ""

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

# Install base deps with tips (legacy resolver to ignore conflicts)
if ($CHINA_MODE) {
    # For China: use Tsinghua mirror, skip PyTorch extra-index-url from requirements.txt
    $process = Start-Process -FilePath $PYTHON_EXE -ArgumentList "-m", "pip", "install", "--upgrade", "-r", "requirements.txt", "--no-cache-dir", "--use-deprecated=legacy-resolver", "-i", $PIP_INDEX_URL, "--trusted-host", "pypi.tuna.tsinghua.edu.cn" -NoNewWindow -PassThru
} else {
    $process = Start-Process -FilePath $PYTHON_EXE -ArgumentList "-m", "pip", "install", "--upgrade", "-r", "requirements.txt", "--no-cache-dir", "--use-deprecated=legacy-resolver" -NoNewWindow -PassThru
}

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

# Legacy resolver can return non-zero even on success, so verify key packages
$verifyResult = & $PYTHON_EXE -c "import torch; import transformers; import webview; import cv2; print('OK')" 2>&1
if ($verifyResult -ne "OK") {
    if ($process.ExitCode -ne 0) {
        Write-Host ""
        Write-Host "  [X] Failed to install dependencies" -ForegroundColor Red
        Read-Host "  Press Enter to exit"
        exit 1
    }
}

# Install iopaint separately without pulling its deps (we already have ours)
Write-Host "  [*] Installing iopaint (no deps)..." -ForegroundColor Cyan
if ($CHINA_MODE) {
    $iopaintProcess = Start-Process -FilePath $PYTHON_EXE -ArgumentList "-m", "pip", "install", "--upgrade", "iopaint", "--no-deps", "--no-cache-dir", "-i", $PIP_INDEX_URL, "--trusted-host", "pypi.tuna.tsinghua.edu.cn" -NoNewWindow -PassThru
} else {
    $iopaintProcess = Start-Process -FilePath $PYTHON_EXE -ArgumentList "-m", "pip", "install", "--upgrade", "iopaint", "--no-deps", "--no-cache-dir" -NoNewWindow -PassThru
}
$iopaintProcess.WaitForExit()

if ($iopaintProcess.ExitCode -ne 0) {
    Write-Host ""
    Write-Host "  [X] Failed to install iopaint" -ForegroundColor Red
    Read-Host "  Press Enter to exit"
    exit 1
}
Write-Host "  [OK] iopaint installed" -ForegroundColor Green

# Install iopaint's required dependencies manually (subset needed for LaMA inpainting)
Write-Host "  [*] Installing iopaint dependencies..." -ForegroundColor Cyan
if ($CHINA_MODE) {
    $iopaintDepsProcess = Start-Process -FilePath $PYTHON_EXE -ArgumentList "-m", "pip", "install", "pydantic", "typer", "einops", "omegaconf", "easydict", "yacs", "--no-cache-dir", "-i", $PIP_INDEX_URL, "--trusted-host", "pypi.tuna.tsinghua.edu.cn" -NoNewWindow -PassThru
} else {
    $iopaintDepsProcess = Start-Process -FilePath $PYTHON_EXE -ArgumentList "-m", "pip", "install", "pydantic", "typer", "einops", "omegaconf", "easydict", "yacs", "--no-cache-dir" -NoNewWindow -PassThru
}
$iopaintDepsProcess.WaitForExit()

if ($iopaintDepsProcess.ExitCode -ne 0) {
    Write-Host ""
    Write-Host "  [X] Failed to install iopaint dependencies" -ForegroundColor Red
    Read-Host "  Press Enter to exit"
    exit 1
}

# Verify iopaint dependencies are properly installed
Write-Host "  [*] Verifying iopaint dependencies..." -ForegroundColor Cyan
$verifyIopaint = & $PYTHON_EXE -c "import pydantic; import typer; import einops; import omegaconf; import easydict; import yacs; print('OK')" 2>&1
if ($verifyIopaint -ne "OK") {
    Write-Host ""
    Write-Host "  [X] iopaint dependencies verification failed" -ForegroundColor Red
    Write-Host "      Missing modules detected. Attempting reinstall..." -ForegroundColor Yellow

    # Try installing one by one to identify issues
    $deps = @("pydantic", "typer", "einops", "omegaconf", "easydict", "yacs")
    foreach ($dep in $deps) {
        if ($CHINA_MODE) {
            & $PYTHON_EXE -m pip install $dep --no-cache-dir -i $PIP_INDEX_URL --trusted-host pypi.tuna.tsinghua.edu.cn 2>&1 | Out-Null
        } else {
            & $PYTHON_EXE -m pip install $dep --no-cache-dir 2>&1 | Out-Null
        }
    }

    # Verify again
    $verifyAgain = & $PYTHON_EXE -c "import pydantic; import typer; import einops; import omegaconf; import easydict; import yacs; print('OK')" 2>&1
    if ($verifyAgain -ne "OK") {
        Write-Host "  [X] Could not install iopaint dependencies" -ForegroundColor Red
        Write-Host "      Please try running: pip install pydantic typer einops omegaconf easydict yacs" -ForegroundColor Yellow
        Read-Host "  Press Enter to exit"
        exit 1
    }
}

Write-Host "  [OK] Dependencies installed and verified" -ForegroundColor Green

# Download LaMA model directly from GitHub (avoids iopaint CLI dependency on fastapi)
Write-Host ""
Write-Host "  [*] Downloading AI model (196MB)..." -ForegroundColor Cyan
Write-Host ""
Write-Host "      Did you know?" -ForegroundColor DarkGray
Write-Host ""

$lamaDir = Join-Path $env:USERPROFILE ".cache\torch\hub\checkpoints"
$lamaFile = Join-Path $lamaDir "big-lama.pt"
$lamaUrl = "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt"

if (-not (Test-Path $lamaFile)) {
    # Create directory if needed
    if (-not (Test-Path $lamaDir)) {
        New-Item -ItemType Directory -Path $lamaDir -Force | Out-Null
    }

    try {
        # Show tips while downloading
        $job = Start-Job -ScriptBlock {
            param($url, $dest)
            Invoke-WebRequest -Uri $url -OutFile $dest -UseBasicParsing
        } -ArgumentList $lamaUrl, $lamaFile

        $lastTipTime = Get-Date
        while ($job.State -eq "Running") {
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

        $result = Receive-Job -Job $job
        Remove-Job -Job $job

        if (Test-Path $lamaFile) {
            Write-Host "  [OK] LaMA model ready" -ForegroundColor Green
        } else {
            Write-Host "  [!] Warning: Could not download LaMA model" -ForegroundColor Yellow
            Write-Host "      It will be downloaded on first use" -ForegroundColor Yellow
        }
    }
    catch {
        Write-Host "  [!] Warning: Could not download LaMA model" -ForegroundColor Yellow
        Write-Host "      It will be downloaded on first use" -ForegroundColor Yellow
    }
}
else {
    Write-Host "  [OK] LaMA model already exists" -ForegroundColor Green
}

# Download Florence-2 model for watermark detection
Write-Host ""
Write-Host "  [*] Downloading Florence-2 detection model (~1.5GB)..." -ForegroundColor Cyan
Write-Host ""
Write-Host "      Did you know?" -ForegroundColor DarkGray
Write-Host ""

if ($CHINA_MODE) {
    # Set HF_ENDPOINT environment variable for China mirror
    $env:HF_ENDPOINT = $HF_ENDPOINT
    Write-Host "      Using HF-Mirror for faster download in China" -ForegroundColor DarkGray
    $florenceScript = @"
import os
os.environ['HF_ENDPOINT'] = '$HF_ENDPOINT'
from huggingface_hub import snapshot_download
snapshot_download('florence-community/Florence-2-large', local_dir_use_symlinks=False)
print('FLORENCE_OK')
"@
} else {
    $florenceScript = @"
from huggingface_hub import snapshot_download
snapshot_download('florence-community/Florence-2-large', local_dir_use_symlinks=False)
print('FLORENCE_OK')
"@
}

$florenceProcess = Start-Process -FilePath $PYTHON_EXE -ArgumentList "-c", "`"$florenceScript`"" -NoNewWindow -PassThru

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

if ($florenceProcess.ExitCode -ne 0) {
    Write-Host "  [!] Warning: Could not download Florence-2 model" -ForegroundColor Yellow
    Write-Host "      It will be downloaded on first use" -ForegroundColor Yellow
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
