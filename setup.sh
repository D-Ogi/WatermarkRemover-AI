#!/usr/bin/env bash
set -e
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

echo ""
echo "  ============================================="
echo "     WatermarkRemover-AI Setup (Linux/macOS)"
echo "  ============================================="
echo ""

# China mirror configuration
CHINA_MODE=0
PIP_MIRROR=""
HF_ENDPOINT=""

# Check if user is in China (for mirror selection)
echo "  [?] Are you in China? (y/n)"
echo "      This will use faster mirrors for downloads"
read -p "      " -n 1 -r china_choice
echo
if [[ $china_choice =~ ^[Yy]$ ]]; then
    CHINA_MODE=1
    PIP_MIRROR="-i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn"
    HF_ENDPOINT="https://hf-mirror.com"
    echo "  [OK] Using China mirrors (Tsinghua PyPI + HF-Mirror)"
else
    echo "  [OK] Using default mirrors"
fi
echo ""

# Detect OS
OS_TYPE="linux"
if [[ "$OSTYPE" == "darwin"* ]]; then
    OS_TYPE="macos"
    echo "  [*] Detected macOS"
else
    echo "  [*] Detected Linux"
fi

# Check Python version
PYTHON_CMD=""
for cmd in python3.12 python3.11 python3.10 python3 python; do
    if command -v $cmd &> /dev/null; then
        version=$($cmd -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null)
        major=$(echo $version | cut -d. -f1)
        minor=$(echo $version | cut -d. -f2)
        if [ "$major" -eq 3 ] && [ "$minor" -ge 10 ]; then
            PYTHON_CMD=$cmd
            echo "  [OK] Found $PYTHON_CMD (version $version)"
            break
        fi
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo "  [X] Python 3.10+ is required but not found."
    echo "      Please install Python 3.10 or higher."
    exit 1
fi

# Create virtual environment
VENV_DIR="venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "  [*] Creating virtual environment..."
    $PYTHON_CMD -m venv $VENV_DIR
    echo "  [OK] Virtual environment created"
else
    echo "  [OK] Virtual environment exists"
fi

# Activate venv
source $VENV_DIR/bin/activate

# Upgrade pip
echo "  [*] Upgrading pip..."
if [ "$CHINA_MODE" == "1" ]; then
    pip install --upgrade pip setuptools wheel $PIP_MIRROR -q
else
    pip install --upgrade pip setuptools wheel -q
fi

# Install PyTorch based on platform
echo "  [*] Installing PyTorch..."
if [ "$OS_TYPE" == "macos" ]; then
    # macOS: Install from main PyPI (supports MPS on Apple Silicon)
    if [ "$CHINA_MODE" == "1" ]; then
        pip install "torch>=2.14.0" "torchvision>=0.29.0" --no-cache-dir $PIP_MIRROR -q
    else
        pip install "torch>=2.14.0" "torchvision>=0.29.0" --no-cache-dir -q
    fi
    echo "  [OK] PyTorch installed (MPS support on Apple Silicon)"
else
    # Select one wheel index; mixing PyPI/mirrors can silently choose a CPU build.
    if command -v nvidia-smi &> /dev/null; then
        echo "  [*] NVIDIA GPU detected, installing CUDA version..."
        python -m pip --isolated install "torch>=2.14.0" "torchvision>=0.29.0" --index-url https://download.pytorch.org/whl/cu126 --no-cache-dir -q
        echo "  [OK] PyTorch installed (CUDA 12.6)"
    else
        echo "  [*] No NVIDIA GPU detected, installing CPU version..."
        python -m pip --isolated install "torch>=2.14.0" "torchvision>=0.29.0" --index-url https://download.pytorch.org/whl/cpu --no-cache-dir -q
        echo "  [OK] PyTorch installed (CPU)"
    fi
fi

# Use the same dependency manifest as the Windows installers and manual installs.
echo "  [*] Installing application dependencies..."
python -m pip install -r requirements.txt --no-cache-dir $PIP_MIRROR
python -m pip check
python -c "import remwm; import webview; import yaml; import psutil"
echo "  [OK] Dependencies installed and verified"

# Shared checksum-verified, atomic download; failure must not look like success.
echo "  [*] Preparing LaMA model (~196MB)..."
if [ "$CHINA_MODE" == "1" ]; then
    echo "      If GitHub is blocked, preseed the verified cache: docs/lama-runtime.md#restricted-networks"
fi
python -m lama_inpaint download

# Download Florence-2 model
echo "  [*] Downloading Florence-2 model (~1.5GB)..."
if [ "$CHINA_MODE" == "1" ]; then
    echo "      Using HF-Mirror for faster download in China"
    HF_ENDPOINT="$HF_ENDPOINT" python -c "import os; os.environ['HF_ENDPOINT']='$HF_ENDPOINT'; from huggingface_hub import snapshot_download; from model_assets import FLORENCE_REPO, FLORENCE_REVISION, MANIFEST, ensure_florence; snapshot_download(FLORENCE_REPO, revision=FLORENCE_REVISION, allow_patterns=[f['name'] for f in MANIFEST['files']]); ensure_florence(download=False)" || echo "  [!] Florence-2 download failed, open Models in the application and choose Download / Retry"
else
    python -c "from huggingface_hub import snapshot_download; from model_assets import FLORENCE_REPO, FLORENCE_REVISION, MANIFEST, ensure_florence; snapshot_download(FLORENCE_REPO, revision=FLORENCE_REVISION, allow_patterns=[f['name'] for f in MANIFEST['files']]); ensure_florence(download=False)" || echo "  [!] Florence-2 download failed, open Models in the application and choose Download / Retry"
fi

echo ""
echo "  ============================================="
echo "     Setup complete!"
echo "  ============================================="
echo ""
echo "  To run the app:"
echo "    source venv/bin/activate"
echo "    python remwmgui.py"
echo ""
echo "  Or for CLI:"
echo "    source venv/bin/activate"
echo "    python remwm.py input.png output/"
echo ""

# Ask to launch
read -p "  Launch now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "  Starting WatermarkRemover-AI..."
    python remwmgui.py
fi
