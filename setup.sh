#!/usr/bin/env bash
set -e

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
        pip install torch>=2.4.0 torchvision>=0.19.0 --no-cache-dir $PIP_MIRROR -q
    else
        pip install torch>=2.4.0 torchvision>=0.19.0 --no-cache-dir -q
    fi
    echo "  [OK] PyTorch installed (MPS support on Apple Silicon)"
else
    # Linux: Try CUDA first, fallback to CPU
    if command -v nvidia-smi &> /dev/null; then
        echo "  [*] NVIDIA GPU detected, installing CUDA version..."
        if [ "$CHINA_MODE" == "1" ]; then
            pip install torch>=2.4.0 torchvision>=0.19.0 --extra-index-url https://download.pytorch.org/whl/cu124 --no-cache-dir $PIP_MIRROR -q
        else
            pip install torch>=2.4.0 torchvision>=0.19.0 --extra-index-url https://download.pytorch.org/whl/cu124 --no-cache-dir -q
        fi
        echo "  [OK] PyTorch installed (CUDA 12.4)"
    else
        echo "  [*] No NVIDIA GPU detected, installing CPU version..."
        if [ "$CHINA_MODE" == "1" ]; then
            pip install torch>=2.4.0 torchvision>=0.19.0 --no-cache-dir $PIP_MIRROR -q
        else
            pip install torch>=2.4.0 torchvision>=0.19.0 --no-cache-dir -q
        fi
        echo "  [OK] PyTorch installed (CPU)"
    fi
fi

# Install other dependencies (without torch lines)
echo "  [*] Installing other dependencies..."
if [ "$CHINA_MODE" == "1" ]; then
    pip install transformers>=4.50.0 diffusers>=0.30.0 "numpy<2" --no-cache-dir $PIP_MIRROR -q
    pip install "opencv-python-headless>=4.8.0,<4.12.0" "Pillow>=10.0.0" --no-cache-dir $PIP_MIRROR -q
    pip install pywebview>=4.0 --no-cache-dir $PIP_MIRROR -q
    pip install loguru click tqdm psutil pyyaml --no-cache-dir $PIP_MIRROR -q
else
    pip install transformers>=4.50.0 diffusers>=0.30.0 "numpy<2" --no-cache-dir -q
    pip install "opencv-python-headless>=4.8.0,<4.12.0" "Pillow>=10.0.0" --no-cache-dir -q
    pip install pywebview>=4.0 --no-cache-dir -q
    pip install loguru click tqdm psutil pyyaml --no-cache-dir -q
fi

# Install iopaint separately (no deps to avoid conflicts)
echo "  [*] Installing iopaint..."
if [ "$CHINA_MODE" == "1" ]; then
    pip install iopaint --no-deps --no-cache-dir $PIP_MIRROR -q
else
    pip install iopaint --no-deps --no-cache-dir -q
fi

# Install iopaint's required dependencies manually (subset needed for LaMA inpainting)
echo "  [*] Installing iopaint dependencies..."
if [ "$CHINA_MODE" == "1" ]; then
    pip install pydantic typer einops omegaconf easydict yacs --no-cache-dir $PIP_MIRROR -q
else
    pip install pydantic typer einops omegaconf easydict yacs --no-cache-dir -q
fi
echo "  [OK] Dependencies installed"

# Download LaMA model directly from GitHub (avoids iopaint CLI dependency on fastapi)
echo "  [*] Downloading LaMA model (~196MB)..."
LAMA_DIR="$HOME/.cache/torch/hub/checkpoints"
LAMA_FILE="$LAMA_DIR/big-lama.pt"
if [ ! -f "$LAMA_FILE" ]; then
    mkdir -p "$LAMA_DIR"
    curl -L -o "$LAMA_FILE" "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt" || echo "  [!] LaMA download failed, will retry on first use"
    echo "  [OK] LaMA model downloaded"
else
    echo "  [OK] LaMA model already exists"
fi

# Download Florence-2 model
echo "  [*] Downloading Florence-2 model (~1.5GB)..."
if [ "$CHINA_MODE" == "1" ]; then
    echo "      Using HF-Mirror for faster download in China"
    HF_ENDPOINT="$HF_ENDPOINT" python -c "import os; os.environ['HF_ENDPOINT']='$HF_ENDPOINT'; from huggingface_hub import snapshot_download; snapshot_download('florence-community/Florence-2-large', local_dir_use_symlinks=False)" || echo "  [!] Florence-2 download failed, will retry on first use"
else
    python -c "from huggingface_hub import snapshot_download; snapshot_download('florence-community/Florence-2-large', local_dir_use_symlinks=False)" || echo "  [!] Florence-2 download failed, will retry on first use"
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
