#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status
set -e

# Default values
ENV_NAME="py312"
ENV_FILE="environment.yml"
INSTALL_GUI=true  # Default is with GUI

# Check if the user wants to skip GUI dependencies
if [[ "$1" == "--no-gui" ]]; then
    INSTALL_GUI=false
    echo "Skipping GUI dependencies..."
fi

# Check if Conda is installed
if ! command -v conda &> /dev/null; then
    echo "Conda could not be found. Please install Conda or Miniconda and try again."
    exit 1
fi

# Check if the environment already exists
if conda env list | grep -q "^${ENV_NAME}"; then
    echo "Environment '${ENV_NAME}' already exists. Activating it..."
    eval "$(conda shell.bash hook)"
    conda activate "${ENV_NAME}"
else
    # Create the Conda environment
    echo "Creating Conda environment '${ENV_NAME}' from '${ENV_FILE}'..."
    conda env create -f "${ENV_FILE}" || {
        echo "Failed to create the Conda environment."
        exit 1
    }
    eval "$(conda shell.bash hook)"
    conda activate "${ENV_NAME}"
fi

# Optionally install GUI dependencies
if [ "${INSTALL_GUI}" = true ]; then
    echo "Installing GUI dependencies..."
    pip install PyQt6 || {
        echo "Failed to install PyQt6 GUI dependencies."
        exit 1
    }
else
    echo "GUI dependencies were skipped as per user request."
fi

# Install IOPaint and transformers manually using pip
echo "Installing iopaint and transformers manually via pip..."
pip install iopaint transformers opencv-python-headless || {
    echo "Failed to install iopaint or transformers."
    exit 1
}

# Verify installation
echo "Verifying environment and dependencies..."
python -c "import torch, iopaint; print('Torch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('iopaint installed.')"

# Start the appropriate script
if [ "${INSTALL_GUI}" = true ]; then
    echo "Starting GUI mode with 'remwmgui.py'..."
    python remwmgui.py
else
    echo "Starting CLI mode with 'remwm.py'..."
    python remwm.py
fi

echo "Setup and execution complete!"
