#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status
set -e

# Default values
ENV_NAME="py310"
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

# Create the Conda environment
echo "Creating Conda environment '${ENV_NAME}' from '${ENV_FILE}'..."
conda env create -f "${ENV_FILE}" || {
    echo "Failed to create the Conda environment. If it already exists, try activating it with 'conda activate ${ENV_NAME}'."
    exit 1
}

# Activate the Conda environment
echo "Activating Conda environment '${ENV_NAME}'..."
eval "$(conda shell.bash hook)"  # Ensure Conda works in non-login shells
conda activate "${ENV_NAME}"

# Optionally install GUI dependencies
if [ "${INSTALL_GUI}" = true ]; then
    echo "Installing GUI dependencies..."
    conda install -y pyqt || {
        echo "Failed to install GUI dependencies."
        exit 1
    }
else
    echo "GUI dependencies were skipped as per user request."
fi

# Verify installation
echo "Conda environment '${ENV_NAME}' is ready. Use 'conda activate ${ENV_NAME}' to activate it."
if [ "${INSTALL_GUI}" = true ]; then
    echo "GUI dependencies are installed."
else
    echo "GUI dependencies are not installed."
fi
