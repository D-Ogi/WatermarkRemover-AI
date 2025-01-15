#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status
set -e

# Default values
ENV_NAME="py312aiwatermark"
ENV_FILE="environment.yml"
INSTALL_DIR=""
FORCE_REINSTALL=false
USE_DEFAULT_DIR=true

# Function to detect previously used directory
find_existing_install_dir() {
    if [ -d "$PWD/conda_envs" ] && conda env list | grep -q "^${ENV_NAME}\\s"; then
        echo "$PWD/conda_envs"
    else
        echo ""
    fi
}

# Function to display usage
usage() {
    echo "Usage: $0 [options] -- [script arguments]"
    echo "Options:"
    echo "  --activate                Activate the Conda environment and provide instructions to deactivate it."
    echo "  --reinstall               Reinstall the Conda environment."
    echo "  --current-dir             Use the current directory as the root for the Conda environment."
    echo "  --help                    Show this help message."
    echo "Script Arguments:"
    echo "  Any arguments after -- will be passed directly to remwm.py."
    exit 1
}

# Parse arguments
ACTIVATE_ONLY=false
SCRIPT_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --activate)
            ACTIVATE_ONLY=true
            shift
            ;;
        --reinstall)
            FORCE_REINSTALL=true
            shift
            ;;
        --current-dir)
            USE_DEFAULT_DIR=false
            INSTALL_DIR="$PWD/conda_envs"
            shift
            ;;
        --help)
            usage
            ;;
        --)
            shift
            SCRIPT_ARGS=("$@")
            break
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Check if Conda is installed
if ! command -v conda &> /dev/null; then
    echo "Conda could not be found. Please install Conda or Miniconda and try again."
    exit 1
fi

# Determine installation directory
if [ "$USE_DEFAULT_DIR" = false ]; then
    echo "Using current directory as the root for Conda environments: $INSTALL_DIR"
    export CONDA_ENVS_PATH="$INSTALL_DIR"
    export CONDA_PKGS_DIRS="$INSTALL_DIR/pkgs"
else
    EXISTING_DIR=$(find_existing_install_dir)
    if [ -n "$EXISTING_DIR" ]; then
        echo "Detected existing installation in: $EXISTING_DIR"
        export CONDA_ENVS_PATH="$EXISTING_DIR"
        export CONDA_PKGS_DIRS="$EXISTING_DIR/pkgs"
    else
        echo "Using default Conda directory."
    fi
fi

# Check if the environment already exists or needs to be reinstalled
if conda env list | grep -q "^${ENV_NAME}\\s"; then
    if [ "$FORCE_REINSTALL" = true ]; then
        echo "Reinstalling environment '${ENV_NAME}'..."
        conda env remove -n "${ENV_NAME}"
    else
        echo "Environment '${ENV_NAME}' already exists. Activating it..."
        eval "$(conda shell.bash hook)"
        conda activate "${ENV_NAME}"
    fi
fi

if ! conda env list | grep -q "^${ENV_NAME}\\s"; then
    # Create the Conda environment
    echo "Creating Conda environment '${ENV_NAME}' from '${ENV_FILE}'..."
    conda env create -f "${ENV_FILE}" || {
        echo "Failed to create the Conda environment."
        exit 1
    }
    eval "$(conda shell.bash hook)"
    conda activate "${ENV_NAME}"
fi

if [ "$ACTIVATE_ONLY" = true ]; then
    echo "Environment '${ENV_NAME}' activated. To deactivate, run 'conda deactivate'."
    exit 0
fi

# Ensure required dependencies are installed
pip list | grep -q PyQt6 || pip install PyQt6
pip list | grep -q transformers || pip install transformers
pip list | grep -q iopaint || pip install iopaint
pip list | grep -q opencv-python-headless || pip install opencv-python-headless

# Run remwm.py with passed arguments
python remwm.py "${SCRIPT_ARGS[@]}"
