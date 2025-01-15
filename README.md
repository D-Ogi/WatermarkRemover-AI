# WatermarkRemover-AI

**AI-Powered Watermark Removal Tool using Florence-2 and LaMA Models**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

![image](https://github.com/user-attachments/assets/8f7fb600-695f-4dd7-958c-0cff516b5c7a)

Example of watermark removal with LaMa inpainting

![image](https://github.com/user-attachments/assets/e89825fb-3b14-4358-96f3-feb526908ad3)

![image](https://github.com/user-attachments/assets/64e63d5c-4ecc-4fe0-954d-1b72e6a29580)


## Overview

`WatermarkRemover-AI` is a cutting-edge application that leverages AI models for precise watermark detection and seamless removal. It uses Florence-2 from Microsoft for watermark identification and LaMA for inpainting to fill in the removed regions naturally. The software offers both a command-line interface (CLI) and a PyQt6-based graphical user interface (GUI), making it accessible to both casual and advanced users.

## Table of Contents

- [Features](#features)
- [Technical Overview](#technical-overview)
- [Installation](#installation)
- [Usage](#usage)
  - [Preferred Way: Setup Script](#preferred-way-setup-script)
  - [Manual Way](#manual-way)
  - [Using the GUI](#using-the-gui)
  - [Using the CLI](#using-the-cli)
- [Upgrade Notes](#upgrade-notes)
- [Alpha Masking](#alpha-masking)
- [Contributing](#contributing)
- [License](#license)

---

## Features

- **Dual Modes**: Process individual images or entire directories of images.
- **Advanced Watermark Detection**: Utilizes Florence-2's open-vocabulary detection for accurate watermark identification.
- **Seamless Inpainting**: Employs LaMA for high-quality, context-aware inpainting.
- **Customizable Output**:
  - Configure maximum bounding box size for watermark detection.
  - Set transparency for watermark regions.
  - Force specific output formats (PNG, WEBP, JPG).
- **Progress Tracking**: Real-time progress updates in both GUI and CLI modes.
- **Dark Mode Support**: GUI automatically adapts to system dark mode settings.
- **Efficient Resource Management**: Optimized for GPU acceleration using CUDA (optional).

---

## Technical Overview

### Florence-2 for Watermark Detection
- Florence-2 detects watermarks using open-vocabulary object detection.
- Bounding boxes are filtered to ensure that only small regions (configurable by the user) are processed.

### LaMA for Inpainting
- The LaMA model seamlessly fills in watermark regions with context-aware content.
- Supports high-resolution inpainting by using cropping and resizing strategies.

### PyQt6 GUI
- User-friendly interface for selecting input/output paths, configuring settings, and tracking progress.
- Dark mode and customization options enhance the user experience.

---

## Installation

### Prerequisites

- Conda/Miniconda installed.
- CUDA (optional for GPU acceleration; the application runs well on CPUs too).

### Steps

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/D-Ogi/WatermarkRemover-AI.git
   cd WatermarkRemover-AI
   ```

2. **Run the Setup Script:**
   ```bash
   bash setup.sh
   ```

   The `setup.sh` script automatically sets up the environment, installs dependencies, and launches the GUI application. It also provides convenient options for CLI usage.

3. **Fast-Track Options:**
   - **To Use the CLI Immediately**: After running `setup.sh`, you can use the CLI directly without activating the environment manually:
     ```bash
     ./setup.sh input_path output_path [options]
     ```
     Example:
     ```bash
     ./setup.sh ./input_images ./output_images --overwrite --transparent
     ```
   - **To Activate the Environment Without Starting the Application**: Use:
     ```bash
     conda activate py312aiwatermark
     ```

---

## Usage

### Preferred Way: Setup Script

1. **Run the Setup Script**:
   ```bash
   bash setup.sh
   ```
   - The GUI will launch automatically, and the environment will be ready for immediate CLI or GUI use.
   - For CLI use, run:
     ```bash
     ./setup.sh input_path output_path [options]
     ```
     Example:
     ```bash
     ./setup.sh ./input_images ./output_images --overwrite --transparent
     ```

### Manual Way

1. **Activate the Environment**:
   ```bash
   conda activate py312aiwatermark
   ```
2. **Launch GUI or CLI**:
   - **GUI**:
     ```bash
     python remwmgui.py
     ```
   - **CLI**:
     ```bash
     python remwm.py input_path output_path [options]
     ```

### Using the GUI

1. **Launch the GUI**:
   If not launched automatically, start it with:
   ```bash
   python remwmgui.py
   ```

2. **Configure Settings**:
   - **Mode**: Select "Process Single Image" or "Process Directory".
   - **Paths**: Browse and set the input/output directories.
   - **Options**:
     - Enable overwriting of existing files (directory processing only, single image processing always overwrites)
     - Enable transparency for watermark regions.
     - Adjust the maximum bounding box size for watermark detection.
   - **Output Format**: Choose between PNG, WEBP, JPG, or retain the original format.

3. **Start Processing**:
   - Click "Start" to begin processing.
   - Monitor progress and logs in the GUI.

### Using the CLI

1. **Basic Command**:
   ```bash
   python remwm.py input_path output_path
   ```

2. **Options**:
   - `--overwrite`: Overwrite existing files.
   - `--transparent`: Make watermark regions transparent instead of removing them.
   - `--max-bbox-percent`: Set the maximum bounding box size for watermark detection (default: 10%).
   - `--force-format`: Force output format (PNG, WEBP, or JPG).

3. **Example**:
   ```bash
   python remwm.py ./input_images ./output_images --overwrite --max-bbox-percent=15 --force-format=PNG
   ```
---

### Upgrade Notes

If you have previously used an older version of the repository or set up an incorrect Conda environment, follow these steps to upgrade:

1. **Update the Repository**:
   ```bash
   git pull
   ```

2. **Remove the Old Environment**:
   ```bash
   conda deactivate
   conda env remove -n py312
   ```

3. **Run the Setup Script**:
   ```bash
   bash setup.sh
   ```

This will recreate the correct environment (`py312aiwatermark`) and ensure all dependencies are up-to-date.


---

## Alpha Masking

We implemented alpha masking to allow selective manipulation of watermark regions without altering other parts of the image.

### Why Alpha Masking?
- **Precision**: Enable box-targeted watermark removal by isolating specific regions.
- **Flexibility**: By controlling opacity in alpha layers, we can achieve a range of effects by complete removal to transparency.
- **Minimal Impact**: This method ensures that areas outside the watermark remain untouched, preserving image quality.


---

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a new branch for your feature.
3. Submit a pull request detailing your changes.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


