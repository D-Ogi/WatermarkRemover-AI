# WatermarkRemover-AI

**AI-Powered Watermark Batch Remover using Florence-2 and LaMA Models**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)

![image](https://github.com/user-attachments/assets/df3203ed-057e-499b-86bc-f9e96be66c1e)

![image](https://github.com/user-attachments/assets/e89825fb-3b14-4358-96f3-feb526908ad3)

![image](https://github.com/user-attachments/assets/64e63d5c-4ecc-4fe0-954d-1b72e6a29580)


## Overview

`WatermarkRemover-AI` is a Python-based application that utilizes state-of-the-art AI models—Florence-2 from Microsoft for detecting watermarks and LaMA (Large Masked Autoregressive) for inpainting—to effectively remove watermarks from images. The application provides a user-friendly interface built with PyQt6, allowing for easy batch processing and previewing of original, cleaned, and difference images. It's effective tool for large datasets processing for AI image models learning and training.

## Table of Contents

- [Features](#features)
- [Technical Overview](#technical-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Batch Image Processing**: Select a directory of images and process them all at once.
- **Advanced Watermark Detection**: Utilizes Florence-2's open vocabulary detection to identify watermark regions accurately.
- **High-Quality Inpainting**: Employs the LaMA model for seamless inpainting, ensuring high-quality results.
- **Customizable Settings**: Configure overwrite behavior

## Technical Overview

### 1. **Florence-2 Model for Watermark Detection**
   - Florence-2, a model from Microsoft, is used for detecting watermarks in images. It leverages open vocabulary detection to identify regions that may contain watermarks.
   - Detected regions are filtered to ensure that only areas covering less than 10% of the image are processed, avoiding false positives.

### 2. **LaMA Model for Inpainting**
   - The LaMA (Large Masked Autoregressive) model is employed for inpainting the detected watermark regions. It provides high-quality, context-aware inpainting, making the watermark removal seamless.
   - The application uses different strategies (resize, crop) to handle images of various sizes, ensuring the best possible results.

### 3. **PyQt6-Based GUI**
   - The application features a PyQt6-based graphical user interface (GUI) that is intuitive and user-friendly. It allows users to select input/output directories, configure settings, and view the results in real-time.

## Installation

### Prerequisites

- Python 3.8+ (3.11 recommended)
- CUDA (for GPU acceleration)

### Steps

1. **Clone the Repository:**

   ```
   git clone https://github.com/yourusername/WatermarkRemover-AI.git
   cd WatermarkRemover-AI
   ```
   

2. **Install Dependencies:** 
   ```
   pip install -r requirements.txt
   pip install transformers>=4.44.0
   ```

4. Run the Application:

   Command-line
   
   ```python remwm.py source_directory output_directory```
   
   GUI:
   
    ```python remwmgui.py```
   

## Usage

### 1. **Selecting Directories**
   - Use the "Input Directory" button to choose the directory containing the images you want to process.
   - Use the "Output Directory" button to choose where the processed images will be saved.

### 2. **Configuring Settings (GUI only)**
   - **If output file exists**: Choose whether to skip or overwrite existing processed images.

### 3. **Processing Images**
   - Click "Start Processing" to begin processing all images in the input directory.
   - The progress bar will update as images are processed.

You may want to change the object detection prompt from `text_input = 'logo'` to 'watermark' or 'text' depending on your use case. 

## Contributing

Contributions are welcome! If you'd like to contribute, please fork this repository, create a feature branch, and submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
   
