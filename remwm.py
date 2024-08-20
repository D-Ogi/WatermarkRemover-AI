import sys
import os
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QHBoxLayout,
    QFileDialog, QProgressBar, QComboBox, QLineEdit, QWidget
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QPixmap
import configparser
import cv2
import numpy as np
from PIL import Image, ImageDraw  # Added ImageDraw import
from transformers import AutoProcessor, AutoModelForCausalLM
from lama_cleaner.model_manager import ModelManager
from lama_cleaner.schema import Config, HDStrategy, LDMSampler
import torch
from utils import TaskType, run_example, draw_polygons, set_model_info  # Assuming these are from utils.py


class WatermarkRemoverApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.load_preferences()

    def initUI(self):
        self.setWindowTitle('Watermark Remover')
        self.setGeometry(100, 100, 1200, 600)  # Adjusted size to accommodate more images

        layout = QVBoxLayout()

        # Directory selectors
        self.input_dir_btn = QPushButton('Select Input Directory')
        self.output_dir_btn = QPushButton('Select Output Directory')
        self.input_dir_btn.clicked.connect(self.select_input_dir)
        self.output_dir_btn.clicked.connect(self.select_output_dir)
        layout.addWidget(self.input_dir_btn)
        layout.addWidget(self.output_dir_btn)

        # Image Previews
        preview_layout = QHBoxLayout()
        
        self.original_image_label = QLabel("Original Image")
        self.original_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        preview_layout.addWidget(self.original_image_label)

        self.cleaned_image_label = QLabel("Cleaned Image")
        self.cleaned_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        preview_layout.addWidget(self.cleaned_image_label)

        self.difference_image_label = QLabel("Difference")
        self.difference_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        preview_layout.addWidget(self.difference_image_label)

        layout.addLayout(preview_layout)

        # Progress Bar
        self.progress_bar = QProgressBar(self)
        layout.addWidget(self.progress_bar)

        # Settings
        settings_layout = QHBoxLayout()
        self.max_size_label = QLabel("Max Width/Height:")
        self.max_size_input = QLineEdit(self)
        self.max_size_input.setPlaceholderText("e.g., 1024")
        settings_layout.addWidget(self.max_size_label)
        settings_layout.addWidget(self.max_size_input)

        self.skip_existing_label = QLabel("If Exists:")
        self.skip_existing_combo = QComboBox(self)
        self.skip_existing_combo.addItems(["Skip", "Overwrite"])
        settings_layout.addWidget(self.skip_existing_label)
        settings_layout.addWidget(self.skip_existing_combo)
        
        layout.addLayout(settings_layout)

        # Start Button
        self.start_btn = QPushButton('Start Batch Processing')
        self.start_btn.clicked.connect(self.start_batch_processing)
        layout.addWidget(self.start_btn)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def select_input_dir(self):
        self.input_dir = QFileDialog.getExistingDirectory(self, 'Select Input Directory')
        if self.input_dir:
            self.input_dir_btn.setText(f"Input Directory: {self.input_dir}")

    def select_output_dir(self):
        self.output_dir = QFileDialog.getExistingDirectory(self, 'Select Output Directory')
        if self.output_dir:
            self.output_dir_btn.setText(f"Output Directory: {self.output_dir}")

    def start_batch_processing(self):
        # Save preferences as soon as the Start button is clicked
        self.save_preferences()

        self.worker = WatermarkRemoverThread(self.input_dir, self.output_dir, self.max_size_input.text(), self.skip_existing_combo.currentText())
        self.worker.progress_update.connect(self.progress_bar.setValue)
        self.worker.image_processed.connect(self.update_image_preview)
        self.worker.start()

    def update_image_preview(self, image_path):
        # Load the original image
        original_image = Image.open(image_path)
        original_pixmap = QPixmap(image_path)
        self.original_image_label.setPixmap(original_pixmap.scaled(self.original_image_label.size(), Qt.AspectRatioMode.KeepAspectRatio))

        # Load the cleaned image
        cleaned_pixmap = QPixmap(image_path)
        self.cleaned_image_label.setPixmap(cleaned_pixmap.scaled(self.cleaned_image_label.size(), Qt.AspectRatioMode.KeepAspectRatio))

        # Generate and display the difference image
        original_image_np = np.array(original_image)
        cleaned_image_np = cv2.imread(image_path)

        if cleaned_image_np is not None:
            cleaned_image_np = cv2.cvtColor(cleaned_image_np, cv2.COLOR_BGR2RGB)
            difference_image_np = cv2.absdiff(original_image_np, cleaned_image_np)
            difference_image = Image.fromarray(difference_image_np)
            difference_image_path = os.path.join(self.output_dir, "difference.png")
            difference_image.save(difference_image_path)

            difference_pixmap = QPixmap(difference_image_path)
            self.difference_image_label.setPixmap(difference_pixmap.scaled(self.difference_image_label.size(), Qt.AspectRatioMode.KeepAspectRatio))

    def load_preferences(self):
        config = configparser.ConfigParser()
        config.read('remwmconfig.ini')  # Updated to 'remwmconfig.ini'
        if 'Preferences' in config:
            self.input_dir = config['Preferences'].get('input_dir', '')
            self.output_dir = config['Preferences'].get('output_dir', '')
            self.max_size_input.setText(config['Preferences'].get('max_size', ''))
            self.skip_existing_combo.setCurrentText(config['Preferences'].get('skip_existing', 'Skip'))
            self.input_dir_btn.setText(f"Input Directory: {self.input_dir}")
            self.output_dir_btn.setText(f"Output Directory: {self.output_dir}")

    def save_preferences(self):
        config = configparser.ConfigParser()
        config['Preferences'] = {
            'input_dir': self.input_dir,
            'output_dir': self.output_dir,
            'max_size': self.max_size_input.text(),
            'skip_existing': self.skip_existing_combo.currentText()
        }
        with open('remwmconfig.ini', 'w') as configfile:
            config.write(configfile)

    def closeEvent(self, event):
        self.save_preferences()
        event.accept()


class WatermarkRemoverThread(QThread):
    progress_update = pyqtSignal(int)
    image_processed = pyqtSignal(str)

    def __init__(self, input_dir, output_dir, max_size, skip_existing):
        super().__init__()
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.max_size = int(max_size) if max_size.isdigit() else None
        self.skip_existing = skip_existing
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.florence_model = AutoModelForCausalLM.from_pretrained('microsoft/Florence-2-large', trust_remote_code=True).to(self.device).eval()
        self.florence_processor = AutoProcessor.from_pretrained('microsoft/Florence-2-large', trust_remote_code=True)
        self.model_manager = ModelManager(name="lama", device=self.device)

        # Set the model and processor in utils.py
        set_model_info(self.florence_model, self.florence_processor)

    def run(self):
        images = [os.path.join(self.input_dir, img) for img in os.listdir(self.input_dir) if img.endswith(('.png', '.jpg', '.jpeg'))]
        total_images = len(images)

        for i, image_path in enumerate(images):
            output_image_path = os.path.join(self.output_dir, os.path.basename(image_path))
            if os.path.exists(output_image_path) and self.skip_existing == "Skip":
                continue
            
            # Open image at full resolution
            image = Image.open(image_path).convert("RGB")
            
            # Only resize for processing if a max size is specified
            if self.max_size:
                image = self.resize_image(image, self.max_size)
            
            mask_image = self.get_watermark_mask(image)
            result_image = self.process_image_with_lama(np.array(image), np.array(mask_image))
            
            # Ensure the result_image conversion
            if result_image.dtype in [np.float64, np.float32]:
                result_image = np.clip(result_image, 0, 255)  # Ensure values are within [0, 255]
                result_image = result_image.astype(np.uint8)  # Convert to uint8

            # Convert from BGR to RGB
            result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)

            # Ensure the shape is correct (H, W, 3)
            if result_image.shape[-1] == 3:  # Check that it's a 3-channel image
                result_image_pil = Image.fromarray(result_image)
            else:
                raise ValueError(f"Unexpected result image shape: {result_image.shape}")

            result_image_pil.save(output_image_path)

            self.progress_update.emit(int((i + 1) / total_images * 100))
            self.image_processed.emit(output_image_path)

        self.progress_update.emit(100)

    def resize_image(self, image, max_size):
        """Resizes the image maintaining aspect ratio, only if it exceeds max_size."""
        if max(image.size) > max_size:
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        return image

    def get_watermark_mask(self, image):
        text_input = 'watermark'
        task_prompt = TaskType.OPEN_VOCAB_DETECTION  # Use OPEN_VOCAB_DETECTION
        parsed_answer = run_example(task_prompt, image, text_input)

        # Debugging: Print the parsed_answer to understand its structure
        print("Parsed Answer:", parsed_answer)

        # Get image dimensions
        image_width, image_height = image.size
        total_image_area = image_width * image_height

        # Create a mask based on bounding boxes since polygons are empty
        mask = Image.new("L", image.size, 0)  # "L" mode for single-channel grayscale
        draw = ImageDraw.Draw(mask)

        if 'bboxes' in parsed_answer['<OPEN_VOCABULARY_DETECTION>']:
            for bbox in parsed_answer['<OPEN_VOCABULARY_DETECTION>']['bboxes']:
                x1, y1, x2, y2 = map(int, bbox)  # Convert float bbox to int

                # Calculate the area of the bounding box
                bbox_area = (x2 - x1) * (y2 - y1)

                # If the area of the bounding box is less than 10% of the image area, include it in the mask
                if bbox_area <= 0.1 * total_image_area:
                    draw.rectangle([x1, y1, x2, y2], fill=255)  # Draw a white rectangle on the mask
                else:
                    print(f"Skipping region: Bounding box covers more than 10% of the image. BBox Area: {bbox_area}, Image Area: {total_image_area}")

        return mask



    def process_image_with_lama(self, image, mask):
        config = Config(
            ldm_steps=50,  # Increased steps for higher quality
            ldm_sampler=LDMSampler.ddim,
            hd_strategy=HDStrategy.CROP,  # Use CROP strategy for higher quality
            hd_strategy_crop_margin=64,  # Increase crop margin to provide more context
            hd_strategy_crop_trigger_size=800,  # Higher trigger size for larger images
            hd_strategy_resize_limit=1600,  # Increase limit for processing larger images
        )
        result = self.model_manager(image, mask, config)
        
        # Ensure result is in the correct format
        if result.dtype in [np.float64, np.float32]:
            result = np.clip(result, 0, 255)
            result = result.astype(np.uint8)
        
        return result



def main():
    app = QApplication(sys.argv)
    window = WatermarkRemoverApp()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
