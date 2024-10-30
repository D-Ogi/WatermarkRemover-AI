import sys
import os
from PyQt6 import QtWidgets, QtCore
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton,
    QVBoxLayout, QHBoxLayout, QFileDialog, QProgressBar, QMessageBox, QComboBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

import remwm  # Ensure remwm.py is in the same directory or in the PYTHONPATH
from PIL import Image

class WorkerThread(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, input_dir, output_dir, overwrite_option):
        super().__init__()
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.overwrite_option = overwrite_option

    def run(self):
        try:
            images = []
            for root, dirs, files in os.walk(self.input_dir):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
                        images.append(os.path.join(root, file))

            total_images = len(images)
            if total_images == 0:
                self.error.emit("No images found in the input directory.")
                self.finished.emit()
                return

            # Initialize models outside the loop to avoid redundant loading
            device = 'cuda' if remwm.torch.cuda.is_available() else 'cpu'
            print(f"Using device: {device}")
            florence_model = remwm.AutoModelForCausalLM.from_pretrained(
                'microsoft/Florence-2-large', trust_remote_code=True
            ).to(device)
            florence_model.eval()
            florence_processor = remwm.AutoProcessor.from_pretrained('microsoft/Florence-2-large', trust_remote_code=True)

            model_manager = remwm.ModelManager(name="lama", device=device)

            for idx, image_path in enumerate(images):
                output_image_path = os.path.join(
                    self.output_dir, os.path.relpath(image_path, self.input_dir)
                )

                # Ensure output directory exists
                os.makedirs(os.path.dirname(output_image_path), exist_ok=True)

                # Check if output file exists
                if os.path.exists(output_image_path):
                    if self.overwrite_option == 'Skip':
                        continue  # Skip processing this image
                    elif self.overwrite_option == 'Overwrite':
                        pass  # Proceed to overwrite
                    else:
                        pass  # Default to overwrite

                try:
                    # Load the image
                    image = Image.open(image_path).convert("RGB")

                    # Get watermark mask
                    mask_image = remwm.get_watermark_mask(image, florence_model, florence_processor, device)

                    # Process image with LaMa
                    result_image = remwm.process_image_with_lama(
                        remwm.np.array(image),
                        remwm.np.array(mask_image),
                        model_manager
                    )

                    # Convert the result from BGR to RGB
                    result_image_rgb = remwm.cv2.cvtColor(result_image, remwm.cv2.COLOR_BGR2RGB)

                    # Convert result_image from NumPy array to PIL Image
                    result_image_pil = remwm.Image.fromarray(result_image_rgb)

                    # Save the result image
                    result_image_pil.save(output_image_path)
                except Exception as e:
                    self.error.emit(f"Error processing image {image_path}: {str(e)}")
                    continue  # Continue with the next image

                self.progress.emit(int((idx + 1) / total_images * 100))

            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))
            self.finished.emit()


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Batch Watermark Remover")
        self.setFixedSize(500, 200)

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Input Directory
        input_layout = QHBoxLayout()
        self.input_label = QLabel("Input Directory:")
        self.input_line_edit = QLineEdit()
        self.input_browse_button = QPushButton("Browse")
        self.input_browse_button.clicked.connect(self.browse_input_directory)
        input_layout.addWidget(self.input_label)
        input_layout.addWidget(self.input_line_edit)
        input_layout.addWidget(self.input_browse_button)

        # Output Directory
        output_layout = QHBoxLayout()
        self.output_label = QLabel("Output Directory:")
        self.output_line_edit = QLineEdit()
        self.output_browse_button = QPushButton("Browse")
        self.output_browse_button.clicked.connect(self.browse_output_directory)
        output_layout.addWidget(self.output_label)
        output_layout.addWidget(self.output_line_edit)
        output_layout.addWidget(self.output_browse_button)

        # Overwrite option
        overwrite_layout = QHBoxLayout()
        self.overwrite_label = QLabel("If output file exists:")
        self.overwrite_combo = QComboBox()
        self.overwrite_combo.addItems(['Overwrite', 'Skip'])
        overwrite_layout.addWidget(self.overwrite_label)
        overwrite_layout.addWidget(self.overwrite_combo)

        # Progress Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)

        # Start Button
        self.start_button = QPushButton("Start Processing")
        self.start_button.clicked.connect(self.start_processing)

        layout.addLayout(input_layout)
        layout.addLayout(output_layout)
        layout.addLayout(overwrite_layout)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.start_button)

        self.setLayout(layout)

    def browse_input_directory(self):
        dir = QFileDialog.getExistingDirectory(self, "Select Input Directory")
        if dir:
            self.input_line_edit.setText(dir)

    def browse_output_directory(self):
        dir = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if dir:
            self.output_line_edit.setText(dir)

    def start_processing(self):
        input_dir = self.input_line_edit.text()
        output_dir = self.output_line_edit.text()
        overwrite_option = self.overwrite_combo.currentText()

        if not os.path.isdir(input_dir):
            QMessageBox.critical(self, "Error", "Invalid input directory.")
            return

        if not os.path.isdir(output_dir):
            QMessageBox.critical(self, "Error", "Invalid output directory.")
            return

        self.start_button.setEnabled(False)

        self.thread = WorkerThread(input_dir, output_dir, overwrite_option)
        self.thread.progress.connect(self.update_progress)
        self.thread.finished.connect(self.processing_finished)
        self.thread.error.connect(self.show_error)
        self.thread.start()

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def processing_finished(self):
        self.start_button.setEnabled(True)
        QMessageBox.information(self, "Finished", "Processing completed.")

    def show_error(self, message):
        QMessageBox.critical(self, "Error", message)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
