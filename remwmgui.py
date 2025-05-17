import os
import sys
import subprocess
import psutil
import yaml
import torch
import time
import threading
import shutil
from pathlib import Path
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QLabel, QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QTextEdit,
    QProgressBar, QComboBox, QMessageBox, QRadioButton, QButtonGroup, QSlider, QCheckBox, QStatusBar
)
from PyQt6.QtCore import Qt, pyqtSignal, QObject, QThread, QTimer
from PyQt6.QtGui import QPalette, QColor
from loguru import logger

CONFIG_FILE = "ui.yml"

class Worker(QObject):
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal()
    error_signal = pyqtSignal(str)

    def __init__(self, process):
        super().__init__()
        self.process = process
        self.running = True

    def run(self):
        try:
            # Créer un thread pour lire stderr
            error_thread = threading.Thread(target=self.read_stderr)
            error_thread.daemon = True
            error_thread.start()
            
            # Lire stdout avec un timeout pour éviter les blocages
            while self.running and self.process.poll() is None:
                line = self.process.stdout.readline()
                if line:
                    self.log_signal.emit(line)
                    if "overall_progress:" in line:
                        try:
                            progress = int(line.strip().split("overall_progress:")[1].strip())
                            self.progress_signal.emit(progress)
                        except (ValueError, IndexError) as e:
                            self.log_signal.emit(f"Erreur de parsing de la progression: {str(e)}")
                else:
                    # Petite pause pour éviter d'utiliser trop de CPU
                    time.sleep(0.1)
            
            # Vérifier si le processus s'est terminé normalement
            if self.process.returncode is not None and self.process.returncode != 0:
                self.error_signal.emit(f"Le processus s'est terminé avec le code d'erreur: {self.process.returncode}")
                
        except Exception as e:
            self.error_signal.emit(f"Erreur dans le worker: {str(e)}")
        finally:
            # S'assurer que les flux sont fermés
            try:
                self.process.stdout.close()
                self.process.stderr.close()
            except:
                pass
            self.finished_signal.emit()
    
    def read_stderr(self):
        """Lire stderr dans un thread séparé pour éviter les blocages"""
        try:
            for line in iter(self.process.stderr.readline, ""):
                if line:
                    self.log_signal.emit(f"ERREUR: {line}")
        except:
            pass

    def stop(self):
        self.running = False

class WatermarkRemoverGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Watermark Remover GUI")
        self.setGeometry(100, 100, 800, 600)

        # Initialize UI elements
        self.radio_single = QRadioButton("Process Single File")
        self.radio_batch = QRadioButton("Process Directory")
        self.radio_single.setChecked(True)
        self.mode_group = QButtonGroup()
        self.mode_group.addButton(self.radio_single)
        self.mode_group.addButton(self.radio_batch)

        self.input_path = QLineEdit(self)
        self.output_path = QLineEdit(self)
        self.overwrite_checkbox = QCheckBox("Overwrite Existing Files", self)
        self.transparent_checkbox = QCheckBox("Make Watermark Transparent", self)
        self.max_bbox_percent_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.max_bbox_percent_slider.setRange(1, 100)
        self.max_bbox_percent_slider.setValue(10)
        self.max_bbox_percent_label = QLabel(f"Max BBox Percent: 10%", self)
        self.max_bbox_percent_slider.valueChanged.connect(self.update_bbox_label)

        self.force_format_png = QRadioButton("PNG")
        self.force_format_webp = QRadioButton("WEBP")
        self.force_format_jpg = QRadioButton("JPG")
        self.force_format_mp4 = QRadioButton("MP4")
        self.force_format_avi = QRadioButton("AVI")
        self.force_format_none = QRadioButton("None")
        self.force_format_none.setChecked(True)
        self.force_format_group = QButtonGroup()
        self.force_format_group.addButton(self.force_format_png)
        self.force_format_group.addButton(self.force_format_webp)
        self.force_format_group.addButton(self.force_format_jpg)
        self.force_format_group.addButton(self.force_format_mp4)
        self.force_format_group.addButton(self.force_format_avi)
        self.force_format_group.addButton(self.force_format_none)

        self.progress_bar = QProgressBar(self)
        self.logs = QTextEdit(self)
        self.logs.setReadOnly(True)
        self.logs.setVisible(False)

        self.start_button = QPushButton("Start", self)
        self.stop_button = QPushButton("Stop", self)
        self.toggle_logs_button = QPushButton("Show Logs", self)
        self.toggle_logs_button.setCheckable(True)
        self.stop_button.setDisabled(True)

        # Status bar for system info
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_system_info)
        self.timer.start(1000)  # Update every second

        self.process = None
        self.thread = None
        self.worker = None

        # Layout
        layout = QVBoxLayout()

        # Mode selection
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(self.radio_single)
        mode_layout.addWidget(self.radio_batch)

        # Input and output paths
        path_layout = QVBoxLayout()
        path_layout.addWidget(QLabel("Input Path:"))
        path_layout.addWidget(self.input_path)
        path_layout.addWidget(QPushButton("Browse", clicked=self.browse_input))
        path_layout.addWidget(QLabel("Output Path:"))
        path_layout.addWidget(self.output_path)
        path_layout.addWidget(QPushButton("Browse", clicked=self.browse_output))

        # Options
        options_layout = QVBoxLayout()
        options_layout.addWidget(self.overwrite_checkbox)
        options_layout.addWidget(self.transparent_checkbox)

        bbox_layout = QVBoxLayout()
        bbox_layout.addWidget(self.max_bbox_percent_label)
        bbox_layout.addWidget(self.max_bbox_percent_slider)
        options_layout.addLayout(bbox_layout)

        force_format_layout = QHBoxLayout()
        force_format_layout.addWidget(QLabel("Force Format:"))
        force_format_layout.addWidget(self.force_format_png)
        force_format_layout.addWidget(self.force_format_webp)
        force_format_layout.addWidget(self.force_format_jpg)
        force_format_layout.addWidget(self.force_format_mp4)
        force_format_layout.addWidget(self.force_format_avi)
        force_format_layout.addWidget(self.force_format_none)
        options_layout.addLayout(force_format_layout)

        # Logs and progress
        progress_layout = QVBoxLayout()
        progress_layout.addWidget(QLabel("Progress:"))
        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.toggle_logs_button)
        progress_layout.addWidget(self.logs)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)

        # Final assembly
        layout.addLayout(mode_layout)
        layout.addLayout(path_layout)
        layout.addLayout(options_layout)
        layout.addLayout(progress_layout)
        layout.addLayout(button_layout)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Connect buttons
        self.start_button.clicked.connect(self.start_processing)
        self.stop_button.clicked.connect(self.stop_processing)
        self.toggle_logs_button.toggled.connect(self.toggle_logs)

        self.apply_dark_mode_if_needed()

        # Load configuration
        self.load_config()

    def update_bbox_label(self, value):
        self.max_bbox_percent_label.setText(f"Max BBox Percent: {value}%")

    def toggle_logs(self, checked):
        self.logs.setVisible(checked)
        self.toggle_logs_button.setText("Hide Logs" if checked else "Show Logs")

    def apply_dark_mode_if_needed(self):
        if QApplication.instance().styleHints().colorScheme() == Qt.ColorScheme.Dark:
            dark_palette = QPalette()
            dark_palette.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
            dark_palette.setColor(QPalette.ColorRole.WindowText, QColor(255, 255, 255))
            dark_palette.setColor(QPalette.ColorRole.Base, QColor(25, 25, 25))
            dark_palette.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
            dark_palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(255, 255, 255))
            dark_palette.setColor(QPalette.ColorRole.ToolTipText, QColor(255, 255, 255))
            dark_palette.setColor(QPalette.ColorRole.Text, QColor(255, 255, 255))
            dark_palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
            dark_palette.setColor(QPalette.ColorRole.ButtonText, QColor(255, 255, 255))
            dark_palette.setColor(QPalette.ColorRole.BrightText, QColor(255, 0, 0))
            dark_palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))

            dark_palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
            dark_palette.setColor(QPalette.ColorRole.HighlightedText, QColor(0, 0, 0))

            QApplication.instance().setPalette(dark_palette)

    def update_system_info(self):
        cuda_available = "CUDA: Available" if torch.cuda.is_available() else "CUDA: Not Available"
        ram = psutil.virtual_memory()
        ram_usage = ram.used // (1024 ** 2)
        ram_total = ram.total // (1024 ** 2)
        ram_percentage = ram.percent

        vram_status = "Not Available"
        if torch.cuda.is_available():
            gpu_info = torch.cuda.get_device_properties(0)
            vram_total = gpu_info.total_memory // (1024 ** 2)
            vram_used = vram_total - (torch.cuda.memory_reserved(0) // (1024 ** 2))
            vram_percentage = (vram_used / vram_total) * 100
            vram_status = f"VRAM: {vram_used} MB / {vram_total} MB ({vram_percentage:.2f}%)"

        status_text = (
            f"{cuda_available} | RAM: {ram_usage} MB / {ram_total} MB ({ram_percentage}%) | {vram_status} | CPU Load: {psutil.cpu_percent()}%"
        )
        self.status_bar.showMessage(status_text)

    def browse_input(self):
        if self.radio_single.isChecked():
            path, _ = QFileDialog.getOpenFileName(
                self, 
                "Select Input File", 
                "", 
                "All Supported Files (*.png *.jpg *.jpeg *.webp *.mp4 *.avi *.mov *.mkv *.flv *.wmv *.webm);;Images (*.png *.jpg *.jpeg *.webp);;Videos (*.mp4 *.avi *.mov *.mkv *.flv *.wmv *.webm)"
            )
        else:
            path = QFileDialog.getExistingDirectory(self, "Select Input Directory")
        if path:
            self.input_path.setText(path)
            # Update format options based on file type
            if path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm')):
                self.force_format_mp4.setEnabled(True)
                self.force_format_avi.setEnabled(True)
                if self.transparent_checkbox.isChecked():
                    self.transparent_checkbox.setChecked(False)
                    QMessageBox.information(self, "Info", "Transparency is not supported for videos. Disabled.")

    def browse_output(self):
        path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if path:
            self.output_path.setText(path)

    def start_processing(self):
        input_path = self.input_path.text()
        output_path = self.output_path.text()

        if not input_path or not output_path:
            QMessageBox.critical(self, "Error", "Input and Output paths are required.")
            return

        # Check if input is a video 
        is_video = input_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm'))
        
        # Check if transparency is enabled for video
        if is_video and self.transparent_checkbox.isChecked():
            QMessageBox.warning(self, "Warning", "Transparency is not supported for videos. Continuing with non-transparent processing.")
            self.transparent_checkbox.setChecked(False)
            
        # Vérifier si FFmpeg est disponible pour les vidéos
        if is_video:
            ffmpeg_available = self.check_ffmpeg_available()
            if not ffmpeg_available:
                response = QMessageBox.warning(
                    self, 
                    "FFmpeg non disponible", 
                    "FFmpeg n'est pas disponible sur votre système. Les vidéos traitées n'auront pas de son.\n\nVoulez-vous continuer quand même?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No
                )
                if response == QMessageBox.StandardButton.No:
                    return

        overwrite = "--overwrite" if self.overwrite_checkbox.isChecked() else ""
        transparent = "--transparent" if self.transparent_checkbox.isChecked() else ""
        max_bbox_percent = self.max_bbox_percent_slider.value()
        force_format = "None"
        if self.force_format_png.isChecked():
            force_format = "PNG"
        elif self.force_format_webp.isChecked():
            force_format = "WEBP"
        elif self.force_format_jpg.isChecked():
            force_format = "JPG"
        elif self.force_format_mp4.isChecked():
            force_format = "MP4"
        elif self.force_format_avi.isChecked():
            force_format = "AVI"

        force_format_option = f"--force-format={force_format}" if force_format != "None" else ""

        command = [
            "python", "remwm.py",
            input_path, output_path,
            overwrite, transparent,
            f"--max-bbox-percent={max_bbox_percent}",
            force_format_option
        ]
        command = [arg for arg in command if arg]  # Remove empty strings

        self.process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # Line buffered
            env={**os.environ, "PYTHONUNBUFFERED": "1"}
        )

        self.worker = Worker(self.process)
        self.worker.log_signal.connect(self.update_logs)
        self.worker.progress_signal.connect(self.update_progress_bar)
        self.worker.finished_signal.connect(self.reset_ui)
        self.worker.error_signal.connect(self.handle_error)

        self.thread = QThread()
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.thread.start()

        self.stop_button.setDisabled(False)
        self.start_button.setDisabled(True)

    def update_logs(self, line):
        self.logs.append(line)

    def update_progress_bar(self, progress):
        self.progress_bar.setValue(progress)

    def stop_processing(self):
        if self.process:
            try:
                if self.worker:
                    self.worker.stop()
                self.process.terminate()
                # Donner un peu de temps au processus pour se terminer proprement
                QTimer.singleShot(500, lambda: self.force_kill_if_needed())
            except Exception as e:
                logger.error(f"Erreur lors de l'arrêt du processus: {str(e)}")
                self.reset_ui()

    def force_kill_if_needed(self):
        """Force l'arrêt du processus s'il est encore en cours d'exécution"""
        if self.process and self.process.poll() is None:
            try:
                self.process.kill()
                self.process.wait(timeout=1)
            except:
                pass
        self.reset_ui()

    def reset_ui(self):
        self.stop_button.setDisabled(True)
        self.start_button.setDisabled(False)
        if self.thread and self.thread.isRunning():
            self.thread.quit()
            self.thread.wait()
        self.process = None
        self.thread = None
        self.worker = None

    def save_config(self):
        config = {
            "input_path": self.input_path.text(),
            "output_path": self.output_path.text(),
            "overwrite": self.overwrite_checkbox.isChecked(),
            "transparent": self.transparent_checkbox.isChecked(),
            "max_bbox_percent": self.max_bbox_percent_slider.value(),
            "force_format": "PNG" if self.force_format_png.isChecked() 
                      else "WEBP" if self.force_format_webp.isChecked() 
                      else "JPG" if self.force_format_jpg.isChecked()
                      else "MP4" if self.force_format_mp4.isChecked()
                      else "AVI" if self.force_format_avi.isChecked()
                      else "None",
            "mode": "single" if self.radio_single.isChecked() else "batch"
        }
        with open(CONFIG_FILE, "w") as f:
            yaml.dump(config, f)

    def load_config(self):
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, "r") as f:
                config = yaml.safe_load(f)
                self.input_path.setText(config.get("input_path", ""))
                self.output_path.setText(config.get("output_path", ""))
                self.overwrite_checkbox.setChecked(config.get("overwrite", False))
                self.transparent_checkbox.setChecked(config.get("transparent", False))
                self.max_bbox_percent_slider.setValue(config.get("max_bbox_percent", 10))
                force_format = config.get("force_format", "None")
                if force_format == "PNG":
                    self.force_format_png.setChecked(True)
                elif force_format == "WEBP":
                    self.force_format_webp.setChecked(True)
                elif force_format == "JPG":
                    self.force_format_jpg.setChecked(True)
                elif force_format == "MP4":
                    self.force_format_mp4.setChecked(True)
                elif force_format == "AVI":
                    self.force_format_avi.setChecked(True)
                else:
                    self.force_format_none.setChecked(True)
                mode = config.get("mode", "single")
                if mode == "single":
                    self.radio_single.setChecked(True)
                else:
                    self.radio_batch.setChecked(True)

    def handle_error(self, error_message):
        """Gère les erreurs signalées par le worker"""
        self.logs.append(f"<span style='color:red'>{error_message}</span>")
        # Rendre les logs visibles en cas d'erreur
        if not self.logs.isVisible():
            self.toggle_logs_button.setChecked(True)
            self.toggle_logs(True)
        QMessageBox.critical(self, "Erreur", f"Une erreur est survenue: {error_message}")

    def closeEvent(self, event):
        self.save_config()
        event.accept()

    def check_ffmpeg_available(self):
        """Vérifie si FFmpeg est disponible sur le système"""
        try:
            # Essayer d'exécuter ffmpeg -version pour vérifier s'il est installé
            subprocess.check_output(["ffmpeg", "-version"], stderr=subprocess.STDOUT)
            return True
        except (subprocess.SubprocessError, FileNotFoundError):
            return False

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = WatermarkRemoverGUI()
    gui.show()
    sys.exit(app.exec())