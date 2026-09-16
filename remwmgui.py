"""
WatermarkRemover-AI GUI - Ohio Edition
PyWebview frontend with brainrot HTML UI
"""

import logging

import webview
import threading
import subprocess
import sys
import os
import json
import yaml
import base64
from pathlib import Path
from desktop_models import ModelPreparation
from desktop_runtime import APP_ROOT, configure_runtime, data_dir, python_executable, worker_options

# Only psutil for system info (lightweight)
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


CONFIG_FILE = str(data_dir() / "ui.yml")


class Api:
    """Python API exposed to JavaScript frontend"""

    def __init__(self):
        self.window = None
        self.process = None
        self._preview_process = None
        self._preview_running = False
        self._closed = False
        self.is_running = False
        self._process_lock = threading.Lock()
        self._stop_requested = threading.Event()
        self.config = self._load_config()
        self._models = ModelPreparation()
        self._models.start(check=True)

    def get_model_status(self):
        """Report verified readiness, byte progress, or an actionable download error."""
        return self._models.status()

    def download_models(self):
        """Download missing assets or retry a failed transfer using verified caches."""
        if self.is_running or self._preview_running:
            return {"error": "Wait until processing or preview finishes"}
        return self._models.start()

    def _close(self):
        """Reap only child processes created by this application window."""
        with self._process_lock:
            self._closed = True
        self.stop_processing()
        self._models.close()

    def set_window(self, window):
        """Set the webview window reference"""
        self.window = window

    def _load_config(self):
        """Load saved configuration from YAML file"""
        source = Path(CONFIG_FILE)
        if not source.exists():
            source = APP_ROOT / "ui.yml"
        if source.exists():
            try:
                with source.open('r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                    return config if isinstance(config, dict) else {}
            except Exception:
                pass
        return {}

    def _save_config(self, config):
        """Save configuration to YAML file"""
        try:
            Path(CONFIG_FILE).parent.mkdir(parents=True, exist_ok=True)
            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False)
        except Exception as e:
            print(f"Failed to save config: {e}")

    def debug_log(self, msg):
        """Print debug message from JavaScript"""
        print(f"[JS DEBUG] {msg}")

    def get_config(self):
        """Return saved configuration to frontend"""
        return self.config

    def save_config(self, config):
        """Save configuration from frontend"""
        self.config = config
        self._save_config(config)

    def browse_file(self):
        """Open file browser dialog"""
        if not self.window:
            return None

        file_types = (
            'All supported files (*.png;*.jpg;*.jpeg;*.webp;*.bmp;*.mp4;*.avi;*.mov;*.mkv;*.flv;*.wmv;*.webm)',
            'Images (*.png;*.jpg;*.jpeg;*.webp;*.bmp)',
            'Videos (*.mp4;*.avi;*.mov;*.mkv;*.flv;*.wmv;*.webm)',
            'All files (*.*)'
        )

        result = self.window.create_file_dialog(
            webview.FileDialog.OPEN,
            file_types=file_types
        )
        return result[0] if result else None

    def browse_folder(self):
        """Open folder browser dialog"""
        if not self.window:
            return None

        result = self.window.create_file_dialog(webview.FileDialog.FOLDER)
        return result[0] if result else None

    def _would_overwrite_input(self, input_path, output_path):
        """Check if output would overwrite the input file."""
        supported_ext = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm'}

        if os.path.isfile(input_path):
            # Single file mode
            output_ext = os.path.splitext(output_path)[1].lower()
            is_output_dir = os.path.isdir(output_path) or (output_ext == '' or output_ext not in supported_ext)

            if is_output_dir:
                output_file = os.path.join(output_path, os.path.basename(input_path))
            else:
                output_file = output_path
            # Compare resolved paths
            return os.path.normcase(os.path.abspath(input_path)) == os.path.normcase(os.path.abspath(output_file))
        else:
            # Directory mode - check if input and output folders are the same
            return os.path.normcase(os.path.abspath(input_path)) == os.path.normcase(os.path.abspath(output_path))

    def _check_file_conflicts(self, input_path, output_path):
        """Check if output files already exist. Returns list of conflicting filenames."""
        conflicts = []
        supported_ext = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm'}

        if os.path.isfile(input_path):
            # Single file mode
            input_name = os.path.basename(input_path)
            # Check if output_path is an existing directory OR looks like a directory path (no file extension)
            output_ext = os.path.splitext(output_path)[1].lower()
            is_output_dir = os.path.isdir(output_path) or (output_ext == '' or output_ext not in supported_ext)

            if is_output_dir:
                output_file = os.path.join(output_path, input_name)
            else:
                output_file = output_path

            # Check for file with same name OR alternate extension (jpg<->jpeg)
            files_to_check = [output_file]
            base, ext = os.path.splitext(output_file)
            if ext.lower() == '.jpg':
                files_to_check.append(base + '.jpeg')
            elif ext.lower() == '.jpeg':
                files_to_check.append(base + '.jpg')

            for check_file in files_to_check:
                if os.path.exists(check_file):
                    conflicts.append(os.path.basename(check_file))
                    break
        else:
            # Directory/batch mode
            if os.path.isdir(input_path):
                for fname in os.listdir(input_path):
                    ext = os.path.splitext(fname)[1].lower()
                    if ext in supported_ext:
                        output_file = os.path.join(output_path, fname)
                        if os.path.exists(output_file):
                            conflicts.append(fname)

        return conflicts

    def get_static_info(self):
        """Get static system info (CUDA, FFmpeg, GPU) - call once on startup"""
        info = {
            'cuda': False,
            'gpu_name': None,
            'ffmpeg': False
        }

        # Windows: hide console windows for subprocesses
        creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0

        # Check CUDA via subprocess (avoid importing torch in GUI)
        try:
            result = subprocess.run(
                [python_executable(), '-c', 'import torch; print("CUDA:" + str(torch.cuda.is_available()) + ":" + (torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""))'],
                capture_output=True, text=True, timeout=30, creationflags=creationflags
            )
            if result.returncode == 0 and 'CUDA:' in result.stdout:
                parts = result.stdout.strip().split(':')
                info['cuda'] = parts[1] == 'True'
                if len(parts) > 2 and parts[2]:
                    info['gpu_name'] = parts[2]
        except Exception:
            pass

        # Check FFmpeg
        try:
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True, timeout=5, creationflags=creationflags)
            info['ffmpeg'] = True
        except (subprocess.SubprocessError, FileNotFoundError):
            info['ffmpeg'] = False

        return info

    def get_dynamic_info(self):
        """Get dynamic system info (RAM, CPU) - call periodically"""
        info = {
            'ram_percent': 0,
            'cpu_percent': 0
        }

        if PSUTIL_AVAILABLE:
            try:
                info['ram_percent'] = psutil.virtual_memory().percent
                info['cpu_percent'] = psutil.cpu_percent()
            except Exception:
                pass

        return info

    def start_processing(self, settings):
        """Start watermark removal processing"""
        if self.is_running:
            return {'error': 'Already running'}

        if self._models.status().get('status') != 'ready':
            return {'error': 'Prepare the AI models before processing. Open Models and choose Download / Retry.'}
        input_path = settings.get('input', '')
        output_path = settings.get('output', '')

        if not input_path:
            return {'error': 'No input path specified'}

        # Use input directory as output if not specified
        if not output_path:
            if os.path.isfile(input_path):
                output_path = os.path.dirname(input_path)
            else:
                output_path = input_path

        # SAFETY: Check if output would overwrite input
        overwrite = settings.get('overwrite', False)
        would_overwrite_input = self._would_overwrite_input(input_path, output_path)
        if would_overwrite_input:
            return {'error': 'Cannot overwrite input file! Choose a different output folder.'}

        # Check for file conflicts if overwrite is not enabled
        if not overwrite:
            conflicts = self._check_file_conflicts(input_path, output_path)
            if conflicts:
                conflict_list = ', '.join(conflicts[:3])
                more = f" (+{len(conflicts)-3} more)" if len(conflicts) > 3 else ""
                error_msg = f'Output files already exist: {conflict_list}{more}. Enable "Overwrite" or choose different output folder.'
                return {'error': error_msg}

        # Get settings
        detection_prompt = settings.get('detection_prompt', 'watermark')
        detection_skip = settings.get('detection_skip', 1)
        fade_in = settings.get('fade_in', 0)
        fade_out = settings.get('fade_out', 0)

        # Save config
        self.save_config({
            'input_path': input_path,
            'output_path': output_path,
            'overwrite': settings.get('overwrite', False),
            'transparent': settings.get('transparent', False),
            'max_bbox_percent': settings.get('max_bbox', 15),
            'force_format': settings.get('format', 'None'),
            'mode': settings.get('mode', 'single'),
            'detection_prompt': detection_prompt,
            'detection_skip': detection_skip,
            'fade_in': fade_in,
            'fade_out': fade_out,
            'theme': settings.get('theme', 'brainrot'),
            'lang': settings.get('lang', 'brainrot')
        })

        # Build command
        cmd = [python_executable(), str(APP_ROOT / 'remwm.py'), input_path, output_path]

        if settings.get('overwrite'):
            cmd.append('--overwrite')

        if settings.get('transparent'):
            cmd.append('--transparent')

        max_bbox = settings.get('max_bbox', 15)
        cmd.append(f'--max-bbox-percent={int(max_bbox)}')

        format_opt = settings.get('format', 'None')
        if format_opt and format_opt != 'None':
            cmd.append(f'--force-format={format_opt}')

        if detection_prompt and detection_prompt != 'watermark':
            cmd.append(f'--detection-prompt={detection_prompt}')

        if detection_skip and int(detection_skip) > 1:
            cmd.append(f'--detection-skip={int(detection_skip)}')

        if fade_in and float(fade_in) > 0:
            cmd.append(f'--fade-in={float(fade_in)}')

        if fade_out and float(fade_out) > 0:
            cmd.append(f'--fade-out={float(fade_out)}')

        # Start processing in background thread
        with self._process_lock:
            if self._closed:
                return {"error": "Window is closing"}
            if self.is_running or self._preview_running:
                return {"error": "Processing or preview is already running"}
            self.is_running = True
            self._stop_requested.clear()
        threading.Thread(target=self._run_process, args=(cmd,), daemon=True).start()
        return {'status': 'started'}

    def _run_process(self, cmd):
        """Run the subprocess and stream output to frontend"""
        try:
            # Log the CLI command for educational purposes
            cli_display = ' '.join(cmd[1:])  # Skip python executable
            cli_display = cli_display.replace('remwm.py ', 'python remwm.py \\\n    ')
            cli_display = cli_display.replace(' --', ' \\\n    --')
            self._call_js(f'addLog("$ {json.dumps(cli_display)[1:-1]}", "text-neon-cyan")')


            working_dir = os.path.dirname(os.path.abspath(__file__))
            script_path = os.path.join(working_dir, 'remwm.py')

            # Verify script exists
            if not os.path.exists(script_path):
                self._call_js(f'addLog({json.dumps("ERROR: remwm.py not found at " + script_path)}, "text-error")')
                self._call_js('processingComplete({success: false})')
                return

            with self._process_lock:
                if self._stop_requested.is_set():
                    self._call_js('processingComplete({success: false, cancelled: true})')
                    return
                self.process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    bufsize=1,
                    **worker_options(offline=True)
                )

            for line in iter(self.process.stdout.readline, ''):
                if self._stop_requested.is_set():
                    break

                line = line.strip()
                if not line:
                    continue

                # Parse progress
                if 'overall_progress:' in line:
                    try:
                        progress_str = line.split('overall_progress:')[1].strip()
                        progress = int(progress_str.replace('%', ''))
                        self._call_js(f'updateProgress({progress})')
                    except (ValueError, IndexError):
                        pass

                # Send log line to frontend
                escaped = json.dumps(line)

                if 'error' in line.lower() or 'failed' in line.lower():
                    color = 'text-error'
                elif 'warning' in line.lower():
                    color = 'text-yellow-400'
                elif 'success' in line.lower() or 'done' in line.lower() or 'saved' in line.lower():
                    color = 'text-neon-green'
                else:
                    color = 'text-gray-400'

                self._call_js(f'addLog({escaped}, "{color}")')

            code = self.process.wait()
            self._call_js(f'processingComplete({json.dumps({"success": code == 0 and not self._stop_requested.is_set(), "cancelled": self._stop_requested.is_set(), "exit_code": code})})')

        except Exception as e:
            import traceback
            error_msg = json.dumps(f"Error: {str(e)}")
            self._call_js(f'addLog({error_msg}, "text-error")')
            # Log full traceback for debugging
            tb = json.dumps(traceback.format_exc())
            self._call_js(f'addLog({tb}, "text-gray-500")')
            self._call_js('processingComplete({success: false})')

        finally:
            with self._process_lock:
                if self.process and self.process.stdout:
                    self.process.stdout.close()
                self.is_running = False
                self.process = None

    def _call_js(self, js_code):
        """Safely call JavaScript in the frontend"""
        if self.window:
            try:
                self.window.evaluate_js(js_code)
            except Exception:
                pass

    def stop_processing(self):
        """Stop the current processing"""
        self._stop_requested.set()
        with self._process_lock:
            processes = (self.process, self._preview_process)

        for process in processes:
            if process is None:
                continue
            try:
                process.terminate()
                try:
                    process.wait(timeout=0.5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=3)
            except Exception:
                pass

        return {'status': 'stopped'}

    def preview_detection(self, settings):
        """Run one owned preview worker and return detections or a visible error."""
        if self._models.status().get('status') != 'ready':
            return {'error': 'Prepare the AI models before processing. Open Models and choose Download / Retry.'}
        input_path = settings.get('input', '')
        if not input_path:
            return {'error': 'No input path specified'}
        with self._process_lock:
            if self._closed:
                return {'error': 'Window is closing'}
            if self.is_running or self._preview_running:
                return {'error': 'Processing or preview is already running'}
            self._preview_running = True
        process = None
        try:
            cmd = [python_executable(), str(APP_ROOT / 'remwm.py'), input_path, '--preview',
                   '--max-bbox-percent', str(int(settings.get('max_bbox', 15))),
                   '--detection-prompt', settings.get('detection_prompt', 'watermark')]
            with self._process_lock:
                if self._closed:
                    return {'error': 'Window is closing'}
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                           **worker_options(offline=True))
                self._preview_process = process
            output, errors = process.communicate(timeout=120)
            if process.returncode != 0:
                return {'error': errors or 'Preview failed or was cancelled'}
            for line in output.strip().splitlines():
                if line.startswith('{'):
                    return json.loads(line)
            return {'error': 'No preview data returned'}
        except subprocess.TimeoutExpired:
            return {'error': 'Preview timed out'}
        except Exception as error:
            return {'error': str(error)}
        finally:
            if process is not None:
                if process.poll() is None:
                    process.kill()
                process.communicate()
            with self._process_lock:
                self._preview_process = None
                self._preview_running = False


def main(startup=None, *, hidden=False):
    """Main entry point"""
    configure_runtime()
    api = Api()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    ui_path = os.path.join(script_dir, 'ui', 'index.html')

    window = webview.create_window(
        'WatermarkRemover AI - Ohio Edition',
        ui_path,
        js_api=api,
        width=950,
        height=860,
        min_size=(800, 600),
        background_color='#050505',
        hidden=hidden
    )

    api.set_window(window)
    window.events.closed += api._close
    webview.start(startup, args=(window, api) if startup else None,
                  http_server=True, gui="qt" if sys.platform == "win32" else None)


if __name__ == '__main__':
    main()
