"""
WatermarkRemover-AI GUI - Ohio Edition
PyWebview frontend with brainrot HTML UI
"""

import logging

# Suppress noisy pywebview WebView2 COM warnings (thread safety noise, doesn't affect functionality)
class PyWebviewFilter(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        # Filter out WebView2 COM interface errors that spam the console
        if 'Error while processing window.native' in msg:
            return False
        if 'CoreWebView2 members can only be accessed' in msg:
            return False
        return True

logging.getLogger('pywebview').addFilter(PyWebviewFilter())

import webview
import threading
import subprocess
import sys
import os
import json
import yaml
import base64
from pathlib import Path

# Only psutil for system info (lightweight)
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


CONFIG_FILE = "ui.yml"


class Api:
    """Python API exposed to JavaScript frontend"""

    def __init__(self):
        self.window = None
        self.process = None
        self.is_running = False
        self.config = self._load_config()

    def set_window(self, window):
        """Set the webview window reference"""
        self.window = window

    def _load_config(self):
        """Load saved configuration from YAML file"""
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f) or {}
            except Exception:
                pass
        return {}

    def _save_config(self, config):
        """Save configuration to YAML file"""
        try:
            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False)
        except Exception as e:
            print(f"Failed to save config: {e}")

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
            webview.OPEN_DIALOG,
            file_types=file_types
        )
        return result[0] if result else None

    def browse_folder(self):
        """Open folder browser dialog"""
        if not self.window:
            return None

        result = self.window.create_file_dialog(webview.FOLDER_DIALOG)
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
                [sys.executable, '-c', 'import torch; print("CUDA:" + str(torch.cuda.is_available()) + ":" + (torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""))'],
                capture_output=True, text=True, timeout=10, creationflags=creationflags
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
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True, creationflags=creationflags)
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
            'fade_out': fade_out
        })

        # Build command
        cmd = [sys.executable, 'remwm.py', input_path, output_path]

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
        self.is_running = True
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

            env = os.environ.copy()
            env['PYTHONUNBUFFERED'] = '1'

            working_dir = os.path.dirname(os.path.abspath(__file__))
            script_path = os.path.join(working_dir, 'remwm.py')

            # Verify script exists
            if not os.path.exists(script_path):
                self._call_js(f'addLog("ERROR: remwm.py not found at {json.dumps(script_path)}", "text-error")')
                self._call_js('processingComplete()')
                return

            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
                cwd=working_dir
            )

            for line in iter(self.process.stdout.readline, ''):
                if not self.is_running:
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

            self.process.wait()
            self._call_js('processingComplete()')

        except Exception as e:
            import traceback
            error_msg = json.dumps(f"Error: {str(e)}")
            self._call_js(f'addLog({error_msg}, "text-error")')
            # Log full traceback for debugging
            tb = json.dumps(traceback.format_exc())
            self._call_js(f'addLog({tb}, "text-gray-500")')
            self._call_js('processingComplete()')

        finally:
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
        self.is_running = False

        if self.process:
            try:
                self.process.terminate()
                try:
                    self.process.wait(timeout=0.5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
            except Exception:
                pass

        return {'status': 'stopped'}

    def preview_detection(self, settings):
        """
        Preview watermark detection via CLI subprocess.
        Returns image with bounding boxes drawn as base64.
        """
        input_path = settings.get('input', '')
        detection_prompt = settings.get('detection_prompt', 'watermark')
        max_bbox = settings.get('max_bbox', 15)

        if not input_path:
            return {'error': 'No input path specified'}

        try:
            # Call CLI with --preview flag
            cmd = [
                sys.executable, 'remwm.py',
                input_path, '--preview',
                '--max-bbox-percent', str(int(max_bbox)),
                '--detection-prompt', detection_prompt
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
                cwd=os.path.dirname(os.path.abspath(__file__))
            )

            if result.returncode != 0:
                return {'error': result.stderr or 'Preview failed'}

            # Parse JSON output from CLI
            output = result.stdout.strip()
            # Find JSON in output (may have log lines before it)
            for line in output.split('\n'):
                if line.startswith('{'):
                    return json.loads(line)

            return {'error': 'No preview data returned'}

        except subprocess.TimeoutExpired:
            return {'error': 'Preview timed out'}
        except Exception as e:
            return {'error': str(e)}


def main():
    """Main entry point"""
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
        background_color='#050505'
    )

    api.set_window(window)
    webview.start()


if __name__ == '__main__':
    main()
