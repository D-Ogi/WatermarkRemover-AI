// OpenWipe — Tauri Shell
// Manages the Python backend process and serves the web UI.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use tauri::Manager;
use std::process::{Command, Child};
use std::sync::Mutex;

struct PythonBackend {
    child: Mutex<Option<Child>>,
}

fn start_backend() -> Option<Child> {
    // Try to start the Python API backend
    let python = if cfg!(target_os = "windows") { "python" } else { "python3" };

    let child = Command::new(python)
        .arg("backend/api.py")
        .spawn()
        .ok();

    if child.is_some() {
        // Give the server a moment to start
        std::thread::sleep(std::time::Duration::from_millis(2000));
    }

    child
}

fn main() {
    // Start Python backend
    let backend_child = start_backend();

    tauri::Builder::default()
        .manage(PythonBackend {
            child: Mutex::new(backend_child),
        })
        .setup(|app| {
            // Navigate to the local API server
            if let Some(window) = app.get_webview_window("main") {
                let _ = window.eval("window.location.href = 'http://127.0.0.1:8765'");
            }
            Ok(())
        })
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::Destroyed = event {
                // Kill Python backend when window closes
                if let Some(state) = window.try_state::<PythonBackend>() {
                    if let Ok(mut child) = state.child.lock() {
                        if let Some(ref mut c) = *child {
                            let _ = c.kill();
                        }
                    }
                }
            }
        })
        .run(tauri::generate_context!())
        .expect("error while running OpenWipe");
}
