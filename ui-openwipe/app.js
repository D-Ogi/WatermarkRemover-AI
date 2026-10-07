// OpenWipe — Frontend App
const API = 'http://127.0.0.1:8765/api';

let currentMode = 'remove';
let sourceImage = null;
let resultImage = null;
let maskCtx = null;
let imageCtx = null;
let isDrawing = false;
let brushSize = 30;

// === Initialization ===
window.addEventListener('DOMContentLoaded', () => {
    checkHealth();
    setupDragDrop();
    setupBrush();
    setupSliders();
});

async function checkHealth() {
    try {
        const res = await fetch(`${API}/health`);
        const data = await res.json();
        document.getElementById('device-badge').textContent =
            data.device === 'cuda' ? 'GPU' : 'CPU';
    } catch {
        document.getElementById('device-badge').textContent = 'offline';
    }
}

// === Mode ===
function setMode(mode) {
    currentMode = mode;
    document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
    document.querySelector(`[data-mode="${mode}"]`).classList.add('active');
    document.getElementById('brush-settings').style.display = mode === 'manual' ? 'block' : 'none';
    const maskCanvas = document.getElementById('mask-canvas');
    maskCanvas.style.pointerEvents = mode === 'manual' ? 'auto' : 'none';
}

// === Drag & Drop ===
function setupDragDrop() {
    const dz = document.getElementById('drop-zone');
    const fi = document.getElementById('file-input');

    dz.addEventListener('dragover', e => { e.preventDefault(); dz.classList.add('drag-over'); });
    dz.addEventListener('dragleave', () => dz.classList.remove('drag-over'));
    dz.addEventListener('drop', e => {
        e.preventDefault();
        dz.classList.remove('drag-over');
        if (e.dataTransfer.files.length) loadImage(e.dataTransfer.files[0]);
    });
    fi.addEventListener('change', e => { if (e.target.files.length) loadImage(e.target.files[0]); });
}

function loadImage(file) {
    const img = new Image();
    img.onload = () => {
        sourceImage = img;
        const c = document.getElementById('image-canvas');
        const m = document.getElementById('mask-canvas');
        c.width = m.width = img.width;
        c.height = m.height = img.height;
        imageCtx = c.getContext('2d');
        maskCtx = m.getContext('2d');
        imageCtx.drawImage(img, 0, 0);
        maskCtx.clearRect(0, 0, m.width, m.height);

        document.getElementById('drop-zone').style.display = 'none';
        document.getElementById('editor').style.display = 'flex';
        document.getElementById('action-bar').style.display = 'block';
        document.getElementById('image-info').textContent = `${img.width} × ${img.height}`;
        document.getElementById('save-btn').style.display = 'none';
    };
    img.src = URL.createObjectURL(file);
}

// === Brush ===
function setupBrush() {
    const m = document.getElementById('mask-canvas');

    m.addEventListener('mousedown', e => {
        if (currentMode !== 'manual') return;
        isDrawing = true;
        drawMask(e);
    });
    m.addEventListener('mousemove', e => { if (isDrawing) drawMask(e); });
    m.addEventListener('mouseup', () => isDrawing = false);
    m.addEventListener('mouseleave', () => isDrawing = false);

    const bs = document.getElementById('brush-size');
    bs.addEventListener('input', e => {
        brushSize = parseInt(e.target.value);
        document.getElementById('brush-size-val').textContent = brushSize + 'px';
    });
}

function drawMask(e) {
    const rect = e.target.getBoundingClientRect();
    const scaleX = e.target.width / rect.width;
    const scaleY = e.target.height / rect.height;
    const x = (e.clientX - rect.left) * scaleX;
    const y = (e.clientY - rect.top) * scaleY;

    maskCtx.fillStyle = 'rgba(108, 92, 231, 0.6)';
    maskCtx.beginPath();
    maskCtx.arc(x, y, brushSize * scaleX, 0, Math.PI * 2);
    maskCtx.fill();
}

function clearMask() {
    if (maskCtx) maskCtx.clearRect(0, 0, maskCtx.canvas.width, maskCtx.canvas.height);
}

// === Sliders ===
function setupSliders() {
    const mb = document.getElementById('max-bbox');
    mb.addEventListener('input', e => {
        document.getElementById('max-bbox-val').textContent = e.target.value + '%';
    });
}

// === Processing ===
async function processImage() {
    if (!sourceImage) return;

    const btn = document.getElementById('process-btn');
    const progressBar = document.getElementById('progress-bar');
    const progressFill = document.getElementById('progress-fill');

    btn.disabled = true;
    btn.textContent = 'Processing...';
    progressBar.style.display = 'block';
    progressFill.style.width = '30%';

    try {
        const imgB64 = canvasToB64('image-canvas');
        let result;

        if (currentMode === 'manual') {
            const maskB64 = canvasToB64('mask-canvas');
            progressFill.style.width = '50%';
            result = await fetch(`${API}/inpaint`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    image_b64: imgB64,
                    mask_b64: maskB64,
                    double_pass: document.getElementById('double-pass').checked,
                }),
            });
        } else {
            progressFill.style.width = '50%';
            result = await fetch(`${API}/remove`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    image_b64: imgB64,
                    prompt: document.getElementById('detect-prompt').value,
                    max_bbox_percent: parseFloat(document.getElementById('max-bbox').value),
                    mask_mode: document.getElementById('mask-mode').value,
                    double_pass: document.getElementById('double-pass').checked,
                    enhance: document.getElementById('enhance-region').checked,
                }),
            });
        }

        progressFill.style.width = '80%';
        const data = await result.json();

        if (data.result_b64) {
            displayResult(data.result_b64);
            progressFill.style.width = '100%';
        }
    } catch (err) {
        alert('Error: ' + err.message);
    } finally {
        btn.disabled = false;
        btn.textContent = 'Remove Watermark';
        setTimeout(() => { progressBar.style.display = 'none'; progressFill.style.width = '0%'; }, 1000);
    }
}

function displayResult(b64) {
    const img = new Image();
    img.onload = () => {
        resultImage = img;
        // Show result on canvas
        const c = document.getElementById('image-canvas');
        c.width = img.width;
        c.height = img.height;
        imageCtx.clearRect(0, 0, c.width, c.height);
        imageCtx.drawImage(img, 0, 0);
        document.getElementById('save-btn').style.display = 'flex';
        document.getElementById('image-info').textContent =
            `${img.width} × ${img.height} — Result (full resolution)`;
    };
    img.src = 'data:image/png;base64,' + b64;
}

function canvasToB64(canvasId) {
    const c = document.getElementById(canvasId);
    return c.toDataURL('image/png').split(',')[1];
}

function saveResult() {
    if (!resultImage) return;
    const fmt = document.getElementById('output-format').value;
    const c = document.getElementById('image-canvas');
    const link = document.createElement('a');

    if (fmt === 'JPEG') {
        link.href = c.toDataURL('image/jpeg', 1.0);
        link.download = 'openwipe-result.jpg';
    } else if (fmt === 'WEBP') {
        link.href = c.toDataURL('image/webp', 1.0);
        link.download = 'openwipe-result.webp';
    } else {
        link.href = c.toDataURL('image/png');
        link.download = 'openwipe-result.png';
    }

    link.click();
}

function resetEditor() {
    sourceImage = null;
    resultImage = null;
    document.getElementById('drop-zone').style.display = 'flex';
    document.getElementById('editor').style.display = 'none';
    document.getElementById('action-bar').style.display = 'none';
    document.getElementById('save-btn').style.display = 'none';
    document.getElementById('file-input').value = '';
}
