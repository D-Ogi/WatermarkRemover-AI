"""OpenWipe API backend — serves ML inference to the Tauri frontend.

Provides REST endpoints for:
- Watermark/object detection (Florence-2)
- Inpainting (LaMa)
- File upload/download
- Batch processing

Runs on localhost:8765. Started automatically by the Tauri shell.
"""

import io
import os
import sys
import uuid
import json
import base64
import tempfile
from pathlib import Path

import numpy as np
import cv2
from PIL import Image
from loguru import logger

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="OpenWipe API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instances (lazy loaded)
_florence_model = None
_florence_processor = None
_lama_model = None
_device = None


def get_device():
    global _device
    if _device is None:
        import torch
        _device = "cuda" if torch.cuda.is_available() else "cpu"
    return _device


def get_florence():
    global _florence_model, _florence_processor
    if _florence_model is None:
        from transformers import AutoProcessor, Florence2ForConditionalGeneration
        from model_assets import ensure_florence
        device = get_device()
        path = str(ensure_florence())
        dtype = None if device == "cuda" else __import__("torch").float32
        _florence_model = Florence2ForConditionalGeneration.from_pretrained(
            path, torch_dtype=dtype
        ).to(device).eval()
        _florence_processor = AutoProcessor.from_pretrained(path)
        logger.info("Florence-2 loaded")
    return _florence_model, _florence_processor


def get_lama():
    global _lama_model
    if _lama_model is None:
        from lama_inpaint.runtime import LamaInpaint
        _lama_model = LamaInpaint(get_device())
        logger.info("LaMa loaded")
    return _lama_model


class DetectRequest(BaseModel):
    image_b64: str
    prompt: str = "watermark"
    max_bbox_percent: float = 10.0


class InpaintRequest(BaseModel):
    image_b64: str
    mask_b64: str
    double_pass: bool = False


class RemoveRequest(BaseModel):
    image_b64: str
    prompt: str = "watermark"
    max_bbox_percent: float = 10.0
    mask_mode: str = "box"
    double_pass: bool = False
    enhance: bool = False


def b64_to_image(b64: str) -> Image.Image:
    data = base64.b64decode(b64.split(",")[-1])
    return Image.open(io.BytesIO(data)).convert("RGB")


def image_to_b64(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    if fmt == "JPEG":
        img.save(buf, format="JPEG", quality=100, subsampling=0)
    elif fmt == "WEBP":
        img.save(buf, format="WEBP", quality=100)
    else:
        img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


@app.get("/api/health")
async def health():
    return {"status": "ok", "device": get_device()}


@app.post("/api/detect")
async def detect(req: DetectRequest):
    """Detect watermarks/objects and return bounding boxes."""
    try:
        image = b64_to_image(req.image_b64)
        model, processor = get_florence()

        from remwm import get_watermark_mask, detect_only
        detections = detect_only(
            np.array(image), model, processor, get_device(),
            req.max_bbox_percent, req.prompt
        )
        return {"detections": detections}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/inpaint")
async def inpaint(req: InpaintRequest):
    """Inpaint masked regions using LaMa."""
    try:
        image = b64_to_image(req.image_b64)
        mask = Image.open(io.BytesIO(base64.b64decode(req.mask_b64.split(",")[-1]))).convert("L")
        lama = get_lama()

        from remwm import process_image_with_lama
        img_np = np.array(image)
        mask_np = np.array(mask)
        result = process_image_with_lama(img_np, mask_np, lama)

        if req.double_pass:
            result = process_image_with_lama(
                cv2.cvtColor(result, cv2.COLOR_BGR2RGB), mask_np, lama
            )

        result_img = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        return {"result_b64": image_to_b64(result_img)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/remove")
async def remove(req: RemoveRequest):
    """One-click: detect + inpaint watermarks."""
    try:
        image = b64_to_image(req.image_b64)
        model, processor = get_florence()
        lama = get_lama()

        from remwm import get_watermark_mask, process_image_with_lama
        mask = get_watermark_mask(
            image, model, processor, get_device(),
            req.max_bbox_percent, req.prompt, mask_mode=req.mask_mode
        )

        img_np = np.array(image)
        mask_np = np.array(mask)
        result = process_image_with_lama(img_np, mask_np, lama)

        if req.double_pass:
            result = process_image_with_lama(
                cv2.cvtColor(result, cv2.COLOR_BGR2RGB), mask_np, lama
            )

        result_img = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))

        if req.enhance:
            from quality_pipeline import QualityPipeline
            qp = QualityPipeline(lama, get_device())
            result_img = qp._enhance_inpainted_region(result_img, mask)

        # Return mask for preview too
        mask_b64 = base64.b64encode(
            cv2.imencode(".png", np.array(mask))[1].tobytes()
        ).decode()

        return {
            "result_b64": image_to_b64(result_img),
            "mask_b64": mask_b64,
            "image_width": image.width,
            "image_height": image.height,
        }
    except Exception as e:
        import traceback
        logger.error(f"/api/remove failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


# Serve frontend static files at root
import os
_ui_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ui-openwipe")
if os.path.isdir(_ui_dir):
    app.mount("/", StaticFiles(directory=_ui_dir, html=True), name="ui")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8765, log_level="info")
