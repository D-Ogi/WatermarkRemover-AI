# OpenWipe — Full-Resolution Watermark & Object Removal

**Fully local. Full resolution. No cloud. No quality loss.**

A desktop application for removing watermarks, logos, text, and unwanted objects from images — powered by Florence-2 detection + LaMa inpainting, running entirely on your machine.

## Why another watermark remover?

Commercial tools like VisioMint upload your images to the cloud for processing. Open-source alternatives often sacrifice quality by downscaling before processing. OpenWipe does neither:

- **100% local** — your images never leave your machine
- **Full resolution** — output matches input dimensions exactly
- **Max quality output** — JPEG quality=100 with 4:4:4 chroma, or lossless PNG
- **Smart processing** — automatic VRAM-aware tiling for large images

## Quality improvements over WatermarkRemover-AI

This project fixes the core quality pipeline issues:

| Issue | WatermarkRemover-AI | OpenWipe |
|---|---|---|
| Resolution | Downscaled via `max_dim` | Full resolution by default |
| JPEG quality | PIL default (75) | 100 + 4:4:4 chroma |
| Large images | Downscaled | Full-res + smart cropping |
| Output dimensions | Reduced | Original preserved |

## Features

- **Auto-detect watermarks/text/logos** using Florence-2
- **AI-powered inpainting** using LaMa (Large Mask Inpainting)
- **Brush-select** any region for manual object removal
- **Batch processing** for folders of images
- **Video support** with frame-by-frame watermark removal
- **Stroke mode** for precise edge-level masks vs box masks
- **Double-pass** inpainting for cleaner results
- **Transparent mode** — make watermark regions transparent instead of inpainting

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Single image
python remwm.py input.jpg output.jpg

# Batch process a folder
python remwm.py ./photos/ ./clean/

# With custom detection prompt
python remwm.py input.jpg output.jpg --detection-prompt "watermark logo"

# Preview detections without processing
python remwm.py input.jpg --preview

# Manual stroke mask (more precise than box)
python remwm.py input.jpg output.jpg --mask-mode stroke
```

## Tech Stack

- **Florence-2** (Microsoft) — open-vocabulary object/text detection
- **LaMa** — Large Mask Inpainting (TorchScript)
- **PyTorch** — inference backend (CPU or CUDA)
- **PyWebview** — native desktop window
- **Alpine.js** — lightweight UI framework

## License

MIT — see [LICENSE](LICENSE).

## Credits

Built on [WatermarkRemover-AI](https://github.com/D-Ogi/WatermarkRemover-AI) by D-Ogi (MIT), with significant quality pipeline improvements.
LaMa inpainting adapted from [IOPaint](https://github.com/Sanster/IOPaint) (Apache-2.0).
