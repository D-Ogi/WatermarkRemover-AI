import sys
import click
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw
from transformers import AutoProcessor, AutoModelForCausalLM
from iopaint.model_manager import ModelManager
from iopaint.schema import HDStrategy, LDMSampler, InpaintRequest as Config
import torch
from torch.nn import Module
import tqdm
from loguru import logger
from enum import Enum
import os
import tempfile
import shutil
import subprocess

try:
    from cv2.typing import MatLike
except ImportError:
    MatLike = np.ndarray

class TaskType(str, Enum):
    OPEN_VOCAB_DETECTION = "<OPEN_VOCABULARY_DETECTION>"
    """Detect bounding box for objects and OCR text"""

def identify(task_prompt: TaskType, image: MatLike, text_input: str, model: AutoModelForCausalLM, processor: AutoProcessor, device: str):
    if not isinstance(task_prompt, TaskType):
        raise ValueError(f"task_prompt must be a TaskType, but {task_prompt} is of type {type(task_prompt)}")

    prompt = task_prompt.value if text_input is None else task_prompt.value + text_input
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        early_stopping=False,
        do_sample=False,
        num_beams=3,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    return processor.post_process_generation(
        generated_text, task=task_prompt.value, image_size=(image.width, image.height)
    )

def get_watermark_mask(image: MatLike, model: AutoModelForCausalLM, processor: AutoProcessor, device: str, max_bbox_percent: float):
    text_input = "watermark"
    task_prompt = TaskType.OPEN_VOCAB_DETECTION
    parsed_answer = identify(task_prompt, image, text_input, model, processor, device)

    mask = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(mask)

    detection_key = "<OPEN_VOCABULARY_DETECTION>"
    if detection_key in parsed_answer and "bboxes" in parsed_answer[detection_key]:
        image_area = image.width * image.height
        for bbox in parsed_answer[detection_key]["bboxes"]:
            x1, y1, x2, y2 = map(int, bbox)
            bbox_area = (x2 - x1) * (y2 - y1)
            if (bbox_area / image_area) * 100 <= max_bbox_percent:
                draw.rectangle([x1, y1, x2, y2], fill=255)
            else:
                logger.warning(f"Skipping large bounding box: {bbox} covering {bbox_area / image_area:.2%} of the image")

    return mask

def process_image_with_lama(image: MatLike, mask: MatLike, model_manager: ModelManager):
    config = Config(
        ldm_steps=50,
        ldm_sampler=LDMSampler.ddim,
        hd_strategy=HDStrategy.CROP,
        hd_strategy_crop_margin=64,
        hd_strategy_crop_trigger_size=800,
        hd_strategy_resize_limit=1600,
    )
    result = model_manager(image, mask, config)

    if result.dtype in [np.float64, np.float32]:
        result = np.clip(result, 0, 255).astype(np.uint8)

    return result

def make_region_transparent(image: Image.Image, mask: Image.Image):
    image = image.convert("RGBA")
    mask = mask.convert("L")
    transparent_image = Image.new("RGBA", image.size)
    for x in range(image.width):
        for y in range(image.height):
            if mask.getpixel((x, y)) > 0:
                transparent_image.putpixel((x, y), (0, 0, 0, 0))
            else:
                transparent_image.putpixel((x, y), image.getpixel((x, y)))
    return transparent_image

def is_video_file(file_path):
    """Check if the file is a video based on its extension"""
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm']
    return Path(file_path).suffix.lower() in video_extensions

def process_video(input_path, output_path, florence_model, florence_processor, model_manager, device, transparent, max_bbox_percent, force_format):
    """Process a video file by extracting frames, removing watermarks, and reconstructing the video"""
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        logger.error(f"Error opening video file: {input_path}")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Determine output format
    if force_format:
        output_format = force_format.upper()
    else:
        output_format = "MP4"  # Default to MP4 for videos
    
    # Create output video file
    output_path = Path(output_path)
    if output_path.is_dir():
        output_file = output_path / f"{input_path.stem}_no_watermark.{output_format.lower()}"
    else:
        output_file = output_path.with_suffix(f".{output_format.lower()}")
    
    # Créer un fichier temporaire pour la vidéo sans audio
    temp_dir = tempfile.mkdtemp()
    temp_video_path = Path(temp_dir) / f"temp_no_audio.{output_format.lower()}"
    
    # Set codec based on output format
    if output_format.upper() == "MP4":
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    elif output_format.upper() == "AVI":
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
    else:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Default to MP4
    
    out = cv2.VideoWriter(str(temp_video_path), fourcc, fps, (width, height))
    
    # Process each frame
    with tqdm.tqdm(total=total_frames, desc="Processing video frames") as pbar:
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert frame to PIL Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            # Get watermark mask
            mask_image = get_watermark_mask(pil_image, florence_model, florence_processor, device, max_bbox_percent)
            
            # Process frame
            if transparent:
                # For video, we can't use transparency, so we'll fill with a color or background
                result_image = make_region_transparent(pil_image, mask_image)
                # Convert RGBA to RGB by filling transparent areas with white
                background = Image.new("RGB", result_image.size, (255, 255, 255))
                background.paste(result_image, mask=result_image.split()[3])
                result_image = background
            else:
                lama_result = process_image_with_lama(np.array(pil_image), np.array(mask_image), model_manager)
                result_image = Image.fromarray(cv2.cvtColor(lama_result, cv2.COLOR_BGR2RGB))
            
            # Convert back to OpenCV format and write to output video
            frame_result = cv2.cvtColor(np.array(result_image), cv2.COLOR_RGB2BGR)
            out.write(frame_result)
            
            # Update progress
            frame_count += 1
            pbar.update(1)
            progress = int((frame_count / total_frames) * 100)
            print(f"Processing frame {frame_count}/{total_frames}, progress:{progress}%")
    
    # Release resources
    cap.release()
    out.release()
    
    # Combiner la vidéo traitée avec l'audio original à l'aide de FFmpeg
    try:
        logger.info("Fusion de la vidéo traitée avec l'audio original...")
        
        # Vérifier si FFmpeg est disponible
        try:
            subprocess.check_output(["ffmpeg", "-version"], stderr=subprocess.STDOUT)
        except (subprocess.SubprocessError, FileNotFoundError):
            logger.warning("FFmpeg n'est pas disponible. La vidéo sera produite sans audio.")
            shutil.copy(str(temp_video_path), str(output_file))
        else:
            # Utiliser FFmpeg pour combiner la vidéo traitée avec l'audio original
            ffmpeg_cmd = [
                "ffmpeg", "-y",
                "-i", str(temp_video_path),  # Vidéo traitée sans audio
                "-i", str(input_path),       # Vidéo originale avec audio
                "-c:v", "copy",              # Copier la vidéo sans réencodage
                "-c:a", "aac",               # Encoder l'audio en AAC pour meilleure compatibilité
                "-map", "0:v:0",             # Utiliser la piste vidéo du premier fichier (vidéo traitée)
                "-map", "1:a:0",             # Utiliser la piste audio du deuxième fichier (vidéo originale)
                "-shortest",                  # Terminer quand la piste la plus courte se termine
                str(output_file)
            ]
            
            # Exécuter FFmpeg
            subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            logger.info("Fusion audio/vidéo terminée avec succès!")
    except Exception as e:
        logger.error(f"Erreur lors de la fusion audio/vidéo: {str(e)}")
        # En cas d'erreur, utiliser la vidéo sans audio
        shutil.copy(str(temp_video_path), str(output_file))
    finally:
        # Nettoyer les fichiers temporaires
        try:
            os.remove(str(temp_video_path))
            os.rmdir(temp_dir)
        except:
            pass
    
    logger.info(f"input_path:{input_path}, output_path:{output_file}, overall_progress:100")
    return output_file

def handle_one(image_path: Path, output_path: Path, florence_model, florence_processor, model_manager, device, transparent, max_bbox_percent, force_format, overwrite):
    if output_path.exists() and not overwrite:
        logger.info(f"Skipping existing file: {output_path}")
        return

    # Check if it's a video file
    if is_video_file(image_path):
        return process_video(image_path, output_path, florence_model, florence_processor, model_manager, device, transparent, max_bbox_percent, force_format)

    # Process image
    image = Image.open(image_path).convert("RGB")
    mask_image = get_watermark_mask(image, florence_model, florence_processor, device, max_bbox_percent)

    if transparent:
        result_image = make_region_transparent(image, mask_image)
    else:
        lama_result = process_image_with_lama(np.array(image), np.array(mask_image), model_manager)
        result_image = Image.fromarray(cv2.cvtColor(lama_result, cv2.COLOR_BGR2RGB))

    # Determine output format
    if force_format:
        output_format = force_format.upper()
    elif transparent:
        output_format = "PNG"
    else:
        output_format = image_path.suffix[1:].upper()
        if output_format not in ["PNG", "WEBP", "JPG"]:
            output_format = "PNG"
    
    # Map JPG to JPEG for PIL compatibility
    if output_format == "JPG":
        output_format = "JPEG"

    if transparent and output_format == "JPG":
        logger.warning("Transparency detected. Defaulting to PNG for transparency support.")
        output_format = "PNG"

    new_output_path = output_path.with_suffix(f".{output_format.lower()}")
    result_image.save(new_output_path, format=output_format)
    logger.info(f"input_path:{image_path}, output_path:{new_output_path}")
    return new_output_path

@click.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
@click.option("--overwrite", is_flag=True, help="Overwrite existing files in bulk mode.")
@click.option("--transparent", is_flag=True, help="Make watermark regions transparent instead of removing.")
@click.option("--max-bbox-percent", default=10.0, help="Maximum percentage of the image that a bounding box can cover.")
@click.option("--force-format", type=click.Choice(["PNG", "WEBP", "JPG", "MP4", "AVI"], case_sensitive=False), default=None, help="Force output format. Defaults to input format.")
def main(input_path: str, output_path: str, overwrite: bool, transparent: bool, max_bbox_percent: float, force_format: str):
    input_path = Path(input_path)
    output_path = Path(output_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    florence_model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True).to(device).eval()
    florence_processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)
    logger.info("Florence-2 Model loaded")

    if not transparent:
        model_manager = ModelManager(name="lama", device=device)
        logger.info("LaMa model loaded")
    else:
        model_manager = None

    if input_path.is_dir():
        if not output_path.exists():
            output_path.mkdir(parents=True)

        # Include video files in the search
        images = list(input_path.glob("*.[jp][pn]g")) + list(input_path.glob("*.webp"))
        videos = list(input_path.glob("*.mp4")) + list(input_path.glob("*.avi")) + list(input_path.glob("*.mov")) + list(input_path.glob("*.mkv"))
        files = images + videos
        total_files = len(files)

        for idx, file_path in enumerate(tqdm.tqdm(files, desc="Processing files")):
            output_file = output_path / file_path.name
            handle_one(file_path, output_file, florence_model, florence_processor, model_manager, device, transparent, max_bbox_percent, force_format, overwrite)
            progress = int((idx + 1) / total_files * 100)
            print(f"input_path:{file_path}, output_path:{output_file}, overall_progress:{progress}")
    else:
        output_file = output_path
        if is_video_file(input_path) and output_path.suffix.lower() not in ['.mp4', '.avi', '.mov', '.mkv']:
            # Ensure video output has proper extension
            if force_format and force_format.upper() in ["MP4", "AVI"]:
                output_file = output_path.with_suffix(f".{force_format.lower()}")
            else:
                output_file = output_path.with_suffix(".mp4")  # Default to mp4
        
        handle_one(input_path, output_file, florence_model, florence_processor, model_manager, device, transparent, max_bbox_percent, force_format, overwrite)
        print(f"input_path:{input_path}, output_path:{output_file}, overall_progress:100")

if __name__ == "__main__":
    main()
