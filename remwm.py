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
from torch import Module
import tqdm
from loguru import logger
from enum import Enum

try:
    from cv2.typing import MatLike
except ImportError:
    MatLike = np.ndarray


class TaskType(str, Enum):
    OPEN_VOCAB_DETECTION = "<OPEN_VOCABULARY_DETECTION>"
    """Detect bounding box for objects and OCR text"""


def identify(
    task_prompt: TaskType,
    image: MatLike,
    text_input: str,
    model: AutoModelForCausalLM,
    processor: AutoProcessor,
    device: str,
):
    """Runs an inference task using the model."""
    if not isinstance(task_prompt, TaskType):
        raise ValueError(
            f"task_prompt must be a TaskType, but {task_prompt} is of type {type(task_prompt)}"
        )

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
    parsed_answer = processor.post_process_generation(
        generated_text, task=task_prompt.value, image_size=(image.width, image.height)
    )
    return parsed_answer


def get_watermark_mask(
    image: MatLike, model: AutoModelForCausalLM, processor: AutoProcessor, device: str
):
    text_input = "watermark"
    task_prompt = TaskType.OPEN_VOCAB_DETECTION  # Use OPEN_VOCAB_DETECTION
    parsed_answer = identify(task_prompt, image, text_input, model, processor, device)

    # Get image dimensions
    image_width, image_height = image.size
    total_image_area = image_width * image_height

    # Create a mask based on bounding boxes
    mask = Image.new("L", image.size, 0)  # "L" mode for single-channel grayscale
    draw = ImageDraw.Draw(mask)

    detection_key = "<OPEN_VOCABULARY_DETECTION>"
    if detection_key in parsed_answer and "bboxes" in parsed_answer[detection_key]:
        for bbox in parsed_answer[detection_key]["bboxes"]:
            x1, y1, x2, y2 = map(int, bbox)  # Convert float bbox to int

            # Calculate the area of the bounding box
            bbox_area = (x2 - x1) * (y2 - y1)

            # If the area of the bounding box is less than 10% of the image area, include it in the mask
            if bbox_area <= 0.1 * total_image_area:
                draw.rectangle(
                    [x1, y1, x2, y2], fill=255
                )  # Draw a white rectangle on the mask
            else:
                print(
                    f"Skipping region: Bounding box covers more than 10% of the image. BBox Area: {bbox_area}, Image Area: {total_image_area}"
                )
    else:
        print("No bounding boxes found in parsed answer.")

    return mask


def process_image_with_lama(image: MatLike, mask: MatLike, model_manager: ModelManager):
    config = Config(
        ldm_steps=50,  # Increased steps for higher quality
        ldm_sampler=LDMSampler.ddim,
        hd_strategy=HDStrategy.CROP,  # Use CROP strategy for higher quality
        hd_strategy_crop_margin=64,  # Increase crop margin to provide more context
        hd_strategy_crop_trigger_size=800,  # Higher trigger size for larger images
        hd_strategy_resize_limit=1600,  # Increase limit for processing larger images
    )
    result = model_manager(image, mask, config)

    # Ensure result is in the correct format
    if result.dtype in [np.float64, np.float32]:
        result = np.clip(result, 0, 255)
        result = result.astype(np.uint8)

    return result


@click.command()
@click.argument("input_image", type=click.Path(exists=True))
@click.argument("output_image", type=click.Path())
def main(input_image: str, output_image: str):
    input_image_path = Path(input_image)
    output_image_path = Path(output_image)

    if not input_image_path.exists():
        logger.error(f"Input not found: {input_image_path}")
        sys.exit(1)

    # Load Florence2 model and processor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    florence_model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(
        "microsoft/Florence-2-large", trust_remote_code=True
    ).to(device)
    florence_model.eval()
    florence_processor: AutoProcessor = AutoProcessor.from_pretrained(
        "microsoft/Florence-2-large", trust_remote_code=True
    )

    # Load LaMa model
    model_manager = ModelManager(name="lama", device=device)
    logger.info("models loaded")

    def handle_one(image_path: Path, output_path: Path):
        image = Image.open(image_path).convert("RGB")

        # Get watermark mask
        mask_image = get_watermark_mask(
            image, florence_model, florence_processor, device
        )

        # Process image with LaMa
        result_image = process_image_with_lama(
            np.array(image), np.array(mask_image), model_manager
        )

        result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
        result_image_pil = Image.fromarray(result_image_rgb)
        result_image_pil.save(output_path)

    if input_image_path.is_dir():
        logger.info("handle input `{}` as directory", input_image_path)

        if not output_image_path.exists():
            logger.info("create output directory `{}`", output_image_path)
            output_image_path.mkdir()

        if output_image_path.is_file():
            logger.error("output {} should be a directory", output_image_path)
            raise ValueError("output should be a directory")

        COMMON_IMAGE_EXTENSIONS = ["jpg", "jpeg", "png", "bmp"]

        def grab_all():
            for ext in COMMON_IMAGE_EXTENSIONS:
                yield from input_image_path.glob(f"*.{ext}")

        for image_path in tqdm.tqdm(list(grab_all())):
            handle_one(image_path, output_image_path / image_path.name)
    else:
        # must be a file
        logger.info("handle input {} as file", input_image_path)
        if output_image_path.is_dir():
            out = output_image_path / input_image_path.name
        else:
            out = output_image_path
        handle_one(input_image_path, out)
        logger.info("output saved to {}", out)


if __name__ == "__main__":
    main()  # py-lint: disable=no-value-for-parameter
