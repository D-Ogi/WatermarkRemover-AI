from enum import Enum
import random
import matplotlib.patches as patches
import numpy as np
from PIL import ImageDraw

# Constants
colormap = ['blue', 'orange', 'green', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'red',
            'lime', 'indigo', 'violet', 'aqua', 'magenta', 'coral', 'gold', 'tan', 'skyblue']

# To be set
model = None
processor = None

def set_model_info(model_, processor_):
    global model, processor
    model = model_
    processor = processor_

class TaskType(str, Enum):
    """The types of tasks supported"""
    CAPTION = '<CAPTION>'
    DETAILED_CAPTION = '<DETAILED_CAPTION>'
    MORE_DETAILED_CAPTION = '<MORE_DETAILED_CAPTION>'

def run_example(task_prompt: TaskType, image, text_input=None):
    """Runs an inference task using the model."""
    if not isinstance(task_prompt, TaskType):
        raise ValueError(f"task_prompt must be a TaskType, but {task_prompt} is of type {type(task_prompt)}")

    prompt = task_prompt.value if text_input is None else task_prompt.value + text_input
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    generated_ids = model.generate(
        input_ids=inputs["input_ids"].cuda(),
        pixel_values=inputs["pixel_values"].cuda(),
        max_new_tokens=1024,
        early_stopping=False,
        do_sample=False,
        num_beams=3,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text,
        task=task_prompt.value,
        image_size=(image.width, image.height)
    )
    return parsed_answer

def draw_polygons(image, prediction, fill_mask=False):
    """Draws segmentation masks with polygons on an image."""
    draw = ImageDraw.Draw(image)
    for polygons, label in zip(prediction['polygons'], prediction['labels']):
        color = random.choice(colormap)
        fill_color = random.choice(colormap) if fill_mask else None

        for polygon in polygons:
            polygon = np.array(polygon).reshape(-1, 2)
            if len(polygon) < 3:
                print('Invalid polygon:', polygon)
                continue

            polygon = (polygon * 1).reshape(-1).tolist()  # No scaling
            draw.polygon(polygon, outline=color, fill=fill_color)
            draw.text((polygon[0] + 8, polygon[1] + 2), label, fill=color)

    return image

def draw_ocr_bboxes(image, prediction):
    """Draws OCR bounding boxes on an image."""
    draw = ImageDraw.Draw(image)
    bboxes, labels = prediction['quad_boxes'], prediction['labels']
    for box, label in zip(bboxes, labels):
        color = random.choice(colormap)
        new_box = (np.array(box) * 1).tolist()  # No scaling
        draw.polygon(new_box, width=3, outline=color)
        draw.text((new_box[0] + 8, new_box[1] + 2), "{}".format(label), align="right", fill=color)
    return image

def convert_bbox_to_relative(box, image):
    """Converts bounding box pixel coordinates to relative coordinates in the range 0-999."""
    return [
        (box[0] / image.width) * 999,
        (box[1] / image.height) * 999,
        (box[2] / image.width) * 999,
        (box[3] / image.height) * 999,
    ]

def convert_relative_to_bbox(relative, image):
    """Converts list of relative coordinates to pixel coordinates."""
    return [
        (relative[0] / 999) * image.width,
        (relative[1] / 999) * image.height,
        (relative[2] / 999) * image.width,
        (relative[3] / 999) * image.height,
    ]

def convert_bbox_to_loc(box, image):
    """Converts bounding box pixel coordinates to position tokens."""
    relative_coordinates = convert_bbox_to_relative(box, image)
    return ''.join([f'<loc_{i}>' for i in relative_coordinates])
