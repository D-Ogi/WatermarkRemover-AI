"""Exercise geometry and mask helpers on CPU without downloading models."""

import pytest
from PIL import Image

import utils


@pytest.mark.parametrize("size", [(320, 200), (200, 320), (1, 1)])
def test_image_edges_map_to_normalized_endpoints(size):
    """The whole image spans zero through 999 on each axis."""
    image = Image.new("RGB", size)
    assert utils.convert_bbox_to_relative([0, 0, *size], image) == [0, 0, 999, 999]
    assert utils.convert_relative_to_bbox([0, 0, 999, 999], image) == [0, 0, *size]


def test_normalized_box_uses_each_axis_dimension():
    """A non-square image must not scale vertical coordinates with its width."""
    image = Image.new("RGB", (400, 200))
    assert utils.convert_relative_to_bbox([249.75, 499.5, 999, 999], image) == pytest.approx(
        [100, 100, 400, 200]
    )


def test_filled_polygon_changes_only_its_region(monkeypatch):
    """A synthetic mask fills its interior without altering distant background."""
    monkeypatch.setattr(utils.random, "choice", lambda values: "red")
    image = Image.new("RGB", (64, 64), "white")
    result = utils.draw_polygons(
        image, {"polygons": [[[10, 10, 30, 10, 30, 30, 10, 30]]], "labels": [""]}, True
    )
    assert result.size == (64, 64)
    assert result.getpixel((20, 20)) == (255, 0, 0)
    assert result.getpixel((50, 50)) == (255, 255, 255)


def test_invalid_polygon_leaves_image_unchanged():
    """A shape with fewer than three vertices cannot form a segmentation mask."""
    image = Image.new("RGB", (32, 32), "white")
    before = image.tobytes()
    utils.draw_polygons(image, {"polygons": [[[1, 1, 2, 2]]], "labels": [""]}, True)
    assert image.tobytes() == before


def test_invalid_task_rejected_before_model_access(monkeypatch):
    """Unsupported task names fail without initializing a model or processor."""
    monkeypatch.setattr(utils, "model", None)
    monkeypatch.setattr(utils, "processor", None)
    with pytest.raises(ValueError, match="task_prompt must be a TaskType"):
        utils.run_example("unsupported", Image.new("RGB", (8, 8)))
