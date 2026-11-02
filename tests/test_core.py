import math
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest

from docscan import core

EXAMPLES = Path(__file__).parents[1] / "img" / "example"


def rotate_about_centre(points: np.ndarray, degrees: float) -> np.ndarray:
    t = math.radians(degrees)
    rotation = np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])
    return (points @ rotation.T + 500).astype(np.int32)


# xfail(strict=True) marks a known bug. The test fails the run once the bug is
# fixed, so the marker has to be removed then.

@pytest.mark.parametrize("name, page_share", [
    ("example_doc.jpg", 0.55),
    pytest.param("example_doc3.JPG", 0.71, marks=pytest.mark.xfail(
        strict=True, reason="picks the table inside the page")),
    pytest.param("example_doc4.jpg", 0.53, marks=pytest.mark.xfail(
        strict=True, reason="picks a small square inside the page")),
])
def test_find_document_finds_whole_page(name: str, page_share: float) -> None:
    image = core.load_image(str(EXAMPLES / name))
    contour = core.find_document(image)

    assert contour is not None
    share = cv2.contourArea(contour) / (image.shape[0] * image.shape[1])
    assert share == pytest.approx(page_share, abs=0.02)


def test_find_document_returns_none_on_blank_image() -> None:
    assert core.find_document(np.full((400, 300, 3), 128, np.uint8)) is None


def test_order_corners_sorts_shuffled_points() -> None:
    contour = np.array([[300, 400], [100, 100], [100, 400], [300, 100]], np.int32)

    assert core.order_corners(contour).tolist() == [
        [100, 100], [300, 100], [300, 400], [100, 400],
    ]


@pytest.mark.xfail(strict=True, reason="one point is picked twice near 45 degrees")
def test_order_corners_keeps_all_corners_of_tilted_page() -> None:
    page = np.array([[-168, -297], [168, -297], [210, 297], [-210, 297]], float)
    contour = rotate_about_centre(page, 45)

    assert len({tuple(p) for p in core.order_corners(contour).tolist()}) == 4


def test_warp_flattens_area_to_its_size() -> None:
    image = np.zeros((500, 600, 3), np.uint8)
    image[100:400, 200:500] = 255
    contour = np.array([[200, 100], [499, 100], [499, 399], [200, 399]], np.int32)

    result = core.warp(image, contour)

    assert result.shape == (299, 299, 3)
    assert result.mean() > 250


def test_adjust_at_zero_keeps_pixels() -> None:
    image = np.random.default_rng(0).integers(0, 256, (20, 20, 3), np.uint8)

    assert np.array_equal(core.adjust(image), image)


def test_adjust_brightness_clips_at_white() -> None:
    image = np.full((4, 4, 3), 200, np.uint8)

    assert (core.adjust(image, brightness=100) == 255).all()


def test_adjust_contrast_moves_pixels_away_from_midpoint() -> None:
    image = np.array([[[100, 100, 100], [156, 156, 156]]], np.uint8)

    assert core.adjust(image, contrast=100).tolist() == [[[72] * 3, [184] * 3]]


def test_adjust_saturation_minus_100_removes_colour() -> None:
    image = np.full((4, 4, 3), (30, 90, 200), np.uint8)

    result = core.adjust(image, saturation=-100)

    assert (result.max(axis=2) == result.min(axis=2)).all()


@pytest.mark.parametrize("sharpness", [
    pytest.param(5, marks=pytest.mark.xfail(
        strict=True, reason="values 1 to 9 blur with size 1, so nothing changes")),
    50,
])
def test_adjust_sharpness_changes_edges(sharpness: int) -> None:
    image = np.zeros((40, 40, 3), np.uint8)
    image[:, 20:] = 200

    assert not np.array_equal(core.adjust(image, sharpness=sharpness), image)


def test_adjust_leaves_input_untouched() -> None:
    image = np.full((4, 4, 3), 100, np.uint8)

    core.adjust(image, brightness=50, contrast=50, saturation=50, sharpness=50)

    assert (image == 100).all()


def test_save_then_load_keeps_pixels(tmp_path: Path) -> None:
    image = np.random.default_rng(0).integers(0, 256, (20, 20, 3), np.uint8)
    path = str(tmp_path / "scan.png")

    assert core.save_image(path, image)
    assert np.array_equal(core.load_image(path), image)


@pytest.mark.xfail(strict=True, reason="OpenCV cannot open non-English paths on Windows")
def test_load_image_with_thai_name(tmp_path: Path) -> None:
    path = tmp_path / "เอกสาร.jpg"
    shutil.copy(EXAMPLES / "example_doc.jpg", path)

    assert core.load_image(str(path)) is not None


@pytest.mark.xfail(strict=True, reason="OpenCV writes the file under a garbled name")
def test_save_image_with_thai_name(tmp_path: Path) -> None:
    path = tmp_path / "ใบเสร็จ.png"

    core.save_image(str(path), np.zeros((4, 4, 3), np.uint8))

    assert path.exists()
