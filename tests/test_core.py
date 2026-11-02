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


def page_share(image: np.ndarray, contour: np.ndarray) -> float:
    return cv2.contourArea(contour) / (image.shape[0] * image.shape[1])


@pytest.mark.parametrize("name, share", [
    ("example_doc.jpg", 0.55),
    ("example_doc3.JPG", 0.71),
    ("example_doc4.jpg", 0.53),
])
def test_find_document_finds_whole_page(name: str, share: float) -> None:
    image = core.load_image(str(EXAMPLES / name))
    contour = core.find_document(image)

    assert contour is not None
    assert page_share(image, contour) == pytest.approx(share, abs=0.02)


def test_find_document_works_on_large_photo() -> None:
    image = core.load_image(str(EXAMPLES / "example_doc.jpg"))
    large = cv2.resize(image, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)

    contour = core.find_document(large)

    assert contour is not None
    assert page_share(large, contour) == pytest.approx(0.55, abs=0.02)


def test_find_document_returns_none_on_blank_image() -> None:
    assert core.find_document(np.full((400, 300, 3), 128, np.uint8)) is None


def test_find_document_ignores_small_shape() -> None:
    image = np.full((1000, 800, 3), 60, np.uint8)
    cv2.rectangle(image, (100, 100), (160, 140), (255, 255, 255), -1)

    assert core.find_document(image) is None


def test_order_corners_sorts_shuffled_points() -> None:
    contour = np.array([[300, 400], [100, 100], [100, 400], [300, 100]], np.int32)

    assert core.order_corners(contour).tolist() == [
        [100, 100], [300, 100], [300, 400], [100, 400],
    ]


@pytest.mark.parametrize("far_edge", [0.7, 0.8, 0.9, 1.0])
def test_order_corners_keeps_all_corners_at_every_tilt(far_edge: float) -> None:
    w = 210 * far_edge
    page = np.array([[-w, -297], [w, -297], [210, 297], [-210, 297]], float)

    for degrees in range(360):
        contour = rotate_about_centre(page, degrees)
        assert len({tuple(p) for p in core.order_corners(contour).tolist()}) == 4


def test_order_corners_repairs_crossed_handles() -> None:
    contour = np.array([[100, 100], [300, 400], [300, 100], [100, 400]], np.int32)

    assert core.order_corners(contour).tolist() == [
        [100, 100], [300, 100], [300, 400], [100, 400],
    ]


@pytest.mark.parametrize("points, valid", [
    ([[100, 100], [700, 100], [700, 900], [100, 900]], True),
    ([[100, 100], [700, 900], [700, 100], [100, 900]], True),   # crossed handles
    ([[50, 50]] * 4, False),
    ([[100, 100], [200, 200], [300, 300], [400, 400]], False),  # straight line
    ([[100, 100], [130, 100], [130, 130], [100, 130]], False),  # too small
    ([[100, 100], [700, 100], [400, 200], [400, 900]], False),  # bends inward
])
def test_is_valid_quad(points: list, valid: bool) -> None:
    image = np.zeros((1000, 800, 3), np.uint8)
    contour = np.array(points, np.int32).reshape(4, 1, 2)

    assert core.is_valid_quad(contour, image) == valid


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


@pytest.mark.parametrize("sharpness", [1, 5, 9, 50, 100])
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


def test_load_image_with_thai_name(tmp_path: Path) -> None:
    path = tmp_path / "เอกสาร.jpg"
    shutil.copy(EXAMPLES / "example_doc.jpg", path)

    assert core.load_image(str(path)) is not None


def test_save_image_with_thai_name(tmp_path: Path) -> None:
    path = tmp_path / "ใบเสร็จ.png"

    assert core.save_image(str(path), np.zeros((4, 4, 3), np.uint8))
    assert path.exists()


def test_load_image_returns_none_for_bad_files(tmp_path: Path) -> None:
    empty = tmp_path / "empty.jpg"
    empty.touch()
    text = tmp_path / "text.jpg"
    text.write_text("not an image")

    assert core.load_image(str(tmp_path / "missing.jpg")) is None
    assert core.load_image(str(empty)) is None
    assert core.load_image(str(text)) is None


@pytest.mark.parametrize("name", ["scan.pdf", "scan", "missing_folder/scan.png"])
def test_save_image_returns_false_on_failure(tmp_path: Path, name: str) -> None:
    assert not core.save_image(str(tmp_path / name), np.zeros((4, 4, 3), np.uint8))
