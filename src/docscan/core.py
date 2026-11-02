from typing import Optional

import cv2
import numpy as np


def load_image(path: str) -> Optional[np.ndarray]:
    return cv2.imread(path)


def save_image(path: str, image: np.ndarray) -> bool:
    return cv2.imwrite(path, image)


def find_document(image: np.ndarray) -> Optional[np.ndarray]:
    """Return the 4-point contour of the document, or None if there is none."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edge = cv2.Canny(blur, 75, 200)

    contours, _ = cv2.findContours(edge, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) == 4:
            return approx

    return None


def order_corners(contour: np.ndarray) -> np.ndarray:
    """Return 4 points as a float32 array ordered TL, TR, BR, BL."""
    points = contour.reshape(4, 2)
    rect = np.zeros((4, 2), dtype='float32')
    s = points.sum(axis=1)
    diff = np.diff(points, axis=1)

    rect[0] = points[np.argmin(s)]
    rect[2] = points[np.argmax(s)]
    rect[1] = points[np.argmin(diff)]
    rect[3] = points[np.argmax(diff)]
    return rect


def warp(image: np.ndarray, contour: np.ndarray) -> np.ndarray:
    """Flatten the 4-point area of image into a straight rectangle."""
    rect = order_corners(contour)
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0])**2) + ((br[1] - bl[1])**2))
    widthB = np.sqrt(((tr[0] - tl[0])**2) + ((tr[1] - tl[1])**2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0])**2) + ((tr[1] - br[1])**2))
    heightB = np.sqrt(((tl[0] - bl[0])**2) + ((tl[1] - bl[1])**2))
    maxHeight = max(int(heightA), int(heightB))

    dist = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype='float32')

    M = cv2.getPerspectiveTransform(rect, dist)
    return cv2.warpPerspective(image, M, (maxWidth, maxHeight))


def adjust(
    image: np.ndarray,
    brightness: int = 0,
    contrast: int = 0,
    saturation: int = 0,
    sharpness: int = 0,
) -> np.ndarray:
    """Return a new image with the slider adjustments applied."""
    img: np.ndarray = image.astype(np.float32)

    # Brightness: -100…100 → pixel offset ±255
    if brightness != 0:
        img = np.clip(img + brightness * 2.55, 0, 255)

    # Contrast: 0…100 — scales pixel values around the midpoint (128).
    # 0 = no change; 100 ≈ doubles contrast.
    if contrast > 0:
        factor: float = 1.0 + contrast / 100.0
        img = np.clip((img - 128.0) * factor + 128.0, 0, 255)

    # Saturation: -100…100 — scale the HSV S channel.
    # +100 doubles saturation; -100 removes colour entirely.
    if saturation != 0:
        hsv: np.ndarray   = cv2.cvtColor(
            img.astype(np.uint8), cv2.COLOR_BGR2HSV
        ).astype(np.float32)
        sat_factor: float = 1.0 + saturation / 100.0
        hsv[:, :, 1]      = np.clip(hsv[:, :, 1] * sat_factor, 0, 255)
        img = cv2.cvtColor(
            hsv.astype(np.uint8), cv2.COLOR_HSV2BGR
        ).astype(np.float32)

    # Sharpness: 0…100 — unsharp mask with adaptive Gaussian kernel.
    if sharpness > 0:
        uint: np.ndarray    = img.astype(np.uint8)
        k: int              = max(1, int(sharpness / 10) * 2 + 1)   # must be odd
        blurred: np.ndarray = cv2.GaussianBlur(uint, (k, k), 0)
        strength: float     = sharpness / 40.0
        img = cv2.addWeighted(
            uint, 1.0 + strength, blurred, -strength, 0
        ).astype(np.float32)

    return np.clip(img, 0, 255).astype(np.uint8)
