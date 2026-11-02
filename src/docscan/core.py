import os
from typing import Optional

import cv2
import numpy as np
from PIL import Image

DETECT_SIZE       = 800    # px — long side of the copy used for detection
MIN_DOCUMENT_AREA = 0.15   # share of the photo a detected page must cover
MIN_MANUAL_AREA   = 0.01   # share of the photo hand-placed corners must cover
PDF_LONG_SIDE     = 11.69  # inches — long side of an A4 page


# cv2.imread and cv2.imwrite cannot handle non-English paths on Windows, so
# files go through numpy and OpenCV only sees the bytes.
def load_image(path: str) -> Optional[np.ndarray]:
    try:
        return cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
    except (OSError, cv2.error):
        return None


def save_image(path: str, image: np.ndarray) -> bool:
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".pdf":
            rgb = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            rgb.save(path, "PDF", quality=90,
                     resolution=max(image.shape[:2]) / PDF_LONG_SIDE)
            return True

        ok, data = cv2.imencode(ext, image)
        if ok:
            data.tofile(path)
        return ok
    except (OSError, cv2.error):
        return False


def find_document(image: np.ndarray) -> Optional[np.ndarray]:
    """Return the 4-point contour of the document, or None if there is none."""
    h, w = image.shape[:2]
    k = min(1.0, DETECT_SIZE / max(h, w))
    small = cv2.resize(image, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edge = cv2.Canny(blur, 50, 150)
    edge = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    min_area = MIN_DOCUMENT_AREA * small.shape[0] * small.shape[1]
    contours, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
        hull = cv2.convexHull(contour)
        if cv2.contourArea(hull) < min_area:
            continue

        epsilon = 0.02 * cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, epsilon, True)

        if len(approx) == 4:
            return (approx / k).astype(np.int32)

    return None


def order_corners(contour: np.ndarray) -> np.ndarray:
    """Return 4 points as a float32 array ordered TL, TR, BR, BL."""
    points = contour.reshape(4, 2).astype('float32')
    offset = points - points.mean(axis=0)
    points = points[np.argsort(np.arctan2(offset[:, 1], offset[:, 0]))]
    return np.roll(points, -int(np.argmin(points.sum(axis=1))), axis=0)


def is_valid_quad(contour: np.ndarray, image: np.ndarray) -> bool:
    """True if the 4 points form a convex shape big enough to flatten."""
    rect = order_corners(contour)
    min_area = MIN_MANUAL_AREA * image.shape[0] * image.shape[1]
    return cv2.isContourConvex(rect) and cv2.contourArea(rect) >= min_area


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


def rotate(image: np.ndarray, quarter_turns: int) -> np.ndarray:
    """Return image turned clockwise by 90 degrees per quarter turn."""
    return np.rot90(image, -quarter_turns).copy()


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

    # Contrast: -100…100 — scales pixel values around the midpoint (128).
    # 0 = no change; 100 ≈ doubles contrast; -100 = flat grey.
    if contrast != 0:
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
        k: int              = int(sharpness / 10) * 2 + 3   # must be odd, 3 or more
        blurred: np.ndarray = cv2.GaussianBlur(uint, (k, k), 0)
        strength: float     = sharpness / 40.0
        img = cv2.addWeighted(
            uint, 1.0 + strength, blurred, -strength, 0
        ).astype(np.float32)

    return np.clip(img, 0, 255).astype(np.uint8)
