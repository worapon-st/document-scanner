import numpy as np
import cv2
from dataclasses import dataclass

@dataclass
class DocScanner():
    image: np.ndarray

    def preprocess_image(self) -> np.ndarray:
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        edge = cv2.Canny(blur, 75, 200)
        return edge

    def find_contour(self, edge: np.ndarray) -> np.ndarray:
        contour, _ = cv2.findContours(edge, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contour = sorted(contour, key=cv2.contourArea, reverse=True)[:5]
        return contour

    def get_document_contour(self, contour: np.ndarray) -> np.ndarray:
        for i in contour:
            epsilon = 0.02 * cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, epsilon, True)

            if len(approx) == 4:
                return approx
        
        return None

    def transform_perspective(self, contour: np.ndarray) -> np.ndarray:
        points = contour.reshape(4, 2)
        rect = np.zeros((4,2), dtype='float32')
        s = points.sum(axis=1)
        diff = np.diff(points, axis=1)
        
        rect[0] = points[np.argmin(s)]
        rect[2] = points[np.argmax(s)]
        rect[1] = points[np.argmin(diff)]
        rect[3] = points[np.argmax(diff)] 

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
        warped = cv2.warpPerspective(self.image, M, (maxWidth, maxHeight))
        
        return warped

