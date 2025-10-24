# Utilities for perspective correction of a sheet in an image

import cv2
import numpy as np


def order_corners(corners: np.ndarray) -> np.ndarray:
    """
    Sorts the 4 corners in the order: top-left, top-right, bottom-right, bottom-left
    """
    rect = np.zeros((4, 2), dtype=np.float32)

    s = corners.sum(axis=1)
    rect[0] = corners[np.argmin(s)]
    rect[2] = corners[np.argmax(s)]

    diff = np.diff(corners, axis=1)
    rect[1] = corners[np.argmin(diff)]
    rect[3] = corners[np.argmax(diff)]

    return rect


def get_output_size(corners: np.ndarray) -> tuple[int, int]:
    """
    Compute the output size (width, height) of a quadrilateral based on its corner coordinates
    """
    # Horizontal distances
    width_top = np.linalg.norm(corners[0] - corners[1])
    width_bottom = np.linalg.norm(corners[2] - corners[3])
    max_width = int(max(width_top, width_bottom))

    # Vertical distances
    height_left = np.linalg.norm(corners[0] - corners[3])
    height_right = np.linalg.norm(corners[1] - corners[2])
    max_height = int(max(height_left, height_right))

    return max_width, max_height


def corrected_perspective(img: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """
    Applies a perspective correction to straighten the sheet
    """
    # Perfect destination rectangle
    corners = order_corners(corners)
    output_width, output_height = get_output_size(corners)
    dst_points = np.array([[0, 0], [output_width - 1, 0], [output_width - 1, output_height - 1], [0, output_height - 1]], dtype=np.float32)

    # Compute the perspective transformation matrix
    matrix = cv2.getPerspectiveTransform(corners, dst_points)

    # Apply the transformation
    corrected = cv2.warpPerspective(img, matrix, (output_width, output_height), flags=cv2.INTER_LANCZOS4)

    return corrected

