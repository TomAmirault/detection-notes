"""
Detects quadrilateral shapes in an image and filters them based on size and color
Only white quadrilaterals likely to represent sheets of paper are kept
"""

import cv2
import numpy as np


# --- HYPERPARAMETERS ---
# Bilateral filtering
BILATERAL_D: int = 5               # Neighborhood diameter for bilateral filtering
BILATERAL_SIGMA_COLOR: int = 20    # Filter strength for color differences
BILATERAL_SIGMA_SPACE: int = 20    # Filter strength for spatial distance

# Edge detection (Canny)
CANNY_THRESHOLD1: int = 75         # Lower threshold for Canny edges
CANNY_THRESHOLD2: int = 200        # Upper threshold for Canny edges

# Morphology (closing edges)
MORPH_KERNEL_SIZE: int = 7         # Kernel size for closing gaps
MORPH_ITERATIONS: int = 1          # Number of closing iterations

# Polygon approximation
POLY_EPSILON_FACTOR: float = 0.01  # Fraction of perimeter for approxPolyDP

# Quadrilateral filtering
MIN_AREA_RATIO: float = 0.05       # Minimum area relative to image to keep polygon

# Color filtering (white areas)
MAX_SATURATION: int = 100          # Maximum saturation to consider a region white
MIN_VALUE: int = 175               # Minimum brightness to consider a region white



def preprocessed_image(img: np.ndarray, bilateral_d: int = BILATERAL_D, bilateral_sigma_color: int = BILATERAL_SIGMA_COLOR, bilateral_sigma_space:int = BILATERAL_SIGMA_SPACE) -> np.ndarray:
    """
    Convert the image to grayscale and apply a bilateral filter to reduce noise while preserving edges.

    Args:
        img (np.ndarray): Input image.
        bilateral_d (int, optional): Neighborhood diameter for bilateral filtering.
        bilateral_sigma_color (int, optional): Filter strength for color differences.
        bilateral_sigma_space (int, optional): Filter strength for spatial distance.
    Returns:
        np.ndarray: Preprocessed image after grayscale conversion and bilateral filtering.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.bilateralFilter(gray, bilateral_d, bilateral_sigma_color, bilateral_sigma_space)
    return denoised


def shape_detector(img: np.ndarray, bilateral_d: int = BILATERAL_D, bilateral_sigma_color: int = BILATERAL_SIGMA_COLOR,
                bilateral_sigma_space: int = BILATERAL_SIGMA_SPACE, canny_threshold1: int = CANNY_THRESHOLD1, canny_threshold2: int = CANNY_THRESHOLD2,
                morph_kernel_size: int = MORPH_KERNEL_SIZE, morph_iterations: int = MORPH_ITERATIONS, poly_epsilon_factor: float = POLY_EPSILON_FACTOR,
                min_area_ratio: float = MIN_AREA_RATIO, max_saturation: int = MAX_SATURATION, min_value: int = MIN_VALUE) -> list[np.ndarray]:
    """
    Detect quadrilateral shapes in the image that likely correspond to sheets of paper.

    Steps:
    - Preprocess image (grayscale + bilateral filter)
    - Detect edges using Canny and close gaps with morphology
    - Approximate contours to polygons
    - Filter quadrilaterals by shape, area, convexity, and color (white)

    Args:
        img (np.ndarray): Input image.
        bilateral_d (int, optional): Neighborhood diameter for bilateral filtering.
        bilateral_sigma_color (int, optional): Filter strength for color differences.
        bilateral_sigma_space (int, optional): Filter strength for spatial distance.
        canny_threshold1 (int, optional): Lower threshold for Canny edge detection.
        canny_threshold2 (int, optional): Upper threshold for Canny edge detection.
        morph_kernel_size (int, optional): Kernel size used for morphological closing.
        morph_iterations (int, optional): Number of iterations for morphological operations.
        poly_epsilon_factor (float, optional): Factor for polygon approximation (fraction of perimeter).
        min_area_ratio (float, optional): Minimum area ratio relative to image to keep polygon.
        max_saturation (int, optional): Maximum saturation to consider a region as white.
        min_value (int, optional): Minimum brightness to consider a region as white.

    Returns:
        list[np.ndarray]: List of detected quadrilateral contours, each with 4 points.
    """
    proc = preprocessed_image(img, bilateral_d, bilateral_sigma_color, bilateral_sigma_space)
    h, w = img.shape[:2]

    # Binarize to find edges
    edges = cv2.Canny(proc, canny_threshold1, canny_threshold2)

    # Close gaps in edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_kernel_size, morph_kernel_size))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=morph_iterations)

    # Find contours
    contours, _ = cv2.findContours(
        edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    valid_shapes = []

    # Convert image to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    for cnt in contours:
        # Convex hull
        hull = cv2.convexHull(cnt)

        # Polygon approximation
        perimeter = cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, poly_epsilon_factor * perimeter, True)

        # Keep only polygons likely to be sheets of paper
        if len(approx) == 4 and cv2.contourArea(cnt) > min_area_ratio * h * w and cv2.isContourConvex(approx):
            # Create mask for quadrilateral
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(mask, [approx], -1, 255, -1)

            # Compute mean HSV values within quadrilateral
            mean_h, mean_s, mean_v = cv2.mean(hsv, mask=mask)[:3]

            # Condition: bright and low saturation (white)
            if mean_s < max_saturation and mean_v > min_value:
                valid_shapes.append(approx)

    return valid_shapes
