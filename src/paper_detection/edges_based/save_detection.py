"""
Handles the detection and saving of paper-like quadrilaterals
"""

import sys, os
REPO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if REPO_PATH not in sys.path:
    sys.path.insert(0, REPO_PATH)
from src.paper_detection.edges_based.perspective_corrector import corrected_perspective
from src.processing.add_data2db import add_data2db
import cv2
import time
import numpy as np
from datetime import datetime


# Output settings
OUT_DIR = os.path.join(REPO_PATH, "src/paper_detection/tmp")
os.makedirs(OUT_DIR, exist_ok=True)

# Minimum cooldown between two saves
COOLDOWN_SEC = 20.0  
last_save_time = 0.0


def save_detection(img: np.ndarray, quads: list[np.ndarray]) -> None:
    """
    Save detected quadrilaterals from an image with perspective correction.

    Args:
        img (np.ndarray): Input image.
        quads (list[np.ndarray]): List of quadrilateral contours, each with 4 points, returned by shape_detector.py.

    Returns:
        None
    """
    global last_save_time

    if not quads:
        return

    # Cooldown check
    now = time.time()
    if now - last_save_time < COOLDOWN_SEC:
        return

    # Unique timestamp
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
    prefix = f"detection_{stamp}"

    # Save each quadrilateral
    for i, quad in enumerate(quads):
        corners = quad.reshape(4, 2).astype(np.float32)
        corrected = corrected_perspective(img, corners)
        
        corrected_path = os.path.join(OUT_DIR, f"{prefix}_q{i}.jpg")
        cv2.imwrite(corrected_path, corrected)

        # Add to database
        add_data2db(corrected_path)

    last_save_time = now
