"""
Handles the detection and saving of paper-like quadrilaterals
"""

import os
from perspective_corrector_raspberrypi import corrected_perspective
import cv2
import time
import numpy as np
from datetime import datetime


# Output settings
OUT_DIR = "/home/projetrte/Documents/photos"
os.makedirs(OUT_DIR, exist_ok=True)

# Minimum cooldown between two saves
COOLDOWN_SEC = 5.0   
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

        corrected_path = os.path.join(
            OUT_DIR, f"{prefix}_q{i}.jpg"
        )
        cv2.imwrite(corrected_path, corrected)

    last_save_time = now
