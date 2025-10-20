import os
from perspective_corrector_raspberrypi import corrected_perspective
import cv2
import time
import numpy as np
from datetime import datetime



# Réglages de sauvegarde
OUT_DIR = "/home/projetrte/Documents/photos"
os.makedirs(OUT_DIR, exist_ok=True)

# Cooldown
COOLDOWN_SEC = 5.0   # délai mini entre deux sauvegardes (évite les doublons)

last_save_time = 0.0


def save_detection(frame, quads):
    """
    Sauvegarde le frame complet et celle dont la perspective a été modifiée
    - quads: liste de contours (4 points) renvoyés par la détection (approxPolyDP)
    """
    global last_save_time

    if not quads:
        return

    # Cooldown
    now = time.time()
    if now - last_save_time < COOLDOWN_SEC:
        return

    # Horodatage unique
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
    prefix = f"detection_{stamp}"

    # Sauvegarde de chaque quadrilatère
    for i, quad in enumerate(quads):
        corners = quad.reshape(4, 2).astype(np.float32)
        corrected = corrected_perspective(frame, corners)

        corrected_path = os.path.join(
            OUT_DIR, f"{prefix}_q{i}.jpg"
        )
        cv2.imwrite(corrected_path, corrected)

        # Ajout en base
        # add_data2db(corrected_path)

    last_save_time = now
