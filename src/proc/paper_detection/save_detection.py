import cv2
import os
import time
import numpy as np
from datetime import datetime
from perspective_corrector import corrected_perspective

# Réglages de sauvegarde
OUT_DIR = "src\proc\paper_detection\screenshots"
os.makedirs(OUT_DIR, exist_ok=True)

# Cooldown
COOLDOWN_SEC = 0.0   # délai mini entre deux sauvegardes (évite les doublons)

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

    # Sauvegarde du frame complet
    frame_path = os.path.join(OUT_DIR, f"{prefix}_frame.jpg")
    cv2.imwrite(frame_path, frame)

    # Sauvegarde de chaque quadrilatère
    for i, quad in enumerate(quads):
        corners = quad.reshape(4, 2).astype(np.float32)
        corrected = corrected_perspective(frame, corners)

        corrected_path = os.path.join(
            OUT_DIR, f"{prefix}_q{i}.jpg"
        )
        cv2.imwrite(corrected_path, corrected)

    print(f"[SAVE] {frame_path} (+ {len(quads)} corrected perspectives)")

    last_save_time = now
