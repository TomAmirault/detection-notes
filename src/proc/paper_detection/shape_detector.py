import cv2
import numpy as np
from image_preprocessing import preprocessed_image

# Détection des contours et filtrage des quadrilatères


def shape_detector(img):
    proc = preprocessed_image(img)
    # Binarisation pour trouver les contours
    # Canny adaptatif (moins sensible aux variations d’éclairage)
    sigma = 0.33
    v = np.median(proc)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edges = cv2.Canny(proc, lower, upper)

    # Extraction des contours
    contours, _ = cv2.findContours(
        edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    valid_shapes = []
    h, w = img.shape[:2]

    for cnt in contours:
        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.1 * perimeter, True)

        # On ne garde que les quadrilatères assez grands
        if len(approx) == 4 and cv2.contourArea(cnt) > 0.05 * h * w and cv2.isContourConvex(approx):
            # Isoler en blanc le quadrilatère
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(mask, [approx], -1, 255, -1)

            # Convertir en HSV
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            # Moyennes dans le quadrilatère
            mean_H, mean_S, mean_V, _ = cv2.mean(hsv, mask=mask)

            if mean_V > 100 and mean_S < 50:
                # V = luminosité (haut = clair)
                # S = saturation (bas = gris/blanc)
                valid_shapes.append(approx)

    return valid_shapes
