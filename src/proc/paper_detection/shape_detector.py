# Détection des contours et filtrage: seuls les quadrilatères "blancs" sont gardés
import cv2
import numpy as np
from image_preprocessing import preprocessed_image


def shape_detector(img: np.ndarray) -> list:
    proc = preprocessed_image(img)
    h, w = img.shape[:2]

    # Binarisation pour trouver les contours
    edges = cv2.Canny(proc, 75, 200)

    # Fermer les trous dans les contours
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Extraction des contours
    contours, _ = cv2.findContours(
        edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    valid_shapes = []

    # Conversion en HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    for cnt in contours:
        # Enveloppe convexe
        hull = cv2.convexHull(cnt)

        # Approximation polygoniale
        perimeter = cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, 0.01 * perimeter, True)

        # On ne garde que les polygones pouvant être des feuilles
        if len(approx) == 4 and cv2.contourArea(cnt) > 0.05 * h * w and cv2.isContourConvex(approx):
            # Masque du quadrilatère
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(mask, [approx], -1, 255, -1)

            # Moyenne des canaux HSV dans la zone
            mean_h, mean_s, mean_v = cv2.mean(hsv, mask=mask)[:3]

            # Condition : zone claire et peu saturée (donc blanche)
            if mean_s < 100 and mean_v > 175:
                valid_shapes.append(approx)

    return valid_shapes
