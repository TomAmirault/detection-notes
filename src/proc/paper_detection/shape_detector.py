import cv2
import numpy as np
from image_preprocessing import preprocessed_image

# Détection des contours et filtrage des quadrilatères


def shape_detector(img):
    proc = preprocessed_image(img)
    # Binarisation pour trouver les contours
    edges = cv2.Canny(proc, 75, 200)

    # Extraction des contours
    contours, _ = cv2.findContours(
        edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    valid_shapes = []
    h, w = img.shape[:2]

    for cnt in contours:
        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)

        # On ne garde que les quadrilatères assez grands
        if len(approx) == 4 and cv2.contourArea(cnt) > 0.05 * h * w and cv2.isContourConvex(approx):
            valid_shapes.append(approx)

    return valid_shapes
