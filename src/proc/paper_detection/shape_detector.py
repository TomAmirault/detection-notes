import cv2
from image_preprocessing import processed_image

# Détection des contours et filtrage des quadrilatères


def detect_paper(img):
    proc = processed_image(img)
    # Détection des bords
    edges = cv2.Canny(proc, 50, 150)
    # Trouver des contours
    contours, _ = cv2.findContours(
        edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for cnt in contours:
        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)

        # On ne garde que les quadrilatères assez grands
        if len(approx) == 4 and cv2.contourArea(cnt) > 1000 and cv2.isContourConvex(approx):
            candidates.append(approx)
    return candidates
