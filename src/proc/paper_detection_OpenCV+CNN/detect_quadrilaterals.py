import cv2
import numpy as np
import math
from typing import List



def order_points_clockwise(pts: np.ndarray) -> np.ndarray:
    """
    Réorganise quatre points en ordre horaire (clockwise) en partant du point en haut à gauche.

    Args:
        pts (np.ndarray): Tableau de 4 points sous forme (x, y).

    Returns:
        np.ndarray: Tableau de 4 points réorganisés dans l'ordre horaire :
                   [haut-gauche, haut-droit, bas-droit, bas-gauche].
    """
    rect = np.zeros((4,2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def angle_between(p0, p1, p2):
    """
    Calcule l'angle (en degrés) entre trois points, où le point central est le sommet de l'angle.

    Args:
        p0 (np.ndarray): Premier point.
        p1 (np.ndarray): Point central (sommet de l'angle).
        p2 (np.ndarray): Deuxième point.

    Returns:
        float: Angle en degrés entre les vecteurs p0-p1 et p2-p1.
    """
    v0 = p0 - p1
    v2 = p2 - p1
    dot = np.dot(v0, v2)
    n0 = np.linalg.norm(v0)
    n2 = np.linalg.norm(v2)
    if n0 * n2 == 0:
        return 0.0
    cosang = dot / (n0 * n2)
    cosang = np.clip(cosang, -1.0, 1.0)
    return abs(math.degrees(math.acos(cosang)))



def detect_quadrilaterals_single_scale(image_bgr: np.ndarray, scale: float=1.0, debug=False) -> List[np.ndarray]:
    """
    Détecte les quadrilatères dans une image à une échelle donnée, en utilisant plusieurs méthodes de détection de contours et de filtrage.

    Args:
        image_bgr (np.ndarray): Image d'entrée au format BGR.
        scale (float, optionnel): Facteur d'échelle pour redimensionner l'image avant traitement. Par défaut, 1.0.
        debug (bool, optionnel): Si True, affiche des informations de débogage. Par défaut, False.

    Returns:
        List[np.ndarray]: Liste des quadrilatères détectés, chacun représenté par 4 points ordonnés dans l'ordre horaire.
    """
    h0, w0 = image_bgr.shape[:2]
    if scale != 1.0:
        small = cv2.resize(image_bgr, (int(w0*scale), int(h0*scale)), interpolation=cv2.INTER_AREA)
    else:
        small = image_bgr.copy()
    h, w = small.shape[:2]
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blurred = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

    # Détection multi-sources
    v = np.median(blurred)
    sigma = 0.33
    lower = int(max(5, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edges_canny = cv2.Canny(blurred, lower, upper, apertureSize=3)
    thr = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 11, 2)
    gx = cv2.Sobel(blurred, cv2.CV_16S, 1, 0, ksize=3)
    gy = cv2.Sobel(blurred, cv2.CV_16S, 0, 1, ksize=3)
    mag = cv2.convertScaleAbs(cv2.addWeighted(cv2.convertScaleAbs(gx), 0.5, cv2.convertScaleAbs(gy), 0.5, 0))
    _, mag_thr = cv2.threshold(mag, 20, 255, cv2.THRESH_BINARY)

    # Fusion des cartes
    edges = cv2.bitwise_or(edges_canny, thr)
    edges = cv2.bitwise_or(edges, mag_thr)

    # Morphologie
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)
    edges = cv2.dilate(edges, kernel, iterations=1)

    # Extraction de contours
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    quads = []
    area_img = w * h
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < max(80, area_img * 0.0002):
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon=0.01*peri, closed=True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            pts = approx.reshape(4,2).astype('float32')
            pts_ordered = order_points_clockwise(pts)
            angles = [angle_between(pts_ordered[i-1], pts_ordered[i], pts_ordered[(i+1)%4]) for i in range(4)]
            if all(15 <= a <= 165 for a in angles):
                scale_back = 1.0/scale
                pts_orig = pts_ordered * scale_back
                quads.append(pts_orig)
                continue
        # fallback: minAreaRect
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.array(box, dtype='float32')
        box_area = rect[1][0] * rect[1][1]
        if box_area > 0 and (area / box_area) > 0.3 and box_area > max(300, area_img * 0.0005):
            box_ord = order_points_clockwise(box)
            box_orig = box_ord * (1.0/scale)
            quads.append(box_orig)
    return quads