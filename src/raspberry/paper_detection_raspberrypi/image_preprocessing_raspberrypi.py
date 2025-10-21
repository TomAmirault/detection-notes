# Ce fichier permet de prétraiter l'image pour faciliter la détection de contours par la suite
import cv2
import numpy as np


def preprocessed_image(img: np.ndarray) -> np.ndarray:
    """
    L'image est convertie en niveaux de gris, puis un filtrage bilatéral est appliqué
    pour réduire le bruit tout en préservant les contours.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.bilateralFilter(gray, d=5, sigmaColor=20, sigmaSpace=20)
    return denoised
