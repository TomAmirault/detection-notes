import cv2
import numpy as np


def preprocessed_image(img):
    '''
    Prétraitement de l'image pour faciliter la détection de contours
    '''
    # Conversion en niveaux de gris
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Amélioration du contraste adaptatif (CLAHE)
    clahe = cv2.createCLAHE(
        clipLimit=2.5, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Réduction du bruit
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Fermeture morphologique pour compléter les contours
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, kernel, iterations=5)

    return closed
