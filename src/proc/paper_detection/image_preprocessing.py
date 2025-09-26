import cv2
import numpy as np


def preprocessed_image(img):
    '''
    Prétraitement de l'image pour faciliter la détection de contours
    '''
    # Conversion en niveaux de gris
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Amélioration du contraste
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    gray = clahe.apply(gray)

    # Réduction du bruit
    denoised = cv2.GaussianBlur(gray, (5, 5), 0)

   # Filtre morphologique : supprime les petits bruits et isole mieux les contours
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    processed = cv2.morphologyEx(denoised, cv2.MORPH_OPEN, kernel)

    return processed
