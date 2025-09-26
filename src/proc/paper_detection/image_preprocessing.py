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
        clipLimit=3.0, tileGridSize=(4, 4))
    gray = clahe.apply(gray)

    # Réduction du bruit (bilateral filtre = garde mieux les bords)
    denoised = cv2.bilateralFilter(gray, 9, 30, 50)

    # Filtrage morphologique
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # Ouverture pour supprimer petits bruits
    opened = cv2.morphologyEx(denoised, cv2.MORPH_OPEN, kernel)
    # Fermeture pour renforcer les contours
    processed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

    return processed
