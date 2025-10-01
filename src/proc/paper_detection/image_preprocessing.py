import cv2
import numpy as np


def preprocessed_image(img):
    '''
    Prétraitement de l'image pour faciliter la détection de contours
    '''
    # Conversion en niveaux de gris
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Réduction du bruit
    denoised = cv2.bilateralFilter(gray, d=5, sigmaColor=20, sigmaSpace=20)

    return denoised
