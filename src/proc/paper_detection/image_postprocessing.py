import cv2
import numpy as np


def postprocessed_image(img):
    '''
    Posttraitement de l'image pour améliorer la qualité du changement de perspective
    '''
    # Application d'un unsharp masking pour corriger le flou
    blurred = cv2.GaussianBlur(img, (5, 5), 1.5)
    unsharp = cv2.addWeighted(img, 2.0, blurred, -1.0, 0)

    # Amélioration du contraste
    gray = cv2.cvtColor(unsharp, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(gray)

    return enhanced
