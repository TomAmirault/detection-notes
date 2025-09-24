import cv2
import numpy as np

# Prétraitement de l'image pour faciliter la détection de contours


def processed_image(img):
    # Masque pour isoler les zones blanches
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sensitivity = 100
    lower_white = np.array([0, 0, 255-sensitivity])
    upper_white = np.array([255, sensitivity, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)

    # Appliquer le masque à l'image en niveaux de gris
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    masked_gray = cv2.bitwise_and(gray_img, gray_img, mask=mask)

    # Améliorer le contraste
    masked_gray = cv2.equalizeHist(masked_gray)

    # Réduire le bruit avec un filtre bilatéral
    blur = cv2.bilateralFilter(masked_gray, 9, 75, 75)

    # Dilatation pour agrandir les zones blanches
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(blur, kernel)

    return dilated
