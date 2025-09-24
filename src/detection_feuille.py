import cv2
import numpy as np
import time

# Timer
start = time.time()

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)


# Prétraitement de l'image pour faciliter la détection de contours

def processed_image(img):
    # Convertir en niveaux de gris
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Flou léger pour réduire le bruit
    blur = cv2.GaussianBlur(gray_img, (5, 5), 0)
    return blur


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
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # On ne garde que les quadrilatères assez grands
        if len(approx) == 4 and cv2.contourArea(cnt) > 1000:
            candidates.append(approx)
    return candidates


# Lancement de la webcam
while True:
    ret, img = cap.read()
    possible_papers = detect_paper(img)
    img_show = img.copy()

    # Dessiner les contours sur l'image originale
    cv2.drawContours(img_show, possible_papers, -1, (0, 255, 0), 2)
    cv2.imshow('Webcam', img_show)

    # if cv2.waitKey(1) == ord('q'):
    #    break

    if time.time() - start > 30:
        break

cap.release()
cv2.destroyAllWindows()
