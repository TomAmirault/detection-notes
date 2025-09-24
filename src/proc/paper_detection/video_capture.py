import cv2
import time
import numpy as np
from shape_detector import detect_paper

# Timer
start = time.time()

cap = cv2.VideoCapture(0)

# Lancement de la webcam
while True:
    ret, img = cap.read()
    possible_papers = detect_paper(img)
    img_show = img.copy()

    # Dessiner les contours sur l'image originale
    cv2.drawContours(img_show, possible_papers, -1, (0, 255, 0), 2)
    cv2.imshow('Webcam', img_show)

    if cv2.waitKey(1) == ord('q'):
        break

    # Timer
    if time.time() - start > 30:
        break

cap.release()
cv2.destroyAllWindows()
