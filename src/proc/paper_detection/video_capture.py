# Ajoute la racine du projet au sys.path pour permettre les imports internes
import sys
import os
REPO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if REPO_PATH not in sys.path:
    sys.path.insert(0, REPO_PATH)

import cv2
import time
from src.proc.paper_detection.shape_detector import shape_detector
from src.proc.paper_detection.save_detection import save_detection
from src.proc.paper_detection.image_preprocessing import preprocessed_image

# Timer
start = time.time()

# Choix de la caméra
cap = cv2.VideoCapture(1)

# Lancement de la webcam
while True:
    ret, img = cap.read()

    # Vérification que l'image est bien capturée
    if not ret or img is None:
        print("⚠️ Aucune image capturée. Vérifie la caméra ou l'index du périphérique.")
        continue  # on saute cette itération au lieu de planter

    possible_papers = shape_detector(img)

    # Pour éviter les erreurs si shape_detector renvoie vide
    if img is not None:
        img_show = img.copy()
    else:
        continue


    # Dessiner les contours sur l'image originale
    cv2.drawContours(img_show, possible_papers, -1, (0, 255, 0), 2)

    # Infos en overlay
    cv2.putText(
        img_show,
        f"Feuilles detectees: {len(possible_papers)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0) if len(possible_papers) > 0 else (0, 0, 255),
        2
    )

    # Screenshot lorsqu'il y a détection
    if len(possible_papers) > 0:
        save_detection(img, possible_papers)

    cv2.imshow('Webcam', img_show)

    if cv2.waitKey(1) == ord('q'):
        break

    # # Timer
    # if time.time() - start > 180:
    #     break

cap.release()
cv2.destroyAllWindows()
