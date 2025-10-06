"""
Ce script utilise un modèle YOLOv8 (finetuné sur un dataset de détection de feuilles de papier) 
pour capturer en continu le flux de la webcam. 

Fonctionnement :
    1. YOLO détecte une feuille de papier en temps réel.
    2. Dès la première détection, le script enregistre les 20 frames suivantes.
    3. Parmi ces frames, la moins floue est sélectionnée grâce à `less_blurred`.
    4. La frame sélectionnée est sauvegardée, avec sa version rognée/rectifiée via `save_masked_image`.
    5. Le chemin de l’image sauvegardée est inséré dans la base de données via `add_data2db`.

Arrêt du programme :
    - Ctrl+C (KeyboardInterrupt)
"""

import os
import sys
import time
from datetime import datetime
from ultralytics import YOLO
from save_image import save_masked_image

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
)
from src.add_data2db import add_data2db
from proc.paper_detection.blurry_detection import less_blurred



# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Modèle YOLOv8 finetuné sur dataset RoboFlow
MODEL_PATH = os.path.join(BASE_DIR, "../detection_model/Yolo-seg.pt")
model = YOLO(MODEL_PATH)

# Timer (pour gérer les délais entre captures)
start = time.time()
checkpoint = start

# Liste temporaire pour stocker les frames détectées
video = []


# --- Boucle principale ---
try:
    for result in model.predict(source=0, show=True, conf=0.8, verbose=False, stream=True):
        boxes = result.boxes

        if boxes and len(boxes) > 0:
            # Un objet (feuille) est détecté
            if len(video) == 0:  # Début d'une séquence
                if time.time() - checkpoint > 10:  # Attente d’1s entre deux captures
                    checkpoint = time.time()
                    video.append(result)

            elif len(video) < 5:  # Capture en cours
                video.append(result)

            elif len(video) == 5:  # Séquence terminée
                best = less_blurred(video)  # Sélection de la frame la plus nette
                stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

                save_dir = os.path.join(BASE_DIR, "../../../tmp")
                final_filename = save_masked_image(video[best], save_dir, stamp)

                add_data2db(final_filename)  # Sauvegarde dans la base
                video = []  # Réinitialisation

        elif len(video) > 0:
            # L’objet a disparu avant la fin des 20 frames
            best = less_blurred(video)
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

            save_dir = os.path.join(BASE_DIR, "../../../tmp")
            final_filename = save_masked_image(video[best], save_dir, stamp)

            add_data2db(final_filename)

            video = []  # Réinitialisation

except KeyboardInterrupt:
    print("Arrêt demandé par l'utilisateur.")
finally:
    print("Fin.")
