import cv2
from shape_detector import shape_detector
from save_detection import save_detection
from image_preprocessing import preprocessed_image
import numpy as np
import os

# Dossier contenant les images de test
test_folder = "src/proc/paper_detection/test_set"

# Compteurs globaux
total_images = 0
images_with_detection = 0
total_notes = 0

# Parcours de toutes les images du dossier
for filename in os.listdir(test_folder):
    if not (filename.lower().endswith(".jpg") or filename.lower().endswith(".png")):
        continue  # ignore les fichiers non-images

    image_path = os.path.join(test_folder, filename)
    img = cv2.imread(image_path)

    if img is None:
        print(f"❌ Impossible de lire l'image : {filename}")
        continue

    total_images += 1

    print(f"\n--- Test sur {filename} ---")

    # Détection
    possible_papers = shape_detector(img)
    nb_detected = len(possible_papers)
    total_notes += nb_detected

    # Résultats
    if nb_detected > 0:
        print(f"✅ {nb_detected} note(s) détectée(s)")
        images_with_detection += 1
        save_detection(img, possible_papers)
    else:
        print("❌ Aucune note détectée")

# Résumé global
print("\n=== Résumé des tests ===")
print(f"Images traitées : {total_images}")
print(f"Images avec détection : {images_with_detection}")
print(f"Nombre total de notes détectées : {total_notes}")
