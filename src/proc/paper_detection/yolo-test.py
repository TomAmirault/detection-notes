from ultralytics import YOLO
import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Charge ton modèle finetuné
model = YOLO(os.path.join(BASE_DIR, "../detection_model/best-segment.pt"))

# Évalue-le sur le dataset de test
metrics = model.val(data=os.path.join(BASE_DIR, "../../../datasets_yolo/blank-sheet-segmentation/data.yaml"), split="test")

print(metrics)

# results = model.predict(
#     source="chemin/vers/images/test",  # dossier ou image unique
#     save=True,  # enregistre les images avec les prédictions dans runs/predict/exp
#     show=True   # affiche les images dans une fenêtre
# )