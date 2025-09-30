# Ce fichier crée la fonction qui prend une image et
# renvoie les bounding boxes des feuilles de papier trouvées

from ultralytics import YOLO
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Modèle YOLOv11 finetuné sur le dataset https://app.roboflow.com/dty-opi9m/detection-de-feuilles-245oo/1/export
model_path = os.path.join(BASE_DIR, '../detection_model/best.pt')
model = YOLO(model_path)

def look_at_picture(image_path):
    output = model(image_path)
    output[0].show()

# look_at_picture('/Users/noahparisse/Downloads/Detection-de-feuilles/test/images/IMG_20240605_161026_jpg.rf.77f2974e94c90c24f55588abe32c7082.jpg')
# look_at_picture('/Users/noahparisse/Downloads/Detection-de-feuilles/test/images/IMG_20250516_162655_jpg.rf.b194f5eced97f93797f65fac7b4be100.jpg')
# look_at_picture('/Users/noahparisse/Downloads/Detection-de-feuilles/test/images/-_202505131939205_jpg.rf.93ba05c681c312d466d06c6e1f04ca87.jpg')