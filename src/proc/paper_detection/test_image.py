import cv2
from shape_detector import shape_detector
from save_detection import save_detection
from image_preprocessing import preprocessed_image
import numpy as np

# Charger une image depuis un fichier
image_path = "src/proc/paper_detection/test_image.jpg"  # mets ton chemin ici
img = cv2.imread(image_path)

if img is None:
    raise FileNotFoundError(f"Impossible de lire l'image : {image_path}")

# Détection
possible_papers = shape_detector(img)
img_show = img.copy()
img_show = preprocessed_image(img)  # si tu veux appliquer ton prétraitement

img_show = cv2.Canny(img_show, 75,  200)

# Dessiner les contours
cv2.drawContours(img_show, possible_papers, -1, (0, 255, 0), 2)

# Screenshot lorsqu'il y a détection
if len(possible_papers) > 0:
    print("✅ Note détectée")
    save_detection(img, possible_papers)
else:
    print("❌ Aucune note détectée")

# Affichage

screen_res = 1280, 720  # par ex. si ton écran est 1280x720
scale_width = screen_res[0] / img_show.shape[1]
scale_height = screen_res[1] / img_show.shape[0]
scale = min(scale_width, scale_height)

# redimensionne l'image
window_width = int(img_show.shape[1] * scale)
window_height = int(img_show.shape[0] * scale)
resized = cv2.resize(img_show, (window_width, window_height))

cv2.imshow("Image", resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
