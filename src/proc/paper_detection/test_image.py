import cv2
from shape_detector import shape_detector
from save_detection import save_detection
from image_preprocessing import preprocessed_image

# Charger une image depuis un fichier
image_path = "src/proc/paper_detection/test_image.jpg"  # mets ton chemin ici
img = cv2.imread(image_path)

if img is None:
    raise FileNotFoundError(f"Impossible de lire l'image : {image_path}")

# Détection
possible_papers = shape_detector(img)
img_show = img.copy()
# img_show = preprocessed_image(img)  # si tu veux appliquer ton prétraitement

# Dessiner les contours
cv2.drawContours(img_show, possible_papers, -1, (0, 255, 0), 2)

# Screenshot lorsqu'il y a détection
if len(possible_papers) > 0:
    print("✅ Note détectée")
    save_detection(img, possible_papers)
else:
    print("❌ Aucune note détectée")

# Affichage
cv2.imshow("Image", img_show)
cv2.waitKey(0)
cv2.destroyAllWindows()
