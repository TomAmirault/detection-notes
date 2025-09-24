### Lit une image depuis un chemin ###
# Entrée : chemin fichier image (jpg/png) issu de la caméra ou d'un dossier
# Sortie : image en mémoire (numpy array)

import cv2

def load_image(path: str):
    """Retourne une image BGR (numpy array) depuis un chemin."""
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"Impossible de charger l'image depuis le chemin : {path}")
    return image



