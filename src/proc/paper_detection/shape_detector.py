# Détection des contours et filtrage: seuls les quadrilatères "blancs" sont gardés
import cv2
import numpy as np


# --- HYPERPARAMÈTRES ---
# Filtrage bilatéral
BILATERAL_D = 5             # Diamètre du voisinage pour le filtrage bilatéral
BILATERAL_SIGMA_COLOR = 20  # Lissage sur les différences de couleur
BILATERAL_SIGMA_SPACE = 20  # Lissage spatial (influence sur la distance entre pixels)

# Détection de contours (Canny)
CANNY_THRESHOLD1 = 75       # Seuil bas pour les contours
CANNY_THRESHOLD2 = 200      # Seuil haut pour les contours

# Morphologie (fermeture des contours)
MORPH_KERNEL_SIZE = 7        # Taille du noyau pour fermer les trous
MORPH_ITERATIONS = 1         # Nombre d'itérations de la fermeture

# Approximation polygonale
POLY_EPSILON_FACTOR = 0.01  # Fraction de la longueur du périmètre pour approxPolyDP

# Filtrage des quadrilatères
MIN_AREA_RATIO = 0.05        # Aire minimale relative à l'image pour garder le polygone

# Filtrage couleur (zones "blanches")
MAX_SATURATION = 100         # Saturation maximale pour considérer la zone blanche
MIN_VALUE = 175              # Luminosité minimale pour considérer la zone blanche



def preprocessed_image(img: np.ndarray, bilateral_d=BILATERAL_D, 
                       bilateral_sigma_color=BILATERAL_SIGMA_COLOR, bilateral_sigma_space=BILATERAL_SIGMA_SPACE) -> np.ndarray:
    """
    L'image est convertie en niveaux de gris, puis un filtrage bilatéral est appliqué
    pour réduire le bruit tout en préservant les contours.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.bilateralFilter(gray, bilateral_d, bilateral_sigma_color, bilateral_sigma_space)
    return denoised


def shape_detector(img: np.ndarray, bilateral_d=BILATERAL_D, 
                       bilateral_sigma_color=BILATERAL_SIGMA_COLOR, bilateral_sigma_space=BILATERAL_SIGMA_SPACE, 
                       canny_threshold1=CANNY_THRESHOLD1, canny_threshold2=CANNY_THRESHOLD2, 
                       morph_kernel_size=MORPH_KERNEL_SIZE, morph_iterations=MORPH_ITERATIONS, 
                       poly_epsilon_factor=POLY_EPSILON_FACTOR, min_area_ratio=MIN_AREA_RATIO, 
                       max_saturation=MAX_SATURATION, min_value=MIN_VALUE) -> list:
    proc = preprocessed_image(img, bilateral_d, bilateral_sigma_color, bilateral_sigma_space)
    h, w = img.shape[:2]

    # Binarisation pour trouver les contours
    edges = cv2.Canny(proc, canny_threshold1, canny_threshold2)

    # Fermer les trous dans les contours
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_kernel_size, morph_kernel_size))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=morph_iterations)

    # Extraction des contours
    contours, _ = cv2.findContours(
        edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    valid_shapes = []

    # Conversion en HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    for cnt in contours:
        # Enveloppe convexe
        hull = cv2.convexHull(cnt)

        # Approximation polygoniale
        perimeter = cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, poly_epsilon_factor * perimeter, True)

        # On ne garde que les polygones pouvant être des feuilles
        if len(approx) == 4 and cv2.contourArea(cnt) > min_area_ratio * h * w and cv2.isContourConvex(approx):
            # Masque du quadrilatère
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(mask, [approx], -1, 255, -1)

            # Moyenne des canaux HSV dans la zone
            mean_h, mean_s, mean_v = cv2.mean(hsv, mask=mask)[:3]

            # Condition : zone claire et peu saturée (donc blanche)
            if mean_s < max_saturation and mean_v > min_value:
                valid_shapes.append(approx)

    return valid_shapes

def get_mask(img):
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    possible_papers = shape_detector(img)
    if len(possible_papers) > 0:
        cv2.drawContours(mask, possible_papers, -1, 255, thickness=cv2.FILLED)

    return mask
