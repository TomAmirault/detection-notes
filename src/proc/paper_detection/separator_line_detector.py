import cv2
import numpy as np


# Affichage


def resized(img):
    screen_res = 1280, 720
    scale_width = screen_res[0] / img.shape[1]
    scale_height = screen_res[1] / img.shape[0]
    scale = min(scale_width, scale_height)

    # redimensionne l'image
    window_width = int(img.shape[1] * scale)
    window_height = int(img.shape[0] * scale)
    return cv2.resize(img, (window_width, window_height))


# Loading file
path = "src/proc/paper_detection/test_image.jpg"
img = cv2.imread(path)
h, w = img.shape[:2]
cv2.imshow("Test image", resized(img))
cv2.waitKey(0)
cv2.destroyAllWindows()


# Preprocessing
# Conversion en niveaux de gris
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Débruitage léger
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Détection des contours
edges = cv2.Canny(blurred, 200, 200)

# Ferme les trous
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

# Epaissit la ligne
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
thick = cv2.dilate(edges, kernel, iterations=2)

cv2.imshow("Edges image", resized(thick))
cv2.waitKey(0)
cv2.destroyAllWindows()

# Détection des lignes par Hough
lines = cv2.HoughLinesP(thick,
                        rho=1,                # résolution en pixels
                        theta=np.pi / 180,    # résolution en radians
                        threshold=30,        # nb min d’intersections dans l’espace Hough
                        # ligne d'au moins la moitié de la largeur
                        minLineLength=gray.shape[1] // 2,
                        maxLineGap=20    # tolérance de "trous" dans la ligne
                        )

output = img.copy()

if lines is not None:
    max_length = 0
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = np.linalg.norm([x2 - x1, y2 - y1])

        margin = 200  # pixels à ignorer près des bords
        if y1 < margin or y2 < margin or y1 > h - margin or y2 > h - margin:
            continue  # ligne trop proche du haut/bas

        cv2.line(output, (x1, y1), (x2, y2), (255, 0, 0), 2)
        # Ligne horizontale la plus grande
        if abs(y2 - y1) < 20 and length >= max_length:
            max_length = length
            best_line = (x1, y1, x2, y2)

if best_line is not None:
    x1, y1, x2, y2 = best_line
    m = (y2 - y1) / (x2 - x1 + 1e-6)  # pente
    b = y1 - m * x1                   # intercept
    # intersections avec bord gauche (x=0) et bord droit (x=w-1)
    y_left = int(b)
    y_right = int(m * (w-1) + b)
    # SI UNE SEULE LIGNE
    y_cut = int((y1 + y2) / 2)
    # Découpage
    top_part = img[:y_cut, :]
    bottom_part = img[y_cut:, :]
    cv2.line(output, (0, y_left), (w-1, y_right), (0, 255, 0), 2)


cv2.imshow("Final image", resized(output))
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow("Final top image", resized(top_part))
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow("Final bottom image", resized(bottom_part))
cv2.waitKey(0)
cv2.destroyAllWindows()
