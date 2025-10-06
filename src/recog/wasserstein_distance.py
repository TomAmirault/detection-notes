import numpy as np
import cv2
import matplotlib.pyplot as plt
import ot

# Charger et prétraiter une image avec OpenCV
def preprocess_image(img, size=(64, 64)):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # charger en niveaux de gris
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)  # redimensionner
    img = img.astype(np.float64)
    img = img / img.sum()  # normalisation en probabilité
    return img

# Calculer la distance de Wasserstein entre deux images
def wasserstein_distance(img1, img2):
    a = img1.flatten()
    b = img2.flatten()

    # Grille de coordonnées (pixels)
    n = img1.shape[0]
    coords = np.array([(i, j) for i in range(n) for j in range(n)])
    M = ot.dist(coords, coords, metric='euclidean')  # matrice des coûts
    M /= M.max()  # normalisation

    # Transport optimal (Earth Mover's Distance)
    distance = ot.emd2(a, b, M)
    return distance

def compare(sourcea, sourceb):
    img1 = preprocess_image(sourcea[0])
    img2 = preprocess_image(sourceb[0])

    d = wasserstein_distance(img1, img2)
    print(f"--------- Distance de Wasserstein entre les images {sourcea[1]} et {sourceb[1]}: {d:.4f}")

    # Visualisation
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(img1, cmap="gray")
    axes[0].set_title("Image 1")
    axes[1].imshow(img2, cmap="gray")
    axes[1].set_title("Image 2")
    plt.show()