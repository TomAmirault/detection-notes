import os, sys
REPO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if REPO_PATH not in sys.path:
    sys.path.insert(0, REPO_PATH)
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
from src.image_similarity.resize_minkowski_interpolation import minkowski_resize
from datetime import datetime
from logger_config import setup_logger, save_fig_with_limit
from pathlib import Path

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Affichage des graphes lors de l'exécution du script ?
print_graphs = False

# Hyperparamètres
topMatchesFactor = 0.2       # Sélectivité des matches entre keypoints
gray_threshold = 135           # entre 120 et 145
minkowski_mean_order = 4        # (best = 4) Ordre de la moyenne de Minkowski utilisée pour l'interpolation des pixels dans le redimensionnement des images
diff_threshold = 110     # 40 si Minkowski_mean_order = 2, 110 si = 4 (posible de mettre un peu plus en threshold si on veut être plus restrictif, mais risque de louper une petite modif), 180 si = 10
shape_of_diff = (40, 40)

def show2_with_cursor(img1, img2):
    # if not print_graphs:
    #     return 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    im = ax1.imshow(img1, cmap='gray')
    im = ax2.imshow(img2, cmap='gray')

    def format_coord1(x, y):
        col = int(x + 0.5)
        row = int(y + 0.5)
        if 0 <= col < img1.shape[1] and 0 <= row < img1.shape[0]:
            z = img1[row, col]
            if z.ndim == 0:
                return f"x={col}, y={row}, value={z:.3f}"
            else:
                return f"x={col}, y={row}, value=({z[0]:.3f}, {z[1]:.3f}, {z[2]:.3f})"
        else:
            return f"x={col}, y={row}"
        
    def format_coord2(x, y):
        col = int(x + 0.5)
        row = int(y + 0.5)
        if 0 <= col < img2.shape[1] and 0 <= row < img2.shape[0]:
            z = img2[row, col]
            if z.ndim == 0:
                return f"x={col}, y={row}, value={z:.3f}"
            else:
                return f"x={col}, y={row}, value=({z[0]:.3f}, {z[1]:.3f}, {z[2]:.3f})"
        else:
            return f"x={col}, y={row}"

    ax1.set_title("Image1")
    ax2.set_title("Image2")
    ax1.format_coord = format_coord1
    ax2.format_coord = format_coord1
    plt.show()

def save_comparison(matches_drawn, img2_aligned, overlay, gray1, gray2, gray1_mink, gray2_mink, diff, isSimilar):

    fig, axes = plt.subplots(2, 4, figsize=(8, 8))

    axes[0, 0].imshow(matches_drawn)
    axes[0, 0].set_title('Matches')
    axes[0, 1].imshow(img2_aligned)
    axes[0, 1].set_title('New image')
    axes[0, 2].imshow(overlay)
    axes[0, 2].set_title('Overlay of both')

    im1 = axes[0, 3].imshow(gray1, cmap='gray')
    axes[0, 3].set_title('Thresholded old image')
    im2 = axes[1, 0].imshow(gray2, cmap='gray')
    axes[1, 0].set_title('Thres. new image')
    im3 = axes[1, 1].imshow(gray1_mink, cmap='gray')
    axes[1, 1].set_title('Preprocessed old image')
    im4 = axes[1, 2].imshow(gray2_mink, cmap='gray')
    axes[1, 2].set_title('Prepr. new image')

    imd = axes[1, 3].imshow(diff, cmap='gray')
    axes[1, 3].set_title('Difference')

    fig.colorbar(im1, ax=axes[0, 3], orientation='vertical')
    fig.colorbar(im2, ax=axes[1, 0], orientation='vertical')
    fig.colorbar(im3, ax=axes[1, 1], orientation='vertical')
    fig.colorbar(im4, ax=axes[1, 2], orientation='vertical')
    fig.colorbar(imd, ax=axes[1, 3], orientation='vertical')


    if isSimilar:
        fig.suptitle("[Comparaison visuelle] Image similaire trouvée")
    else:
        fig.suptitle("[Comparaison visuelle] Pas d'image similaire trouvée")

    stamp = f"{datetime.now():%Y%m%d-%H%M%S}-{datetime.now().microsecond//1000:03d}"
    file_name =f"logs/image-comparison/comparison_{stamp}.jpg"
    save_fig_with_limit(file_name, fig)
    plt.close(fig)



def isSimilar(old_img_path:Path, new_img_path:Path) -> bool:
    try:
        old_img = cv2.imread(old_img_path)
        new_img = cv2.imread(new_img_path)
        img1 = cv2.cvtColor(old_img, cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)

        # ---- ALIGNEMENT ----

        # - Détection des keypoints
        orb = cv2.ORB_create(nfeatures = 1000)
        keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
        keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

        if descriptors1 is None or descriptors2 is None:
            print("Pas assez de points d'intérêt pour comparer les images.")
            return False

        print(descriptors1.dtype,descriptors2.dtype)

        # - Matching des keypoints
        matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
        all_matches = list(matcher.match(descriptors1, descriptors2, None))
        sorted_matches = sorted(all_matches, key=lambda x: x.distance, reverse=False)
        numGoodMatches = int(len(sorted_matches) * topMatchesFactor)
        matches = sorted_matches[:numGoodMatches]

        # Draw first keypoints and matches
        matches_drawn = cv2.drawMatches(old_img,keypoints1,new_img,keypoints2,matches,None)

        # - Calcul de la transformation
        points1 = np.zeros((len(matches), 2), dtype = np.float32)
        points2 = np.zeros((len(matches), 2), dtype = np.float32)
        for i, match in enumerate(matches):
            points1[i, :] = keypoints1[match.queryIdx].pt
            points2[i, :] = keypoints2[match.trainIdx].pt
        h, mask = cv2.findHomography(points2, points1, cv2.RANSAC)
        height, width, channels = old_img.shape
        img2_reg = cv2.warpPerspective(new_img, h, (width, height), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
        # cv2.imshow("Image1", sourcea[0])
        # cv2.imshow("Image2", sourceb[0])  
        # cv2.imshow("Image2 realigned", img2_reg)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # - Superposition
        overlay = np.zeros_like(old_img)
        cv2.addWeighted(old_img, 0.5, img2_reg, 0.5, 0, overlay)

        # ---- DIFFERENCES ----

        color1 = old_img.copy()
        color2 = img2_reg
        gray1 = cv2.cvtColor(color1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(color2, cv2.COLOR_BGR2GRAY)
        # _, gray1 = cv2.threshold(gray1, gray_threshold, 255, cv2.THRESH_BINARY)
        # _, gray2 = cv2.threshold(gray2, gray_threshold, 255, cv2.THRESH_BINARY)

        # on inverse le blanc et le noir pour donner plus d'importance aux noirs lorsqu'on applique la moyenne de Minkowski
        # gray1_inv = cv2.bitwise_not(gray1)
        # gray2_inv = cv2.bitwise_not(gray2)
        gray1_inv = 255 - gray1
        gray2_inv = 255 - gray2
        h = min(gray1_inv.shape[0], gray2_inv.shape[0])
        w = min(gray1_inv.shape[1], gray2_inv.shape[1])
        gray1_mink = minkowski_resize(gray1_inv, shape_of_diff, minkowski_mean_order)
        gray2_mink = minkowski_resize(gray2_inv, shape_of_diff, minkowski_mean_order)

        diff = cv2.absdiff(gray1_mink, gray2_mink)
        diff_cut = (diff>diff_threshold) * diff
        answer = np.all(diff<=diff_threshold)
        print("Les 2 feuilles de papier sont identiques :", answer)

        # Pour le debug
        save_comparison(matches_drawn, img2_reg, overlay, gray1, gray2, gray1_mink, gray2_mink, diff_cut, answer)

        return answer.item()
    except Exception as e:
        print("Exception lors de l'exécution de isSimilar:", e)
        return False

if __name__ == '__main__':
    tmp_dir = os.path.join(REPO_PATH, "tmp/test-similarity")
    files = os.listdir(tmp_dir)
    files = [f for f in files if f.endswith('.jpg')]
    print("Longueur de files :", len(files))

    # im0_path = "/Users/noahparisse/Documents/Paris-digital-lab/P1 RTE/detection-notes/tmp/test-similarity/detection_20251022-102208-532_q0.jpg"
    im0_path = "/Users/noahparisse/Documents/Paris-digital-lab/P1 RTE/detection-notes/tmp/test-similarity/detection_20251022-101029-996_q1.jpg"

    for f in files:
        if f.endswith(".jpg"):
            print("L'image traitée est :", f)
            filename = os.path.join(tmp_dir, f)
            isSimilar(im0_path, filename)