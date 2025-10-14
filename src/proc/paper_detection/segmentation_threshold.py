'''Ce script définit toutes les fonctions nécessaires à la segmentation de l'image entre la feuille de papier détectée
l'arrière-plan.'''
import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
from perspective_corrector import corrected_perspective
from datetime import datetime

BASE_DIR = os.path.abspath(os.path.dirname(__file__))


def get_mask(img : np.ndarray) -> np.ndarray :  
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(img_gray, 100, 255, type = cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Meilleur enchainement : morph_close suivi de morph_open
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21,21))
    mask1 = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask1 = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask1)
    if num_labels>1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    else :
        largest_label = 0
    largest_mask = np.zeros_like(mask, dtype=np.uint8)
    largest_mask[labels == largest_label] = 255

    return largest_mask

def get_extreme_points(mask):
    h, w = mask.shape[:2]
    corners_ref = np.array([
        [0, 0],      # haut gauche
        [w-1, 0],    # haut droite
        [w-1, h-1],  # bas droite
        [0, h-1]     # bas gauche
    ])
    extreme_points = []
    ys, xs = np.nonzero(mask)
    points = np.column_stack((xs, ys)) 
    for corner in corners_ref:
        distances = np.linalg.norm(points - corner, axis=1)
        closest_idx = np.argmin(distances)
        extreme_points.append(points[closest_idx])
    extreme_points = np.array(extreme_points)
    return extreme_points

def test_segmentation(img : np.ndarray) -> np.ndarray :  
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(img_gray, 100, 255, type = cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    fig, axes = plt.subplots(3, 4, figsize=(10, 10))
    axes[0, 0].imshow(img_gray, cmap = 'gray')
    axes[0, 0].set_title("Img_gray")
    axes[0, 1].imshow(mask, cmap = 'gray')
    axes[0, 1].set_title("Seuillé")
    # Meilleur enchainement : morph_close suivi de morph_open
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21,21))
    mask1 = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask1 = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    axes[0, 2].imshow(mask1, cmap = 'gray')
    axes[0, 2].set_title("morph_close suivi de morph_open")
    # kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    # mask2 = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel2)
    # mask2 = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel2)
    # axes[1, 0].imshow(mask2, cmap = 'gray')
    # axes[1, 0].set_title("morph_open suivi de morph_close")
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask1)
    if num_labels>1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    else :
        largest_label = 0
    largest_mask = np.zeros_like(mask, dtype=np.uint8)
    largest_mask[labels == largest_label] = 255
    axes[0, 3].imshow(largest_mask, cmap = 'gray')
    axes[0, 3].set_title("plus grande CC du mask")
    masked_image = img.copy()
    masked_image[largest_mask==0]=[255, 255, 255]
    axes[1, 0].imshow(masked_image)
    axes[1, 0].set_title("Image masquée")

    # On fait tourner minAreaRect pour pouvoir croper
    points = np.column_stack(np.where(largest_mask > 0))
    rect = cv2.minAreaRect(points[:, ::-1]) # inverser y, x → x, y
    box = cv2.boxPoints(rect)
    box = np.int32(box)
    mask_rect = largest_mask.copy()
    mask_rect = cv2.cvtColor(mask_rect, cv2.COLOR_GRAY2RGB)
    cv2.drawContours(mask_rect,[box],0,(255,0,0),2)
    axes[1, 1].imshow(mask_rect)
    axes[1, 1].set_title("rectangle sur mask")
    img_rect = img.copy()
    img_rect = cv2.cvtColor(img_rect, cv2.COLOR_BGR2RGB)
    cv2.drawContours(img_rect,[box],0,(255,0,0),2)
    axes[1, 2].imshow(img_rect)
    axes[1, 2].set_title("rectangle")

    # On utilise le rectangle pour croper et remettre vertical
    corrected_image = corrected_perspective(img, box)
    axes[1, 3].imshow(corrected_image)
    axes[1, 3].set_title("Image corrigée")
    masked_corrected_img = corrected_perspective(masked_image, box)
    axes[2, 0].imshow(masked_corrected_img)
    axes[2, 0].set_title("Image masquée et corrigée")

    # On va chercher les 4 points extrêmes du masque pour affiner l'application de corrected_perspective.
    corrected_mask = corrected_perspective(largest_mask, box)
    extreme_points = get_extreme_points(corrected_mask)
    print("longueur de extreme_points :", len(extreme_points))
    image_with_extreme_points = corrected_image.copy()
    for point in extreme_points:
        cv2.circle(image_with_extreme_points, point, radius=5, color=(0, 0, 255), thickness=-1)
    axes[2, 1].imshow(image_with_extreme_points)
    axes[2, 1].set_title("Image corrigée avec les points extrêmes du masque")


    plt.show()
    return largest_mask


def crop_image_around_object(img:np.ndarray, rect:tuple) -> np.ndarray:
    """crop_image_around_object fait tourner l'algorithme grabCut sur l'image et rogne ensuite l'image autour de la feuille,
    en redressant la feuille à la verticale.

    Args:
        img (np.ndarray): frame captée par la caméra
        rect (tuple): tuple représentant la bounding box de l'objet au format (x_left, y_top, width, height)

    Returns:
        np.ndarray: l'image rognée autour de la feuille
    """    
    x, y, w, h = rect
    roi = img[y:y+h, x:x+w]
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    cv2.imwrite(f"/Users/noahparisse/Documents/Paris-digital-lab/P1 RTE/detection-notes/tmp/paper/roi_{stamp}.jpg", roi)
    rect = (1,1,w-2,h-2)
    mask = get_mask(roi)

    mask_color = np.zeros_like(roi)
    mask_color[mask>0]=np.array([0, 0, 255])
    overlay = np.zeros_like(roi)
    overlay = cv2.addWeighted(roi, 1.0, mask_color, 0.5, 0, overlay)
    cv2.imwrite(f"/Users/noahparisse/Documents/Paris-digital-lab/P1 RTE/detection-notes/tmp/paper/overlay_{stamp}.jpg", overlay)

    points = np.column_stack(np.where(mask > 0))
    rect = cv2.minAreaRect(points[:, ::-1])
    box = cv2.boxPoints(rect)
    box = np.int32(box)
    mask_rect = mask.copy()
    mask_rect = cv2.cvtColor(mask_rect, cv2.COLOR_GRAY2RGB)
    cv2.drawContours(mask_rect,[box],0,(255,0,0),2)
    img_rect = img.copy()
    img_rect = cv2.cvtColor(img_rect, cv2.COLOR_BGR2RGB)
    cv2.drawContours(img_rect,[box],0,(255,0,0),2)
    return corrected_perspective(roi, box)

    # hull = cv2.convexHull(points[:, ::-1])  # inverser (y,x) → (x,y)
    # epsilon = 0.02 * cv2.arcLength(hull, True)  # tolérance d’approximation
    # approx = cv2.approxPolyDP(hull, epsilon, True)

    # epsilon_add = 0.02*cv2.arcLength(hull, True)
    # max_iter = 10
    # i = 1
    # while len(approx) != 4 and i<max_iter:

    #     if len(approx>4):
    #         epsilon += epsilon_add
    #         epsilon_add *= 9/10
    #         approx = cv2.approxPolyDP(hull, epsilon, True)
    #     if len(approx<4):
    #         epsilon -= epsilon_add
    #         epsilon_add *= 9/10
    #         approx = cv2.approxPolyDP(hull, epsilon, True)
    #     i+=1
    # if len(approx)!=4:
    #     print("Erreur : l'objet n'a pas d'approximation quadrilatérale.")
    #     return img

    # img_copy = img.copy()

    # cv2.polylines(img_copy, [quadri_points.astype(np.int32)], True, (0, 0, 255), 2)

    # cv2.imshow("Quadrilatère", img_copy)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # box = np.int32(box)          # convertir en int pour tracer

    # # img_copy = img.copy()
    # # cv2.drawContours(img_copy, [box], 0, (0, 0, 255), thickness = 2)  # rouge, épaisseur 2
    # # cv2.imshow("Résultat", img_copy)
    # # cv2.waitKey(0)

if __name__ == "__main__":
    tmp_dir = os.path.join(BASE_DIR, "../../../tmp/")
    files = os.listdir(tmp_dir)

    for f in files:
        if f.endswith(".jpg"):
            print("L'image traitée est :", f)
            img = cv2.imread(os.path.join(tmp_dir, f))
        
            mask = test_segmentation(img)
            # cv2.imshow("PErspectivé", corrected)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()