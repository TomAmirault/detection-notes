'''This module provides functions to detect a sheet of paper on an image, segment it, and correct its perspective.'''
import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
from perspective_corrector import corrected_perspective
from datetime import datetime

BASE_DIR = os.path.abspath(os.path.dirname(__file__))


def get_mask(img : np.ndarray, rect : tuple) -> np.ndarray :
    """Computes the binary mask of the sheet of paper in the image using the GrabCut algorithm.

    Args:
        img (np.ndarray): frame shot by the camera
        rect (tuple): tuple représentant la bounding box de la feuille au format (x_left, y_top, width, height)

    Returns:
        np.ndarray: masque binaire de la taille de l'image, démarquant la feuille de l'arrière-plan
    """     
    mask = np.zeros(img.shape[:2],np.uint8)
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    return mask2


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
    # stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    # cv2.imwrite(f"/Users/noahparisse/Documents/Paris-digital-lab/P1 RTE/detection-notes/tmp/paper/roi_{stamp}.jpg", roi)
    rect = (1,1,w-2,h-2)
    mask = get_mask(roi, rect)

    # mask_color = np.zeros_like(roi)
    # mask_color[mask>0]=np.array([0, 0, 255])
    # overlay = np.zeros_like(roi)
    # overlay = cv2.addWeighted(roi, 1.0, mask_color, 0.5, 0, overlay)
    # cv2.imwrite(f"/Users/noahparisse/Documents/Paris-digital-lab/P1 RTE/detection-notes/tmp/paper/overlay_{stamp}.jpg", overlay)

    # mask : tableau 2D numpy avec 0=background, 1=objet
    points = np.column_stack(np.where(mask > 0))  # shape = (N, 2), (y, x)

    # cv2.minAreaRect attend (x, y) pour chaque point
    rect = cv2.minAreaRect(points[:, ::-1])  # inverser y, x → x, y
    # rect = ((center_x, center_y), (width, height), angle)

    box = cv2.boxPoints(rect)   # renvoie 4 points float (x, y)
    box = np.int32(box)
    # roiplusrect = roi.copy()
    # cv2.drawContours(roiplusrect,[box],0,(0,0,255),2)
    # cv2.imwrite(f"/Users/noahparisse/Documents/Paris-digital-lab/P1 RTE/detection-notes/tmp/paper/rect_{stamp}.jpg", roiplusrect)

    return corrected_perspective(roi, box)


def search_and_print_object(img, rect):
    mask = np.zeros(img.shape[:2],np.uint8)
    # height, width = img.shape[:2]
    # center_x = int(width / 2)
    # center_y = int(height / 2)
    # mask[center_x, center_y] = 1
    # mask[center_x+1, center_y+1] = 1
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 1, cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img = img*mask2[:,:,np.newaxis]
    # cv2.imwrite('/Users/noahparisse/Downloads/photoqrcode.jpg', img)

    plt.imshow(img)
    plt.colorbar()
    plt.show()

if __name__ == "__main__":
    files = os.listdir(os.path.join(BASE_DIR, "../../../tmp/"))
    tmp_dir = os.path.join(BASE_DIR, "../../../tmp/")

    for f in files:
        if f.endswith(".jpg"):
            print("L'image traitée est :", f)
            img = cv2.imread(os.path.join(tmp_dir, f))
            height, width = img.shape[:2]
            stamp = f[6:-4]
            textfilename = os.path.join(tmp_dir, f"output_{stamp}.txt")
            with open(textfilename, "r") as t:
                for ligne in t:
                    valeurs = ligne.strip().split()  # découpe la ligne
                    classe, x, y, w, h = valeurs
                    break
            box_x_center = int(float(x)*width)
            box_y_center = int(float(y)*height)
            box_width = int(float(w)*width)
            box_height = int(float(h)*height)
            box_x_left = int(box_x_center-0.5*box_width)
            box_y_top = int(box_y_center-0.5*box_height)
            rect = (box_x_left, box_y_top, box_width, box_height)
            # search_and_print_object(img, rect)
            
            corrected = crop_image_around_object(img, rect)
            cv2.imshow("PErspectivé", corrected)
            cv2.waitKey(0)
            cv2.destroyAllWindows()