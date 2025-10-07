import numpy as np
import cv2
from matplotlib import pyplot as plt
import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

def search_and_print_object(img, rect):
    mask = np.zeros(img.shape[:2],np.uint8)
    # height, width = img.shape[:2]
    # center_x = int(width / 2)
    # center_y = int(height / 2)
    # mask[center_x, center_y] = 1
    # mask[center_x+1, center_y+1] = 1
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img = img*mask2[:,:,np.newaxis]
    cv2.imwrite('/Users/noahparisse/Downloads/photoqrcode.jpg', img)

    plt.imshow(img)
    plt.colorbar()
    plt.show()

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
        search_and_print_object(img, rect)