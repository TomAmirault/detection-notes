import os, sys
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.paper_detection.edges_based.shape_detector import get_mask
import cv2
import matplotlib.pyplot as plt

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

test_set_dir = os.path.join(BASE_DIR, "./data/set/images")

# Données pour l'affichage des masques de segmentation sur les images
image_index = 0
window_index = 1
max_col = 4
max_row = 4
fig, axes = plt.subplots(4, 4, figsize=(5 * max_col, 5 * max_row))
axes = axes.flatten()

for img_name in os.listdir(test_set_dir):
    img = cv2.imread(os.path.join(test_set_dir, img_name))
    mask = get_mask(img)
    mask = np.where((mask==255),1,0).astype('uint8')
    np.save(os.path.join(BASE_DIR, "./data/alex-model/", img_name[:-4]+".npy"), mask)

    img_display = cv2.resize(img, (mask.shape[1], mask.shape[0]))
    # Colorer le masque en bleu (BGR) puis convertir en RGB
    mask_color = np.zeros_like(img_display)
    mask_color[mask > 0] = [0, 0, 255]  # rouge en BGR (donc bleu en RGB si cv2.imshow)
    overlay = cv2.addWeighted(img_display, 1.0, mask_color, 0.5, 0)
    # Afficher
    axes[image_index].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axes[image_index].set_title(f"Image {img_name}")
    axes[image_index].axis("off")
    image_index+=1

    if image_index==max_col*max_row:
        plt.tight_layout()
        plt.savefig(os.path.join(BASE_DIR, f"./data/alex-model/display/pred_alex-model_{window_index}.png"), dpi=300)
        image_index = 0
        window_index+=1
        fig, axes = plt.subplots(4, 4, figsize=(5 * max_col, 5 * max_row))
        axes = axes.flatten()

if image_index>0:
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, f"./data/alex-model/display/pred_alex-model_{window_index}.png"), dpi=300)