from ultralytics import YOLO
import os, sys
import numpy as np
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from segmentation import get_mask
import cv2
import matplotlib.pyplot as plt

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Charge ton modèle finetuné
model = YOLO(os.path.join(BASE_DIR, "../../detection_model/best-detect.pt"))

# metrics = model.val(data=os.path.join(BASE_DIR, "../../../datasets_yolo/blank-sheet-segmentation/data.yaml"), split="test")
# print(metrics)

test_set_dir = os.path.join(BASE_DIR, "./data/set/images")
results = model.predict(source = test_set_dir)

# Données pour l'affichage des masques de segmentation sur les images
image_index = 0
window_index = 1
max_col = 4
max_row = 4
fig, axes = plt.subplots(4, 4, figsize=(5 * max_col, 5 * max_row))
axes = axes.flatten()

for result in results:
    mask_sum = np.zeros(result.orig_img.shape[:2],np.uint8)
    for i in range(result.boxes.shape[0]):
        x, y, width, height = result.boxes.xywh[i]
        box_x_center, box_y_center, width, height = int(x), int(y), int(width), int(height)
        box_x_left = int(box_x_center-0.5*width)
        box_y_top = int(box_y_center-0.5*height)
        rect = (box_x_left, box_y_top, width, height)
        mask = get_mask(result.orig_img, rect)
        mask_sum += mask
    mask_sum = np.where((mask_sum>=1),1,0).astype('uint8')
    image_name = os.path.basename(result.path)
    np.save(os.path.join(BASE_DIR, "./data/yolo-detect/", image_name[:-4]+".npy"), mask_sum)

    img_display = cv2.resize(result.orig_img, (mask_sum.shape[1], mask_sum.shape[0]))

    # Colorer le masque en bleu (BGR) puis convertir en RGB
    mask_color = np.zeros_like(img_display)
    mask_color[mask_sum > 0] = [0, 0, 255]  # rouge en BGR (donc bleu en RGB si cv2.imshow)
    overlay = cv2.addWeighted(img_display, 1.0, mask_color, 0.5, 0)

    # Afficher
    axes[image_index].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axes[image_index].set_title(f"Image {image_name}")
    axes[image_index].axis("off")
    image_index+=1
    if image_index==max_col*max_row:
        plt.tight_layout()
        plt.savefig(os.path.join(BASE_DIR, f"./data/yolo-detect/display/pred_yolo-detect_{window_index}.png"), dpi=300)
        image_index = 0
        window_index+=1
        fig, axes = plt.subplots(4, 4, figsize=(5 * max_col, 5 * max_row))
        axes = axes.flatten()
if image_index>0:
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, f"./data/yolo-detect/display/pred_yolo-detect_{window_index}.png"), dpi=300)