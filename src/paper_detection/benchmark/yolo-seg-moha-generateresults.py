from ultralytics import YOLO
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Charge ton modèle finetuné
model = YOLO(os.path.join(BASE_DIR, "../../detection_model/Yolo-seg.pt"))

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
    if result.masks == None :
        mask = np.zeros(result.orig_img.shape[:2])
    else :
        mask = result.masks.data.numpy()[0]
    image_name = os.path.basename(result.path)
    np.save(os.path.join(BASE_DIR, "./data/yolo-seg-moha/", image_name[:-4]+".npy"), mask)

    img_display = cv2.resize(result.orig_img, (mask.shape[1], mask.shape[0]))

    # Colorer le masque en bleu (BGR) puis convertir en RGB
    mask_color = np.zeros_like(img_display)
    mask_color[mask > 0] = [0, 0, 255]  # rouge en BGR (donc bleu en RGB si cv2.imshow)
    overlay = cv2.addWeighted(img_display, 1.0, mask_color, 0.5, 0)

    # Afficher
    axes[image_index].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axes[image_index].set_title(f"Image {image_name}")
    axes[image_index].axis("off")
    image_index+=1
    if image_index==max_col*max_row:
        plt.tight_layout()
        plt.savefig(os.path.join(BASE_DIR, f"./data/yolo-seg-moha/display/pred_yolo-seg-moha_{window_index}.png"), dpi=300)
        image_index = 0
        window_index+=1
        fig, axes = plt.subplots(4, 4, figsize=(5 * max_col, 5 * max_row))
        axes = axes.flatten()
if image_index>0:
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, f"./data/yolo-seg-moha/display/pred_yolo-seg-moha_{window_index}.png"), dpi=300)