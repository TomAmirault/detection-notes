# Ce fichier, une fois exécuté, capte en continu le flux de la webcam de l'ordinateur, et, dès que le
# modèle YOLO détecte une feuille de papier, il enregistre les 20 frames suivantes, pour sélectionner
# la frame la moins floue et l'enregistrer, ainsi que sa version rognée autour de la bounding box.
import sys, os
REPO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if REPO_PATH not in sys.path:
    sys.path.insert(0, REPO_PATH)

import cv2
import time
from datetime import datetime
from ultralytics import YOLO
from blurry_detection import less_blurred, less_blurred_roi
# from segmentation import crop_image_around_object
from segmentation_threshold import crop_image_around_object, get_binary_image_of_text
from blurry_detection import laplacian_variance

from src.processing.add_data2db import add_data2db
from logger_config import save_fig_with_limit
import matplotlib.pyplot as plt
import numpy as np
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Paramètres ---
frame_width = 1552      # pour Webcam Mac
frame_height = 1552     # pour Webcam Mac
detection_conf = 0.8    # Seuil de confiance pour le modèle YOLO
s_max = 100
v_min = 210


# Modèle YOLOv11 finetuné sur le dataset https://universe.roboflow.com/dty-opi9m/detection-de-feuilles-245oo
model_path = os.path.join(REPO_PATH, 'src/proc/detection_model/best-detect.pt')
model = YOLO(model_path)

# # Timer
# start = time.time()
# checkpoint = start


# Initialisation du buffer d'images (indice 0 bloqué pour les informations sur les buffers en cours)
buffers = [{'max_buffer_len': 0, "buffer_num": 0}]

# Lancement de la webcam
try :
    cap = cv2.VideoCapture(0)
    print("Lancement de la caméra. Pour arrêter, taper 'q'.")
    print("\n A l'ouverture de la caméra : cam_height, cam_width =", cap.get(cv2.CAP_PROP_FRAME_HEIGHT), "pixels, ", cap.get(cv2.CAP_PROP_FRAME_WIDTH), "pixels.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    print("Après modification : cam_height, cam_width =", cap.get(cv2.CAP_PROP_FRAME_HEIGHT), "pixels, ", cap.get(cv2.CAP_PROP_FRAME_WIDTH), "pixels.")
    while True:
        ret, frame = cap.read()
        if ret:
            result = model.predict(source=frame, conf=detection_conf, verbose=False)[0]
            boxes = result.boxes
            if boxes and len(boxes)>0:      # Si un objet (une feuille de papier) est détectée sur la frame
                
                for i in range(len(boxes.xywh)):
                    (x, y, w, h) = boxes.xywh[i]
                    box_x_left = int(x-0.5*w)
                    box_y_top = int(y-0.5*h)
                    rect = (box_x_left, box_y_top, int(w), int(h))

                    cropped = crop_image_around_object(result.orig_img, rect)

                    hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
                    mean_h, mean_s, mean_v = cv2.mean(hsv)[:3]

                    # Condition : zone claire et peu saturée (donc blanche)
                    if mean_s < s_max and mean_v > v_min: 
                        blur_value = laplacian_variance(cropped)
                        if i+1<=buffers[0]['buffer_num']:
                            buffers[i+1].append({'image':cropped, 'blur_value':blur_value})
                            buffers[0]['max_buffer_len']=max(buffers[0]['max_buffer_len'], len(buffers[i+1]))
                        else:
                            buffers.append([])
                            buffers[i+1]=[{'image':cropped, 'blur_value':blur_value}]
                            buffers[0]['buffer_num']+=1
                            buffers[0]['max_buffer_len']=max(buffers[0]['max_buffer_len'], len(buffers[i+1]))
                    else:
                        fig, ax = plt.subplots()
                        ax.imshow(cropped)
                        stamp = f"{datetime.now():%Y%m%d-%H%M%S}-{datetime.now().microsecond//1000:03d}"
                        file_name =f"logs/color-criteria/image_not_conservated_{stamp}.jpg"
                        save_fig_with_limit(file_name, fig)
                        print("Image non conservée sur critère de couleur.")

                if buffers[0]["max_buffer_len"]>=5:
                    print("BUFFER")
                    for i in range(buffers[0]["buffer_num"]):
                        best_image = max(buffers[i+1], key=lambda x: x['blur_value'])
                        img = best_image['image']
                        blur_value = best_image['blur_value']
                        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                        filename_frame = os.path.join(REPO_PATH, "tmp/paper", f"detected_sheet_{stamp}_{blur_value:.1f}.jpg")
                        cv2.imwrite(filename_frame, img)
                        add_data2db(filename_frame)
                    buffers = [{'max_buffer_len': 0, "buffer_num": 0}]
            else :
                buffers = [{'max_buffer_len': 0, "buffer_num": 0}]
            cv2.destroyAllWindows()
            if boxes and len(boxes)>0:
                for i in range(len(boxes.xywh)):
                    (x, y, w, h) = boxes.xywh[i]
                    rect = ((int(x), int(y)), (int(w), int(h)), 0)
                    box = cv2.boxPoints(rect)
                    box = np.int32(box)
                    cv2.drawContours(frame, [box], -1, (0, 255, 0), 2)
            cv2.imshow('Webcam', frame)
            if cv2.waitKey(1) == ord('q'):
                break
            time.sleep(1)
except KeyboardInterrupt :
    print(" Arrêt demandé par l'utilisateur.")
finally :
    if cap:
        cap.release()
    print("Caméra arrêtée.")