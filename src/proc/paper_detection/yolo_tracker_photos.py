# Ce fichier, une fois exécuté, capte en continu le flux de la webcam de l'ordinateur, et, dès que le
# modèle YOLO détecte une feuille de papier, il enregistre les 20 frames suivantes, pour sélectionner
# la frame la moins floue et l'enregistrer, ainsi que sa version rognée autour de la bounding box.
import cv2
import time
from datetime import datetime
from ultralytics import YOLO
import os
from blurry_detection import less_blurred, less_blurred_roi
# from segmentation import crop_image_around_object
from segmentation_threshold import crop_image_around_object, get_binary_image_of_text
from blurry_detection import laplacian_variance

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Modèle YOLOv11 finetuné sur le dataset https://universe.roboflow.com/dty-opi9m/detection-de-feuilles-245oo
model_path = os.path.join(BASE_DIR, '../detection_model/best-detect.pt')
model = YOLO(model_path)

# # Timer
# start = time.time()
# checkpoint = start

# Initialisation du buffer d'images
buffer = []

# Lancement de la webcam
try :
    cap = cv2.VideoCapture(0)
    while True:
        print("cam_height, cam_width =", cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
        print("cam_height, cam_width =", cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        ret, frame = cap.read()
        if ret:
            result = model.predict(source=frame, conf=0.8)[0]
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
                    if mean_s < 100 and mean_v > 210:
                        blur_value = laplacian_variance(cropped)
                        buffer.append({'image':cropped, 'blur_value':blur_value})

                if len(buffer)>=5:
                    best_image = max(buffer, key=lambda x: x['blur_value'])
                    img = best_image['image']
                    blur_value = best_image['blur_value']
                    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                    filename_frame = os.path.join(BASE_DIR, "../../../tmp/paper", f"detected_sheet_{stamp}_{blur_value:.1f}.jpg")
                    cv2.imwrite(filename_frame, img)
                    buffer = []
            else :
                buffer = []
            time.sleep(1)
except KeyboardInterrupt :
    print("Arrêt demandé par l'utilisateur.")
finally :
    if cap:
        cap.release()
    print("Caméra arrêtée.")
    print("Fin.")