# Ce fichier, une fois exécuté, capte en continu le flux de la webcam de l'ordinateur, et, dès que le
# modèle YOLO détecte une feuille de papier, il enregistre les 20 frames suivantes, pour sélectionner
# la frame la moins floue et l'enregistrer, ainsi que sa version rognée autour de la bounding box.
import cv2
import time
from datetime import datetime
from ultralytics import YOLO
import os
from blurry_detection import less_blurred
# from segmentation import crop_image_around_object
from segmentation_threshold import crop_image_around_object

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Modèle YOLOv11 finetuné sur le dataset https://universe.roboflow.com/dty-opi9m/detection-de-feuilles-245oo
model_path = os.path.join(BASE_DIR, '../detection_model/best-detect.pt')
model = YOLO(model_path)

# Timer
start = time.time()
checkpoint = start

# Initialisation des outils de capture des objets
video = []

# Lancement de la webcam
try :
    for result in model.predict(source = 0, show = True, conf = 0.8, verbose = False, stream = True):
        boxes = result.boxes
        if boxes and len(boxes)>0:      # Si un objet (une feuille de papier) est détectée sur la frame
            if len(video)==0:       # Si aucun objet n'était en cours de capture
                if time.time()-checkpoint>5:
                    checkpoint = time.time()
                    video.append(result)
            elif len(video)<20: # Si un objet est en cours de capture
                video.append(result)
            elif len(video)==20:    # Si on a fini de capturer l'objet
                best = less_blurred(video)  # Indice de la frame la moins floue
                stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

                # filename_frame = os.path.join(BASE_DIR, "../../../tmp", f"photo_{stamp}.jpg")
                save_dir_object = os.path.join(BASE_DIR, "../../../tmp")
                # video[best].save_crop(save_dir_object, file_name = f"object_{stamp}.jpg")    # On enregistre la bounding box en tant qu'image
                # video[best].save_txt(os.path.join(save_dir_object, f"output_{stamp}.txt"))
                # cv2.imwrite(filename_frame, video[best].orig_img)                              # On enregistre la frame avec la bounding box tracée
                
                x, y, w, h = video[best].boxes.xywh[0]
                box_x_left = int(x-0.5*w)
                box_y_top = int(y-0.5*h)
                rect = (box_x_left, box_y_top, int(w), int(h))
                # print("rect :", rect)
                # print("shape :", result.orig_img.shape)

                processed = crop_image_around_object(video[best].orig_img, rect)
                filename_frame = os.path.join(BASE_DIR, "../../../tmp/paper", f"paper_{stamp}.jpg")
                cv2.imwrite(filename_frame, processed) 

                video = []      # On réinitialise la sous-vidéo capturée
                
        elif len(video)>0:  # Si l'objet a disparu avant la fin de la capture des 20 frames
            best = less_blurred(video)  # Indice de la frame la moins floue
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

            filename_frame = os.path.join(BASE_DIR, "../../../tmp/paper", f"paper_{stamp}.jpg")
            # save_dir_object = os.path.join(BASE_DIR, "../../../tmp")
            # video[best].save_crop(save_dir_object, file_name = f"object_{stamp}.jpg")    # On enregistre la bounding box en tant qu'image
            # video[best].save_txt(os.path.join(save_dir_object, f"output_{stamp}.txt"))
            x, y, w, h = video[best].boxes.xywh[0]
            box_x_left = int(x-0.5*w)
            box_y_top = int(y-0.5*h)
            rect = (box_x_left, box_y_top, int(w), int(h))

            processed = crop_image_around_object(video[best].orig_img, rect)
            cv2.imwrite(filename_frame, processed)                              # On enregistre la frame avec la bounding box tracée
            
            video = []      # On réinitialise la sous-vidéo capturée
except KeyboardInterrupt :
    print("Arrêt demandé par l'utilisateur.")
finally :
    print("Fin.")