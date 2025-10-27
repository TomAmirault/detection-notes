'''This script reads, at the speed of 1 fps, the webcam stream and run a finetyuned YOLOv11 model
to detect sheets of paper. When a sheet is detected, the frame is cropped (using crop_image_around_object)
and saved in a buffer. Once 5 images are in the buffer, it keeps the less blurred (according to
the laplacian_variance function defined in the blury_detection) and send it to the rest of the
pipeline (add_data2db function).
'''
import sys, os
REPO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if REPO_PATH not in sys.path:
    sys.path.insert(0, REPO_PATH)

import cv2
import time
from datetime import datetime
from logger_config import setup_logger
from ultralytics import YOLO
from segmentation_threshold import crop_image_around_object, get_binary_image_of_text
from blurry_detection import laplacian_variance
from src.processing.add_data2db import add_data2db
from logger_config import save_fig_with_limit
import matplotlib.pyplot as plt
import numpy as np

logger = setup_logger("yolo_tracker_photos")

# --- Parameters ---
frame_width = 1552      # for Mac Webcam
frame_height = 1552     # for Mac Webcam
detection_conf = 0.8    # Confidence threshold for YOLO model
s_max = 100
v_min = 210


# YOLOv11 detection model finetuned on the following dataset https://universe.roboflow.com/dty-opi9m/detection-de-feuilles-245oo
model_path = os.path.join(REPO_PATH, 'src/proc/detection_model/best-detect.pt')
model = YOLO(model_path)


# The buffers (in case of multiple sheets detected on the same frame) are initialized here
# Each buffer is a list of dictionaries with keys 'image' and 'blur_value' stocked in buffers,
# starting from index 1.
buffers = [{'max_buffer_len': 0, "buffer_num": 0}]

try :
    cap = cv2.VideoCapture(0)
    logger.info("Launching the camera.")
    logger.debug("At camera opening: cam_height, cam_width =" + str(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) + " pixels, " + str(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) + " pixels.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    logger.debug("After modification: cam_height, cam_width =" + str(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) + " pixels, " + str(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) + " pixels.")
    while True:
        ret, frame = cap.read()
        if ret:
            result = model.predict(source=frame, conf=detection_conf, verbose=False)[0]
            boxes = result.boxes

            # If a sheet of paper is detected
            if boxes and len(boxes)>0:
                for i in range(len(boxes.xywh)):
                    (x, y, w, h) = boxes.xywh[i]
                    box_x_left = int(x-0.5*w)
                    box_y_top = int(y-0.5*h)
                    rect = (box_x_left, box_y_top, int(w), int(h))

                    cropped = crop_image_around_object(result.orig_img, rect)

                    hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
                    mean_h, mean_s, mean_v = cv2.mean(hsv)[:3]

                    # We keep the image only if it is mostly white
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
                        file_name =f"logs/color-criteria/image_not_conserved_{stamp}.jpg"
                        save_fig_with_limit(file_name, fig)
                        logger.debug("Image not conserved on color criteria: " + file_name)

                if buffers[0]["max_buffer_len"]>=5:
                    logger.debug("BUFFER - Buffer reached its maximum length. Saving the best image.")
                    for i in range(buffers[0]["buffer_num"]):
                        best_image = max(buffers[i+1], key=lambda x: x['blur_value'])
                        img = best_image['image']
                        blur_value = best_image['blur_value']
                        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                        filename_frame = os.path.join(REPO_PATH, "tmp/paper", f"detected_sheet_{stamp}_{blur_value:.1f}.jpg")
                        cv2.imwrite(filename_frame, img)
                        add_data2db(filename_frame)
                    buffers = [{'max_buffer_len': 0, "buffer_num": 0}]
                    
            # If no sheet of paper is detected, we reset the buffers because if it was not empty,
            # it means that the previous images stayed in the field of vision of the camera less
            # than 5 seconds : it was not a sheet to be scanned.
            else :
                buffers = [{'max_buffer_len': 0, "buffer_num": 0}]
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
    logger.info("The user asked to stop the camera.")
finally :
    if cap:
        cap.release()
    logger.info("Camera stopped.")