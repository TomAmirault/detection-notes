import cv2
import time
from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
import os
import subprocess

# Création des dossiers de travail

# SAVE_DIR = "/home/projetrte/Documents/video_detection"
# os.makedirs(SAVE_DIR, exist_ok=True)

os.makedirs("./sampleH264", exist_ok=True)
os.makedirs("./sampleMP4", exist_ok=True)
os.makedirs("./photos", exist_ok=True)

# Initialisation de la caméra

picam2 = Picamera2()
picam2.start()
# time.sleep(2)  # laisser la caméra ajuster la lumière

# Paramètres vidéo
# fps = 20
# frame = picam2.capture_array()
# width, height, _ = frame.shape
# video_filename = os.path.join(SAVE_DIR, "detection.avi")
# fourcc = cv2.VideoWriter_fourcc(*'MJPG')
# out = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))

# Fonctions de traitement

def processed_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    return blur

def detect_paper(img):
    proc = processed_image(img)
    edges = cv2.Canny(proc, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for cnt in contours:
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) == 4 and cv2.contourArea(cnt) > 1000:
            candidates.append(approx)
    return candidates

start = time.time()
duration = 30  # secondes
interval = 4    # secondes
encoder = H264Encoder(bitrate=10000000)
i = 0

# Capture des images

try:
    while time.time() - start < duration:
        i+=1
        time.sleep(interval)

        # Capture d'images
        
        picam2.capture_file("./photos/photo"+str(i)+".jpg")

        # Capture de vidéos

        # filename = "./sampleH264/sample" + str(i) + ".h264"
        # picam2.start_recording(encoder, filename)
        # time.sleep(1)
        # picam2.stop_recording()

        # Création d'une vidéo à partir de frames

        # frame = picam2.capture_array()
        # papers = detect_paper(frame)
        # cv2.drawContours(frame, papers, -1, (0,255,0), 2)
        # out.write(frame)  # écrire la frame dans la vidéo
finally:
    picam2.stop()
    # out.release()
    # print("Vidéo enregistrée dans :", video_filename)
    # source_folder = "./sampleH264"
    # dest_folder = "./sampleMP4"
    # filenames = os.listdir(source_folder)
    # for f in filenames :
    #     # Conversion en MP4
    #     h264_file = os.path.join(source_folder, f)
    #     mp4_file = os.path.join(dest_folder, f[:-5]+".mp4")
    #     subprocess.run([
    #         "ffmpeg",
    #         "-framerate", "30",       # framerate utilisé à l'enregistrement
    #         "-i", h264_file,          # fichier source
    #         "-c", "copy",             # copie le flux H264 sans ré-encodage
    #         mp4_file                  # fichier de sortie
    #     ])
    #     # os.remove(h264_file)

