from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
import time

picam2 = Picamera2()
picam2.start()

# Durée en secondes
duration = 10
output_file = "/home/projetrte/Documents/video2.h264"
encoder = H264Encoder(bitrate=10000000)
picam2.start_recording(encoder, output_file)  # démarrer l’enregistrement
time.sleep(duration)                  # attendre la durée
picam2.stop_recording()               # arrêter l’enregistrement

picam2.stop()
print("Vidéo enregistrée dans", output_file)
