from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
import time
import subprocess

picam2 = Picamera2()
picam2.start()

# Durée en secondes
duration = 20
output_file = "/home/projetrte/Documents/video.h264"
mp4_file = "/home/projetrte/Documents/video.mp4"
encoder = H264Encoder(bitrate=10000000)
picam2.start_recording(encoder, output_file)  # démarrer l’enregistrement
time.sleep(duration)                  # attendre la durée
picam2.stop_recording()               # arrêter l’enregistrement
subprocess.run([
    "ffmpeg",
    "-framerate", "30",       # framerate utilisé à l'enregistrement
    "-i", output_file,          # fichier source
    "-c", "copy",             # copie le flux H264 sans ré-encodage
    mp4_file                  # fichier de sortie
])

picam2.stop()
print("Vidéo enregistrée dans", output_file)
