from picamera2 import Picamera2
import time
from shape_detector_raspberrypi import shape_detector
from save_detection_raspberrypi import save_detection

FPS = 30
DELAY = 1 / FPS  

picam = Picamera2()
config = picam.create_video_configuration(main={"size": (4608, 2592)})
picam.configure(config)
picam.start()

while True:
    img = picam.capture_array()
    possible_papers = shape_detector(img)
    if possible_papers:
        save_detection(img, possible_papers)
    # time.sleep(DELAY)

picam.stop()  # sera exécuté si script est tué proprement
