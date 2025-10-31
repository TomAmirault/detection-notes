import time
import os, sys
REPO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if REPO_PATH not in sys.path:
    sys.path.insert(0, REPO_PATH)
from src.processing.add_data2db import add_data2db
import paramiko
from logger_config import setup_logger

RASPBERRY_IP = "raspberrypi.local"
RASPBERRY_USER = "projetrte"
port = 22
local_campilots_relative_paths = ["src/raspberry/paper_detection_raspberrypi/perspective_corrector_raspberrypi.py",
                                  "src/raspberry/paper_detection_raspberrypi/save_detection_raspberrypi.py",
                                  "src/raspberry/paper_detection_raspberrypi/shape_detector_raspberrypi.py",
                                  "src/raspberry/paper_detection_raspberrypi/video_capture_raspberrypi.py"]
LOCALxREMOTE_CAMPILOTS_PATHS = {os.path.join(REPO_PATH, rp):os.path.join("/home", RASPBERRY_USER, "Documents/pilots", os.path.basename(rp)) for rp in local_campilots_relative_paths}
REMOTE_MAIN_PATH = LOCALxREMOTE_CAMPILOTS_PATHS[os.path.join(REPO_PATH, "src/raspberry/paper_detection_raspberrypi/video_capture_raspberrypi.py")]
REMOTE_OUTPUT_DIR = os.path.join("/home", RASPBERRY_USER, "Documents/photos")
LOCAL_OUTPUT_DIR = os.path.join(REPO_PATH, "tmp")

REMOTE_LOG_DIR = os.path.join("/home", RASPBERRY_USER, "Documents/logs")
STOP_FILE_PATH = os.path.join("/home", RASPBERRY_USER, "Documents/stop.txt")

logger = setup_logger("launch_rasp")

try:
    # Connection 
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(hostname=RASPBERRY_IP, port=port, username=RASPBERRY_USER)
    sftp = ssh.open_sftp() 
    logger.info("Connection established to the Raspberry Pi.")

    # Creating the necessary directories locally and on the Raspberry Pi 
    os.makedirs(LOCAL_OUTPUT_DIR, exist_ok=True)

    try:
        sftp.mkdir(REMOTE_OUTPUT_DIR)
    except (FileNotFoundError, IOError) as e:
        logger.error("Error" + str(e) + " raised while creating" + REMOTE_OUTPUT_DIR + " on the Raspberry.")

    try:
        sftp.mkdir(REMOTE_LOG_DIR)
    except (FileNotFoundError, IOError) as e:
        logger.error("Error" + str(e) + " raised while creating" + REMOTE_LOG_DIR + " on the Raspberry.")
    
    try:
        sftp.mkdir(os.path.dirname(REMOTE_MAIN_PATH))
    except (FileNotFoundError, IOError) as e:
        logger.error("Error" + str(e) + " raised while creating" + os.path.dirname(REMOTE_MAIN_PATH) + " on the Raspberry.")

    # Deleting the STOP file on the Raspberry Pi 
    try:
        sftp.remove(STOP_FILE_PATH)
        logger.debug("Stop file cleaned.")
    except (FileNotFoundError, IOError) as e:
        logger.debug("Stop file already cleaned:")


    # Copying the camera driver to the Raspberry Pi
    for local, remote in LOCALxREMOTE_CAMPILOTS_PATHS.items():
        sftp.put(local, remote)
    logger.info("Installation of the camera driver on the Raspberry Pi completed.")

    # Launching the camera capture
    logger.info("Starting camera capture")
    stdin, stdout, stderr = ssh.exec_command(f"nohup python3 {REMOTE_MAIN_PATH} > {os.path.join(REMOTE_LOG_DIR, "main.log")} 2>&1 &")
    logger.info("The camera is now filming.")

    # While loop to collect the captured images
    while True:
        logger.debug("Checking for new files...")
        files = sftp.listdir(REMOTE_OUTPUT_DIR)

        for f in files:
            if f.endswith((".jpg", ".png", ".jpeg")):
                sftp.get(os.path.join(REMOTE_OUTPUT_DIR, f), os.path.join(os.path.join(LOCAL_OUTPUT_DIR, f)))
                sftp.remove(os.path.join(REMOTE_OUTPUT_DIR, f))
                add_data2db(os.path.join(LOCAL_OUTPUT_DIR, f))
                logger.debug("New file uploaded to the database:" + str(f))

        time.sleep(2)

except KeyboardInterrupt :
    logger.info("Camera stop requested by the user.")
finally :
    with sftp.file(STOP_FILE_PATH, 'w') as f:
        f.write("This is an empty file to signal the Raspberry Pi to stop the camera.") 
    logger.info("Stop file configured.")
    sftp.close()
    ssh.close()
    logger.info("Connection with the Raspberry Pi closed.")