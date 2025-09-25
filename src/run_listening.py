import subprocess
import time
import os

RASPBERRY_IP = "raspberrypi.local"
RASPBERRY_USER = "projetrte"
LOCAL_SCRIPT_PATH = "./raspberry/pilote_camera_photo.py"
REMOTE_SCRIPT_PATH = "/home/projetrte/Documents"
REMOTE_OUTPUT_DIR = "/home/projetrte/Documents/photos"
LOCAL_DEST_DIR = "../tmp"

os.makedirs(LOCAL_DEST_DIR, exist_ok=True)

# --- Copier le script sur la Pi ---
subprocess.run([
    "scp", LOCAL_SCRIPT_PATH, f"{RASPBERRY_USER}@{RASPBERRY_IP}:{REMOTE_SCRIPT_PATH}"
])

# --- Lancer le script sur la Pi en arrière-plan ---
subprocess.run([
    "ssh", f"{RASPBERRY_USER}@{RASPBERRY_IP}", f"nohup python3 {REMOTE_SCRIPT_PATH} > /home/projetrte/pilote_camera.log 2>&1 &"
])

# --- Boucle de récupération des fichiers ---
downloaded_files = set()
while True:
    # lister les fichiers sur la Pi
    result = subprocess.run(
        ["ssh", f"{RASPBERRY_USER}@{RASPBERRY_IP}", f"ls {REMOTE_OUTPUT_DIR}"],
        capture_output=True,
        text=True
    )
    files = result.stdout.splitlines()

    for f in files:
        if f not in downloaded_files and f.endswith(".jpg"):
            subprocess.run([
                "scp",
                f"{RASPBERRY_USER}@{RASPBERRY_IP}:{REMOTE_OUTPUT_DIR}/{f}",
                os.path.join(LOCAL_DEST_DIR, f)
            ])

            downloaded_files.add(f)

    time.sleep(10)
