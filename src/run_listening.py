# from paramiko import SSHClient, AutoAddPolicy
# from scp import SCPClient
import subprocess
import time
import os
from add_data2db import add_data2db

WORKING_DIR = "/Users/noahparisse/Documents/Paris-digital-lab/P1 RTE/"
RASPBERRY_IP = "raspberrypi.local"
RASPBERRY_USER = "projetrte"
LOCAL_SCRIPT_PATH = WORKING_DIR + "detection-notes/src/raspberry/pilote_camera_photo.py"
REMOTE_SCRIPT_PATH = "/home/projetrte/Documents/pilote_camera_photo.py"
REMOTE_OUTPUT_DIR = "/home/projetrte/Documents/photos"
LOCAL_DEST_DIR = WORKING_DIR + "detection-notes/tmp"

os.makedirs(LOCAL_DEST_DIR, exist_ok=True)

# result = subprocess.run(
#     ["/bin/bash", "/Users/noahparisse/Documents/Paris-digital-lab/P1 RTE/detection-notes/src/exec/connect_raspberry.sh"],
#     capture_output=True,
#     text=True
# )
# print("STDOUT:\n", result.stdout)
# print("STDERR:\n", result.stderr)

# ssh = SSHClient()

# ssh.load_system_host_keys()
# ssh.set_missing_host_key_policy(AutoAddPolicy())
# ssh.connect("raspberrypi.local", username="projetrte", allow_agent=True, look_for_keys=True,)

# --- Copier le script sur la Pi ---
print("Installation du pilote de la caméra sur la Raspberry Pi...")

# with SCPClient(ssh.get_transport()) as scp:
#     scp.put("local_script.py", "/home/projetrte/remote_script.py")

# env = os.environ
# subprocess.run([
#     "ssh-add", "-l"
# ])
subprocess.run([
    "scp", LOCAL_SCRIPT_PATH, f"{RASPBERRY_USER}@{RASPBERRY_IP}:{REMOTE_SCRIPT_PATH}"
])

# --- Lancer le script sur la Pi en arrière-plan ---
print("Démarrage de la capture caméra")
subprocess.run([
    "ssh", f"{RASPBERRY_USER}@{RASPBERRY_IP}", f"nohup python3 {REMOTE_SCRIPT_PATH} > /home/projetrte/Documents/pilote_camera_photo.log 2>&1 &"
])

print("La caméra a bien démarré.")
# ssh.exec_command(f"nohup python3 {REMOTE_SCRIPT_PATH} > /home/projetrte/timelapse.log 2>&1 &")

# --- Boucle de récupération des fichiers ---
downloaded_files = set()
while True:
    # lister les fichiers sur la Pi
    print("Lecture des fichiers sur la Raspberry...")
    result = subprocess.run(
        ["ssh", f"{RASPBERRY_USER}@{RASPBERRY_IP}", f"ls {REMOTE_OUTPUT_DIR}"],
        capture_output=True,
        text=True
    )
    files = result.stdout.splitlines()

    for f in files:
        if f not in downloaded_files and f.endswith(".jpg"):
            local_filepath = os.path.join(LOCAL_DEST_DIR, f)
            subprocess.run([
                "scp",
                f"{RASPBERRY_USER}@{RASPBERRY_IP}:{REMOTE_OUTPUT_DIR}/{f}",
                local_filepath
            ])
            downloaded_files.add(f)
            add_data2db(local_filepath)
            print("Nouveau fichier chargé dans la BDD :",f)

    time.sleep(10)

# try:
#     while True:
#         # lister les fichiers sur la Pi
#         stdin, stdout, stderr = ssh.exec_command(f"ls {REMOTE_OUTPUT_DIR}")
#         files = stdout.read().decode().splitlines()

#         with SCPClient(ssh.get_transport()) as scp:
#             for f in files:
#                 if f not in downloaded_files and f.endswith(".jpg"):  # filtre par type si besoin
#                     remote_path = os.path.join(REMOTE_OUTPUT_DIR, f)
#                     local_path = os.path.join(LOCAL_DEST_DIR, f)
#                     scp.get(remote_path, local_path)
#                     downloaded_files.add(f)
#                     print(f"Téléchargé : {f}")

#         time.sleep(10)

# except KeyboardInterrupt:
#     print("Arrêt du script par l'utilisateur.")

# finally:
#     ssh.close()
#     print("Connexion SSH fermée.")