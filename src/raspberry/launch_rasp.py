# from paramiko import SSHClient, AutoAddPolicy
# from scp import SCPClient
import subprocess
import time
import os, sys
REPO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if REPO_PATH not in sys.path:
    sys.path.insert(0, REPO_PATH)
from src.processing.add_data2db import add_data2db
import paramiko

# 
RASPBERRY_IP = "raspberrypi.local"
RASPBERRY_USER = "projetrte"
port = 22
local_campilots_relative_paths = ["src/raspberry/paper_detection_raspberrypi/image_preprocessing_raspberrypi.py",
                                   "src/raspberry/paper_detection_raspberrypi/perspective_corrector_raspberrypi.py",
                                   "src/raspberry/paper_detection_raspberrypi/save_detection_raspberrypi.py",
                                   "src/raspberry/paper_detection_raspberrypi/shape_detector_raspberrypi.py",
                                   "src/raspberry/paper_detection_raspberrypi/video_capture_raspberrypi.py"]
LOCALxREMOTE_CAMPILOTS_PATHS = {os.path.join(REPO_PATH, rp):os.path.join("/home/projetrte/Documents/pilots", os.path.basename(rp)) for rp in local_campilots_relative_paths}
REMOTE_MAIN_PATH = LOCALxREMOTE_CAMPILOTS_PATHS[os.path.join(REPO_PATH, "src/raspberry/paper_detection_raspberrypi/video_capture_raspberrypi.py")]
REMOTE_OUTPUT_DIR = "/home/projetrte/Documents/photos"
LOCAL_OUTPUT_DIR = os.path.join(REPO_PATH, "tmp")

REMOTE_LOG_DIR = "/home/projetrte/Documents/logs"
STOP_FILE_PATH = "/home/projetrte/Documents/stop.txt"

try:
    # Connexion
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(hostname=RASPBERRY_IP, port=port, username=RASPBERRY_USER)
    print("Connexion à la Raspberry établie.")
    sftp = ssh.open_sftp() 
    print("Ouverture du client SFTP.")

    # --- Création des répertoires d'intérêt ---
    os.makedirs(LOCAL_OUTPUT_DIR, exist_ok=True)

    try:
        sftp.mkdir(REMOTE_OUTPUT_DIR)
    except (FileNotFoundError, IOError) as e:
        print("Erreur", e, " levée lors de la création de", REMOTE_OUTPUT_DIR)

    try:
        sftp.mkdir(REMOTE_LOG_DIR)
    except (FileNotFoundError, IOError) as e:
        print("Erreur", e, " levée lors de la création de", REMOTE_LOG_DIR)
    
    try:
        sftp.mkdir(os.path.dirname(REMOTE_MAIN_PATH))
    except (FileNotFoundError, IOError) as e:
        print("Erreur", e, " levée lors de la création de", os.path.dirname(REMOTE_MAIN_PATH))

    # --- Supprimer le fichier d'arrêt de la Raspberry Pi ---
    try:
        sftp.remove(STOP_FILE_PATH)
        print("Fichier d'arrêt nettoyé.")
    except (FileNotFoundError, IOError) as e:
        print(f"Fichier d'arrêt déjà nettoyé :", e)
  

    # --- Copier le script sur la Pi ---
    print("Installation du pilote de la caméra sur la Raspberry Pi...")

    for local, remote in LOCALxREMOTE_CAMPILOTS_PATHS.items():
        sftp.put(local, remote)
    print("Installation du pilote de la caméra sur la Raspberry Pi terminée.")

    # --- Lancer le script sur la Pi en arrière-plan ---
    print("Démarrage de la capture caméra")
    stdin, stdout, stderr = ssh.exec_command(f"nohup python3 {REMOTE_MAIN_PATH} > {os.path.join(REMOTE_LOG_DIR, "main.log")} 2>&1 &")
    print("La caméra est en train de filmer.")

    # --- Boucle de récupération des fichiers ---
    while True:
        # lister les fichiers sur la Pi
        print("Lecture des fichiers sur la Raspberry...")
        files = sftp.listdir(REMOTE_OUTPUT_DIR)

        for f in files:
            if f.endswith((".jpg", ".png", ".jpeg")):
                sftp.get(os.path.join(REMOTE_OUTPUT_DIR, f), os.path.join(os.path.join(LOCAL_OUTPUT_DIR, f)))
                sftp.remove(os.path.join(REMOTE_OUTPUT_DIR, f))
                add_data2db(os.path.join(LOCAL_OUTPUT_DIR, f))
                print("Nouveau fichier chargé dans la BDD :",f)

        time.sleep(2)

except KeyboardInterrupt :
    print(" Arrêt demandé par l'utilisateur.")
finally :
    with sftp.file(STOP_FILE_PATH, 'w') as f:
        f.write("Ceci est un fichier vide créé pour signaler aux processus sur la Raspberry de s'arrêter.") 
    print("Fichier d'arrêt configuré.")
    sftp.close()
    ssh.close()
    print("Connexion avec la Raspberry fermée.")