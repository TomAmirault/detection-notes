# Fichier principal : lance tous les services nécessaires à l'application
import sys
import os
import subprocess
import time

def main():
    # BASE_DIR correspond au chemin absolu du dossier contenant ce script (src)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Pour le front :
    streamlit_path = os.path.join(BASE_DIR, "frontend/app_streamlit.py")

    # Pour la capture vidéo :
    video_capture_path = os.path.abspath(os.path.join(BASE_DIR, "proc/paper_detection/video_capture.py"))
    yolo_path = os.path.join(BASE_DIR, "proc/paper_detection/yolo_tracker_photos.py")
    raspberry_path = os.path.join(BASE_DIR, "raspberry/launch_rasp.py")

    # Pour l'audio
    audio_path = os.path.join(BASE_DIR, "audio/pipeline_watcher.py")

    processes = []
    try:
        # Nettoyer la base de données au lancement
        clear_db_path = os.path.join(BASE_DIR, "backend/clear_db.py")
        subprocess.run([sys.executable, clear_db_path], check=True)

        # Lancer Streamlit en arrière-plan avec logs dans le terminal principal
        processes.append(subprocess.Popen(["streamlit", "run", streamlit_path], stdout=sys.stdout, stderr=sys.stderr))
        print("1) Front (UI) lancé.")

        # Lancer la caméra en arrière-plan avec logs dans le terminal principal et cwd forcé à la racine du projet
        project_root = os.path.abspath(os.path.join(BASE_DIR, ".."))
        processes.append(subprocess.Popen([sys.executable, video_capture_path], stdout=sys.stdout, stderr=sys.stderr, cwd=project_root))
        # processes.append(subprocess.Popen([sys.executable, yolo_path], stdout=sys.stdout, stderr=sys.stderr, cwd=project_root))
        # processes.append(subprocess.Popen([sys.executable, raspberry_path], stdout=sys.stdout, stderr=sys.stderr, cwd=project_root))
        print("2) Lancement du système de détection de feuilles de papier.")

        # Lancer l'audio en arrière-plan
        processes.append(subprocess.Popen([sys.executable, audio_path], stdout=sys.stdout, stderr=sys.stderr))
        print("3) Microphone actif — numérisation audio en cours.")

        print("=> Tous les process ont été lancés en arrière-plan.")
        print("Le système numérise toutes les informations (notes manuscrites et parole).")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Arrêt demandé par l'utilisateur.")
    except Exception as e:
        print(f"Erreur inattendue : {e}")
    finally:
        for p in processes:
            p.terminate()
        print("Tous les processus ont été terminés.")

if __name__ == "__main__":
    main()