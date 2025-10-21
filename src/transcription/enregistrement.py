import sounddevice as sd
import soundfile as sf
import numpy as np
import noisereduce as nr
from datetime import datetime
import os
import json

# python3 src/transcription/enregistrement.py
# python3 -m sounddevice   /  Pour voir les entrées audio
 

def record_loop(duration, stop_event, device=0,  bruit_reduction=True, samplerate=16000):
    """
    Enregistre des segments audio consécutifs et les sauvegarde dans un dossier temporaire.
    L'enregistrement s'arrête manuellement avec Ctrl+C ou automatiquement après `duration` secondes.

    Args:
        duration (float): Durée de chaque segment audio.
        bruit_reduction (bool): Appliquer une réduction de bruit.
        samplerate (int): Taux d'échantillonnage audio.
    """
    
    os.makedirs("src/transcription/tests", exist_ok=True)
    log_path = os.path.join("src/transcription/tests", "audio_brut.json")

    # Supprimer le fichier JSON existant
    if os.path.exists(log_path):
        os.remove(log_path)



    k = 1

    print("Enregistrement de la parole.")

    log= []

    try:
        with sd.InputStream(samplerate=samplerate, channels=1, dtype='float32') as stream:
            while not stop_event.is_set():
                
                
                audio_folder = "src/transcription/tests"
                # Récupérer la liste des fichiers .wav existants dans le dossier
                existing_files = {f for f in os.listdir(audio_folder) if f.endswith(".wav")}

                # Filtrer les entrées : on garde uniquement celles dont le filename existe vraiment
                log = [entry for entry in log if entry["filename"] in existing_files]

                # Sauvegarder le JSON nettoyé
                with open(log_path, "w", encoding="utf-8") as f:
                    json.dump(log, f, indent=4, ensure_ascii=False)
                       
                       
                start_time = datetime.now()
                safe_time = start_time.strftime("%Y%m%d_%H%M%S")
                filename = f"record_chunk_{k}_{safe_time}.wav"
                filepath = os.path.join("src/transcription/tests", filename)

                frames = []
                while (datetime.now() - start_time).total_seconds() < duration:
                    block = stream.read(1024)[0]
                    frames.append(block)

                recording = np.concatenate(frames, axis=0).squeeze()

                if bruit_reduction:
                    recording = nr.reduce_noise(y=recording, sr=samplerate)
                sf.write(filepath, recording, samplerate)
                end_time = datetime.now()
            
            
                # Charger le JSON existant
                if os.path.exists(log_path):
                    with open(log_path, "r", encoding="utf-8") as f:
                        log = json.load(f)
                else:
                    log = []
                    
                    
         
                    
                    
                entry = {
                    "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "end_time": end_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "filename": filename
                }
                
                log.append(entry)
                
                                # Sauvegarde du JSON après chaque segment
                with open(log_path, "w", encoding="utf-8") as f:
                    json.dump(log, f, indent=4, ensure_ascii=False)


                
                
                
                k += 1
            print("Event est bien set")

    except KeyboardInterrupt:
        print("\nArrêt demandé par l'utilisateur (Ctrl+C).")

        # Sauvegarde du dernier segment partiel (si existant)
        if 'frames' in locals() and len(frames) > 0:
            recording = np.concatenate(frames, axis=0).squeeze()
            if bruit_reduction:
                recording = nr.reduce_noise(y=recording, sr=samplerate)

            sf.write(filepath, recording, samplerate)
            
            end_time = datetime.now()

            entry = {
                "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
                "end_time": end_time.strftime("%Y-%m-%d %H:%M:%S"),
                "filename": filename
            }

            with open(log_path, "w", encoding="utf-8") as f:
                json.dump([entry], f, indent=4, ensure_ascii=False)
                
                
if __name__ == "__main__":
    from pathlib import Path
    folder_tests = Path("src/transcription/tests")
    
    for f in folder_tests.glob("*.wav"):
        try:
            f.unlink()
        except Exception as e:
            print(f"Could not remove {f}: {e}")
            
    import threading
    
    try:
        stop_event = threading.Event()
        enregistrement_thread = threading.Thread(target = record_loop, args=(5, stop_event, 0))
        enregistrement_thread.start()
    except KeyboardInterrupt:
        print("Arrêt demandé par l'utilisateur")
        stop_event.set()
        print("Attente de la fin de record_loop")
        enregistrement_thread.join()
        print("Fin.")