import sounddevice as sd
import soundfile as sf
import numpy as np
import noisereduce as nr
from datetime import datetime
import os
from denoise import denoise_audio
from normalize import normalize_volume

def record_loop(duration=10, bruit_reduction=True):
    """
    Enregistre des segments audio en boucle et les sauvegarde dans un dossier temporaire.

    À chaque itération, cette fonction :
      - démarre un enregistrement audio d'une durée donnée ;
      - applique éventuellement une réduction de bruit ;
      - enregistre le fichier .wav dans un dossier `tmp` avec un nom horodaté ;
      - consigne l'heure de début et de fin dans un fichier `transcriptions_log.txt`.

    L'enregistrement s'interrompt manuellement avec `Ctrl+C`.

    Parameters
    ----------
    duration : int or float, optional (default=10)
        Durée (en secondes) de chaque segment audio enregistré.
    bruit_reduction : bool, optional (default=True)
        Si True, applique une réduction de bruit à l’aide de la fonction 
        `noisereduce.reduce_noise`.

    Outputs
    -------
    - Aucun retour direct (None).
    - Effets de bord :
        * Crée un dossier `tmp/` s’il n’existe pas déjà.
        * Enregistre les fichiers audio nommés `record_chunk_<n>_<timestamp>.wav`.
        * Écrit dans `tmp/transcriptions_log.txt` les intervalles temporels
          correspondant à chaque enregistrement.
    """
    os.makedirs("tmp", exist_ok=True)
    log_path = os.path.join("tmp", "transcriptions_log.txt")
    try:
        k = 1
        while True:
            
            start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            safe_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"record_chunk_{k}_{safe_time}.wav"
            filepath = os.path.join("tmp", filename)
            
            if k == 1 :
                print(f"Parlez maintenant")
            recording = sd.rec(int(duration * 16000), samplerate=16000, channels=1, dtype="float32")
            sd.wait()
            recording = np.squeeze(recording)
            
            if bruit_reduction:
                # Réduction du bruit
                reduced_noise = nr.reduce_noise(y=recording, sr=16000)
                sf.write(filepath, reduced_noise, 16000)
            else:
                sf.write(filepath, recording, 16000)
                
            end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            with open(log_path, "a", encoding="utf-8") as log_file:
                log_file.write(f"[{start_time} → {end_time}] - {filename}\n")
                
            k = k+1

    except KeyboardInterrupt:
        print("\n Arrêt demandé par l'utilisateur (Ctrl+C).")