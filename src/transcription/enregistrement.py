import sounddevice as sd
import soundfile as sf
import numpy as np
import noisereduce as nr
from datetime import datetime
import os


def record_loop(duration=10, bruit_reduction=True):
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