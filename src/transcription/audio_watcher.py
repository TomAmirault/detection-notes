
print("Chargement des Modèles")

from vad import VADe
from enregistrement import record_loop
from transcribe import transcribe_w2v2_clean
import threading
import os
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

print("Modèles Prêts")


if __name__ == "__main__":
    try:
        record_duration = 50
        stop_event = threading.Event()
        device_index = 0
        enregistrement_thread = threading.Thread(target = record_loop, args=(record_duration, stop_event, device_index))
        enregistrement_thread.start()
        
        folder_tmp = Path("src/transcription/tmp")
        folder_tests = Path("src/transcription/tests")
        
        while True:
            
            for audio_path in folder_tests.glob("*.wav"): 
                VADe(audio_path, min_duration_on=2, min_duration_off=5)
                
            for audio_path in folder_tmp.glob("*.wav"): 
                transcribe_w2v2_clean(audio_path)  
                
    except KeyboardInterrupt:
        print("Arrêt demandé par l'utilisateur")
    except Exception as e:
        print("Erreur:", e)
        
    stop_event.set()
    print("Attente de la fin de record_loop")
    enregistrement_thread.join()
    
    for audio_path in folder_tests.glob("*.wav"): 
        VADe(audio_path, min_duration_on=2, min_duration_off=5)
        
    for audio_path in folder_tmp.glob("*.wav"): 
        transcribe_w2v2_clean(audio_path) 
    print("Fin.")


