from vad import VADe
from enregistrement import record_loop
from transcribe import transcribe_w2v2_clean
from transcribe_whisper import transcribe_whisper_clean
import threading

# Ajoute la racine du projet au sys.path pour permettre les imports internes
import sys
import os
REPO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if REPO_PATH not in sys.path:
    sys.path.insert(0, REPO_PATH)

from src.processing.add_data2db import add_audio2db
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

min_duration_on_choice = 3
min_duration_off_choice = 2
prompt = "Abréviations officielles (ne pas développer ; corrige variantes proches vers la forme officielle): SNCF, ABC, RSD, TIR, PF, GEH, SMACC, COSE, TRX, VPL, MNV, N-1, COSE-P"

if __name__ == "__main__":
    try:
        record_duration = 10
        stop_event = threading.Event()
        device_index = 0
        enregistrement_thread = threading.Thread(target = record_loop, args=(record_duration, stop_event, device_index))
        enregistrement_thread.start()
        
        folder_tmp = Path("src/transcription/tmp")
        folder_tests = Path("src/transcription/tests")
        
        while True:
            
            for audio_path in folder_tests.glob("*.wav"): 
                VADe(audio_path, min_duration_on_choice, min_duration_off_choice)
                
            for audio_path in folder_tmp.glob("*.wav"): 
                try:
                    res = transcribe_whisper_clean(audio_path, prompt)
                    if not res:
                        # transcribe returned None -> already transcribed or skipped
                        continue
                    raw, clean = res
                    add_audio2db(str(audio_path), raw, clean)
                except Exception as e:
                    print(f"Erreur transcription/insertion audio pour {audio_path}: {e}")
                
    except KeyboardInterrupt:
        print("Arrêt demandé par l'utilisateur")
    except Exception as e:
        print("Erreur:", e)
        
    stop_event.set()
    print("Attente de la fin de record_loop")
    enregistrement_thread.join()
    
    for audio_path in folder_tests.glob("*.wav"): 
        VADe(audio_path, min_duration_on_choice, min_duration_off_choice)
        
    for audio_path in folder_tmp.glob("*.wav"): 
        transcribe_whisper_clean(audio_path, prompt) 
    print("Fin.")


