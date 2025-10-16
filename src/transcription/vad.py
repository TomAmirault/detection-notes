from pathlib import Path
from huggingface_hub import login
from pyannote.audio import Model
from pyannote.audio.pipelines import VoiceActivityDetection
import torchaudio
import sounddevice as sd
import soundfile as sf
import numpy as np
import noisereduce as nr
import torch
import json
from datetime import datetime, timedelta

# Ajoute la racine du projet au sys.path pour permettre les imports internes
import sys
import os
REPO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if REPO_PATH not in sys.path:
    sys.path.insert(0, REPO_PATH)

# python3 src/transcription/

model_segm = Model.from_pretrained(
"pyannote/segmentation-3.0")

pipeline = VoiceActivityDetection(segmentation=model_segm)

def VADe(audio_path, min_duration_on=3.0, min_duration_off=2.0):

    HYPER_PARAMETERS = {
    # Si un segment de parole détecté dure moins de 3 secondes, il sera ignoré.
    "min_duration_on": min_duration_on,
    # Si une pause est plus courte que 10 secondes, elle peut être remplie ou fusionnée avec les segments voisins.
    "min_duration_off": min_duration_off
    }
    pipeline.instantiate(HYPER_PARAMETERS)
    vad = pipeline(audio_path)

    log_path = os.path.join("src/transcription/tmp", "transcriptions_log.json")

    # Charger le JSON existant s'il existe
    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8") as f:
            try:
                logs = json.load(f)
            except json.JSONDecodeError:
                logs = []
    else:
        logs = []
        
        
        
    #Time
    filename_brut = audio_path.name

    brut_json_path = "src/transcription/tests/audio_brut.json"
    if not os.path.exists(brut_json_path):
        print(f"Fichier introuvable: {brut_json_path} — saut de VADe pour {audio_path}")
        return

    with open(brut_json_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print(f"JSON invalide dans {brut_json_path} — saut de VADe pour {audio_path}")
            return

    entry = next((item for item in data if item.get("filename") == filename_brut), None)
    if not entry:
        print(f"Aucun enregistrement trouvé pour ce fichier ({filename_brut}) dans {brut_json_path}.")
        return

    start_time = entry.get("start_time")
    if not start_time:
        print(f"Aucun start_time pour {filename_brut} — saut de VADe.")
        return

    
    waveform, sr = torchaudio.load(audio_path)
    i=0
    for segment, _, _ in vad.itertracks(yield_label=True):
        #print(segment)
        i = i+1
        
        s = int(segment.start)
        e = int(segment.end)
        
        start = int(segment.start * sr)
        end = int(segment.end * sr)
        
        segment = waveform[:, start:end]
        
        segment_filename = f"{os.path.splitext(os.path.basename(audio_path))[0]}_segment_{i}.wav"
        segment_path = os.path.join("src/transcription/tmp", segment_filename)
        
        torchaudio.save(segment_path, segment, sr)
        
        
        base_start_str = start_time
        base_start = datetime.strptime(base_start_str, "%Y-%m-%d %H:%M:%S")


        absolute_start = base_start + timedelta(seconds=s)
        absolute_end = base_start + timedelta(seconds=e)
        
        
        entry = {
            "start_time": absolute_start.strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": absolute_end.strftime("%Y-%m-%d %H:%M:%S"),
            "filename": segment_filename
        }
        logs.append(entry)
        
    # --- AJOUT DU TRI PAR DATE APRÈS SAUVEGARDE ---


    logs.sort(key=lambda x: datetime.strptime(x["start_time"], "%Y-%m-%d %H:%M:%S"))


    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=4, ensure_ascii=False)
    # ------------------------------------------------ #

    with open("src/transcription/tests/audio_brut.json", "r") as f:
        data = json.load(f)
    
    data = [entry for entry in data if entry.get("filename") != filename_brut]
    for entry in data : 
        print(entry.get("filename"))
        print(filename_brut)
        
    with open("src/transcription/tests/audio_brut.json", "w") as f:
        json.dump(data, f, indent=4)
    
    os.remove(audio_path)
    
if __name__ == "__main__":
    folder = Path("src/transcription/tests")
    for audio_path in folder.glob("*.wav"): 
        print(audio_path)
        VADe(audio_path, min_duration_on=1, min_duration_off=2)    
