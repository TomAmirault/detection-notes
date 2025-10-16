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
from transformers import AutoModelForCTC, Wav2Vec2Processor
import os
import json
from datetime import datetime, timedelta
import whisper

from nettoyage_audio import nettoyer_transcription_audio


model = whisper.load_model("large-v3-turbo")

def transcribe_whisper_clean(audio_path, prompt="Abréviations officielles (ne pas développer ; corrige variantes proches vers la forme officielle): SNCF, ABC, RSD, TIR, PF, GEH, SMACC, COSE, TRX, VPL, MNV, N-1, COSE-P"):
    """
    Transcrit un fichier audio en texte à l’aide d’un modèle Whisper.

    Cette fonction charge un modèle Whisper préentraîné de la taille spécifiée,
    puis effectue la transcription du fichier audio fourni.  
    Elle peut également utiliser un prompt initial pour guider la transcription

    """
    
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

    entry = next((item for item in logs if item["filename"] == filename_brut ), None)

    if entry and entry.get("transcription"):
        return
    
    result = model.transcribe(str(audio_path), prompt = prompt, language="fr")
    
    predicted_sentence = result["text"]

    entry["transcription"] = predicted_sentence
    cleaned = nettoyer_transcription_audio(predicted_sentence)
    entry["transcription_clean"] = cleaned

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=4)

    # Return raw and cleaned transcriptions so callers can insert them in DB
    return predicted_sentence, cleaned

if __name__ == "__main__":
    folder = Path("src/transcription/tmp")
    for audio_path in folder.glob("*.wav"): 
        print(audio_path)
        transcribe_whisper_clean(audio_path)