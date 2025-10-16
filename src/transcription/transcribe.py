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

from nettoyage_audio import nettoyer_transcription_audio


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_ctc = AutoModelForCTC.from_pretrained("bhuang/asr-wav2vec2-french").to(device)
processor = Wav2Vec2Processor.from_pretrained("bhuang/asr-wav2vec2-french")
model_sample_rate = processor.feature_extractor.sampling_rate



def transcribe_w2v2_clean(audio_path):
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

    
    
    wav_path = audio_path
    waveform, sample_rate = torchaudio.load(wav_path)
    waveform = waveform.squeeze(axis=0)  # mono

    # resample
    if sample_rate != model_sample_rate:
        resampler = torchaudio.transforms.Resample(sample_rate, model_sample_rate)
        waveform = resampler(waveform)

    # normalize
    input_dict = processor(waveform, sampling_rate=model_sample_rate, return_tensors="pt")

    with torch.inference_mode():
        logits = model_ctc(input_dict.input_values.to(device)).logits

    # decode
    predicted_ids = torch.argmax(logits, dim=-1)
    predicted_sentence = processor.batch_decode(predicted_ids)[0]

    entry["transcription"] = predicted_sentence
    
    entry["transcription_clean"] = nettoyer_transcription_audio(predicted_sentence)

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    folder = Path("src/transcription/tmp")
    for audio_path in folder.glob("*.wav"): 
        print(audio_path)
        transcribe_w2v2_clean(audio_path)