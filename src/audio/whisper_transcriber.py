import whisper
import warnings
import json
import sys
import os
import numpy as np
from pathlib import Path
from typing import Tuple, Union, Optional
REPO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if REPO_PATH not in sys.path:
    sys.path.insert(0, REPO_PATH)
from src.audio.audio_cleaner import clean_audio_transcription
from src.audio.dictionary.prompts import WHISPER_PROMPT
from src.audio.dictionary.vocabulary import KNOWN_ABBREVIATIONS, KNOWN_CITY, KNOWN_NAMES


# 'tiny',  'base', 'small', 'medium', 'large-v1', 'large-v2', 'large-v3', 'large', 'large-v3-turbo', 'turbo'
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU")
model = whisper.load_model("large-v3-turbo")

def whisper_transcribe(
    audio_path: Union[str, Path],
    pause: bool = True
) -> Optional[Tuple[str, str]]:
    """
    Transcribes an audio file to text using a Whisper model.

    Args:
        audio_path (Path or str): Path to the audio file to transcribe.
        pause (bool, optional): If True, performs transcription and logs results. 
                                If False, removes existing logs for the file and deletes the audio. Defaults to True.
    Returns:
        Optional[Tuple[str, str]]: 
            Tuple of (raw transcription, cleaned transcription) if transcribed, 
            otherwise None.
    """
    
    log_path = os.path.join("src/audio/tmp", "transcriptions_log.json")

    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8") as f:
            try:
                logs = json.load(f)
            except json.JSONDecodeError:
                logs = []
    else:
        logs = []
    
    filename_brut = audio_path.name

    entry = next((item for item in logs if item["filename"] == filename_brut ), None)

    if entry and entry.get("transcription"):
        return
    
    if pause :

        result = model.transcribe(str(audio_path), fp16=False, prompt=WHISPER_PROMPT.format(
        KNOWN_ABBREVIATIONS=KNOWN_ABBREVIATIONS,
        KNOWN_NAMES=KNOWN_NAMES,
        KNOWN_CITY=KNOWN_CITY
    ))
        
        predicted_sentence = result["text"]

        entry["transcription"] = predicted_sentence
        cleaned = clean_audio_transcription(predicted_sentence)
        entry["transcription_clean"] = cleaned
        
        list_conf = []
        for seg in result["segments"]:
            conf = np.exp(seg["avg_logprob"])  
            list_conf.append(conf)
        avg_conf = sum(list_conf) / len(list_conf)
        avg_conf = round(avg_conf, 2)
        
        entry["score"] = avg_conf


        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(logs, f, ensure_ascii=False, indent=4)
        
    else :
        
        with open(log_path, "r") as f:
            data = json.load(f)
        
        data = [entry for entry in data if entry.get("filename") != filename_brut]

            
        with open(log_path, "w") as f:
            json.dump(data, f, indent=4)
        
        os.remove(audio_path)
            
        return

    return predicted_sentence, cleaned

