from audio_cleaner import clean_audio_transcription
from dictionary.prompts import WHISPER_PROMPT
from pathlib import Path

import numpy as np

import whisper
import os
import json

# 'tiny',  'base', 'small', 'medium', 'large-v1', 'large-v2', 'large-v3', 'large', 'large-v3-turbo', 'turbo'
model = whisper.load_model("large-v3-turbo")

def whisper_transcribe(audio_path, prompt=WHISPER_PROMPT, pause=True):
    """
    Transcribes an audio file to text using a Whisper model.

    Args:
        audio_path (Path or str): Path to the audio file to transcribe.
        prompt (str, optional): Initial prompt to guide transcription. Defaults to WHISPER_PROMPT.
        pause (bool, optional): If True, performs transcription and logs results. 
                                If False, removes existing logs for the file and deletes the audio. Defaults to True.
    """
    
    log_path = os.path.join("src/transcription/tmp", "transcriptions_log.json")


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

        result = model.transcribe(str(audio_path), prompt=prompt)
        
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

if __name__ == "__main__":
    folder = Path("src/transcription/tmp")
    for audio_path in folder.glob("*.wav"): 
        print(audio_path)
        whisper_transcribe(audio_path)