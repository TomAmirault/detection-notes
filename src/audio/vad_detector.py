from pathlib import Path
from pyannote.audio import Model
from pyannote.audio.pipelines import VoiceActivityDetection
from datetime import datetime, timedelta

import torchaudio
import json
import re
import os

model_segm = Model.from_pretrained(
"pyannote/segmentation-3.0")

pipeline = VoiceActivityDetection(segmentation=model_segm)

def speech_activity_splitter(audio_path, min_duration_on=3.0, min_duration_off=2.0):
    """
    Performs voice activity detection (VAD) on an audio file, 
    segments the audio into speech parts, and saves each segment as a WAV file.
    
    Information about each segment is recorded in a JSON file.

    Args:
        audio_path (Path): Path to the audio file to process.
        min_duration_on (float): Minimum duration of a speech segment to keep (in seconds).
        min_duration_off (float): Minimum duration of a pause to consider it as a separation (in seconds).
    """
    
    hyper_parameters = {
    "min_duration_on": min_duration_on,
    
    "min_duration_off": min_duration_off
    }
    
    pipeline.instantiate(hyper_parameters)
    vad = pipeline(audio_path)

    log_path = os.path.join("src/transcription/tmp", "transcriptions_log.json")
    logs_folder = os.path.dirname(log_path)
    os.makedirs(logs_folder, exist_ok=True)


    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8") as f:
            try:
                logs = json.load(f)
            except json.JSONDecodeError:
                logs = []
    else:
        logs = []
        
        
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
        

    logs.sort(key=lambda x: datetime.strptime(x["start_time"], "%Y-%m-%d %H:%M:%S"))


    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=4, ensure_ascii=False)

    with open("src/transcription/tests/audio_brut.json", "r") as f:
        data = json.load(f)
    
    data = [entry for entry in data if entry.get("filename") != filename_brut]

        
    with open("src/transcription/tests/audio_brut.json", "w") as f:
        json.dump(data, f, indent=4)
    
    os.remove(audio_path)
    
    
if __name__ == "__main__":
    
    folder = Path("src/transcription/tests")

    files = list(folder.glob("*.wav"))

    files_sorted = sorted(
        files,
        key=lambda f: int(re.search(r"chunk_(\d+)_", f.name).group(1))
    )

    for audio_path in files_sorted: 
        speech_activity_splitter(audio_path, min_duration_on=1, min_duration_off=2)    

