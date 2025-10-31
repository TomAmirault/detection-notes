"""
This script continuously records audio segments from a microphone, detects
speech activity, splits audio files based on silence, and automatically 
transcribes spoken content into text using Whisper. Transcriptions are then
cleaned and stored in a database. The script also handles cleanup of old files
and recovery in case of interruption.

Main features:
- Continuous segmented audio recording (record_audio_segments)
- Voice activity detection (speech_activity_splitter)
- Transcription using Whisper (whisper_transcribe)
- Database insertion of transcripts (add_audio2db)
- Automatic cleanup of temporary files
- Pause/resume handling and multithreading support
"""

from pathlib import Path

import threading
import json
import time
import re
import concurrent.futures
import warnings
warnings.filterwarnings("ignore")

import sys
import os
REPO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if REPO_PATH not in sys.path:
    sys.path.insert(0, REPO_PATH)
from src.processing.add_data2db import add_audio2db
from src.audio.vad_detector import speech_activity_splitter
from src.audio.audio_recorder import record_audio_segments
from src.audio.whisper_transcriber import whisper_transcribe



# duration (float): Length of each audio segment in seconds.
RECORD_DURATION =20

# device (int, optional): Audio input device index. Defaults to 0.
# To list available audio input devices :
# python3 -m sounddevice
DEVICE_INDEX = 0

# min_duration_on (float): Minimum duration of a speech segment to keep (in seconds).
MIN_DURATION_ON = 3

# min_duration_off (float): Minimum duration of a pause to consider it as a separation (in seconds).
MIN_DURATION_OFF = 10


if __name__ == "__main__":
    try:
        pause_status = True
        stop_event = threading.Event()
        recording_thread = threading.Thread(target=record_audio_segments, args=(RECORD_DURATION, stop_event, DEVICE_INDEX))
        recording_thread.start()
        
        folder_tmp = Path("src/audio/tmp")
        folder_tests = Path("src/audio/tests")
        
        with open("src/audio/pause_status.json", "r") as f:
            cfg = json.load(f)
        cfg["pause"] = True  
        with open("src/audio/pause_status.json", "w") as f:
            json.dump(cfg, f)
            
        # Cleanup leftover recording chunks from previous runs
        for folder in (folder_tmp, folder_tests):
            for f in folder.glob("record_chunk*.wav"):
                try:
                    f.unlink()
                except Exception as e:
                    print(f"Could not remove {f}: {e}")

        # Cleanup JSON logs referencing record_chunk*.wav
        transcriptions_log = folder_tmp / "transcriptions_log.json"
        try:
            if transcriptions_log.exists():
                data = json.loads(transcriptions_log.read_text(encoding='utf-8'))
                before = len(data)
                data = [entry for entry in data if not entry.get('filename','').startswith('record_chunk')]
                after = len(data)
                if after != before:
                    transcriptions_log.write_text(json.dumps(data, ensure_ascii=False, indent=4), encoding='utf-8')
        except Exception as e:
            print(f"Could not clean {transcriptions_log}: {e}")

        audio_raw = folder_tests / "audio_brut.json"
        try:
            if audio_raw.exists():
                data = json.loads(audio_raw.read_text(encoding='utf-8'))
                before = len(data)
                data = [entry for entry in data if not entry.get('filename','').startswith('record_chunk')]
                after = len(data)
                if after != before:
                    audio_raw.write_text(json.dumps(data, ensure_ascii=False, indent=4), encoding='utf-8')
        except Exception as e:
            print(f"Could not clean {audio_raw}: {e}")
        
        while True:
            # Split test audio files into speech segments
            for audio_path in folder_tests.glob("*.wav"): 
                speech_activity_splitter(audio_path, MIN_DURATION_ON, MIN_DURATION_OFF)
                
            files = list(folder_tmp.glob("*.wav"))
            files_sorted = sorted(
                files,
                key=lambda x: (
                    int(re.search(r'record_chunk_(\d+)', x.name).group(1)),
                    int(re.search(r'segment_(\d+)', x.name).group(1)) if 'segment_' in x.name else 0
                )
            )
            
            with open("src/audio/pause_status.json", "r") as f:
                cfg = json.load(f)
            pause_status = cfg.get("pause", True)
            
            for audio_path in files_sorted:
                try:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(whisper_transcribe, audio_path, pause=pause_status)
                        try:
                            res = future.result(timeout=60*5)
                        except concurrent.futures.TimeoutError:
                            print(f"Timeout: transcription of {audio_path} exceeded {60*5} seconds. Retrying...")
                            continue  
                    if not res:
                        continue
                    raw, clean = res
                    add_audio2db(str(audio_path), raw, clean)
                except Exception as e:
                    print(f"Error during transcription/insertion for {audio_path}: {e}")

            time.sleep(2)
            
    except KeyboardInterrupt:
        print("User requested stop.")
    except Exception as e:
        print("Error:", e)
        
    stop_event.set()
    print("Waiting for recording loop to finish...")
    recording_thread.join()

    # Final segmentation and transcription after stop
    for audio_path in folder_tests.glob("*.wav"): 
        speech_activity_splitter(audio_path, MIN_DURATION_ON, MIN_DURATION_OFF)
      
    files = list(folder_tmp.glob("*.wav"))
    files_sorted = sorted(
        files,
        key=lambda x: (
            int(re.search(r'record_chunk_(\d+)', x.name).group(1)),  
            int(re.search(r'segment_(\d+)', x.name).group(1)) if 'segment_' in x.name else 0  
        )
    )
    
    for audio_path in files_sorted:
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(whisper_transcribe, audio_path, pause=pause_status)
                try:
                    res = future.result(timeout=60*5)
                except concurrent.futures.TimeoutError:
                    print(f"Timeout: transcription of {audio_path} exceeded {60*5} seconds. Retrying...")
                    continue  
            if not res:
                continue
            raw, clean = res
            add_audio2db(str(audio_path), raw, clean)
        except Exception as e:
            print(f"Error during transcription/insertion for {audio_path}: {e}")

    print("End of process.")