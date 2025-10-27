from pathlib import Path
from datetime import datetime
from threading import Event

import sounddevice as sd
import soundfile as sf
import numpy as np
import noisereduce as nr

import os
import json
import threading

# To list available audio input devices :
# python3 -m sounddevice

def record_audio_segments(
    duration: float,
    stop_event: Event,
    device: int = 0,
    bruit_reduction: bool = True,
    samplerate: int = 16000
) -> None:
    """
    Continuously records consecutive audio segments and saves them in a temporary folder.
    Recording stops either manually with Ctrl+C or automatically after `duration` seconds per segment.

    Args:
        duration (float): Length of each audio segment in seconds.
        stop_event (threading.Event): Event used to stop the recording loop externally.
        device (int, optional): Audio input device index. Defaults to 0.
        bruit_reduction (bool, optional): If True, applies noise reduction to the recorded audio. Defaults to True.
        samplerate (int, optional): Audio sampling rate in Hz. Defaults to 16000.
    """
    
    os.makedirs("src/audio/tests", exist_ok=True)
    log_path = os.path.join("src/audio/tests", "audio_brut.json")

    if os.path.exists(log_path):
        os.remove(log_path)

    k = 1

    print("Speech recording.")

    log= []

    try:
        with sd.InputStream(samplerate=samplerate, channels=1, dtype='float32', device=device) as stream:
            while not stop_event.is_set():
                
                
                audio_folder = "src/audio/tests"

                existing_files = {f for f in os.listdir(audio_folder) if f.endswith(".wav")}


                log = [entry for entry in log if entry["filename"] in existing_files]


                with open(log_path, "w", encoding="utf-8") as f:
                    json.dump(log, f, indent=4, ensure_ascii=False)
                       
                       
                start_time = datetime.now()
                safe_time = start_time.strftime("%Y%m%d_%H%M%S")
                filename = f"record_chunk_{k}_{safe_time}.wav"
                filepath = os.path.join("src/audio/tests", filename)

                frames = []
                while (datetime.now() - start_time).total_seconds() < duration:
                    block = stream.read(1024)[0]
                    frames.append(block)

                recording = np.concatenate(frames, axis=0).squeeze()

                if bruit_reduction:
                    if np.any(np.abs(recording) > 1e-8):
                        recording = nr.reduce_noise(y=recording, sr=samplerate)
                sf.write(filepath, recording, samplerate)
                end_time = datetime.now()
            
            

                if os.path.exists(log_path):
                    with open(log_path, "r", encoding="utf-8") as f:
                        log = json.load(f)
                else:
                    log = []
                      
                entry = {
                    "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "end_time": end_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "filename": filename
                }
                
                log.append(entry)
                

                with open(log_path, "w", encoding="utf-8") as f:
                    json.dump(log, f, indent=4, ensure_ascii=False)

                k += 1

    except KeyboardInterrupt:
        print("\nUser requested shutdown (Ctrl+C).")


        if 'frames' in locals() and len(frames) > 0:
            recording = np.concatenate(frames, axis=0).squeeze()
            if bruit_reduction:
                recording = nr.reduce_noise(y=recording, sr=samplerate)

            sf.write(filepath, recording, samplerate)
            
            end_time = datetime.now()

            entry = {
                "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
                "end_time": end_time.strftime("%Y-%m-%d %H:%M:%S"),
                "filename": filename
            }

            with open(log_path, "w", encoding="utf-8") as f:
                json.dump([entry], f, indent=4, ensure_ascii=False)
                     
           
if __name__ == "__main__":

    folder_tests = Path("src/audio/tests")
    
    for f in folder_tests.glob("*.wav"):
        try:
            f.unlink()
        except Exception as e:
            print(f"Could not remove {f}: {e}")
    try:
        stop_event = threading.Event()
        enregistrement_thread = threading.Thread(target = record_audio_segments, args=(5, stop_event, 1))
        enregistrement_thread.start()
    except KeyboardInterrupt:
        print("User requested shutdown (Ctrl+C).")
        stop_event.set()
        enregistrement_thread.join()
        print("End.")