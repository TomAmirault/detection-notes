import whisper
import sounddevice as sd
import soundfile as sf
import numpy as np
import noisereduce as nr

# model_size = "tiny", "base", "small", "medium", "turbo", "large"

def record_and_transcribe_loop(duration=10, model_size="tiny", prompt=""):
    model = whisper.load_model(model_size)  

    try:
        k = 1
        while True:
            filename = f"record_chunk_{k}.wav"
            if k == 1 :
                print(f"Parlez maintenant")
            recording = sd.rec(int(duration * 16000), samplerate=16000, channels=1, dtype="float32")
            sd.wait()
            recording = np.squeeze(recording)
            
            # Réduction du bruit
            reduced_noise = nr.reduce_noise(y=recording, sr=16000)
            sf.write(filename, reduced_noise, 16000)

            result = model.transcribe(filename, initial_prompt=prompt)
            print(f"Transcription [{k}] :", result["text"], "\n")

            k = k+1

    except KeyboardInterrupt:
        print("\n Arrêt demandé par l'utilisateur (Ctrl+C).")