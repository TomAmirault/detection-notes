
import torch
import whisper
import librosa
from sklearn.cluster import KMeans
import numpy as np


def diarization_with_whisper(audio_path, n_clusters=2, model_size="tiny", prompt=""):
    
    model = whisper.load_model(model_size)  

    audio, sr = librosa.load(audio_path, sr=16000)
    
    result = model.transcribe(audio_path, initial_prompt=prompt)

    list_start = []
    list_end = []
    list_text = []
    list_spectrogram = []

    for segment in result["segments"]:
        
        start_sec = segment["start"]
        end_sec = segment["end"]
        list_start.append(start_sec)
        list_end.append(end_sec)
        start_sample = int(start_sec * sr)
        end_sample = int(end_sec * sr)
        audio_segment = audio[start_sample:end_sample]
        
        list_text.append(segment["text"])

        # Transformer en tensor et normaliser
        audio_segment = torch.from_numpy(audio_segment).float()
        audio_segment = whisper.pad_or_trim(audio_segment)  # ajuste si trop court ou long


        audio_segment = audio_segment.unsqueeze(0)  # ajouter la dimension batch
        with torch.no_grad():
            mel = whisper.log_mel_spectrogram(audio_segment)
            embeddings = model.encoder(mel)
        segment_emb = embeddings.mean(dim=1).squeeze(0).numpy()  
        list_spectrogram.append(segment_emb)
        
    X = np.stack(list_spectrogram)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X)

    return list_start, list_end, list_text, labels


    
    