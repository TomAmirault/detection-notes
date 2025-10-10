
import torch
import whisper
import librosa
from sklearn.cluster import KMeans
import numpy as np
from pyannote.audio import Pipeline
import whisper
from pydub import AudioSegment

from transformers import AutoModelForCTC, Wav2Vec2Processor
import torch
import torchaudio

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = AutoModelForCTC.from_pretrained("bhuang/asr-wav2vec2-french").to(device)
processor = Wav2Vec2Processor.from_pretrained("bhuang/asr-wav2vec2-french")
model_sample_rate = processor.feature_extractor.sampling_rate


def diarization_with_whisper(audio_path, num_speakers=None, min_speakers=None, max_speakers=None):

    if num_speakers is not None and (min_speakers is not None or max_speakers is not None):
        raise ValueError("Tu ne peux pas utiliser à la fois 'num_speakers' et ('min_speakers' ou 'max_speakers').")

    
    audio = AudioSegment.from_file(audio_path)

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1", 
        token="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")


    if num_speakers is not None:
        output = pipeline(audio_path, num_speakers=num_speakers)
    else:
        output = pipeline(audio_path, min_speakers=min_speakers, max_speakers=max_speakers)
        
        
    diarization = output.speaker_diarization


    list_start = []
    list_end = []
    list_speaker = []
    i=0
    for turn, _, speaker in diarization.itertracks(yield_label=True):

        list_start.append(turn.start)
        list_end.append(turn.end)
        list_speaker.append(speaker)
        
        if len(list_speaker) == 1:
            list_speaker.append(speaker)
        

        if list_speaker[-2] != speaker and list_end[-2]-list_start[0] > 1:
            
            i=i+1
            start_time = list_start[0] * 1000   
            end_time = (list_end[-2]) * 1000  

            segment = audio[start_time:end_time]

            segment.export(f"tmp/audio/segment_{i}.wav", format="wav")
            
            
            
            
            wav_path =f"tmp/audio/segment_{i}.wav"  # path to your audio file
            waveform, sample_rate = torchaudio.load(wav_path)
            waveform = waveform.squeeze(axis=0)  # mono

            # resample
            if sample_rate != model_sample_rate:
                resampler = torchaudio.transforms.Resample(sample_rate, model_sample_rate)
                waveform = resampler(waveform)

            # normalize
            input_dict = processor(waveform, sampling_rate=model_sample_rate, return_tensors="pt")

            with torch.inference_mode():
                logits = model(input_dict.input_values.to(device)).logits

            # decode
            predicted_ids = torch.argmax(logits, dim=-1)
            predicted_sentence = processor.batch_decode(predicted_ids)[0]

            
            print(f"{list_start[0]:.1f}s - {list_end[-2]:.1f}s : {list_speaker[-2]} : {predicted_sentence}")
            
            
            list_start = [list_start[-1]]
            list_end = [list_end[-1]]
            list_speaker = [list_speaker[-1]]

    i=i+1
    start_time = list_start[0] * 1000   
    end_time = list_end[-1] * 1000  

    segment = audio[start_time:end_time]

    segment.export(f"tmp/audio/segment_{i}.wav", format="wav")

    wav_path =f"tmp/audio/segment_{i}.wav"  # path to your audio file
    waveform, sample_rate = torchaudio.load(wav_path)
    waveform = waveform.squeeze(axis=0)  # mono

    # resample
    if sample_rate != model_sample_rate:
        resampler = torchaudio.transforms.Resample(sample_rate, model_sample_rate)
        waveform = resampler(waveform)

    # normalize
    input_dict = processor(waveform, sampling_rate=model_sample_rate, return_tensors="pt")

    with torch.inference_mode():
        logits = model(input_dict.input_values.to(device)).logits

    # decode
    predicted_ids = torch.argmax(logits, dim=-1)
    predicted_sentence = processor.batch_decode(predicted_ids)[0]

    print(f"{list_start[0]:.1f}s - {list_end[-1]:.1f}s : {list_speaker[-1]} : {predicted_sentence}")



    
    