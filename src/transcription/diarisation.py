
import torch
import whisper
import librosa
import numpy as np
from pyannote.audio import Pipeline
from pydub import AudioSegment
import torch
import torchaudio

# avoir Python 3.12.11
def diarization(audio_path, num_speakers=None, min_speakers=None, max_speakers=None):
    
    if num_speakers is not None and (min_speakers is not None or max_speakers is not None):
        raise ValueError("Tu ne peux pas utiliser à la fois 'num_speakers' et ('min_speakers' ou 'max_speakers').")

    audio = AudioSegment.from_file(audio_path)

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1", 
        token="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")

    if num_speakers is not None:
        output = pipeline(audio_path, num_speakers=num_speakers)
        
    elif min_speakers is not None or max_speakers is not None:
        output = pipeline(audio_path, min_speakers=min_speakers, max_speakers=max_speakers)
    else:
        output = pipeline(audio_path)
        
    diarization = output.speaker_diarization

    list_start = []
    list_end = []
    list_speaker = []
    
    segment_start = []
    segment_end = []
    segment_speaker = []
    
    for turn, _, speaker in diarization.itertracks(yield_label=True):

        list_start.append(turn.start)
        list_end.append(turn.end)
        list_speaker.append(speaker)
        
        if len(list_speaker) == 1:
            list_speaker.append(speaker) 

        if list_speaker[-2] != speaker and list_end[-2]-list_start[0] > 1:
            
            segment_start.append(list_start[0])
            segment_end.append(list_end[-2])
            segment_speaker.append(list_speaker[-2])
            
            #print(f"{list_start[0]:.1f}s - {list_end[-2]:.1f}s : {list_speaker[-2]}")
            
            list_start = [list_start[-1]]
            list_end = [list_end[-1]]
            list_speaker = [list_speaker[-1]]
            
    segment_start.append(list_start[0])
    segment_end.append(list_end[-1])
    segment_speaker.append(list_speaker[-1])
    
    #print(f"{list_start[0]:.1f}s - {list_end[-1]:.1f}s : {list_speaker[-1]}")
    
    return segment_start, segment_end, segment_speaker
    