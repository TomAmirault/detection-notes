import os
import torch
import torchaudio
import torchaudio.functional as F
import pandas as pd

def normalize_volume(waveform, target_dBFS=-20.0):
    """Normalise le volume audio à un niveau cible (en dBFS)."""
    rms = waveform.pow(2).mean().sqrt()
    if rms > 0:
        current_dBFS = 20 * torch.log10(rms)
        gain = 10 ** ((target_dBFS - current_dBFS) / 20)
        waveform = waveform * gain
    return waveform