import os
import torch
import torchaudio
import torchaudio.functional as F
import pandas as pd

def denoise_audio(waveform, noise_reduction=0.02):
    """Réduction simple du bruit : filtrage spectral par seuil."""
    # FFT
    spec = torch.fft.rfft(waveform)
    magnitude = torch.abs(spec)
    phase = torch.angle(spec)

    # Atténuation des composantes faibles (bruit)
    threshold = magnitude.mean() * noise_reduction
    magnitude = torch.where(magnitude > threshold, magnitude, magnitude * 0.1)

    # Reconstruction
    spec_denoised = magnitude * torch.exp(1j * phase)
    waveform_denoised = torch.fft.irfft(spec_denoised)
    return waveform_denoised