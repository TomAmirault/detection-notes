import whisper
import torch
import torchaudio
from transformers import AutoModelForCTC, Wav2Vec2Processor


# model_size = "tiny", "base", "small", "medium", "turbo", "large"

def transcribe_audio(audio_path, model_size="tiny", prompt=""):
    """
    Transcrit un fichier audio en texte à l’aide d’un modèle Whisper.

    Cette fonction charge un modèle Whisper préentraîné de la taille spécifiée,
    puis effectue la transcription du fichier audio fourni.  
    Elle peut également utiliser un prompt initial pour guider la transcription
    (utile pour le contexte ou la cohérence des phrases).

    Parameters
    ----------
    audio_path : str
        Chemin du fichier audio à transcrire (par ex. "tmp/record_chunk_1.wav").
    model_size : str, optional (default="tiny")
        Taille du modèle Whisper à charger.  
        Exemples valides : "tiny", "base", "small", "medium", "large".  
        Plus le modèle est grand, plus la transcription est précise mais lente.
    prompt : str, optional (default="")
        Texte initial servant de contexte au modèle pour améliorer la transcription
        (utile si le contenu audio est dans un domaine spécifique ou s’il contient des noms propres).

    Returns
    -------
    str
        Le texte transcrit à partir de l’audio.
    """
    model = whisper.load_model(model_size, initial_prompt=prompt)
    result = model.transcribe(audio_path)
    return result["text"]



def transcribe_w2v2(audio_path):
    """
    Transcrit un fichier audio en texte à l’aide d’un modèle Whisper.

    Cette fonction charge un modèle Whisper préentraîné de la taille spécifiée,
    puis effectue la transcription du fichier audio fourni.  
    Elle peut également utiliser un prompt initial pour guider la transcription
    (utile pour le contexte ou la cohérence des phrases).

    Parameters
    ----------
    audio_path : str
        Chemin du fichier audio à transcrire (par ex. "tmp/record_chunk_1.wav").
    model_size : str, optional (default="tiny")
        Taille du modèle Whisper à charger.  
        Exemples valides : "tiny", "base", "small", "medium", "large".  
        Plus le modèle est grand, plus la transcription est précise mais lente.
    prompt : str, optional (default="")
        Texte initial servant de contexte au modèle pour améliorer la transcription
        (utile si le contenu audio est dans un domaine spécifique ou s’il contient des noms propres).

    Returns
    -------
    str
        Le texte transcrit à partir de l’audio.
    """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = AutoModelForCTC.from_pretrained("bhuang/asr-wav2vec2-french").to(device)
    processor = Wav2Vec2Processor.from_pretrained("bhuang/asr-wav2vec2-french")
    model_sample_rate = processor.feature_extractor.sampling_rate

    wav_path = audio_path
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

    return predicted_sentence