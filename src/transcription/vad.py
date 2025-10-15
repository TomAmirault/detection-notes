from huggingface_hub import login
from pyannote.audio import Model
from pyannote.audio.pipelines import VoiceActivityDetection

# Remplace "TON_TOKEN" par ton token Hugging Face
login(token="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")


def VAD(audio_path, min_duration_on=2, min_duration_of=2):
    
    model = Model.from_pretrained(
    "pyannote/segmentation-3.0")

    pipeline = VoiceActivityDetection(segmentation=model)
    HYPER_PARAMETERS = {
    # Si un segment de parole détecté dure moins de 3 secondes, il sera ignoré.
    "min_duration_on": min_duration_on,
    # Si une pause est plus courte que 10 secondes, elle peut être remplie ou fusionnée avec les segments voisins.
    "min_duration_off": min_duration_of
    }
    pipeline.instantiate(HYPER_PARAMETERS)
    vad = pipeline(audio_path)
    segment_start = []
    segment_end = []
    for segment, _, _ in vad.itertracks(yield_label=True):
        #print(segment)
        segment_start.append(segment.start)
        segment_end.append(segment.end)
        
    return segment_start,segment_end

