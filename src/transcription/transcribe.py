import whisper

# model_size = "tiny", "base", "small", "medium", "turbo", "large"

def transcribe_audio(audio_path, model_size="tiny", prompt=""):
    model = whisper.load_model(model_size, initial_prompt=prompt)
    result = model.transcribe(audio_path)
    return result["text"]