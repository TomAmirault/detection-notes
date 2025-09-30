import whisper

# model_size = "tiny", "base", "small", "medium", "turbo", "large"

def transcribe_audio(file_path, model_size="tiny"):
    model = whisper.load_model(model_size)
    result = model.transcribe(file_path)
    text_transcribe = result["text"]
    return text_transcribe