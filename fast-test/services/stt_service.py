from faster_whisper import WhisperModel

class STTService:
    def __init__(self, model_size="medium", device="auto"):
        self.model = WhisperModel(model_size, device=device)

    def speech_to_text(self, audio_path):
        segments, info = self.model.transcribe(audio_path)
        del info
        return "".join(segment.text for segment in segments).strip()
    
