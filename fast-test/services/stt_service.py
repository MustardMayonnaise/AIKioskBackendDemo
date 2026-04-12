from faster_whisper import WhisperModel
from core.errors import ErrorCode, STTServiceError

class STTService:
    def __init__(self, model_size="large-v3", device="cuda"):
        try:
            self.model = WhisperModel(model_size, device=device)
        except Exception as exc:
            raise STTServiceError(
                code=ErrorCode.STT_MODEL_INIT_FAILED,
                message="STT model initialization failed",
                http_status=503,
                details={"model_size": model_size, "device": device, "reason": str(exc)},
            ) from exc

    def speech_to_text(self, audio_path):
        if not audio_path:
            raise STTServiceError(
                code=ErrorCode.STT_INVALID_INPUT,
                message="audio_path is required",
                http_status=400,
            )

        try:
            segments, info = self.model.transcribe(
                audio_path,
                language="ko",
                task="transcribe",
                beam_size=2,
                best_of=1,
                patience=1,
                temperature=0.0,
                condition_on_previous_text=True,
                word_timestamps=False,
                vad_filter=True,
            )
            del info
            return "".join(segment.text for segment in segments).strip()
        except ValueError as exc:
            raise STTServiceError(
                code=ErrorCode.STT_DECODE_FAILED,
                message="Audio decode failed",
                http_status=422,
                details={"reason": str(exc)},
            ) from exc
        except STTServiceError:
            raise
        except Exception as exc:
            raise STTServiceError(
                code=ErrorCode.STT_TRANSCRIBE_FAILED,
                message="STT transcription failed",
                http_status=500,
                details={"reason": str(exc)},
            ) from exc