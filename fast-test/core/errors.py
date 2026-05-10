from enum import Enum
from typing import Any, Optional


class ErrorCode(str, Enum):
    STT_MODEL_INIT_FAILED = "STT-5001"
    STT_INVALID_INPUT = "STT-4001"
    STT_DECODE_FAILED = "STT-4221"
    STT_TRANSCRIBE_FAILED = "STT-5002"

    TTS_MODEL_INIT_FAILED = "TTS-5001"
    TTS_INVALID_INPUT = "TTS-4001"
    TTS_GENERATE_FAILED = "TTS-5021"
    TTS_EMPTY_WAVEFORM = "TTS-4221"
    TTS_ENCODE_FAILED = "TTS-5002"
    TTS_BAD_PAYLOAD = "TTS-5022"

    COMMON_INTERNAL_ERROR = "COMMON-5000"


class ServiceError(Exception):
    def __init__(
        self,
        code: ErrorCode,
        message: str,
        http_status: int,
        details: Optional[Any] = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.http_status = http_status
        self.details = details

    def to_dict(self) -> dict:
        payload = {
            "code": self.code.value,
            "message": self.message,
        }
        if self.details is not None:
            payload["details"] = self.details
        return payload


class STTServiceError(ServiceError):
    pass


class TTSServiceError(ServiceError):
    pass