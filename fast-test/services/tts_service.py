from omnivoice import OmniVoice
import io
import logging
from typing import Any
import numpy as np
import soundfile as sf
import torch
from core.errors import ErrorCode, TTSServiceError



logging.basicConfig(level=logging.INFO)

class TTSService:
    def __init__(self):
        logging.info("Initializing TTSService...")

        # Device fallback
        if torch.cuda.is_available():
            device_map = "cuda:0"
            dtype = torch.float16
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device_map = "mps"
            dtype = torch.float32
        else:
            device_map = "cpu"
            dtype = torch.float32

        self.sample_rate = 24000
        try:
            self.model = OmniVoice.from_pretrained(
                "k2-fsa/OmniVoice",
                device_map=device_map,
                dtype=dtype,
            )
        except Exception as exc:
            raise TTSServiceError(
                code=ErrorCode.TTS_MODEL_INIT_FAILED,
                message="TTS model initialization failed",
                http_status=503,
                details={"device_map": device_map, "reason": str(exc)},
            ) from exc

    def _unwrap_audio(self, obj: Any) -> Any:
        # OmniVoice 출력이 list/tuple 중첩인 경우를 안전하게 해제
        while isinstance(obj, (list, tuple)):
            if len(obj) == 0:
                raise TTSServiceError(
                    code=ErrorCode.TTS_EMPTY_WAVEFORM,
                    message="TTS model returned an empty list/tuple",
                    http_status=422,
                )
            obj = obj[0]
        return obj

    def _to_numpy_1d(self, audio_obj: Any) -> np.ndarray:
        try:
            audio_obj = self._unwrap_audio(audio_obj)

            if torch.is_tensor(audio_obj):
                arr = audio_obj.detach().float().cpu().numpy()
            elif isinstance(audio_obj, np.ndarray):
                arr = audio_obj.astype(np.float32)
            else:
                arr = np.asarray(audio_obj, dtype=np.float32)

            if arr.size == 0:
                raise TTSServiceError(
                    code=ErrorCode.TTS_EMPTY_WAVEFORM,
                    message="TTS waveform is empty",
                    http_status=422,
                )

            # [T, C] 또는 [C, T] 케이스를 mono로 정규화
            if arr.ndim == 2:
                if arr.shape[0] <= 4 and arr.shape[1] > arr.shape[0]:
                    arr = np.mean(arr, axis=0)
                else:
                    arr = np.mean(arr, axis=1)
            elif arr.ndim > 2:
                arr = np.ravel(arr)

            arr = np.clip(arr, -1.0, 1.0).astype(np.float32)
            return arr
        except TTSServiceError:
            raise
        except Exception as exc:
            raise TTSServiceError(
                code=ErrorCode.TTS_GENERATE_FAILED,
                message="Failed to normalize TTS waveform",
                http_status=502,
                details={"reason": str(exc)},
            ) from exc

    def text_to_speech(self, text: str) -> bytes:
        if not text or not text.strip():
            raise TTSServiceError(
                code=ErrorCode.TTS_INVALID_INPUT,
                message="text is required",
                http_status=400,
            )

        logging.info("Converting text to speech: %s", text)

        try:
            audio = self.model.generate(
                preprocess_prompt="announcer, clear voice, high quality",
                num_step=24,
                speed=1.1,
                language="Korean",
                text=text,
            )
        except Exception as exc:
            raise TTSServiceError(
                code=ErrorCode.TTS_GENERATE_FAILED,
                message="TTS generation failed",
                http_status=502,
                details={"reason": str(exc)},
            ) from exc

        waveform = self._to_numpy_1d(audio)

        buffer = io.BytesIO()
        try:
            sf.write(buffer, waveform, self.sample_rate, format="WAV", subtype="PCM_16")
        except Exception as exc:
            raise TTSServiceError(
                code=ErrorCode.TTS_ENCODE_FAILED,
                message="Failed to encode wav",
                http_status=500,
                details={"reason": str(exc)},
            ) from exc

        wav_bytes = buffer.getvalue()

        if len(wav_bytes) <= 44:
            raise TTSServiceError(
                code=ErrorCode.TTS_BAD_PAYLOAD,
                message="TTS wav payload too small",
                http_status=502,
                details={"size": len(wav_bytes)},
            )

        logging.info(
            "TTS done. waveform_shape=%s wav_bytes=%d",
            waveform.shape,
            len(wav_bytes),
        )
        return wav_bytes