from omnivoice import OmniVoice
import torch
import numpy as np
import io
import wave
import logging

logging.basicConfig(level=logging.INFO)

class TTSService:
    def __init__(self):
        logging.info("Initializing TTSService...")
        self.model = OmniVoice.from_pretrained(
            "k2-fsa/OmniVoice",
            device_map="cuda:0",
            dtype=torch.float16
        )

    def _to_numpy_1d(self, audio_obj):
        if isinstance(audio_obj, list):
            if not audio_obj:
                raise ValueError("TTS model returned an empty list.")
            audio_obj = audio_obj[0]

        if torch.is_tensor(audio_obj):
            arr = audio_obj.detach().float().cpu().numpy()
        elif isinstance(audio_obj, np.ndarray):
            arr = audio_obj.astype(np.float32)
        else:
            raise TypeError(f"Unsupported audio type from TTS model: {type(audio_obj)}")

        if arr.ndim > 1:
            arr = np.mean(arr, axis=-1)

        return np.clip(arr, -1.0, 1.0).astype(np.float32)

    def text_to_speech(self, text: str) -> bytes:
        logging.info("Converting text to speech: %s", text)
        audio = self.model.generate(
            preprocess_prompt="announcer, clear voice, high quality",
            num_step=32,
            speed=1.3,
            language="Korean",
            instruct="female, young adult, high pitch",
            text=text
        )

        waveform = self._to_numpy_1d(audio)
        pcm16 = (waveform * 32767.0).astype(np.int16)

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)   # int16
            wf.setframerate(24000)
            wf.writeframes(pcm16.tobytes())

        wav_bytes = buf.getvalue()
        logging.info("Text to speech conversion completed (bytes=%d)", len(wav_bytes))
        return wav_bytes