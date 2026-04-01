from melo.api import TTS
import io
from typing import Any, Dict
import numpy as np
import soundfile as sf
import torch
import time
import logging

class TTSService:
    def __init__(self):
        logging.info("Initializing TTSService...")
        if torch.cuda.is_available():
            device = "cuda"
            logging.info("CUDA is available. Using GPU for TTS.")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
            logging.info("MPS is available. Using Apple Silicon GPU for TTS.")
        else:
            device = "cpu"
            logging.info("No GPU available. Using CPU for TTS.")

        self.tts = TTS(language='KR', device=device)
        hps: Any = getattr(self.tts, "hps", None)
        data_cfg: Any = getattr(hps, "data", None)

        spk2id = getattr(data_cfg, "spk2id", {})
        self.speaker_ids: Dict[str, int] = dict(spk2id) if spk2id else {"KR": 0}
        self.sample_rate = int(getattr(data_cfg, "sampling_rate", 24000))
        logging.info(f"TTSService initialized with device: {device}, sample_rate: {self.sample_rate}, speaker_ids: {self.speaker_ids}")
        
        
    def text_to_speech(self, text: str) -> bytes:
        start_time = time.perf_counter()
        logging.info(f"Converting text to speech: {text}")

        try:
            speaker_id = self.speaker_ids.get("KR", next(iter(self.speaker_ids.values())))
            tts_to_file_fn: Any = getattr(self.tts, "tts_to_file")
            audio_data = tts_to_file_fn(text, speaker_id=speaker_id, output_path=None, speed=1.5, quiet=True)

            if isinstance(audio_data, tuple):
                audio_data = audio_data[0]

            if isinstance(audio_data, torch.Tensor):
                audio_data = audio_data.detach().cpu().numpy()

            if not isinstance(audio_data, np.ndarray):
                audio_data = np.asarray(audio_data, dtype=np.float32)

            buffer = io.BytesIO()
            sf.write(buffer, audio_data, self.sample_rate, format='WAV')
            wav_bytes = buffer.getvalue()

            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logging.info("Text to speech conversion completed in %.2f ms (chars=%d, bytes=%d)", elapsed_ms, len(text), len(wav_bytes))
            return wav_bytes
        except Exception:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logging.exception("Text to speech conversion failed after %.2f ms (chars=%d)", elapsed_ms, len(text))
            raise