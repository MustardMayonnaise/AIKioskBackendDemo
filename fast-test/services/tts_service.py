from melo.api import TTS
import io
from typing import Any, Dict

import numpy as np
import soundfile as sf
import torch

class TTSService:
    def __init__(self):
        # Select device with fallbacks: CUDA -> MPS -> CPU
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        self.tts = TTS(language='KR', device=device)
        hps: Any = getattr(self.tts, "hps", None)
        data_cfg: Any = getattr(hps, "data", None)

        spk2id = getattr(data_cfg, "spk2id", {})
        self.speaker_ids: Dict[str, int] = dict(spk2id) if spk2id else {"KR": 0}
        self.sample_rate = int(getattr(data_cfg, "sampling_rate", 24000))

    def text_to_speech(self, text: str) -> bytes:
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
        return wav_bytes