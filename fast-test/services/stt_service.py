from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import soundfile as sf
import torch
import numpy as np
import io
from pathlib import Path
from typing import Union
import tempfile
import subprocess
import shutil
import os

TARGET_SR = 16000

class STTService:
    def __init__(self):
        self.processor = Wav2Vec2Processor.from_pretrained("kresnik/wav2vec2-large-xlsr-korean")
        self.model = Wav2Vec2ForCTC.from_pretrained("kresnik/wav2vec2-large-xlsr-korean").to("cuda")

    def resample_audio(self, audio, orig_sr, target_sr=TARGET_SR):
        if orig_sr == target_sr:
            return audio.astype(np.float32)
        new_length = int(len(audio) * target_sr / orig_sr)
        old_indices = np.arange(len(audio))
        new_indices = np.linspace(0, len(audio) - 1, new_length)
        return np.interp(new_indices, old_indices, audio).astype(np.float32)

    def transcribe_audio(self, speech: np.ndarray) -> str:
        inputs = self.processor(speech, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
        input_values = inputs.input_values.to("cuda")
        with torch.no_grad():
            logits = self.model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        return self.processor.batch_decode(predicted_ids)[0].strip()

    def _read_audio_any(self, audio: Union[bytes, str, Path]):
        # 1) 기본 경로: soundfile 직접 읽기
        try:
            if isinstance(audio, (str, Path)):
                return sf.read(str(audio))
            if isinstance(audio, (bytes, bytearray)):
                return sf.read(io.BytesIO(audio))
            raise TypeError(f"Unsupported audio input type: {type(audio)}")
        except Exception:
            # 2) webm/opus 등 실패 시 ffmpeg fallback
            if shutil.which("ffmpeg") is None:
                raise RuntimeError("오디오 디코딩 실패. ffmpeg가 필요합니다 (webm/opus 지원).")

            if isinstance(audio, (bytes, bytearray)):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as src:
                    src.write(audio)
                    src_path = src.name
                cleanup_src = True
            else:
                src_path = str(audio)
                cleanup_src = False

            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as dst:
                dst_path = dst.name

            try:
                subprocess.run(
                    ["ffmpeg", "-y", "-i", src_path, "-ac", "1", "-ar", str(TARGET_SR), dst_path],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                )
                return sf.read(dst_path)
            finally:
                if cleanup_src and os.path.exists(src_path):
                    os.remove(src_path)
                if os.path.exists(dst_path):
                    os.remove(dst_path)

    def speech_to_text(self, audio: Union[bytes, str, Path]) -> str:
        audio_array, sr = self._read_audio_any(audio)

        if audio_array.ndim > 1:
            audio_array = np.mean(audio_array, axis=1)

        audio_array = self.resample_audio(audio_array, orig_sr=sr)
        return self.transcribe_audio(audio_array)