from fastapi import FastAPI, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from services.tts_service import TTSService
from services.stt_service import STTService
import os
import tempfile
import uvicorn

# 모델 초기화(무거움)
tts_service = TTSService()
stt_service = STTService()

# FastAPI 앱 생성
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/tts")
def get_tts(text: str):
    audio_data = tts_service.text_to_speech(text)
    return Response(content=audio_data, media_type="audio/wav")

@app.post("/stt")
def post_stt(audio_file: UploadFile):
    audio_data = audio_file.file.read()

    suffix = os.path.splitext(audio_file.filename or "recording.webm")[1] or ".webm"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_audio:
        temp_audio.write(audio_data)
        temp_path = temp_audio.name

    try:
        text = stt_service.speech_to_text(temp_path)
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass

    return {"text": text}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
