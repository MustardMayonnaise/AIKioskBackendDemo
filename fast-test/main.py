from fastapi import FastAPI, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from services.tts_service import TTSService
from services.stt_service import STTService
from services.llm_service import LLMService
from core.log_config import setup_logging
from pathlib import Path
from typing import Iterator
import os
import tempfile
import uvicorn
import logging
import time


setup_logging()

# 모델 초기화(무거움)
logging.info("Initializing services...")
chatbot = LLMService()
chatbot.load_gemma_quant()
stt_service = STTService()
tts_service = TTSService()
# FastAPI 앱 생성
app = FastAPI()
logging.info("Services initialized successfully.")



app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
ROLE_FILES = [
    BASE_DIR / "markdown" / "agent.md",
    BASE_DIR / "markdown" / "menu.md",
]

def load_role_text() -> str:
    chunks = []
    for role_file in ROLE_FILES:
        if not role_file.exists():
            raise FileNotFoundError(f"Role file not found: {role_file}")
        chunks.append(role_file.read_text(encoding="utf-8").strip())
    return "\n\n".join(chunks)


def iter_audio_chunks(audio_data: bytes, chunk_size: int = 4096) -> Iterator[bytes]:
    for i in range(0, len(audio_data), chunk_size):
        yield audio_data[i:i + chunk_size]


@app.get("/")
def read_root():
    logging.info("Received request at root endpoint.")
    return {"Hello": "World"}

@app.get("/tts")
def get_tts(text: str):
    request_start = time.perf_counter()
    logging.info("Received TTS request (chars=%d)", len(text))

    try:
        audio_data = tts_service.text_to_speech(text)
        audio_ready_ms = (time.perf_counter() - request_start) * 1000
        logging.info("TTS audio generated in %.2f ms (bytes=%d)", audio_ready_ms, len(audio_data))

        response = Response(content=audio_data, media_type="audio/wav")
        response_ready_ms = (time.perf_counter() - request_start) * 1000
        logging.info("TTS response prepared in %.2f ms", response_ready_ms)
        return response
    except Exception:
        elapsed_ms = (time.perf_counter() - request_start) * 1000
        logging.exception("TTS request failed after %.2f ms (chars=%d)", elapsed_ms, len(text))
        raise

@app.post("/stt")
def post_stt(audio_file: UploadFile):
    logging.info("Received STT request.")
    audio_data = audio_file.file.read()
    suffix = os.path.splitext(audio_file.filename or "recording.webm")[1] or ".webm"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_audio:
        temp_audio.write(audio_data)
        temp_path = temp_audio.name
    try:
        
        text = stt_service.speech_to_text(temp_path)
        logging.info(f"Speech to text conversion completed. Extracted text: {text}")
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass

    return {"text": text}

@app.post("/llm")
def post_llm(text: str):
    logging.info("Received LLM request (chars=%d)", len(text))
    # prompts = [
    #         {
    #             "role": "system",
    #             "content": [{"type": "text", "text": load_role_text()}]
    #         },
    #         {
    #             "role": "user",
    #             "content": [{"type": "text", "text": text}]
    #         }
    # ]
    prompts = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "Your role is Mcdonald's kiosk. Answer the user's questions based on the menu and your role. If you don't know the answer, say you don't know. and You must speak in Korean. "}]
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": text}]
            }
    ]
    answer = chatbot.question(prompts)
    logging.info("LLM generated answer (chars=%d): %s", len(answer), answer)
    return {"message": answer}

@app.post("/chat")
def post_chat(audio_file: UploadFile):
    request_start = time.perf_counter()
    logging.info("Received Chat request with audio file: %s (size=%d bytes)", audio_file.filename, audio_file.spool_max_size)

    audio_data = audio_file.file.read()
    suffix = os.path.splitext(audio_file.filename or "recording.webm")[1] or ".webm"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_audio:
        temp_audio.write(audio_data)
        temp_path = temp_audio.name

    try:
        stt_start = time.perf_counter()
        user_text = stt_service.speech_to_text(temp_path)
        stt_ms = (time.perf_counter() - stt_start) * 1000
        logging.info("Chat STT completed in %.2f ms", stt_ms)

        llm_start = time.perf_counter()
        try:
            role_text = load_role_text()
        except FileNotFoundError as exc:
            logging.exception("Role markdown file is missing.")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        # prompts = [
        #     {
        #         "role": "system",
        #         "content": [{"type": "text", "text": role_text}],
        #     },
        #     {
        #         "role": "user",
        #         "content": [{"type": "text", "text": user_text}],
        #     },
        # ]
        logging.info("Sending chat request with user text (chars=%d): %s", len(user_text), user_text)
        prompts = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "Your role is Mcdonald's kiosk. Answer the user's questions based on the menu and your role. If you don't know the answer, say you don't know. and You must speak in Korean."}]
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": user_text}]
            }
        ]
        answer = chatbot.question(prompts)
        llm_ms = (time.perf_counter() - llm_start) * 1000
        logging.info("Chat LLM completed in %.2f ms", llm_ms)

        tts_start = time.perf_counter()
        wav_bytes = tts_service.text_to_speech(answer)
        tts_ms = (time.perf_counter() - tts_start) * 1000
        total_ms = (time.perf_counter() - request_start) * 1000

        logging.info(
            "Chat pipeline completed in %.2f ms (stt=%.2f ms, llm=%.2f ms, tts=%.2f ms, user_chars=%d, answer_chars=%d)",
            total_ms,
            stt_ms,
            llm_ms,
            tts_ms,
            len(user_text),
            len(answer),
        )

        return StreamingResponse(iter_audio_chunks(wav_bytes), media_type="audio/wav")
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
