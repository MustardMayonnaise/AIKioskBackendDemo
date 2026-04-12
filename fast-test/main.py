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


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    )

def main(stt_model_dir: str, host: str = "0.0.0.0", port: int = 8000):
    setup_logging()
    global stt_service, tts_service, chatbot, app
    logging.info("Initializing services...")
    stt_service = STTService()
    chatbot = LLMService()
    chatbot.load_gemma_quant()
    tts_service = TTSService()    
    uvicorn.run(app, host=host, port=port)
    logging.info("Services initialized successfully.")




BASE_DIR = Path(__file__).resolve().parent
ROLE_FILES = [
    BASE_DIR / "markdown" / "agent.md",
    # BASE_DIR / "markdown" / "menu.md",
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
        return Response(
            content=audio_data,
            media_type="audio/wav",
            headers={"Content-Disposition": "inline; filename=tts.wav"}
        )
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - request_start) * 1000
        logging.exception("TTS request failed after %.2f ms (chars=%d)", elapsed_ms, len(text))
        raise HTTPException(status_code=500, detail=f"TTS failed: {exc}") from exc

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
        logging.info("Speech to text conversion completed. Extracted text: %s", text)
        return {"text": text}
    except Exception as exc:
        logging.exception("STT failed.")
        raise HTTPException(status_code=400, detail=f"STT decode/transcribe failed: {exc}") from exc
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass

@app.post("/llm")
def post_llm(text: str):
    logging.info("Received LLM request (chars=%d)", len(text))
    prompts = [
            {
                "role": "system",
                "content": [{"type": "text", "text": load_role_text()}]
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
    logging.info("Received Chat request.")

    audio_data = audio_file.file.read()
    suffix = os.path.splitext(audio_file.filename or "recording.webm")[1] or ".webm"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_audio:
        temp_audio.write(audio_data)
        temp_path = temp_audio.name

    try:
        stt_start = time.perf_counter()
        user_text = stt_service.speech_to_text(temp_path)
        stt_ms = (time.perf_counter() - stt_start) * 1000
        logging.info("Extracted STT user text (chars=%d): %s", len(user_text), user_text)
        logging.info("Chat STT completed in %.2f ms", stt_ms)

        llm_start = time.perf_counter()
        try:
            role_text = load_role_text()
        except FileNotFoundError as exc:
            logging.exception("Role markdown file is missing.")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        prompts = [
            {
                "role": "system",
                "content": [{"type": "text", "text": role_text}],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": user_text}],
            },
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
    raise SystemExit(main(stt_model_dir="./models/Qwen3-ASR-1.7B-FineTune-2026-04-10"))
