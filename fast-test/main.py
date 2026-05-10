from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from services.tts_service import TTSService
from services.stt_service import STTService
from services.chat_cart_service import ChatCartService
from services.rag_chat_service import RagChatService
from core.log_config import setup_logging
from pathlib import Path
import base64
import os
import tempfile
import uvicorn
import logging
import time
from uuid import uuid4
from core.errors import STTServiceError, TTSServiceError

BASE_DIR = Path(__file__).resolve().parent
setup_logging()
services_initialized = False


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    )

def initialize_services() -> None:
    global stt_service, tts_service, chatbot, chat_cart_service, rag_chat_service, app
    global services_initialized
    if services_initialized:
        return
    logging.info("Initializing services...")
    chatbot = None
    chat_cart_service = ChatCartService(BASE_DIR)
    rag_chat_service = RagChatService(BASE_DIR)
    logging.info("Preloading RAG model before STT/TTS services.")
    rag_chat_service.preload("startup")
    stt_service = STTService()
    tts_service = TTSService()
    services_initialized = True
    logging.info("Services initialized successfully.")

def main(host: str = "0.0.0.0", port: int = 35660):
    setup_logging()
    initialize_services()
    uvicorn.run(app, host=host, port=port)

@app.get("/")
def read_root():
    logging.info("Received request at root endpoint.")
    return {"Hello": "World"}

@app.post("/chat")
def post_chat(audio_file: UploadFile, session_id: str | None = Form(default=None)):
    request_start = time.perf_counter()
    request_id = f"req_{uuid4().hex[:12]}"
    logging.info(
        "Received Chat request (request_id=%s, session_id=%s, filename=%s)",
        request_id,
        session_id,
        audio_file.filename,
    )

    audio_data = audio_file.file.read()
    suffix = os.path.splitext(audio_file.filename or "recording.webm")[1] or ".webm"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_audio:
        temp_audio.write(audio_data)
        temp_path = temp_audio.name

    try:
        stt_start = time.perf_counter()
        user_text = stt_service.speech_to_text(temp_path)
        stt_ms = (time.perf_counter() - stt_start) * 1000
        logging.info(
            "Chat STT completed (request_id=%s, stt_ms=%.2f, user_chars=%d, user_text=%s)",
            request_id,
            stt_ms,
            len(user_text),
            user_text,
        )

        response_session_id = session_id or f"s_{uuid4().hex[:12]}"

        rag_start = time.perf_counter()
        rag_result = rag_chat_service.ask(response_session_id, user_text, request_id)
        rag_ms = (time.perf_counter() - rag_start) * 1000
        answer = rag_result["answer"]
        logging.info(
            "Chat RAG completed (request_id=%s, session_id=%s, rag_ms=%.2f, answer_chars=%d, search_required=%s, docs_chars=%d)",
            request_id,
            response_session_id,
            rag_ms,
            len(answer),
            rag_result["search_required"],
            len(rag_result["retrieved_docs_text"]),
        )

        cart_start = time.perf_counter()
        response = chat_cart_service.build_response(
            session_id=response_session_id,
            user_text=user_text,
            answer=answer,
            order_info=rag_result["order_info"],
            order_process=rag_result["order_process"],
        )
        cart_ms = (time.perf_counter() - cart_start) * 1000

        tts_start = time.perf_counter()
        wav_bytes = tts_service.text_to_speech(answer)
        tts_ms = (time.perf_counter() - tts_start) * 1000
        response["audio"] = {
            "mime_type": "audio/wav",
            "base64": base64.b64encode(wav_bytes).decode("ascii"),
        }
        total_ms = (time.perf_counter() - request_start) * 1000

        logging.info(
            "Chat pipeline completed (request_id=%s, session_id=%s, total_ms=%.2f, stt_ms=%.2f, rag_ms=%.2f, cart_ms=%.2f, tts_ms=%.2f, user_chars=%d, answer_chars=%d, step=%s, menu_id=%s, total_price=%s)",
            request_id,
            response["session_id"],
            total_ms,
            stt_ms,
            rag_ms,
            cart_ms,
            tts_ms,
            len(user_text),
            len(answer),
            response["current_step"],
            response["active_order"]["menu"]["id"],
            response["active_order"]["price"]["total_price"],
        )
        return response
    except STTServiceError as exc:
        logging.exception("Chat STT domain error.")
        raise HTTPException(status_code=exc.http_status, detail={"error": exc.to_dict()})  
    except TTSServiceError as exc:
        logging.exception("Chat TTS domain error.")
        raise HTTPException(status_code=exc.http_status, detail={"error": exc.to_dict()}) from exc
    except Exception as exc:
        logging.exception("Chat pipeline failed.")
        raise HTTPException(status_code=500, detail={"error": {"code": "COMMON_INTERNAL_ERROR", "message": str(exc)}}) from exc
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass

@app.get("/chat/session/{session_id}")
def get_chat_session(session_id: str):
    response = chat_cart_service.get_response(session_id)
    if response is None:
        raise HTTPException(status_code=404, detail={"error": {"code": "CHAT-4040", "message": "Chat session not found"}})
    return response


if __name__ == "__main__":
    raise SystemExit(main())#stt_model_dir="./models/Qwen3-ASR-1.7B-FineTune-2026-04-10"))
