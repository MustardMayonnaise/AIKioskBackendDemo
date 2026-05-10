# AI Kiosk Demo Backend

FastAPI 기반 데모 백엔드입니다. 프론트와 AI 모델 사이의 브릿지 역할을 하며, 음성 입력을 받아 주문 상태 JSON과 TTS 음성을 함께 반환합니다.

## 실행 환경

- Conda 환경: `ai-kiosk`
- Python: 3.10 계열
- 권장: CUDA GPU 환경
- 서버 기본 포트: `35660`
- 기본 URL: `http://localhost:35660`

## 설치

`ai-kiosk` 환경을 활성화한 뒤 필요한 패키지를 설치합니다.

```powershell
conda activate ai-kiosk
pip install -r requirements.txt
```

현재 구현에서 사용하는 주요 패키지는 `fastapi`, `uvicorn`, `python-multipart`, `torch`, `transformers`, `bitsandbytes`, `faster-whisper`, `openai`, `python-dotenv`, `sentence-transformers`, `faiss-cpu`, `pandas`, `omnivoice`입니다.

## 환경 파일

AI 팀원이 준 `set\.env`를 데모 백엔드 루트의 `.env`로 복사해서 사용합니다.

```text
fast-test/.env
```

`.env`에는 API 키가 들어갈 수 있으므로 외부 공유나 커밋에 주의해야 합니다.

임베딩 모델 `google/embeddinggemma-300m`은 Hugging Face gated model입니다. 서버 시작 중 403 Forbidden 또는 gated repo 오류가 나오면 Hugging Face에서 해당 모델 접근 권한을 받은 뒤 아래 중 하나를 `fast-test/.env`에 추가해야 합니다.

```dotenv
HF_TOKEN=...
# 또는
HUGGINGFACE_HUB_TOKEN=...
```

## 모델 파일

`set` 폴더의 구현체는 내부 내용을 수정하지 않고 `models` 폴더로 복사해 보존했습니다.

```text
fast-test/models/main3_use_gemma.py
fast-test/models/vectorstore/SUBWAY_MENU.csv
fast-test/models/vectorstore/SUBWAY_MENU.index
```

원본 구현체는 CLI 흐름과 상대경로 의존성이 있어서 직접 수정하지 않습니다. `/chat`에서는 `services/rag_chat_service.py`가 원본 `RAGChatbot`을 adapter로 감싸서 호출하고, `services/chat_cart_service.py`가 주문 JSON 형태로 변환합니다.

서버 시작 시 `RagChatService.preload()`가 `RAGChatbot`의 Gemma 모델, 임베딩 모델, FAISS 인덱스를 한 번 로드합니다. 이후 `/chat` 요청은 이미 로드된 RAG 인스턴스를 재사용합니다. 초기화 순서는 무거운 RAG 모델을 먼저 올린 뒤 STT/TTS 서비스를 준비하도록 맞췄습니다.

Gemma 양자화 모델이 GPU VRAM에 전부 올라가지 않는 환경을 고려해 adapter에서 `llm_int8_enable_fp32_cpu_offload=True`를 로드 직전에 주입합니다. 원본 `models/main3_use_gemma.py`는 수정하지 않고, `services/rag_chat_service.py`에서 실행 환경에 필요한 로드 옵션만 보강합니다. 기존 `/llm` 테스트 엔드포인트의 `LLMService`는 중복 로드를 피하기 위해 `/llm`이 실제 호출될 때만 lazy-load됩니다.

사이즈와 토스팅 여부는 adapter에서 기존 RAG 프롬프트에 최소 문구만 추가해 처리합니다. 주문 순서는 `메인 메뉴 선택 완료 -> 사이즈 선택 완료 -> 빵 선택 완료 -> 치즈 선택 완료 -> 토스팅 여부 선택 완료 -> 야채 선택 완료`이며, 주문 상태에는 `사이즈: 15cm` 또는 `사이즈: 30cm`, `토스팅: 미정`, `토스팅: 함`, `토스팅: 안 함` 형태로 기록되도록 유도합니다.

## 실행

```powershell
conda activate ai-kiosk
python main.py
```

개발 모드로 실행하려면:

```powershell
conda activate ai-kiosk
uvicorn main:app --reload --host 0.0.0.0 --port 35660
```

## API

### GET /

헬스 체크용 엔드포인트입니다.

```bash
curl http://localhost:35660/
```

응답:

```json
{"Hello": "World"}
```

### GET /tts

텍스트를 받아 `audio/wav`로 반환합니다.

```bash
curl "http://localhost:35660/tts?text=안녕하세요" --output output.wav
```

### POST /stt

음성 파일을 받아 텍스트로 변환합니다.

```bash
curl -X POST -F "audio_file=@audio.webm" http://localhost:35660/stt
```

응답:

```json
{
  "text": "터키 샌드위치 하나 주세요"
}
```

### POST /llm

텍스트 질문을 받아 LLM 응답을 반환합니다.

```bash
curl -X POST "http://localhost:35660/llm?text=메뉴 추천해줘"
```

응답:

```json
{
  "message": "추천 응답"
}
```

### POST /chat

데모 핵심 API입니다. 음성 파일을 받아 다음 흐름을 수행합니다.

```text
STT -> RAGChatbot adapter -> 주문 상태 파싱 -> TTS -> JSON 반환
```

주문 JSON은 사용자에게 읽어줄 `answer` 문장을 기준으로 채우지 않습니다. RAG가 분리해서 반환한 `order_info`의 현재 주문 상태를 우선 파싱하고, 필요한 경우 이전 JSON `current_step`에 해당하는 사용자 발화만 보조로 반영합니다. 예를 들어 답변에 "화이트 빵으로 할까요?"가 포함되어도 현재 단계가 빵 선택 전이고 고객이 빵을 확정하지 않았다면 `bread`는 채우지 않습니다.

요청:

```bash
curl -X POST ^
  -F "audio_file=@audio.webm" ^
  -F "session_id=s_abc123" ^
  http://localhost:35660/chat
```

`session_id`는 선택값입니다. 없으면 서버가 새 세션 ID를 생성합니다. 프론트는 응답의 `session_id`를 저장했다가 다음 요청에 다시 보내 주문 상태를 이어갑니다.

응답 예시:

```json
{
  "session_id": "s_abc123",
  "answer": "알겠습니다. 빵은 어떤 걸로 하시겠어요?",
  "current_step": "BREAD_SELECT",
  "active_order": {
    "order_item_id": "oi_123456789abc",
    "menu": {
      "id": "터키",
      "name": "터키"
    },
    "size": "15cm",
    "bread": {
      "id": null,
      "name": null
    },
    "cheese": {
      "id": null,
      "name": null
    },
    "is_toasted": null,
    "vegetables": [
      {
        "id": null,
        "name": null
      }
    ],
    "sauces": [
      {
        "id": null,
        "name": null
      }
    ],
    "side_menu_items": [
      {
        "id": null,
        "name": null
      }
    ],
    "extras": [
      {
        "id": null,
        "name": null
      }
    ],
    "quantity": 1,
    "price": {
      "base_price": null,
      "extra_price": null,
      "total_price": null
    }
  },
  "audio": {
    "mime_type": "audio/wav",
    "base64": "..."
  }
}
```

`audio.base64`는 TTS 결과 wav 파일입니다. 프론트는 이 값을 Blob으로 복원해 재생합니다. 화면 표시용 JSON에서는 base64 전체 대신 길이만 보여주는 것이 좋습니다.

CPU 환경 또는 외부 프록시 환경에서 `/chat` 요청이 오래 걸리면 브라우저가 `NetworkError when attempting to fetch resource`를 낼 수 있습니다. 이를 복구하기 위해 프론트는 요청 전에 `session_id`를 먼저 생성해 전송하고, 네트워크 오류가 발생하면 `GET /chat/session/{session_id}`를 polling해 마지막 주문 JSON을 다시 가져옵니다. 이 복구 응답은 JSON 상태 갱신용이며, 끊긴 요청의 TTS audio base64는 재전송하지 않습니다.

### GET /chat/session/{session_id}

세션의 마지막 주문 JSON을 조회합니다. `/chat` 연결이 중간에 끊겼을 때 프론트 복구용으로 사용합니다.

```bash
curl http://localhost:35660/chat/session/s_abc123
```

## 주문 상태 단계

`current_step`은 아래 순서로 진행됩니다.

```text
MENU_SELECT
SIZE_SELECT
BREAD_SELECT
CHEESE_SELECT
TOAST_SELECT
VEGETABLE_SELECT
SAUCE_SELECT
SIDE_SELECT
ORDER_CONFIRM
```

음료는 `side_menu_items`에 들어갑니다. 현재 CSV 기준으로 `탄산음료`, `커피`를 지원하며, `콜라`, `사이다`, `스프라이트` 발화는 `탄산음료`로 파싱합니다.

야채, 소스, 사이드, 추가 재료처럼 여러 개를 고를 수 있는 항목은 제외 표현을 별도로 처리합니다. 예를 들어 `토마토 빼고 다 넣어` 또는 RAG 주문 상태의 `야채: 토마토 제외, 나머지 모두`는 토마토를 제외한 나머지 야채 전체로 확장합니다. 반대로 `토마토는 빼주세요`처럼 특정 항목 제외만 말하고 대체 선택이 없으면 해당 단계에 머무릅니다.

## 로깅

로그는 콘솔과 `fast-test/logs/app.log`에 함께 기록됩니다. 파일 로그는 5MB 단위로 rotate되며 최대 3개까지 보관합니다.

`/chat` 요청은 전체 파이프라인 시간과 함께 STT, RAG, 주문 JSON 파싱, TTS 시간을 남깁니다. RAG adapter는 원본 `RAGChatbot`을 수정하지 않고 아래 단계별 입력, 출력, 소요시간을 기록합니다.

- `RAG input received`: 사용자 입력, 세션 상태, 기존 주문 정보
- `RAG search decision completed`: 검색 필요 여부 판단 프롬프트와 모델 출력
- `RAG retrieval completed` 또는 `RAG retrieval skipped`: 검색용 질의, FAISS 검색 결과 문서, 검색 단계 시간
- `RAG answer generated`: 최종 답변 생성 프롬프트, 원본 combined answer, 파싱된 주문 정보와 답변
- `RAG order process updated`: 주문 단계 변경 전후 값
- `RAG answer condensed`: 사용자에게 읽어줄 최종 축약 답변
- `RAG output ready`: RAG 전체 결과, 단계별 `timings`

프롬프트, 검색 문서, 답변처럼 길어질 수 있는 텍스트는 기본 1200자까지만 로그에 남기고, 실제 길이는 `*_chars` 필드로 함께 기록합니다.

## 에러 응답

서비스 에러는 FastAPI `HTTPException` 형태로 반환됩니다.

```json
{
  "detail": {
    "error": {
      "code": "TTS-4221",
      "message": "TTS waveform is empty"
    }
  }
}
```

주요 에러 코드:

- `STT-4001`, `STT-4221`, `STT-5001`, `STT-5002`
- `TTS-4001`, `TTS-4221`, `TTS-5001`, `TTS-5002`, `TTS-5021`, `TTS-5022`
- `COMMON-5000`

## 파일 구조

```text
fast-test/
  main.py
  requirements.txt
  .env
  core/
    errors.py
    log_config.py
  data/
    cart.json
  markdown/
    agent.md
    menu.md
  models/
    main3_use_gemma.py
    vectorstore/
      SUBWAY_MENU.csv
      SUBWAY_MENU.index
  services/
    chat_cart_service.py
    rag_chat_service.py
    llm_service.py
    stt_service.py
    tts_service.py
```

## 검증 명령

```powershell
conda activate ai-kiosk
python -m py_compile main.py services\rag_chat_service.py services\chat_cart_service.py models\main3_use_gemma.py
```

프론트 문법 검사는 프로젝트 루트에서 실행합니다.

```powershell
node --check fast-front\app.js
```
