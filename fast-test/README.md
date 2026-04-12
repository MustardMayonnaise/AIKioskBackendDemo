# fast-test

FastAPI 기반 STT/LLM/TTS 백엔드 서버입니다.

## 주요 기능

- STT (Speech-to-Text): `faster-whisper` 기반 한국어 음성 인식
- LLM: Gemma 3 양자화 모델 기반 답변 생성
- TTS (Text-to-Speech): `OmniVoice` 기반 한국어 음성 합성
- Chat 파이프라인: `STT -> LLM -> TTS`를 하나의 API로 제공
- 구조화된 에러 응답: HTTP 상태코드 + 도메인 에러코드 반환

## 필수 요구사항

- Python 3.10 이상
- pip 또는 conda
- CUDA GPU 권장 (CPU 동작 가능하나 매우 느릴 수 있음)
- 충분한 디스크/메모리 (모델 다운로드 및 로딩)

## 설치

### 1. 가상 환경 생성

```bash
conda create -n fast-test python=3.10 -y
conda activate fast-test
```

### 2. PyTorch 설치

CUDA 환경 예시:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
```

CPU 환경 예시:

```bash
pip install torch torchvision torchaudio
```

### 3. 백엔드 의존성 설치

아래 패키지는 현재 코드 import 기준으로 확인된 설치 대상입니다.

```bash
pip install fastapi uvicorn python-multipart
pip install faster-whisper transformers accelerate bitsandbytes
pip install omnivoice numpy soundfile
```

옵션(환경에 따라 필요):

```bash
# 음성 디코딩 이슈가 있으면 ffmpeg 설치
# Windows: winget install Gyan.FFmpeg
# macOS: brew install ffmpeg
```

## 실행

### 기본 실행

```bash
python main.py
```

기본 포트: `http://localhost:8000`

### 개발 모드 실행 (Hot-reload)

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## API 엔드포인트

### GET /

헬스 체크 엔드포인트

```bash
curl http://localhost:8000/
```

예시 응답:

```json
{"Hello": "World"}
```

### GET /tts

텍스트를 음성으로 변환해 `audio/wav`로 반환합니다.

```bash
curl "http://localhost:8000/tts?text=안녕하세요" --output output.wav
```

파라미터:

- `text` (string): 음성으로 변환할 텍스트

### POST /stt

업로드한 음성 파일을 텍스트로 변환합니다.

```bash
curl -X POST -F "audio_file=@audio.webm" http://localhost:8000/stt
```

예시 응답:

```json
{"text": "인식된 텍스트"}
```

### POST /llm

텍스트 프롬프트를 받아 답변을 생성합니다.

```bash
curl -X POST "http://localhost:8000/llm?text=메뉴 추천해줘"
```

예시 응답:

```json
{"message": "추천 답변"}
```

### POST /chat

음성 파일을 받아 STT -> LLM -> TTS를 수행하고, 응답 음성을 `audio/wav` 스트림으로 반환합니다.

```bash
curl -X POST -F "audio_file=@audio.webm" http://localhost:8000/chat --output chat_reply.wav
```

## 에러 응답 규격

서비스 오류는 아래 형태로 반환됩니다.

```json
{
	"detail": {
		"error": {
			"code": "TTS-4221",
			"message": "TTS waveform is empty",
			"details": {
				"reason": "..."
			}
		}
	}
}
```

대표 에러 코드:

- STT: `STT-4001`, `STT-4221`, `STT-5001`, `STT-5002`
- TTS: `TTS-4001`, `TTS-4221`, `TTS-5001`, `TTS-5002`, `TTS-5021`, `TTS-5022`
- 공통: `COMMON-5000`

## 파일 구조

```text
fast-test/
├── main.py
├── core/
│   ├── errors.py
│   └── log_config.py
├── services/
│   ├── stt_service.py
│   ├── tts_service.py
│   └── llm_service.py
├── markdown/
│   └── agent.md
└── README.md
```
