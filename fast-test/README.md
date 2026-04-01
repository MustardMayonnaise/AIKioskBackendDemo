# fast-test

FastAPI 기반의 STT(음성인식)와 TTS(음성합성) 백엔드 서버입니다.

## 주요 기능

- **STT (Speech-to-Text)**: Faster Whisper를 사용한 고속 음성인식
- **TTS (Text-to-Speech)**: MELO를 사용한 한국어 음성합성
- **CORS 지원**: 웹 프론트엔드와의 크로스 오리진 통신 가능
- **자동 디바이스 선택**: CUDA → MPS → CPU 자동 선택

## 필수 요구사항

- Python 3.8 이상
- pip 또는 conda
- 4GB 이상의 메모리 (모델 로딩)
- (선택) CUDA 지원 GPU (가속)

## 설치

### 1. 가상 환경 생성

```bash
# venv 사용
python -m venv venv
source venv/bin/activate  # macOS/Linux
# 또는
venv\Scripts\activate  # Windows

# conda 사용 (권장)
conda create -n fast-test python=3.10
conda activate fast-test
```

### 2. 의존성 설치

```bash
pip install -r requirements.txt
```

또는 개별 설치:

```bash
pip install fastapi uvicorn faster-whisper melo torch torchaudio soundfile python-multipart
```

### 3. 한국어 모델 다운로드 (처음 실행 시에만)

```bash
# MELO 모델 다운로드 (선택사항 - 첫 실행 시 자동)
python -c "from melo.api import TTS; TTS(language='KR')"

# Whisper 모델은 처음 실행 시 자동으로 다운로드됩니다
```

## 실행

### 기본 실행

```bash
python main.py
```

기본 포트: `http://localhost:8000`

### 커스텀 포트 지정

```bash
# main.py 마지막 줄 수정 후 실행
# uvicorn.run(app, host="0.0.0.0", port=8000)
# 위 줄의 port를 원하는 포트로 변경
```

### 개발 모드 실행 (Hot-reload)

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## API 엔드포인트

### GET /

헬스 체크 엔드포인트

```bash
curl http://localhost:8000/
# 응답: {"Hello": "World"}
```

### GET /tts

텍스트를 음성으로 변환합니다.

```bash
curl "http://localhost:8000/tts?text=안녕하세요" --output output.wav
```

**파라미터:**
- `text` (string): 음성으로 변환할 텍스트 (URL 인코딩 필수)

**응답:**
- Content-Type: `audio/wav`
- 바이너리 오디오 데이터

### POST /stt

음성 파일을 텍스트로 변환합니다.

```bash
curl -X POST -F "audio_file=@audio.webm" http://localhost:8000/stt
# 응답: {"text": "인식된 텍스트"}
```

**파라미터:**
- `audio_file` (file): 음성 파일 (multipart/form-data)

**지원하는 형식:**
- WebM, MP3, WAV, OGG, FLAC 등

**응답:**
- `{"text": "인식된 텍스트"}`

## 파일 구조

```
fast-test/
├── main.py              # FastAPI 메인 서버 파일
├── services/            # 서비스 구현
│   ├── stt_service.py   # Faster Whisper STT 서비스
│   └── tts_service.py   # MELO TTS 서비스
├── models/              # (예약됨) 커스텀 모델
├── core/                # (예약됨) 공통 로직
└── README.md            # 이 파일
```

## 기술 스택

- **Framework**: FastAPI
- **Server**: Uvicorn
- **STT**: Faster Whisper (OpenAI Whisper 최적화 버전)
- **TTS**: MELO (한국어 지원)
- **Deep Learning**: PyTorch
- **Audio Processing**: SoundFile, torchaudio

## 주요 클래스

### STTService

```python
from services.stt_service import STTService

stt = STTService(model_size="medium", device="auto")
text = stt.speech_to_text("path/to/audio.webm")
```

**파라미터:**
- `model_size`: tiny, base, small, medium, large (크기와 정확도 트레이드오프)
- `device`: auto, cuda, mps, cpu

### TTSService

```python
from services.tts_service import TTSService

tts = TTSService()
audio_bytes = tts.text_to_speech("안녕하세요")
```

## 환경 변수 (선택사항)

```bash
# GPU 사용 제어
export CUDA_VISIBLE_DEVICES=0  # 특정 GPU만 사용

# 모델 캐시 경로 설정
export HF_HOME=/path/to/cache
```

## 성능 최적화

### 1. 모델 크기 선택

```python
# fast-test/main.py 수정
stt_service = STTService(model_size="small")  # tiny, small, medium, large
```

- `tiny`: 가장 빠름, 낮은 정확도
- `small`: 빠름, 중간 정확도
- `medium`: 중간 속도, 높은 정확도 (기본값)
- `large`: 가장 느림, 최고 정확도

### 2. GPU 사용

CUDA가 설치된 경우 자동으로 GPU 사용:

```bash
# GPU 확인
python -c "import torch; print(torch.cuda.is_available())"
```

### 3. 메모리 최적화

```bash
# 모델 오프로드 (필요시)
export DEVICE=cpu  # CPU로 강제 실행
```

## 트러블슈팅

### 포트 이미 사용 중

```bash
# 다른 포트 사용
uvicorn main:app --port 8001
```

### 메모리 부족 (OOM)

```python
# main.py에서 모델 크기 줄임
stt_service = STTService(model_size="tiny")
```

### CUDA/GPU 오류

```bash
# CPU로 강제 실행
export CUDA_VISIBLE_DEVICES=""
python main.py
```

### 모델 다운로드 오류

```bash
# 캐시 삭제 후 재시도
rm -rf ~/.cache/huggingface
python main.py
```

## 개발 가이드

### 새로운 엔드포인트 추가

`main.py`에 다음과 같이 추가:

```python
@app.get("/new-endpoint")
def new_endpoint(param: str):
    return {"result": param}
```

### 커스텀 서비스 생성

`services/` 폴더에 새 파일 생성:

```python
# services/custom_service.py
class CustomService:
    def __init__(self):
        pass
    
    def process(self, data):
        return data
```

## 라이선스

MIT

## 참고 링크

- [FastAPI 문서](https://fastapi.tiangolo.com/)
- [Faster Whisper](https://github.com/guillaumekln/faster-whisper)
- [MELO TTS](https://github.com/myshell-ai/MeloTTS)
- [PyTorch](https://pytorch.org/)
