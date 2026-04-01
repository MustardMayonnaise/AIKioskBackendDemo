# fast-test

FastAPI 기반의 STT(음성인식)와 TTS(음성합성) 백엔드 서버입니다.

## 주요 기능

- **STT (Speech-to-Text)**: Faster Whisper를 사용한 고속 음성인식
- **TTS (Text-to-Speech)**: MELO를 사용한 한국어 음성합성
- **CORS 지원**: 웹 프론트엔드와의 크로스 오리진 통신 가능
- **자동 디바이스 선택**: CUDA → MPS → CPU 자동 선택

## 필수 요구사항

- Python 3.10 이상
- pip 또는 conda
- 4GB 이상의 메모리 (모델 로딩)
- CUDA 지원 GPU (가속)
- 저장 공간 2GB 이상 필요 (모델 저장)

## 설치

### 1. 가상 환경 생성

```bash
conda create -n fast-test python=3.10
conda activate fast-test
```

### 2. 의존성 설치

- 메인 참조


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

# 현재 오류가 있습니다.

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
