# fast-front

간단한 웹 프론트입니다.

- STT: 녹음(마이크) 또는 오디오 파일 업로드 -> /stt 전송 -> 텍스트 출력
- TTS: 텍스트 입력 -> /tts 전송 -> 음성 재생

## 1) 백엔드 실행

fast-test에서 FastAPI를 실행하세요.

```bash
cd /Users/admin/data/fast-test
python main.py
```

기본 주소는 http://localhost:8000 입니다.

## 2) 프론트 실행

아주 단순하게 정적 서버로 실행할 수 있습니다.

```bash
cd /Users/admin/data/fast-front
python -m http.server 5500
```

브라우저에서 http://localhost:5500 을 열면 됩니다.

## 3) 사용 방법

1. 상단 API 주소를 확인하고 저장
2. STT:
   - 녹음 시작 -> 녹음 중지 -> 녹음 전송
   - 또는 파일 업로드 후 파일 전송
3. TTS:
   - 텍스트 입력 -> 음성 생성

fetch 기본 동작은 타임아웃 제한이 없으므로 CPU 환경에서 시간이 오래 걸려도 완료까지 대기합니다.
