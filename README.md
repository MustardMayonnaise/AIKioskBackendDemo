# fast-front

설명: AI를 이용해서 백엔드 테스트를 할 수 있도록 구성한 프로젝트입니다.
환경:
- node 버전: 25.8.2


```

macos 기준 

brew install node

npx serve -l 5500

```

# fast-test

설명: FastAPI 기반 STT/LLM/TTS 백엔드입니다.

환경:
- python 3.11.20
- cuda 13.1

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
pip install fastapi uvicorn python-multipart
pip install faster-whisper transformers accelerate bitsandbytes
pip install omnivoice numpy soundfile
```

상세 API/에러코드 문서는 [fast-test/README.md](fast-test/README.md) 참고.
