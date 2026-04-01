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

설명: fastapi를 활용한 stt/tts 구성 예제입니다. 코드가 아직 정상적으로 동작하지 않으며, ai kiosk 백엔드로 확장할 수 있습니다.

환경:
- python 3.11.20
- cuda 13.1

```

pip install fastapi uvicorn

윈도우, 쿠다 버전은 아래와 같이 설치

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

macos, cpu 버전은 아래와 같이 설치

pip install torch torchvision

pip install faster-whisper git+httpsL//github.com/myshell-ai/MeloTTS.git mecab-python3 unidic eunjeon

python -m unidic download

```
