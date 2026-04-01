# fast-front

STT(음성인식)와 TTS(음성합성) 기능을 제공하는 웹 프론트엔드입니다.

## 주요 기능

- **STT**: 마이크 녹음 또는 오디오 파일 업로드 → 텍스트 변환
- **TTS**: 텍스트 입력 → 음성 생성 및 재생

## 필수 요구사항

- 웹 브라우저 (Chrome, Firefox, Safari 등)
- Python 3.x (정적 서버 실행용)

## 설치 및 실행

### 1. 백엔드 실행

먼저 `fast-test` 디렉토리에서 FastAPI 서버를 실행하세요.

```bash
cd ../fast-test
python main.py
```

기본 API 주소: `http://localhost:8000`

### 2. 프론트엔드 실행

정적 웹 서버를 실행합니다.

```bash
cd ../fast-front
python -m http.server 5500
```

웹 브라우저에서 `http://localhost:5500` 을 열어주세요.

### 3. Node.js/npm 사용 시 (선택사항)

```bash
# http-server 설치
npm install -g http-server

# 서버 실행
http-server -p 5500
```

## 사용 방법

1. **API 설정**
   - 화면 상단에서 백엔드 API 주소 입력 (기본값: `http://localhost:8000`)
   - "저장" 버튼 클릭 (localStorage에 저장됨)

2. **STT (음성인식)**
   - 녹음: 마이크 버튼으로 녹음 시작/중지 → 전송 버튼
   - 파일 업로드: 오디오 파일 선택 → 전송 버튼
   - 인식된 텍스트가 화면에 표시됩니다

3. **TTS (음성합성)**
   - 텍스트 입력칸에 텍스트 입력
   - "음성 생성" 버튼 클릭
   - 생성된 음성이 자동으로 재생됩니다

## 파일 구조

```
fast-front/
├── index.html    # HTML 구조
├── app.js        # JavaScript 로직
├── styles.css    # 스타일시트
└── README.md     # 이 파일
```

## 기술 스택

- **Frontend**: HTML5, CSS3, JavaScript ES6+
- **API 통신**: Fetch API
- **음성**: Web Audio API, MediaRecorder API
- **저장소**: localStorage

## 주요 기능 세부사항

- **음성 녹음**: WebRTC MediaRecorder 사용
- **오디오 형식**: WebM (opus 코덱), MP4, OGG 자동 선택
- **CORS**: 크로스 오리진 요청 지원
- **타임아웃**: fetch 요청 제한 없음 (느린 환경에서도 완료까지 대기)

## 주의사항

- HTTPS가 아닌 HTTP에서는 음성 녹음 기능이 제한될 수 있습니다 (localhost 제외)
- 백엔드 서버가 기본 포트(8000)에 실행되지 않을 경우 API 주소 변경 필요
- 브라우저의 마이크 사용 권한 허용이 필요합니다

## 트러블슈팅

**마이크 연결 안 됨**
- 브라우저 권한 설정 확인
- localhost가 아닌 경우 HTTPS 필수

**API 연결 실패**
- 백엔드 서버가 실행 중인지 확인
- 입력한 API 주소가 정확한지 확인
- 브라우저 개발자 도구 콘솔에서 에러 메시지 확인

## 라이선스

MIT
