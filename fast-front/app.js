const state = {
  mediaRecorder: null,
  stream: null,
  recordedBlob: null,
  uploadedFile: null,
  recordingMimeType: "audio/webm",
  chatRecordingMimeType: "audio/webm",
  ttsObjectUrl: null,
  sttObjectUrl: null,
  chatObjectUrl: null,
  chatUploadedFile: null,
  chatMediaRecorder: null,
  chatStream: null,
  chatRecordedBlob: null,
};

const els = {
  apiBase: document.getElementById("apiBase"),
  saveApiButton: document.getElementById("saveApiButton"),
  statusText: document.getElementById("statusText"),

  startRecordButton: document.getElementById("startRecordButton"),
  stopRecordButton: document.getElementById("stopRecordButton"),
  sendRecordButton: document.getElementById("sendRecordButton"),
  audioFileInput: document.getElementById("audioFileInput"),
  sendFileButton: document.getElementById("sendFileButton"),
  sttAudioPreview: document.getElementById("sttAudioPreview"),
  sttResult: document.getElementById("sttResult"),

  ttsInput: document.getElementById("ttsInput"),
  synthesizeButton: document.getElementById("synthesizeButton"),
  ttsAudioPlayer: document.getElementById("ttsAudioPlayer"),

  llmInput: document.getElementById("llmInput"),
  askLlmButton: document.getElementById("askLlmButton"),
  llmResult: document.getElementById("llmResult"),

  chatAudioFileInput: document.getElementById("chatAudioFileInput"),
  sendChatFileButton: document.getElementById("sendChatFileButton"),
  chatAudioPreview: document.getElementById("chatAudioPreview"),
  startChatRecordButton: document.getElementById("startChatRecordButton"),
  stopChatRecordButton: document.getElementById("stopChatRecordButton"),
  sendChatRecordButton: document.getElementById("sendChatRecordButton"),
};

const storedApiBase = localStorage.getItem("fastSpeechApiBase");
if (storedApiBase) {
  els.apiBase.value = storedApiBase;
}

function setStatus(message) {
  els.statusText.textContent = message;
}

function getApiBase() {
  return els.apiBase.value.trim().replace(/\/$/, "");
}

function cleanObjectUrl(key) {
  if (state[key]) {
    URL.revokeObjectURL(state[key]);
    state[key] = null;
  }
}

function getSupportedMimeType() {
  const candidates = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4", "audio/ogg"];
  for (const type of candidates) {
    if (window.MediaRecorder && MediaRecorder.isTypeSupported(type)) {
      return type;
    }
  }
  return "";
}

async function startRecording() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new Error("이 브라우저는 마이크 녹음을 지원하지 않습니다.");
  }

  state.stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  const mimeType = getSupportedMimeType();
  state.recordingMimeType = mimeType || "audio/webm";

  const chunks = [];
  state.mediaRecorder = mimeType
    ? new MediaRecorder(state.stream, { mimeType })
    : new MediaRecorder(state.stream);

  state.mediaRecorder.ondataavailable = (event) => {
    if (event.data && event.data.size > 0) {
      chunks.push(event.data);
    }
  };

  state.mediaRecorder.onstop = () => {
    state.recordedBlob = new Blob(chunks, { type: state.recordingMimeType });
    cleanObjectUrl("sttObjectUrl");
    state.sttObjectUrl = URL.createObjectURL(state.recordedBlob);
    els.sttAudioPreview.src = state.sttObjectUrl;
    els.sendRecordButton.disabled = false;

    if (state.stream) {
      state.stream.getTracks().forEach((track) => track.stop());
      state.stream = null;
    }

    setStatus("녹음 완료. 전송 버튼을 눌러 STT를 실행하세요.");
  };

  state.mediaRecorder.start();
}

function stopRecording() {
  if (state.mediaRecorder && state.mediaRecorder.state !== "inactive") {
    state.mediaRecorder.stop();
  }
}

async function startChatRecording() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new Error("이 브라우저는 마이크 녹음을 지원하지 않습니다.");
  }

  state.chatStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  const mimeType = getSupportedMimeType();
  state.chatRecordingMimeType = mimeType || "audio/webm";

  const chunks = [];
  state.chatMediaRecorder = mimeType
    ? new MediaRecorder(state.chatStream, { mimeType })
    : new MediaRecorder(state.chatStream);

  state.chatMediaRecorder.ondataavailable = (event) => {
    if (event.data && event.data.size > 0) {
      chunks.push(event.data);
    }
  };

  state.chatMediaRecorder.onstop = () => {
    state.chatRecordedBlob = new Blob(chunks, { type: state.chatRecordingMimeType });
    cleanObjectUrl("chatObjectUrl");
    state.chatObjectUrl = URL.createObjectURL(state.chatRecordedBlob);
    els.chatAudioPreview.src = state.chatObjectUrl;
    els.sendChatRecordButton.disabled = false;

    if (state.chatStream) {
      state.chatStream.getTracks().forEach((track) => track.stop());
      state.chatStream = null;
    }

    setStatus("Chat 녹음 완료. 전송 버튼을 눌러 Chat을 실행하세요.");
  };

  state.chatMediaRecorder.start();
}

function stopChatRecording() {
  if (state.chatMediaRecorder && state.chatMediaRecorder.state !== "inactive") {
    state.chatMediaRecorder.stop();
  }
}

async function sendAudioToStt(fileOrBlob, filename) {
  const apiBase = getApiBase();
  if (!apiBase) {
    throw new Error("API 주소를 입력하세요.");
  }

  const formData = new FormData();
  formData.append("audio_file", fileOrBlob, filename);

  setStatus("STT 처리 중... 완료까지 기다려 주세요.");
  const response = await fetch(`${apiBase}/stt`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const body = await response.text();
    throw new Error(`STT 요청 실패 (${response.status}): ${body}`);
  }

  const data = await response.json();
  let text = "";

  if (typeof data.text === "string") {
    text = data.text;
  } else if (Array.isArray(data.text)) {
    text = data.text.join(" ");
  } else {
    text = JSON.stringify(data, null, 2);
  }

  els.sttResult.value = text || "(빈 결과)";
  setStatus("STT 완료");
}

async function requestTts() {
  const apiBase = getApiBase();
  const text = els.ttsInput.value.trim();

  if (!apiBase) {
    throw new Error("API 주소를 입력하세요.");
  }
  if (!text) {
    throw new Error("TTS 텍스트를 입력하세요.");
  }

  setStatus("TTS 생성 중... 완료까지 기다려 주세요.");
  const response = await fetch(`${apiBase}/tts?text=${encodeURIComponent(text)}`, {
    method: "GET",
  });

  if (!response.ok) {
    const body = await response.text();
    throw new Error(`TTS 요청 실패 (${response.status}): ${body}`);
  }

  const audioBlob = await response.blob();
  cleanObjectUrl("ttsObjectUrl");
  state.ttsObjectUrl = URL.createObjectURL(audioBlob);
  els.ttsAudioPlayer.src = state.ttsObjectUrl;
  await els.ttsAudioPlayer.play().catch(() => undefined);
  setStatus("TTS 완료");
}

async function requestLlm() {
  const apiBase = getApiBase();
  const text = els.llmInput.value.trim();

  if (!apiBase) {
    throw new Error("API 주소를 입력하세요.");
  }
  if (!text) {
    throw new Error("LLM 질의 텍스트를 입력하세요.");
  }

  setStatus("LLM 질의 중... 완료까지 기다려 주세요.");
  const response = await fetch(`${apiBase}/llm?text=${encodeURIComponent(text)}`, {
    method: "POST",
  });

  if (!response.ok) {
    const body = await response.text();
    throw new Error(`LLM 요청 실패 (${response.status}): ${body}`);
  }

  const data = await response.json();
  const message = typeof data.message === "string" ? data.message : JSON.stringify(data, null, 2);
  els.llmResult.value = message || "(빈 결과)";
  setStatus("LLM 완료");
}

async function sendAudioToChat(fileOrBlob, filename) {
  const apiBase = getApiBase();
  if (!apiBase) {
    throw new Error("API 주소를 입력하세요.");
  }

  const formData = new FormData();
  formData.append("audio_file", fileOrBlob, filename);

  setStatus("Chat 처리 중... 완료까지 기다려 주세요.");
  const response = await fetch(`${apiBase}/chat`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const body = await response.text();
    throw new Error(`Chat 요청 실패 (${response.status}): ${body}`);
  }

  const audioBlob = await response.blob();
  cleanObjectUrl("chatObjectUrl");
  state.chatObjectUrl = URL.createObjectURL(audioBlob);
  els.chatAudioPreview.src = state.chatObjectUrl;
  await els.chatAudioPreview.play().catch(() => undefined);
  setStatus("Chat 완료");
}

els.saveApiButton.addEventListener("click", () => {
  const apiBase = getApiBase();
  localStorage.setItem("fastSpeechApiBase", apiBase);
  setStatus(`API 주소 저장: ${apiBase}`);
});

els.startRecordButton.addEventListener("click", async () => {
  try {
    els.startRecordButton.disabled = true;
    els.stopRecordButton.disabled = false;
    els.sendRecordButton.disabled = true;
    await startRecording();
    setStatus("녹음 중...");
  } catch (error) {
    els.startRecordButton.disabled = false;
    els.stopRecordButton.disabled = true;
    setStatus(error.message || "녹음을 시작할 수 없습니다.");
  }
});

els.stopRecordButton.addEventListener("click", () => {
  els.startRecordButton.disabled = false;
  els.stopRecordButton.disabled = true;
  stopRecording();
});

els.sendRecordButton.addEventListener("click", async () => {
  if (!state.recordedBlob) {
    setStatus("먼저 녹음을 완료하세요.");
    return;
  }

  try {
    await sendAudioToStt(state.recordedBlob, "recording.webm");
  } catch (error) {
    setStatus(error.message || "STT 처리 중 오류가 발생했습니다.");
  }
});

els.audioFileInput.addEventListener("change", (event) => {
  const files = event.target.files;
  state.uploadedFile = files && files[0] ? files[0] : null;
  els.sendFileButton.disabled = !state.uploadedFile;

  if (state.uploadedFile) {
    cleanObjectUrl("sttObjectUrl");
    state.sttObjectUrl = URL.createObjectURL(state.uploadedFile);
    els.sttAudioPreview.src = state.sttObjectUrl;
    setStatus(`파일 선택 완료: ${state.uploadedFile.name}`);
  }
});

els.sendFileButton.addEventListener("click", async () => {
  if (!state.uploadedFile) {
    setStatus("먼저 오디오 파일을 선택하세요.");
    return;
  }

  try {
    await sendAudioToStt(state.uploadedFile, state.uploadedFile.name);
  } catch (error) {
    setStatus(error.message || "STT 처리 중 오류가 발생했습니다.");
  }
});

els.synthesizeButton.addEventListener("click", async () => {
  try {
    await requestTts();
  } catch (error) {
    setStatus(error.message || "TTS 처리 중 오류가 발생했습니다.");
  }
});

els.askLlmButton.addEventListener("click", async () => {
  try {
    await requestLlm();
  } catch (error) {
    setStatus(error.message || "LLM 처리 중 오류가 발생했습니다.");
  }
});

els.chatAudioFileInput.addEventListener("change", (event) => {
  const files = event.target.files;
  state.chatUploadedFile = files && files[0] ? files[0] : null;
  els.sendChatFileButton.disabled = !state.chatUploadedFile;
  state.chatRecordedBlob = null;
  els.sendChatRecordButton.disabled = true;

  if (state.chatUploadedFile) {
    setStatus(`Chat 파일 선택 완료: ${state.chatUploadedFile.name}`);
  }
});

els.startChatRecordButton.addEventListener("click", async () => {
  try {
    els.startChatRecordButton.disabled = true;
    els.stopChatRecordButton.disabled = false;
    els.sendChatRecordButton.disabled = true;
    state.chatUploadedFile = null;
    els.sendChatFileButton.disabled = true;
    if (els.chatAudioFileInput) {
      els.chatAudioFileInput.value = "";
    }
    await startChatRecording();
    setStatus("Chat 녹음 중...");
  } catch (error) {
    els.startChatRecordButton.disabled = false;
    els.stopChatRecordButton.disabled = true;
    setStatus(error.message || "Chat 녹음을 시작할 수 없습니다.");
  }
});

els.stopChatRecordButton.addEventListener("click", () => {
  els.startChatRecordButton.disabled = false;
  els.stopChatRecordButton.disabled = true;
  stopChatRecording();
});

els.sendChatRecordButton.addEventListener("click", async () => {
  if (!state.chatRecordedBlob) {
    setStatus("먼저 Chat 녹음을 완료하세요.");
    return;
  }

  try {
    await sendAudioToChat(state.chatRecordedBlob, "chat-recording.webm");
  } catch (error) {
    setStatus(error.message || "Chat 처리 중 오류가 발생했습니다.");
  }
});

els.sendChatFileButton.addEventListener("click", async () => {
  if (!state.chatUploadedFile) {
    setStatus("먼저 Chat 오디오 파일을 선택하세요.");
    return;
  }

  try {
    await sendAudioToChat(state.chatUploadedFile, state.chatUploadedFile.name);
  } catch (error) {
    setStatus(error.message || "Chat 처리 중 오류가 발생했습니다.");
  }
});
