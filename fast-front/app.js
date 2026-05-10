const API_BASE = "http://localhost:35660";

const state = {
  chatObjectUrl: null,
  chatUploadedFile: null,
  chatMediaRecorder: null,
  chatStream: null,
  chatRecordedBlob: null,
  chatRecordingMimeType: "audio/webm",
  chatSessionId: localStorage.getItem("fastSpeechChatSessionId") || "",
};

const els = {
  statusText: document.getElementById("statusText"),
  chatAudioFileInput: document.getElementById("chatAudioFileInput"),
  sendChatFileButton: document.getElementById("sendChatFileButton"),
  chatAudioPreview: document.getElementById("chatAudioPreview"),
  startChatRecordButton: document.getElementById("startChatRecordButton"),
  stopChatRecordButton: document.getElementById("stopChatRecordButton"),
  sendChatRecordButton: document.getElementById("sendChatRecordButton"),
  chatJsonResult: document.getElementById("chatJsonResult"),
};

function setStatus(message) {
  els.statusText.textContent = message;
}

function cleanObjectUrl() {
  if (state.chatObjectUrl) {
    URL.revokeObjectURL(state.chatObjectUrl);
    state.chatObjectUrl = null;
  }
}

function base64ToBlob(base64, mimeType) {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) {
    bytes[i] = binary.charCodeAt(i);
  }
  return new Blob([bytes], { type: mimeType });
}

function toDisplayChatJson(data) {
  if (!data.audio || typeof data.audio.base64 !== "string") {
    return data;
  }
  return {
    ...data,
    audio: {
      mime_type: data.audio.mime_type,
      base64_length: data.audio.base64.length,
    },
  };
}

function createSessionId() {
  if (window.crypto && typeof window.crypto.randomUUID === "function") {
    return `s_${window.crypto.randomUUID().replace(/-/g, "").slice(0, 12)}`;
  }
  return `s_${Date.now().toString(36)}${Math.random().toString(36).slice(2, 8)}`;
}

function ensureChatSessionId() {
  if (!state.chatSessionId) {
    state.chatSessionId = createSessionId();
    localStorage.setItem("fastSpeechChatSessionId", state.chatSessionId);
  }
  return state.chatSessionId;
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
    cleanObjectUrl();
    state.chatObjectUrl = URL.createObjectURL(state.chatRecordedBlob);
    els.chatAudioPreview.src = state.chatObjectUrl;
    els.sendChatRecordButton.disabled = false;

    if (state.chatStream) {
      state.chatStream.getTracks().forEach((track) => track.stop());
      state.chatStream = null;
    }

    setStatus("녹음 완료. 녹음 전송 버튼을 눌러 Chat을 실행하세요.");
  };

  state.chatMediaRecorder.start();
}

function stopChatRecording() {
  if (state.chatMediaRecorder && state.chatMediaRecorder.state !== "inactive") {
    state.chatMediaRecorder.stop();
  }
}

async function sendAudioToChat(fileOrBlob, filename) {
  const formData = new FormData();
  formData.append("audio_file", fileOrBlob, filename);
  formData.append("session_id", ensureChatSessionId());

  setStatus("Chat 처리 중입니다. 완료까지 기다려 주세요.");
  const response = await fetch(`${API_BASE}/chat`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const body = await response.text();
    throw new Error(`Chat 요청 실패 (${response.status}): ${body}`);
  }

  const data = await response.json();
  if (typeof data.session_id === "string" && data.session_id) {
    state.chatSessionId = data.session_id;
    localStorage.setItem("fastSpeechChatSessionId", data.session_id);
  }

  if (data.audio && typeof data.audio.base64 === "string") {
    const audioBlob = base64ToBlob(data.audio.base64, data.audio.mime_type || "audio/wav");
    cleanObjectUrl();
    state.chatObjectUrl = URL.createObjectURL(audioBlob);
    els.chatAudioPreview.src = state.chatObjectUrl;
    await els.chatAudioPreview.play().catch(() => undefined);
  }

  els.chatJsonResult.value = JSON.stringify(toDisplayChatJson(data), null, 2);
  setStatus("Chat 완료");
}

els.chatAudioFileInput.addEventListener("change", (event) => {
  const files = event.target.files;
  state.chatUploadedFile = files && files[0] ? files[0] : null;
  els.sendChatFileButton.disabled = !state.chatUploadedFile;
  state.chatRecordedBlob = null;
  els.sendChatRecordButton.disabled = true;

  if (state.chatUploadedFile) {
    setStatus(`파일 선택 완료: ${state.chatUploadedFile.name}`);
  }
});

els.startChatRecordButton.addEventListener("click", async () => {
  try {
    els.startChatRecordButton.disabled = true;
    els.stopChatRecordButton.disabled = false;
    els.sendChatRecordButton.disabled = true;
    state.chatUploadedFile = null;
    els.sendChatFileButton.disabled = true;
    els.chatAudioFileInput.value = "";
    await startChatRecording();
    setStatus("녹음 중입니다.");
  } catch (error) {
    els.startChatRecordButton.disabled = false;
    els.stopChatRecordButton.disabled = true;
    setStatus(error.message || "녹음을 시작할 수 없습니다.");
  }
});

els.stopChatRecordButton.addEventListener("click", () => {
  els.startChatRecordButton.disabled = false;
  els.stopChatRecordButton.disabled = true;
  stopChatRecording();
});

els.sendChatRecordButton.addEventListener("click", async () => {
  if (!state.chatRecordedBlob) {
    setStatus("먼저 녹음을 완료하세요.");
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
    setStatus("먼저 오디오 파일을 선택하세요.");
    return;
  }

  try {
    await sendAudioToChat(state.chatUploadedFile, state.chatUploadedFile.name);
  } catch (error) {
    setStatus(error.message || "Chat 처리 중 오류가 발생했습니다.");
  }
});
