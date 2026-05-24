// --- 變數定義 ---
let isCamOn = true;
let sendOnEnter = true; // 設定：是否按 Enter 發送
let localStream = null;

const selfVideo = document.getElementById('selfVideo');
const selfPlaceholder = document.getElementById('selfPlaceholder');
const camBtn = document.getElementById('camBtn');
const chatInput = document.getElementById('chatInput');
const chatArea = document.getElementById('chatArea');
const sendBtn = document.getElementById('sendBtn');
const enterSendConfig = document.getElementById('enterSendConfig');

// --- 1. 相機開關邏輯 ---
async function initCamera() {
    try {
        localStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
        selfVideo.srcObject = localStream;
    } catch (e) {
        console.error("相機權限錯誤");
    }
}

camBtn.onclick = () => {
    isCamOn = !isCamOn;
    
    if (localStream) {
        // 取得視訊軌道並切換啟用狀態
        const videoTrack = localStream.getVideoTracks()[0];
        videoTrack.enabled = isCamOn;
    }

    if (isCamOn) {
        selfVideo.classList.remove('hidden');
        selfPlaceholder.classList.add('hidden');
        camBtn.textContent = "📹";
    } else {
        selfVideo.classList.add('hidden');
        selfPlaceholder.classList.remove('hidden');
        camBtn.textContent = "🚫"; // 顯示關閉圖示
    }
};

// --- 2. 訊息發送邏輯 ---
function sendMessage() {
    const text = chatInput.value.trim();
    if (!text) return;

    const msgDiv = document.createElement('div');
    msgDiv.className = 'msg-me';
    msgDiv.innerHTML = `<p class="bubble">${text.replace(/\n/g, '<br>')}</p>`;
    chatArea.appendChild(msgDiv);
    
    chatInput.value = ""; // 清空
    chatInput.style.height = 'auto'; // 重置高度
    chatArea.scrollTop = chatArea.scrollHeight; // 捲動到底部
}

// 處理 Textarea 按鍵動作
chatInput.addEventListener('keydown', (e) => {
    if (sendOnEnter) {
        // 如果設定為 Enter 發送
        if (e.key === 'Enter' && !e.altKey) {
            e.preventDefault(); // 防止預設換行
            sendMessage();
        } else if (e.key === 'Enter' && e.altKey) {
            // Alt + Enter：手動加入換行符號
            const start = chatInput.selectionStart;
            const end = chatInput.selectionEnd;
            chatInput.value = chatInput.value.substring(0, start) + "\n" + chatInput.value.substring(end);
            chatInput.selectionStart = chatInput.selectionEnd = start + 1;
            autoResizeTextarea();
        }
    }
});

// 自動調整 Textarea 高度
function autoResizeTextarea() {
    chatInput.style.height = 'auto';
    chatInput.style.height = (chatInput.scrollHeight) + 'px';
}
chatInput.addEventListener('input', autoResizeTextarea);

sendBtn.onclick = sendMessage;

// --- 3. 設定彈窗邏輯 ---
const settingsModal = document.getElementById('settingsModal');
document.getElementById('openSettings').onclick = () => settingsModal.classList.add('active');
document.getElementById('closeSettings').onclick = () => {
    sendOnEnter = enterSendConfig.checked;
    settingsModal.classList.remove('active');
};

// --- 初始化 ---
window.onload = () => {
    initCamera();
    // 更新時間邏輯 (略...)
};