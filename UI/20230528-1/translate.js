// --- DOM 選項 ---
const selfVideo = document.getElementById('selfVideo');
const selfAvatar = document.getElementById('selfAvatar');
const camBtn = document.getElementById('camBtn');
const chatInput = document.getElementById('chatInput');
const chatArea = document.getElementById('chatArea'); // 補上漏掉的選取
const sendBtn = document.getElementById('sendBtn');
const recordBtn = document.getElementById('recordBtn'); // 補上漏掉的選取
const exitBtn = document.getElementById('exitBtn'); // 補上漏掉的選取
const enterSetting = document.getElementById('enterToSendSetting');
const timeDisplay = document.getElementById('currentDateTime');

let camOn = true;
let sendOnEnter = true; // 定義 Enter 發送狀態變數
let localStream = null;

// --- 時間與日期功能 ---
function updateDateTime() {
    const now = new Date();
    const dateStr = now.toLocaleDateString('zh-TW', { year: 'numeric', month: '2-digit', day: '2-digit' });
    const timeStr = now.toLocaleTimeString('zh-TW', { hour12: false });
    timeDisplay.textContent = `${dateStr} ${timeStr}`;
}
setInterval(updateDateTime, 1000);
updateDateTime();

// --- 攝影機初始化功能 ---
async function initCamera() {
    try {
        localStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
        selfVideo.srcObject = localStream;
        selfAvatar.classList.add('hidden'); // 預設開啟時隱藏頭像
    } catch (err) {
        console.error("相機初始化失敗:", err);
    }
}

// 點擊鏡頭開關
camBtn.onclick = () => {
    if (!localStream) return;
    
    camOn = !camOn;
    const videoTrack = localStream.getVideoTracks()[0];
    
    if (videoTrack) {
        videoTrack.enabled = camOn; // 實質關閉/開啟鏡頭感應
        
        if (camOn) {
            selfVideo.classList.remove('off'); // 如果你CSS是用 off
            selfVideo.classList.remove('hidden'); // 安全起見同時相容兩種寫法
            selfAvatar.classList.add('hidden');
            camBtn.textContent = "📹";
        } else {
            selfVideo.classList.add('off');
            selfVideo.classList.add('hidden');
            selfAvatar.classList.remove('hidden'); // 顯示用戶頭像
            camBtn.textContent = "🚫";
        }
    }
};

// 點擊退出
if (exitBtn) {
    exitBtn.onclick = () => {
        if(confirm("確定要退出工作站嗎？")) {
            window.location.href = 'home.html';
        }
    };
}

// --- 訊息發送邏輯 (Enter/Alt+Enter) ---

let lastSender = null;

function sendMessage() {
    const text = chatInput.value.trim();
    if (!text) return;

    const currentSender = 'me'; // 目前發送者標記
    const currentDisplayName = '我'; // 顯示的名稱

    // 建立外層訊息容器
    const msgContainer = document.createElement('div');
    msgContainer.className = `msg-container ${currentSender}`;

    // 【核心邏輯】如果跟上一次發送者不同，才建立並顯示名稱
    if (lastSender !== currentSender) {
        const nameDiv = document.createElement('div');
        nameDiv.className = 'msg-username';
        nameDiv.textContent = currentDisplayName;
        msgContainer.appendChild(nameDiv);
        
        // 如果是連續訊息的第一則，可以微調與上方的間距
        msgContainer.classList.add('new-group');
    } else {
        // 如果是連續發送，縮短與上一條氣泡的上下間距
        msgContainer.classList.add('consecutive');
    }

    // 建立對話氣泡
    const bubbleP = document.createElement('p');
    bubbleP.className = 'bubble';
    bubbleP.textContent = text; // 支援換行

    // 組合並加入對話區
    msgContainer.appendChild(bubbleP);
    chatArea.appendChild(msgContainer);
    
    // 更新最後發送者紀錄
    lastSender = currentSender;

    // 重置輸入框與滾動條
    chatInput.value = "";
    chatInput.style.height = 'auto';
    chatArea.scrollTop = chatArea.scrollHeight; // 自動捲動到底部
}

// 自動調整高度函式
function autoResizeTextarea() {
    chatInput.style.height = 'auto';
    chatInput.style.height = (chatInput.scrollHeight) + 'px';
}

chatInput.addEventListener('keydown', (e) => {
    if (sendOnEnter) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault(); // 防止預設換行
            sendMessage();
        } else if (e.key === 'Enter' && e.shiftKey) {
            // Alt + Enter：手動加入換行符號
            const start = chatInput.selectionStart;
            const end = chatInput.selectionEnd;
            chatInput.value = chatInput.value.substring(0, start) + "\n" + chatInput.value.substring(end);
            chatInput.selectionStart = chatInput.selectionEnd = start + 1;
            autoResizeTextarea();
        }
    }
});

chatInput.addEventListener('input', autoResizeTextarea);
sendBtn.onclick = sendMessage; // 修正原本寫錯的對應名稱

// --- 錄音功能 ---
let mediaRecorder;
let audioChunks = [];
if (recordBtn) {
    recordBtn.onclick = async () => {
        if (!mediaRecorder) {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                mediaRecorder = new MediaRecorder(stream);
                mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
                mediaRecorder.onstop = () => {
                    const blob = new Blob(audioChunks, { type: 'audio/webm' });
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url; a.download = 'record.webm'; a.click();
                    audioChunks = [];
                };
            } catch (err) {
                console.error("錄音設備獲取失敗:", err);
                return;
            }
        }
        if (mediaRecorder.state === "inactive") {
            mediaRecorder.start();
            document.getElementById('recordStatus').classList.add('active');
            recordBtn.textContent = "🛑";
        } else {
            mediaRecorder.stop();
            document.getElementById('recordStatus').classList.remove('active');
            recordBtn.textContent = "🎤";
        }
    };
}

// --- 設定視窗控制 ---
const settingsModal = document.getElementById('settingsModal');
document.getElementById('settingsBtn').onclick = () => settingsModal.classList.add('active');
document.getElementById('closeSettings').onclick = () => {
    if (enterSetting) sendOnEnter = enterSetting.checked; // 連動設定狀態
    settingsModal.classList.remove('active');
};

// --- 收合/展開邏輯 ---
const chatPanel = document.getElementById('chatPanel');
const expandChat = document.getElementById('expandChat');

document.getElementById('toggleChat').onclick = () => {
    chatPanel.classList.add('collapsed');
    expandChat.classList.remove('hidden');
};

expandChat.onclick = () => {
    chatPanel.classList.remove('collapsed');
    expandChat.classList.add('hidden');
};

// --- 核心修復：統一在唯一的 window.onload 內執行初始化 ---
window.onload = () => {
    initCamera();
};