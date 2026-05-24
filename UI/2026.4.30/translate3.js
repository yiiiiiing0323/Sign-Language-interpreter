// --- DOM 選項 ---
const selfVideo = document.getElementById('selfVideo');
const selfAvatar = document.getElementById('selfAvatar');
const camBtn = document.getElementById('camBtn');
const chatInput = document.getElementById('chatInput');
const sendBtn = document.getElementById('sendBtn');
const enterSetting = document.getElementById('enterToSendSetting');
const timeDisplay = document.getElementById('currentDateTime');

let camOn = true;
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

// --- 攝影機暫時關閉功能 ---
async function initCamera() {
    try {
        localStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
        selfVideo.srcObject = localStream;
        selfAvatar.classList.add('hidden'); // 預設開啟時隱藏頭像
    } catch (err) {
        console.error("相機初始化失敗:", err);
    }
}

//click cam
camBtn.onclick = () => {
    if (!localStream) return;
    
    camOn = !camOn;
    const videoTrack = localStream.getVideoTracks()[0];
    
    if (videoTrack) {
        videoTrack.enabled = camOn; // 實質關閉鏡頭感應
        
        if (camOn) {
            selfVideo.classList.remove('off');
            selfAvatar.classList.add('hidden');
            camBtn.textContent = "📹";
        } else {
            selfVideo.classList.add('off');
            selfAvatar.classList.remove('hidden'); // 顯示用戶頭像
            camBtn.textContent = "🚫";
        }
    }
};

//click exit
exitBtn.onclick = () => {
    if(confirm("確定要退出工作站嗎？")) {
        window.close(); // 或跳轉到其他頁面
    }
};

// --- 訊息發送邏輯 (Enter/Alt+Enter) ---
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
// Textarea 自動增高
chatInput.addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = (this.scrollHeight) + 'px';
});

sendBtn.onclick = sendMsg;


// 錄音功能 (延續上次邏輯)
let mediaRecorder;
let audioChunks = [];
recordBtn.onclick = async () => {
    if (!mediaRecorder) {
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


// --- 設定視窗控制 ---
const settingsModal = document.getElementById('settingsModal');
document.getElementById('settingsBtn').onclick = () => settingsModal.classList.add('active');
document.getElementById('closeSettings').onclick = () => settingsModal.classList.remove('active');

// 初始化
window.onload = initCamera;

// 收合/展開邏輯
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