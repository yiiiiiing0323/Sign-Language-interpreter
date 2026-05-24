// --- 功能控制 ---
const selfVideo = document.getElementById('selfVideo');
const selfPlaceholder = document.getElementById('selfPlaceholder');
const recordBtn = document.getElementById('recordBtn');
const camBtn = document.getElementById('camBtn');
const exitBtn = document.getElementById('exitBtn');
const timeDisplay = document.getElementById('currentTime');

// 1. 更新現在時間
function updateTime() {
    const now = new Date();
    timeDisplay.textContent = now.toLocaleTimeString('zh-TW', { hour12: false });
}
setInterval(updateTime, 1000);
updateTime();

// 2. 視訊處理 (畫面會填滿 block)
async function startCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
        selfVideo.srcObject = stream;
        // 開啟後隱藏頭像，顯示視訊
        selfPlaceholder.style.display = 'none';
        selfVideo.style.opacity = '1';
    } catch (err) {
        console.error("相機開啟失敗:", err);
    }
}

// 退出鍵功能
exitBtn.onclick = () => {
    if(confirm("確定要退出工作站嗎？")) {
        window.close(); // 或跳轉到其他頁面
    }
};

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

// 3. 收合/展開邏輯
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

// 初始化
window.onload = startCamera;