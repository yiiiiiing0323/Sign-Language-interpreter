// meeting_new.js

// --- DOM 選項 ---
const chatPanel = document.getElementById('chatPanel');
const toggleChat = document.getElementById('toggleChat');
const expandChat = document.getElementById('expandChat');
const recordBtn = document.getElementById('recordBtn');
const recordStatus = document.getElementById('recordStatus');

// --- 錄音相關變數 ---
let mediaRecorder;
let audioChunks = [];
let isRecording = false;

// --- 1. 收合邏輯 ---
toggleChat.addEventListener('click', () => {
    chatPanel.classList.add('collapsed');
    expandChat.classList.remove('hidden');
});

expandChat.addEventListener('click', () => {
    chatPanel.classList.remove('collapsed');
    expandChat.classList.add('hidden');
});

// --- 2. 錄音與儲存功能 ---
async function setupRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        
        mediaRecorder = new MediaRecorder(stream);

        mediaRecorder.ondataavailable = (event) => {
            audioChunks.push(event.data);
        };

        mediaRecorder.onstop = () => {
            const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
            const audioUrl = URL.createObjectURL(audioBlob);
            
            // 建立下載連結
            const link = document.createElement('a');
            link.href = audioUrl;
            link.download = `meeting-record-${Date.now()}.webm`;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            
            // 重置
            audioChunks = [];
        };

    } catch (err) {
        console.error("無法取得麥克風權限:", err);
        alert("請允許麥克風權限以進行錄音");
    }
}

recordBtn.addEventListener('click', async () => {
    if (!mediaRecorder) await setupRecording();

    if (!isRecording) {
        // 開始錄音
        mediaRecorder.start();
        isRecording = true;
        recordBtn.textContent = "🛑"; // 變成停止符號
        recordStatus.classList.add('active');
        console.log("錄音中...");
    } else {
        // 停止錄音
        mediaRecorder.stop();
        isRecording = false;
        recordBtn.textContent = "🎤";
        recordStatus.classList.remove('active');
        console.log("錄音結束，準備下載...");
    }
});

// --- 3. 鏡頭初始化 ---
async function initVideo() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        document.getElementById('selfVideo').srcObject = stream;
    } catch (e) {
        console.log("鏡頭開啟失敗");
    }
}

window.onload = initVideo;