// -------------------- DOM 選取 --------------------
const chatToggleBtn = document.querySelector('.ctrl.chat-toggle');
const chatSide = document.querySelector('.meet-side');

const micBtn = document.querySelector('.ctrl.mic');
const camBtn = document.querySelector('.ctrl.cam');
const endBtn = document.querySelector('.ctrl.end');
const infoBtn = document.querySelector('.ctrl.info');
const membersBtn = document.querySelector('.ctrl.members');
const settingsBtn = document.querySelector('.ctrl.settings');

const endModal = document.getElementById('endCallModal');
const infoPanel = document.getElementById('infoPanel');
const membersPanel = document.getElementById('membersPanel');
const settingsModal = document.getElementById('settingsModal');

const myVideoContainer = document.getElementById('myVideoContainer');
const selfVideo = document.getElementById('selfVideo');
const toggleMiniBtn = document.getElementById('toggleMiniVideo');

const subtitleContainer = document.querySelector('.subtitle-container');

// -------------------- 狀態 --------------------
let micOn = true;
let camOn = true;
let chatVisible = false;
let localStream = null;
let isMini = false;

// -------------------- 面板統一管理 --------------------
const panels = {
    chat: chatSide,
    info: infoPanel,
    members: membersPanel
};

function closeAllPanels() {
    Object.values(panels).forEach(panel => {
        panel?.classList.add('hidden');
    });
}

function togglePanel(type) {
    const panel = panels[type];
    if (!panel) return;
    const hidden = panel.classList.contains('hidden');
    closeAllPanels();
    if (hidden) panel.classList.remove('hidden');
}

document.addEventListener('click', (e) => {
    if (
        e.target.closest('.ctrl') ||
        e.target.closest('.meet-side') ||
        e.target.closest('#infoPanel') ||
        e.target.closest('#membersPanel')
    ) return;
    closeAllPanels();
});

// -------------------- 聊天側欄 --------------------
chatToggleBtn.addEventListener('click', () => {
    chatVisible = !chatVisible;
    chatSide.classList.toggle('hidden', !chatVisible);
});

// -------------------- 麥克風 / 攝影機 --------------------
async function startMedia() {
    try {
        localStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
        selfVideo.srcObject = localStream;
    } catch (err) {
        console.error('無法取得鏡頭或麥克風:', err);
        alert('請允許使用鏡頭與麥克風');
    }
}

micBtn.addEventListener('click', () => {
    if (!localStream) return;
    const audioTrack = localStream.getAudioTracks()[0];
    micOn = !micOn;
    audioTrack.enabled = micOn;
    micBtn.textContent = micOn ? '🎤' : '🔇';
});

camBtn.addEventListener('click', () => {
    if (!localStream) return;
    const videoTrack = localStream.getVideoTracks()[0];
    camOn = !camOn;
    videoTrack.enabled = camOn;
    camBtn.textContent = camOn ? '📹' : '🚫';
});

// -------------------- 掛斷 --------------------
endBtn.addEventListener('click', () => endModal.classList.add('active'));
document.getElementById('cancelEnd').onclick = () => endModal.classList.remove('active');
document.getElementById('confirmEnd').onclick = () => window.location.href = 'home.html';

// -------------------- info / members / settings --------------------
infoBtn.onclick = () => togglePanel('info');
membersBtn.onclick = () => togglePanel('members');
settingsBtn.onclick = () => settingsModal.classList.add('active');
document.getElementById('closeSettings').onclick = () => settingsModal.classList.remove('active');

// -------------------- 模擬即時字幕 --------------------
const demoSubtitles = [
    "Hello, how are you?",
    "我正在使用 AI 生成翻譯",
    "今天我們討論專案進度",
    "字幕會即時更新",
    "感謝您的參與！"
];

let subtitleIndex = 0;
setInterval(() => {
    const text = demoSubtitles[subtitleIndex];
    const subEl = document.createElement('div');
    subEl.className = 'subtitle';
    subEl.textContent = text;
    subtitleContainer.appendChild(subEl);
    if (subtitleContainer.children.length > 3) subtitleContainer.removeChild(subtitleContainer.firstChild);
    subtitleIndex = (subtitleIndex + 1) % demoSubtitles.length;
}, 3000);

// -------------------- 小視訊拖移（整個螢幕） ------------------
let isDragging = false, offsetX = 0, offsetY = 0;

myVideoContainer.addEventListener('mousedown', (e) => {
    if (e.target === toggleMiniBtn) return; // 避開收合按鈕
    isDragging = true;
    offsetX = e.clientX - myVideoContainer.offsetLeft;
    offsetY = e.clientY - myVideoContainer.offsetTop;
    myVideoContainer.style.transition = 'none';
});

document.addEventListener('mousemove', (e) => {
    if (!isDragging) return;
    // 計算新的位置
    let x = e.clientX - offsetX;
    let y = e.clientY - offsetY;
    // 限制範圍：整個視窗內
    x = Math.max(0, Math.min(window.innerWidth - myVideoContainer.offsetWidth, x));
    y = Math.max(0, Math.min(window.innerHeight - myVideoContainer.offsetHeight, y));
    myVideoContainer.style.left = x + 'px';
    myVideoContainer.style.top = y + 'px';
});

document.addEventListener('mouseup', () => {
    isDragging = false;
    myVideoContainer.style.transition = 'all 0.2s';
});


// -------------------- 小視訊收合 ------------------
toggleMiniBtn.addEventListener('click', () => {
    if (!isMini) {
        myVideoContainer.style.width = '50px';
        myVideoContainer.style.height = '40px';
        selfVideo.style.display = 'none';
        toggleMiniBtn.textContent = '➡️';
        isMini = true;
    } else {
        myVideoContainer.style.width = '200px';
        myVideoContainer.style.height = '150px';
        selfVideo.style.display = 'block';
        toggleMiniBtn.textContent = '⬅️';
        isMini = false;
    }
});

// -------------------- 小視訊縮放 ------------------
myVideoContainer.addEventListener('wheel', (e) => {
    e.preventDefault();
    let w = myVideoContainer.offsetWidth + (e.deltaY < 0 ? 20 : -20);
    let h = myVideoContainer.offsetHeight + (e.deltaY < 0 ? 20 : -20);
    w = Math.max(120, Math.min(400, w));
    h = Math.max(90, Math.min(300, h));
    myVideoContainer.style.width = w + 'px';
    myVideoContainer.style.height = h + 'px';
});

// -------------------- 啟動攝影機 ------------------
startMedia();
