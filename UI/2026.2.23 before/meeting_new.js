// ==================== DOM 選取 ====================
const micBtn = document.querySelector('.ctrl.mic');
const camBtn = document.querySelector('.ctrl.cam');
const endBtn = document.querySelector('.ctrl.end');

const infoBtn = document.querySelector('.ctrl.info');
const membersBtn = document.querySelector('.ctrl.members');
const chatToggleBtn = document.querySelector('.ctrl.chat-toggle');
const settingsBtn = document.querySelector('.ctrl.settings');

const infoPanel = document.getElementById('infoPanel');
const membersPanel = document.getElementById('membersPanel');
const chatSide = document.querySelector('.meet-side');

const endModal = document.getElementById('endCallModal');
const cancelEndBtn = document.getElementById('cancelEnd');
const confirmEndBtn = document.getElementById('confirmEnd');

const settingsModal = document.getElementById('settingsModal');
const closeSettingsBtn = document.getElementById('closeSettings');

const selfVideo = document.getElementById('selfVideo');
const subtitleContainer = document.querySelector('.subtitle-container');

// ==================== 狀態 ====================
let micOn = true;
let camOn = true;
let localStream = null;

// ==================== Panel 統一管理 ====================
const panels = {
    info: infoPanel,
    members: membersPanel,
    chat: chatSide
};

function closeAllPanels() {
    Object.values(panels).forEach(panel => {
        panel?.classList.add('hidden');
    });
}

function togglePanel(type) {
    const panel = panels[type];
    if (!panel) return;

    const isHidden = panel.classList.contains('hidden');
    closeAllPanels();
    if (isHidden) panel.classList.remove('hidden');
}

// ==================== 點擊空白處自動關閉 ====================
document.addEventListener('click', (e) => {
    if (
        e.target.closest('.ctrl') ||
        e.target.closest('#infoPanel') ||
        e.target.closest('#membersPanel') ||
        e.target.closest('.meet-side') ||
        e.target.closest('.modal-content')
    ) {
        return;
    }
    closeAllPanels();
});

// ==================== 按鈕事件 ====================
infoBtn?.addEventListener('click', (e) => {
    e.stopPropagation();
    togglePanel('info');
});

membersBtn?.addEventListener('click', (e) => {
    e.stopPropagation();
    togglePanel('members');
});

chatToggleBtn?.addEventListener('click', (e) => {
    e.stopPropagation();
    togglePanel('chat');
});

settingsBtn?.addEventListener('click', (e) => {
    e.stopPropagation();
    settingsModal.classList.add('active');
});

// ==================== 設定視窗 ====================
closeSettingsBtn?.addEventListener('click', () => {
    settingsModal.classList.remove('active');
});

// ==================== 掛斷會議 ====================
endBtn?.addEventListener('click', () => {
    endModal.classList.add('active');
});

cancelEndBtn?.addEventListener('click', () => {
    endModal.classList.remove('active');
});

confirmEndBtn?.addEventListener('click', () => {
    window.location.href = 'record.html';
});

// ==================== 麥克風 / 鏡頭 ====================
async function startMedia() {
    try {
        localStream = await navigator.mediaDevices.getUserMedia({
            video: true,
            audio: true
        });
        selfVideo.srcObject = localStream;
    } catch (err) {
        alert('請允許使用麥克風與鏡頭');
        console.error(err);
    }
}

micBtn?.addEventListener('click', async () => {
    if (!localStream) await startMedia();

    const audioTrack = localStream?.getAudioTracks()[0];
    if (!audioTrack) return;

    micOn = !micOn;
    audioTrack.enabled = micOn;
    micBtn.textContent = micOn ? '🎤' : '🔇';
});

camBtn?.addEventListener('click', async () => {
    if (!localStream) await startMedia();

    const videoTrack = localStream?.getVideoTracks()[0];
    if (!videoTrack) return;

    camOn = !camOn;
    videoTrack.enabled = camOn;
    camBtn.textContent = camOn ? '📹' : '🚫';
});

// ==================== 模擬即時字幕 ====================
const demoSubtitles = [
    "Hello, how are you?",
    "我正在使用 AI 生成即時翻譯",
    "今天我們討論專案進度",
    "字幕會同步顯示",
    "感謝您的參與"
];

let subtitleIndex = 0;

setInterval(() => {
    if (!subtitleContainer) return;

    const el = document.createElement('div');
    el.className = 'subtitle';
    el.textContent = demoSubtitles[subtitleIndex];
    subtitleContainer.appendChild(el);

    if (subtitleContainer.children.length > 3) {
        subtitleContainer.removeChild(subtitleContainer.firstChild);
    }

    subtitleIndex = (subtitleIndex + 1) % demoSubtitles.length;
}, 3000);

// ==================== 初始化 ====================
window.addEventListener('load', () => {
    closeAllPanels();
});
