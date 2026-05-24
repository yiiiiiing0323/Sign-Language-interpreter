// -------------------- DOM 選取 --------------------
const chatToggleBtn = document.querySelector('.ctrl.chat-toggle');
const chatSide = document.querySelector('.meet-side');

const myVideo = document.querySelector('.my-video');
const toggleSelfVideoBtn = document.querySelector('.toggle-self-video');
const selfVideoArrow = document.querySelector('.self-video-arrow');

const micBtn = document.querySelector('.ctrl.mic');
const camBtn = document.querySelector('.ctrl.cam');

const subtitleContainer = document.querySelector('.subtitle-container');

const endBtn = document.querySelector('.ctrl.end');
const endModal = document.getElementById('endCallModal');
const infoBtn = document.querySelector('.ctrl.info');
const infoPanel = document.getElementById('infoPanel');
const membersBtn = document.querySelector('.ctrl.members');
const membersPanel = document.getElementById('membersPanel');
const settingsBtn = document.querySelector('.ctrl.settings');
const settingsModal = document.getElementById('settingsModal');
const selfVideo = document.getElementById('selfVideo');

// 狀態
let micOn = true;
let camOn = true;
let selfVideoVisible = true;
let chatVisible = false;
let localStream = null;

const panels = {
    chat: document.querySelector('.meet-side'),
    info: document.getElementById('infoPanel'),
    members: document.getElementById('membersPanel')
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

    if (isHidden) {
        panel.classList.remove('hidden');
    }
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

async function startMedia() {
    try {
        localStream = await navigator.mediaDevices.getUserMedia({
            video: true,
            audio: true
        });

        selfVideo.srcObject = localStream;
    } catch (err) {
        console.error('無法存取裝置：', err);
        alert('請允許使用麥克風與鏡頭');
    }
}


// -------------------- 聊天側欄收放 --------------------
if (chatToggleBtn && chatSide) {
    chatToggleBtn.addEventListener('click', () => {
        chatVisible = !chatVisible;
        chatSide.classList.toggle('hidden', !chatVisible);
    });
}


// -------------------- 自己小畫面收合 --------------------
toggleSelfVideoBtn.addEventListener('click', () => {
    selfVideoVisible = false;
    myVideo.style.display = 'none';
    selfVideoArrow.classList.remove('hidden'); // 顯示箭頭
});

selfVideoArrow.addEventListener('click', () => {
    selfVideoVisible = true;
    myVideo.style.display = 'flex';
    selfVideoArrow.classList.add('hidden'); // 隱藏箭頭
});

// -------------------- 麥克風 / 攝影機開關 --------------------
micBtn.addEventListener('click', async () => {
    if (!localStream) {
        await startMedia();
        micOn = true;
    } else {
        const audioTrack = localStream.getAudioTracks()[0];
        micOn = !micOn;
        audioTrack.enabled = micOn;
    }

    micBtn.textContent = micOn ? '🎤' : '🔇';
});


camBtn.addEventListener('click', async () => {
    if (!localStream) {
        await startMedia();
        camOn = true;
    } else {
        const videoTrack = localStream.getVideoTracks()[0];
        camOn = !camOn;
        videoTrack.enabled = camOn;
    }

    camBtn.textContent = camOn ? '📹' : '🚫';
});

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

    // 保持最多 3 條字幕
    if (subtitleContainer.children.length > 3) {
        subtitleContainer.removeChild(subtitleContainer.firstChild);
    }

    subtitleIndex = (subtitleIndex + 1) % demoSubtitles.length;
}, 3000);

// -------------------- 掛斷按鈕 --------------------
endBtn.addEventListener('click', () => {
    endModal.classList.add('active');
});

document.getElementById('cancelEnd').onclick = () => {
    endModal.classList.remove('active');
};

document.getElementById('confirmEnd').onclick = () => {
    window.location.href = 'record.html';
};

// -------------------- info按鈕 --------------------
infoBtn.onclick = () => togglePanel('info');

// -------------------- members按鈕 --------------------
membersBtn.onclick = () => togglePanel('members');

// -------------------- settings按鈕 --------------------
settingsBtn.onclick = () => {
    settingsModal.classList.add('active');
};

document.getElementById('closeSettings').onclick = () => {
    settingsModal.classList.remove('active');
};
