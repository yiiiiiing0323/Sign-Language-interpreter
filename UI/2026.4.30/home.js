// --- 下拉選單邏輯 ---
const userIcon = document.getElementById('userIcon');
const userDropdown = document.getElementById('userDropdown');

userIcon.addEventListener('click', (e) => {
    e.stopPropagation(); // 防止點擊頭像時觸發 document 的關閉事件
    userDropdown.classList.toggle('active');
});

// 點擊頁面其他地方時，關閉下拉選單
document.addEventListener('click', () => {
    userDropdown.classList.remove('active');
});

// --- 會議功能邏輯 (保持原本功能) ---
const joinBtn = document.getElementById('join-btn');
const startBtn = document.querySelector('.btn-start-meeting');
const modal = document.getElementById('meeting-modal');
const cancelBtn = document.getElementById('cancel-meeting');
const confirmBtn = document.getElementById('confirm-meeting');

// 加入會議
joinBtn?.addEventListener('click', () => {
    window.location.href = 'meeting.html';
});

// 開啟發起會議彈窗
startBtn?.addEventListener('click', () => {
    modal.classList.add('active');
});

// 取消彈窗
cancelBtn?.addEventListener('click', () => {
    modal.classList.remove('active');
});

// 確認發起會議
confirmBtn?.addEventListener('click', () => {
    const myAI = document.getElementById('my-ai').value;
    const peerAI = document.getElementById('peer-ai').value;

    console.log("我的 AI:", myAI, "對方 AI:", peerAI);

    // 導向會議頁面並傳遞參數
    window.location.href = `meeting.html?myAI=${myAI}&peerAI=${peerAI}`;
});