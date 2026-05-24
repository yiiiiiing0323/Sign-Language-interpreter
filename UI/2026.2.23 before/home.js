document.getElementById('join-btn').addEventListener('click', () => {
    window.location.href = 'meeting.html';
});

const hamburger = document.querySelector('.hamburger-menu');
const layout = document.querySelector('.main-layout');

hamburger.addEventListener('click', () => {
    layout.classList.toggle('sidebar-collapsed');
});


const accountMenu = document.querySelector('.account-menu');
const accountIcon = document.querySelector('.account-icon');

accountIcon.addEventListener('click', (e) => {
    e.stopPropagation();
    accountMenu.classList.toggle('open');
});

document.addEventListener('click', () => {
    accountMenu.classList.remove('open');
});

const startBtn = document.querySelector('.btn-start-meeting');
const modal = document.getElementById('meeting-modal');
const cancelBtn = document.getElementById('cancel-meeting');
const confirmBtn = document.getElementById('confirm-meeting');

startBtn.addEventListener('click', () => {
    modal.classList.add('active');
});

cancelBtn.addEventListener('click', () => {
    modal.classList.remove('active');
});

// 點確認後進入會議頁面（可以帶上選擇值）
confirmBtn.addEventListener('click', () => {
    const myAI = document.getElementById('my-ai').value;
    const peerAI = document.getElementById('peer-ai').value;

    console.log("我的 AI:", myAI, "對方 AI:", peerAI);

    // 這裡可以改成 window.location.href 導頁
    window.location.href = `meeting.html?myAI=${myAI}&peerAI=${peerAI}`;
});
