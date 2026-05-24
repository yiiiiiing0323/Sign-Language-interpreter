// --- 1. 頭像下拉選單邏輯 ---
const userIcon = document.getElementById('userIcon');
const userDropdown = document.getElementById('userDropdown');

// 點擊頭像：切換選單顯示/隱藏
userIcon?.addEventListener('click', (e) => {
    e.stopPropagation(); // 防止觸發下方的 document 點擊事件
    userDropdown.classList.toggle('active');
});

// 點擊頁面任何地方：自動關閉下拉選單
document.addEventListener('click', () => {
    userDropdown?.classList.remove('active');
});

// --- 2. 登出確認彈窗邏輯 ---
const logoutLink = document.querySelector('.logout-link');
const logoutModal = document.getElementById('logout-modal');
const confirmLogoutBtn = document.getElementById('confirm-logout');
const cancelLogoutBtn = document.getElementById('cancel-logout');

// 點擊選單中的 Logout：攔截跳轉並開啟確認視窗
logoutLink?.addEventListener('click', (e) => {
    e.preventDefault(); 
    logoutModal.classList.add('active');
});

// 彈窗內的「取消」按鈕：關閉視窗
cancelLogoutBtn?.addEventListener('click', () => {
    logoutModal.classList.remove('active');
});

// 彈窗內的「確認登出」按鈕：跳轉回登入頁面
confirmLogoutBtn?.addEventListener('click', () => {
    // 若有需要清除登入狀態（如 localStorage），可在這加入
    window.location.href = 'index.html';
});

// 點擊彈窗半透明背景：關閉視窗
logoutModal?.addEventListener('click', (e) => {
    if (e.target === logoutModal) {
        logoutModal.classList.remove('active');
    }
});

// --- 3. 針對 AI 頁面的延遲跳轉邏輯 ---
const aiLink = document.querySelector('a[href="translate.html"]');

aiLink?.addEventListener('click', function(e) {
    e.preventDefault(); // 先攔截，不讓它立刻跳轉
    
    const targetUrl = this.href;

    // 1. 幫 body 加上淡出效果的 class
    document.body.classList.add('page-exit');

    // 2. 設定延遲時間 (例如 800 毫秒)
    setTimeout(() => {
        window.location.href = targetUrl;
    }, 800); 
});