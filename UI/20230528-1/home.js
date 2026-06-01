// =========================================================================
// 1. 頭像下拉選單邏輯
// =========================================================================
const userIcon = document.getElementById('userIcon');
const userDropdown = document.getElementById('userDropdown');

// 點擊頭像：切換選單顯示/隱藏
userIcon?.addEventListener('click', (e) => {
    e.stopPropagation(); // 防止觸發下方的 document 點擊事件
    userDropdown?.classList.toggle('active');
});

// 點擊頁面任何地方：自動關閉下拉選單
document.addEventListener('click', () => {
    userDropdown?.classList.remove('active');
});


// =========================================================================
// 2. 【核心修改】登出確認邏輯（改為內建 confirm 彈窗）
// =========================================================================
// 供 HTML 上的 onclick="openLogoutModal()" 呼叫的全域函式
function openLogoutModal() {
    // 1. 先主動把頭像下拉選單收起來
    if (userDropdown) {
        userDropdown.classList.remove('active');
    }

    // 2. 彈出與 translate.js 相同風格的內建確認視窗
    if (confirm("確定要登出系統嗎？")) {
        window.location.href = 'index.html'; // 使用者點選「確定」後跳轉回登入頁
    }
}


// =========================================================================
// 3. 側邊欄抽屜選單邏輯
// =========================================================================
const menuToggle = document.getElementById('menuToggle');
const closeSidebar = document.getElementById('closeSidebar');
const sidebarMenu = document.getElementById('sidebarMenu');
const sidebarOverlay = document.getElementById('sidebarOverlay');

// 開啟側邊欄
function openSidebar() {
    sidebarMenu?.classList.add('active');
    sidebarOverlay?.classList.add('active');
}

// 關閉側邊欄
function closeSidebarMenu() {
    sidebarMenu?.classList.remove('active');
    sidebarOverlay?.classList.remove('active');
}

// 綁定點擊事件
menuToggle?.addEventListener('click', openSidebar);
closeSidebar?.addEventListener('click', closeSidebarMenu);
sidebarOverlay?.addEventListener('click', closeSidebarMenu);

// =========================================================================
// 4. 初始歡迎彈窗與勾選解鎖邏輯
// =========================================================================
document.addEventListener('DOMContentLoaded', () => {
    const welcomeModal = document.getElementById('welcomeModal');
    const readConfirmCheck = document.getElementById('readConfirmCheck');
    const enterSystemBtn = document.getElementById('enterSystemBtn');

    // 監聽 Checkbox 的勾選狀態切換
    readConfirmCheck?.addEventListener('change', function() {
        if (enterSystemBtn) {
            // 當打勾時，解除 disabled (false)；取消打勾時，重新加上 disabled (true)
            enterSystemBtn.disabled = !this.checked;
        }
    });

    // 點擊「進入系統」按鈕：關閉彈窗
    enterSystemBtn?.addEventListener('click', () => {
        welcomeModal?.classList.remove('active');
    });
});