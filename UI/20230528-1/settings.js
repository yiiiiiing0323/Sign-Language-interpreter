// =========================================================================
// 1. 內部設定 Tabs 分頁切換邏輯
// =========================================================================
const tabBtns = document.querySelectorAll('.tab-btn');
const tabPanels = document.querySelectorAll('.tab-panel');

tabBtns.forEach(btn => {
    btn.addEventListener('click', function() {
        // 移除所有按鈕的 active 樣式
        tabBtns.forEach(b => b.classList.remove('active'));
        // 隱藏所有內容面板
        tabPanels.forEach(p => p.classList.remove('active'));

        // 幫目前點擊的按鈕加上 active
        this.classList.add('active');
        
        // 對應秀出對應的 ID 面板
        const targetTab = this.getAttribute('data-tab');
        const targetPanel = document.getElementById(`panel-${targetTab}`);
        if (targetPanel) {
            targetPanel.classList.add('active');
        }
    });
});

// 儲存按鈕模擬反饋
document.getElementById('saveSettingsBtn')?.addEventListener('click', () => {
    alert('設定已成功儲存！');
});

// =========================================================================
// 2. 【核心修改】全域函式：供 HTML 的 onclick="openLogoutModal()" 呼叫
// =========================================================================
function openLogoutModal() {
    const userDropdown = document.getElementById('userDropdown');
    // 1. 先主動把頭像下拉選單收起來
    if (userDropdown) {
        userDropdown.classList.remove('active');
    }

    // 2. 彈出內建確認視窗
    if (confirm("確定要登出系統嗎？")) {
        window.location.href = 'index.html'; // 使用者點選「確定」後跳轉回登入頁
    }
}

// =========================================================================
// 3. 外部大側邊欄抽屜選單控制 (與主頁一致)
// =========================================================================
const menuToggle = document.getElementById('menuToggle');
const closeSidebar = document.getElementById('closeSidebar');
const sidebarMenu = document.getElementById('sidebarMenu');
const sidebarOverlay = document.getElementById('sidebarOverlay');

menuToggle?.addEventListener('click', () => {
    sidebarMenu?.classList.add('active');
    sidebarOverlay?.classList.add('active');
});

function closeSidebarMenu() {
    sidebarMenu?.classList.remove('active');
    sidebarOverlay?.classList.remove('active');
}
closeSidebar?.addEventListener('click', closeSidebarMenu);
sidebarOverlay?.addEventListener('click', closeSidebarMenu);

// =========================================================================
// 4. 頭像下拉選單邏輯 (與主頁一致)
// =========================================================================
const userIcon = document.getElementById('userIcon');
const userDropdown = document.getElementById('userDropdown');

// 點擊頭像開關選單
userIcon?.addEventListener('click', (e) => {
    e.stopPropagation();
    userDropdown?.classList.toggle('active');
});

document.addEventListener('click', () => {
    userDropdown?.classList.remove('active');
});