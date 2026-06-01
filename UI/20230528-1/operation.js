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

// --- 3. 大綱導覽聯動高亮 (ScrollSpy 閱讀進度追蹤) ---
    const mainContentArea = document.querySelector('.operation-main-content');
    const tocItems = document.querySelectorAll('.toc-item');
    const sections = document.querySelectorAll('.content-section');

    if (mainContentArea) {
        mainContentArea.addEventListener('scroll', () => {
            let currentActiveSectionId = "";
            
            // 抓取當前滾動到的區塊位置
            sections.forEach(section => {
                const sectionTop = section.offsetTop;
                // 扣除一些微調空間，讓滾動到接近頂端時就觸發
                if (mainContentArea.scrollTop >= sectionTop - 60) {
                    currentActiveSectionId = section.getAttribute('id');
                }
            });

            // 更新左側大綱的高亮狀態
            tocItems.forEach(item => {
                item.classList.remove('active');
                if (item.getAttribute('href') === `#${currentActiveSectionId}`) {
                    item.classList.add('active');
                }
            });
        });
    }

    // 點擊大綱按鈕時的手動點擊高亮修正
    tocItems.forEach(item => {
        item.addEventListener('click', function() {
            tocItems.forEach(i => i.classList.remove('active'));
            this.classList.add('active');
        });
    });
});