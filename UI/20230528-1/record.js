// =========================================================================
// 模擬歷史對話數據庫
// =========================================================================
const mockRecords = [
    {
        id: 1,
        title: "關於專案排程與 UI 設計討論",
        time: "2026/05/26 14:32",
        preview: "我想確認一下首頁導覽列的粗黑線風格...",
        dialogue: [
            { sender: "me", name: "我", text: "我想確認一下首頁導覽列的粗黑線風格，這樣算切換頁面嗎？" },
            { sender: "ai", name: "AI", text: "這在技術上被稱為「單頁面切換」 (SPA)，透過 JavaScript 控制顯示隱藏，體驗會比傳統網頁跳轉更流暢！" },
            { sender: "me", name: "我", text: "原來如此，那我想要改成真的切換頁面，且網址列也要變動。" }
        ]
    },
    {
        id: 2,
        title: "前端相機權限異常排查",
        time: "2026/05/24 10:15",
        preview: "為什麼我的 window.onload 沒有開鏡頭？",
        dialogue: [
            { sender: "me", name: "我", text: "為什麼我的 window.onload 沒有成功開啟鏡頭？" },
            { sender: "ai", name: "AI", text: "檢查後發現您的程式碼中 window.onload 被重複宣告覆蓋掉了！請將初始化統一寫在同一個載入函式內。" },
            { sender: "me", name: "我", text: "改掉之後就順利開了，謝謝！" }
        ]
    },
    {
        id: 3,
        title: "Neubrutalism 彈窗設計規範",
        time: "2026/05/20 18:00",
        preview: "我想做一個好看的偏好設定介面...",
        dialogue: [
            { sender: "me", name: "我", text: "我想做一個好看的偏好設定介面，要在設定彈窗裡面。" },
            { sender: "ai", name: "AI", text: "為您推薦左側兩欄式選單搭配仿 iOS 滑塊開關（Switch）的設計，具有強烈的俐落幾何感，非常契合您的專案樣式！" }
        ]
    }
];

// =========================================================================
// 【核心修改】全域函式：改用與 translate.js 相同的內建 confirm 彈窗
// =========================================================================
function openLogoutModal() {
    const userDropdown = document.getElementById('userDropdown');
    if (userDropdown) userDropdown.classList.remove('active'); // 先收起頭像選單

    // 使用與 translate.js 相同的瀏覽器內建確認視窗
    if (confirm("確定要登出系統嗎？")) {
        window.location.href = 'index.html'; // 點擊確定後導向登入頁
    }
}

// =========================================================================
// 核心互動邏輯
// =========================================================================
document.addEventListener('DOMContentLoaded', () => {
    const searchInput = document.getElementById('searchInput');
    const recordList = document.getElementById('recordList');
    const emptyView = document.getElementById('emptyView');
    const chatDetailPanel = document.getElementById('chatDetailPanel');
    
    const detailTitle = document.getElementById('detailTitle');
    const detailTime = document.getElementById('detailTime');
    const detailChatArea = document.getElementById('detailChatArea');

    let currentActiveId = null;

    // --- 1. 渲染左側列表功能 ---
    function renderList(recordsToRender) {
        if (!recordList) return;
        recordList.innerHTML = "";
        
        if (recordsToRender.length === 0) {
            recordList.innerHTML = `<div style="text-align:center; color:#888; padding:20px; font-weight:bold;">❌ 找不到相關紀錄</div>`;
            return;
        }

        recordsToRender.forEach(data => {
            const itemDiv = document.createElement('div');
            itemDiv.className = `record-item ${currentActiveId === data.id ? 'active' : ''}`;
            itemDiv.innerHTML = `
                <div class="item-meta">
                    <span>📁 對話紀錄</span>
                    <span>${data.time.split(' ')[0]}</span>
                </div>
                <h4 class="item-title">${data.title}</h4>
                <p class="item-preview">${data.preview}</p>
            `;

            itemDiv.addEventListener('click', () => {
                currentActiveId = data.id;
                renderList(recordsToRender);
                showChatDetail(data);
            });

            recordList.appendChild(itemDiv);
        });
    }

    // --- 2. 顯示右側詳細對話氣泡 ---
    function showChatDetail(record) {
        if (!chatDetailPanel || !emptyView) return;
        emptyView.classList.add('hidden');
        chatDetailPanel.classList.remove('hidden');

        if (detailTitle) detailTitle.textContent = record.title;
        if (detailTime) detailTime.textContent = record.time;
        
        if (detailChatArea) {
            detailChatArea.innerHTML = "";
            record.dialogue.forEach(msg => {
                const msgContainer = document.createElement('div');
                msgContainer.className = `msg-container ${msg.sender}`;
                msgContainer.innerHTML = `
                    <div class="msg-username">${msg.name}</div>
                    <div class="bubble">${msg.text}</div>
                `;
                detailChatArea.appendChild(msgContainer);
            });
            detailChatArea.scrollTop = detailChatArea.scrollHeight;
        }
    }

    // --- 3. 關鍵字查詢即時篩選功能 ---
    searchInput?.addEventListener('input', function() {
        const keyword = this.value.toLowerCase().trim();
        const filtered = mockRecords.filter(item => {
            const matchTitle = item.title.toLowerCase().includes(keyword);
            const matchContent = item.dialogue.some(msg => msg.text.toLowerCase().includes(keyword));
            return matchTitle || matchContent;
        });
        renderList(filtered);
    });

    // 執行初始列表渲染
    renderList(mockRecords);

    // =========================================================================
    // 4. 側邊欄抽屜選單控制 (與主頁完全一致)
    // =========================================================================
    const menuToggle = document.getElementById('menuToggle');
    const closeSidebar = document.getElementById('closeSidebar');
    const sidebarMenu = document.getElementById('sidebarMenu');
    const sidebarOverlay = document.getElementById('sidebarOverlay');

    menuToggle?.addEventListener('click', () => {
        sidebarMenu?.classList.add('active');
        sidebarOverlay?.addEventListener('active');
    });

    const closeMenu = () => {
        sidebarMenu?.classList.remove('active');
        sidebarOverlay?.classList.remove('active');
    };
    closeSidebar?.addEventListener('click', closeMenu);
    sidebarOverlay?.addEventListener('click', closeMenu);

    // =========================================================================
    // 5. 個人頭像選單按鈕功能
    // =========================================================================
    const userIcon = document.getElementById('userIcon');
    const userDropdown = document.getElementById('userDropdown');

    // 點擊頭像打開/關閉下拉選單
    userIcon?.addEventListener('click', (e) => {
        e.stopPropagation();
        userDropdown?.classList.toggle('active');
    });

    // 點擊頁面任何地方自動關閉頭像下拉
    document.addEventListener('click', () => {
        userDropdown?.classList.remove('active');
    });
});