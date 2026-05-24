document.getElementById('loginForm').addEventListener('submit', function(e) {
    e.preventDefault(); // 防止表單真正送出刷新頁面

    const email = document.getElementById('email').value;
    const password = document.getElementById('password').value;

    // 這裡可以加入簡單的驗證邏輯
    if (email && password) {
        console.log("登入嘗試:", email);
        
        // 模擬登入成功，導向你的主頁面
        // 假設你的主頁面檔案名稱是 home.html
        window.location.href = 'home.html';
    } else {
        alert("請輸入帳號與密碼");
    }
});