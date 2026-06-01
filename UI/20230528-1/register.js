

document.addEventListener('DOMContentLoaded', () => {
    const registerForm = document.getElementById('registerForm');
    const passwordInput = document.getElementById('password');
    const confirmPasswordInput = document.getElementById('confirmPassword');
    const errorDiv = document.getElementById('errorMessage');

    // --- 1. 即時檢查兩次密碼是否一致 ---
    function checkPasswordMatch() {
        // 如果確認密碼欄位還空著，先不提示錯誤
        if (!confirmPasswordInput.value) {
            errorDiv.classList.add('hidden');
            confirmPasswordInput.style.borderColor = '#000';
            return;
        }

        if (passwordInput.value !== confirmPasswordInput.value) {
            errorDiv.textContent = "❌ 兩次輸入的密碼不相同！";
            errorDiv.classList.remove('hidden');
            confirmPasswordInput.style.borderColor = '#d9534f'; // 變成警告紅邊框
        } else {
            errorDiv.classList.add('hidden');
            confirmPasswordInput.style.borderColor = '#2b7a43'; // 一致時變成安全綠邊框
        }
    }

    // 綁定輸入事件，讓使用者在打字時就能馬上有視覺回饋
    passwordInput.addEventListener('input', checkPasswordMatch);
    confirmPasswordInput.addEventListener('input', checkPasswordMatch);

    // --- 2. 表單提交處理 ---
    registerForm?.addEventListener('submit', (e) => {
        e.preventDefault(); // 攔截預設跳轉

        const username = document.getElementById('username').value.trim();
        const email = document.getElementById('email').value.trim();
        const password = passwordInput.value;
        const confirmPassword = confirmPasswordInput.value;

        // 安全檢查：密碼長度驗證
        if (password.length < 6) {
            errorDiv.textContent = "❌ 為了您的安全，密碼長度至少需要 6 個字元！";
            errorDiv.classList.remove('hidden');
            passwordInput.focus();
            return;
        }

        // 安全檢查：再次確認密碼一致性
        if (password !== confirmPassword) {
            errorDiv.textContent = "❌ 兩次輸入的密碼不相同，請重新確認！";
            errorDiv.classList.remove('hidden');
            confirmPasswordInput.focus();
            return;
        }

        // 隱藏錯誤提示並執行註冊成功
        errorDiv.classList.add('hidden');

        /* 💡 提示：未來在這裡可以串接後端 API 
           fetch('/api/register', { method: 'POST', body: ... })
        */
        
        // 模擬儲存本機狀態（選填）
        localStorage.setItem('registeredEmail', email);
        localStorage.setItem('registeredUser', username);

        alert(`🎉 註冊成功！歡迎您，${username}。\n將自動為您跳轉至登入頁面。`);
        window.location.href = 'index.html'; // 導向登入頁
    });
});