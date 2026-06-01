

document.addEventListener('DOMContentLoaded', () => {
    const forgotForm = document.getElementById('forgotForm');
    const emailInput = document.getElementById('forgotEmail');
    const infoDiv = document.getElementById('infoMessage');
    const submitBtn = document.querySelector('.btn-auth');

    forgotForm?.addEventListener('submit', (e) => {
        e.preventDefault(); // 攔截表單預設動作

        const email = emailInput.value.trim();

        // 1. 顯示成功發送的提示字與樣式
        infoDiv.textContent = `📩 重設密碼連結已發送至 ${email}，請至您的信箱查收！`;
        infoDiv.className = "info-msg success"; // 切換成漂亮的綠色成功底色
        infoDiv.classList.remove('hidden');

        // 2. 鎖定輸入框與按鈕，啟動倒數機制（防止重複點擊）
        emailInput.disabled = true;
        submitBtn.disabled = true;
        submitBtn.style.opacity = '0.6';
        submitBtn.style.cursor = 'not-allowed';

        /* 💡 提示：未來可在這裡呼叫後端發信 API
           fetch('/api/forgot-password', { method: 'POST', body: ... })
        */

        // 3. 按鈕倒數計時功能 (例如 60 秒後才可以重新發送)
        let countdown = 60;
        submitBtn.textContent = `重新發送 (${countdown}s)`;

        const timer = setInterval(() => {
            countdown--;
            if (countdown > 0) {
                submitBtn.textContent = `重新發送 (${countdown}s)`;
            } else {
                // 倒數結束，恢復按鈕功能
                clearInterval(timer);
                emailInput.disabled = false;
                submitBtn.disabled = false;
                submitBtn.style.opacity = '1';
                submitBtn.style.cursor = 'pointer';
                submitBtn.textContent = '發送重設連結';
                infoDiv.classList.add('hidden'); // 隱藏舊的提示訊息
            }
        }, 1000);
    });
});