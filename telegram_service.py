import asyncio  # Diperbaiki dari 'Import' menjadi 'import' kecil
import httpx
from config import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID

async def send_telegram_alert_async(message):
    if not TELEGRAM_TOKEN or "ENTER_TOKEN" in TELEGRAM_TOKEN:
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        # Gunakan timeout total yang ketat agar tidak menggantung jika jaringan Render tidak stabil
        strict_timeout = httpx.Timeout(5.0, connect=2.0, read=5.0)
        async with httpx.AsyncClient(timeout=strict_timeout) as client:
            res = await client.post(url, json=payload)
            return res.status_code == 200
    except Exception as e:
        print(f"Failed to send Telegram alert: {e}")
        return False

def send_telegram_in_worker_thread(message):
    """
    Mengirim notifikasi Telegram secara non-blocking di latar belakang.
    Dioptimalkan menggunakan asyncio.create_task untuk menghemat RAM dan CPU di Render.
    """
    try:
        # Ambil event loop utama yang sedang berjalan
        loop = asyncio.get_running_loop()
        # Jalankan pengiriman pesan sebagai background task tanpa mengganggu/menunggu proses utama
        loop.create_task(send_telegram_alert_async(message))
    except RuntimeError:
        # Fallback jika fungsi ini dipanggil dari luar event loop asyncio utama
        import threading
        def worker():
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                new_loop.run_until_complete(send_telegram_alert_async(message))
            finally:
                new_loop.close()
        threading.Thread(target=worker, daemon=True).start()
