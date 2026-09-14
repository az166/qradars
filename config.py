import os
from dotenv import load_dotenv

# Set pencarian cepat O(1) untuk menyaring koin hitam (Blacklist)
COIN_BLACKLIST = {'UPUSDT', 'DOWNUSDT', 'BUSDUSDT', 'USDCUSDT', 'FDUSDUSDT', 'EURUSDT', 'USD1USDT', 'RLUSDUSDT', 'UUSDT'}

# Memuat variabel dari file .env
load_dotenv()

# Membaca token dengan aman menggunakan os.getenv
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "@cryptoradar_quantum")

# Pengaman tambahan jika file .env lupa belum terisi
if not TELEGRAM_TOKEN:
    raise ValueError("TELEGRAM_TOKEN belum terkonfigurasi di dalam file .env")


# Engine configuration
CACHE_TTL_SECONDS = 120
