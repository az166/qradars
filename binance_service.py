import asyncio
import httpx  
from config import COIN_BLACKLIST

async def check_bitcoin_circuit_breaker(client):
    try:
        url = "https://api2.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1h&limit=48"
        url_5m = "https://api2.binance.com/api/v3/klines?symbol=BTCUSDT&interval=5m&limit=6"

        # Menggunakan timeout bawaan dari instance client (tanpa timeout=None)
        res_1h, res_5m = await asyncio.gather(
            client.get(url), 
            client.get(url_5m)
        )

        if res_1h.status_code == 200 and res_5m.status_code == 200:
            klines = res_1h.json()
            klines_5m = res_5m.json()

            closes = [float(k[4]) for k in klines]
            live_btc = closes[-1]
            btc_ma24 = sum(closes[-24:]) / 24 if len(closes) >= 24 else live_btc
            
            btc_open_1h = float(klines[-1][1])
            btc_change_1h = ((live_btc - btc_open_1h) / btc_open_1h * 100) if btc_open_1h > 0 else 0.0

            btc_open_25m = float(klines_5m[0][1])
            btc_flash_change = ((live_btc - btc_open_25m) / btc_open_25m * 100) if btc_open_25m > 0 else 0.0

            local_returns = []
            for i in range(-24, 0):
                try:
                    c_open = float(klines[i][1])
                    c_close = float(klines[i][4])
                    if c_open > 0:
                        local_returns.append((c_close - c_open) / c_open)
                except Exception:
                    pass

            if btc_flash_change <= -1.0:
                status = {"is_safe": False, "reason": f"⚡ BTC FLASH DUMP ({btc_flash_change:.1f}%)"}
            elif btc_change_1h <= -1.5:
                status = {"is_safe": False, "reason": f"BTC DUMP ({btc_change_1h:.1f}%)"}
            elif live_btc < btc_ma24:
                status = {"is_safe": False, "reason": "BTC BEARISH (Below MA24)"}
            else:
                status = {"is_safe": True, "reason": "BTC SAFE"}

            return status, local_returns
        else:
            print(f"[BTC WARNING] Binance returned status: {res_1h.status_code} / {res_5m.status_code}")

    except Exception as e:
        print(f"Error checking BTC status: {e}")

    return {"is_safe": True, "reason": "BTC CHECK DELAYED"}, []


async def get_combined_tickers_data_async(client, portfolio=None, extra_symbols=None, **kwargs):
    """
    Mengambil data ticker 24 jam untuk:
    1. Top 50 koin berdasarkan volume 24h
    2. Koin yang ada di portofolio pengguna
    3. Koin kustom tambahan (extra_symbols)
    """
    url = "https://api.binance.com/api/v3/ticker/24hr"
    try:
        response = await client.get(url)
        if response.status_code == 200:
            all_tickers = response.json()
            ticker_dict = {}
            filtered_list = []
            local_live_prices = {}

            for t in all_tickers:
                symbol = t['symbol']
                if symbol.endswith('USDT') and (symbol not in COIN_BLACKLIST):
                    live_p = float(t['lastPrice'])
                    local_live_prices[symbol] = live_p

                    filtered_list.append({
                        "symbol": symbol,
                        "pure_vol_24h": float(t['quoteVolume']),
                        "price_change_pct_24h": float(t['priceChangePercent'])
                    })

            # Urutkan berdasarkan volume transaksi terbesar (Top 50)
            filtered_list.sort(key=lambda x: x['pure_vol_24h'], reverse=True)
            top_50_symbols = [item['symbol'] for item in filtered_list[:50]]

            # Ekstraksi simbol dari portofolio dengan pemeriksaan tipe data yang aman
            portfolio_symbols = []
            if portfolio and isinstance(portfolio, dict):
                for k, v in portfolio.items():
                    if not k:
                        continue
                    clean_k = str(k).strip().upper()
                    # Cek apakah v adalah sub-dict per-koin (format single device) atau sub-dict per-device
                    if isinstance(v, dict) and any(sub_key in v for sub_key in ["amount", "costPrice", "entry"]):
                        # Ini adalah format single device snapshot: { "BTC": {"amount": 1} }
                        sym = clean_k if clean_k.endswith("USDT") else f"{clean_k}USDT"
                        portfolio_symbols.append(sym)
                    elif isinstance(v, dict):
                        # Ini adalah format multi-device: { "dev_123": { "BTC": {...} } }
                        for coin in v.keys():
                            if coin:
                                clean_c = str(coin).strip().upper()
                                sym = clean_c if clean_c.endswith("USDT") else f"{clean_c}USDT"
                                portfolio_symbols.append(sym)

            # Ekstraksi simbol dari extra_symbols (koin kustom)
            custom_symbols = []
            if extra_symbols and isinstance(extra_symbols, list):
                for s in extra_symbols:
                    if s:
                        clean_s = str(s).strip().upper()
                        sym = clean_s if clean_s.endswith("USDT") else f"{clean_s}USDT"
                        custom_symbols.append(sym)

            # Gabungkan seluruh simbol target secara unik
            target_symbols = set(top_50_symbols + portfolio_symbols + custom_symbols)

            for item in filtered_list:
                if item['symbol'] in target_symbols:
                    ticker_dict[item['symbol']] = {
                        "pure_vol_24h": item['pure_vol_24h'],
                        "price_change_pct_24h": item['price_change_pct_24h']
                    }

            return ticker_dict, local_live_prices
    except Exception as e:
        print(f"Failed to update master ticker data: {e}")
    return {}, {}


async def fetch_klines_safely_async(client, symbol, interval, limit):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    try:
        response = await client.get(url)
        if response.status_code == 200: 
            return response.json()
    except Exception as e:
        print(f"[KLINES ERROR] Fetch {symbol} failed: {e}")
    return None


async def fetch_order_book_imbalance(client, symbol):
    url = f"https://api.binance.com/api/v3/depth?symbol={symbol}&limit=20"
    try:
        response = await client.get(url)
        if response.status_code == 200:
            depth = response.json()
            bids_vol = sum(float(b[1]) for b in depth.get('bids', []))
            asks_vol = sum(float(a[1]) for a in depth.get('asks', []))
            if asks_vol == 0: 
                return 2.0
            return bids_vol / asks_vol
    except Exception:
        pass
    return 1.0
