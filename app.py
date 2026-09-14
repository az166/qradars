import copy
import time
import asyncio
from threading import Thread, Lock
from flask import Flask, render_template, jsonify, request
import httpx
import numpy as np
import nest_asyncio

# Pengaman komponen async di dalam arsitektur multi-thread
nest_asyncio.apply()

from config import CACHE_TTL_SECONDS
from services.binance_service import (
    check_bitcoin_circuit_breaker, get_combined_tickers_data_async
)
from services.engine import process_single_coin_pipeline, hitung_matriks_atr_dinamis
from services.telegram_service import send_telegram_in_worker_thread

app = Flask(__name__)

class GlobalStateManager:
    """Mengelola data global terbagi menggunakan pengaman Thread Lock yang aman."""
    def __init__(self):
        self.lock = Lock()
        self.market_data_cache = []
        self.last_alerts_state = {}
        self.last_successful_scan_time = 0
        self.btc_returns = []
        self.live_price_map = {}
        self.btc_status = {"is_safe": True, "reason": "Connecting"}
        self.trailing_peaks = {}
        self.portfolio_dynamics = {}

    def get_live_price(self, symbol, default):
        with self.lock:
            return self.live_price_map.get(symbol, default)

    def get_btc_returns(self):
        with self.lock:
            return list(self.btc_returns)

    def is_alert_state_differs(self, coin_name, fase):
        with self.lock:
            return self.last_alerts_state.get(coin_name) != fase

    def update_alert_state(self, coin_name, fase):
        with self.lock:
            self.last_alerts_state[coin_name] = fase

    def update_trailing_peak(self, device_id, coin_name, entry_price, live_price):
        with self.lock:
            if device_id not in self.trailing_peaks:
                self.trailing_peaks[device_id] = {}
            old_peak = self.trailing_peaks[device_id].get(coin_name, entry_price)
            current_peak = max(old_peak, live_price)
            self.trailing_peaks[device_id][coin_name] = current_peak
            return current_peak

    def sync_portfolio_and_clean_peaks(self, device_id, portfolio_data):
        """Thread-safe update portofolio & pembersihan trailing peak yang tidak aktif."""
        with self.lock:
            self.portfolio_dynamics[device_id] = portfolio_data or {}
            if device_id in self.trailing_peaks:
                active_coins = set(self.portfolio_dynamics[device_id].keys())
                self.trailing_peaks[device_id] = {
                    k: v for k, v in self.trailing_peaks[device_id].items() if k in active_coins
                }

    def get_all_custom_coins(self):
        """Mengumpulkan koin kustom secara aman tanpa memblokir lock terlalu lama."""
        with self.lock:
            raw_portfolios = [dict(p) for p in self.portfolio_dynamics.values() if isinstance(p, dict)]

        custom_coins = set()
        for portfolio in raw_portfolios:
            for coin in portfolio.keys():
                if coin:
                    clean_coin = str(coin).strip().upper()
                    if clean_coin.endswith("USDT"):
                        clean_coin = clean_coin[:-4]
                    custom_coins.add(f"{clean_coin}USDT")
        return list(custom_coins)

state = GlobalStateManager()
ENGINE_INITIALIZED = False
STARTUP_LOCK = Lock()

async def execute_one_market_scan(target_device_id=None, minimal_bootstrap=False):
    async with httpx.AsyncClient(
        limits=httpx.Limits(max_connections=50, max_keepalive_connections=20),
        timeout=httpx.Timeout(12.0)
    ) as brass_client:
        try:
            semaphore = asyncio.Semaphore(10)  

            # 1. Update BTC Circuit Breaker
            btc_status, btc_returns = await check_bitcoin_circuit_breaker(brass_client)
            with state.lock:
                state.btc_status = btc_status
                state.btc_returns = btc_returns

            # 2. Ekstraksi portofolio gabungan & custom coins
            dev_key = target_device_id if target_device_id else "default_guest_device"
            with state.lock:
                portfolio_snapshot = copy.deepcopy(state.portfolio_dynamics.get(dev_key, {}))

            custom_coins_list = state.get_all_custom_coins()

            # 3. Ambil data koin bawaan + Custom coins
            ticker_master_data, prices_update = await get_combined_tickers_data_async(
                brass_client, 
                portfolio_snapshot, 
                extra_symbols=custom_coins_list
            )

            if not ticker_master_data:
                return

            with state.lock:
                state.live_price_map.update(prices_update)

            if minimal_bootstrap:
                ticker_master_data = dict(list(ticker_master_data.items())[:4])

            with state.lock:
                active_portfolio = dict(state.portfolio_dynamics.get(dev_key, {}))

            # 4. Paralelisasi eksekusi pipeline
            tasks = [
                process_single_coin_pipeline(brass_client, symbol, m_data, active_portfolio, semaphore, state, dev_key) 
                for symbol, m_data in ticker_master_data.items()
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            temp_data = [r for r in results if r is not None and not isinstance(r, Exception)]
            temp_data.sort(key=lambda x: (x.get('is_portfolio', False), x.get('skor') or 0), reverse=True)

            with state.lock:
                if minimal_bootstrap and state.market_data_cache:
                    existing_coins = {x.get('koin') for x in temp_data if 'koin' in x}
                    for old_item in state.market_data_cache:
                        if old_item.get('koin') not in existing_coins:
                            temp_data.append(old_item)

                state.market_data_cache = temp_data
                state.last_successful_scan_time = time.time()
        except Exception as e:
            app.logger.error(f"Error during core scan execution: {e}")

def run_loop_in_bg():
    local_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(local_loop)
    while True:
        try:
            local_loop.run_until_complete(execute_one_market_scan())
        except Exception as e:
            print(f"Background Loop Error: {e}")
        time.sleep(30)  

@app.before_request
def trigger_engine_startup():
    global ENGINE_INITIALIZED
    if not ENGINE_INITIALIZED:
        with STARTUP_LOCK:
            if not ENGINE_INITIALIZED:
                worker_thread = Thread(target=run_loop_in_bg, daemon=True)
                worker_thread.start()
                ENGINE_INITIALIZED = True

@app.route('/')
def index(): 
    return render_template('index.html')

@app.route('/api/data', methods=['POST'])
def get_data():
    req = request.json or {}
    device_id = req.get("device_id", "default_guest_device")

    try:
        # Penanganan portofolio & trailing peak yang thread-safe
        state.sync_portfolio_and_clean_peaks(device_id, req.get("portfolio", {}))
    except Exception as e:
        app.logger.error(f"Failed to synchronize device dynamic cache: {e}")

    try:
        with state.lock:
            active_portfolio = dict(state.portfolio_dynamics.get(device_id, {}))
            cache_snapshot = copy.deepcopy(state.market_data_cache)
            btc_safe_snapshot = state.btc_status.get("is_safe", True)
            btc_reason_snapshot = str(state.btc_status.get("reason", "")).upper()
            btc_returns_snapshot = list(state.btc_returns)

        # Sanitasi nilai return BTC dari NaN/None agar perhitungan aman
        valid_returns = [float(r) for r in btc_returns_snapshot if r is not None and not np.isnan(r)]
        avg_btc_return = float(np.mean(valid_returns)) if len(valid_returns) > 0 else 0.0

        if not btc_safe_snapshot and ("CRASH" in btc_reason_snapshot or "CAPITULATION" in btc_reason_snapshot or avg_btc_return < -0.04):
            btc_risk_level = 4
        elif not btc_safe_snapshot or "BREAKDOWN" in btc_reason_snapshot or avg_btc_return < -0.02:
            btc_risk_level = 3
        elif "SQUEEZE" in btc_reason_snapshot or "CONSOLIDATION" in btc_reason_snapshot or -0.01 <= avg_btc_return < 0.01:
            btc_risk_level = 2
        else:
            btc_risk_level = 1

        user_market_data = []
        normalized_active_portfolio = {
            str(k).strip().upper(): v for k, v in active_portfolio.items()
        }

        # Menggunakan item langsung dari cache_snapshot tanpa deepcopy berulang
        for item in cache_snapshot:
            coin_raw = str(item.get("koin", "")).strip().upper()
            coin_clean = coin_raw[:-4] if coin_raw.endswith("USDT") else coin_raw

            matched_key = None
            if coin_raw in normalized_active_portfolio:
                matched_key = coin_raw
            elif coin_clean in normalized_active_portfolio:
                matched_key = coin_clean

            live_price = float(item.get("harga", 0.0))
            rsi_val = float(item.get("rsi", 50.0))
            p_atas = float(item.get("proyeksi_atas", 0.0))

            if matched_key:
                item["is_portfolio"] = True
                coin_p_data = normalized_active_portfolio[matched_key]
                item["amount"] = float(coin_p_data.get("amount", 0.0))
                item["entry"] = float(coin_p_data.get("costPrice", 0.0))
                current_peak = 0.0
                if item["entry"] > 0 and item["amount"] > 0 and live_price > 0:
                    current_peak = state.update_trailing_peak(device_id, matched_key, item["entry"], live_price)

                if item["entry"] > 0 and item["amount"] > 0:
                    # --- PENYESUAIAN KOMPOUNDING & TRAILING TP ---
                    base_compounding_price = item["entry"]
                    if current_peak > item["entry"]:
                        base_compounding_price = current_peak

                    dtp, dcl = hitung_matriks_atr_dinamis(
                        live_price=live_price,
                        entry_price=base_compounding_price,  # Menggeser basis harga compounding
                        atr=item.get("atr", 0.0),
                        vol_spike_ratio=item.get("rasio", 1.0),
                        whale_dominance=item.get("whale", 0.0),
                        btc_risk_level=btc_risk_level,
                        rsi_saat_ini=rsi_val,
                        highest_peak=current_peak,
                        proyeksi_atas=p_atas
                    )

                    item["tp"] = dtp
                    item["cl"] = dcl
                    item["current_value"] = item["amount"] * live_price
                    initial_val = item["amount"] * item["entry"]
                    item["pnl_val"] = item["current_value"] - initial_val
                    item["pnl_pct"] = (item["pnl_val"] / initial_val) * 100.0 if initial_val > 0 else 0.0

                    if live_price >= item["tp"]: 
                        item["status_aksi"] = "COMPOUNDING / TP HIT"
                    elif live_price <= item["cl"]: 
                        item["status_aksi"] = "CUT LOSS"
                    else: 
                        item["status_aksi"] = "HOLDING (RIDING TREND)"

            else:
                item["is_portfolio"] = False
                item["amount"] = 0.0
                item["entry"] = 0.0

                dtp, dcl = hitung_matriks_atr_dinamis(
                    live_price=live_price,
                    entry_price=0.0,
                    atr=item.get("atr", 0.0),
                    vol_spike_ratio=item.get("rasio", 1.0),
                    whale_dominance=item.get("whale", 0.0),
                    btc_risk_level=btc_risk_level,
                    rsi_saat_ini=rsi_val,
                    highest_peak=0.0,
                    proyeksi_atas=p_atas
                )
                item["tp"] = dtp
                item["cl"] = dcl
                item["pnl_val"] = 0.0
                item["pnl_pct"] = 0.0
                item["current_value"] = 0.0

                if "ENGINE LOCKED" not in item.get("fase", ""):
                    if item.get("fase") in ["INSTITUTIONAL BUY", "VALID BREAKOUT", "EARLY RALLY", "⚡ SQUEEZE BREAKOUT (EARLY TREND)", "🐳 WHALE ACCUMULATION (SILENT)", "🔄 MOMENTUM REVERSAL (BOTTOMING)"]:
                        item["status_aksi"] = "BUY STAGE"
                    elif item.get("fase") == "OVERBOUGHT PEAK":
                        item["status_aksi"] = "TAKE PROFIT"
                    else:
                        item["status_aksi"] = "WAIT & SEE"

            user_market_data.append(item)

        user_market_data.sort(key=lambda x: (x.get('is_portfolio', False), x.get('skor') or 0), reverse=True)

        with state.lock:
            btc_status_response = dict(state.btc_status)
            btc_status_response["risk_level"] = btc_risk_level

        return jsonify({
            "btc_status": btc_status_response,
            "market": user_market_data
        }), 200

    except Exception as e:
        app.logger.error(f"Error executing quantitative data route: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/telegram/send_manual', methods=['POST'])
def send_manual_alert():
    try:
        req = request.json or {}

        coin = str(req.get("koin", "")).strip().upper()
        fase = req.get("fase", "MONITORING")
        harga = float(req.get("harga", 0))
        skor = req.get("skor", 0)
        status_aksi = req.get("status_aksi", "WAIT & SEE")
        z_score = req.get("z_score", 0.0)
        rasio = float(req.get("rasio", 0))
        whale = req.get("whale", 0)

        tren_pendek = req.get("tren_pendek", "SCANNING")
        probabilitas = req.get("probabilitas_prediksi", "50.0%")
        p_bawah = float(req.get("proyeksi_bawah", 0))
        p_atas = float(req.get("proyeksi_atas", 0))

        harga_fmt = f"${harga:.8f}" if harga < 1.0 else f"${harga:.4f}"
        fmt_atas = f"${p_atas:.8f}" if p_atas < 1.0 else f"${p_atas:.4f}"
        fmt_bawah = f"${p_bawah:.8f}" if p_bawah < 1.0 else f"${p_bawah:.4f}"

        msg = (
            f"📢 *MANUAL QUANTUM SIGNAL ALERT*\n\n"
            f"Coin: *{coin}*\n"
            f"Confidence Score: `{skor}/100` (`{status_aksi}`)\n"
            f"Vol Z-Score: `{z_score}` (Rasio: {rasio:.1f}x)\n"
            f"Whale Dominance: `{whale}%`\n"
            f"Market Phase: *{fase}*\n\n"
            f"🔮 *Trend Prediction*: `{tren_pendek}` ({probabilitas})\n"
            f"🎯 *Projected Range*: {fmt_bawah} - {fmt_atas}\n"
            f"Live Price: *{harga_fmt}*"
        )

        send_telegram_in_worker_thread(msg)
        return jsonify({"status": "success", "message": "Manual signal broadcast initiated!"}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
