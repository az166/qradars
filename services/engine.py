import asyncio
import json
import logging
import os
import time
from datetime import datetime
from math import exp
import numpy as np

from services.binance_service import fetch_klines_safely_async, fetch_order_book_imbalance
from services.telegram_service import send_telegram_in_worker_thread
from utils.indicators import (
    calculate_ma,
    calculate_macd_efficient,
    calculate_obv_trend,
    calculate_pearson_correlation,
    calculate_std_dev,
    calculate_technical_envelope_single_pass,
    detect_bullish_divergence,
)

# ==============================================================================
# DATA CACHE STORE
# ==============================================================================
_kline_cache = {}


async def fetch_klines_cached(client, symbol, interval, limit, ttl_seconds):
    key = f"{symbol}_{interval}_{limit}"
    now = time.time()

    # Pembersihan cache yang sudah kedaluwarsa secara berkala
    expired_keys = [k for k, (_, exp_t) in _kline_cache.items() if now > exp_t]
    for k in expired_keys:
        _kline_cache.pop(k, None)

    if key in _kline_cache:
        cached_data, expire_time = _kline_cache[key]
        if now < expire_time:
            return cached_data

    data = await fetch_klines_safely_async(client, symbol, interval, limit)
    if data:
        _kline_cache[key] = (data, now + ttl_seconds)
    return data


# ==============================================================================
# 1. PERFORMANCE LOGGER (Thread-safe File Atomic Write)
# ==============================================================================
class TradingPerformanceLogger:
    def __init__(self, log_filepath="logs/signal_performance.json"):
        self.log_filepath = log_filepath
        self.lock = asyncio.Lock()
        os.makedirs(os.path.dirname(self.log_filepath) or ".", exist_ok=True)
        if not os.path.exists(self.log_filepath):
            with open(self.log_filepath, "w") as f:
                json.dump([], f)

    def _read_data(self):
        if not os.path.exists(self.log_filepath):
            return []
        try:
            with open(self.log_filepath, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return []

    def _write_entry_sync(self, log_entry):
        try:
            data = self._read_data()
            has_open = any(
                e.get("symbol") == log_entry["symbol"] and e.get("status") == "OPEN"
                for e in data
            )
            if has_open:
                return None

            data.append(log_entry)
            temp_path = f"{self.log_filepath}.tmp"
            with open(temp_path, "w") as f:
                json.dump(data, f, indent=4)
            os.replace(temp_path, self.log_filepath)
            return log_entry["signal_id"]
        except Exception as e:
            logging.error(f"[LOGGER ERROR] Gagal mencatat entry signal: {e}")
            return None

    async def log_entry_signal_async(
        self,
        symbol,
        entry_price,
        score,
        action,
        z_score,
        btc_risk_status,
        tp_level,
        cl_level,
    ):
        async with self.lock:
            log_entry = {
                "signal_id": f"{symbol}_{int(time.time())}",
                "timestamp_entry": datetime.now().isoformat(),
                "symbol": symbol,
                "entry_price": entry_price,
                "confidence_score": score,
                "engine_action": action,
                "volume_z_score": z_score,
                "btc_risk": btc_risk_status,
                "target_tp": tp_level,
                "target_cl": cl_level,
                "status": "OPEN",
                "exit_price": None,
                "pnl_pct": 0.0,
                "timestamp_exit": None,
            }
            return await asyncio.to_thread(self._write_entry_sync, log_entry)

    def _write_close_sync(self, symbol, exit_price, exit_time):
        try:
            data = self._read_data()
            if not data:
                return

            updated = False
            for entry in data:
                if entry.get("symbol") == symbol and entry.get("status") == "OPEN":
                    entry["status"] = "CLOSED"
                    entry["exit_price"] = exit_price
                    entry["timestamp_exit"] = exit_time
                    pnl = (
                        ((exit_price - entry["entry_price"]) / entry["entry_price"]) * 100
                        if entry.get("entry_price", 0) > 0
                        else 0.0
                    )
                    entry["pnl_pct"] = round(pnl, 2)
                    updated = True
                    break

            if updated:
                temp_path = f"{self.log_filepath}.tmp"
                with open(temp_path, "w") as f:
                    json.dump(data, f, indent=4)
                os.replace(temp_path, self.log_filepath)
        except Exception as e:
            logging.error(f"[LOGGER ERROR] Gagal menutup posisi {symbol}: {e}")

    async def close_logged_signal_async(self, symbol, exit_price, current_time=None):
        async with self.lock:
            exit_time = current_time if current_time else datetime.now().isoformat()
            await asyncio.to_thread(self._write_close_sync, symbol, exit_price, exit_time)


perf_logger = TradingPerformanceLogger()


# ==============================================================================
# 2. QUANTITATIVE & PREDICTIVE FUNCTIONS
# ==============================================================================
def calculate_rsi_efficient(closes, period=14):
    if len(closes) < period + 1:
        return 50.0

    closes_arr = np.array(closes, dtype=float)
    deltas = np.diff(closes_arr)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(float(100.0 - (100.0 / (1.0 + rs))), 2)


def detect_bottoming_pattern(klines_1h, klines_15m, rsi_1h, is_bullish_div):
    if len(klines_1h) < 30 or len(klines_15m) < 4:
        return False, "NONE"

    closes_1h = np.array([float(k[4]) for k in klines_1h])
    lows_1h = np.array([float(k[3]) for k in klines_1h])
    vols_1h = np.array([float(k[7]) for k in klines_1h])

    closes_15m = np.array([float(k[4]) for k in klines_15m])
    lows_15m = np.array([float(k[3]) for k in klines_15m])

    is_oversold = rsi_1h <= 35.0

    sub_lows = lows_1h[-20:]
    part1 = sub_lows[:10]
    part2 = sub_lows[10:]

    if len(part1) == 0 or len(part2) == 0:
        return False, "NONE"

    val_1 = np.min(part1)
    val_2 = np.min(part2)

    is_double_bottom = (
        abs(val_1 - val_2) / max(1e-8, val_1) <= 0.008 and closes_1h[-1] > val_2
    )

    last_candle_open = float(klines_1h[-1][1])
    last_candle_close = float(klines_1h[-1][4])
    last_candle_high = float(klines_1h[-1][2])
    last_candle_low = float(klines_1h[-1][3])

    candle_range = max(0.0001, last_candle_high - last_candle_low)
    lower_wick = min(last_candle_open, last_candle_close) - last_candle_low
    wick_ratio = lower_wick / candle_range

    avg_vol = np.mean(vols_1h[-10:-1]) if len(vols_1h) >= 10 else 1.0
    is_vol_spike = vols_1h[-1] > (avg_vol * 1.8)
    is_absorption = wick_ratio >= 0.45 and is_vol_spike

    is_15m_higher_low = lows_15m[-1] > lows_15m[-2] and closes_15m[-1] > closes_15m[-2]

    if is_bullish_div and (is_double_bottom or is_absorption):
        return True, "CONFIRMED_BOTTOMING"
    elif is_oversold and is_absorption:
        return True, "VOLUME_ABSORPTION_BOTTOM"
    elif is_oversold and is_double_bottom and is_15m_higher_low:
        return True, "DOUBLE_BOTTOM_ACCUMULATION"
    elif is_bullish_div and is_15m_higher_low:
        return True, "EARLY_BULLISH_DIVERGENCE"

    return False, "NONE"


def prediksi_arah_tren(
    klines_1w,
    klines_1d,
    klines_1h,
    klines_15m,
    atr_sekarang,
    vol_spike_ratio,
    is_squeeze,
    is_confirmed_breakout,
    is_15m_volume_burst,
    btc_correlation,
    btc_risk_level,
    pure_vol_24h=20000000,
):
    if (
        not klines_1w
        or len(klines_1w) < 3
        or not klines_1d
        or len(klines_1d) < 99
        or not klines_1h
        or len(klines_1h) < 10
        or not klines_15m
        or len(klines_15m) < 4
    ):
        return "NEUTRAL", 50.0, 0.0, 0.0

    closes_1w = [float(k[4]) for k in klines_1w]
    closes_1d = [float(k[4]) for k in klines_1d]
    closes_1h = [float(k[4]) for k in klines_1h]
    volumes_1h = [float(k[7]) for k in klines_1h]
    closes_15m = [float(k[4]) for k in klines_15m]

    live_price = closes_1h[-1]
    relative_atr = (atr_sekarang / max(1e-8, live_price)) * 100

    is_weekly_bullish = closes_1w[-1] >= closes_1w[-2]

    ma25_daily = sum(closes_1d[-25:]) / 25
    ma99_daily = sum(closes_1d[-99:]) / 99
    is_daily_bullish = live_price > ma25_daily and live_price > ma99_daily

    momentum_1h_curr = closes_1h[-1] - closes_1h[-3]
    momentum_1h_prev = closes_1h[-3] - closes_1h[-6]
    akselerasi_1h = momentum_1h_curr - momentum_1h_prev
    momentum_15m_curr = closes_15m[-1] - closes_15m[-3]
    is_15m_micro_turning_up = momentum_15m_curr > 0 and (closes_15m[-1] > closes_15m[-2])

    slices_pendek = min(3, len(volumes_1h))
    slices_panjang = min(10, len(volumes_1h))

    v_short = volumes_1h[-slices_pendek:]
    if slices_pendek == 3:
        vol_ema_pendek = v_short[0] * 0.2 + v_short[1] * 0.3 + v_short[2] * 0.5
    else:
        vol_ema_pendek = sum(v_short) / slices_pendek

    v_long = volumes_1h[-slices_panjang:]
    w_long = [exp(x) for x in np.linspace(-1, 0, slices_panjang)]
    w_long_sum = sum(w_long)
    vol_ema_panjang = (
        sum(v * w for v, w in zip(v_long, w_long)) / w_long_sum
        if w_long_sum > 0
        else 1.0
    )

    volume_velocity = vol_ema_pendek / max(0.000001, vol_ema_panjang)

    prediksi_tren = "SIDEWAYS / REGRESSION"
    probabilitas_sukses = 50.0

    if akselerasi_1h > 0 and vol_spike_ratio > 1.3:
        if momentum_1h_curr > 0:
            prediksi_tren = "PREDICTIVE BULLISH CONTINUATION"
            probabilitas_sukses = 65.0 + (volume_velocity * 4)
        else:
            prediksi_tren = "POTENTIAL REVERSAL UP (BOTTOMING)"
            probabilitas_sukses = 58.0
    elif akselerasi_1h < 0 or (momentum_1h_curr < 0 and volume_velocity > 1.2):
        if momentum_1h_curr < 0 and is_15m_micro_turning_up and is_15m_volume_burst:
            prediksi_tren = "POTENTIAL REVERSAL UP (EARLY 15M ACCELERATION)"
            probabilitas_sukses = 60.0 + (vol_spike_ratio * 2)
        elif momentum_1h_curr < 0:
            prediksi_tren = "PREDICTIVE BEARISH CONTINUATION"
            probabilitas_sukses = 68.0 + (volume_velocity * 3)
        else:
            prediksi_tren = "POTENTIAL TOPPING / BULL TRAP"
            probabilitas_sukses = 62.0

    if "BULLISH" in prediksi_tren or "UP" in prediksi_tren:
        if is_weekly_bullish and is_daily_bullish:
            probabilitas_sukses += 12.0
        if not is_weekly_bullish:
            probabilitas_sukses -= 15.0
        if not is_daily_bullish:
            probabilitas_sukses -= 10.0
    else:
        if not is_weekly_bullish and not is_daily_bullish:
            probabilitas_sukses += 10.0
        if is_weekly_bullish:
            probabilitas_sukses -= 12.0

    if "BULLISH" in prediksi_tren or "UP" in prediksi_tren:
        if btc_correlation > 0.70 and btc_risk_level >= 3:
            probabilitas_sukses -= 18.0
        elif btc_correlation < 0.20:
            probabilitas_sukses += 5.0
    else:
        if btc_correlation > 0.70 and btc_risk_level >= 3:
            probabilitas_sukses += 12.0

    if relative_atr > 8.0:
        probabilitas_sukses -= 10.0

    probabilitas_sukses = max(10.0, min(95.0, probabilitas_sukses))

    mult_atas_bullish, mult_bawah_bullish = 1.5, 0.75
    mult_atas_bearish, mult_bawah_bearish = 0.75, 1.5

    if is_squeeze:
        mult_atas_bullish, mult_bawah_bullish = 1.1, 0.5
        mult_atas_bearish, mult_bawah_bearish = 0.5, 1.1
    elif is_confirmed_breakout:
        vol_cap_ratio = pure_vol_24h / 100000000
        boost_factor = min(1.5, max(1.0, vol_cap_ratio))
        mult_atas_bullish, mult_bawah_bullish = 2.5 * boost_factor, 1.2
        mult_atas_bearish, mult_bawah_bearish = 0.8, 1.8 * boost_factor

    if "BULLISH" in prediksi_tren or "UP" in prediksi_tren:
        proyeksi_atas = live_price + (atr_sekarang * mult_atas_bullish)
        proyeksi_bawah = max(0.0, live_price - (atr_sekarang * mult_bawah_bullish))
    else:
        proyeksi_atas = live_price + (atr_sekarang * mult_atas_bearish)
        proyeksi_bawah = max(0.0, live_price - (atr_sekarang * mult_bawah_bearish))

    return prediksi_tren, round(probabilitas_sukses, 1), proyeksi_atas, proyeksi_bawah


def detect_fair_value_gap(klines_1h):
    if len(klines_1h) < 3:
        return False, 0.0

    high_1 = float(klines_1h[-3][2])
    low_3 = float(klines_1h[-1][3])

    if low_3 > high_1:
        return True, (low_3 + high_1) / 2

    return False, 0.0


def calculate_volume_metrics(klines_1h, window=20):
    if len(klines_1h) < window + 1:
        return 0.0, 50.0, 1.0

    historical_volumes = [float(k[7]) for k in klines_1h[-(window + 1) : -1]]
    current_volume = float(klines_1h[-1][7])
    n_hist = len(historical_volumes)

    if n_hist == 0:
        return 0.0, 50.0, 1.0

    mean_vol = sum(historical_volumes) / n_hist
    variance = sum((x - mean_vol) ** 2 for x in historical_volumes) / n_hist

    std_vol = max(0.0, variance) ** 0.5

    z_score = (current_volume - mean_vol) / std_vol if std_vol > 0.00001 else 0.0
    z_score = max(-3.0, min(6.0, z_score))

    all_vols = historical_volumes + [current_volume]
    percentile = (
        sum(1 for v in all_vols if v <= current_volume) / len(all_vols)
    ) * 100
    vol_spike_ratio = current_volume / mean_vol if mean_vol > 0 else 1.0

    return round(z_score, 2), round(percentile, 1), round(vol_spike_ratio, 2)


def analyze_market_structure(klines_1h, window=5):
    n = len(klines_1h)
    if n < 30:
        return "CONSOLIDATION", 0.0, 0.0

    highs = np.array([float(k[2]) for k in klines_1h])
    lows = np.array([float(k[3]) for k in klines_1h])
    closes = np.array([float(k[4]) for k in klines_1h])

    is_swing_high = np.ones(n, dtype=bool)
    is_swing_low = np.ones(n, dtype=bool)

    for offset in range(-window, window + 1):
        if offset == 0:
            continue
        is_swing_high &= highs >= np.roll(highs, -offset)
        is_swing_low &= lows <= np.roll(lows, -offset)

    valid_range = (np.arange(n) >= window) & (np.arange(n) < n - window)
    swing_high_indices = np.where(is_swing_high & valid_range)[0]
    swing_low_indices = np.where(is_swing_low & valid_range)[0]

    if len(swing_high_indices) < 2 or len(swing_low_indices) < 2:
        return "CONSOLIDATION", float(np.max(highs[-10:])), float(np.min(lows[-10:]))

    last_sh = float(highs[swing_high_indices[-1]])
    prev_sh = float(highs[swing_high_indices[-2]])
    last_sl = float(lows[swing_low_indices[-1]])
    prev_sl = float(lows[swing_low_indices[-2]])

    if last_sh > prev_sh and last_sl > prev_sl:
        structure = "BULLISH_STRUCTURE"
    elif last_sh < prev_sh and last_sl < prev_sl:
        structure = "BEARISH_STRUCTURE"
    else:
        structure = "CONSOLIDATION"

    if structure == "BEARISH_STRUCTURE" and closes[-1] > last_sh:
        structure = "MSS_BULLISH_BREAKOUT"

    return structure, last_sh, last_sl


def verify_breakout_status(
    live_price, last_close, open_price, high_price, low_price, local_swing_high, z_score
):
    if local_swing_high == 0:
        return "NO_BREAKOUT"

    candle_body = abs(live_price - open_price)
    candle_range = max(live_price * 0.0001, high_price - low_price)
    body_to_range_ratio = candle_body / candle_range

    if live_price > local_swing_high:
        if (
            last_close > local_swing_high
            and body_to_range_ratio >= 0.45
            and z_score >= 1.5
        ):
            return "CONFIRMED_BREAKOUT"
        return "PENDING_BREAKOUT"

    return "NO_BREAKOUT"


def calculate_atr_and_spread(klines_1d, klines_1h):
    if not klines_1d or len(klines_1d) < 2 or not klines_1h:
        return 0.0, 1.0

    true_ranges = []
    for i in range(1, len(klines_1d)):
        high = float(klines_1d[i][2])
        low = float(klines_1d[i][3])
        prev_close = float(klines_1d[i - 1][4])
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        true_ranges.append(tr)

    atr = sum(true_ranges) / len(true_ranges) if true_ranges else 0.0
    current_spread = float(klines_1h[-1][2]) - float(klines_1h[-1][3])
    recent_spreads = [float(k[2]) - float(k[3]) for k in klines_1h[-6:-1]]

    avg_recent_spread = (
        sum(recent_spreads) / len(recent_spreads) if recent_spreads else 1.0
    )
    if avg_recent_spread <= 0:
        avg_recent_spread = 0.00001
    return atr, current_spread / avg_recent_spread


def calculate_btc_risk_level(btc_status_dict, btc_returns_24h):
    if not isinstance(btc_status_dict, dict):
        btc_status_dict = {}

    is_safe = btc_status_dict.get("is_safe", True)
    reason = str(btc_status_dict.get("reason", "")).upper()
    avg_return = (
        sum(btc_returns_24h) / len(btc_returns_24h) if btc_returns_24h else 0.0
    )

    if not is_safe and (
        "CRASH" in reason or "CAPITULATION" in reason or avg_return < -0.04
    ):
        return {
            "level": 4,
            "status": "EXTREME RISK",
            "allocation_pct": 0.0,
            "allowed_trade": False,
        }
    if not is_safe or "BREAKDOWN" in reason or avg_return < -0.02:
        return {
            "level": 3,
            "status": "HIGH RISK",
            "allocation_pct": 25.0,
            "allowed_trade": True,
        }
    if (
        "SQUEEZE" in reason
        or "CONSOLIDATION" in reason
        or -0.01 <= avg_return < 0.01
    ):
        return {
            "level": 2,
            "status": "MEDIUM RISK",
            "allocation_pct": 50.0,
            "allowed_trade": True,
        }
    return {
        "level": 1,
        "status": "LOW RISK",
        "allocation_pct": 100.0,
        "allowed_trade": True,
    }


def calculate_confidence_score(
    market_struct,
    breakout_status,
    z_score,
    percentile,
    ob_ratio,
    ma_bullish,
    btc_risk,
    is_bottoming=False,
    bottoming_type="NONE",
):
    if btc_risk["level"] == 4:
        return 0, "WAIT & SEE"

    score = 0
    if ma_bullish:
        score += 30

    if z_score >= 2.0 and percentile >= 85:
        score += 25
    elif z_score >= 1.0 and percentile >= 70:
        score += 15
    elif z_score >= 0.0:
        score += 5

    if ob_ratio >= 1.5:
        score += 20
    elif ob_ratio >= 1.1:
        score += 10

    if (
        market_struct == "MSS_BULLISH_BREAKOUT"
        or breakout_status == "CONFIRMED_BREAKOUT"
    ):
        score += 15
    elif breakout_status == "PENDING_BREAKOUT":
        score += 7

    if is_bottoming:
        if bottoming_type == "CONFIRMED_BOTTOMING":
            score += 25
        elif bottoming_type in [
            "VOLUME_ABSORPTION_BOTTOM",
            "DOUBLE_BOTTOM_ACCUMULATION",
        ]:
            score += 18
        elif bottoming_type == "EARLY_BULLISH_DIVERGENCE":
            score += 12

    if btc_risk["level"] == 1:
        score += 10
    elif btc_risk["level"] == 2:
        score += 5
    elif btc_risk["level"] == 3:
        score -= 15

    score = max(0, min(100, score))
    action = (
        "STRONG BUY"
        if score >= 75
        else (
            "SPECULATIVE BUY / EARLY TREND"
            if score >= 55
            else "HOLDING / MONITOR" if score >= 40 else "WAIT & SEE"
        )
    )
    return score, action


def hitung_matriks_atr_dinamis(
    live_price: float,
    entry_price: float,
    atr: float,
    vol_spike_ratio: float,
    whale_dominance: float,
    btc_risk_level: float,
    rsi_saat_ini: float = 50.0,
    highest_peak: float = 0.0,
    proyeksi_atas: float = 0.0,
) -> tuple[float, float]:
    try:
        base_entry = entry_price if entry_price > 0 else live_price

        if atr <= 0 or live_price <= 0:
            return round(base_entry * 1.03, 6), round(base_entry * 0.98, 6)

        is_strong_bullish = (rsi_saat_ini >= 65.0) and (vol_spike_ratio >= 2.0)
        is_bearish_momentum = (rsi_saat_ini <= 40.0) or (vol_spike_ratio < 0.7)
        risk_level = int(btc_risk_level)

        if risk_level >= 3:
            cl_multiplier = 0.8
        elif is_strong_bullish:
            cl_multiplier = 1.2
        elif is_bearish_momentum:
            cl_multiplier = 1.0
        else:
            cl_multiplier = 1.5

        calculated_cl = base_entry - (atr * cl_multiplier)

        if highest_peak > base_entry:
            trailing_stop = highest_peak - (atr * 1.2)
            calculated_cl = max(calculated_cl, trailing_stop)

        calculated_cl = min(calculated_cl, live_price * 0.995)

        if is_strong_bullish:
            tp_multiplier = 3.5
        elif is_bearish_momentum or risk_level >= 3:
            tp_multiplier = 1.5
        else:
            tp_multiplier = 2.5

        calculated_tp = base_entry + (atr * tp_multiplier)

        if proyeksi_atas > base_entry:
            realistic_target = proyeksi_atas * 0.98
            if realistic_target > base_entry:
                calculated_tp = min(calculated_tp, realistic_target)

        risk = base_entry - calculated_cl
        if risk > 0:
            min_tp = base_entry + (risk * 1.2)
            calculated_tp = max(calculated_tp, min_tp)

        return round(calculated_tp, 6), round(calculated_cl, 6)

    except Exception as e:
        logging.error(f"[ENGINE RECOVERY] Fallback ATR terpicu akibat: {e}")
        base_p = entry_price if entry_price > 0 else live_price
        return round(base_p * 1.05, 6), round(base_p * 0.97, 6)


# ==============================================================================
# 3. MAIN ASYNC PIPELINE ENGINE
# ==============================================================================
async def process_single_coin_pipeline(
    client,
    symbol,
    m_data,
    user_portfolio,
    semaphore,
    state_manager,
    device_id="default_guest_device",
):
    async with semaphore:
        (
            klines_1w,
            klines_1d,
            klines_1h,
            klines_15m,
            order_book_ratio,
        ) = await asyncio.gather(
            fetch_klines_cached(client, symbol, "1w", 5, ttl_seconds=7200),
            fetch_klines_cached(client, symbol, "1d", 105, ttl_seconds=3600),
            fetch_klines_cached(client, symbol, "1h", 60, ttl_seconds=300),
            fetch_klines_cached(client, symbol, "15m", 10, ttl_seconds=60),
            fetch_order_book_imbalance(client, symbol),
        )

        if (
            not klines_1w
            or not klines_1d
            or not klines_1h
            or not klines_15m
            or len(klines_1h) < 40
            or len(klines_1d) < 99
        ):
            return None

        try:
            coin_name = symbol.replace("USDT", "")
            w1_close, w2_close = float(klines_1w[-1][4]), float(klines_1w[-2][4])

            live_price = (
                state_manager.get_live_price(symbol, float(klines_1h[-1][4]))
                if hasattr(state_manager, "get_live_price")
                else float(klines_1h[-1][4])
            )
            btc_returns_snapshot = (
                state_manager.get_btc_returns()
                if hasattr(state_manager, "get_btc_returns")
                else []
            )

            open_price = float(klines_1h[-1][1])
            high_price = float(klines_1h[-1][2])
            low_price = float(klines_1h[-1][3])
            last_close_1h = float(klines_1h[-1][4])

            price_pct_1h = (
                (((live_price - open_price) / open_price) * 100)
                if open_price > 0
                else 0.0
            )
            atr, spread_ratio = calculate_atr_and_spread(klines_1d, klines_1h)

            daily_closes = [float(k[4]) for k in klines_1d]
            hourly_closes = [float(k[4]) for k in klines_1h]
            ma25_daily, ma99_daily = calculate_ma(daily_closes, 25), calculate_ma(
                daily_closes, 99
            )

            (
                macd_line,
                signal_line,
                macd_hist,
                hist_list,
            ) = await asyncio.to_thread(calculate_macd_efficient, hourly_closes)
            is_ma_trend_bullish = live_price > ma25_daily and live_price > ma99_daily

            rsi_1h = calculate_rsi_efficient(hourly_closes, period=14)

            vol_z_score, vol_percentile, vol_spike_ratio = calculate_volume_metrics(
                klines_1h, window=20
            )
            volatility_based_threshold = max(
                0.2, min(1.5, (atr / max(1e-8, live_price)) * 100 * 0.15)
            )

            v_15m_curr = float(klines_15m[-1][7])
            v_15m_ma = (
                sum(float(k[7]) for k in klines_15m[-5:-1]) / 4
                if len(klines_15m) >= 5
                else 1.0
            )
            is_15m_volume_burst = (
                v_15m_curr > (v_15m_ma * 2.5) if v_15m_ma > 0 else False
            )

            coin_returns = []
            recent_klines = klines_1h[-25:]
            for kline in recent_klines[:-1]:
                c_open = float(kline[1])
                c_close = float(kline[4])
                if c_open > 0:
                    coin_returns.append((c_close - c_open) / max(1e-8, c_open))

            btc_correlation = await asyncio.to_thread(
                calculate_pearson_correlation, coin_returns, btc_returns_snapshot
            )
            is_uncorrelated_or_decoupled = btc_correlation < 0.20

            has_fvg, fvg_target_price = detect_fair_value_gap(klines_1h)
            prev_volume = float(klines_1h[-2][7]) if len(klines_1h) >= 2 else 1.0
            vol_velocity = (float(klines_1h[-1][7]) - prev_volume) / max(
                0.000001, prev_volume
            )

            (
                ma20_hourly,
                bb_upper,
                bb_lower,
                kc_upper,
                kc_lower,
                is_squeeze,
            ) = calculate_technical_envelope_single_pass(
                prices=hourly_closes,
                atr=atr,
                period=20,
                num_std_dev=2.0,
                num_atr_mult=1.5,
            )

            is_bullish_div = (
                detect_bullish_divergence(hourly_closes, hist_list, period=10)
                if len(hourly_closes) >= 10
                and max(hourly_closes[-10:]) != min(hourly_closes[-10:])
                else False
            )

            is_bottoming, bottoming_type = detect_bottoming_pattern(
                klines_1h=klines_1h,
                klines_15m=klines_15m,
                rsi_1h=rsi_1h,
                is_bullish_div=is_bullish_div,
            )

            market_struct, last_sh, last_sl = analyze_market_structure(klines_1h, 5)

            breakout_status = verify_breakout_status(
                live_price=live_price,
                last_close=last_close_1h,
                open_price=open_price,
                high_price=high_price,
                low_price=low_price,
                local_swing_high=last_sh,
                z_score=vol_z_score,
            )

            market_liquidity_pool = m_data.get("pure_vol_24h", 0) if m_data else 0
            required_vol_spike = (
                1.5
                if market_liquidity_pool >= 50000000
                else (3.5 if market_liquidity_pool <= 5000000 else 2.0)
            )

            candle_range_denom = max(0.0001, high_price - low_price)
            body_to_range_ratio = abs(live_price - open_price) / candle_range_denom
            is_confirmed_breakout = breakout_status == "CONFIRMED_BREAKOUT"

            is_whale_churning = (
                vol_spike_ratio > required_vol_spike * 1.5
            ) and (body_to_range_ratio < 0.20)
            base_whale = 30.0 + (vol_spike_ratio * 12.0)
            if is_whale_churning:
                base_whale -= 25.0
            whale_dominance = round(max(10.0, min(99.0, base_whale)), 1)

            btc_risk = calculate_btc_risk_level(
                getattr(state_manager, "btc_status", {}), btc_returns_snapshot
            )

            (
                prediksi_tren,
                probabilitas_prediksi,
                proyeksi_atas,
                proyeksi_bawah,
            ) = await asyncio.to_thread(
                prediksi_arah_tren,
                klines_1w,
                klines_1d,
                klines_1h,
                klines_15m,
                atr,
                vol_spike_ratio,
                is_squeeze,
                is_confirmed_breakout,
                is_15m_volume_burst,
                btc_correlation,
                btc_risk["level"],
                market_liquidity_pool,
            )

            if w1_close >= w2_close and live_price > ma25_daily:
                tren_panjang_skenario = "MACRO BULLISH STRUCTURE"
            elif w1_close < w2_close and live_price < ma25_daily:
                tren_panjang_skenario = "MACRO BEARISH STRUCTURE"
            else:
                tren_panjang_skenario = "MACRO SIDEWAYS DYNAMICS"

            momentum_score, status_rencana_otomatis = calculate_confidence_score(
                market_struct=market_struct,
                breakout_status=breakout_status,
                z_score=vol_z_score,
                percentile=vol_percentile,
                ob_ratio=order_book_ratio,
                ma_bullish=is_ma_trend_bullish,
                btc_risk=btc_risk,
                is_bottoming=is_bottoming,
                bottoming_type=bottoming_type,
            )

            fase = "CONSOLIDATION"
            if btc_risk["level"] == 4 and coin_name not in user_portfolio:
                btc_reason = getattr(state_manager, "btc_status", {}).get(
                    "reason", "CRASH"
                )
                fase = f"ENGINE LOCKED ({btc_reason})"
            else:
                if status_rencana_otomatis == "STRONG BUY":
                    fase = (
                        "INSTITUTIONAL BUY"
                        if is_ma_trend_bullish and order_book_ratio > 1.2
                        else "VALID BREAKOUT"
                    )
                elif is_bottoming and bottoming_type in [
                    "CONFIRMED_BOTTOMING",
                    "VOLUME_ABSORPTION_BOTTOM",
                ]:
                    fase = f"🛡️ BOTTOMING CONFIRMED ({bottoming_type})"
                elif (
                    is_squeeze
                    and (vol_velocity > 1.8 or is_15m_volume_burst)
                    and price_pct_1h > volatility_based_threshold
                ):
                    fase = "⚡ SQUEEZE BREAKOUT (EARLY TREND)"
                elif (
                    abs(price_pct_1h) < volatility_based_threshold
                    and vol_velocity > 2.5
                    and vol_z_score < 0.5
                ):
                    fase = "🐳 WHALE ACCUMULATION (SILENT)"
                elif is_bullish_div and price_pct_1h > volatility_based_threshold:
                    fase = "🔄 MOMENTUM REVERSAL (BOTTOMING)"

            if hasattr(
                state_manager, "is_alert_state_differs"
            ) and state_manager.is_alert_state_differs(coin_name, fase):
                if (
                    status_rencana_otomatis == "STRONG BUY"
                    or fase
                    in [
                        "VALID BREAKOUT",
                        "INSTITUTIONAL BUY",
                        "⚡ SQUEEZE BREAKOUT (EARLY TREND)",
                    ]
                    or "BOTTOMING" in fase
                ):
                    if btc_risk["allowed_trade"] or is_uncorrelated_or_decoupled:
                        emoji = (
                            "🛡️ BOTTOMING DETECTED!"
                            if "BOTTOMING" in fase
                            else (
                                "👑 BRAND NEW MSS BREAKOUT!"
                                if fase == "INSTITUTIONAL BUY"
                                else "🔥 BREAKOUT SPIKE"
                            )
                        )
                        fvg_info = (
                            f"\n⚠️ Fair Value Gap Spotted: Yes (Retest Area: ${fvg_target_price:.4f})"
                            if has_fvg
                            else ""
                        )
                        decouple_info = (
                            f"\n🔄 BTC Correlation: {btc_correlation:.2f} (Decoupled)"
                            if is_uncorrelated_or_decoupled
                            else f"\n🔄 BTC Correlation: {btc_correlation:.2f}"
                        )
                        bottom_info = (
                            f"\n🎯 Bottom Type: `{bottoming_type}`"
                            if is_bottoming
                            else ""
                        )

                        harga_terformat = (
                            f"${live_price:.8f}"
                            if live_price < 1.0
                            else f"${live_price:.4f}"
                        )
                        fmt_atas = (
                            f"${proyeksi_atas:.8f}"
                            if proyeksi_atas < 1.0
                            else f"${proyeksi_atas:.4f}"
                        )
                        fmt_bawah = (
                            f"${proyeksi_bawah:.8f}"
                            if proyeksi_bawah < 1.0
                            else f"${proyeksi_bawah:.4f}"
                        )

                        send_telegram_in_worker_thread(
                            f"{emoji}\n\nCoin: *{coin_name}*\nConfidence Score: `{momentum_score}/100` (`{status_rencana_otomatis}`)\n"
                            f"Vol Z-Score: `{vol_z_score}` (Pct: {vol_percentile}%)\n"
                            f"Structure: `{market_struct}` | Breakout: `{breakout_status}`\n"
                            f"Whale Dominance: `{whale_dominance}%`{bottom_info}{decouple_info}{fvg_info}\n"
                            f"🔮 *Trend Prediction*: `{prediksi_tren}` ({probabilitas_prediksi}%)\n"
                            f"🎯 *Projected Range*: {fmt_bawah} - {fmt_atas}\n"
                            f"Live Price: *{harga_terformat}*"
                        )
                if hasattr(state_manager, "update_alert_state"):
                    state_manager.update_alert_state(coin_name, fase)

            coin_p_data = user_portfolio.get(coin_name, {})
            entry_price = float(coin_p_data.get("costPrice") or 0.0)
            amount = float(coin_p_data.get("amount") or 0.0)

            current_peak = 0.0
            if (
                entry_price > 0
                and amount > 0
                and hasattr(state_manager, "update_trailing_peak")
            ):
                current_peak = state_manager.update_trailing_peak(
                    device_id, coin_name, entry_price, live_price
                )

            dynamic_tp, dynamic_cl = hitung_matriks_atr_dinamis(
                live_price=live_price,
                entry_price=entry_price,
                atr=atr,
                vol_spike_ratio=vol_spike_ratio,
                whale_dominance=whale_dominance,
                btc_risk_level=btc_risk["level"],
                rsi_saat_ini=rsi_1h,
                highest_peak=current_peak,
                proyeksi_atas=proyeksi_atas,
            )

            if status_rencana_otomatis == "STRONG BUY" and entry_price == 0:
                await perf_logger.log_entry_signal_async(
                    symbol=symbol,
                    entry_price=live_price,
                    score=momentum_score,
                    action=status_rencana_otomatis,
                    z_score=vol_z_score,
                    btc_risk_status=btc_risk["status"],
                    tp_level=dynamic_tp,
                    cl_level=dynamic_cl,
                )

            max_allowed_atr = live_price * 0.15
            smoothed_atr = max(
                0.000001,
                min(atr, max_allowed_atr) if atr > 0 else (live_price * 0.02),
            )

            if status_rencana_otomatis == "STRONG BUY" and has_fvg:
                saran_entry = fvg_target_price
            elif status_rencana_otomatis == "STRONG BUY" and is_confirmed_breakout:
                saran_entry = last_sh + (0.15 * smoothed_atr)
            else:
                saran_entry = max(0.0, live_price - (0.5 * smoothed_atr))

            if entry_price > 0:
                if live_price >= dynamic_tp:
                    status_rencana_otomatis = "TAKE PROFIT"
                    await perf_logger.close_logged_signal_async(symbol, live_price)
                elif live_price <= dynamic_cl:
                    status_rencana_otomatis = "CUT LOSS"
                    await perf_logger.close_logged_signal_async(symbol, live_price)
                else:
                    status_rencana_otomatis = "HOLDING"
                saran_entry = max(0.0, live_price - (0.75 * smoothed_atr))

            pnl_val, pnl_pct, current_value = 0.0, 0.0, 0.0
            if entry_price > 0 and amount > 0:
                current_value = amount * live_price
                pnl_val = current_value - (amount * entry_price)
                pnl_pct = (pnl_val / (amount * entry_price)) * 100

            return {
                "koin": coin_name,
                "harga": live_price,
                "persen_harga": price_pct_1h,
                "rasio": vol_spike_ratio,
                "fase": fase,
                "atr": atr,
                "whale": whale_dominance,
                "skor": momentum_score,
                "is_portfolio": coin_name in user_portfolio,
                "amount": amount,
                "entry": entry_price,
                "tp": dynamic_tp,
                "cl": dynamic_cl,
                "status_aksi": status_rencana_otomatis,
                "saran_entry": round(saran_entry, 8 if live_price < 1.0 else 4),
                "pnl_val": pnl_val,
                "pnl_pct": pnl_pct,
                "current_value": current_value,
                "vol_velocity_pct": f"{round(vol_velocity * 100, 1)}%",
                "z_score": round(vol_z_score, 2),
                "rsi": rsi_1h,
                "is_bottoming": is_bottoming,
                "bottoming_type": bottoming_type,
                "tren_pendek": prediksi_tren.upper(),
                "tren_panjang": tren_panjang_skenario.upper(),
                "probabilitas_prediksi": f"{probabilitas_prediksi}%",
                "proyeksi_atas": (
                    round(proyeksi_atas, 8)
                    if live_price < 1.0
                    else round(proyeksi_atas, 4)
                ),
                "proyeksi_bawah": (
                    round(proyeksi_bawah, 8)
                    if live_price < 1.0
                    else round(proyeksi_bawah, 4)
                ),
            }
        except Exception as e:
            logging.error(f"[PIPELINE ERROR] Gagal memproses {symbol}: {e}")
            return None
