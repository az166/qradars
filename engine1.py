import asyncio
from utils.indicators import (
    calculate_ma, calculate_std_dev, calculate_obv_trend,
    detect_bullish_divergence, calculate_macd_efficient, calculate_pearson_correlation
)
from services.binance_service import fetch_klines_safely_async, fetch_order_book_imbalance
from services.telegram_service import send_telegram_in_worker_thread

def detect_fair_value_gap(klines_1h):
    if len(klines_1h) < 3:
        return False, 0.0
    high_1 = float(klines_1h[-3][2])
    low_3 = float(klines_1h[-1][3])
    if low_3 > high_1:
        fvg_midpoint = (low_3 + high_1) / 2
        return True, fvg_midpoint
    return False, 0.0

def extract_market_structure_and_vol_ma(klines_1h):
    volumes = [float(k[7]) for k in klines_1h]
    if len(volumes) > 1:
        historical_vols = volumes[:-1]
        vol_ma20 = sum(historical_vols[-20:]) / min(len(historical_vols), 20)
    else:
        vol_ma20 = 1.0

    highs_48h = [float(k[2]) for k in klines_1h[-49:-1]]
    local_swing_high = max(highs_48h) if highs_48h else 0.0
    return vol_ma20, local_swing_high

def calculate_atr_and_spread(klines_1d, klines_1h):
    true_ranges = []
    for i in range(1, len(klines_1d)):
        high = float(klines_1d[i][2])
        low = float(klines_1d[i][3])
        prev_close = float(klines_1d[i-1][4])
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        true_ranges.append(tr)
    atr = sum(true_ranges) / len(true_ranges) if true_ranges else 0
    current_spread = float(klines_1h[-1][2]) - float(klines_1h[-1][3])
    recent_spreads = [float(k[2]) - float(k[3]) for k in klines_1h[-6:-1]]

    avg_recent_spread = sum(recent_spreads) / len(recent_spreads) if recent_spreads else 1
    if avg_recent_spread == 0: 
        avg_recent_spread = 0.00001
    return atr, current_spread / avg_recent_spread

def hitung_matriks_atr_dinamis(live_price, entry_price, atr, vol_spike_ratio, whale_dominance, is_btc_safe, highest_peak=0.0):
    base_tp_multiplier = 2.0
    base_cl_multiplier = 1.5

    if vol_spike_ratio > 3.0:
        base_tp_multiplier += 0.6  
    elif vol_spike_ratio > 1.8:
        base_tp_multiplier += 0.3

    if whale_dominance > 75.0:
        base_cl_multiplier -= 0.3  
    elif whale_dominance > 55.0:
        base_cl_multiplier -= 0.1

    if not is_btc_safe:
        base_tp_multiplier -= 0.6  
        base_cl_multiplier += 0.3  

    base_tp_multiplier = max(1.2, base_tp_multiplier)
    base_cl_multiplier = max(0.8, base_cl_multiplier)

    if entry_price > 0:
        tp_level = entry_price + (base_tp_multiplier * atr)
        cl_level = entry_price - (base_cl_multiplier * atr)

        if highest_peak > entry_price:
            chandelier_exit_floor = highest_peak - (1.3 * atr)
            if chandelier_exit_floor > cl_level:
                cl_level = chandelier_exit_floor
    else:
        tp_level = live_price + (base_tp_multiplier * atr)
        cl_level = live_price - (base_cl_multiplier * atr)

    return tp_level, cl_level

async def process_single_coin_pipeline(client, symbol, m_data, user_portfolio, semaphore, state_manager, device_id="default_guest_device"):
    async with semaphore:
        task_1w = fetch_klines_safely_async(client, symbol, '1w', 4)   
        task_1d = fetch_klines_safely_async(client, symbol, '1d', 105)  
        task_1h = fetch_klines_safely_async(client, symbol, '1h', 60)  
        task_15m = fetch_klines_safely_async(client, symbol, '15m', 10)
        task_depth = fetch_order_book_imbalance(client, symbol)

        klines_1w, klines_1d, klines_1h, klines_15m, order_book_ratio = await asyncio.gather(
            task_1w, task_1d, task_1h, task_15m, task_depth
        )
        if not klines_1w or not klines_1d or not klines_1h or not klines_15m or len(klines_1h) < 40 or len(klines_1d) < 99: 
            return None

        try:
            coin_name = symbol.replace("USDT", "")
            w1_close, w2_close = float(klines_1w[-2][4]), float(klines_1w[-3][4])
            is_macro_bullish = w1_close >= w2_close  

            # Ambil state global menggunakan state_manager pengaman thread
            live_price = state_manager.get_live_price(symbol, float(klines_1h[-1][4]))
            btc_returns_snapshot = state_manager.get_btc_returns()
            btc_safe_snapshot = state_manager.btc_status["is_safe"]
            btc_reason_snapshot = state_manager.btc_status["reason"]

            open_price = float(klines_1h[-1][1])
            price_pct_1h = ((live_price - open_price) / open_price) * 100
            atr, spread_ratio = calculate_atr_and_spread(klines_1d, klines_1h)

            vol_ma20, local_swing_high = extract_market_structure_and_vol_ma(klines_1h)
            daily_closes = [float(k[4]) for k in klines_1d]
            hourly_closes = [float(k[4]) for k in klines_1h]
            ma25_daily, ma99_daily = calculate_ma(daily_closes, 25), calculate_ma(daily_closes, 99)

            macd_line, signal_line, macd_hist, hist_list = await asyncio.to_thread(calculate_macd_efficient, hourly_closes)

            is_ma_trend_bullish = live_price > ma25_daily and live_price > ma99_daily
            is_macd_momentum_bullish = macd_hist > 0

            live_volume = float(klines_1h[-1][7])
            vol_spike_ratio = live_volume / vol_ma20 if vol_ma20 > 0 else 0
            volatility_based_threshold = max(0.2, min(1.5, (atr / live_price) * 100 * 0.15)) if live_price > 0 else 0.4

            v_15m_curr = float(klines_15m[-1][7])
            v_15m_ma = sum(float(k[7]) for k in klines_15m[-5:-1]) / 4
            is_15m_volume_burst = v_15m_curr > (v_15m_ma * 2.5)

            is_obv_healthy = calculate_obv_trend(klines_1h)

            coin_returns = []
            for i in range(-24, 0):
                try:
                    c_open = float(klines_1h[i][1])
                    c_close = float(klines_1h[i][4])
                    coin_returns.append((c_close - c_open) / c_open)
                except:
                    pass

            btc_correlation = await asyncio.to_thread(calculate_pearson_correlation, coin_returns, btc_returns_snapshot)
            is_uncorrelated_or_decoupled = btc_correlation < 0.20

            has_fvg, fvg_target_price = detect_fair_value_gap(klines_1h)

            prev_volume = float(klines_1h[-2][7]) if len(klines_1h) >= 2 else 1.0
            vol_velocity = (live_volume - prev_volume) / prev_volume if prev_volume > 0 else 0.0

            std_dev_20 = calculate_std_dev(hourly_closes, 20)
            bb_upper = calculate_ma(hourly_closes, 20) + (2.0 * std_dev_20)
            bb_lower = calculate_ma(hourly_closes, 20) - (2.0 * std_dev_20)
            kc_upper = calculate_ma(hourly_closes, 20) + (1.5 * atr)
            kc_lower = calculate_ma(hourly_closes, 20) - (1.5 * atr)
            is_squeeze = (bb_upper < kc_upper) and (bb_lower > kc_lower)

            is_bullish_div = False
            if len(hourly_closes) >= 10 and max(hourly_closes[-10:]) != min(hourly_closes[-10:]):
                is_bullish_div = detect_bullish_divergence(hourly_closes, hist_list, period=10)

            z_score = (live_price - ma25_daily) / std_dev_20 if std_dev_20 > 0 else 0.0

            market_liquidity_pool = m_data.get("pure_vol_24h", 0)
            if market_liquidity_pool >= 50000000: 
                required_vol_spike = 1.5
            elif market_liquidity_pool <= 5000000: 
                required_vol_spike = 3.5
            else: 
                required_vol_spike = 2.0

            current_hour_close = float(klines_1h[-1][4])
            hour_high, hour_low = float(klines_1h[-1][2]), float(klines_1h[-1][3])
            candle_body = abs(live_price - open_price)
            candle_total_range = max(0.0001, hour_high - hour_low)
            body_to_range_ratio = candle_body / candle_total_range

            is_mss_breakout = (current_hour_close > local_swing_high) and local_swing_high > 0
            is_confirmed_breakout = is_mss_breakout and (vol_spike_ratio >= required_vol_spike) and (body_to_range_ratio > 0.45)

            is_whale_churning = (vol_spike_ratio > required_vol_spike * 1.5) and (body_to_range_ratio < 0.20)

            base_whale = 30.0 + (vol_spike_ratio * 12.0)
            if is_whale_churning: 
                base_whale -= 25.0  
            whale_dominance = round(max(10.0, min(99.0, base_whale)), 1)

            fase = "CONSOLIDATION"
            momentum_score = (vol_spike_ratio * 35) + (whale_dominance * 0.5) + (price_pct_1h * 15)
            is_overextended = (live_price > ma25_daily + (2.5 * atr)) and atr > 0

            is_engine_locked = not btc_safe_snapshot and not is_uncorrelated_or_decoupled

            if is_engine_locked and coin_name not in user_portfolio:
                fase = f"ENGINE LOCKED ({btc_reason_snapshot})"
                momentum_score = 0
            else:
                if is_squeeze and (vol_velocity > 1.8 or is_15m_volume_burst) and price_pct_1h > volatility_based_threshold and is_obv_healthy:
                    fase = "⚡ SQUEEZE BREAKOUT (EARLY TREND)"
                    momentum_score += 140
                elif abs(price_pct_1h) < volatility_based_threshold and vol_velocity > 2.5 and abs(z_score) < 0.5 and is_obv_healthy:
                    fase = "🐳 WHALE ACCUMULATION (SILENT)"
                    momentum_score += 110
                elif is_bullish_div and price_pct_1h > volatility_based_threshold:
                    fase = "🔄 MOMENTUM REVERSAL (BOTTOMING)"
                    momentum_score += 100
                elif price_pct_1h > volatility_based_threshold and vol_spike_ratio >= required_vol_spike and not is_whale_churning:
                    if is_confirmed_breakout and is_macro_bullish and is_ma_trend_bullish and is_macd_momentum_bullish and (live_price > ma99_daily) and order_book_ratio > 1.2:
                        fase, momentum_score = "INSTITUTIONAL BUY", momentum_score + 150
                    elif is_confirmed_breakout and order_book_ratio > 1.0:
                        fase, momentum_score = "VALID BREAKOUT", momentum_score + 80
                    else:
                        fase = "EARLY RALLY"
                elif (price_pct_1h > 3.0 and vol_spike_ratio >= required_vol_spike * 2) or is_overextended: 
                    fase = "OVERBOUGHT PEAK"
                elif price_pct_1h < -1.5 and vol_spike_ratio >= required_vol_spike: 
                    fase, momentum_score = "EARLY DOWNTREND", momentum_score - 100
                elif is_whale_churning and price_pct_1h > 0:
                    fase = "WHALE CHURNING (HIGH RISK)"

            if "ENGINE LOCKED" in fase:
                status_rencana_otomatis = "WAIT & SEE"
            else:
                if fase in ["INSTITUTIONAL BUY", "VALID BREAKOUT", "EARLY RALLY", "⚡ SQUEEZE BREAKOUT (EARLY TREND)", "🐳 WHALE ACCUMULATION (SILENT)", "🔄 MOMENTUM REVERSAL (BOTTOMING)"]:
                    status_rencana_otomatis = "BUY STAGE"
                elif fase == "OVERBOUGHT PEAK":
                    status_rencana_otomatis = "TAKE PROFIT"
                else:
                    status_rencana_otomatis = "WAIT & SEE"

            harga_terformat = f"${live_price:.8f}" if live_price < 1.0 else f"${live_price:.4f}"

            # Verifikasi perubahan status untuk alert telegram
            if state_manager.is_alert_state_differs(coin_name, fase):
                if fase in ["VALID BREAKOUT", "INSTITUTIONAL BUY", "⚡ SQUEEZE BREAKOUT (EARLY TREND)", "🐳 WHALE ACCUMULATION (SILENT)"] and (btc_safe_snapshot or is_uncorrelated_or_decoupled) and vol_spike_ratio >= 2.0:
                    if fase == "INSTITUTIONAL BUY": 
                        emoji = "👑 BRAND NEW MSS BREAKOUT!"
                    elif fase == "⚡ SQUEEZE BREAKOUT (EARLY TREND)": 
                        emoji = "⚡ SQUEEZE BREAKOUT"
                    elif fase == "🐳 WHALE ACCUMULATION (SILENT)": 
                        emoji = "🐳 DECOUPLED WHALE ACCUMULATION"
                    else: 
                        emoji = "🔥 BREAKOUT SPIKE"

                    fvg_info = f"\n⚠️ Fair Value Gap Spotted: Yes (Retest Area: ${fvg_target_price:.4f})" if has_fvg else ""
                    decouple_info = f"\n🔄 BTC Correlation: {btc_correlation:.2f} (Decoupled From Core)" if is_uncorrelated_or_decoupled else f"\n🔄 BTC Correlation: {btc_correlation:.2f}"
                    depth_info = f"\n📊 Order Book Bid/Ask Ratio: {order_book_ratio:.2f}x"

                    send_telegram_in_worker_thread(
                        f"{emoji}\n\nCoin: *{coin_name}*\nMarket Phase: `{fase}`\n"
                        f"Vol vs MA20 Speed: `{vol_spike_ratio:.1f}x` (Velocity: {vol_velocity*100:.1f}%)\n"
                        f"Whale Dominance: `{whale_dominance}%` (Z-Score: {z_score:.2f}){decouple_info}{fvg_info}{depth_info}\n"
                        f"Live Price: *{harga_terformat}*"
                    )
                state_manager.update_alert_state(coin_name, fase)

            coin_p_data = user_portfolio.get(coin_name, {})
            entry_price = coin_p_data.get("costPrice", 0.0)
            amount = coin_p_data.get("amount", 0.0)

            current_peak = 0.0
            if entry_price > 0 and amount > 0:
                current_peak = state_manager.update_trailing_peak(device_id, coin_name, entry_price, live_price)

            dynamic_tp, dynamic_cl = hitung_matriks_atr_dinamis(
                live_price=live_price,
                entry_price=entry_price,
                atr=atr,
                vol_spike_ratio=vol_spike_ratio,
                whale_dominance=whale_dominance,
                is_btc_safe=btc_safe_snapshot,
                highest_peak=current_peak
            )

            max_allowed_atr = live_price * 0.15
            smoothed_atr = min(atr, max_allowed_atr) if atr > 0 else (live_price * 0.02)

            if status_rencana_otomatis == "BUY STAGE" and has_fvg:
                saran_entry = fvg_target_price
            elif status_rencana_otomatis == "BUY STAGE" and is_confirmed_breakout:
                saran_entry = local_swing_high + (0.15 * smoothed_atr)
            else:
                saran_entry = live_price - (0.5 * smoothed_atr)

            if entry_price > 0:
                if live_price >= dynamic_tp: 
                    status_rencana_otomatis = "TAKE PROFIT"
                elif live_price <= dynamic_cl: 
                    status_rencana_otomatis = "CUT LOSS"
                else: 
                    status_rencana_otomatis = "HOLDING"
                saran_entry = live_price - (0.75 * smoothed_atr)

            pnl_val, pnl_pct, current_value = 0.0, 0.0, 0.0
            if entry_price > 0 and amount > 0:
                current_value = amount * live_price
                initial_value = amount * entry_price
                pnl_val = current_value - initial_value
                pnl_pct = (pnl_val / initial_value) * 100

            return {
                "koin": coin_name, "harga": live_price, "persen_harga": price_pct_1h,
                "rasio": vol_spike_ratio, "fase": fase, "atr": atr, "whale": whale_dominance, "skor": momentum_score,
                "is_portfolio": coin_name in user_portfolio,
                "amount": amount, "entry": entry_price, "tp": dynamic_tp, "cl": dynamic_cl,
                "status_aksi": status_rencana_otomatis, "saran_entry": saran_entry,
                "pnl_val": pnl_val, "pnl_pct": pnl_pct, "current_value": current_value,
                "vol_velocity_pct": f"{round(vol_velocity * 100, 1)}%", "z_score": round(z_score, 2)
            }
        except Exception as e:
            print(f"Error processing {symbol}: {e}")
            return None
