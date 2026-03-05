# =============================================================================
# Gold Weekly Bias Dashboard — ICT Analysis Engine
# =============================================================================
"""
Applies Inner Circle Trader (ICT) concepts algorithmically to gold price data
across monthly, weekly, and daily timeframes. Generates 3 daily trade setups:
  1. Primary Trend Trade  — with monthly + weekly market structure
  2. OTE Retracement      — Fibonacci 0.618–0.705 optimal entry
  3. Liquidity Hunt       — counter-trend sweep of key liquidity levels

All data is pure pandas computation on OHLCV DataFrames from yfinance.
No new API keys or packages required.

Disclaimer: Educational ICT analysis only. Not financial advice.
"""
import numpy as np
import pandas as pd

import config


# ---------------------------------------------------------------------------
# Swing Point Detection
# ---------------------------------------------------------------------------

def find_swing_points(df: pd.DataFrame, order: int = config.ICT_SWING_ORDER) -> dict:
    """
    Detect confirmed swing highs and swing lows using a fractal approach.

    A swing high at bar i means df['High'][i] is the highest among the
    `order` bars to its left AND `order` bars to its right.
    The last `order` bars cannot be confirmed (right side incomplete).

    Returns:
        {"highs": pd.Series (index=date, value=high price at swing bars only),
         "lows":  pd.Series (index=date, value=low price at swing bars only)}
    """
    if df.empty or len(df) < 2 * order + 1:
        empty = pd.Series(dtype=float)
        return {"highs": empty, "lows": empty}

    highs: dict = {}
    lows: dict  = {}

    high_arr = df["High"].values
    low_arr  = df["Low"].values
    idx      = df.index

    for i in range(order, len(df) - order):
        window_h = high_arr[i - order : i + order + 1]
        window_l = low_arr[i - order : i + order + 1]
        if high_arr[i] == np.max(window_h):
            highs[idx[i]] = float(high_arr[i])
        if low_arr[i] == np.min(window_l):
            lows[idx[i]] = float(low_arr[i])

    return {
        "highs": pd.Series(highs, dtype=float),
        "lows":  pd.Series(lows,  dtype=float),
    }


# ---------------------------------------------------------------------------
# Market Structure
# ---------------------------------------------------------------------------

def detect_market_structure(df: pd.DataFrame, lookback: int = 10) -> str:
    """
    Determine HTF market structure from swing points.

    Bullish  = Higher High + Higher Low (last confirmed swing > prior)
    Bearish  = Lower High  + Lower Low
    Ranging  = mixed signals or insufficient swing history

    Returns: 'bullish' | 'bearish' | 'ranging'
    """
    swings = find_swing_points(df)
    sh = swings["highs"].dropna()
    sl = swings["lows"].dropna()

    if len(sh) < 2 or len(sl) < 2:
        return "ranging"

    sh_tail = sh.tail(lookback)
    sl_tail = sl.tail(lookback)

    hh = float(sh_tail.iloc[-1]) > float(sh_tail.iloc[-2])   # Higher High
    hl = float(sl_tail.iloc[-1]) > float(sl_tail.iloc[-2])   # Higher Low
    lh = float(sh_tail.iloc[-1]) < float(sh_tail.iloc[-2])   # Lower High
    ll = float(sl_tail.iloc[-1]) < float(sl_tail.iloc[-2])   # Lower Low

    if hh and hl:
        return "bullish"
    if lh and ll:
        return "bearish"
    return "ranging"


# ---------------------------------------------------------------------------
# Fibonacci Levels
# ---------------------------------------------------------------------------

def calc_fibonacci_levels(swing_high: float, swing_low: float) -> dict:
    """
    Compute Fibonacci retracement levels between swing_high and swing_low.

    Convention (bullish move from low to high):
        price_at_fib = swing_high - level * (swing_high - swing_low)
        fib 0.0 = swing_high (top of range)
        fib 1.0 = swing_low  (bottom of range)

    Returns dict with float fib levels as keys plus metadata keys.
    Returns {} if swing_high <= swing_low.
    """
    if swing_high <= swing_low:
        return {}

    rng = swing_high - swing_low
    result: dict = {
        level: round(swing_high - level * rng, 2)
        for level in config.ICT_FIB_LEVELS
    }
    result["swing_high"] = round(swing_high, 2)
    result["swing_low"]  = round(swing_low,  2)
    result["range"]      = round(rng, 2)
    return result


# ---------------------------------------------------------------------------
# Fair Value Gaps (FVG)
# ---------------------------------------------------------------------------

def _fvg_filled(df: pd.DataFrame, gap_bar_idx: int,
                top: float, bottom: float, direction: str) -> bool:
    """Check if a FVG has been (at least half) filled by subsequent price action."""
    subsequent = df.iloc[gap_bar_idx + 1 :]
    if subsequent.empty:
        return False
    midpoint = (top + bottom) / 2.0
    if direction == "bullish":
        return bool((subsequent["Low"] <= midpoint).any())
    else:
        return bool((subsequent["High"] >= midpoint).any())


def find_fvgs(df: pd.DataFrame,
              n_recent: int = config.ICT_FVG_LOOKBACK) -> list[dict]:
    """
    Scan for Fair Value Gaps (3-candle imbalance pattern).

    Bullish FVG: candle[n-2].High < candle[n].Low
        → unfilled gap zone between those two prices (blue zone)
    Bearish FVG: candle[n-2].Low > candle[n].High
        → unfilled gap zone between those two prices (orange zone)

    Returns list of dicts (most recent first):
        {direction, top, bottom, midpoint, date, filled}
    """
    if df.empty or len(df) < 3:
        return []

    fvgs: list[dict] = []
    scan_start = max(0, len(df) - n_recent - 2)

    for i in range(scan_start + 2, len(df)):
        c0_high = float(df["High"].iloc[i - 2])
        c0_low  = float(df["Low"].iloc[i - 2])
        c2_high = float(df["High"].iloc[i])
        c2_low  = float(df["Low"].iloc[i])
        date    = df.index[i]

        # Bullish FVG
        if c0_high < c2_low:
            top    = c2_low
            bottom = c0_high
            if top > bottom:
                filled = _fvg_filled(df, i, top, bottom, "bullish")
                fvgs.append({
                    "direction": "bullish",
                    "top":       round(top,    2),
                    "bottom":    round(bottom, 2),
                    "midpoint":  round((top + bottom) / 2, 2),
                    "date":      date,
                    "filled":    filled,
                })

        # Bearish FVG
        elif c0_low > c2_high:
            top    = c0_low
            bottom = c2_high
            if top > bottom:
                filled = _fvg_filled(df, i, top, bottom, "bearish")
                fvgs.append({
                    "direction": "bearish",
                    "top":       round(top,    2),
                    "bottom":    round(bottom, 2),
                    "midpoint":  round((top + bottom) / 2, 2),
                    "date":      date,
                    "filled":    filled,
                })

    return list(reversed(fvgs))   # most recent first


# ---------------------------------------------------------------------------
# Order Blocks (OB)
# ---------------------------------------------------------------------------

def find_order_blocks(df: pd.DataFrame,
                      n_recent: int = config.ICT_OB_LOOKBACK,
                      min_impulse_pct: float = config.ICT_OB_MIN_IMPULSE
                      ) -> list[dict]:
    """
    Identify institutional Order Blocks.

    Bullish OB: the last bearish (down) candle immediately before a
                significant bullish impulse. Zone = that candle's High/Low.
                Valid as long as no subsequent candle's Low traded below OB Low.

    Bearish OB: the last bullish (up) candle immediately before a
                significant bearish impulse. Zone = that candle's High/Low.
                Valid as long as no subsequent candle's High traded above OB High.

    Returns list of dicts (most recent first):
        {direction, high, low, date, valid, impulse_pct}
    """
    if df.empty or len(df) < 3:
        return []

    obs: list[dict] = []
    scan_start = max(0, len(df) - n_recent - 1)

    for i in range(scan_start + 1, len(df) - 1):
        prev_o = float(df["Open"].iloc[i - 1])
        prev_c = float(df["Close"].iloc[i - 1])
        prev_h = float(df["High"].iloc[i - 1])
        prev_l = float(df["Low"].iloc[i - 1])
        nxt_o  = float(df["Open"].iloc[i + 1])
        nxt_c  = float(df["Close"].iloc[i + 1])
        date   = df.index[i - 1]   # OB date = the OB candle itself

        # Bullish OB candidate: previous candle was bearish
        if prev_c < prev_o and nxt_o != 0:
            impulse_pct = (nxt_c - nxt_o) / abs(nxt_o) * 100
            if impulse_pct >= min_impulse_pct:
                subsequent = df.iloc[i + 1 :]
                valid = not bool((subsequent["Low"] < prev_l).any())
                obs.append({
                    "direction":   "bullish",
                    "high":        round(prev_h, 2),
                    "low":         round(prev_l, 2),
                    "date":        date,
                    "valid":       valid,
                    "impulse_pct": round(impulse_pct, 2),
                })

        # Bearish OB candidate: previous candle was bullish
        elif prev_c > prev_o and nxt_o != 0:
            impulse_pct = (nxt_o - nxt_c) / abs(nxt_o) * 100
            if impulse_pct >= min_impulse_pct:
                subsequent = df.iloc[i + 1 :]
                valid = not bool((subsequent["High"] > prev_h).any())
                obs.append({
                    "direction":   "bearish",
                    "high":        round(prev_h, 2),
                    "low":         round(prev_l, 2),
                    "date":        date,
                    "valid":       valid,
                    "impulse_pct": round(impulse_pct, 2),
                })

    return list(reversed(obs))   # most recent first


# ---------------------------------------------------------------------------
# Key Levels
# ---------------------------------------------------------------------------

def get_key_levels(monthly_df: pd.DataFrame, weekly_df: pd.DataFrame) -> dict:
    """
    Extract key price levels from most recent completed and in-progress bars.

    Uses iloc[-2] for the last completed bar and iloc[-1] for the current
    in-progress bar (which may not be fully formed yet).

    Returns dict with keys:
        PMH, PML  = Previous Month High / Low
        CMH, CML  = Current Month High / Low (in progress)
        PWH, PWL  = Previous Week High / Low
        CWH, CWL  = Current Week High / Low (in progress)
    """
    def _safe_val(df: pd.DataFrame, row_idx: int, col: str) -> float | None:
        try:
            if len(df) > abs(row_idx):
                return float(df[col].iloc[row_idx])
        except Exception:
            pass
        return None

    return {
        "PMH": _safe_val(monthly_df, -2, "High"),
        "PML": _safe_val(monthly_df, -2, "Low"),
        "CMH": _safe_val(monthly_df, -1, "High"),
        "CML": _safe_val(monthly_df, -1, "Low"),
        "PWH": _safe_val(weekly_df,  -2, "High"),
        "PWL": _safe_val(weekly_df,  -2, "Low"),
        "CWH": _safe_val(weekly_df,  -1, "High"),
        "CWL": _safe_val(weekly_df,  -1, "Low"),
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_major_swing(df: pd.DataFrame, lookback_bars: int = 60
                      ) -> tuple[float, float, str]:
    """
    Find the most significant swing high and swing low in the last
    `lookback_bars` rows of df (full OHLCV required).

    Returns (swing_high, swing_low, direction) where direction is:
        'up'   if the swing high occurred MORE recently than the swing low
               (price has been rising — currently in retracement)
        'down' if the swing low occurred MORE recently than the swing high
               (price has been falling — currently in a bounce)
    """
    window = df.tail(lookback_bars)
    if window.empty:
        return 0.0, 0.0, "ranging"

    high_idx = window["High"].idxmax()
    low_idx  = window["Low"].idxmin()
    sh = float(window["High"].max())
    sl = float(window["Low"].min())

    direction = "up" if high_idx > low_idx else "down"
    return sh, sl, direction


def _calc_rr(entry: float | None, stop: float | None,
             target: float | None) -> float | None:
    """Risk/Reward ratio. Returns None if any input is missing or risk is zero."""
    if entry is None or stop is None or target is None:
        return None
    risk   = abs(entry - stop)
    reward = abs(target - entry)
    if risk == 0:
        return None
    return round(reward / risk, 2)


def _build_rationale(direction: str,
                     m_struct: str,
                     w_struct: str,
                     current_price: float | None,
                     fifty_pct: float | None,
                     ob_used: dict | None,
                     fvg_used: dict | None) -> str:
    """Assemble a human-readable 2–3 sentence ICT rationale."""
    parts = []

    # Market structure
    parts.append(
        f"Monthly structure: {m_struct.upper()}. "
        f"Weekly structure: {w_struct.upper()}."
    )

    # Premium / discount zone
    if current_price is not None and fifty_pct is not None:
        zone = "discount" if current_price < fifty_pct else "premium"
        parts.append(
            f"Gold at ${current_price:,.1f} is in a {zone} zone "
            f"(50% fib = ${fifty_pct:,.1f})."
        )

    # Order Block
    if ob_used:
        d = ob_used["direction"].title()
        parts.append(
            f"{d} Order Block at ${ob_used['low']:,.1f}–${ob_used['high']:,.1f} "
            f"providing {'support' if ob_used['direction'] == 'bullish' else 'resistance'}."
        )

    # Fair Value Gap
    if fvg_used:
        d = fvg_used["direction"].title()
        parts.append(
            f"Unfilled {d} FVG at ${fvg_used['bottom']:,.1f}–${fvg_used['top']:,.1f} "
            f"acting as a {'target' if direction == 'LONG' else 'magnet'}."
        )

    return " ".join(parts)


def _wait_trade(trade_id: int, reason: str = "Insufficient data") -> dict:
    """Return a WAIT trade dict for edge cases."""
    return {
        "id":              trade_id,
        "direction":       "WAIT",
        "setup_name":      "No Setup",
        "entry":           None,
        "stop":            None,
        "target1":         None,
        "target2":         None,
        "rr1":             None,
        "rr2":             None,
        "confidence":      "LOW",
        "rationale":       reason,
        "key_levels_used": [],
        "timeframe":       "—",
    }


# ---------------------------------------------------------------------------
# Master Trade Generator
# ---------------------------------------------------------------------------

def generate_ict_trades(monthly_df: pd.DataFrame,
                        weekly_df:  pd.DataFrame,
                        daily_df:   pd.DataFrame,
                        bias_score: float) -> list[dict]:
    """
    Generate 3 daily ICT trade setups for gold.

    Trade 1 — Primary Trend   : with monthly + weekly bias (Order Block entry)
    Trade 2 — OTE Retracement : Fibonacci 0.618–0.705 optimal entry
    Trade 3 — Liquidity Hunt  : counter-trend sweep of PWH/PWL

    Args:
        monthly_df: Monthly OHLCV DataFrame (10 years)
        weekly_df:  Weekly OHLCV DataFrame  (5 years)
        daily_df:   Daily OHLCV DataFrame   (90 days)
        bias_score: Aggregate bias score from scoring.py (-1 to +1)

    Returns:
        List of exactly 3 trade dicts.
    """
    # ── Guard: need sufficient data ──────────────────────────────────────
    MIN_BARS = 10
    if (monthly_df.empty or weekly_df.empty or daily_df.empty
            or len(monthly_df) < MIN_BARS
            or len(weekly_df)  < MIN_BARS
            or len(daily_df)   < 3):
        reason = "Insufficient historical data to compute ICT setups."
        return [_wait_trade(i, reason) for i in range(1, 4)]

    # ── Compute building blocks ───────────────────────────────────────────
    try:
        monthly_structure = detect_market_structure(monthly_df)
        weekly_structure  = detect_market_structure(weekly_df)
        key_levels        = get_key_levels(monthly_df, weekly_df)
        current_price     = float(daily_df["Close"].iloc[-1])

        # Use DAILY swing for fibonacci so OTE zone stays near current price.
        # 30-day window captures the most recent local swing rather than the
        # macro multi-year move which would put OTE far from current price.
        sh, sl, swing_dir = _find_major_swing(daily_df, lookback_bars=30)
        # Fallback: if daily range is too tight (<1%), widen to 60 daily bars
        if sh - sl < current_price * 0.01:
            sh, sl, swing_dir = _find_major_swing(daily_df, lookback_bars=60)

        fib       = calc_fibonacci_levels(sh, sl)
        fifty_pct = fib.get(0.5)

        weekly_obs  = [o for o in find_order_blocks(weekly_df) if o["valid"]]
        daily_obs   = [o for o in find_order_blocks(daily_df)  if o["valid"]]
        weekly_fvgs = [f for f in find_fvgs(weekly_df) if not f["filled"]]
        daily_fvgs  = [f for f in find_fvgs(daily_df)  if not f["filled"]]
    except Exception as exc:
        reason = f"ICT computation error: {exc}"
        return [_wait_trade(i, reason) for i in range(1, 4)]

    # ── Overall bias ──────────────────────────────────────────────────────
    if (monthly_structure == "bullish"
            and weekly_structure == "bullish"
            and bias_score > 0.10):
        overall_bias = "bullish"
    elif (monthly_structure == "bearish"
            and weekly_structure == "bearish"
            and bias_score < -0.10):
        overall_bias = "bearish"
    else:
        overall_bias = "ranging"

    # ─────────────────────────────────────────────────────────────────────
    # TRADE 1 — Primary Trend Trade
    # ─────────────────────────────────────────────────────────────────────
    def _trade1() -> dict:
        base = {
            "id":              1,
            "timeframe":       "Monthly + Weekly",
            "key_levels_used": [],
        }

        if overall_bias == "bullish":
            # Find nearest valid bullish OB within 8% of current price.
            # Sort by proximity (closest high to current price first).
            candidates = sorted(
                [o for o in (daily_obs + weekly_obs)
                 if o["direction"] == "bullish"
                 and o["high"] <= current_price * 1.03   # at or just above
                 and o["high"] >= current_price * 0.92], # within 8% below
                key=lambda o: abs(o["high"] - current_price),
            )
            ob_used = candidates[0] if candidates else None

            if ob_used:
                ob_range = ob_used["high"] - ob_used["low"]
                entry  = round((ob_used["high"] + ob_used["low"]) / 2, 1)
                stop   = round(ob_used["low"] - max(ob_range * 0.5, current_price * 0.003), 1)
                conf   = "HIGH" if abs(bias_score) > 0.3 else "MEDIUM"
                levels = [f"Bullish OB ${ob_used['low']:.1f}–${ob_used['high']:.1f}"]
            else:
                # Fallback: use PWL as support; entry at current price
                pwl = key_levels.get("PWL") or key_levels.get("CWL")
                entry  = round(current_price, 1)
                stop   = round(pwl * 0.998, 1) if pwl else round(current_price * 0.985, 1)
                conf   = "MEDIUM"
                levels = [f"No nearby OB — stop below PWL ${stop:.1f}"]

            tp1_candidates = [v for v in [key_levels.get("PWH"), key_levels.get("CMH")] if v
                              and v > current_price]
            tp1 = round(min(tp1_candidates), 1) if tp1_candidates else None
            # TP2: use PMH if it's above current price, else recent swing high
            tp2_candidates = [v for v in [key_levels.get("PMH"), sh] if v and v > current_price]
            tp2 = round(max(tp2_candidates), 1) if tp2_candidates else None

            if tp1: levels.append(f"TP1 = PWH/CMH ${tp1:.1f}")
            if tp2: levels.append(f"TP2 = PMH ${tp2:.1f}")

            # Pick a nearby FVG to mention in rationale
            fvg_target = next((f for f in (weekly_fvgs + daily_fvgs)
                               if f["direction"] == "bullish" and f["bottom"] > entry), None)

            return {
                **base,
                "direction":       "LONG",
                "setup_name":      "Weekly OB — Primary Trend Long",
                "entry":           entry,
                "stop":            stop,
                "target1":         tp1,
                "target2":         tp2,
                "rr1":             _calc_rr(entry, stop, tp1),
                "rr2":             _calc_rr(entry, stop, tp2),
                "confidence":      conf,
                "rationale":       _build_rationale(
                    "LONG", monthly_structure, weekly_structure,
                    current_price, fifty_pct, ob_used, fvg_target),
                "key_levels_used": levels,
            }

        elif overall_bias == "bearish":
            candidates = sorted(
                [o for o in (daily_obs + weekly_obs)
                 if o["direction"] == "bearish"
                 and o["low"] >= current_price * 0.97   # at or just below
                 and o["low"] <= current_price * 1.08], # within 8% above
                key=lambda o: abs(o["low"] - current_price),
            )
            ob_used = candidates[0] if candidates else None

            if ob_used:
                ob_range = ob_used["high"] - ob_used["low"]
                entry  = round((ob_used["high"] + ob_used["low"]) / 2, 1)
                stop   = round(ob_used["high"] + max(ob_range * 0.5, current_price * 0.003), 1)
                conf   = "HIGH" if abs(bias_score) > 0.3 else "MEDIUM"
                levels = [f"Bearish OB ${ob_used['low']:.1f}–${ob_used['high']:.1f}"]
            else:
                pwh = key_levels.get("PWH") or key_levels.get("CWH")
                entry  = round(current_price, 1)
                stop   = round(pwh * 1.002, 1) if pwh else round(current_price * 1.015, 1)
                conf   = "MEDIUM"
                levels = [f"No nearby OB — stop above PWH ${stop:.1f}"]

            tp1_cands = [v for v in [key_levels.get("PWL"), key_levels.get("CML")] if v
                         and v < current_price]
            tp1 = round(max(tp1_cands), 1) if tp1_cands else None
            tp2_cands = [v for v in [key_levels.get("PML"), sl] if v and v < current_price]
            tp2 = round(min(tp2_cands), 1) if tp2_cands else None

            if tp1: levels.append(f"TP1 = PWL/CML ${tp1:.1f}")
            if tp2: levels.append(f"TP2 = PML ${tp2:.1f}")

            fvg_target = next((f for f in (weekly_fvgs + daily_fvgs)
                               if f["direction"] == "bearish" and f["top"] < entry), None)

            return {
                **base,
                "direction":       "SHORT",
                "setup_name":      "Weekly OB — Primary Trend Short",
                "entry":           entry,
                "stop":            stop,
                "target1":         tp1,
                "target2":         tp2,
                "rr1":             _calc_rr(entry, stop, tp1),
                "rr2":             _calc_rr(entry, stop, tp2),
                "confidence":      conf,
                "rationale":       _build_rationale(
                    "SHORT", monthly_structure, weekly_structure,
                    current_price, fifty_pct, ob_used, fvg_target),
                "key_levels_used": levels,
            }

        else:
            return _wait_trade(1, (
                f"Ranging market — Monthly: {monthly_structure.upper()}, "
                f"Weekly: {weekly_structure.upper()}. "
                "No directional primary trade. Wait for structure to resolve."
            ))

    # ─────────────────────────────────────────────────────────────────────
    # TRADE 2 — OTE Retracement Trade
    # ─────────────────────────────────────────────────────────────────────
    def _trade2() -> dict:
        base = {
            "id":        2,
            "timeframe": "Weekly Fibonacci",
        }

        if not fib:
            return _wait_trade(2, "Fibonacci not computable — swing_high equals swing_low.")

        ote_low  = fib.get(config.ICT_OTE_LOW)   # 0.618
        ote_high = fib.get(config.ICT_OTE_HIGH)  # 0.705

        if ote_low is None or ote_high is None:
            return _wait_trade(2, "OTE zone levels unavailable.")

        if overall_bias == "bullish":
            # In bullish trend: OTE zone is a DISCOUNT retracement area to buy
            entry  = round((ote_low + ote_high) / 2, 1)
            stop   = round(fib.get(0.786, ote_high * 0.995) - 5.0, 1)
            tp1    = round(sh, 1)   # back to swing high

            # TP2: first unfilled bullish FVG above entry, or swing high + buffer
            above_fvg = next((f for f in (weekly_fvgs + daily_fvgs)
                              if f["direction"] == "bullish" and f["bottom"] > entry), None)
            tp2 = round(above_fvg["top"], 1) if above_fvg else round(sh * 1.005, 1)

            # Confidence: based on how close price is to OTE zone
            if ote_low <= current_price <= ote_high:
                conf = "HIGH"
                note = "Price currently IN OTE zone."
            elif abs(current_price - ote_high) / ote_high < 0.02:
                conf = "MEDIUM"
                note = "Price approaching OTE zone."
            elif current_price > ote_low:
                # Price has already passed through OTE zone downward — missed
                return _wait_trade(2, (
                    f"Price ${current_price:,.1f} is above OTE zone "
                    f"(${ote_high:,.1f}–${ote_low:,.1f}). "
                    "OTE entry opportunity may have passed — wait for next retracement."
                ))
            else:
                conf = "LOW"
                note = "Price not yet near OTE zone."

            levels = [
                f"OTE Zone ${ote_high:.1f}–${ote_low:.1f} (Fib 0.618–0.705)",
                f"Stop below Fib 0.786 = ${fib.get(0.786, 0):.1f}",
                f"TP1 = Swing High ${tp1:.1f}",
            ]
            if above_fvg:
                levels.append(f"TP2 = FVG ${above_fvg['bottom']:.1f}–${above_fvg['top']:.1f}")

            rationale = (
                f"Monthly: {monthly_structure.upper()}, Weekly: {weekly_structure.upper()}. "
                f"OTE zone at ${ote_high:.1f}–${ote_low:.1f} (Fib 0.618–0.705) "
                f"is in discount (below 50% at ${fifty_pct:.1f}) — ideal for trend entries. "
                f"{note}"
            ) if fifty_pct else (
                f"Monthly: {monthly_structure.upper()}, Weekly: {weekly_structure.upper()}. "
                f"OTE retracement zone at ${ote_high:.1f}–${ote_low:.1f}. {note}"
            )

            return {
                **base,
                "direction":       "LONG",
                "setup_name":      "OTE Retracement — Fib 0.618–0.705",
                "entry":           entry,
                "stop":            stop,
                "target1":         tp1,
                "target2":         tp2,
                "rr1":             _calc_rr(entry, stop, tp1),
                "rr2":             _calc_rr(entry, stop, tp2),
                "confidence":      conf,
                "rationale":       rationale,
                "key_levels_used": levels,
            }

        elif overall_bias == "bearish":
            # In bearish trend: OTE zone is a PREMIUM bounce area to short into
            entry  = round((ote_low + ote_high) / 2, 1)
            stop   = round(fib.get(0.786, ote_high * 1.005) + 5.0, 1)
            tp1    = round(sl, 1)

            below_fvg = next((f for f in (weekly_fvgs + daily_fvgs)
                              if f["direction"] == "bearish" and f["top"] < entry), None)
            tp2 = round(below_fvg["bottom"], 1) if below_fvg else round(sl * 0.995, 1)

            if ote_low <= current_price <= ote_high:
                conf = "HIGH"
                note = "Price currently IN OTE premium zone."
            elif abs(current_price - ote_low) / ote_low < 0.02:
                conf = "MEDIUM"
                note = "Price approaching OTE premium zone."
            else:
                conf = "LOW"
                note = "Price not yet at OTE premium zone."

            levels = [
                f"OTE Zone ${ote_high:.1f}–${ote_low:.1f} (Fib 0.618–0.705)",
                f"Stop above Fib 0.786 = ${fib.get(0.786, 0):.1f}",
                f"TP1 = Swing Low ${tp1:.1f}",
            ]

            rationale = (
                f"Monthly: {monthly_structure.upper()}, Weekly: {weekly_structure.upper()}. "
                f"OTE zone at ${ote_low:.1f}–${ote_high:.1f} (Fib 0.618–0.705) "
                f"in premium area — ideal for short entries in bearish structure. {note}"
            )

            return {
                **base,
                "direction":       "SHORT",
                "setup_name":      "OTE Retracement — Fib 0.618–0.705",
                "entry":           entry,
                "stop":            stop,
                "target1":         tp1,
                "target2":         tp2,
                "rr1":             _calc_rr(entry, stop, tp1),
                "rr2":             _calc_rr(entry, stop, tp2),
                "confidence":      conf,
                "rationale":       rationale,
                "key_levels_used": levels,
            }

        else:
            # Ranging: use range midpoint as OTE of the range
            pwh = key_levels.get("PWH")
            pwl = key_levels.get("PWL")
            if pwh and pwl and current_price:
                mid = (pwh + pwl) / 2
                ote_range_low  = round(mid - (pwh - pwl) * 0.118, 1)
                ote_range_high = round(mid + (pwh - pwl) * 0.118, 1)
                entry = round(mid, 1)
                if current_price < mid:
                    return {
                        **base,
                        "direction":       "LONG",
                        "setup_name":      "Range OTE — Buy Range Low",
                        "entry":           round(pwl * 1.001, 1),
                        "stop":            round(pwl * 0.995, 1),
                        "target1":         entry,
                        "target2":         round(pwh, 1),
                        "rr1":             _calc_rr(round(pwl * 1.001, 1), round(pwl * 0.995, 1), entry),
                        "rr2":             _calc_rr(round(pwl * 1.001, 1), round(pwl * 0.995, 1), round(pwh, 1)),
                        "confidence":      "LOW",
                        "rationale":       (
                            f"Market ranging between PWH ${pwh:.1f} and PWL ${pwl:.1f}. "
                            f"Price at ${current_price:.1f} in lower half. Buy near range low targeting midpoint and PWH."
                        ),
                        "key_levels_used": [f"PWL ${pwl:.1f}", f"Range Mid ${mid:.1f}", f"PWH ${pwh:.1f}"],
                    }
                else:
                    return {
                        **base,
                        "direction":       "SHORT",
                        "setup_name":      "Range OTE — Sell Range High",
                        "entry":           round(pwh * 0.999, 1),
                        "stop":            round(pwh * 1.005, 1),
                        "target1":         entry,
                        "target2":         round(pwl, 1),
                        "rr1":             _calc_rr(round(pwh * 0.999, 1), round(pwh * 1.005, 1), entry),
                        "rr2":             _calc_rr(round(pwh * 0.999, 1), round(pwh * 1.005, 1), round(pwl, 1)),
                        "confidence":      "LOW",
                        "rationale":       (
                            f"Market ranging between PWH ${pwh:.1f} and PWL ${pwl:.1f}. "
                            f"Price at ${current_price:.1f} in upper half. Short near range high targeting midpoint and PWL."
                        ),
                        "key_levels_used": [f"PWH ${pwh:.1f}", f"Range Mid ${mid:.1f}", f"PWL ${pwl:.1f}"],
                    }
            return _wait_trade(2, "Ranging market with insufficient key level data for OTE.")

    # ─────────────────────────────────────────────────────────────────────
    # TRADE 3 — Liquidity Hunt (counter-trend, always LOW confidence)
    # ─────────────────────────────────────────────────────────────────────
    def _trade3() -> dict:
        base = {
            "id":        3,
            "timeframe": "Weekly Key Levels",
        }

        if overall_bias == "bullish":
            # Bullish bias: anticipate a short-term SWEEP below PWL/CML before up
            liq_targets = [v for v in [key_levels.get("PWL"), key_levels.get("CML")] if v]
            if not liq_targets or current_price is None:
                return _wait_trade(3, "No liquidity level found for hunt setup.")
            liq_target = min(liq_targets)   # deepest low = most likely sweep target

            if current_price <= liq_target:
                return _wait_trade(3, (
                    f"Price ${current_price:.1f} already swept through liquidity ${liq_target:.1f}. "
                    "Hunt may have completed — wait for next setup."
                ))

            entry  = round(current_price, 1)
            stop   = round(current_price * 1.005, 1)
            tp1    = round(liq_target, 1)
            tp2    = round(fib.get(0.618, liq_target), 1) if fib else tp1

            return {
                **base,
                "direction":       "SHORT",
                "setup_name":      "Liquidity Hunt — Stop Sweep Below PWL",
                "entry":           entry,
                "stop":            stop,
                "target1":         tp1,
                "target2":         tp2,
                "rr1":             _calc_rr(entry, stop, tp1),
                "rr2":             _calc_rr(entry, stop, tp2),
                "confidence":      "LOW",
                "rationale":       (
                    f"Bullish overall bias but price at ${current_price:.1f} may sweep "
                    f"buy-side liquidity below PWL ${liq_target:.1f} before resuming upward. "
                    "Counter-trend scalp only — tight stop and small size."
                ),
                "key_levels_used": [f"PWL/CML Liquidity ${liq_target:.1f}"],
            }

        elif overall_bias == "bearish":
            # Bearish bias: anticipate sweep ABOVE PWH/CMH before continuing down
            liq_targets = [v for v in [key_levels.get("PWH"), key_levels.get("CMH")] if v]
            if not liq_targets or current_price is None:
                return _wait_trade(3, "No liquidity level found for hunt setup.")
            liq_target = max(liq_targets)

            if current_price >= liq_target * 0.99:
                return _wait_trade(3, (
                    f"Price ${current_price:.1f} already at/above liquidity level ${liq_target:.1f}. "
                    "Sweep may have occurred — wait for confirmation."
                ))

            entry  = round(current_price, 1)
            stop   = round(current_price * 0.995, 1)
            tp1    = round(liq_target, 1)
            tp2    = round(fib.get(0.618, liq_target), 1) if fib else tp1

            return {
                **base,
                "direction":       "LONG",
                "setup_name":      "Liquidity Hunt — Stop Sweep Above PWH",
                "entry":           entry,
                "stop":            stop,
                "target1":         tp1,
                "target2":         tp2,
                "rr1":             _calc_rr(entry, stop, tp1),
                "rr2":             _calc_rr(entry, stop, tp2),
                "confidence":      "LOW",
                "rationale":       (
                    f"Bearish overall bias but price at ${current_price:.1f} may sweep "
                    f"sell-side liquidity above PWH ${liq_target:.1f} before continuing lower. "
                    "Counter-trend scalp only — tight stop and small size."
                ),
                "key_levels_used": [f"PWH/CMH Liquidity ${liq_target:.1f}"],
            }

        else:
            # Ranging: hunt the range extremes
            pwh = key_levels.get("PWH")
            pwl = key_levels.get("PWL")
            if not (pwh and pwl and current_price):
                return _wait_trade(3, "Ranging market — no clear liquidity hunt setup.")

            mid = (pwh + pwl) / 2
            if current_price > mid:
                # In upper half → expect sweep of PWH then drop
                entry  = round(pwh * 1.001, 1)
                stop   = round(pwh * 1.006, 1)
                tp1    = round(mid, 1)
                tp2    = round(pwl, 1)
                return {
                    **base,
                    "direction":       "SHORT",
                    "setup_name":      "Range Liquidity — Sell PWH Sweep",
                    "entry":           entry,
                    "stop":            stop,
                    "target1":         tp1,
                    "target2":         tp2,
                    "rr1":             _calc_rr(entry, stop, tp1),
                    "rr2":             _calc_rr(entry, stop, tp2),
                    "confidence":      "LOW",
                    "rationale":       (
                        f"Price ${current_price:.1f} in upper half of range. "
                        f"Expect sweep of equal highs (PWH ${pwh:.1f}) then reversal. "
                        "Short after sweep confirmation into midpoint and PWL."
                    ),
                    "key_levels_used": [f"PWH ${pwh:.1f}", f"Range Mid ${mid:.1f}", f"PWL ${pwl:.1f}"],
                }
            else:
                # In lower half → expect sweep of PWL then rise
                entry  = round(pwl * 0.999, 1)
                stop   = round(pwl * 0.994, 1)
                tp1    = round(mid, 1)
                tp2    = round(pwh, 1)
                return {
                    **base,
                    "direction":       "LONG",
                    "setup_name":      "Range Liquidity — Buy PWL Sweep",
                    "entry":           entry,
                    "stop":            stop,
                    "target1":         tp1,
                    "target2":         tp2,
                    "rr1":             _calc_rr(entry, stop, tp1),
                    "rr2":             _calc_rr(entry, stop, tp2),
                    "confidence":      "LOW",
                    "rationale":       (
                        f"Price ${current_price:.1f} in lower half of range. "
                        f"Expect sweep of equal lows (PWL ${pwl:.1f}) then reversal. "
                        "Long after sweep confirmation targeting midpoint and PWH."
                    ),
                    "key_levels_used": [f"PWL ${pwl:.1f}", f"Range Mid ${mid:.1f}", f"PWH ${pwh:.1f}"],
                }

    # ── Assemble final 3 trades ───────────────────────────────────────────
    try:
        t1 = _trade1()
    except Exception as e:
        t1 = _wait_trade(1, f"Trade 1 error: {e}")

    try:
        t2 = _trade2()
    except Exception as e:
        t2 = _wait_trade(2, f"Trade 2 error: {e}")

    try:
        t3 = _trade3()
    except Exception as e:
        t3 = _wait_trade(3, f"Trade 3 error: {e}")

    return [t1, t2, t3]
