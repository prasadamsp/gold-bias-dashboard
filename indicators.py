# =============================================================================
# Gold Weekly Bias Dashboard — Indicator Calculations
# =============================================================================
import numpy as np
import pandas as pd

import config


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def _safe_last(series: pd.Series, n: int = 1) -> float | None:
    """Return the last n-th value of a series, or None if too short."""
    s = series.dropna()
    if len(s) < n:
        return None
    return float(s.iloc[-n])


def pct_change_weekly(series: pd.Series) -> float | None:
    """Percent change last close vs previous close."""
    s = series.dropna()
    if len(s) < 2:
        return None
    return float((s.iloc[-1] / s.iloc[-2] - 1) * 100)


def yoy_change(series: pd.Series) -> float | None:
    """Year-over-year percent change using monthly FRED series."""
    s = series.dropna()
    if len(s) < 13:
        return None
    return float((s.iloc[-1] / s.iloc[-13] - 1) * 100)


# ---------------------------------------------------------------------------
# Moving Averages
# ---------------------------------------------------------------------------

def calc_moving_averages(close: pd.Series, periods: list[int] = config.WEEKLY_MA_PERIODS) -> dict:
    """
    Returns dict with MA values and price-vs-MA status.
    {period: {"ma": float, "above": bool, "pct_diff": float}}
    """
    result = {}
    price = _safe_last(close)
    if price is None:
        return result
    for p in periods:
        ma = close.rolling(p).mean()
        ma_val = _safe_last(ma)
        if ma_val is None:
            continue
        result[p] = {
            "ma": round(ma_val, 2),
            "above": price > ma_val,
            "pct_diff": round((price / ma_val - 1) * 100, 2),
        }
    return result


# ---------------------------------------------------------------------------
# RSI
# ---------------------------------------------------------------------------

def calc_rsi(close: pd.Series, period: int = config.RSI_PERIOD) -> float | None:
    """Wilder's RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return _safe_last(rsi)


# ---------------------------------------------------------------------------
# MACD
# ---------------------------------------------------------------------------

def calc_macd(
    close: pd.Series,
    fast: int = config.MACD_FAST,
    slow: int = config.MACD_SLOW,
    signal: int = config.MACD_SIGNAL,
) -> dict:
    """
    Returns {macd_line, signal_line, histogram, bullish (bool)}.
    """
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line

    m = _safe_last(macd_line)
    s = _safe_last(signal_line)
    h = _safe_last(histogram)
    h_prev = _safe_last(histogram, 2)

    if None in (m, s, h):
        return {"macd_line": None, "signal_line": None, "histogram": None, "bullish": None}

    return {
        "macd_line":   round(m, 4),
        "signal_line": round(s, 4),
        "histogram":   round(h, 4),
        "bullish":     bool(h > 0),
        "crossing_up": bool(h > 0 and h_prev is not None and h_prev < 0),
    }


# ---------------------------------------------------------------------------
# COT Index (percentile)
# ---------------------------------------------------------------------------

def calc_cot_index(cot_df: pd.DataFrame, window: int = config.COT_PERCENTILE_WINDOW) -> dict:
    """
    COT Index = percentile rank of current net speculator position
    over the trailing `window` weeks.
    Returns {net_pos, cot_index (0-100), extreme_long (bool), extreme_short (bool)}
    """
    if cot_df.empty or "noncomm_net" not in cot_df.columns:
        return {"net_pos": None, "cot_index": None, "extreme_long": None, "extreme_short": None}

    net = cot_df["noncomm_net"].dropna()
    if len(net) < 2:
        return {"net_pos": None, "cot_index": None, "extreme_long": None, "extreme_short": None}

    current = float(net.iloc[-1])
    window_data = net.tail(window)
    mn, mx = window_data.min(), window_data.max()
    cot_index = float((current - mn) / (mx - mn) * 100) if mx != mn else 50.0

    return {
        "net_pos":      int(current),
        "cot_index":    round(cot_index, 1),
        "extreme_long":  cot_index > 80,
        "extreme_short": cot_index < 20,
    }


# ---------------------------------------------------------------------------
# ETF Flow (shares outstanding delta)
# ---------------------------------------------------------------------------

def calc_etf_flow(
    prices: dict,
    etf_shares: dict,
) -> dict:
    """
    For each ETF, approximate weekly flow = weekly shares change * end price.
    Since yfinance sharesOutstanding is a snapshot (not historical),
    we use weekly Close change × AUM as a directional proxy.

    Returns per-ETF and combined metrics.
    """
    etf_keys = ["gld", "iau", "gldm", "phys", "gdx", "gdxj"]
    result = {}
    total_flow_proxy = 0.0
    n_valid = 0

    for key in etf_keys:
        close = prices.get(key, pd.DataFrame()).get("Close", pd.Series())
        shares = etf_shares.get(key)
        price = _safe_last(close)
        price_prev = _safe_last(close, 2)

        if price is None or price_prev is None:
            result[key] = {
                "price": None, "weekly_chg_pct": None,
                "aum_m": None, "flow_direction": "unknown"
            }
            continue

        weekly_chg_pct = round((price / price_prev - 1) * 100, 2)
        aum_m = round((shares * price / 1e6), 1) if shares else None

        # Flow proxy: if shares data unavailable, use price momentum direction
        flow_direction = "inflow" if weekly_chg_pct > 0 else "outflow"
        total_flow_proxy += weekly_chg_pct
        n_valid += 1

        result[key] = {
            "price":         round(price, 2),
            "weekly_chg_pct": weekly_chg_pct,
            "aum_m":         aum_m,
            "flow_direction": flow_direction,
            "shares":        shares,
        }

    # Miners/Gold ratio
    gold_close = prices.get("gold", pd.DataFrame()).get("Close", pd.Series())
    gdx_close  = prices.get("gdx",  pd.DataFrame()).get("Close", pd.Series())
    gdxj_close = prices.get("gdxj", pd.DataFrame()).get("Close", pd.Series())

    def _ratio_trend(a: pd.Series, b: pd.Series) -> str | None:
        av, ap = _safe_last(a), _safe_last(a, 2)
        bv, bp = _safe_last(b), _safe_last(b, 2)
        if None in (av, ap, bv, bp) or bv == 0 or bp == 0:
            return None
        ratio_now  = av / bv
        ratio_prev = ap / bp
        return "rising" if ratio_now > ratio_prev else "falling"

    result["gdx_gold_ratio_trend"] = _ratio_trend(gdx_close, gold_close)
    result["gdxj_gdx_ratio_trend"] = _ratio_trend(gdxj_close, gdx_close)
    result["combined_flow_avg_pct"] = round(total_flow_proxy / n_valid, 2) if n_valid else None

    return result


# ---------------------------------------------------------------------------
# Macro snapshot
# ---------------------------------------------------------------------------

def calc_macro_snapshot(prices: dict, fred: dict) -> dict:
    """
    Build a single dict of current macro readings.
    """
    def last(series: pd.Series) -> float | None:
        return _safe_last(series) if not series.empty else None

    def delta(series: pd.Series) -> float | None:
        """Change vs prior reading."""
        s = series.dropna()
        if len(s) < 2:
            return None
        return float(s.iloc[-1] - s.iloc[-2])

    # DXY from Yahoo
    dxy_close = prices.get("dxy", pd.DataFrame()).get("Close", pd.Series())

    # 2Y and 10Y from FRED (daily, resample to weekly)
    def _resample_weekly(s: pd.Series) -> pd.Series:
        if s.empty or not isinstance(s.index, pd.DatetimeIndex):
            return pd.Series(dtype=float)
        return s.resample("W").last()

    t2y  = _resample_weekly(fred.get("treasury_2y",  pd.Series()))
    t10y = _resample_weekly(fred.get("treasury_10y", pd.Series()))

    t2y_val  = last(t2y)
    t10y_val = last(t10y)
    yield_curve = round(t10y_val - t2y_val, 3) if (t2y_val and t10y_val) else None
    yield_curve_prev = None
    if not t2y.empty and not t10y.empty:
        p2 = _safe_last(t2y, 2)
        p10 = _safe_last(t10y, 2)
        yield_curve_prev = round(p10 - p2, 3) if (p2 and p10) else None

    return {
        "dxy": {
            "value":      last(dxy_close),
            "weekly_chg": pct_change_weekly(dxy_close),
        },
        "real_yield_10y": {
            "value": last(fred.get("real_yield_10y", pd.Series())),
            "delta": delta(fred.get("real_yield_10y", pd.Series())),
        },
        "breakeven_10y": {
            "value": last(fred.get("breakeven_10y", pd.Series())),
            "delta": delta(fred.get("breakeven_10y", pd.Series())),
        },
        "fed_funds": {
            "value": last(fred.get("fed_funds", pd.Series())),
            "delta": delta(fred.get("fed_funds", pd.Series())),
        },
        "cpi_yoy": {
            "value": yoy_change(fred.get("cpi_yoy", pd.Series())),
        },
        "pce_yoy": {
            "value": yoy_change(fred.get("pce_yoy", pd.Series())),
        },
        "treasury_2y": {
            "value": t2y_val,
            "delta": delta(t2y),
        },
        "treasury_10y": {
            "value": t10y_val,
            "delta": delta(t10y),
        },
        "yield_curve": {
            "value":   yield_curve,
            "prev":    yield_curve_prev,
            "steepening": (yield_curve > yield_curve_prev) if (yield_curve and yield_curve_prev) else None,
        },
    }


# ---------------------------------------------------------------------------
# Cross-asset snapshot
# ---------------------------------------------------------------------------

def calc_cross_asset_snapshot(prices: dict) -> dict:
    def _last(key: str) -> float | None:
        s = prices.get(key, pd.DataFrame()).get("Close", pd.Series())
        return _safe_last(s)

    def _chg(key: str) -> float | None:
        s = prices.get(key, pd.DataFrame()).get("Close", pd.Series())
        return pct_change_weekly(s)

    gold_price = _last("gold")
    copper_price = _last("copper")
    copper_gold_ratio = round(copper_price / gold_price, 6) if (copper_price and gold_price and gold_price != 0) else None
    copper_gold_close = (
        prices.get("copper", pd.DataFrame()).get("Close", pd.Series()) /
        prices.get("gold",   pd.DataFrame()).get("Close", pd.Series())
    ).dropna() if (not prices.get("copper", pd.DataFrame()).empty and not prices.get("gold", pd.DataFrame()).empty) else pd.Series()

    return {
        "vix":   {"value": _last("vix"),    "weekly_chg": _chg("vix")},
        "spx":   {"value": _last("spx"),    "weekly_chg": _chg("spx")},
        "eurusd":{"value": _last("eurusd"), "weekly_chg": _chg("eurusd")},
        "usdjpy":{"value": _last("usdjpy"), "weekly_chg": _chg("usdjpy")},
        "wti":   {"value": _last("wti"),    "weekly_chg": _chg("wti")},
        "copper_gold_ratio": {
            "value":      copper_gold_ratio,
            "weekly_chg": pct_change_weekly(copper_gold_close) if not copper_gold_close.empty else None,
        },
    }


# ---------------------------------------------------------------------------
# Master indicator builder
# ---------------------------------------------------------------------------

def build_all_indicators(data: dict) -> dict:
    """
    Takes the raw data dict from fetch_all_data() and returns
    a fully computed indicator snapshot.
    """
    prices    = data["prices"]
    etf_shares = data["etf_shares"]
    fred      = data["fred"]
    cot       = data["cot"]

    gold_close = prices.get("gold", pd.DataFrame()).get("Close", pd.Series())

    return {
        "gold_price":   _safe_last(gold_close),
        "gold_weekly_chg": pct_change_weekly(gold_close),
        "macro":        calc_macro_snapshot(prices, fred),
        "technical": {
            "moving_averages": calc_moving_averages(gold_close),
            "rsi":             calc_rsi(gold_close),
            "macd":            calc_macd(gold_close),
        },
        "sentiment": {
            "cot":         calc_cot_index(cot),
            "cot_df":      cot,
            "etf":         calc_etf_flow(prices, etf_shares),
            "gold_silver_ratio": {
                "value": round(
                    _safe_last(prices.get("gold", pd.DataFrame()).get("Close", pd.Series())) /
                    _safe_last(prices.get("silver", pd.DataFrame()).get("Close", pd.Series())), 2
                ) if (
                    _safe_last(prices.get("gold", pd.DataFrame()).get("Close", pd.Series())) and
                    _safe_last(prices.get("silver", pd.DataFrame()).get("Close", pd.Series()))
                ) else None,
                "weekly_chg": pct_change_weekly(
                    (
                        prices.get("gold",  pd.DataFrame()).get("Close", pd.Series()) /
                        prices.get("silver", pd.DataFrame()).get("Close", pd.Series())
                    ).dropna()
                ),
            },
        },
        "cross_asset":  calc_cross_asset_snapshot(prices),
        "prices":       prices,
    }
