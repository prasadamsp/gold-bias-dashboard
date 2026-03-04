# =============================================================================
# Gold Weekly Bias Dashboard — Scoring Engine
# =============================================================================
"""
Each scorer function returns -1 (bearish), 0 (neutral), or +1 (bullish).
The master `score_all` function computes the weighted aggregate score
and maps it to a bias label.
"""
import config


# ---------------------------------------------------------------------------
# Individual scorers
# ---------------------------------------------------------------------------

def _sign_score(val, bullish_if_positive: bool = True) -> int:
    """Simple sign-based score with a small neutral band."""
    if val is None:
        return 0
    if bullish_if_positive:
        return 1 if val > 0 else (-1 if val < 0 else 0)
    else:
        return 1 if val < 0 else (-1 if val > 0 else 0)


# ── Macro ──────────────────────────────────────────────────────────────────

def score_dxy(macro: dict) -> int:
    """Falling DXY → bullish gold."""
    chg = macro.get("dxy", {}).get("weekly_chg")
    return _sign_score(chg, bullish_if_positive=False)


def score_real_yield(macro: dict) -> int:
    """Falling or negative real yield → bullish gold."""
    val = macro.get("real_yield_10y", {}).get("value")
    delta = macro.get("real_yield_10y", {}).get("delta")
    if val is None or delta is None:
        return 0
    if val < 0 or delta < 0:
        return 1
    if delta > 0.05:      # rising by more than 5 bps
        return -1
    return 0


def score_breakeven(macro: dict) -> int:
    """Rising inflation expectations → bullish gold."""
    delta = macro.get("breakeven_10y", {}).get("delta")
    return _sign_score(delta, bullish_if_positive=True)


def score_fed_funds(macro: dict) -> int:
    """Fed cutting cycle → bullish gold."""
    delta = macro.get("fed_funds", {}).get("delta")
    return _sign_score(delta, bullish_if_positive=False)


def score_cpi(macro: dict) -> int:
    """CPI above 2.5% → modest bullish; below 2% → bearish."""
    yoy = macro.get("cpi_yoy", {}).get("value")
    if yoy is None:
        return 0
    if yoy > 2.5:
        return 1
    if yoy < 2.0:
        return -1
    return 0


def score_pce(macro: dict) -> int:
    """PCE above 2% target → bullish; well below → neutral/bearish."""
    yoy = macro.get("pce_yoy", {}).get("value")
    if yoy is None:
        return 0
    if yoy > 2.5:
        return 1
    if yoy < 1.5:
        return -1
    return 0


def score_yield_curve(macro: dict) -> int:
    """
    Inverted but steepening from inversion → early bullish.
    Deeply inverted and re-inverting → bearish.
    """
    val = macro.get("yield_curve", {}).get("value")
    steepening = macro.get("yield_curve", {}).get("steepening")
    if val is None:
        return 0
    # Inversion can signal recession → risk-off → gold positive
    if val < -0.20 and steepening:      # steepening from inversion = rates about to cut
        return 1
    if val > 0.50 and not steepening:   # flattening toward inversion again
        return -1
    return 0


# ── Sentiment ──────────────────────────────────────────────────────────────

def score_cot_index(sentiment: dict) -> int:
    """
    Contrarian: extreme speculative LONGS (>80th pct) = bearish;
    extreme speculative SHORTS (<20th pct) = bullish.
    """
    cot_idx = sentiment.get("cot", {}).get("cot_index")
    if cot_idx is None:
        return 0
    if cot_idx < 20:
        return 1
    if cot_idx > 80:
        return -1
    return 0


def score_cot_trend(sentiment: dict) -> int:
    """Trend-following: net speculative long rising → bullish."""
    # Use the direction of net change (last 2 readings)
    cot_df = sentiment.get("cot_df")
    if cot_df is None or cot_df.empty or "noncomm_net" not in cot_df.columns:
        return 0
    net = cot_df["noncomm_net"].dropna()
    if len(net) < 2:
        return 0
    delta = float(net.iloc[-1]) - float(net.iloc[-2])
    return _sign_score(delta, bullish_if_positive=True)


def score_etf_flows(sentiment: dict) -> int:
    """Net positive ETF flows across GLD+IAU+GLDM → bullish."""
    avg = sentiment.get("etf", {}).get("combined_flow_avg_pct")
    if avg is None:
        return 0
    if avg > 0.5:
        return 1
    if avg < -0.5:
        return -1
    return 0


def score_gold_silver(sentiment: dict) -> int:
    """
    Rising Gold/Silver ratio → risk-off (more defensive), slight positive for gold.
    Falling → risk-on, gold may lag.
    """
    chg = sentiment.get("gold_silver_ratio", {}).get("weekly_chg")
    if chg is None:
        return 0
    # Small magnitude → neutral
    if abs(chg) < 0.5:
        return 0
    return 1 if chg > 0 else -1


# ── Technical ──────────────────────────────────────────────────────────────

def score_ma(technical: dict, period: int) -> int:
    ma_data = technical.get("moving_averages", {}).get(period)
    if ma_data is None:
        return 0
    return 1 if ma_data["above"] else -1


def score_rsi(technical: dict) -> int:
    rsi = technical.get("rsi")
    if rsi is None:
        return 0
    if 50 <= rsi <= 70:     # bullish momentum zone
        return 1
    if rsi > 75:            # overbought
        return -1
    if rsi < 40:            # weak
        return -1
    return 0


def score_macd(technical: dict) -> int:
    macd = technical.get("macd", {})
    bullish = macd.get("bullish")
    crossing_up = macd.get("crossing_up")
    if bullish is None:
        return 0
    if crossing_up:
        return 1   # fresh bullish cross
    if bullish:
        return 1
    return -1


# ── Cross-asset ────────────────────────────────────────────────────────────

def score_vix(cross: dict) -> int:
    """Rising VIX → risk-off → bullish gold."""
    chg = cross.get("vix", {}).get("weekly_chg")
    val = cross.get("vix", {}).get("value")
    if chg is None:
        return 0
    if chg > 5 or (val and val > 25):
        return 1
    if chg < -5 and val and val < 15:
        return -1
    return 0


def score_spx(cross: dict) -> int:
    """Falling S&P500 → safe-haven demand → bullish gold."""
    chg = cross.get("spx", {}).get("weekly_chg")
    if chg is None:
        return 0
    if chg < -2:
        return 1
    if chg > 2:
        return -1
    return 0


def score_eurusd(cross: dict) -> int:
    """Rising EUR/USD = weak USD → bullish gold."""
    chg = cross.get("eurusd", {}).get("weekly_chg")
    return _sign_score(chg, bullish_if_positive=True) if chg and abs(chg) > 0.3 else 0


def score_usdjpy(cross: dict) -> int:
    """Falling USD/JPY = JPY strengthening = risk-off → bullish gold."""
    chg = cross.get("usdjpy", {}).get("weekly_chg")
    return _sign_score(chg, bullish_if_positive=False) if chg and abs(chg) > 0.3 else 0


def score_wti(cross: dict) -> int:
    """Rising WTI can signal inflation expectations → modestly bullish gold."""
    chg = cross.get("wti", {}).get("weekly_chg")
    if chg is None:
        return 0
    if chg > 3:
        return 1
    if chg < -3:
        return -1
    return 0


def score_copper_gold(cross: dict) -> int:
    """Falling Copper/Gold ratio = growth concern → bullish gold."""
    chg = cross.get("copper_gold_ratio", {}).get("weekly_chg")
    return _sign_score(chg, bullish_if_positive=False) if chg and abs(chg) > 0.5 else 0


# ---------------------------------------------------------------------------
# Master scoring
# ---------------------------------------------------------------------------

def score_all(indicators: dict) -> dict:
    """
    Returns:
    {
        "score":       float in [-1, 1],
        "label":       str,
        "color":       str,
        "breakdown":   {group: {indicator: score, ...}},
        "group_scores": {group: float},
    }
    """
    macro    = indicators.get("macro", {})
    tech     = indicators.get("technical", {})
    sent     = indicators.get("sentiment", {})
    cross    = indicators.get("cross_asset", {})

    # ── Macro scores ──
    macro_scores = {
        "dxy":         score_dxy(macro),
        "real_yield":  score_real_yield(macro),
        "breakeven":   score_breakeven(macro),
        "fed_funds":   score_fed_funds(macro),
        "cpi":         score_cpi(macro),
        "pce":         score_pce(macro),
        "yield_curve": score_yield_curve(macro),
    }
    macro_group = sum(
        macro_scores[k] * config.MACRO_SUB_WEIGHTS.get(k, 0)
        for k in macro_scores
    )

    # ── Sentiment scores ──
    sentiment_scores = {
        "cot_index":   score_cot_index(sent),
        "cot_trend":   score_cot_trend(sent),
        "etf_flows":   score_etf_flows(sent),
        "gold_silver": score_gold_silver(sent),
    }
    sentiment_group = sum(
        sentiment_scores[k] * config.SENTIMENT_SUB_WEIGHTS.get(k, 0)
        for k in sentiment_scores
    )

    # ── Technical scores ──
    technical_scores = {
        "ma_20w":  score_ma(tech, 20),
        "ma_50w":  score_ma(tech, 50),
        "ma_200w": score_ma(tech, 200),
        "rsi":     score_rsi(tech),
        "macd":    score_macd(tech),
    }
    technical_group = sum(
        technical_scores[k] * config.TECHNICAL_SUB_WEIGHTS.get(k, 0)
        for k in technical_scores
    )

    # ── Cross-asset scores ──
    cross_scores = {
        "vix":         score_vix(cross),
        "spx":         score_spx(cross),
        "eurusd":      score_eurusd(cross),
        "usdjpy":      score_usdjpy(cross),
        "wti":         score_wti(cross),
        "copper_gold": score_copper_gold(cross),
    }
    cross_group = sum(
        cross_scores[k] * config.CROSS_ASSET_SUB_WEIGHTS.get(k, 0)
        for k in cross_scores
    )

    # ── Weighted aggregate ──
    w = config.SCORING_WEIGHTS
    aggregate = (
        macro_group     * w["macro"] +
        sentiment_group * w["sentiment"] +
        technical_group * w["technical"] +
        cross_group     * w["cross_asset"]
    )
    aggregate = round(max(-1.0, min(1.0, aggregate)), 4)

    # ── Map to label ──
    label, color = "NEUTRAL", "#FFD740"
    for lo, hi, lbl, clr in config.BIAS_LEVELS:
        if lo <= aggregate <= hi:
            label, color = lbl, clr
            break

    return {
        "score":   aggregate,
        "label":   label,
        "color":   color,
        "breakdown": {
            "macro":      macro_scores,
            "sentiment":  sentiment_scores,
            "technical":  technical_scores,
            "cross_asset": cross_scores,
        },
        "group_scores": {
            "macro":      round(macro_group,     4),
            "sentiment":  round(sentiment_group, 4),
            "technical":  round(technical_group, 4),
            "cross_asset": round(cross_group,    4),
        },
    }
