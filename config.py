# =============================================================================
# Gold Weekly Bias Dashboard — Configuration
# =============================================================================

# ---------------------------------------------------------------------------
# Yahoo Finance tickers
# ---------------------------------------------------------------------------
TICKERS = {
    # Gold & Precious Metals
    "gold":    "GC=F",       # Gold futures (continuous)
    "silver":  "SI=F",       # Silver futures (continuous)
    "copper":  "HG=F",       # Copper futures (continuous)

    # USD
    "dxy":     "DX-Y.NYB",   # US Dollar Index

    # Rates
    "tnx":     "^TNX",       # 10Y Treasury yield
    "irx":     "^IRX",       # 13-week T-Bill / 2Y proxy (closest free)
    "tyx":     "^TYX",       # 30Y Treasury yield

    # Equities / Risk
    "spx":     "^GSPC",      # S&P 500
    "vix":     "^VIX",       # CBOE VIX

    # FX
    "eurusd":  "EURUSD=X",   # Euro vs USD
    "usdjpy":  "JPY=X",      # USD vs JPY

    # Energy
    "wti":     "CL=F",       # WTI Crude Oil futures

    # Gold ETFs (for flow tracking)
    "gld":     "GLD",        # SPDR Gold Shares
    "iau":     "IAU",        # iShares Gold Trust
    "gldm":    "GLDM",       # SPDR Gold MiniShares
    "phys":    "PHYS",       # Sprott Physical Gold Trust

    # Gold Miners ETFs
    "gdx":     "GDX",        # VanEck Gold Miners ETF
    "gdxj":    "GDXJ",       # VanEck Junior Gold Miners ETF
}

# ---------------------------------------------------------------------------
# FRED series IDs (requires free API key)
# ---------------------------------------------------------------------------
FRED_SERIES = {
    "real_yield_10y":    "DFII10",      # 10Y TIPS real yield
    "breakeven_10y":     "T10YIE",      # 10Y Breakeven Inflation
    "fed_funds":         "FEDFUNDS",    # Effective Fed Funds Rate
    "cpi_yoy":           "CPIAUCSL",    # CPI All Urban Consumers
    "pce_yoy":           "PCEPI",       # PCE Price Index
    "treasury_2y":       "DGS2",        # 2Y Treasury Yield (daily)
    "treasury_10y":      "DGS10",       # 10Y Treasury Yield (daily)
}

# ---------------------------------------------------------------------------
# CFTC COT — Gold futures contract code (COMEX)
# ---------------------------------------------------------------------------
COT_GOLD_CODE = "088691"
COT_REPORT_URL_TEMPLATE = (
    "https://www.cftc.gov/files/dea/history/fut_disagg_txt_{year}.zip"
)
COT_HISTORICAL_YEARS = 2   # years of COT history to download for percentile calc

# ---------------------------------------------------------------------------
# Indicator parameters
# ---------------------------------------------------------------------------
WEEKLY_MA_PERIODS = [20, 50, 200]   # weeks
RSI_PERIOD = 14                      # weeks
MACD_FAST = 12                       # weeks
MACD_SLOW = 26                       # weeks
MACD_SIGNAL = 9                      # weeks
COT_PERCENTILE_WINDOW = 52           # weeks for COT index percentile

# ---------------------------------------------------------------------------
# Scoring weights (must sum to 1.0)
# ---------------------------------------------------------------------------
SCORING_WEIGHTS = {
    "macro":       0.35,
    "sentiment":   0.30,
    "technical":   0.25,
    "cross_asset": 0.10,
}

# Macro sub-weights (within macro group, must sum to 1.0)
MACRO_SUB_WEIGHTS = {
    "dxy":           0.25,   # USD direction
    "real_yield":    0.25,   # Real interest rate
    "breakeven":     0.15,   # Inflation expectations
    "fed_funds":     0.15,   # Rate cycle
    "cpi":           0.10,   # CPI trend
    "pce":           0.05,   # PCE trend
    "yield_curve":   0.05,   # 10Y-2Y spread direction
}

# Sentiment sub-weights
SENTIMENT_SUB_WEIGHTS = {
    "cot_index":   0.40,   # COT percentile (contrarian)
    "cot_trend":   0.20,   # COT net direction (non-contrarian trend)
    "etf_flows":   0.30,   # Combined ETF net flows
    "gold_silver": 0.10,   # Gold/Silver ratio trend
}

# Technical sub-weights
TECHNICAL_SUB_WEIGHTS = {
    "ma_20w":    0.15,
    "ma_50w":    0.25,
    "ma_200w":   0.30,
    "rsi":       0.15,
    "macd":      0.15,
}

# Cross-asset sub-weights
CROSS_ASSET_SUB_WEIGHTS = {
    "vix":          0.25,
    "spx":          0.15,
    "eurusd":       0.20,
    "usdjpy":       0.20,
    "wti":          0.10,
    "copper_gold":  0.10,
}

# ---------------------------------------------------------------------------
# Bias score thresholds
# ---------------------------------------------------------------------------
BIAS_LEVELS = [
    ( 0.60,  1.00, "STRONG BULLISH", "#00C853"),
    ( 0.20,  0.60, "BULLISH",        "#69F0AE"),
    (-0.20,  0.20, "NEUTRAL",        "#FFD740"),
    (-0.60, -0.20, "BEARISH",        "#FF6D00"),
    (-1.00, -0.60, "STRONG BEARISH", "#D50000"),
]

# ---------------------------------------------------------------------------
# Data history (for charts and calculations)
# ---------------------------------------------------------------------------
PRICE_HISTORY_YEARS = 5    # years of weekly price data to fetch
