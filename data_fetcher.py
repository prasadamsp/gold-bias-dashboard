# =============================================================================
# Gold Weekly Bias Dashboard — Data Fetcher
# =============================================================================
import io
import os
import zipfile
from datetime import datetime, timedelta

import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv

import config

load_dotenv()


def _get_fred_key() -> str:
    """Read FRED API key from .env (local) or Streamlit secrets (cloud)."""
    key = os.getenv("FRED_API_KEY", "")
    if not key:
        try:
            import streamlit as st
            key = st.secrets["FRED_API_KEY"]
        except (KeyError, FileNotFoundError, Exception):
            pass
    return key


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _years_ago(n: int) -> str:
    return (datetime.today() - timedelta(days=365 * n)).strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# Yahoo Finance
# ---------------------------------------------------------------------------

def fetch_weekly_prices(years: int = config.PRICE_HISTORY_YEARS) -> dict[str, pd.DataFrame]:
    """
    Download weekly OHLCV for every ticker in config.TICKERS.
    Returns dict {key: DataFrame with weekly Close}.
    """
    start = _years_ago(years)
    result = {}
    tickers_list = list(config.TICKERS.values())
    keys_list = list(config.TICKERS.keys())

    raw = yf.download(
        tickers_list,
        start=start,
        interval="1wk",
        auto_adjust=True,
        progress=False,
        group_by="ticker",
    )

    for key, ticker in zip(keys_list, tickers_list):
        try:
            if len(tickers_list) == 1:
                df = raw[["Close"]].dropna()
            else:
                df = raw[ticker][["Close"]].dropna()
            df.index = pd.to_datetime(df.index).tz_localize(None)
            result[key] = df
        except Exception:
            result[key] = pd.DataFrame(columns=["Close"])

    return result


def fetch_etf_shares_outstanding() -> dict[str, float | None]:
    """
    Return current shares outstanding for each gold ETF.
    Used to track week-over-week flow.
    """
    etf_keys = ["gld", "iau", "gldm", "phys", "gdx", "gdxj"]
    result = {}
    for key in etf_keys:
        ticker = config.TICKERS[key]
        try:
            info = yf.Ticker(ticker).info
            shares = info.get("sharesOutstanding") or info.get("impliedSharesOutstanding")
            result[key] = shares
        except Exception:
            result[key] = None
    return result


# ---------------------------------------------------------------------------
# FRED API
# ---------------------------------------------------------------------------

def fetch_fred_series(years: int = config.PRICE_HISTORY_YEARS, api_key: str = "") -> dict[str, pd.Series]:
    """
    Download FRED series via direct REST API (no fredapi library needed).
    Falls back gracefully if FRED_API_KEY is missing.
    """
    api_key = api_key or _get_fred_key()
    result = {}

    if not api_key:
        for k in config.FRED_SERIES:
            result[k] = pd.Series(dtype=float, name=k)
        return result

    start = _years_ago(years)
    base_url = "https://api.stlouisfed.org/fred/series/observations"

    for key, series_id in config.FRED_SERIES.items():
        try:
            resp = requests.get(base_url, params={
                "series_id":         series_id,
                "api_key":           api_key,
                "file_type":         "json",
                "observation_start": start,
            }, timeout=15)
            resp.raise_for_status()
            obs = resp.json().get("observations", [])
            dates  = [o["date"] for o in obs]
            values = [float(o["value"]) if o["value"] != "." else float("nan") for o in obs]
            s = pd.Series(values, index=pd.to_datetime(dates), name=key).dropna()
            result[key] = s
        except Exception:
            result[key] = pd.Series(dtype=float, name=key)

    return result


# ---------------------------------------------------------------------------
# CFTC COT Report
# ---------------------------------------------------------------------------

def _cot_url(year: int) -> str:
    return config.COT_REPORT_URL_TEMPLATE.format(year=year)


def _download_cot_year(year: int) -> pd.DataFrame | None:
    url = _cot_url(year)
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            fname = z.namelist()[0]
            with z.open(fname) as f:
                df = pd.read_csv(f, low_memory=False)
        return df
    except Exception:
        return None


def fetch_cot_gold(years: int = config.COT_HISTORICAL_YEARS) -> pd.DataFrame:
    """
    Download CFTC legacy futures-only COT report for all requested years.
    Returns weekly DataFrame for Gold (COMEX) with:
        date, noncomm_long, noncomm_short, noncomm_net, comm_long, comm_short, comm_net
    """
    current_year = datetime.today().year
    frames = []

    for y in range(current_year - years + 1, current_year + 1):
        df = _download_cot_year(y)
        if df is None:
            continue
        # Filter to gold contract
        gold = df[df["CFTC_Contract_Market_Code"].astype(str).str.strip() == config.COT_GOLD_CODE].copy()
        if gold.empty:
            # Try alternate column name
            for col in df.columns:
                if "contract" in col.lower() and "code" in col.lower():
                    gold = df[df[col].astype(str).str.strip() == config.COT_GOLD_CODE].copy()
                    break
        if gold.empty:
            continue
        frames.append(gold)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)

    # Identify date column
    date_col = None
    for c in ["Report_Date_as_YYYY-MM-DD", "As_of_Date_In_Form_YYMMDD", "Report_Date_as_MM_DD_YYYY"]:
        if c in combined.columns:
            date_col = c
            break
    if date_col is None:
        return pd.DataFrame()

    combined["date"] = pd.to_datetime(combined[date_col], errors="coerce")
    combined = combined.dropna(subset=["date"]).sort_values("date")

    # Column mapping (legacy report column names)
    col_map = {
        "NonComm_Positions_Long_All":  "noncomm_long",
        "NonComm_Positions_Short_All": "noncomm_short",
        "Comm_Positions_Long_All":     "comm_long",
        "Comm_Positions_Short_All":    "comm_short",
        "Open_Interest_All":           "open_interest",
    }

    out = combined[["date"] + [c for c in col_map if c in combined.columns]].copy()
    out = out.rename(columns=col_map)
    out = out.drop_duplicates(subset=["date"]).set_index("date")

    if "noncomm_long" in out.columns and "noncomm_short" in out.columns:
        out["noncomm_net"] = out["noncomm_long"] - out["noncomm_short"]
    if "comm_long" in out.columns and "comm_short" in out.columns:
        out["comm_net"] = out["comm_long"] - out["comm_short"]

    out.index = pd.to_datetime(out.index).tz_localize(None)
    return out


# ---------------------------------------------------------------------------
# Aggregate fetch — single call that returns everything
# ---------------------------------------------------------------------------

def fetch_all_data(fred_key: str = "") -> dict:
    """
    Master fetcher. Returns:
    {
        "prices":   dict of weekly OHLCV DataFrames keyed by config name,
        "etf_shares": dict of current shares outstanding,
        "fred":     dict of FRED Series,
        "cot":      DataFrame of weekly gold COT data,
        "fetched_at": datetime
    }
    """
    prices = fetch_weekly_prices()
    etf_shares = fetch_etf_shares_outstanding()
    fred = fetch_fred_series(api_key=fred_key)
    cot = fetch_cot_gold()

    return {
        "prices":     prices,
        "etf_shares": etf_shares,
        "fred":       fred,
        "cot":        cot,
        "fetched_at": datetime.now(),
    }
