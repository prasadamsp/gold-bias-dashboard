# =============================================================================
# Gold Weekly Bias Dashboard — Streamlit App
# =============================================================================
import os

import streamlit as st
from dotenv import load_dotenv

load_dotenv()


def _get_fred_key() -> str:
    """Read FRED key from .env (local) or Streamlit secrets (cloud)."""
    key = os.getenv("FRED_API_KEY", "")
    if not key:
        try:
            key = str(st.secrets["FRED_API_KEY"])
        except Exception:
            key = ""
    return key

st.set_page_config(
    page_title="Gold Weekly Bias Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .main { background-color: #0E1117; }
    .block-container { padding: 1rem 2rem; }
    .metric-card {
        background: #1E2129; border-radius: 10px;
        padding: 14px 18px; margin-bottom: 8px;
    }
    .metric-label  { font-size: 12px; color: #888; text-transform: uppercase; letter-spacing: 0.5px; }
    .metric-value  { font-size: 22px; font-weight: 700; color: #FAFAFA; }
    .metric-change { font-size: 13px; margin-top: 2px; }
    .bull-text  { color: #00C853; }
    .bear-text  { color: #D50000; }
    .neut-text  { color: #FFD740; }
    .score-pill {
        display: inline-block; padding: 6px 20px;
        border-radius: 50px; font-weight: 700;
        font-size: 18px; letter-spacing: 1px;
    }
    h2, h3 { color: #FFD700 !important; }
    section[data-testid="stSidebar"] { background-color: #1E2129; }
    .stButton > button {
        background: #FFD700; color: #000; font-weight: 700;
        border: none; border-radius: 8px; padding: 10px 28px;
        font-size: 15px; cursor: pointer;
    }
    .stButton > button:hover { background: #FFC200; }
    .divider { border-top: 1px solid #2A2D35; margin: 16px 0; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Imports (after page config)
# ---------------------------------------------------------------------------
import charts
import data_fetcher
import indicators
import scoring
import config


# ---------------------------------------------------------------------------
# Data fetching with caching
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner="Fetching market data...")
def load_data(fred_key: str = ""):
    return data_fetcher.fetch_all_data(fred_key=fred_key)


def get_data(force_refresh: bool = False):
    fred_key = _get_fred_key()
    if force_refresh:
        st.cache_data.clear()
    return load_data(fred_key=fred_key)


# ---------------------------------------------------------------------------
# Helper: coloured metric card
# ---------------------------------------------------------------------------

def metric_card(label: str, value: str, change: str = "", change_positive: bool | None = None):
    if change_positive is True:
        chg_class = "bull-text"
    elif change_positive is False:
        chg_class = "bear-text"
    else:
        chg_class = "neut-text"

    chg_html = f'<div class="metric-change {chg_class}">{change}</div>' if change else ""
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {chg_html}
    </div>
    """, unsafe_allow_html=True)


def score_badge(score: float, label: str, color: str):
    st.markdown(f"""
    <div style="text-align:center; margin: 10px 0;">
        <span class="score-pill" style="background:{color}33; color:{color}; border: 2px solid {color};">
            {label}
        </span>
        <div style="color:#888; font-size:13px; margin-top:6px;">
            Score: <strong style="color:{color};">{score:+.2f}</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)


def _fmt(val, decimals: int = 2, suffix: str = "") -> str:
    if val is None:
        return "N/A"
    return f"{val:.{decimals}f}{suffix}"


def _arrow(chg) -> tuple[str, bool | None]:
    if chg is None:
        return "", None
    arrow = "▲" if chg >= 0 else "▼"
    return f"{arrow} {abs(chg):.2f}%", chg >= 0


# ---------------------------------------------------------------------------
# ── MAIN APP ─────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def main():
    # ── Header ──────────────────────────────────────────────────────────
    col_title, col_btn = st.columns([4, 1])
    with col_title:
        st.markdown("## Gold Weekly Bias Dashboard")
        st.markdown("<div style='color:#888; font-size:13px; margin-top:-12px;'>Free sources: Yahoo Finance · FRED · CFTC · All indicators weekly</div>", unsafe_allow_html=True)
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        refresh = st.button("Refresh Data", key="refresh_btn", use_container_width=True)

    # ── FRED key warning ─────────────────────────────────────────────────
    if not os.getenv("FRED_API_KEY"):
        st.warning(
            "**FRED API key not set** — macro indicators (Real Yield, Breakeven, CPI, PCE, Yield Curve) "
            "will show N/A. Add `FRED_API_KEY=your_key` to the `.env` file. "
            "Get a free key at [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html).",
            icon="⚠️",
        )

    # ── Load data ────────────────────────────────────────────────────────
    data = get_data(force_refresh=refresh)

    with st.spinner("Computing indicators..."):
        ind  = indicators.build_all_indicators(data)
        bias = scoring.score_all(ind)

    fetched_at = data.get("fetched_at")
    if fetched_at:
        st.caption(f"Last updated: {fetched_at.strftime('%Y-%m-%d %H:%M:%S')}")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════
    # SECTION 1: BIAS SCORECARD
    # ════════════════════════════════════════════════════════════════════
    st.markdown("### Bias Scorecard")

    gauge_col, breakdown_col, detail_col = st.columns([1.2, 1, 1.8])

    with gauge_col:
        fig_gauge = charts.chart_bias_gauge(bias["score"], bias["label"], bias["color"])
        st.plotly_chart(fig_gauge, use_container_width=True, config={"displayModeBar": False})

    with breakdown_col:
        fig_breakdown = charts.chart_score_breakdown(bias["group_scores"], bias["breakdown"])
        st.plotly_chart(fig_breakdown, use_container_width=True, config={"displayModeBar": False})

    with detail_col:
        st.markdown("**Indicator Signals**")
        all_bd = bias["breakdown"]

        def signal_row(name: str, score: int):
            icon = "🟢" if score > 0 else ("🔴" if score < 0 else "⚪")
            label = "Bullish" if score > 0 else ("Bearish" if score < 0 else "Neutral")
            st.markdown(f"{icon} **{name}** — {label}")

        with st.expander("Macro", expanded=True):
            for k, v in all_bd["macro"].items():
                signal_row(k.replace("_", " ").title(), v)

        with st.expander("Sentiment"):
            for k, v in all_bd["sentiment"].items():
                signal_row(k.replace("_", " ").title(), v)

        with st.expander("Technical"):
            for k, v in all_bd["technical"].items():
                signal_row(k.replace("_", " ").title(), v)

        with st.expander("Cross-Asset"):
            for k, v in all_bd["cross_asset"].items():
                signal_row(k.replace("_", " ").title(), v)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Gold headline
    gold_price = ind.get("gold_price")
    gold_chg   = ind.get("gold_weekly_chg")
    arrow, is_pos = _arrow(gold_chg)
    chg_class = "bull-text" if is_pos else "bear-text"
    st.markdown(
        f"<h3 style='margin:0;'>Gold (XAU/USD) &nbsp; "
        f"<span style='color:#FFD700;'>${_fmt(gold_price)}</span> &nbsp;"
        f"<span class='{chg_class}' style='font-size:16px;'>{arrow} wk</span></h3>",
        unsafe_allow_html=True
    )

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════
    # SECTION 2: MACRO PANEL
    # ════════════════════════════════════════════════════════════════════
    st.markdown("### Macro Indicators")

    macro = ind.get("macro", {})

    mc1, mc2, mc3, mc4, mc5, mc6 = st.columns(6)
    with mc1:
        dxy = macro.get("dxy", {})
        arrow, pos = _arrow(dxy.get("weekly_chg"))
        metric_card("DXY", _fmt(dxy.get("value"), 2), f"{arrow} wk", change_positive=not pos if pos is not None else None)

    with mc2:
        ry = macro.get("real_yield_10y", {})
        delta = ry.get("delta")
        metric_card("10Y Real Yield", _fmt(ry.get("value"), 3, "%"),
                    f"{'▲' if delta and delta>0 else '▼'} {abs(delta):.3f}%" if delta else "",
                    change_positive=delta < 0 if delta else None)

    with mc3:
        be = macro.get("breakeven_10y", {})
        delta = be.get("delta")
        metric_card("10Y Breakeven", _fmt(be.get("value"), 2, "%"),
                    f"{'▲' if delta and delta>0 else '▼'} {abs(delta):.3f}%" if delta else "",
                    change_positive=delta > 0 if delta else None)

    with mc4:
        ff = macro.get("fed_funds", {})
        metric_card("Fed Funds Rate", _fmt(ff.get("value"), 2, "%"), "")

    with mc5:
        metric_card("CPI YoY", _fmt(macro.get("cpi_yoy", {}).get("value"), 2, "%"), "")

    with mc6:
        metric_card("PCE YoY", _fmt(macro.get("pce_yoy", {}).get("value"), 2, "%"), "")

    # 2Y yield & yield curve
    mc_a, mc_b, mc_c = st.columns(3)
    with mc_a:
        t2 = macro.get("treasury_2y", {})
        delta = t2.get("delta")
        metric_card("2Y Treasury Yield", _fmt(t2.get("value"), 3, "%"),
                    f"{'▲' if delta and delta>0 else '▼'} {abs(delta):.3f}%" if delta else "",
                    change_positive=delta < 0 if delta else None)
    with mc_b:
        t10 = macro.get("treasury_10y", {})
        delta = t10.get("delta")
        metric_card("10Y Treasury Yield", _fmt(t10.get("value"), 3, "%"),
                    f"{'▲' if delta and delta>0 else '▼'} {abs(delta):.3f}%" if delta else "",
                    change_positive=delta < 0 if delta else None)
    with mc_c:
        yc = macro.get("yield_curve", {})
        val = yc.get("value")
        steep = yc.get("steepening")
        trend = "Steepening" if steep else ("Flattening" if steep is False else "")
        metric_card("Yield Curve 10Y-2Y", _fmt(val, 3, "%"), trend,
                    change_positive=steep)

    # Charts: Gold + MA, DXY, Real Yield, Yield Curve
    ch1, ch2 = st.columns(2)
    with ch1:
        fig = charts.chart_real_yield(data["fred"])
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    with ch2:
        fig = charts.chart_yield_curve(data["fred"])
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    fig_dxy = charts.chart_dxy(data["prices"])
    st.plotly_chart(fig_dxy, use_container_width=True, config={"displayModeBar": False})

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════
    # SECTION 3: SENTIMENT + COT
    # ════════════════════════════════════════════════════════════════════
    st.markdown("### Sentiment — COT & Positioning")

    cot_data = ind.get("sentiment", {}).get("cot", {})
    gs_ratio  = ind.get("sentiment", {}).get("gold_silver_ratio", {})

    sc1, sc2, sc3, sc4 = st.columns(4)
    with sc1:
        net = cot_data.get("net_pos")
        metric_card("COT Net Spec Long", f"{net:,}" if net else "N/A")
    with sc2:
        ci = cot_data.get("cot_index")
        extreme = "Extreme Long (Contrarian Bearish)" if cot_data.get("extreme_long") else \
                  ("Extreme Short (Contrarian Bullish)" if cot_data.get("extreme_short") else "")
        metric_card("COT Index (52W %ile)", _fmt(ci, 1), extreme,
                    change_positive=(ci < 20) if ci else None)
    with sc3:
        metric_card("Gold/Silver Ratio", _fmt(gs_ratio.get("value"), 2),
                    (_arrow(gs_ratio.get("weekly_chg"))[0]) + " wk")
    with sc4:
        avg_flow = ind.get("sentiment", {}).get("etf", {}).get("combined_flow_avg_pct")
        metric_card("ETF Avg Flow (wk)", _fmt(avg_flow, 2, "%"),
                    change_positive=avg_flow >= 0 if avg_flow is not None else None)

    fig_cot = charts.chart_cot(data["cot"])
    st.plotly_chart(fig_cot, use_container_width=True, config={"displayModeBar": False})

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════
    # SECTION 4: GOLD ETF PANEL
    # ════════════════════════════════════════════════════════════════════
    st.markdown("### Gold ETFs")

    etf_data = ind.get("sentiment", {}).get("etf", {})
    etf_keys = ["gld", "iau", "gldm", "phys", "gdx", "gdxj"]
    etf_labels = ["GLD", "IAU", "GLDM", "PHYS", "GDX", "GDXJ"]
    etf_desc = [
        "SPDR Gold Shares",
        "iShares Gold Trust",
        "SPDR Gold MiniShares",
        "Sprott Physical Gold",
        "VanEck Gold Miners",
        "VanEck Junior Miners",
    ]

    cols = st.columns(6)
    for i, (key, label, desc) in enumerate(zip(etf_keys, etf_labels, etf_desc)):
        d = etf_data.get(key, {})
        price = d.get("price")
        chg   = d.get("weekly_chg_pct")
        aum   = d.get("aum_m")
        arrow_str, is_pos = _arrow(chg)
        with cols[i]:
            aum_str = f"AUM ~${aum:,.0f}M" if aum else ""
            metric_card(f"{label} — {desc}", _fmt(price), f"{arrow_str} wk | {aum_str}", change_positive=is_pos)

    # Miners ratios
    mr1, mr2 = st.columns(2)
    with mr1:
        ratio_trend = etf_data.get("gdx_gold_ratio_trend")
        icon = "▲" if ratio_trend == "rising" else ("▼" if ratio_trend == "falling" else "")
        metric_card("GDX / Gold Ratio Trend", f"{icon} {ratio_trend or 'N/A'}".strip(),
                    "Rising = Miners leading = Bullish confirmation",
                    change_positive=ratio_trend == "rising" if ratio_trend else None)
    with mr2:
        gj_trend = etf_data.get("gdxj_gdx_ratio_trend")
        icon = "▲" if gj_trend == "rising" else ("▼" if gj_trend == "falling" else "")
        metric_card("GDXJ / GDX Ratio Trend", f"{icon} {gj_trend or 'N/A'}".strip(),
                    "Rising = Junior miners outperforming = Higher risk appetite",
                    change_positive=gj_trend == "rising" if gj_trend else None)

    fig_etf = charts.chart_etf_flows(etf_data)
    st.plotly_chart(fig_etf, use_container_width=True, config={"displayModeBar": False})

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════
    # SECTION 5: TECHNICAL PANEL
    # ════════════════════════════════════════════════════════════════════
    st.markdown("### Technical — Weekly")

    tech = ind.get("technical", {})
    mas  = tech.get("moving_averages", {})
    rsi  = tech.get("rsi")
    macd = tech.get("macd", {})

    tc1, tc2, tc3, tc4, tc5 = st.columns(5)
    for col, period in zip([tc1, tc2, tc3], config.WEEKLY_MA_PERIODS):
        ma_d = mas.get(period, {})
        above = ma_d.get("above")
        diff  = ma_d.get("pct_diff")
        with col:
            metric_card(
                f"{period}W MA",
                _fmt(ma_d.get("ma")),
                f"{'Above' if above else 'Below'} ({diff:+.2f}%)" if diff is not None else "",
                change_positive=above,
            )
    with tc4:
        rsi_zone = "Momentum (50-70)" if (rsi and 50 <= rsi <= 70) else \
                   ("Overbought >75" if (rsi and rsi > 75) else \
                   ("Weak <40" if (rsi and rsi < 40) else "Neutral"))
        metric_card("RSI (14W)", _fmt(rsi, 1), rsi_zone,
                    change_positive=(rsi and 50 <= rsi <= 70))
    with tc5:
        macd_bull = macd.get("bullish")
        macd_cross = macd.get("crossing_up")
        macd_label = "Bullish Cross!" if macd_cross else ("Above 0" if macd_bull else "Below 0")
        metric_card("MACD Weekly", _fmt(macd.get("histogram"), 4), macd_label, change_positive=macd_bull)

    # Price + MA chart
    fig_gold = charts.chart_gold_price_ma(data["prices"], mas)
    st.plotly_chart(fig_gold, use_container_width=True, config={"displayModeBar": False})

    rsi_col, macd_col = st.columns(2)
    with rsi_col:
        fig_rsi = charts.chart_rsi(data["prices"])
        st.plotly_chart(fig_rsi, use_container_width=True, config={"displayModeBar": False})
    with macd_col:
        fig_macd = charts.chart_macd(data["prices"])
        st.plotly_chart(fig_macd, use_container_width=True, config={"displayModeBar": False})

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════
    # SECTION 6: CROSS-ASSET PANEL
    # ════════════════════════════════════════════════════════════════════
    st.markdown("### Cross-Asset")

    cross = ind.get("cross_asset", {})

    xc1, xc2, xc3, xc4, xc5, xc6 = st.columns(6)
    cross_items = [
        (xc1, "VIX",     cross.get("vix",    {}), True,  False),
        (xc2, "S&P 500", cross.get("spx",    {}), True,  False),
        (xc3, "EUR/USD", cross.get("eurusd", {}), False, True),
        (xc4, "USD/JPY", cross.get("usdjpy", {}), False, False),
        (xc5, "WTI",     cross.get("wti",    {}), False, True),
        (xc6, "Cu/Au",   cross.get("copper_gold_ratio", {}), False, False),
    ]

    for col, label, d, invert_pos, normal_pos in cross_items:
        val = d.get("value")
        chg = d.get("weekly_chg")
        arrow_str, is_pos = _arrow(chg)
        # Gold-bullish direction
        if label == "VIX":
            gold_pos = is_pos         # rising VIX = gold bullish
        elif label == "S&P 500":
            gold_pos = not is_pos if is_pos is not None else None  # falling SPX = gold bullish
        elif label == "EUR/USD":
            gold_pos = is_pos         # rising EUR/USD = gold bullish (weak USD)
        elif label == "USD/JPY":
            gold_pos = not is_pos if is_pos is not None else None  # falling USDJPY = gold bullish
        elif label == "WTI":
            gold_pos = is_pos         # rising oil = inflation = gold bullish
        elif label == "Cu/Au":
            gold_pos = not is_pos if is_pos is not None else None  # falling Cu/Au = gold bullish
        else:
            gold_pos = None
        with col:
            metric_card(label, _fmt(val, 4 if "USD" in label or "JPY" in label else 2),
                        f"{arrow_str} wk", change_positive=gold_pos)

    fig_cross = charts.chart_cross_asset(data["prices"])
    st.plotly_chart(fig_cross, use_container_width=True, config={"displayModeBar": False})

    # ════════════════════════════════════════════════════════════════════
    # FOOTER
    # ════════════════════════════════════════════════════════════════════
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.caption(
        "Data: Yahoo Finance (prices, ETFs) · FRED API (macro) · CFTC (COT) — all free sources. "
        "This dashboard is for informational purposes only. Not financial advice."
    )


if __name__ == "__main__":
    main()
