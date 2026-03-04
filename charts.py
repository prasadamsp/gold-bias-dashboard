# =============================================================================
# Gold Weekly Bias Dashboard — Chart Builders (Plotly)
# =============================================================================
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import config

# Common theme colours
BULL_COLOR  = "#00C853"
BEAR_COLOR  = "#D50000"
NEUTRAL_COLOR = "#FFD740"
GOLD_COLOR  = "#FFD700"
BG_COLOR    = "#0E1117"
GRID_COLOR  = "#1E2129"
TEXT_COLOR  = "#FAFAFA"

_LAYOUT = dict(
    paper_bgcolor=BG_COLOR,
    plot_bgcolor=BG_COLOR,
    font=dict(color=TEXT_COLOR, family="Inter, Arial, sans-serif", size=12),
    margin=dict(l=10, r=10, t=40, b=10),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
    xaxis=dict(gridcolor=GRID_COLOR, zeroline=False),
    yaxis=dict(gridcolor=GRID_COLOR, zeroline=False),
)


def _apply_layout(fig, **extra) -> go.Figure:
    fig.update_layout(**{**_LAYOUT, **extra})
    return fig


# ---------------------------------------------------------------------------
# Gold Price + Moving Averages
# ---------------------------------------------------------------------------

def chart_gold_price_ma(prices: dict, ma_data: dict) -> go.Figure:
    gold = prices.get("gold", pd.DataFrame()).get("Close", pd.Series())
    if gold.empty:
        return go.Figure()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=gold.index, y=gold.values,
        name="Gold (GC=F)", line=dict(color=GOLD_COLOR, width=2),
    ))

    colors = {20: "#40C4FF", 50: "#FF6D00", 200: "#CE93D8"}
    for p, data in ma_data.items():
        ma_series = gold.rolling(p).mean()
        fig.add_trace(go.Scatter(
            x=ma_series.index, y=ma_series.values,
            name=f"{p}W MA", line=dict(color=colors.get(p, "#888"), width=1.2, dash="dot"),
        ))

    _apply_layout(fig, title="Gold Price — Weekly + MAs", height=350)
    return fig


# ---------------------------------------------------------------------------
# RSI Chart
# ---------------------------------------------------------------------------

def chart_rsi(prices: dict) -> go.Figure:
    from indicators import calc_rsi
    gold = prices.get("gold", pd.DataFrame()).get("Close", pd.Series())
    if gold.empty:
        return go.Figure()

    # Compute RSI series (rolling, not just last value)
    import numpy as np
    delta = gold.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / config.RSI_PERIOD, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / config.RSI_PERIOD, adjust=False).mean()
    rs   = avg_gain / avg_loss.replace(0, np.nan)
    rsi_series = 100 - (100 / (1 + rs))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rsi_series.index, y=rsi_series.values,
        name="RSI (14)", line=dict(color="#40C4FF", width=1.8),
        fill="tozeroy", fillcolor="rgba(64,196,255,0.08)",
    ))
    # Reference lines
    for level, color, label in [(70, BEAR_COLOR, "OB 70"), (50, NEUTRAL_COLOR, "50"), (30, BULL_COLOR, "OS 30")]:
        fig.add_hline(y=level, line=dict(color=color, dash="dot", width=1), annotation_text=label,
                      annotation_position="right")

    _apply_layout(fig, title="RSI (14) — Weekly", height=200, yaxis=dict(range=[0, 100], gridcolor=GRID_COLOR))
    return fig


# ---------------------------------------------------------------------------
# MACD Chart
# ---------------------------------------------------------------------------

def chart_macd(prices: dict) -> go.Figure:
    gold = prices.get("gold", pd.DataFrame()).get("Close", pd.Series())
    if gold.empty:
        return go.Figure()

    ema_fast   = gold.ewm(span=config.MACD_FAST,   adjust=False).mean()
    ema_slow   = gold.ewm(span=config.MACD_SLOW,   adjust=False).mean()
    macd_line  = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=config.MACD_SIGNAL, adjust=False).mean()
    histogram  = macd_line - signal_line

    bar_colors = [BULL_COLOR if v >= 0 else BEAR_COLOR for v in histogram.values]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=histogram.index, y=histogram.values,
        name="Histogram", marker_color=bar_colors, opacity=0.7,
    ))
    fig.add_trace(go.Scatter(
        x=macd_line.index, y=macd_line.values,
        name="MACD", line=dict(color="#40C4FF", width=1.5),
    ))
    fig.add_trace(go.Scatter(
        x=signal_line.index, y=signal_line.values,
        name="Signal", line=dict(color=NEUTRAL_COLOR, width=1.2, dash="dot"),
    ))
    fig.add_hline(y=0, line=dict(color="#555", width=1))

    _apply_layout(fig, title="MACD — Weekly", height=220)
    return fig


# ---------------------------------------------------------------------------
# COT Chart
# ---------------------------------------------------------------------------

def chart_cot(cot_df: pd.DataFrame) -> go.Figure:
    if cot_df.empty or "noncomm_net" not in cot_df.columns:
        fig = go.Figure()
        fig.add_annotation(text="COT data unavailable", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font=dict(color=TEXT_COLOR, size=14))
        _apply_layout(fig, title="COT — Non-Commercial Net Positions", height=250)
        return fig

    net = cot_df["noncomm_net"].dropna()
    bar_colors = [BULL_COLOR if v >= 0 else BEAR_COLOR for v in net.values]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=net.index, y=net.values,
        name="Net Spec Long", marker_color=bar_colors, opacity=0.8,
    ))
    fig.add_hline(y=0, line=dict(color="#555", width=1))

    _apply_layout(fig, title="COT — Managed Money Net Positions (Gold Futures)", height=250)
    return fig


# ---------------------------------------------------------------------------
# ETF Flows Bar Chart
# ---------------------------------------------------------------------------

def chart_etf_flows(etf_data: dict) -> go.Figure:
    etf_keys  = ["gld", "iau", "gldm", "phys", "gdx", "gdxj"]
    labels    = ["GLD", "IAU", "GLDM", "PHYS", "GDX", "GDXJ"]
    values    = [etf_data.get(k, {}).get("weekly_chg_pct", 0) or 0 for k in etf_keys]
    colors    = [BULL_COLOR if v >= 0 else BEAR_COLOR for v in values]

    fig = go.Figure(go.Bar(
        x=labels, y=values,
        marker_color=colors, opacity=0.85,
        text=[f"{v:+.2f}%" for v in values],
        textposition="outside",
    ))
    fig.add_hline(y=0, line=dict(color="#555", width=1))
    _apply_layout(fig, title="Gold ETFs — Weekly Price Change %", height=280,
                  yaxis=dict(gridcolor=GRID_COLOR, zeroline=False, ticksuffix="%"))
    return fig


# ---------------------------------------------------------------------------
# DXY Chart
# ---------------------------------------------------------------------------

def chart_dxy(prices: dict) -> go.Figure:
    dxy = prices.get("dxy", pd.DataFrame()).get("Close", pd.Series())
    if dxy.empty:
        return go.Figure()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dxy.index, y=dxy.values,
        name="DXY", line=dict(color="#FF6D00", width=2),
        fill="tozeroy", fillcolor="rgba(255,109,0,0.07)",
    ))
    _apply_layout(fig, title="US Dollar Index (DXY) — Weekly", height=260)
    return fig


# ---------------------------------------------------------------------------
# Real Yield Chart
# ---------------------------------------------------------------------------

def chart_real_yield(fred: dict) -> go.Figure:
    ry = fred.get("real_yield_10y", pd.Series())
    be = fred.get("breakeven_10y", pd.Series())

    fig = go.Figure()
    if not ry.empty:
        fig.add_trace(go.Scatter(x=ry.index, y=ry.values, name="10Y Real Yield",
                                 line=dict(color="#CE93D8", width=2)))
    if not be.empty:
        fig.add_trace(go.Scatter(x=be.index, y=be.values, name="10Y Breakeven Inflation",
                                 line=dict(color=NEUTRAL_COLOR, width=1.5, dash="dot")))
    fig.add_hline(y=0, line=dict(color=BEAR_COLOR, dash="dot", width=1),
                  annotation_text="0% (real rate threshold)", annotation_position="right")

    _apply_layout(fig, title="10Y Real Yield & Breakeven Inflation", height=280,
                  yaxis=dict(ticksuffix="%", gridcolor=GRID_COLOR))
    return fig


# ---------------------------------------------------------------------------
# Yield Curve Chart
# ---------------------------------------------------------------------------

def chart_yield_curve(fred: dict) -> go.Figure:
    def _safe_resample(s: pd.Series) -> pd.Series:
        if s.empty or not isinstance(s.index, pd.DatetimeIndex):
            return pd.Series(dtype=float)
        return s.resample("W").last()

    t2  = _safe_resample(fred.get("treasury_2y",  pd.Series()))
    t10 = _safe_resample(fred.get("treasury_10y", pd.Series()))

    if t2.empty or t10.empty:
        fig = go.Figure()
        _apply_layout(fig, title="Yield Curve (10Y-2Y)", height=220)
        return fig

    spread = (t10 - t2).dropna()
    bar_colors = [BULL_COLOR if v >= 0 else BEAR_COLOR for v in spread.values]

    fig = go.Figure(go.Bar(x=spread.index, y=spread.values,
                           marker_color=bar_colors, opacity=0.8, name="10Y-2Y Spread"))
    fig.add_hline(y=0, line=dict(color="#555", width=1.5))
    _apply_layout(fig, title="Yield Curve — 10Y minus 2Y Spread (%)", height=220,
                  yaxis=dict(ticksuffix="%", gridcolor=GRID_COLOR))
    return fig


# ---------------------------------------------------------------------------
# Cross-asset overview
# ---------------------------------------------------------------------------

def chart_cross_asset(prices: dict) -> go.Figure:
    assets = {
        "VIX":    prices.get("vix",    pd.DataFrame()).get("Close", pd.Series()),
        "SPX":    prices.get("spx",    pd.DataFrame()).get("Close", pd.Series()),
        "EUR/USD": prices.get("eurusd", pd.DataFrame()).get("Close", pd.Series()),
        "USD/JPY": prices.get("usdjpy", pd.DataFrame()).get("Close", pd.Series()),
        "WTI":    prices.get("wti",    pd.DataFrame()).get("Close", pd.Series()),
    }

    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=list(assets.keys()),
        vertical_spacing=0.10,
        horizontal_spacing=0.06,
    )
    colors = ["#FF6D00", "#40C4FF", "#00C853", "#CE93D8", "#FFD740"]
    positions = [(1,1), (1,2), (2,1), (2,2), (3,1)]

    for (name, series), color, (row, col) in zip(assets.items(), colors, positions):
        if series.empty:
            continue
        fig.add_trace(
            go.Scatter(x=series.index, y=series.values, name=name,
                       line=dict(color=color, width=1.5)),
            row=row, col=col,
        )

    _apply_layout(fig, title="Cross-Asset — Weekly", height=550)
    fig.update_xaxes(gridcolor=GRID_COLOR)
    fig.update_yaxes(gridcolor=GRID_COLOR)
    return fig


# ---------------------------------------------------------------------------
# Bias Gauge (speedometer-style)
# ---------------------------------------------------------------------------

def chart_bias_gauge(score: float, label: str, color: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score * 100,                # scale to -100 … +100
        number={"suffix": "", "font": {"size": 28, "color": color}},
        gauge={
            "axis": {"range": [-100, 100], "tickwidth": 1, "tickcolor": TEXT_COLOR},
            "bar": {"color": color, "thickness": 0.3},
            "bgcolor": BG_COLOR,
            "borderwidth": 0,
            "steps": [
                {"range": [-100, -60], "color": "#D50000"},
                {"range": [-60,  -20], "color": "#FF6D00"},
                {"range": [-20,   20], "color": "#FFD740"},
                {"range": [ 20,   60], "color": "#69F0AE"},
                {"range": [ 60,  100], "color": "#00C853"},
            ],
            "threshold": {"line": {"color": "white", "width": 3}, "thickness": 0.8, "value": score * 100},
        },
        title={"text": label, "font": {"size": 20, "color": color}},
    ))
    _apply_layout(fig, height=260, margin=dict(l=20, r=20, t=40, b=20))
    return fig


# ---------------------------------------------------------------------------
# Score breakdown bar chart
# ---------------------------------------------------------------------------

def chart_score_breakdown(group_scores: dict, breakdown: dict) -> go.Figure:
    groups = list(group_scores.keys())
    scores = [group_scores[g] for g in groups]
    colors = [BULL_COLOR if s > 0 else (BEAR_COLOR if s < 0 else NEUTRAL_COLOR) for s in scores]

    fig = go.Figure(go.Bar(
        y=[g.replace("_", " ").title() for g in groups],
        x=scores,
        orientation="h",
        marker_color=colors,
        opacity=0.85,
        text=[f"{s:+.2f}" for s in scores],
        textposition="outside",
    ))
    fig.add_vline(x=0, line=dict(color="#555", width=1.5))
    _apply_layout(fig, title="Bias Score by Group", height=220,
                  xaxis=dict(range=[-1, 1], gridcolor=GRID_COLOR),
                  yaxis=dict(gridcolor=GRID_COLOR))
    return fig
