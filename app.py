import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime

# ==========================================================
# CONFIG
# ==========================================================
st.set_page_config(page_title="BNB Signal Engine v2", layout="wide")
st.title("🏦 BNB Signal Engine v2 — Institutional Grade (FIXED)")

# ==========================================================
# SESSION STATE (HISTÓRICO)
# ==========================================================
if "signal_log" not in st.session_state:
    st.session_state.signal_log = []

# ==========================================================
# DATA
# ==========================================================
@st.cache_data(ttl=3600)
def load_data():
    df = yf.download(
        "BNB-USD",
        start="2010-07-17",
        interval="1d",
        auto_adjust=True,
        progress=False
    )

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.reset_index(inplace=True)
    return df


df = load_data()

if df.empty:
    st.error("Sem dados disponíveis")
    st.stop()

# ==========================================================
# POWER LAW
# ==========================================================
def power_law(df):
    df = df.copy()

    df["Date"] = pd.to_datetime(df["Date"])
    genesis = pd.Timestamp("2009-01-03")

    df["Days"] = (df["Date"] - genesis).dt.days.astype(float)
    df = df[df["Days"] > 0].copy()

    x = np.log10(df["Days"].to_numpy())
    y = np.log10(df["Close"].to_numpy())

    slope, intercept = np.polyfit(x, y, 1)

    df["PowerLaw"] = 10 ** (intercept + slope * x)

    return df


df = power_law(df)

# ==========================================================
# INDICADORES
# ==========================================================
def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def rsi(series, period=14):
    delta = series.diff()

    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    avg_gain = pd.Series(gain).rolling(period).mean()
    avg_loss = pd.Series(loss).rolling(period).mean()

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


df["EMA9"] = ema(df["Close"], 9)
df["EMA29"] = ema(df["Close"], 29)
df["EMA69"] = ema(df["Close"], 69)
df["EMA169"] = ema(df["Close"], 169)

df["RSI"] = rsi(df["Close"], 14)

df = df.dropna()

# ==========================================================
# STATE
# ==========================================================
price = float(df["Close"].iloc[-1])
ema69 = float(df["EMA69"].iloc[-1])
ema9 = float(df["EMA9"].iloc[-1])
ema29 = float(df["EMA29"].iloc[-1])
ema169 = float(df["EMA169"].iloc[-1])

rsi_now = float(df["RSI"].iloc[-1])
pl = float(df["PowerLaw"].iloc[-1])

trend_ok = price > ema69

# ==========================================================
# EMA RIBBON DETECTION
# ==========================================================
ema_max = max(ema9, ema29, ema69, ema169)
ema_min = min(ema9, ema29, ema69, ema169)

compression = (ema_max - ema_min) / ema69

if ema9 > ema29 > ema69 > ema169:
    ribbon_state = "BULLISH"
elif ema9 < ema29 < ema69 < ema169:
    ribbon_state = "BEARISH"
elif compression < 0.08:
    ribbon_state = "COMPRESSION"
else:
    ribbon_state = "NEUTRAL"

# ==========================================================
# SCORE ENGINE (COM RIBBON)
# ==========================================================
trend_score = 60 if trend_ok else 0
momentum_score = np.clip((40 - rsi_now) * 1.5, 0, 25)
quality_score = 15 if rsi_now < 45 else 5 if rsi_now < 55 else 0

# RIBBON SCORE (NOVO)
if ribbon_state == "BULLISH":
    ribbon_score = 15
elif ribbon_state == "COMPRESSION":
    ribbon_score = 8
elif ribbon_state == "NEUTRAL":
    ribbon_score = 3
else:
    ribbon_score = 0

score = trend_score + momentum_score + quality_score + ribbon_score

# ==========================================================
# STATE MACHINE
# ==========================================================
if not trend_ok:
    state = "BLOCKED"
    signal = "⛔ BLOQUEADO (ABAIXO DA EMA 169)"

elif score >= 75:
    state = "LONG"
    signal = "🟢 LONG SETUP CONFIRMADO"

elif score >= 50:
    state = "WAIT"
    signal = "🟡 AGUARDAR CONFIRMAÇÃO"

else:
    state = "NO_TRADE"
    signal = "🔴 SEM TRADE"

# ==========================================================
# LOG
# ==========================================================
last = st.session_state.signal_log[-1]["state"] if st.session_state.signal_log else None

entry = {
    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "price": price,
    "ema169": ema169,
    "rsi": rsi_now,
    "score": score,
    "state": state,
    "signal": signal,
    "ribbon": ribbon_state
}

if last != state:
    st.session_state.signal_log.append(entry)

# ==========================================================
# UI
# ==========================================================
if state == "LONG":
    st.success(signal)
elif state == "WAIT":
    st.warning(signal)
else:
    st.error(signal)

# ==========================================================
# METRICS
# ==========================================================
c1, c2, c3, c4 = st.columns(4)

c1.metric("BNB", f"${price:,.0f}")
c2.metric("EMA 169", f"${ema169:,.0f}")
c3.metric("Power Law", f"${pl:,.0f}")
c4.metric("Score", f"{score:.1f}/100")

st.divider()

# ==========================================================
# CHART
# ==========================================================
fig = go.Figure()

fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"], name="BNB"))

fig.add_trace(go.Scatter(x=df["Date"], y=df["EMA9"], name="EMA 9"))
fig.add_trace(go.Scatter(x=df["Date"], y=df["EMA29"], name="EMA 29"))
fig.add_trace(go.Scatter(x=df["Date"], y=df["EMA69"], name="EMA 69"))
fig.add_trace(go.Scatter(x=df["Date"], y=df["EMA169"], name="EMA 169"))

fig.add_trace(go.Scatter(
    x=df["Date"],
    y=df["PowerLaw"],
    name="Power Law",
    line=dict(dash="dot", width=2)
))

fig.update_layout(height=650, yaxis_type="log")

st.plotly_chart(fig, use_container_width=True)

# ==========================================================
# HISTÓRICO
# ==========================================================
st.subheader("📊 Histórico de Sinais")

log_df = pd.DataFrame(st.session_state.signal_log)

if not log_df.empty:
    st.dataframe(log_df, use_container_width=True)

    st.download_button(
        "📥 Baixar histórico",
        log_df.to_csv(index=False),
        file_name="signal_log.csv",
        mime="text/csv"
    )

# ==========================================================
# RESUMO
# ==========================================================
st.subheader("Resumo Institucional")

st.write({
    "Preço": price,
    "EMA9": ema9,
    "EMA29": ema29,
    "EMA69": ema69,
    "EMA169": ema169,
    "Power Law": pl,
    "RSI": rsi_now,
    "Ribbon State": ribbon_state,
    "Score": score,
    "State": state
})
