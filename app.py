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
st.title("🏦 BNB Signal Engine v2 — Institutional Grade (PROB ENGINE)")

# ==========================================================
# SESSION STATE
# ==========================================================
if "signal_log" not in st.session_state:
    st.session_state.signal_log = []

# ==========================================================
# INPUTS (GAIN / LOSS EM ATR)
# ==========================================================
st.sidebar.header("🎯 Risk Model (ATR Engine)")

gain_atr = st.sidebar.slider("Take Profit (ATR)", 1.0, 10.0, 3.0, 0.5)
loss_atr = st.sidebar.slider("Stop Loss (ATR)", 0.5, 5.0, 1.5, 0.5)

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
# ATR
# ==========================================================
def atr(df, period=14):
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)

    return tr.rolling(period).mean()


df["ATR"] = atr(df, 14)
df = df.dropna().reset_index(drop=True)

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


df["EMA169"] = ema(df["Close"], 169)
df["RSI"] = rsi(df["Close"], 14)

df = df.dropna()

# ==========================================================
# STATE
# ==========================================================
price = float(df["Close"].iloc[-1])
ema169 = float(df["EMA169"].iloc[-1])
rsi_now = float(df["RSI"].iloc[-1])
atr_now = float(df["ATR"].iloc[-1])

trend_ok = price > ema169

# ==========================================================
# SCORE
# ==========================================================
trend_score = 60 if trend_ok else 0
momentum_score = np.clip((40 - rsi_now) * 1.5, 0, 25)
quality_score = 15 if rsi_now < 45 else 5 if rsi_now < 55 else 0

score = trend_score + momentum_score + quality_score

# ==========================================================
# STATE MACHINE
# ==========================================================
if not trend_ok:
    state = "BLOCKED"
    signal = "⛔ BLOQUEADO"

elif score >= 75:
    state = "LONG"
    signal = "🟢 LONG SETUP CONFIRMADO"

elif score >= 50:
    state = "WAIT"
    signal = "🟡 AGUARDAR"

else:
    state = "NO_TRADE"
    signal = "🔴 SEM TRADE"

# ==========================================================
# 🧠 PROBABILITY ENGINE (ATR BASED HISTORICAL SIMULATION)
# ==========================================================
def probability_engine(df, gain_atr, loss_atr, samples=300):

    wins = 0

    valid = df.iloc[:-MAX_DAYS] if "MAX_DAYS" in globals() else df.iloc[:-100]

    for _ in range(samples):

        idx = np.random.randint(50, len(valid) - 1)

        entry = valid.iloc[idx]
        price = entry["Close"]
        atr = entry["ATR"]

        if np.isnan(atr) or atr == 0:
            continue

        for i in range(1, 60):

            future = df.iloc[idx + i]["Close"]

            if future >= price + (gain_atr * atr):
                wins += 1
                break

            if future <= price - (loss_atr * atr):
                break

    return wins / samples if samples > 0 else 0


prob = probability_engine(df, gain_atr, loss_atr)

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
    "prob_gain_first": prob
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

st.sidebar.markdown("### 📊 Probabilidade histórica")
st.sidebar.metric("Gain antes do Loss", f"{prob*100:.1f}%")

# ==========================================================
# METRICS
# ==========================================================
c1, c2, c3, c4 = st.columns(4)

c1.metric("BNB", f"${price:,.0f}")
c2.metric("EMA 169", f"${ema169:,.0f}")
c3.metric("Score", f"{score:.1f}/100")
c4.metric("Prob ATR Edge", f"{prob*100:.1f}%")

st.divider()

# ==========================================================
# CHART
# ==========================================================
fig = go.Figure()

fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"], name="BNB"))
fig.add_trace(go.Scatter(x=df["Date"], y=df["EMA169"], name="EMA 169"))

fig.add_trace(go.Scatter(
    x=df["Date"],
    y=df["PowerLaw"],
    name="Power Law",
    line=dict(dash="dot")
))

fig.update_layout(height=650, yaxis_type="log")

st.plotly_chart(fig, use_container_width=True)

# ==========================================================
# RESUMO
# ==========================================================
st.subheader("Resumo Institucional")

st.write({
    "Preço": price,
    "Score": score,
    "State": state,
    "Gain ATR": gain_atr,
    "Loss ATR": loss_atr,
    "Probabilidade": prob
})
