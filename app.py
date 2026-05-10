import streamlit as st
import yfinance as yf
import requests
import pandas as pd
from datetime import datetime
import numpy as np
from scipy.stats import norm

# --- CORE MATH ENGINES ---
def calculate_greeks(S, K, T, r, sigma, type="call"):
    if T <= 0 or sigma <= 0 or S <= 0: return 0.0, 0.0, 0.0, 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    delta = norm.cdf(d1) if type == "call" else norm.cdf(d1) - 1
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    theta = (- (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    vega = (S * norm.pdf(d1) * np.sqrt(T)) / 100
    return round(delta, 3), round(gamma, 4), round(theta, 3), round(vega, 3)

def bs_price(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0 or S <= 0: return max(0, S-K)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

# --- PAGE CONFIG ---
st.set_page_config(page_title="Analyst Pro v6.1", layout="wide")

# Safe Initialization
for key in ['price', 'trend', 'sma20', 'pct_change', 'stock_name', 'expiries', 'current_ticker', 'credits_used']:
    if key not in st.session_state:
        if key == 'expiries': st.session_state[key] = []
        elif key == 'credits_used': st.session_state[key] = 0
        else: st.session_state[key] = None

# --- DATA FETCHERS ---
def fetch_alpha_price(ticker, api_key):
    url = f'https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={ticker}&apikey={api_key}'
    try:
        r = requests.get(url, timeout=10)
        data = r.json()
        if "Global Quote" in data:
            st.session_state.credits_used += 1
            return float(data["Global Quote"]["05. price"])
    except: pass
    return None

def get_trend_analysis(ticker):
    try:
        hist = yf.Ticker(ticker).history(period="50d")
        if hist.empty: return "Neutral", 0, 0
        sma20 = hist['Close'].rolling(window=20).mean().iloc[-1]
        current = hist['Close'].iloc[-1]
        trend = "Bullish" if current > sma20 else "Bearish"
        change = ((current / hist['Close'].iloc[-20]) - 1) * 100 if len(hist) > 20 else 0
        return trend, sma20, change
    except: return "Neutral", 0, 0

# --- SIDEBAR CONTROL ---
with st.sidebar:
    st.header("🎮 Control Center")
    st.metric("API Credits", f"{st.session_state.credits_used} / 25")
    ticker = st.text_input("Ticker:", "SHOP").upper()
    fetch_btn = st.button("🚀 Analyze Ticker")
    
    st.divider()
    st.header("🧪 Simulator")
    target_p_slider = st.slider("Target Price Mult.", 0.7, 1.3, 1.0)
    days_sim = st.slider("Days in Future", 0, 30, 0)

# --- MAIN LOGIC ---
if fetch_btn or st.session_state.price:
    api_key = st.secrets.get("ALPHA_VANTAGE_KEY")
    
    if fetch_btn:
        st.session_state.current_ticker = ticker
        st.session_state.price = fetch_alpha_price(ticker, api_key) or yf.Ticker(ticker).fast_info['lastPrice']
        st.session_state.trend, st.session_state.sma20, st.session_state.pct_change = get_trend_analysis(ticker)
        stock_obj = yf.Ticker(ticker)
        st.session_state.stock_name = stock_obj.info.get('longName', ticker)
        st.session_state.expiries = list(stock_obj.options)

    if st.session_state.price:
        S = st.session_state.price
        st.header(f"{st.session_state.stock_name} ({st.session_state.current_ticker})")
        
        col_p, col_t = st.columns(2)
        col_p.metric("Current Price", f"${S:.2f}")
        
        # Safe check for trend color
        t_val = st.session_state.trend or "Neutral"
        t_pct = st.session_state.pct_change or 0.0
        trend_color = "normal" if t_val == "Bullish" else "inverse"
        col_t.metric("20-Day Trend", t_val, f"{t_pct:.1f}%", delta_color=trend_color)

        if st.session_state.expiries:
            expiry = st.selectbox("Select Expiry:", st.session_state.expiries)
            expiry_dt = pd.to_datetime(expiry)
            days_to_expiry = (expiry_dt.date() - datetime.now().date()).days
            T_years = max(days_to_expiry, 0.5) / 365

            target_val = S * target_p_slider
            chain = yf.Ticker(st.session_state.current_ticker).option_chain(expiry).calls
            avg_iv = chain['impliedVolatility'].median()
            expected_move = S * avg_iv * np.sqrt(T_years)

            # Recommendations
            cons_strike = chain.iloc[(chain['strike'] - (S + expected_move*0.3)).abs().argsort()[:1]].iloc[0]
            aggr_strike = chain.iloc[(chain['strike'] - (S + expected_move*0.8)).abs().argsort()[:1]].iloc[0]

            st.divider()
            tabs = st.tabs(["🛡️ Conservative Strategy", "⚡ Aggressive Strategy"])

            def render_analysis(tab, contract, mode):
                with tab:
                    mid_price = (contract['bid'] + contract['ask']) / 2 if contract['bid'] > 0 else contract['lastPrice']
                    d, g, t, v = calculate_greeks(S, contract['strike'], T_years, 0.05, contract['impliedVolatility'])
                    
                    T_future = max(days_to_expiry - days_sim, 0.5) / 365
                    sim_price = bs_price(target_val, contract['strike'], T_future, 0.05, contract['impliedVolatility'])
                    roi = ((sim_price / mid_price) - 1) * 100

                    score = 0
                    if st.session_state.trend == "Bullish": score += 1
                    if d > 0.3: score += 1
                    if abs(t) < (mid_price * 0.1): score += 1
                    
                    col_v, col_g, col_ch = st.columns([1.5, 1.5, 2])
                    with col_v:
                        st.subheader("Verdict")
                        if score >= 3: st.success("✅ HIGH CONVICTION BUY")
                        elif score == 2: st.warning("⚠️ CAUTION: SETUP WEAK")
                        else: st.error("❌ NO-BUY: POOR PROBABILITY")
                        st.write(f"**Target Entry (Mid):** `${mid_price:.2f}`")
                        st.write(f"**Prob. ITM:** `{d*100:.1f}%`主力")
                        st.metric("Simulated ROI", f"{roi:.1f}%")

                    with col_g:
                        st.subheader("The Greeks")
                        st.write(f"🔹 **Delta:** `{d}`")
                        st.write(f"🔸 **Theta:** `-${abs(t):.3f}/day`")
                        st.write(f"🔺 **Gamma:** `{g}`")
                        st.write(f"🌊 **Vega:** `{v}`")

                    with col_ch:
                        st.write("**30-Day Option Price**")
                        h = yf.Ticker(contract['contractSymbol']).history(period="1mo")
                        if not h.empty: st.line_chart(h['Close'])

            render_analysis(tabs[0], cons_strike, "CONS")
            render_analysis(tabs[1], aggr_strike, "AGGR")
else:
    st.info("Enter ticker and click 'Analyze Ticker' to begin.")
