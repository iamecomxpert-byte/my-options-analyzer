import streamlit as st
import yfinance as yf
import requests
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from scipy.stats import norm

# --- BLACK-SCHOLES & PROBABILITY ENGINES ---
def calculate_greeks(S, K, T, r, sigma, type="call"):
    if T <= 0 or sigma <= 0 or S <= 0: return 0.0, 0.0, 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    delta = norm.cdf(d1) if type == "call" else norm.cdf(d1) - 1
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    theta = (- (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    return round(delta, 2), round(gamma, 4), round(theta, 3)

def bs_price(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0 or S <= 0: return max(0, S-K)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

# --- PAGE CONFIG ---
st.set_page_config(page_title="Options Analyst Pro v5.9", layout="wide")

# Initialize Session State
if 'price' not in st.session_state: st.session_state.price = None
if 'expiries' not in st.session_state: st.session_state.expiries = []
if 'credits_used' not in st.session_state: st.session_state.credits_used = 0
if 'current_ticker' not in st.session_state: st.session_state.current_ticker = ""

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

def fetch_yahoo_meta(ticker):
    try:
        stock = yf.Ticker(ticker)
        return stock.info.get('longName', ticker), list(stock.options)
    except: return ticker, []

# --- MAIN UI HEADER ---
st.title("🏦 Options Analyst Pro v5.9")
api_key = st.secrets.get("ALPHA_VANTAGE_KEY")

with st.sidebar:
    st.header("📊 Session")
    st.metric("API Credits", f"{st.session_state.credits_used} / 25")
    ticker_input = st.text_input("Enter Ticker:", "SHOP").upper()
    if st.button("🚀 Fetch Data"):
        st.session_state.current_ticker = ticker_input
        st.session_state.price = fetch_alpha_price(ticker_input, api_key)
        st.session_state.stock_name, st.session_state.expiries = fetch_yahoo_meta(ticker_input)
        if not st.session_state.price:
            st.session_state.price = yf.Ticker(ticker_input).fast_info['lastPrice']

# --- DASHBOARD START ---
if st.session_state.price and st.session_state.expiries:
    S = st.session_state.price
    st.subheader(f"{st.session_state.stock_name} ({st.session_state.current_ticker})")
    st.metric("Current Price", f"${S:.2f}")

    # Expiry Selection (Default to closest date)
    expiry = st.selectbox("Choose Expiry Date:", st.session_state.expiries, index=0)
    expiry_dt = pd.to_datetime(expiry)
    days_total = (expiry_dt.date() - datetime.now().date()).days
    T = max(days_total, 0.5) / 365

    # Fetch Option Chain
    stock_obj = yf.Ticker(st.session_state.current_ticker)
    chain = stock_obj.option_chain(expiry).calls

    # --- SIMULATOR SETTINGS (Sidebar) ---
    with st.sidebar:
        st.divider()
        st.header("🧪 Simulator")
        target_p = st.slider("Target Price at Expiry", float(S*0.7), float(S*1.3), float(S))
        days_sim = st.slider("Days from Today", 0, max(0, days_total), 0)
        T_sim = max(days_total - days_sim, 0.5) / 365

    # --- RECOMMENDATION LOGIC ---
    # Expected Move = S * IV * sqrt(T)
    avg_iv = chain['impliedVolatility'].median()
    expected_move = S * avg_iv * np.sqrt(T)
    
    # Conservative: Strikes inside the expected move
    cons_strike = S + (expected_move * 0.5)
    cons_contract = chain.iloc[(chain['strike']-cons_strike).abs().argsort()[:1]].iloc[0]
    
    # Aggressive: Strikes at/outside the expected move
    aggr_strike = S + (expected_move * 1.2)
    aggr_contract = chain.iloc[(chain['strike']-aggr_strike).abs().argsort()[:1]].iloc[0]

    st.divider()
    t1, t2 = st.tabs(["🛡️ Conservative Recommendation", "⚡ Aggressive Recommendation"])

    def render_strategy(tab, contract, label):
        with tab:
            c1, c2, c3 = st.columns([1, 1, 2])
            mid = (contract['bid'] + contract['ask']) / 2 if contract['bid'] > 0 else contract['lastPrice']
            K, iv = contract['strike'], contract['impliedVolatility']
            
            # Simulator Math
            val_sim = bs_price(target_p, K, T_sim, 0.05, iv)
            roi = ((val_sim / mid) - 1) * 100
            
            # Likelihood (Delta is a proxy for probability of expiring ITM)
            d, g, th = calculate_greeks(S, K, T, 0.05, iv)
            prob = abs(d) * 100

            with c1:
                st.write(f"**{label} Strike**")
                st.subheader(f"${K}")
                st.write(f"Entry: `${mid:.2f}`")
                st.write(f"IV: `{iv:.1%}`")
            
            with c2:
                st.write("**Simulated Return**")
                st.metric("ROI %", f"{roi:.1f}%")
                st.write(f"Prob. ITM: `{prob:.1f}%`主力")
                if prob > 60: st.success("Likely Move")
                elif prob > 30: st.warning("Moderate Risk")
                else: st.error("Low Probability")

            with c3:
                # 30 Day History Graph
                hist = yf.Ticker(contract['contractSymbol']).history(period="1mo")
                if not hist.empty:
                    st.write("Last 30 Days Option Price")
                    st.line_chart(hist['Close'])

    render_strategy(t1, cons_call := cons_contract, "Conservative")
    render_strategy(t2, aggr_call := aggr_contract, "Aggressive")

else:
    st.info("👈 Enter a ticker and click 'Fetch Data' in the sidebar to begin.")
