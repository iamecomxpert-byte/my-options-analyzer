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

# --- PAGE CONFIG & SESSION STATE ---
st.set_page_config(page_title="Analyst Pro v6.3", layout="wide")

for key in ['price', 'trend', 'sma20', 'pct_change', 'stock_name', 'expiries', 'current_ticker', 'credits_used']:
    if key not in st.session_state:
        if key == 'expiries': st.session_state[key] = []
        elif key == 'credits_used': st.session_state[key] = 0
        else: st.session_state[key] = None

# --- SIDEBAR ---
with st.sidebar:
    st.header("🎮 Control Center")
    st.metric("API Credits", f"{st.session_state.credits_used} / 25")
    ticker = st.text_input("Ticker:", "SHOP").upper()
    fetch_btn = st.button("🚀 Analyze Ticker")
    
    st.divider()
    st.header("🧪 Simulator")
    # Percentage-based target price slider
    target_pct = st.slider("Target Price Change (%)", -30, 50, 0, step=1)
    days_sim = st.slider("Days in Future", 0, 30, 0)

# --- DATA LOGIC ---
if fetch_btn or st.session_state.price:
    if fetch_btn:
        st.session_state.current_ticker = ticker
        api_key = st.secrets.get("ALPHA_VANTAGE_KEY")
        url = f'https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={ticker}&apikey={api_key}'
        try:
            r = requests.get(url, timeout=10).json()
            st.session_state.price = float(r["Global Quote"]["05. price"])
            st.session_state.credits_used += 1
        except:
            st.session_state.price = yf.Ticker(ticker).fast_info['lastPrice']
        
        stock_obj = yf.Ticker(ticker)
        st.session_state.stock_name = stock_obj.info.get('longName', ticker)
        st.session_state.expiries = list(stock_obj.options)
        hist = stock_obj.history(period="50d")
        sma20 = hist['Close'].rolling(window=20).mean().iloc[-1]
        st.session_state.trend = "Bullish" if hist['Close'].iloc[-1] > sma20 else "Bearish"
        st.session_state.pct_change = ((hist['Close'].iloc[-1] / hist['Close'].iloc[-20]) - 1) * 100

    S = st.session_state.price
    st.header(f"{st.session_state.stock_name} ({st.session_state.current_ticker})")
    
    col_p, col_t = st.columns(2)
    col_p.metric("Current Price", f"${S:.2f}")
    col_t.metric("20-Day Trend", st.session_state.trend, f"{st.session_state.pct_change:.1f}%")

    expiry = st.selectbox("Select Expiry Date:", st.session_state.expiries)
    days_to_expiry = (pd.to_datetime(expiry).date() - datetime.now().date()).days
    T_years = max(days_to_expiry, 0.5) / 365

    # Fetch Chain
    chain = yf.Ticker(st.session_state.current_ticker).option_chain(expiry).calls
    all_strikes = chain['strike'].tolist()
    avg_iv = chain['impliedVolatility'].median()
    expected_move = S * avg_iv * np.sqrt(T_years)

    # AI Suggestion logic
    ai_cons = all_strikes[min(range(len(all_strikes)), key=lambda i: abs(all_strikes[i] - (S + expected_move*0.3)))]
    ai_aggr = all_strikes[min(range(len(all_strikes)), key=lambda i: abs(all_strikes[i] - (S + expected_move*0.8)))]

    # Calculate target price from percentage
    final_target_price = S * (1 + target_pct / 100)

    st.divider()
    t_cons, t_aggr = st.tabs(["🛡️ Conservative", "⚡ Aggressive"])

    def render_strategy(tab, ai_suggested_strike, label):
        with tab:
            selected_k = st.selectbox(f"Select {label} Strike:", all_strikes, index=all_strikes.index(ai_suggested_strike), key=f"k_{label}")
            contract = chain[chain['strike'] == selected_k].iloc[0]
            
            mid_price = (contract['bid'] + contract['ask']) / 2 if contract['bid'] > 0 else contract['lastPrice']
            d, g, t, v = calculate_greeks(S, selected_k, T_years, 0.05, contract['impliedVolatility'])
            
            # SIM ROI using percentage-derived target price
            sim_p = bs_price(final_target_price, selected_k, max(days_to_expiry - days_sim, 0.5)/365, 0.05, contract['impliedVolatility'])
            roi = ((sim_p / mid_price) - 1) * 100

            score = (1 if st.session_state.trend == "Bullish" else 0) + (1 if d > 0.3 else 0) + (1 if abs(t) < (mid_price * 0.1) else 0)
            
            st.markdown(f"### Selected Strike: **${selected_k}**")
            
            c1, c2, c3 = st.columns([1.5, 1.5, 2])
            with c1:
                if score >= 3: st.success("✅ HIGH CONVICTION BUY")
                elif score == 2: st.warning("⚠️ CAUTION: SETUP WEAK")
                else: st.error("❌ NO-BUY: POOR PROBABILITY")
                st.metric("Target Entry", f"${mid_price:.2f}")
                st.metric("Simulated ROI", f"{roi:.1f}%")
            with c2:
                st.write(f"**Greeks & Probability**")
                st.write(f"Delta: `{d}`")
                st.write(f"Theta: `-{abs(t):.2f}/day`")
                st.write(f"Prob. ITM: `{d*100:.1f}%`主力")
            with c3:
                h = yf.Ticker(contract['contractSymbol']).history(period="1mo")
                if not h.empty: 
                    st.write("**Contract Price History (30d)**")
                    st.line_chart(h['Close'])

    render_strategy(t_cons, ai_cons, "Conservative")
    render_strategy(t_aggr, ai_aggr, "Aggressive")
