import streamlit as st
import yfinance as yf
import requests
import pandas as pd
from datetime import datetime
import numpy as np
from scipy.stats import norm

# --- THE BLACK-SCHOLES ENGINE ---
def bs_price(S, K, T, r, sigma, type="call"):
    if T <= 0 or sigma <= 0 or S <= 0:
        return max(0, S - K) if type == "call" else max(0, K - S)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2) if type == "call" else K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def calculate_greeks(S, K, T, r, sigma, type="call"):
    if T <= 0 or sigma <= 0 or S <= 0: return 0.0, 0.0, 0.0
    try:
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        delta = norm.cdf(d1) if type == "call" else norm.cdf(d1) - 1
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        theta = (- (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
        return round(delta, 2), round(gamma, 4), round(theta, 3)
    except: return 0.0, 0.0, 0.0

# --- DATA FETCHERS WITH TIMEOUTS ---
@st.cache_data(ttl=300)
def get_api_price(ticker, api_key):
    url = f'https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={ticker}&apikey={api_key}'
    try:
        r = requests.get(url, timeout=5)
        data = r.json()
        if "Global Quote" in data and "05. price" in data["Global Quote"]:
            return float(data["Global Quote"]["05. price"])
    except: pass
    return None

@st.cache_data(ttl=600)
def get_expiries(ticker):
    try:
        # We manually trigger a 'fast_info' call to wake up the connection
        stock = yf.Ticker(ticker)
        _ = stock.fast_info 
        
        options = stock.options
        if options:
            return list(options)
        return []
    except Exception as e:
        # If Yahoo is totally blocked, we log it for the user
        st.sidebar.error(f"Yahoo Connection Error: {str(e)}")
        return []

# --- PAGE SETUP ---
st.set_page_config(page_title="Modular Options Analyst v5.5", layout="wide")
st.title("📈 Options Analyst (Modular v5.5)")

api_key = st.secrets.get("ALPHA_VANTAGE_KEY")
ticker = st.text_input("Enter Ticker:", "SHOP").upper()

if ticker and api_key:
    # --- PHASE 1: DIAGNOSTICS & PRICE ---
    col_price, col_status = st.columns([1, 1])
    
    with col_status:
        st.write("**System Status**")
        price_status = st.empty()
        options_status = st.empty()

    # Get Price
    price = get_api_price(ticker, api_key)
    if price:
        price_status.success("✅ Price API: Connected")
        col_price.metric(f"{ticker} Price", f"${price:.2f}")
    else:
        price_status.warning("⚠️ Price API: Throttled. Trying Fallback...")
        try:
            price = yf.Ticker(ticker).fast_info['lastPrice']
            col_price.metric(f"{ticker} Price (Fallback)", f"${price:.2f}")
        except:
            st.error("Could not fetch price from any source.")
            st.stop()

    # --- PHASE 2: OPTIONS EXPIRY ---
    expiries = get_expiries(ticker)
    if not expiries:
        options_status.info("🕒 Options: Waiting for Yahoo Finance...")
        st.info(f"Scanning for options on {ticker}... If this persists, the ticker may not have options.")
        st.stop()
    else:
        options_status.success(f"✅ Options: {len(expiries)} dates found")

    # --- PHASE 3: SELECTION & MATH ---
    expiry_selection = st.selectbox("Select Expiry:", expiries, index=min(2, len(expiries)-1))
    
    # Process Dates
    expiry_dt = pd.to_datetime(expiry_selection)
    expiry_str = expiry_dt.strftime('%Y-%m-%d')
    days_total = (expiry_dt.date() - datetime.now().date()).days

    # Load Chain
    chain = yf.Ticker(ticker).option_chain(expiry_str).calls
    
    if chain.empty:
        st.error("Chain loaded but is empty. Try a different expiry.")
    else:
        # Simulator Settings
        with st.sidebar:
            st.header("🧪 Simulator")
            target_p = st.slider("Target Price", float(price*0.8), float(price*1.2), float(price))
            days_sim = st.slider("Days in Future", 0, max(0, days_total), 0)
            risk_amt = st.number_input("Risk $", value=100)

        # Strategy Extraction
        itm = chain[chain['strike'] < price * 0.98]
        cons_call = itm.iloc[-1] if not itm.empty else chain.iloc[0]
        otm = chain[chain['strike'] > price * 1.05]
        aggr_call = otm.iloc[0] if not otm.empty else chain.iloc[-1]

        t1, t2 = st.tabs(["🛡️ Conservative", "⚡ Aggressive"])

        def show_trade(tab, contract):
            with tab:
                c1, c2 = st.columns(2)
                mid = (contract['bid'] + contract['ask']) / 2 if contract['bid'] > 0 else contract['lastPrice']
                S, K, vol = price, contract['strike'], contract['impliedVolatility']
                
                # Math Engine
                T_now = max(days_total, 0.5) / 365
                T_sim = max(days_total - days_sim, 0.5) / 365
                d, g, t = calculate_greeks(S, K, T_now, 0.05, vol)
                v_sim = bs_price(target_p, K, T_sim, 0.05, vol)
                
                with c1:
                    st.metric("New Value", f"${v_sim:.2f}", f"{((v_sim/mid)-1)*100:.1f}%")
                    st.write(f"**Strike:** ${K}")
                with c2:
                    st.write(f"**Delta:** {d}")
                    st.write(f"**Theta:** -${abs(t*100):.2f}/day")

        show_trade(t1, cons_call)
        show_trade(t2, aggr_call)
