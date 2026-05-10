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
    if T <= 0 or sigma <= 0 or S <= 0: return 0, 0, 0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    delta = norm.cdf(d1) if type == "call" else norm.cdf(d1) - 1
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    theta = (- (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    return round(delta, 2), round(gamma, 4), round(theta, 3)

# --- INDEPENDENT DATA FETCHERS ---
@st.cache_data(ttl=300)
def get_api_price(ticker, api_key):
    url = f'https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={ticker}&apikey={api_key}'
    try:
        r = requests.get(url)
        data = r.json()
        if "Global Quote" in data and "05. price" in data["Global Quote"]:
            return float(data["Global Quote"]["05. price"])
    except: pass
    return None

@st.cache_data(ttl=600)
def get_expiries(ticker):
    try: return list(yf.Ticker(ticker).options)
    except: return []

@st.cache_data(ttl=600)
def get_call_chain(ticker, expiry_str):
    try: return yf.Ticker(ticker).option_chain(expiry_str).calls
    except: return pd.DataFrame()

# --- PAGE SETUP ---
st.set_page_config(page_title="Modular Options Analyst", layout="wide")
st.title("📈 Pro Options Analyst (Modular v5.4)")

api_key = st.secrets.get("ALPHA_VANTAGE_KEY")
ticker = st.text_input("Enter Ticker:", "SHOP").upper()

if ticker and api_key:
    # --- BIT 1: STOCK PRICE (Alpha Vantage) ---
    price_container = st.container()
    price = get_api_price(ticker, api_key)
    
    if price:
        price_container.metric(f"{ticker} Current Price", f"${price:.2f}")
    else:
        price_container.warning("⚠️ Stock price (API) is loading or rate-limited. Using last known data...")
        # Fallback to Yahoo for price if API fails
        try: price = yf.Ticker(ticker).fast_info['lastPrice']
        except: price = None
        if price: price_container.metric(f"{ticker} Price (Fallback)", f"${price:.2f}")

    # --- BIT 2: EXPIRY SELECTION (Yahoo) ---
    st.divider()
    expiries = get_expiries(ticker)
    
    if not expiries:
        st.info("🕒 Fetching Options Expiry dates from Yahoo...")
        st.stop() # Wait here until we have dates to continue
    
    expiry_selection = st.selectbox("Select Expiry:", expiries, index=min(2, len(expiries)-1))
    
    # --- BIT 3: CHAIN & SIMULATION ---
    expiry_dt = pd.to_datetime(expiry_selection)
    expiry_str = expiry_dt.strftime('%Y-%m-%d')
    days_total = (expiry_dt.date() - datetime.now().date()).days

    with st.sidebar:
        st.header("🧪 Simulation Settings")
        target_price_sim = st.slider("Target Price ($)", float(price*0.7 if price else 50), float(price*1.3 if price else 150), float(price if price else 100))
        days_passed_sim = st.slider("Days from Today", 0, max(0, days_total), 0)
        risk_dollars = st.number_input("Risk Amount ($)", value=100)

    chain = get_call_chain(ticker, expiry_str)

    if chain.empty:
        st.error(f"❌ Could not load the option chain for {expiry_str}. Yahoo might be rate-limiting.")
    else:
        # Strategy Logic
        itm = chain[chain['strike'] < (price if price else 0) * 0.98]
        cons_call = itm.iloc[-1] if not itm.empty else chain.iloc[0]
        otm = chain[chain['strike'] > (price if price else 0) * 1.05]
        aggr_call = otm.iloc[0] if not otm.empty else chain.iloc[-1]

        tab1, tab2 = st.tabs(["🛡️ Conservative", "⚡ Aggressive"])

        def render_strategy(tab, contract):
            with tab:
                col1, col2 = st.columns(2)
                mid = (contract['bid'] + contract['ask']) / 2 if (contract['bid'] > 0) else contract['lastPrice']
                
                # Math
                S = price if price else mid
                K = contract['strike']
                sigma = contract['impliedVolatility']
                T_now = max(days_total, 0.5) / 365
                T_sim = max(days_total - days_passed_sim, 0.5) / 365
                
                delta, gamma, theta = calculate_greeks(S, K, T_now, 0.05, sigma)
                val_sim = bs_price(target_price_sim, K, T_sim, 0.05, sigma)
                
                with col1:
                    st.metric("Simulated Value", f"${val_sim:.2f}", f"{((val_sim/mid)-1)*100:.1f}%")
                    st.write(f"**Strike:** ${K}")
                    st.write(f"**Delta:** {delta}")
                with col2:
                    st.write(f"**Theta:** -${abs(theta*100):.2f}/day")
                    st.write(f"**Gamma:** {gamma}")
                    st.caption("Greeks calculated via Black-Scholes Engine")

        render_strategy(tab1, cons_call)
        render_strategy(tab2, aggr_call)
