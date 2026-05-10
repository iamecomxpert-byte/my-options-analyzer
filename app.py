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
    if type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def calculate_greeks(S, K, T, r, sigma, type="call"):
    if T <= 0 or sigma <= 0 or S <= 0: return 0, 0, 0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    delta = norm.cdf(d1) if type == "call" else norm.cdf(d1) - 1
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    theta = (- (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    return round(delta, 2), round(gamma, 4), round(theta, 3)

# --- PAGE SETUP ---
st.set_page_config(page_title="Pro Options Analyst v5.1", layout="wide", page_icon="📈")

# --- CACHED DATA FETCHING ---
@st.cache_data(ttl=300)
def get_api_price(ticker, api_key):
    url = f'https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={ticker}&apikey={api_key}'
    try:
        r = requests.get(url)
        data = r.json()
        if "Global Quote" in data and "05. price" in data["Global Quote"]:
            return float(data["Global Quote"]["05. price"]), None
        return None, "API Limit/Ticker Error"
    except: return None, "Connection Error"

@st.cache_data(ttl=600)
def get_expiries(ticker):
    try: 
        stock = yf.Ticker(ticker)
        return list(stock.options)
    except: return []

@st.cache_data(ttl=600)
def get_call_chain(ticker, expiry):
    try: return yf.Ticker(ticker).option_chain(expiry).calls
    except: return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_option_history(contract_symbol):
    try: return yf.Ticker(contract_symbol).history(period="1mo")
    except: return pd.DataFrame()

# --- MAIN INTERFACE ---
st.title("📈 Pro Options Analyst v5.1")
api_key = st.secrets.get("ALPHA_VANTAGE_KEY")
ticker = st.text_input("Enter Ticker:", "SHOP").upper()

if ticker and api_key:
    price, p_error = get_api_price(ticker, api_key)
    expiries = get_expiries(ticker)
    
    if price:
        # --- SIDEBAR SIMULATOR ---
        with st.sidebar:
            st.header("💰 Risk & Wallet")
            total_budget = st.number_input("Total Budget ($)", value=5000)
            risk_percent = st.slider("Risk per Trade (%)", 1, 10, 2)
            risk_dollars = total_budget * (risk_percent / 100)
            
            st.divider()
            st.header("🧪 What-If Simulator")
            target_price_sim = st.slider("Simulated Stock Price ($)", 
                                         float(price*0.7), float(price*1.3), float(price))
            
            expiry = st.selectbox("Select Expiry for Simulation:", expiries, index=min(2, len(expiries)-1))
            
            # --- FIX: ROBUST DATE HANDLING ---
            expiry_str = expiry if isinstance(expiry, str) else expiry.strftime('%Y-%m-%d')
            days_total = (datetime.strptime(expiry_str, '%Y-%m-%d') - datetime.now()).days
            
            days_passed_sim = st.slider("Days from Today", 0, max(0, days_total), 0)

        # --- DATA DISPLAY ---
        st.metric(f"{ticker} Current Price", f"${price:.2f}")
        chain = get_call_chain(ticker, expiry_str)

        if not chain.empty:
            itm = chain[chain['strike'] < price * 0.98]
            cons_call = itm.iloc[-1] if not itm.empty else chain.iloc[0]
            otm = chain[chain['strike'] > price * 1.05]
            aggr_call = otm.iloc[0] if not otm.empty else chain.iloc[-1]

            tab1, tab2 = st.tabs(["🛡️ Conservative Trade", "⚡ Aggressive Trade"])

            def render_strategy(tab, contract):
                with tab:
                    col_pnl, col_greeks, col_chart = st.columns([1, 1, 2])
                    
                    S_now = price
                    K = contract['strike']
                    sigma = contract['impliedVolatility']
                    r = 0.05
                    T_now = max(days_total, 0.5) / 365
                    T_sim = max(days_total - days_passed_sim, 0.5) / 365
                    
                    mid_entry = (contract['bid'] + contract['ask']) / 2
                    if np.isnan(mid_entry) or mid_entry <= 0: mid_entry = contract['lastPrice']
                    
                    val_sim = bs_price(target_price_sim, K, T_sim, r, sigma)
                    delta, gamma, theta = calculate_greeks(S_now, K, T_now, r, sigma)
                    
                    with col_pnl:
                        st.markdown("### Simulation P/L")
                        profit_per_share = val_sim - mid_entry
                        lots = int(risk_dollars / (mid_entry * 50))
                        total_pnl = profit_per_share * 100 * lots
                        
                        st.metric("Est. P/L", f"${total_pnl:,.2f}", f"{((val_sim/mid_entry)-1)*100:.1f}%")
                        st.write(f"Value at T+{days_passed_sim} days: **${val_sim:.2f}**")

                    with col_greeks:
                        st.markdown("### Today's Greeks")
                        st.write(f"🔹 **Delta:** {delta}")
                        st.write(f"🔸 **Theta:** -${abs(theta*100):.2f}/day")
                        st.write(f"🔺 **Gamma:** {gamma}")

                    with col_chart:
                        st.write("**30-Day History**")
                        hist = get_option_history(contract['contractSymbol'])
                        if not hist.empty:
                            st.line_chart(hist['Close'])

            render_strategy(tab1, cons_call)
            render_strategy(tab2, aggr_call)
