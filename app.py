import streamlit as st
import yfinance as yf
import requests
import pandas as pd
from datetime import datetime
import numpy as np

# Page Config
st.set_page_config(page_title="Pro Options Analyst v3", layout="wide", page_icon="📈")

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
    except:
        return None, "Connection Error"

@st.cache_data(ttl=600)
def get_expiries(ticker):
    try:
        return list(yf.Ticker(ticker).options)
    except:
        return []

@st.cache_data(ttl=600)
def get_call_chain(ticker, expiry):
    try:
        # We ensure greeks are included by fetching the full chain
        return yf.Ticker(ticker).option_chain(expiry).calls
    except:
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_option_history(contract_symbol):
    try:
        contract = yf.Ticker(contract_symbol)
        return contract.history(period="1mo")
    except:
        return pd.DataFrame()

# --- SIDEBAR ---
with st.sidebar:
    st.header("💰 Risk Settings")
    total_budget = st.number_input("Total Budget ($)", value=5000)
    risk_percent = st.slider("Risk per Trade (%)", 1, 10, 2)
    risk_dollars = total_budget * (risk_percent / 100)
    st.divider()
    st.write("**Pro Tip:** Watch Theta on your 'Aggressive' trades. If it's high, don't hold the trade overnight.")

# --- MAIN APP ---
st.title("📈 Pro Options Analyst v3 (Greeks Edition)")

api_key = st.secrets.get("ALPHA_VANTAGE_KEY")
ticker = st.text_input("Enter Ticker:", "SHOP").upper()

if ticker and api_key:
    price, p_error = get_api_price(ticker, api_key)
    expiries = get_expiries(ticker)
    
    if price:
        st.metric(f"{ticker} Price", f"${price:.2f}")
        expiry = st.selectbox("Select Expiry:", expiries, index=min(2, len(expiries)-1))
        chain = get_call_chain(ticker, expiry)

        if not chain.empty:
            # Strategy selection
            itm = chain[chain['strike'] < price * 0.98]
            cons_call = itm.iloc[-1] if not itm.empty else chain.iloc[0]
            otm = chain[chain['strike'] > price * 1.05]
            aggr_call = otm.iloc[0] if not otm.empty else chain.iloc[-1]

            tab1, tab2 = st.tabs(["🛡️ Conservative Trade", "⚡ Aggressive Trade"])

            def render_strategy(tab, contract):
                with tab:
                    col_info, col_greeks, col_chart = st.columns([1, 1, 2])
                    
                    mid = (contract['bid'] + contract['ask']) / 2
                    if np.isnan(mid) or mid <= 0: mid = contract['lastPrice']
                    
                    with col_info:
                        st.markdown("### Trade Info")
                        st.write(f"**Strike:** ${contract['strike']}")
                        st.success(f"**Target Buy:** ${mid:.2f}")
                        lots = int(risk_dollars / (mid * 50))
                        st.metric("Recommended Size", f"{lots} Lots")
                        st.caption(f"Risk: Stop at ${mid*0.5:.2f}")

                    with col_greeks:
                        st.markdown("### The Greeks")
                        # Fetch Greeks if available, otherwise show N/A
                        delta = contract.get('delta', 'N/A')
                        theta = contract.get('theta', 'N/A')
                        gamma = contract.get('gamma', 'N/A')
                        
                        st.write(f"🔹 **Delta:** {delta}")
                        st.write(f"🔸 **Theta:** {theta}")
                        st.write(f"🔺 **Gamma:** {gamma}")
                        
                        if delta != 'N/A' and float(delta) > 0.7:
                            st.info("High Delta: Moves like the stock.")

                    with col_chart:
                        st.write("**30-Day Price Movement**")
                        hist = get_option_history(contract['contractSymbol'])
                        if not hist.empty:
                            st.line_chart(hist['Close'])
                        else:
                            st.info("No historical data available.")

            render_strategy(tab1, cons_call)
            render_strategy(tab2, aggr_call)
