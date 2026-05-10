import streamlit as st
import yfinance as yf
import requests
import pandas as pd
from datetime import datetime
import numpy as np

# Page Config
st.set_page_config(page_title="Hybrid Options Analyst", layout="wide", page_icon="🎯")

# --- DATA FETCHING (ALPHA VANTAGE FOR PRICE) ---
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

# --- DATA FETCHING (YFINANCE FOR OPTIONS) ---
@st.cache_data(ttl=600)
def get_options_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        # Using a simple data pull to verify the ticker works
        expiries = stock.options
        return stock, expiries
    except:
        return None, None

# --- SIDEBAR: RISK MANAGEMENT ---
with st.sidebar:
    st.header("💰 Wallet & Risk")
    total_budget = st.number_input("Total Budget ($)", value=5000, step=500)
    risk_percent = st.slider("Risk per Trade (%)", 1, 10, 2)
    risk_dollars = total_budget * (risk_percent / 100)
    st.info(f"Max Loss Allowed: **${risk_dollars:.2f}**")

# --- MAIN APP ---
st.title("🎯 Hybrid Options Analyst")

# Pull key from Secrets
api_key = st.secrets.get("ALPHA_VANTAGE_KEY")
if not api_key:
    st.error("Please add ALPHA_VANTAGE_KEY to your Streamlit Secrets.")
    st.stop()

ticker = st.text_input("Enter Ticker:", "SHOP").upper()

if ticker:
    # 1. Fetch Data
    price, p_error = get_api_price(ticker, api_key)
    stock_obj, expiries = get_options_data(ticker)
    
    if p_error or not price:
        st.error(f"Price Error: {p_error}. Try again in 1 minute.")
    elif not expiries:
        st.warning("Options data currently unavailable for this ticker.")
    else:
        st.metric(f"{ticker} Current Price", f"${price:.2f}")
        
        expiry = st.selectbox("Select Expiry:", expiries, index=min(2, len(expiries)-1))
        chain = stock_obj.option_chain(expiry).calls

        # Strategy Logic: Conservative (ITM) vs Aggressive (OTM)
        itm_options = chain[chain['strike'] < price * 0.98]
        cons_call = itm_options.iloc[-1] if not itm_options.empty else chain.iloc[0]

        otm_options = chain[chain['strike'] > price * 1.05]
        aggr_call = otm_options.iloc[0] if not otm_options.empty else chain.iloc[-1]

        col1, col2 = st.columns(2)

        def display_box(title, contract, risk_amt):
            mid = (contract['bid'] + contract['ask']) / 2
            if np.isnan(mid) or mid <= 0: mid = contract['lastPrice']
            
            cost = mid * 100
            # Sizing based on a 50% stop-loss threshold
            max_lots = int(risk_amt / (cost * 0.5)) if cost > 0 else 0

            st.markdown(f"### {title}")
            st.write(f"**Strike:** ${contract['strike']}")
            st.success(f"**Target Buy Price:** ${mid:.2f}")
            
            if max_lots > 0:
                st.metric("Recommended Size", f"{max_lots} Lot(s)", f"Total Cost: ${max_lots*cost:.2f}")
            else:
                st.error(f"Budget too small. 1 lot risks ${(cost*0.5):.2f}.")
            
            st.write(f"🔴 Stop Loss: **${mid * 0.5:.2f}**")
            st.write(f"🟢 Take Profit: **${mid * 2.0:.2f}**")

        with col1: display_box("🛡️ Conservative", cons_call, risk_dollars)
        with col2: display_box("⚡ Aggressive", aggr_call, risk_dollars)

        # 4. Market Probability (Expected Move)
        st.divider()
        iv = aggr_call['impliedVolatility']
        days = (datetime.strptime(expiry, '%Y-%m-%d') - datetime.now()).days
        exp_move = price * iv * np.sqrt(max(days, 1) / 365)
        
        st.subheader("📊 Market Probability Check")
        st.write(f"Expected Move: **±${exp_move:.2f}**")
        if (aggr_call['strike'] - price) < exp_move:
            st.success("✅ Strike is within expected move. High statistical probability.")
        else:
            st.error("⚠️ Strike is outside expected move. Low statistical probability.")
