import streamlit as st
import requests
import pandas as pd
from datetime import datetime
import numpy as np

# Page Config
st.set_page_config(page_title="Alpha Options Pro", layout="wide", page_icon="🎯")

# --- DATA FETCHING (ALPHA VANTAGE API) ---
@st.cache_data(ttl=3600) # Cache for 1 hour to save your 25 daily requests
def get_options_data(ticker, api_key):
    # Function 2026: HISTORICAL_OPTIONS provides the full chain
    url = f'https://www.alphavantage.co/query?function=HISTORICAL_OPTIONS&symbol={ticker}&apikey={api_key}'
    try:
        response = requests.get(url)
        data = response.json()
        
        if "data" not in data:
            return None, "API Error: Check if ticker is valid or key is correct."
        
        df = pd.DataFrame(data['data'])
        # Convert numeric columns
        cols = ['strike', 'bid', 'ask', 'underlying_price', 'implied_volatility']
        for col in cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df, None
    except Exception as e:
        return None, str(e)

# --- SIDEBAR: RISK MANAGEMENT ---
with st.sidebar:
    st.header("💰 Wallet & Risk")
    total_budget = st.number_input("Total Budget ($)", value=5000, step=500)
    risk_percent = st.slider("Risk per Trade (%)", 1, 10, 2)
    risk_dollars = total_budget * (risk_percent / 100)
    st.info(f"Max Loss Allowed: **${risk_dollars:.2f}**")

# --- MAIN APP ---
st.title("🎯 Alpha Options Analyst")

# Pull key from Streamlit Secrets
try:
    api_key = st.secrets["ALPHA_VANTAGE_KEY"]
except:
    st.error("API Key not found in Secrets. Please add ALPHA_VANTAGE_KEY to your Streamlit settings.")
    st.stop()

ticker = st.text_input("Enter Ticker:", "SHOP").upper()

if ticker:
    df, error = get_options_data(ticker, api_key)
    
    if error:
        st.error(error)
    else:
        # 1. Get Current Context
        current_price = df.iloc[0]['underlying_price']
        st.metric(f"{ticker} Current Price", f"${current_price:.2f}")

        # 2. Filter for upcoming Monthly Expiry (e.g., 30-45 days out)
        df['expiration'] = pd.to_datetime(df['expiration'])
        expiries = sorted(df['expiration'].unique())
        # Pick an expiry at least 30 days out
        target_date = datetime.now() + pd.Timedelta(days=30)
        selected_expiry = next((d for d in expiries if d > target_date), expiries[-1])
        
        expiry_str = selected_expiry.strftime('%Y-%m-%d')
        st.write(f"Analyzing Expiry: **{expiry_str}**")
        
        # Filter chain for selected expiry and Calls
        chain = df[(df['expiration'] == selected_expiry) & (df['type'] == 'call')]

        # 3. Strategy Logic
        # Conservative: Strike ~3% below current price
        cons_chain = chain[chain['strike'] < current_price * 0.97]
        cons_call = cons_chain.iloc[-1] if not cons_chain.empty else chain.iloc[0]

        # Aggressive: Strike ~5% above current price
        aggr_chain = chain[chain['strike'] > current_price * 1.05]
        aggr_call = aggr_chain.iloc[0] if not aggr_chain.empty else chain.iloc[-1]

        col1, col2 = st.columns(2)

        def display_box(title, contract, risk_amt):
            mid = (contract['bid'] + contract['ask']) / 2
            cost = mid * 100
            # Sizing based on 50% stop-loss
            max_lots = int(risk_amt / (cost * 0.5)) if cost > 0 else 0

            st.markdown(f"### {title}")
            st.write(f"**Strike:** ${contract['strike']}")
            st.success(f"**Target Buy:** ${mid:.2f}")
            
            if max_lots > 0:
                st.metric("Size", f"{max_lots} Lots", f"Total: ${max_lots*cost:.2f}")
            else:
                st.error("Too expensive for risk profile.")
            
            st.write(f"🔴 Stop Loss: **${mid * 0.5:.2f}**")
            st.write(f"🟢 Take Profit: **${mid * 2.0:.2f}**")

        with col1: display_box("🛡️ Conservative", cons_call, risk_dollars)
        with col2: display_box("⚡ Aggressive", aggr_call, risk_dollars)

        # 4. Probability Check
        st.divider()
        iv = aggr_call['implied_volatility']
        days = (selected_expiry - datetime.now()).days
        exp_move = current_price * iv * np.sqrt(max(days, 1) / 365)
        
        st.subheader("📊 Market Probability")
        st.write(f"Market expected move: **±${exp_move:.2f}**")
        if (aggr_call['strike'] - current_price) < exp_move:
            st.success("✅ Strike is within expected move.")
        else:
            st.error("⚠️ Strike is outside expected move.")
