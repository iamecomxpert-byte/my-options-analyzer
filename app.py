import streamlit as st
import yfinance as yf
from datetime import datetime
import numpy as np

st.set_page_config(page_title="Pro Options Analyst", layout="wide")

# --- SIDEBAR: RISK MANAGEMENT ---
with st.sidebar:
    st.header("💰 Wallet & Risk")
    total_budget = st.number_input("Total Trading Budget ($)", min_value=100, value=5000, step=500)
    risk_percent = st.slider("Risk per Trade (%)", 1, 10, 2)
    
    risk_dollars = total_budget * (risk_percent / 100)
    st.info(f"Max Loss Allowed: **${risk_dollars:.2f}**")

# --- MAIN INTERFACE ---
st.title("🎯 Pro Options Analyst")
ticker = st.text_input("Enter Ticker:", "KTOS").upper()

if ticker:
    stock = yf.Ticker(ticker)
    price = stock.fast_info['lastPrice']
    st.metric(f"{ticker} Price", f"${price:.2f}")

    expiries = stock.options
    expiry = st.selectbox("Select Expiry:", expiries, index=min(3, len(expiries)-1))
    calls = stock.option_chain(expiry).calls
    
    # Logic: Conservative (ITM) vs Aggressive (OTM)
    cons_call = calls[calls['strike'] < price * 0.95].iloc[-1]
    aggr_call = calls[calls['strike'] > price * 1.05].iloc[0]

    st.subheader("🚀 Recommendation Engine")
    col1, col2 = st.columns(2)

    def display_strategy(title, contract, risk_amt, css_class):
        mid_price = (contract['bid'] + contract['ask']) / 2
        # One contract covers 100 shares, so cost = mid_price * 100
        cost_per_contract = mid_price * 100
        
        # Position Sizing: How many contracts can we afford if we lose 50%?
        # If we exit at a 50% loss, the loss is cost_per_contract * 0.5
        max_contracts = int(risk_amt / (cost_per_contract * 0.5)) if cost_per_contract > 0 else 0
        
        st.markdown(f"### {title}")
        st.write(f"**Strike:** ${contract['strike']}")
        st.success(f"**Target Buy Price:** ${mid_price:.2f} (Limit Order)")
        
        if max_contracts > 0:
            st.metric("Recommended Size", f"{max_contracts} Contracts", f"Total: ${max_contracts * cost_per_contract:.2f}")
        else:
            st.error("Budget too small for this contract.")
            
        st.divider()
        st.write("**Exit Strategy:**")
        st.write(f"🔴 Stop Loss (Options Price): **${mid_price * 0.5:.2f}**")
        st.write(f"🟢 Take Profit (Options Price): **${mid_price * 2.0:.2f}**")

    with col1:
        display_strategy("🛡️ Conservative", cons_call, risk_dollars, "info")

    with col2:
        display_strategy("⚡ Aggressive", aggr_call, risk_dollars, "warning")

    # --- EXPECTED MOVE PROBABILITY ---
    st.divider()
    iv = aggr_call['impliedVolatility']
    days = (datetime.strptime(expiry, '%Y-%m-%d') - datetime.now()).days
    exp_move = price * iv * np.sqrt(days / 365)
    
    st.subheader("📊 Market Probability")
    st.write(f"Expected Move: **±${exp_move:.2f}**")
    if (aggr_call['strike'] - price) < exp_move:
        st.success("✅ Strike is mathematically 'reachable' based on current volatility.")
    else:
        st.error("⚠️ Strike is 'unreachable' based on market expectations. High risk of $0.")
