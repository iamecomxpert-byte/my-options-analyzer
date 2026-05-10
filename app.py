import streamlit as st
import yfinance as yf
from datetime import datetime
import numpy as np

# Page Configuration
st.set_page_config(page_title="Pro Options Analyst", layout="wide", page_icon="🎯")

# --- SIDEBAR: RISK MANAGEMENT ---
with st.sidebar:
    st.header("💰 Wallet & Risk")
    total_budget = st.number_input("Total Trading Budget ($)", min_value=100, value=5000, step=500)
    risk_percent = st.slider("Risk per Trade (%)", 1, 10, 2)
    
    risk_dollars = total_budget * (risk_percent / 100)
    st.info(f"Max Loss Allowed: **${risk_dollars:.2f}**")
    st.caption("Risk is calculated as 50% loss of the position value.")

# --- MAIN INTERFACE ---
st.title("🎯 Pro Options Analyst")
ticker = st.text_input("Enter Ticker (e.g., KTOS, SOXL, NVDA):", "KTOS").upper()

if ticker:
    try:
        stock = yf.Ticker(ticker)
        # Use fast_info for real-time price; fallback to regular info if needed
        try:
            price = stock.fast_info['lastPrice']
        except:
            price = stock.history(period="1d")['Close'].iloc[-1]
            
        st.metric(f"{ticker} Current Price", f"${price:.2f}")

        expiries = stock.options
        if not expiries:
            st.warning("No options available for this ticker.")
        else:
            expiry = st.selectbox("Select Expiration Date:", expiries, index=min(3, len(expiries)-1))
            
            # Fetch the options chain
            opts = stock.option_chain(expiry)
            calls = opts.calls

            # --- CRASH-PROOF SELECTION LOGIC ---
            # Conservative: Looking for ITM (Strike < Price)
            itm_options = calls[calls['strike'] < price * 0.97]
            if not itm_options.empty:
                cons_call = itm_options.iloc[-1]
            else:
                cons_call = calls.iloc[0] # Take lowest strike available

            # Aggressive: Looking for OTM (Strike > Price)
            otm_options = calls[calls['strike'] > price * 1.03]
            if not otm_options.empty:
                aggr_call = otm_options.iloc[0]
            else:
                aggr_call = calls.iloc[-1] # Take highest strike available

            st.subheader("🚀 Recommendation Engine")
            col1, col2 = st.columns(2)

            def display_strategy(title, contract, risk_amt):
                # Calculate Mid Price (Mark) with fallback for closed markets
                bid, ask = contract['bid'], contract['ask']
                mid_price = (bid + ask) / 2 if (bid > 0 and ask > 0) else contract['lastPrice']
                
                cost_per_contract = mid_price * 100
                
                # Position Sizing based on 50% stop loss
                # (Risk Dollars / (Cost * 0.5))
                if cost_per_contract > 0:
                    max_contracts = int(risk_amt / (cost_per_contract * 0.5))
                else:
                    max_contracts = 0

                st.markdown(f"### {title}")
                st.write(f"**Strike Price:** ${contract['strike']}")
                st.success(f"**Target Buy Price:** ${mid_price:.2f} (Limit Order)")
                
                if max_contracts > 0:
                    st.metric("Recommended Size", f"{max_contracts} Contracts", f"Total Cost: ${max_contracts * cost_per_contract:.2f}")
                else:
                    st.error("Budget/Risk too low for this contract.")
                
                st.divider()
                st.write("**🛡️ Exit Strategy (Broker's Logic):**")
                st.write(f"🔴 **Stop Loss:** Exit if option price hits **${mid_price * 0.5:.2f}**")
                st.write(f"🟢 **Take Profit:** Exit if option price hits **${mid_price * 2.0:.2f}**")

            with col1:
                display_strategy("🛡️ Conservative (Safe)", cons_call, risk_dollars)

            with col2:
                display_strategy("⚡ Aggressive (Growth)", aggr_call, risk_dollars)

            # --- MARKET PROBABILITY ---
            st.divider()
            st.subheader("📊 Market Probability Check")
            iv = aggr_call['impliedVolatility']
            days = (datetime.strptime(expiry, '%Y-%m-%d') - datetime.now()).days
            # Calculate Expected Move
            exp_move = price * iv * np.sqrt(max(days, 1) / 365)
            
            st.write(f"The market expects a move of **±${exp_move:.2f}** by expiry.")
            if (aggr_call['strike'] - price) < exp_move:
                st.success("✅ **Mathematically Sound:** The target strike is within the expected move.")
            else:
                st.error("⚠️ **Statistical Warning:** Strike is outside the expected move. High probability of expiring worthless.")

    except Exception as e:
        st.error(f"Waiting for market data or invalid ticker. Error: {e}")

st.caption("Real-time data via Yahoo Finance. Position sizing assumes a 50% stop-loss threshold.")
