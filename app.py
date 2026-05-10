import streamlit as st
import yfinance as yf
from datetime import datetime
import numpy as np
import time

# Page Configuration
st.set_page_config(page_title="Pro Options Analyst", layout="wide", page_icon="🎯")

# --- DATA FETCHING WITH CACHING & RETRY ---
@st.cache_data(ttl=600)  # Cache results for 10 minutes to avoid Rate Limits
def get_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        # Attempt to access a property to trigger the API call
        _ = stock.fast_info['lastPrice']
        return stock
    except Exception as e:
        return None

# --- SIDEBAR: RISK MANAGEMENT ---
with st.sidebar:
    st.header("💰 Wallet & Risk")
    total_budget = st.number_input("Total Trading Budget ($)", min_value=100, value=5000, step=500)
    risk_percent = st.slider("Risk per Trade (%)", 1, 10, 2)
    
    risk_dollars = total_budget * (risk_percent / 100)
    st.info(f"Max Loss Allowed: **${risk_dollars:.2f}**")
    st.caption("This is your 'Safety Stop'. If you lose this much, exit the trade.")

# --- MAIN INTERFACE ---
st.title("🎯 Pro Options Analyst")
ticker_input = st.text_input("Enter Ticker:", "KTOS").upper()

if ticker_input:
    stock = get_stock_data(ticker_input)
    
    if stock is None:
        st.error("⚠️ Rate Limited or Invalid Ticker. Yahoo Finance is busy. Please wait 1-2 minutes and refresh.")
    else:
        try:
            # Price logic with fallbacks
            try:
                price = stock.fast_info['lastPrice']
            except:
                price = stock.history(period="1d")['Close'].iloc[-1]
                
            st.metric(f"{ticker_input} Current Price", f"${price:.2f}")

            expiries = stock.options
            if not expiries:
                st.warning("No options data available for this ticker.")
            else:
                expiry = st.selectbox("Select Expiration Date:", expiries, index=min(3, len(expiries)-1))
                
                # Fetch options chain
                opts = stock.option_chain(expiry)
                calls = opts.calls

                # Selection Logic
                itm_options = calls[calls['strike'] < price * 0.97]
                cons_call = itm_options.iloc[-1] if not itm_options.empty else calls.iloc[0]

                otm_options = calls[calls['strike'] > price * 1.03]
                aggr_call = otm_options.iloc[0] if not otm_options.empty else calls.iloc[-1]

                st.subheader("🚀 Recommendation Engine")
                col1, col2 = st.columns(2)

                def display_strategy(title, contract, risk_amt):
                    bid, ask = contract['bid'], contract['ask']
                    # Handle zero/NaN bids during off-market hours
                    mid_price = (bid + ask) / 2 if (bid > 0 and ask > 0) else contract['lastPrice']
                    cost_per_contract = mid_price * 100
                    
                    # Risk-based sizing: 1 contract risks (Cost * 50%)
                    risk_per_contract = cost_per_contract * 0.5
                    max_contracts = int(risk_amt / risk_per_contract) if risk_per_contract > 0 else 0

                    st.markdown(f"### {title}")
                    st.write(f"**Strike Price:** ${contract['strike']}")
                    st.success(f"**Target Buy Price:** ${mid_price:.2f}")
                    
                    if max_contracts > 0:
                        st.metric("Recommended Size", f"{max_contracts} Lot(s)", f"Total Cost: ${max_contracts * cost_per_contract:.2f}")
                    else:
                        st.error(f"Budget too low. 1 lot risks ${risk_per_contract:.2f}, but your max allowed is ${risk_amt:.2f}.")
                    
                    st.divider()
                    st.write("**Exit Strategy:**")
                    st.write(f"🔴 Stop Loss: **${mid_price * 0.5:.2f}**")
                    st.write(f"🟢 Take Profit: **${mid_price * 2.0:.2f}**")

                with col1:
                    display_strategy("🛡️ Conservative", cons_call, risk_dollars)
                with col2:
                    display_strategy("⚡ Aggressive", aggr_call, risk_dollars)

                # Probability Logic
                st.divider()
                st.subheader("📊 Market Probability")
                iv = aggr_call['impliedVolatility']
                days = (datetime.strptime(expiry, '%Y-%m-%d') - datetime.now()).days
                exp_move = price * iv * np.sqrt(max(days, 1) / 365)
                
                st.write(f"Market expected move: **±${exp_move:.2f}**")
                if (aggr_call['strike'] - price) < exp_move:
                    st.success("✅ Strike is within the expected move.")
                else:
                    st.error("⚠️ Strike is outside the expected move (Low Probability).")

        except Exception as e:
            st.error(f"Data error: {e}. Try a different ticker or expiry.")

st.caption("Note: Every 1 contract covers 100 shares. 'Cost' is Price x 100.")
