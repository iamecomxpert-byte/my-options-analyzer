import streamlit as st
import yfinance as yf
from datetime import datetime
import numpy as np

# Page Configuration
st.set_page_config(page_title="Pro Options Analyst", layout="wide", page_icon="🎯")

# --- SMART DATA FETCHING (CACHING DATA, NOT OBJECTS) ---
@st.cache_data(ttl=600)
def get_option_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        # We extract the data we need into a simple format
        price = stock.fast_info['lastPrice']
        expiries = stock.options
        return {"price": price, "expiries": expiries}
    except:
        return None

@st.cache_data(ttl=600)
def get_specific_chain(ticker, expiry):
    stock = yf.Ticker(ticker)
    opts = stock.option_chain(expiry)
    return opts.calls

# --- SIDEBAR: RISK MANAGEMENT ---
with st.sidebar:
    st.header("💰 Wallet & Risk")
    total_budget = st.number_input("Total Trading Budget ($)", min_value=100, value=5000, step=500)
    risk_percent = st.slider("Risk per Trade (%)", 1, 10, 2)
    risk_dollars = total_budget * (risk_percent / 100)
    st.info(f"Max Loss Allowed: **${risk_dollars:.2f}**")

# --- MAIN INTERFACE ---
st.title("🎯 Pro Options Analyst")
ticker_input = st.text_input("Enter Ticker:", "KTOS").upper()

if ticker_input:
    data = get_option_data(ticker_input)
    
    if not data:
        st.error("⚠️ Could not fetch data. Ticker might be invalid or Yahoo is rate-limiting. Try again in a minute.")
    else:
        price = data['price']
        expiries = data['expiries']
        
        st.metric(f"{ticker_input} Current Price", f"${price:.2f}")

        if not expiries:
            st.warning("No options available.")
        else:
            expiry = st.selectbox("Select Expiry:", expiries, index=min(3, len(expiries)-1))
            calls = get_specific_chain(ticker_input, expiry)

            # Strategy Selection
            itm_options = calls[calls['strike'] < price * 0.97]
            cons_call = itm_options.iloc[-1] if not itm_options.empty else calls.iloc[0]

            otm_options = calls[calls['strike'] > price * 1.03]
            aggr_call = otm_options.iloc[0] if not otm_options.empty else calls.iloc[-1]

            st.subheader("🚀 Recommendation Engine")
            col1, col2 = st.columns(2)

            def display_strategy(title, contract, risk_amt):
                bid, ask = contract['bid'], contract['ask']
                mid_price = (bid + ask) / 2 if (bid > 0 and ask > 0) else contract['lastPrice']
                cost_per_contract = mid_price * 100
                
                # Sizing: Risk 50% of the premium
                risk_per_contract = cost_per_contract * 0.5
                max_contracts = int(risk_amt / risk_per_contract) if risk_per_contract > 0 else 0

                st.markdown(f"### {title}")
                st.write(f"**Strike:** ${contract['strike']}")
                st.success(f"**Target Buy Price:** ${mid_price:.2f}")
                
                if max_contracts > 0:
                    st.metric("Recommended Size", f"{max_contracts} Lot(s)", f"Total: ${max_contracts * cost_per_contract:.2f}")
                else:
                    st.error(f"Budget too low. 1 lot risks ${risk_per_contract:.2f}.")
                
                st.write(f"🔴 Stop Loss: **${mid_price * 0.5:.2f}**")
                st.write(f"🟢 Take Profit: **${mid_price * 2.0:.2f}**")

            with col1:
                display_strategy("🛡️ Conservative", cons_call, risk_dollars)
            with col2:
                display_strategy("⚡ Aggressive", aggr_call, risk_dollars)

            # Probability Check
            st.divider()
            iv = aggr_call['impliedVolatility']
            days = (datetime.strptime(expiry, '%Y-%m-%d') - datetime.now()).days
            exp_move = price * iv * np.sqrt(max(days, 1) / 365)
            
            st.subheader("📊 Market Probability")
            st.write(f"Expected Move: **±${exp_move:.2f}**")
            if (aggr_call['strike'] - price) < exp_move:
                st.success("✅ Strike is within expected move.")
            else:
                st.error("⚠️ Strike is outside expected move (Low Prob).")

st.caption("Positions sized so that a 50% drop equals your Max Loss Allowed.")
