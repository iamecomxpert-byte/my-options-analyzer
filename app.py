import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# Page Config
st.set_page_config(page_title="Options Analyst", page_icon="🎯")

st.title("🎯 Options Analyst Pro")
st.write("Enter a ticker to find the best 'Buy the Dip' call options.")

ticker_input = st.text_input("Enter Ticker (e.g. KTOS, TSLA, AAPL):", "KTOS").upper()

if ticker_input:
    try:
        stock = yf.Ticker(ticker_input)
        current_price = stock.fast_info['lastPrice']
        
        st.metric(label=f"Current {ticker_input} Price", value=f"${current_price:.2f}")

        # 1. Get Expirations
        expiries = stock.options
        if not expiries:
            st.error("No options data found for this ticker.")
        else:
            # Select an expiry roughly 3-4 months out
            selected_expiry = st.selectbox("Select Expiration Date:", expiries, index=min(3, len(expiries)-1))
            
            # 2. Load the Call Options
            opts = stock.option_chain(selected_expiry)
            calls = opts.calls
            
            # 3. Filter Logic: Delta 0.50 (At The Money)
            # Since free data lacks Delta, we find the Strike closest to Current Price
            calls['Diff'] = abs(calls['strike'] - current_price)
            best_call = calls.sort_values('Diff').iloc[0]
            
            st.divider()
            st.subheader("💡 Recommended Contract")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Strike Price", f"${best_call['strike']}")
            col2.metric("Market Price (Ask)", f"${best_call['ask']}")
            
            breakeven = best_call['strike'] + best_call['ask']
            move_needed = ((breakeven / current_price) - 1) * 100
            col3.metric("Breakeven Price", f"${breakeven:.2f}", f"{move_needed:.1f}% Move")

            # 4. Analysis Logic
            st.info(f"**Analysis:** This option requires {ticker_input} to rise {move_needed:.1f}% by {selected_expiry} to be profitable at expiration. The Implied Volatility is **{best_call['impliedVolatility']*100:.1f}%**.")

    except Exception as e:
        st.error(f"Error fetching data: {e}")

st.caption("Data provided by Yahoo Finance (15m delay). Use for educational purposes.")
