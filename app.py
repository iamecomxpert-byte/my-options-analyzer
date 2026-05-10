import streamlit as st
import yfinance as yf
import requests
import pandas as pd
from datetime import datetime
import numpy as np
from scipy.stats import norm
from google import genai
from google.genai import types

# --- CORE MATH ENGINES ---
def calculate_greeks(S, K, T, r, sigma, type="call"):
    if T <= 0 or sigma <= 0 or S <= 0: return 0.0, 0.0, 0.0, 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    delta = norm.cdf(d1) if type == "call" else norm.cdf(d1) - 1
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    theta = (- (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    vega = (S * norm.pdf(d1) * np.sqrt(T)) / 100
    return round(delta, 3), round(gamma, 4), round(theta, 3), round(vega, 3)

def bs_price(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0 or S <= 0: return max(0, S-K)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

# --- TECHNICAL ANALYSIS ENGINE ---
def get_technicals(df):
    # EMA 8 and 20
    df['ema8'] = df['Close'].ewm(span=8, adjust=False).mean()
    df['ema20'] = df['Close'].ewm(span=20, adjust=False).mean()
    
    # MACD (12, 26, 9)
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['hist'] = df['macd'] - df['signal']
    
    # Bollinger Bands (20, 2)
    df['sma20'] = df['Close'].rolling(window=20).mean()
    df['std20'] = df['Close'].rolling(window=20).std()
    df['upper'] = df['sma20'] + (df['std20'] * 2)
    df['lower'] = df['sma20'] - (df['std20'] * 2)
    
    return df.iloc[-1], df.iloc[-2] # Current and Previous for cross detection

# --- AI RESEARCH ENGINE (Google Search AI Grounding) ---
def get_ai_research(ticker):
    api_key = st.secrets.get("GEMINI_API_KEY")
    if not api_key: return "⚠️ Please add GEMINI_API_KEY to Streamlit Secrets."
    client = genai.Client(api_key=api_key)
    model_id = "gemini-2.0-flash" 
    
    prompt = f"""
    Perform a live web search for the stock ticker {ticker}. 
    Today is {datetime.now().strftime('%B %d, %Y')}.
    Provide a factual bulleted cheat sheet:
    1. Analyst Consensus: Current median price target and rating.
    2. Catalyst Calendar: Next earnings date and any upcoming investor days.
    3. Sentiment: Top 3 news drivers from the last 7 days.
    4. View: Factual 'Buy' or 'Wait' summary based on the latest analyst updates.
    MANDATORY: Use the Google Search tool for all data. Cite specific dates.
    """
    try:
        response = client.models.generate_content(
            model=model_id,
            contents=prompt,
            config=types.GenerateContentConfig(tools=[types.Tool(google_search=types.GoogleSearch())])
        )
        return response.text
    except Exception as e:
        if "429" in str(e):
            return "❌ **Quota Exhausted.** Google Search AI is rate-limited. Please wait 60s."
        return f"❌ **Search AI Unavailable.** (Detail: {str(e)[:60]}...)"

# --- PAGE CONFIG & SESSION STATE ---
st.set_page_config(page_title="Analyst Pro v6.9", layout="wide")

state_keys = {
    'price': None, 'trend': None, 'sma20': 0, 'pct_change': 0, 
    'stock_name': None, 'expiries': [], 'current_ticker': "", 
    'credits_used': 0, 'ai_cons_strike': None, 'ai_aggr_strike': None,
    'ai_brief': "", 'last_refresh': "Never", 'hist_data': pd.DataFrame()
}
for key, default in state_keys.items():
    if key not in st.session_state:
        st.session_state[key] = default

# --- SIDEBAR ---
with st.sidebar:
    st.header("🎮 Control Center")
    st.metric("API Credits", f"{st.session_state.credits_used} / 25")
    ticker_input = st.text_input("Ticker:", "SHOP").upper()
    fetch_btn = st.button("🚀 Analyze Ticker")
    st.divider()
    st.header("🧪 Simulator")
    target_pct = st.slider("Target Price Change (%)", -30, 50, 0, step=1)
    days_sim = st.slider("Days in Future", 0, 30, 0)

# --- DATA FETCHING ---
if fetch_btn:
    st.session_state.current_ticker = ticker_input
    st.session_state.ai_brief = "" 
    
    try:
        stock_obj = yf.Ticker(ticker_input)
        hist = stock_obj.history(period="100d")
        
        if hist.empty or 'Close' not in hist.columns:
            st.error(f"❌ No valid history found for {ticker_input}. The symbol may be incorrect or delisted.")
            st.session_state.price = None
        else:
            st.session_state.hist_data = hist
            st.session_state.price = hist['Close'].iloc[-1]
            st.session_state.stock_name = stock_obj.info.get('longName', ticker_input)
            st.session_state.expiries = list(stock_obj.options)
            
            # Calculate Summary Metrics
            sma20_val = hist['Close'].rolling(window=20).mean().iloc[-1]
            st.session_state.trend = "Bullish" if st.session_state.price > sma20_val else "Bearish"
            st.session_state.pct_change = ((st.session_state.price / hist['Close'].iloc[-20]) - 1) * 100
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")

# --- MAIN DASHBOARD ---
if st.session_state.price and st.session_state.expiries:
    S = st.session_state.price
    st.header(f"{st.session_state.stock_name} ({st.session_state.current_ticker})")
    
    col_p, col_t = st.columns(2)
    col_p.metric("Current Price", f"${S:.2f}")
    col_t.metric("20-Day Trend", st.session_state.trend, f"{st.session_state.pct_change:.1f}%")

    expiry = st.selectbox("Select Expiry Date:", st.session_state.expiries)
    days_to_expiry = (pd.to_datetime(expiry).date() - datetime.now().date()).days
    T_years = max(days_to_expiry, 0.5) / 365

    chain = yf.Ticker(st.session_state.current_ticker).option_chain(expiry).calls
    all_strikes = sorted(chain['strike'].tolist())
    avg_iv = chain['impliedVolatility'].median()
    expected_move = S * avg_iv * np.sqrt(T_years)

    st.session_state.ai_cons_strike = all_strikes[min(range(len(all_strikes)), key=lambda i: abs(all_strikes[i] - (S + expected_move*0.3)))]
    st.session_state.ai_aggr_strike = all_strikes[min(range(len(all_strikes)), key=lambda i: abs(all_strikes[i] - (S + expected_move*0.8)))]

    final_target_price = S * (1 + target_pct / 100)

    st.divider()
    t_cons, t_aggr, t_tech, t_ai, t_edu = st.tabs(["🛡️ Conservative", "⚡ Aggressive", "📊 Technicals", "🤖 AI Research", "📖 Strategy Guide"])

    def render_strategy(tab, ai_val, label, key_suffix):
        with tab:
            selected_k = st.selectbox(f"Override {label} Strike (Currently viewing ${ai_val}):", all_strikes, index=all_strikes.index(ai_val), key=f"sel_{key_suffix}_{st.session_state.current_ticker}_{expiry}")
            st.info(f"💡 AI Suggestion: **${ai_val}**")
            contract = chain[chain['strike'] == selected_k].iloc[0]
            mid_price = (contract['bid'] + contract['ask']) / 2 if contract['bid'] > 0 else contract['lastPrice']
            d, g, t, v = calculate_greeks(S, selected_k, T_years, 0.05, contract['impliedVolatility'])
            sim_p = bs_price(final_target_price, selected_k, max(days_to_expiry - days_sim, 0.5)/365, 0.05, contract['impliedVolatility'])
            roi = ((sim_p / mid_price) - 1) * 100
            score = (1 if st.session_state.trend == "Bullish" else 0) + (1 if d > 0.3 else 0) + (1 if abs(t) < (mid_price * 0.1) else 0)
            
            c1, c2, c3 = st.columns([1.5, 1.5, 2])
            with c1:
                if score >= 3: st.success("✅ HIGH CONVICTION BUY")
                elif score == 2: st.warning("⚠️ CAUTION: SETUP WEAK")
                else: st.error("❌ NO-BUY: POOR PROBABILITY")
                st.metric("Target Entry (Mid)", f"${mid_price:.2f}")
                st.metric("Simulated ROI", f"{roi:.1f}%")
            with c2:
                st.write(f"**Greeks**")
                st.write(f"Delta: `{d}` | Theta: `-{abs(t):.2f}`")
                st.write(f"Prob. ITM: `{d*100:.1f}%`主力")
            with c3:
                h = yf.Ticker(contract['contractSymbol']).history(period="1mo")
                if not h.empty: st.line_chart(h['Close'])

    render_strategy(t_cons, st.session_state.ai_cons_strike, "Conservative", "cons")
    render_strategy(t_aggr, st.session_state.ai_aggr_strike, "Aggressive", "aggr")

with t_tech:
        if not st.session_state.hist_data.empty and 'Close' in st.session_state.hist_data.columns:
            # 1. RUN THE CALCULATIONS AND SAVE TO A LOCAL VARIABLE
            # We pass a copy to get_technicals, and it returns the updated rows, 
            # but we need the FULL dataframe for the chart.
            df_tech = st.session_state.hist_data.copy()
            
            # Re-run calculations on the local df_tech so the columns exist for the chart
            # We call the engine but use the dataframe it modifies
            curr, prev = get_technicals(df_tech) 
            
            st.subheader("Momentum & Volatility Health")
            c1, c2, c3 = st.columns(3)
            
            # EMA CROSS
            ema_status = "Bullish Cross" if curr['ema8'] > curr['ema20'] else "Bearish Separation"
            c1.metric("8/20 EMA Status", ema_status, f"{curr['ema8'] - curr['ema20']:.2f} delta")
            if curr['ema8'] > curr['ema20'] and prev['ema8'] <= prev['ema20']:
                c1.success("🔥 JUST CROSSED BULLISH")
            
            # MACD
            macd_dir = "Improving" if curr['hist'] > prev['hist'] else "Fading"
            c2.metric("MACD Momentum", macd_dir, f"{curr['hist']:.3f} hist")
            if curr['macd'] > 0: c2.caption("Trend Battery: Positive")
            
            # BOLLINGER
            pos = "Upper Half" if S > curr['sma20'] else "Lower Half"
            c3.metric("Bollinger Position", pos, f"{((S - curr['lower'])/(curr['upper'] - curr['lower']))*100:.1f}% Band")
            if S > curr['upper']: c3.warning("⚠️ OVEREXTENDED (Above Upper Band)")

            st.divider()
            # 2. USE THE UPDATED LOCAL DATAFRAME FOR THE CHART
            # This ensures 'ema8', 'ema20', etc. actually exist in the index
            st.line_chart(df_tech[['Close', 'ema8', 'ema20', 'upper', 'lower']])
        else:
            st.warning("⚠️ Historical technical data is unavailable for this ticker.")

    with t_ai:
        c1, c2 = st.columns([4, 1])
        with c1: st.subheader(f"🌐 Search AI Intelligence: {st.session_state.current_ticker}")
        with c2:
            if st.button("🔄 Refresh AI", use_container_width=True):
                st.session_state.ai_brief = ""
                st.rerun()
        if not st.session_state.ai_brief:
            with st.spinner("Scanning the live web..."):
                st.session_state.ai_brief = get_ai_research(st.session_state.current_ticker)
                st.session_state.last_refresh = datetime.now().strftime("%H:%M:%S")
        st.markdown(st.session_state.ai_brief)

    with t_edu:
        st.subheader("📖 Strategy Playbook")
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.markdown("""
            #### ⚖️ High-Conviction Checklist
            - **Trend:** 20-Day SMA Bullish & 8EMA > 20EMA.
            - **MACD:** Look for a green, rising histogram.
            - **Bollinger:** Best entries occur when price bounces off the Middle Band.
            """)
        with col_g2:
            st.markdown("""
            #### 📉 Risk Management
            - **Exit Alarms:** Mandatory for stop-loss and profit targets.
            - **IV Crush:** Use AI tab to ensure no earnings occur during your trade.
            """)
else:
    st.info("👈 Enter a ticker and click 'Analyze Ticker' to start.")
