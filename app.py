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
    df['ema8'] = df['Close'].ewm(span=8, adjust=False).mean()
    df['ema20'] = df['Close'].ewm(span=20, adjust=False).mean()
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['hist'] = df['macd'] - df['signal']
    df['sma20'] = df['Close'].rolling(window=20).mean()
    df['std20'] = df['Close'].rolling(window=20).std()
    df['upper'] = df['sma20'] + (df['std20'] * 2)
    df['lower'] = df['sma20'] - (df['std20'] * 2)
    return df.iloc[-1], df.iloc[-2]

# --- AI RESEARCH ENGINE ---
def get_ai_research(ticker):
    api_key = st.secrets.get("GEMINI_API_KEY")
    if not api_key: return "⚠️ Please add GEMINI_API_KEY to Streamlit Secrets."
    client = genai.Client(api_key=api_key)
    model_id = "gemini-2.0-flash" 
    prompt = f"Live search {ticker} for Analyst Consensus, Catalysts, and News Sentiment for {datetime.now().strftime('%B %d, %Y')}."
    try:
        response = client.models.generate_content(
            model=model_id, contents=prompt,
            config=types.GenerateContentConfig(tools=[types.Tool(google_search=types.GoogleSearch())])
        )
        return response.text
    except Exception as e:
        return f"❌ AI Unavailable: {str(e)[:50]}"

# --- PAGE CONFIG ---
st.set_page_config(page_title="Analyst Pro v6.9", layout="wide")

state_keys = {
    'price': None, 'trend': None, 'sma20': 0, 'pct_change': 0, 
    'stock_name': None, 'expiries': [], 'current_ticker': "", 
    'credits_used': 0, 'ai_brief': "", 'last_refresh': "Never", 'hist_data': pd.DataFrame()
}
for key, default in state_keys.items():
    if key not in st.session_state: st.session_state[key] = default

# --- SIDEBAR ---
with st.sidebar:
    st.header("🎮 Control Center")
    ticker_input = st.text_input("Ticker:", "SHOP").upper()
    fetch_btn = st.button("🚀 Analyze Ticker")
    st.divider()
    st.header("🧪 Simulator")
    target_pct = st.slider("Target Price Change (%)", -30, 50, 0)
    days_sim = st.slider("Days in Future", 0, 30, 0)

if fetch_btn:
    st.session_state.current_ticker = ticker_input
    st.session_state.ai_brief = "" 
    try:
        stock_obj = yf.Ticker(ticker_input)
        hist = stock_obj.history(period="100d")
        if not hist.empty:
            st.session_state.hist_data = hist
            st.session_state.price = hist['Close'].iloc[-1]
            st.session_state.stock_name = stock_obj.info.get('longName', ticker_input)
            st.session_state.expiries = list(stock_obj.options)
            sma20_val = hist['Close'].rolling(window=20).mean().iloc[-1]
            st.session_state.trend = "Bullish" if st.session_state.price > sma20_val else "Bearish"
            st.session_state.pct_change = ((st.session_state.price / hist['Close'].iloc[-20]) - 1) * 100
    except Exception as e: st.error(f"Error: {e}")

# --- DASHBOARD ---
if st.session_state.price and st.session_state.expiries:
    S = st.session_state.price
    st.header(f"{st.session_state.stock_name} ({st.session_state.current_ticker})")
    
    t_cons, t_aggr, t_tech, t_ai, t_edu = st.tabs(["🛡️ Conservative", "⚡ Aggressive", "📊 Technicals", "🤖 AI Research", "📖 Strategy Guide"])

    # Strategy Render Function (Internal)
    def render_strategy(tab, ai_val, label, key_suffix, chain):
        with tab:
            all_strikes = sorted(chain['strike'].tolist())
            selected_k = st.selectbox(f"Strike:", all_strikes, index=all_strikes.index(ai_val), key=f"s_{key_suffix}")
            contract = chain[chain['strike'] == selected_k].iloc[0]
            mid = (contract['bid'] + contract['ask']) / 2 if contract['bid'] > 0 else contract['lastPrice']
            d, g, t, v = calculate_greeks(S, selected_k, T_years, 0.05, contract['impliedVolatility'])
            sim_p = bs_price(S*(1+target_pct/100), selected_k, max(days_to_expiry-days_sim, 0.5)/365, 0.05, contract['impliedVolatility'])
            roi = ((sim_p / mid) - 1) * 100
            st.metric("Expected ROI", f"{roi:.1f}%")
            st.write(f"Delta: {d} | Theta: -{abs(t):.2f}")

    # Tabs Implementation
    expiry = st.selectbox("Expiry:", st.session_state.expiries)
    days_to_expiry = (pd.to_datetime(expiry).date() - datetime.now().date()).days
    T_years = max(days_to_expiry, 0.5) / 365
    chain = yf.Ticker(st.session_state.current_ticker).option_chain(expiry).calls

    # sugg logic
    avg_iv = chain['impliedVolatility'].median()
    all_strikes = sorted(chain['strike'].tolist())
    cons_k = all_strikes[min(range(len(all_strikes)), key=lambda i: abs(all_strikes[i] - (S + (S*avg_iv*np.sqrt(T_years))*0.3)))]
    aggr_k = all_strikes[min(range(len(all_strikes)), key=lambda i: abs(all_strikes[i] - (S + (S*avg_iv*np.sqrt(T_years))*0.8)))]

    render_strategy(t_cons, cons_k, "Cons", "c", chain)
    render_strategy(t_aggr, aggr_k, "Aggr", "a", chain)

    with t_tech:
        df_tech = st.session_state.hist_data.copy()
        curr, prev = get_technicals(df_tech)
        
        st.subheader("Momentum & Volatility Health")
        c1, c2, c3 = st.columns(3)
        
        ema_status = "Bullish Cross" if curr['ema8'] > curr['ema20'] else "Bearish Separation"
        c1.metric("8/20 EMA Status", ema_status, f"{curr['ema8'] - curr['ema20']:.2f}")
        
        macd_dir = "Improving" if curr['hist'] > prev['hist'] else "Fading"
        c2.metric("MACD Momentum", macd_dir, f"{curr['hist']:.3f} hist")
        
        pos = "Upper Half" if S > curr['sma20'] else "Lower Half"
        c3.metric("Bollinger Position", pos, f"{((S - curr['lower'])/(curr['upper'] - curr['lower']))*100:.1f}%")

        st.line_chart(df_tech[['Close', 'ema8', 'ema20', 'upper', 'lower']])

        # --- FINAL RECOMMENDATION LOGIC ---
        st.divider()
        st.subheader("🏁 Final Technical Verdict")
        
        tech_score = 0
        reasons = []
        
        if curr['ema8'] > curr['ema20']: 
            tech_score += 1
            reasons.append("Short-term momentum is above the 20-day average.")
        if curr['hist'] > prev['hist']:
            tech_score += 1
            reasons.append("MACD histogram is rising (momentum accelerating).")
        if S > curr['sma20']:
            tech_score += 1
            reasons.append("Price is holding above the Middle Bollinger Band.")
            
        if tech_score >= 3:
            st.success(f"✅ **STRONGLY BULLISH:** All indicators aligned. High probability of upward continuation.")
        elif tech_score == 2:
            st.warning(f"⚠️ **NEUTRAL/CAUTION:** Technicals are mixed. Wait for the 8/20 EMA to confirm direction.")
        else:
            st.error(f"❌ **BEARISH / STAY AWAY:** Momentum is fading or price is breaking down. High risk for Calls.")
        
        with st.expander("Why this rating?"):
            for r in reasons: st.write(f"- {r}")

    with t_ai:
        if st.button("Refresh AI"): st.session_state.ai_brief = ""
        if not st.session_state.ai_brief: st.session_state.ai_brief = get_ai_research(st.session_state.current_ticker)
        st.markdown(st.session_state.ai_brief)

    with t_edu:
        st.subheader("📖 Technical Decoder Guide")
        st.markdown("""
        ### 1. 8/20 EMA Crossover
        * **Bullish Cross:** The 8-day line is above the 20-day. Buying pressure is fresh.
        * **Bearish Separation:** The 8-day is below the 20-day. The stock is in a "cool down" or crash phase. **Avoid calls.**
        
        ### 2. MACD Momentum
        * **Improving (e.g., -1.927):** Even if the value is negative, "Improving" means the selling is stopping. The "bars" on the histogram are getting shorter. This is often the first sign of a reversal.
        * **Fading:** Even if the stock is up, "Fading" means the buyers are getting tired. 
        
        ### 3. Bollinger Bands
        * **Upper Half:** The stock is strong. 
        * **Lower Half:** The stock is weak. 
        * **Touch Upper Band:** Be careful! The stock is "Overbought." It might pull back to the middle line soon.
        """)
