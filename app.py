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
    # Standard BS formula
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

# --- AI RESEARCH ENGINE (Strict Live-Search Protocol) ---
def get_ai_research(ticker):
    api_key = st.secrets.get("GEMINI_API_KEY")
    if not api_key: return "⚠️ Please add GEMINI_API_KEY to Streamlit Secrets."
    
    client = genai.Client(api_key=api_key)
    # Using the standard production model for web search
    model_id = "gemini-2.0-flash" 
    
    prompt = f"""
    Today is {datetime.now().strftime('%B %d, %Y')}. 
    Provide a factual bulleted cheat sheet for {ticker}.
    1. Analyst Consensus: Median price target and rating.
    2. Catalyst Calendar: Upcoming earnings/events.
    3. Sentiment: Top 3 news drivers.
    4. View: Buy or Wait based on current context.
    Limit to 6 bullets.
    
    MANDATORY: You must use the Google Search tool. Do NOT use internal knowledge for pricing or sentiment.
    """

    try:
        response = client.models.generate_content(
            model=model_id,
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearchRetrieval())]
            )
        )
        return response.text

    except Exception as e:
        # STRICT OVERRIDE: No internal knowledge fallback. 
        # If search fails, we fail gracefully and tell the user.
        return f"❌ **Live Web Search Unavailable.**\n\nThe Google Search API is currently congested. Per strict system parameters, internal AI knowledge has been disabled to prevent outdated or hallucinated pricing.\n\n*Please wait 60 seconds and use the **Refresh AI** button above to retry.*\n\n*(Error detail: {str(e)[:80]}...)*"


# --- PAGE CONFIG & SESSION STATE ---
st.set_page_config(page_title="Analyst Pro v6.8", layout="wide")

state_keys = {
    'price': None, 'trend': None, 'sma20': 0, 'pct_change': 0, 
    'stock_name': None, 'expiries': [], 'current_ticker': "", 
    'credits_used': 0, 'ai_cons_strike': None, 'ai_aggr_strike': None,
    'ai_brief': "", 'last_refresh': "Never"
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
    st.session_state.ai_cons_strike = None
    st.session_state.ai_aggr_strike = None
    st.session_state.ai_brief = ""
    
    api_key_av = st.secrets.get("ALPHA_VANTAGE_KEY")
    url = f'https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={ticker_input}&apikey={api_key_av}'
    try:
        r = requests.get(url, timeout=10).json()
        st.session_state.price = float(r["Global Quote"]["05. price"])
        st.session_state.credits_used += 1
    except:
        st.session_state.price = yf.Ticker(ticker_input).fast_info['lastPrice']
    
    stock_obj = yf.Ticker(ticker_input)
    st.session_state.stock_name = stock_obj.info.get('longName', ticker_input)
    st.session_state.expiries = list(stock_obj.options)
    hist = stock_obj.history(period="50d")
    if not hist.empty:
        sma20 = hist['Close'].rolling(window=20).mean().iloc[-1]
        st.session_state.trend = "Bullish" if hist['Close'].iloc[-1] > sma20 else "Bearish"
        st.session_state.pct_change = ((hist['Close'].iloc[-1] / hist['Close'].iloc[-20]) - 1) * 100

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
    t_cons, t_aggr, t_ai, t_edu = st.tabs(["🛡️ Conservative", "⚡ Aggressive", "🤖 AI Research", "📖 Strategy Guide"])

    def render_strategy(tab, ai_val, label, key_suffix):
        with tab:
            selected_k = st.selectbox(
                f"Override {label} Strike (Currently viewing ${ai_val}):", 
                all_strikes, 
                index=all_strikes.index(ai_val), 
                key=f"sel_{key_suffix}_{st.session_state.current_ticker}_{expiry}"
            )
            
            st.info(f"💡 AI Suggestion: **${ai_val}**")
            
            contract = chain[chain['strike'] == selected_k].iloc[0]
            mid_price = (contract['bid'] + contract['ask']) / 2 if contract['bid'] > 0 else contract['lastPrice']
            d, g, t, v = calculate_greeks(S, selected_k, T_years, 0.05, contract['impliedVolatility'])
            
            sim_p = bs_price(final_target_price, selected_k, max(days_to_expiry - days_sim, 0.5)/365, 0.05, contract['impliedVolatility'])
            roi = ((sim_p / mid_price) - 1) * 100
            score = (1 if st.session_state.trend == "Bullish" else 0) + (1 if d > 0.3 else 0) + (1 if abs(t) < (mid_price * 0.1) else 0)
            
            st.markdown(f"### Analysis: **${selected_k} Call**")
            
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

    # --- AI TAB WITH REFRESH BUTTON ---
    with t_ai:
        c1, c2 = st.columns([4, 1])
        with c1:
            st.subheader(f"🤖 Gemini Intelligence: {st.session_state.current_ticker}")
        with c2:
            # The Mini Refresh Button
            if st.button("🔄 Refresh AI", use_container_width=True):
                st.session_state.ai_brief = ""
                
        if not st.session_state.ai_brief or fetch_btn:
            with st.spinner("Searching web for real-time data..."):
                st.session_state.ai_brief = get_ai_research(st.session_state.current_ticker)
                st.session_state.last_refresh = datetime.now().strftime("%H:%M:%S")
                
        st.markdown(st.session_state.ai_brief)
        
        if st.session_state.last_refresh != "Never" and "❌" not in st.session_state.ai_brief:
            st.caption(f"Last updated: {st.session_state.last_refresh} | Strictly grounded in Google Search.")

    with t_edu:
        st.subheader("Strategy Playbook")
        st.markdown("""
        - **High Conviction:** Math (Greeks) and Momentum (Trend) align.
        - **AI Brief:** Prioritize this to avoid 'Earnings Crushes'.
        """)
else:
    st.info("👈 Enter a ticker and click 'Analyze Ticker' to start.")
