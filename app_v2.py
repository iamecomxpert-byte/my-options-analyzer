import streamlit as st
import yfinance as yf
import requests
import pandas as pd
from datetime import datetime
import numpy as np
from scipy.stats import norm
from google import genai
from google.genai import types

# --- CORE MATH & OPTIONS QUANT ENGINES ---
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

def calculate_p_touch(S, K, T, sigma):
    """Calculates the probability of touching the strike price before expiration."""
    if T <= 0 or sigma <= 0 or S <= 0: return 0.0
    # Dynamic approximation under risk-neutral assumption (double the probability of expiring ITM)
    d1 = (np.log(S / K) + (0.05 + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    p_itm = norm.cdf(d1) if S < K else 1.0 - norm.cdf(d1)
    p_touch = min(p_itm * 2.0, 0.99)
    return round(p_touch, 3)

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
st.set_page_config(page_title="Analyst Pro Options Suite", layout="wide")

state_keys = {
    'price': None, 'trend': None, 'sma20': 0, 'pct_change': 0, 
    'stock_name': None, 'expiries': [], 'current_ticker': "", 
    'credits_used': 0, 'ai_brief': "", 'last_refresh': "Never", 'hist_data': pd.DataFrame()
}
for key, default in state_keys.items():
    if key not in st.session_state:
        st.session_state[key] = default

# --- SIDEBAR ---
with st.sidebar:
    st.header("🎮 Control Center")
    ticker_input = st.text_input("Ticker:", "SHOP").upper()
    fetch_btn = st.button("🚀 Analyze Options Structure")
    st.divider()
    st.header("🧪 Exit & Hold Adjuster")
    profit_target_pct = st.slider("Target Option Profit Booking (%)", 10, 150, 40, step=5)
    stop_loss_pct = st.slider("Max Stop Loss (%)", 10, 100, 30, step=5)

# --- DATA FETCHING ---
if fetch_btn:
    st.session_state.current_ticker = ticker_input
    st.session_state.ai_brief = "" 
    
    try:
        stock_obj = yf.Ticker(ticker_input)
        hist = stock_obj.history(period="100d")
        
        if hist.empty or 'Close' not in hist.columns:
            st.error(f"❌ No valid history found for {ticker_input}. Symbol may be incorrect.")
            st.session_state.price = None
        else:
            st.session_state.hist_data = hist
            st.session_state.price = hist['Close'].iloc[-1]
            st.session_state.stock_name = stock_obj.info.get('longName', ticker_input)
            
            # CRITICAL RULES FILTER: Expiries must be 2+ months out (>= 60 days)
            today = datetime.now().date()
            valid_expiries = []
            for exp in stock_obj.options:
                days_diff = (pd.to_datetime(exp).date() - today).days
                if days_diff >= 60:
                    valid_expiries.append(exp)
            
            st.session_state.expiries = valid_expiries
            
            if not valid_expiries:
                st.error("❌ No options found with a 2+ month (>= 60 days) expiry window.")
            
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
    col_p.metric("Current Underlying Price", f"${S:.2f}")
    col_t.metric("20-Day Baseline Trend", st.session_state.trend, f"{st.session_state.pct_change:.1f}%")

    # Filter selection box to present only valid 2+ month expiries
    expiry = st.selectbox("Select Filtered Expiry Date (Minimum 60 Days required):", st.session_state.expiries)
    days_to_expiry = (pd.to_datetime(expiry).date() - datetime.now().date()).days
    T_years = days_to_expiry / 365

    # Fetch and configure Options Table 
    chain = yf.Ticker(st.session_state.current_ticker).option_chain(expiry).calls
    
    # Precompute structural data elements
    tech_score = 0
    if not st.session_state.hist_data.empty:
        df_tech = st.session_state.hist_data.copy()
        curr, prev = get_technicals(df_tech)
        if curr['ema8'] > curr['ema20']: tech_score += 1
        if curr['hist'] > prev['hist']: tech_score += 1
        if S > curr['sma20']: tech_score += 1

    st.divider()
    t_cons, t_aggr, t_spec, t_tech, t_ai = st.tabs(["🛡️ Conservative Buy", "⚡ Aggressive Buy", "🎰 Speculative Buy", "📊 Technical Analysis", "🤖 AI Grounding"])

    def process_tier_strategy(tab_component, delta_min, delta_max, tier_label):
        with tab_component:
            tier_contracts = []
            
            # Loop contract architecture to check mathematically suitable targets
            for index, row in chain.iterrows():
                mid = (row['bid'] + row['ask']) / 2 if row['bid'] > 0 else row['lastPrice']
                if mid <= 0 or row['impliedVolatility'] <= 0: continue
                
                d, g, t, v = calculate_greeks(S, row['strike'], T_years, 0.05, row['impliedVolatility'])
                
                if delta_min <= d <= delta_max:
                    p_touch = calculate_p_touch(S, row['strike'], T_years, row['impliedVolatility'])
                    
                    # Expected value equation execution
                    pot_profit = mid * (1 + profit_target_pct / 100)
                    pot_loss = mid * (stop_loss_pct / 100)
                    ev = (p_touch * pot_profit) - ((1 - p_touch) * pot_loss)
                    
                    tier_contracts.append({
                        'strike': row['strike'], 'mid': mid, 'delta': d, 'theta': t, 
                        'iv': row['impliedVolatility'], 'p_touch': p_touch, 'ev': ev, 'symbol': row['contractSymbol']
                    })
            
            if not tier_contracts:
                st.error(f"No contracts on this expiry date match the requested Delta parameters for a {tier_label} strategy.")
                return

            df_tier = pd.DataFrame(tier_contracts).sort_values(by='ev', ascending=False)
            optimal_contract = df_tier.iloc[0]
            
            # User choice override selection matrix
            strike_list = sorted(df_tier['strike'].tolist())
            selected_k = st.selectbox(f"Select Available {tier_label} Strike:", strike_list, index=strike_list.index(optimal_contract['strike']), key=f"sel_{tier_label}_{expiry}")
            
            chosen = df_tier[df_tier['strike'] == selected_k].iloc[0]
            
            # Quantitative Synthesis - Composite Trading Score (CTS) Calculations
            normalized_ev = 1 if chosen['ev'] > 0 else 0
            cts_score = int(((chosen['delta'] * 0.4) + (chosen['p_touch'] * 0.4) + (tech_score / 3.0 * 0.2)) * 100)
            
            # Display metrics columns
            c1, c2, c3 = st.columns([1.5, 1.5, 2])
            with c1:
                if cts_score >= 55 and chosen['ev'] > 0:
                    st.success("🎯 RECOMMENDATION: STRONG BUY SETUP")
                elif cts_score >= 40 and chosen['ev'] > 0:
                    st.warning("⚖️ RECOMMENDATION: CAUTIOUS / WATCH")
                else:
                    st.error("🛑 RECOMMENDATION: AVOID / NEGATIVE EXPECTANCY")
                    
                st.metric("Composite Trade Score", f"{cts_score}/100")
                st.metric("Target Entry (Midpoint Price)", f"${chosen['mid']:.2f}")
                
            with c2:
                target_exit_p = chosen['mid'] * (1 + profit_target_pct / 100)
                stop_loss_p = chosen['mid'] * (1 - stop_loss_pct / 100)
                
                st.metric("Take Profit Target Price", f"${target_exit_p:.2f}")
                st.metric("Stop Loss Level", f"${stop_loss_p:.2f}")
                
                # Calculation metrics for recommended hold timeframe bounds
                theta_decay_cutoff_days = min(int(days_to_expiry * 0.4), 45)
                st.write(f"⏱️ **Max Hold Target:** `{theta_decay_cutoff_days} days` *(Exit before accelerated Theta curve)*")

            with c3:
                st.write("**Stochastic & Valuation Profiles**")
                st.write(f"- Stat Probability of Success ($P_{{\\text{{ITM}}}}$ Delta Proxy): `{chosen['delta'] * 100:.1f}%`主力")
                st.write(f"- Path Probability of Touching Strike: `{chosen['p_touch'] * 100:.1f}%`")
                st.write(f"- Pure Mathematical Expectancy ($E[X]$ value): `{chosen['ev']:.3f}`")
                st.write(f"- Implied Volatility (IV): `{chosen['iv']*100:.1f}%` | Daily Theta: `-{abs(chosen['theta']):.3f}`")
                
                h_chart = yf.Ticker(chosen['symbol']).history(period="1mo")
                if not h_chart.empty: 
                    st.caption("Contract Price History (1 Month)")
                    st.line_chart(h_chart['Close'])

    # Map the strategy profiles systematically based on your exact mathematical definitions
    process_tier_strategy(t_cons, 0.50, 0.60, "Conservative")
    process_tier_strategy(t_aggr, 0.40, 0.49, "Aggressive")
    process_tier_strategy(t_spec, 0.30, 0.39, "Speculative")

    with t_tech:
        if not st.session_state.hist_data.empty and 'Close' in st.session_state.hist_data.columns:
            df_tech = st.session_state.hist_data.copy()
            curr, prev = get_technicals(df_tech)
            
            st.subheader("Momentum & Volatility Health")
            c1, c2, c3 = st.columns(3)
            
            ema_status = "Bullish Cross" if curr['ema8'] > curr['ema20'] else "Bearish Separation"
            c1.metric("8/20 EMA Status", ema_status, f"{curr['ema8'] - curr['ema20']:.2f} delta")
            
            macd_dir = "Improving" if curr['hist'] > prev['hist'] else "Fading"
            c2.metric("MACD Momentum", macd_dir, f"{curr['hist']:.3f} hist")
            
            pos = "Upper Half" if S > curr['sma20'] else "Lower Half"
            c3.metric("Bollinger Position", pos, f"{((S - curr['lower'])/(curr['upper'] - curr['lower']))*100:.1f}% Band")

            st.divider()
            st.line_chart(df_tech[['Close', 'ema8', 'ema20', 'upper', 'lower']])
        else:
            st.warning("⚠️ Technical analysis stream offline.")

    with t_ai:
        c1, c2 = st.columns([4, 1])
        with c1: st.subheader(f"🌐 Search AI Grounding Vector: {st.session_state.current_ticker}")
        with c2:
            if st.button("🔄 Refresh Data Vector", use_container_width=True):
                st.session_state.ai_brief = ""
                st.rerun()
        if not st.session_state.ai_brief:
            with st.spinner("Executing real-time web scan query..."):
                st.session_state.ai_brief = get_ai_research(st.session_state.current_ticker)
        st.markdown(st.session_state.ai_brief)

else:
    st.info("👈 Input a valid trading ticker to trigger the options matrix models.")
