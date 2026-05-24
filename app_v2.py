import streamlit as st
import yfinance as yf
import requests
import pandas as pd
from datetime import datetime, timedelta
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
st.set_page_config(page_title="Analyst Pro Options Suite v2", layout="wide")

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
            
            # Change 1: Show all expiry options without filtering them out completely
            st.session_state.expiries = list(stock_obj.options)
            
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

    # Dropdown displaying all available options
    expiry = st.selectbox("Select Expiry Date:", st.session_state.expiries)
    days_to_expiry = (pd.to_datetime(expiry).date() - datetime.now().date()).days
    T_years = max(days_to_expiry, 1) / 365

    # Visual dynamic warning if the chosen date violates our rule structure
    if days_to_expiry < 60:
        st.warning(f"⚠️ **Rule Warning:** Selected expiry is {days_to_expiry} days away. Framework rules recommend choosing an option $\ge$ 60 days (2+ months) out.")

    # Fetch Call Chain Options 
    chain = yf.Ticker(st.session_state.current_ticker).option_chain(expiry).calls
    
    # Pre-calculate the core technical indicator values for scoring inputs
    tech_score = 0
    verdict_reasons = []
    if not st.session_state.hist_data.empty:
        df_tech = st.session_state.hist_data.copy()
        curr, prev = get_technicals(df_tech)
        if curr['ema8'] > curr['ema20']:
            tech_score += 1
            verdict_reasons.append("Short-term momentum (8 EMA) is leading the long-term trend.")
        if curr['hist'] > prev['hist']:
            tech_score += 1
            verdict_reasons.append("MACD histogram is rising, indicating selling pressure is exhausting or buying is accelerating.")
        if S > curr['sma20']:
            tech_score += 1
            verdict_reasons.append("Price is holding above the 20-day baseline (Middle Bollinger Band).")

    st.divider()
    t_cons, t_aggr, t_spec, t_tech, t_ai, t_edu = st.tabs([
        "🛡️ Conservative Buy", "⚡ Aggressive Buy", "🎰 Speculative Buy", 
        "📊 Technical Analysis", "🤖 AI Grounding", "📖 Strategy Guide"
    ])

    def process_tier_strategy(tab_component, delta_min, delta_max, tier_label):
        with tab_component:
            tier_contracts = []
            
            # Map structural data loops
            for index, row in chain.iterrows():
                mid = (row['bid'] + row['ask']) / 2 if row['bid'] > 0 else row['lastPrice']
                if mid <= 0 or row['impliedVolatility'] <= 0: continue
                
                d, g, t, v = calculate_greeks(S, row['strike'], T_years, 0.05, row['impliedVolatility'])
                
                if delta_min <= d <= delta_max:
                    p_touch = calculate_p_touch(S, row['strike'], T_years, row['impliedVolatility'])
                    pot_profit = mid * (1 + profit_target_pct / 100)
                    pot_loss = mid * (stop_loss_pct / 100)
                    ev = (p_touch * pot_profit) - ((1 - p_touch) * pot_loss)
                    
                    tier_contracts.append({
                        'strike': row['strike'], 'mid': mid, 'delta': d, 'theta': t, 'gamma': g, 'vega': v,
                        'iv': row['impliedVolatility'], 'p_touch': p_touch, 'ev': ev, 'symbol': row['contractSymbol']
                    })
            
            if not tier_contracts:
                st.error(f"No contracts available on this expiry option matching {delta_min*100:.0f}%-{delta_max*100:.0f}% Delta.")
                return

            df_tier = pd.DataFrame(tier_contracts).sort_values(by='ev', ascending=False)
            optimal_contract = df_tier.iloc[0]
            
            # Calculate optimal target parameters
            opt_cts = int(((optimal_contract['delta'] * 0.4) + (optimal_contract['p_touch'] * 0.4) + (tech_score / 3.0 * 0.2)) * 100)
            opt_exit_p = optimal_contract['mid'] * (1 + profit_target_pct / 100)
            opt_stop_p = optimal_contract['mid'] * (1 - stop_loss_pct / 100)
            
            hold_days_limit = min(int(days_to_expiry * 0.4), 45)
            target_calendar_date = (datetime.now() + timedelta(days=hold_days_limit)).strftime('%B %d, %Y')

            # Change 2 & 3: The Persistent Recommendation Anchor Container
            st.markdown(f"### 🎯 Core Engine Recommendation ({tier_label} Profile)")
            
            box_html = f"""
            <div style="border: 2px solid #4CAF50; padding: 15px; border-radius: 8px; background-color: rgba(76, 175, 80, 0.1); margin-bottom: 20px;">
                <h4 style="margin-top:0; color:#4CAF50;">System Choice Strike: ${optimal_contract['strike']:.2f} Call</h4>
                <table style="width:100%; border:none; color:inherit;">
                    <tr>
                        <td><b>Composite Score:</b> {opt_cts}/100</td>
                        <td><b>Entry Limit (Mid):</b> ${optimal_contract['mid']:.2f}</td>
                        <td><b>Take Profit:</b> ${opt_exit_p:.2f} ({profit_target_pct}%)</td>
                    </tr>
                    <tr>
                        <td><b>Stop Loss Target:</b> ${opt_stop_p:.2f} (-{stop_loss_pct}%)</td>
                        <td><b>Max Hold Frame:</b> {hold_days_limit} Days</td>
                        <td><b>Hard Exit Calendar Cutoff:</b> {target_calendar_date}</td>
                    </tr>
                </table>
            </div>
            """
            st.markdown(box_html, unsafe_allowed_html=True)
            
            st.divider()
            
            # Separate Interactive Exploration Filters 
            st.markdown("### 🔍 Manual Strike Inspection Sandbox")
            strike_list = sorted(df_tier['strike'].tolist())
            selected_k = st.selectbox(f"Select Alternative {tier_label} Strike to Chart:", strike_list, index=strike_list.index(optimal_contract['strike']), key=f"sel_{tier_label}_{expiry}")
            
            chosen = df_tier[df_tier['strike'] == selected_k].iloc[0]
            chosen_cts = int(((chosen['delta'] * 0.4) + (chosen['p_touch'] * 0.4) + (tech_score / 3.0 * 0.2)) * 100)
            
            c1, c2, c3 = st.columns([1.5, 1.5, 2])
            with c1:
                if chosen_cts >= 55 and chosen['ev'] > 0:
                    st.success("✅ HIGH CONVICTION SETUP")
                elif chosen_cts >= 40 and chosen['ev'] > 0:
                    st.warning("⚠️ WEAK CONVICTION MATRIX")
                else:
                    st.error("❌ NEGATIVE EXPECTANCY AVOID")
                    
                st.metric("Inspected Composite Score", f"{chosen_cts}/100")
                st.metric("Inspected Entry Target", f"${chosen['mid']:.2f}")
                
            with c2:
                st.metric("Inspected Take Profit", f"${chosen['mid'] * (1 + profit_target_pct / 100):.2f}")
                st.metric("Inspected Stop Loss", f"${chosen['mid'] * (1 - stop_loss_pct / 100):.2f}")
                st.write(f"⏱️ **Hold Warning:** Exit prior to `{hold_days_limit} days` ({target_calendar_date}) to maintain safe theta exposure.")

            with c3:
                st.write("**Stochastic Pricing Models**")
                st.write(f"- Stat Probability ($P_{{\\text{{ITM}}}}$ Delta Proxy): `{chosen['delta'] * 100:.1f}%`主力")
                st.write(f"- Path Touch Probability ($P_{{\\text{{touch}}}}$): `{chosen['p_touch'] * 100:.1f}%`")
                st.write(f"- Expected Valuation Return Matrix ($E[X]$): `{chosen['ev']:.3f}`")
                st.write(f"- Daily Theta Drag: `-{abs(chosen['theta']):.3f}` | Vega Sensitive coefficient: `{chosen['vega']}`")
                
                h_chart = yf.Ticker(chosen['symbol']).history(period="1mo")
                if not h_chart.empty: 
                    st.line_chart(h_chart['Close'])

    # Map the three explicit strategy tiers based on your parameters
    process_tier_strategy(t_cons, 0.50, 0.60, "Conservative")
    process_tier_strategy(t_aggr, 0.40, 0.49, "Aggressive")
    process_tier_strategy(t_spec, 0.30, 0.39, "Speculative")

    # Change 4: Restored detailed Technical Analysis Engine
    with t_tech:
        if not st.session_state.hist_data.empty and 'Close' in st.session_state.hist_data.columns:
            df_tech = st.session_state.hist_data.copy()
            curr, prev = get_technicals(df_tech)
            
            st.subheader("Momentum & Volatility Health")
            c1, c2, c3 = st.columns(3)
            
            ema_status = "Bullish Cross" if curr['ema8'] > curr['ema20'] else "Bearish Separation"
            c1.metric("8/20 EMA Status", ema_status, f"{curr['ema8'] - curr['ema20']:.2f} delta")
            if curr['ema8'] > curr['ema20'] and prev['ema8'] <= prev['ema20']:
                c1.success("🔥 JUST CROSSED BULLISH")
            
            macd_dir = "Improving" if curr['hist'] > prev['hist'] else "Fading"
            c2.metric("MACD Momentum", macd_dir, f"{curr['hist']:.3f} hist")
            if curr['macd'] > 0: c2.caption("Trend Battery: Positive")
            
            pos = "Upper Half" if S > curr['sma20'] else "Lower Half"
            c3.metric("Bollinger Position", pos, f"{((S - curr['lower'])/(curr['upper'] - curr['lower']))*100:.1f}% Band")
            if S > curr['upper']: c3.warning("⚠️ OVEREXTENDED (Above Upper Band)")

            st.divider()
            st.line_chart(df_tech[['Close', 'ema8', 'ema20', 'upper', 'lower']])

            st.subheader("🏁 Final Technical Verdict")
            if tech_score == 3:
                st.success("🎯 **VERDICT: INVEST.** All technical indicators are aligned for a bullish continuation.")
            elif tech_score == 2:
                st.warning("⚖️ **VERDICT: CAUTION.** Technicals are mixed. Consider a smaller position or wait for confirmation.")
            else:
                st.error("🛑 **VERDICT: STAY AWAY.** Momentum is bearish and the price structure is weak.")
            
            with st.expander("View Verdict Logic"):
                for reason in verdict_reasons:
                    st.write(f"- {reason}")
                if tech_score < 2:
                    st.write("- Multiple indicators show declining strength or bearish crossovers.")
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

    # Change 5: Maintained the full strategy guide 
    with t_edu:
        st.subheader("📖 Technical Decoder & Playbook")
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.markdown("""
            #### 📊 Momentum Decoder
            * **Bullish Cross:** 8 EMA > 20 EMA. Short-term buyers are in control.
            * **Bearish Separation:** 8 EMA < 20 EMA. The stock is in a downtrend; avoid entering new Call positions.
            * **MACD Improving:** The histogram is rising (e.g., going from -2.0 to -1.5). This suggests a **reversal** or "buying the dip" opportunity.
            * **MACD Fading:** Histogram is falling. Buyers are losing steam.
            """)
        with col_g2:
            st.markdown("""
            #### ⚖️ High-Conviction Checklist
            - **Trend:** 20-Day SMA Bullish & 8 EMA > 20 EMA.
            - **MACD:** Look for a green, rising histogram.
            - **Bollinger:** Best entries occur when price bounces off the Middle Band (SMA 20).
            - **Risk:** Always set exit alarms for stop-loss or profit booking.
            """)
else:
    st.info("👈 Input a valid trading ticker to trigger the options matrix models.")
