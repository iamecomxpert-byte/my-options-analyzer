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

# --- NEW: Fetch news from Yahoo Finance (no API limits!) ---
def fetch_news_for_ticker(ticker):
    """Fetch latest news for a ticker using yfinance - completely free, no rate limits."""
    try:
        stock = yf.Ticker(ticker)
        news_list = stock.news
        
        if not news_list:
            return None
        
        # Get up to 8 most recent news articles
        formatted_news = []
        for item in news_list[:8]:
            formatted_news.append({
                'title': item.get('title', 'No title'),
                'link': item.get('link', '#'),
                'publisher': item.get('publisher', 'Unknown'),
                'providerPublishTime': item.get('providerPublishTime', '')
            })
        return formatted_news
    except Exception as e:
        st.warning(f"Could not fetch news: {str(e)[:100]}")
        return None

# --- AI RESEARCH ENGINE (Now using Yahoo Finance news + Gemini summarization) ---
def get_ai_research(ticker):
    api_key = st.secrets.get("GEMINI_API_KEY")
    if not api_key:
        return "⚠️ Please add GEMINI_API_KEY to Streamlit Secrets."
    
    # Fetch news using yfinance (free, no rate limits)
    news_articles = fetch_news_for_ticker(ticker)
    
    if not news_articles:
        return f"ℹ️ No recent news found for {ticker}. The stock may have low coverage or the ticker is invalid."
    
    # Format news for Gemini prompt
    news_text = "\n\n".join([
        f"**News {i+1}** (Source: {item['publisher']})\nTitle: {item['title']}\nLink: {item['link']}"
        for i, item in enumerate(news_articles)
    ])
    
    prompt = f"""
    You are a financial analyst. Below are the latest {len(news_articles)} news articles for stock {ticker}.
    
    NEWS ARTICLES:
    {news_text}
    
    Based ONLY on these news articles, provide a concise analysis:
    
    1. **Analyst Consensus**: What are analysts saying? (price targets, upgrades/downgrades if mentioned)
    2. **Key Catalysts**: Upcoming events, earnings dates, product launches mentioned
    3. **Sentiment Drivers**: Top 3 themes from the last 7 days
    4. **Actionable View**: Based strictly on this news flow, give a "Bullish", "Neutral", or "Cautious" rating with 1-sentence reasoning
    
    Keep it factual and concise. Do not invent information not in the articles.
    """
    
    try:
        client = genai.Client(api_key=api_key)
        model_id = "gemini-2.0-flash"
        
        response = client.models.generate_content(
            model=model_id,
            contents=prompt
        )
        
        # Add source attribution
        return f"### 📰 News Summary for {ticker}\n\n{response.text}\n\n---\n*📌 Sources: {len(news_articles)} recent news articles from Yahoo Finance*"
        
    except Exception as e:
        if "429" in str(e):
            return "❌ **Gemini API rate limited.** Please wait 60 seconds and try again."
        return f"❌ **AI Summary Failed.** Error: {str(e)[:100]}"

# --- PAGE CONFIG & SESSION STATE ---
st.set_page_config(page_title="Analyst Pro Options Suite v2", layout="wide")

state_keys = {
    'price': None, 'trend': None, 'sma20': 0, 'pct_change': 0, 
    'stock_name': None, 'expiries': [], 'current_ticker': "", 
    'credits_used': 0, 'ai_brief': "", 'last_refresh': "Never", 'hist_data': pd.DataFrame(),
    'global_conservative': None, 'global_aggressive': None, 'global_speculative': None
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

# --- DATA FETCHING & GLOBAL SCANS ---
if fetch_btn:
    st.session_state.current_ticker = ticker_input
    st.session_state.ai_brief = "" 
    st.session_state.global_conservative = None
    st.session_state.global_aggressive = None
    st.session_state.global_speculative = None
    
    try:
        stock_obj = yf.Ticker(ticker_input)
        hist = stock_obj.history(period="100d")
        
        if hist.empty or 'Close' not in hist.columns:
            st.error(f"❌ No valid history found for {ticker_input}. Symbol may be incorrect.")
            st.session_state.price = None
        else:
            st.session_state.hist_data = hist
            st.session_state.price = hist['Close'].iloc[-1]
            st.session_name = stock_obj.info.get('longName', ticker_input)
            st.session_state.stock_name = st.session_name
            st.session_state.expiries = list(stock_obj.options)
            
            sma20_val = hist['Close'].rolling(window=20).mean().iloc[-1]
            st.session_state.trend = "Bullish" if st.session_state.price > sma20_val else "Bearish"
            st.session_state.pct_change = ((st.session_state.price / hist['Close'].iloc[-20]) - 1) * 100
            
            df_tech_init = hist.copy()
            curr_init, prev_init = get_technicals(df_tech_init)
            
            tech_score = 0
            if curr_init['ema8'] > curr_init['ema20']: tech_score += 1
            if curr_init['hist'] > prev_init['hist']: tech_score += 1
            if st.session_state.price > sma20_val: tech_score += 1

            # Multi-expiration background scanner for the Summary Tab
            today = datetime.now().date()
            valid_global_expiries = [exp for exp in stock_obj.options if (pd.to_datetime(exp).date() - today).days >= 60]
            
            cons_candidates = []
            aggr_candidates = []
            spec_candidates = []
            
            with st.spinner("Processing mathematical matrix across options chain..."):
                for exp_date in valid_global_expiries[:6]:
                    try:
                        opt_chain = stock_obj.option_chain(exp_date).calls
                        days_exp = (pd.to_datetime(exp_date).date() - today).days
                        t_yrs = days_exp / 365
                        
                        for _, row in opt_chain.iterrows():
                            mid_p = (row['bid'] + row['ask']) / 2 if row['bid'] > 0 else row['lastPrice']
                            if mid_p <= 0 or row['impliedVolatility'] <= 0: continue
                            
                            d, g, t, v = calculate_greeks(st.session_state.price, row['strike'], t_yrs, 0.05, row['impliedVolatility'])
                            p_t = calculate_p_touch(st.session_state.price, row['strike'], t_yrs, row['impliedVolatility'])
                            ev_val = (p_t * (mid_p * (1 + profit_target_pct / 100))) - ((1 - p_t) * (mid_p * (stop_loss_pct / 100)))
                            cts = int(((d * 0.4) + (p_t * 0.4) + (tech_score / 3.0 * 0.2)) * 100)
                            
                            c_data = {
                                'strike': row['strike'], 'expiry': exp_date, 'mid': mid_p, 'delta': d, 
                                'p_touch': p_t, 'ev': ev_val, 'cts': cts, 'days': days_exp, 'iv': row['impliedVolatility']
                            }
                            
                            if 0.50 <= d <= 0.60: cons_candidates.append(c_data)
                            elif 0.40 <= d <= 0.49: aggr_candidates.append(c_data)
                            elif 0.30 <= d <= 0.39: spec_candidates.append(c_data)
                    except:
                        continue
            
            if cons_candidates: st.session_state.global_conservative = max(cons_candidates, key=lambda x: x['ev'])
            if aggr_candidates: st.session_state.global_aggressive = max(aggr_candidates, key=lambda x: x['ev'])
            if spec_candidates: st.session_state.global_speculative = max(spec_candidates, key=lambda x: x['ev'])

    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")

# --- MAIN DASHBOARD VIEW ---
if st.session_state.price and st.session_state.expiries:
    S = st.session_state.price
    st.header(f"{st.session_state.stock_name} ({st.session_state.current_ticker})")
    
    col_p, col_t = st.columns(2)
    col_p.metric("Current Underlying Price", f"${S:.2f}")
    col_t.metric("20-Day Baseline Trend", st.session_state.trend, f"{st.session_state.pct_change:.1f}%")

    st.divider()
    
    # Tabs Container
    t_summary, t_cons, t_aggr, t_spec, t_tech, t_ai, t_edu = st.tabs([
        "📋 Global Recommendations", "🛡️ Conservative Buy", "⚡ Aggressive Buy", 
        "🎰 Speculative Buy", "📊 Technical Analysis", "🤖 AI Research", "📖 Strategy Guide"
    ])

    # Permanent Summary Tab Rendering Engine
    with t_summary:
        st.subheader("🏁 Automated Quantitative Trading Dashboard")
        st.markdown("This panel displays the mathematically optimal contract selection across the *entire chain life* matching our risk filters ($\ge$ 60 Days Expiry).")
        
        sum_data = []
        profiles = [
            ("🛡️ Conservative Buy", st.session_state.global_conservative, "50-60%"),
            ("⚡ Aggressive Buy", st.session_state.global_aggressive, "40-49%"),
            ("🎰 Speculative Buy", st.session_state.global_speculative, "30-39%")
        ]
        
        for name, profile, target_d in profiles:
            if profile:
                t_exit = profile['mid'] * (1 + profit_target_pct / 100)
                s_loss = profile['mid'] * (1 - stop_loss_pct / 100)
                h_days = min(int(profile['days'] * 0.4), 45)
                h_date = (datetime.now() + timedelta(days=h_days)).strftime('%b %d, %Y')
                
                sum_data.append({
                    "Strategy Profile": name,
                    "Target Delta": target_d,
                    "Optimal Strike": f"${profile['strike']:.2f} Call",
                    "Best Expiry Date": profile['expiry'],
                    "Entry Target (Mid)": f"${profile['mid']:.2f}",
                    "Take Profit Target": f"${t_exit:.2f}",
                    "Stop Loss Target": f"${s_loss:.2f}",
                    "Max Hold Time": f"{h_days} Days ({h_date})",
                    "Composite Score": f"{profile['cts']}/100"
                })
        
        if sum_data:
            df_summary_table = pd.DataFrame(sum_data)
            st.table(df_summary_table.set_index("Strategy Profile"))
        else:
            st.warning("No contracts met the strict mathematical baseline definitions across the processed options chain.")

    # Shared Dropdown Matrix for Individual Workspace Tabs
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔍 Workspace Adjuster")
    expiry = st.sidebar.selectbox("Select Expiry for Individual Tabs Below:", st.session_state.expiries)
    days_to_expiry = (pd.to_datetime(expiry).date() - datetime.now().date()).days
    T_years = max(days_to_expiry, 1) / 365

    if days_to_expiry < 60:
        st.sidebar.warning(f"⚠️ Selected expiry ({days_to_expiry} days) is under the 2+ month framework recommendation rule.")

    # Fetch Call Chain Options table for specific selections
    chain = yf.Ticker(st.session_state.current_ticker).option_chain(expiry).calls
    
    # Recalculate full tech variables for individual interactive views
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

    def process_tier_strategy(tab_component, delta_min, delta_max, tier_label):
        with tab_component:
            tier_contracts = []
            all_available_contracts = []
            
            for index, row in chain.iterrows():
                mid = (row['bid'] + row['ask']) / 2 if row['bid'] > 0 else row['lastPrice']
                if mid <= 0 or row['impliedVolatility'] <= 0: continue
                
                d, g, t, v = calculate_greeks(S, row['strike'], T_years, 0.05, row['impliedVolatility'])
                p_touch = calculate_p_touch(S, row['strike'], T_years, row['impliedVolatility'])
                pot_profit = mid * (1 + profit_target_pct / 100)
                pot_loss = mid * (stop_loss_pct / 100)
                ev = (p_touch * pot_profit) - ((1 - p_touch) * pot_loss)
                cts = int(((d * 0.4) + (p_touch * 0.4) + (tech_score / 3.0 * 0.2)) * 100)
                
                item = {
                    'strike': row['strike'], 'mid': mid, 'delta': d, 'theta': t, 'gamma': g, 'vega': v,
                    'iv': row['impliedVolatility'], 'p_touch': p_touch, 'ev': ev, 'cts': cts, 'symbol': row['contractSymbol']
                }
                
                all_available_contracts.append(item)
                if delta_min <= d <= delta_max:
                    tier_contracts.append(item)
            
            if not all_available_contracts:
                st.error("No valid options contracts returned from data stream for this expiry.")
                return

            if tier_contracts:
                df_tier = pd.DataFrame(tier_contracts).sort_values(by='ev', ascending=False)
                default_strike = df_tier.iloc[0]['strike']
            else:
                df_all_fallback = pd.DataFrame(all_available_contracts).sort_values(by='strike')
                default_strike = df_all_fallback.iloc[(df_all_fallback['strike'] - S).abs().argsort()[:1]].iloc[0]['strike']

            df_all = pd.DataFrame(all_available_contracts)
            strike_list = sorted(df_all['strike'].tolist())

            # Interactive Dropdown showing all strikes
            selected_k = st.selectbox(f"Select Alternative Strike to Inspect ({tier_label} Sandbox):", strike_list, index=strike_list.index(default_strike), key=f"sel_{tier_label}_{expiry}")
            
            chosen = df_all[df_all['strike'] == selected_k].iloc[0]
            chosen_exit = chosen['mid'] * (1 + profit_target_pct / 100)
            chosen_stop = chosen['mid'] * (1 - stop_loss_pct / 100)
            chosen_hold = min(int(days_to_expiry * 0.4), 45)
            chosen_date = (datetime.now() + timedelta(days=chosen_hold)).strftime('%B %d, %Y')
            
            st.markdown(f"### 🎯 Active Selection Recommendation Metrics (${selected_k:.2f} Call)")
            box_html = f"""
            <div style="border: 2px solid #2196F3; padding: 15px; border-radius: 8px; background-color: rgba(33, 150, 243, 0.1); margin-bottom: 25px;">
                <h4 style="margin-top:0; color:#2196F3;">Calculated Directives for the ${chosen['strike']:.2f} Strike Structure:</h4>
                <table style="width:100%; border:none; color:inherit; margin-top:10px;">
                    <tr>
                        <td><b>Composite Score:</b> {chosen['cts']}/100</td>
                        <td><b>Entry Mid Price:</b> ${chosen['mid']:.2f}</td>
                        <td><b>Take Profit Target:</b> ${chosen_exit:.2f}</td>
                    </tr>
                    <tr>
                        <td><b>Stop Loss Point:</b> ${chosen_stop:.2f}</td>
                        <td><b>Max Hold Limit:</b> {chosen_hold} Days</td>
                        <td><b>Calendar Cutoff Date:</b> {chosen_date}</td>
                    </tr>
                </table>
            </div>
            """
            st.markdown(box_html, unsafe_allow_html=True)

            st.divider()
            
            st.markdown("### 🔍 Mathematical Output Summary")
            c1, c2, c3 = st.columns([1.5, 1.5, 2])
            with c1:
                # Conviction Status Indicators with Embedded Definitions
                if chosen['cts'] >= 55 and chosen['ev'] > 0:
                    st.success("✅ STRUCTURAL BUY INSTANCE")
                    st.markdown("""
                    <p style='font-size:0.85rem; color:rgba(255,255,255,0.75);line-height:1.3;'>
                    <b>What this means in plain English:</b> The odds are highly in your favor. 
                    The combination of healthy upward stock momentum, a strong mathematical win rate, and fair contract pricing makes this a premier risk-reward setup.
                    </p>
                    """, unsafe_allow_html=True)
                elif chosen['cts'] >= 40 and chosen['ev'] > 0:
                    st.warning("⚠️ WEAK EDGE PATTERN")
                    st.markdown("""
                    <p style='font-size:0.85rem; color:rgba(255,255,255,0.75);line-height:1.3;'>
                    <b>What this means in plain English:</b> This option has a mathematical edge, but it is thin. 
                    Some charts are flashing mixed signals, meaning you have a decent shot, but you must keep your position size smaller and stay strict with your stop-loss.
                    </p>
                    """, unsafe_allow_html=True)
                else:
                    st.error("❌ NEGATIVE EXPECTANCY AVOID")
                    st.markdown("""
                    <p style='font-size:0.85rem; color:rgba(255,255,255,0.75);line-height:1.3;'>
                    <b>What this means in plain English:</b> Stay away. The pricing math on this strike is either too expensive or too far out of reach. 
                    Statistically, playing these setups results in an outright loss over time.
                    </p>
                    """, unsafe_allow_html=True)
                    
                st.metric("Inspected Score Metric", f"{chosen['cts']}/100")
                st.metric("Inspected Entry Target", f"${chosen['mid']:.2f}")
                
            with c2:
                st.metric("Inspected Take Profit", f"${chosen_exit:.2f}")
                st.metric("Inspected Stop Loss", f"${chosen_stop:.2f}")
                st.write(f"⏱️ **Hold Cutoff:** `{chosen_hold} days` ({chosen_date})")

            with c3:
                st.write("**Stochastic Engine Outputs**")
                st.write(f"- Stat Probability ($P_{{\\text{{ITM}}}}$ Delta Proxy): `{chosen['delta'] * 100:.1f}%`")
                st.write(f"- Path Touch Probability ($P_{{\\text{{touch}}}}$): `{chosen['p_touch'] * 100:.1f}%`")
                st.write(f"- Expected Valuation Return Matrix ($E[X]$): `{chosen['ev']:.3f}`")
                st.write(f"- Volatility Index (IV): `{chosen['iv']*100:.1f}%` | Daily Theta: `-{abs(chosen['theta']):.3f}`")
                
                st.markdown("""
                <div style="background-color: rgba(255,255,255,0.05); padding: 8px 12px; border-radius: 5px; font-size: 0.85rem; border-left: 3px solid #888;">
                <b>📈 What these numbers mean in plain English:</b><br>
                • <b>Delta Proxy:</b> The estimated mathematical chance this contract finishes completely inside the money at expiration.<br>
                • <b>Path Touch Probability:</b> The likelihood that the stock price flashes or touches this strike at least <i>once</i> before expiry (giving you a chance to exit early). This is usually double the Delta value.<br>
                • <b>Expected Valuation ($E[X]$):</b> The net profit expectancy. A positive value means the risk-to-reward math favors long setups over time.<br>
                • <b>Daily Theta:</b> The amount of premium value this option drops every single day just from time moving forward. Higher IV speeds up this decay.
                </div>
                """, unsafe_allow_html=True)
                
                st.write("")
                h_chart = yf.Ticker(chosen['symbol']).history(period="1mo")
                if not h_chart.empty: 
                    st.line_chart(h_chart['Close'])

    # Map strategies into isolated tiers 
    process_tier_strategy(t_cons, 0.50, 0.60, "Conservative")
    process_tier_strategy(t_aggr, 0.40, 0.49, "Aggressive")
    process_tier_strategy(t_spec, 0.30, 0.39, "Speculative")

    # Technical Analysis Verdict Core block
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

    # --- MODIFIED AI RESEARCH TAB (Now uses Yahoo Finance news, not Google Search) ---
    with t_ai:
        c1, c2 = st.columns([4, 1])
        with c1: 
            st.subheader(f"📰 News & AI Analysis: {st.session_state.current_ticker}")
            st.caption("Powered by Yahoo Finance news + Gemini AI summarization (no rate limits)")
        with c2:
            if st.button("🔄 Refresh News", use_container_width=True):
                st.session_state.ai_brief = ""
                st.rerun()
        
        if not st.session_state.ai_brief:
            with st.spinner("Fetching latest news and generating AI summary..."):
                st.session_state.ai_brief = get_ai_research(st.session_state.current_ticker)
        
        st.markdown(st.session_state.ai_brief)

    # Built out Strategy Guide Master Encyclopedia
    with t_edu:
        st.header("📖 System Strategy Guide & Indicator Dictionary")
        st.markdown("Welcome to the complete manual documentation. Below is the full breakdown of how our metrics work, what they mean, and how to execute positions.")
        
        st.divider()
        
        st.subheader("🎯 Conviction Indicator Glossary")
        st.markdown("""
        When you select a contract inside the manual sandbox, the engine evaluates it against historical prices and probability matrices to assign a conviction state:
        * **`STRUCTURAL BUY INSTANCE`**: High mathematical conviction. It implies that your chosen option benefits from an excellent trend, strong probability metrics, and favorable options pricing.
        * **`WEAK EDGE PATTERN`**: Moderate caution. There is a statistical edge favoring a profitable outcome, but indicators are mixed. Position size should be scaled down to minimize exposure.
        * **`NEGATIVE EXPECTANCY AVOID`**: Severe statistical disadvantage. Over a large sample size, playing this exact contract layout loses money due to excessive time decay, high overpricing, or bad trend health.
        """)
        
        st.divider()
        
        st.subheader("📊 Stochastic Engine Glossary")
        st.markdown("""
        * **Delta Proxy ($P_{\\text{ITM}}$)**: Measures the theoretical probability that the option will expire deep inside the money. A Delta of `0.50` means a roughly 50% chance of expiring profitable.
        * **Path Touch Probability ($P_{\\text{touch}}$)**: Calculates the mathematical probability that the stock price hits or crosses your strike price at least *once* during the life of the option. This is almost always double your raw Delta proxy.
        * **Expected Valuation ($E[X]$)**: Represents your long-term expectancy. It calculates: `(Touch Probability × Target Take Profit Value) - (Loss Probability × Max Allowed Stop Loss Value)`. A positive number signifies a structurally sound trade layout.
        * **Daily Theta**: The daily rent fee or time decay your option contract experiences. Every 24 hours that pass, the option premium drops by this exact amount, all else remaining equal.
        * **Implied Volatility (IV)**: The market's expectation of future asset fluctuation. High IV indicates expensive option premiums, accelerating time decay.
        """)
        
        st.divider()
        
        st.subheader("📈 Momentum Indicator Reference")
        st.markdown("""
        * **8 & 20 Day EMA Crossover**: Standard trend filter. When the fast 8 EMA stays above the slow 20 EMA, buyers control short-term swing momentum.
        * **MACD Histogram**: Monitors acceleration. When the histogram expands upward, buying pressure is accelerating. When it shrinks, momentum is exhausting.
        * **Bollinger Bands**: Volatility boundaries. The middle band represents the 20-day simple moving average baseline. Bounces off this line indicate healthy technical trend retention.
        """)
else:
    st.info("👈 Input a valid trading ticker to trigger the options matrix models.")
