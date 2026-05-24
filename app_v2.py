import streamlit as st
import yfinance as yf
import requests
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from scipy.stats import norm
from groq import Groq
import time
from datetime import datetime, timedelta

# --- SIMPLE CACHE FOR AI RESPONSES ---
class SimpleCache:
    def __init__(self, ttl_seconds=300):  # 5 minute TTL
        self.cache = {}
        self.ttl = ttl_seconds
    
    def get(self, key):
        if key in self.cache:
            entry = self.cache[key]
            if datetime.now() < entry['expires']:
                return entry['data']
            else:
                del self.cache[key]
        return None
    
    def set(self, key, value):
        self.cache[key] = {
            'data': value,
            'expires': datetime.now() + timedelta(seconds=self.ttl)
        }

# --- GROQ RETRY LOGIC ---
def call_groq_with_retry(client, prompt, max_retries=3, base_delay=2):
    """Call Groq with exponential backoff retry logic."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a financial analyst specializing in stock market news summarization."},
                    {"role": "user", "content": prompt}
                ],
                model="llama-3.3-70b-versatile",  # 30 req/min, high quality
                max_tokens=600,
                temperature=0.3,  # Lower temp for factual consistency
            )
            return response.choices[0].message.content
        except Exception as e:
            error_str = str(e)
            # Check if it's a rate limit error
            if "429" in error_str or "rate limit" in error_str.lower() or "quota" in error_str.lower():
                if attempt < max_retries - 1:
                    wait_time = base_delay * (2 ** attempt)  # 2, 4, 8 seconds
                    st.warning(f"⏳ Groq rate limit hit. Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                    continue
                else:
                    return None  # All retries exhausted
            else:
                # Non-rate-limit error, don't retry
                return None
    return None

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

# --- FETCH NEWS FROM FINNHUB (RELIABLE, WORKING LINKS) ---
def fetch_news_finnhub(ticker):
    """Fetch latest news for a ticker using Finnhub API - working links guaranteed."""
    api_key = st.secrets.get("FINNHUB_API_KEY")
    if not api_key:
        st.warning("⚠️ FINNHUB_API_KEY not found in secrets. News will be unavailable.")
        return None
    
    try:
        # Get news from last 7 days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        url = "https://finnhub.io/api/v1/company-news"
        params = {
            'symbol': ticker,
            'from': start_date.strftime('%Y-%m-%d'),
            'to': end_date.strftime('%Y-%m-%d'),
            'token': api_key
        }
        
        response = requests.get(url, params=params)
        
        if response.status_code != 200:
            st.warning(f"Finnhub API error: {response.status_code}")
            return None
        
        articles = response.json()
        
        if not articles:
            return None
        
        # Format articles with proper fields
        formatted_news = []
        for item in articles[:8]:  # Get up to 8 most recent
            formatted_news.append({
                'title': item.get('headline', 'No title'),
                'link': item.get('url', '#'),
                'publisher': item.get('source', 'Unknown'),
                'datetime': datetime.fromtimestamp(item.get('datetime', 0)).strftime('%Y-%m-%d %H:%M'),
                'summary': item.get('summary', '')[:200]  # Preview text
            })
        return formatted_news
    except Exception as e:
        st.warning(f"Could not fetch news: {str(e)[:100]}")
        return None

# --- AI RESEARCH ENGINE (Finnhub News + Groq AI) ---
def get_ai_research(ticker):
    groq_api_key = st.secrets.get("GROQ_API_KEY")
    finnhub_api_key = st.secrets.get("FINNHUB_API_KEY")
    
    if not groq_api_key:
        return "⚠️ Please add GROQ_API_KEY to Streamlit Secrets."
    
    if not finnhub_api_key:
        return "⚠️ Please add FINNHUB_API_KEY to Streamlit Secrets."
    
    # Check cache first
    cache_key = f"news_summary_{ticker}"
    cached_response = st.session_state.ai_cache.get(cache_key)
    if cached_response:
        return cached_response
    
    # Fetch news using Finnhub
    news_articles = fetch_news_finnhub(ticker)
    
    if not news_articles:
        return f"ℹ️ No recent news found for {ticker} in the last 7 days."
    
    # Format news for Groq prompt
    news_text = "\n\n".join([
        f"**News {i+1}** (Source: {item['publisher']}, Time: {item['datetime']})\n"
        f"Title: {item['title']}\n"
        f"Summary: {item['summary']}\n"
        f"Link: {item['link']}"
        for i, item in enumerate(news_articles)
    ])
    
    prompt = f"""
    You are a financial analyst. Below are the latest {len(news_articles)} news articles for stock {ticker} from the last 7 days.
    
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
        client = Groq(api_key=groq_api_key)
        
        # Use retry logic
        response_text = call_groq_with_retry(client, prompt)
        
        if response_text is None:
            # Fallback: Show raw news without AI summary
            fallback = f"### 📰 Recent News for {ticker}\n\n"
            fallback += "*(AI summary temporarily unavailable due to rate limits. Here are the raw headlines:)*\n\n"
            for i, item in enumerate(news_articles[:5]):
                fallback += f"**{i+1}. {item['title']}**  \n"
                fallback += f"📌 Source: {item['publisher']} | 🕐 {item['datetime']}  \n"
                fallback += f"🔗 [Read full article]({item['link']})  \n\n"
            fallback += "---\n*💡 Tip: Click 'Refresh News' in 30-60 seconds to try AI summary again.*"
            
            result = fallback
        else:
            # Add sources section
            sources_text = "\n".join([f"- [{item['title']}]({item['link']}) ({item['publisher']})" for item in news_articles[:5]])
            result = f"### 📰 AI Summary for {ticker}\n\n{response_text}\n\n---\n### 🔗 Sources\n{sources_text}\n\n*📌 Data provided by Finnhub.io*"
        
        # Cache the result
        st.session_state.ai_cache.set(cache_key, result)
        return result
        
    except Exception as e:
        # Ultimate fallback: just show raw news
        fallback = f"### 📰 Recent News for {ticker}\n\n"
        fallback += "*(AI service unavailable. Here are the latest headlines:)*\n\n"
        for i, item in enumerate(news_articles[:5]):
            fallback += f"**{i+1}. {item['title']}**  \n"
            fallback += f"📌 {item['publisher']} | 🕐 {item['datetime']}  \n"
            fallback += f"🔗 [Read full article]({item['link']})  \n\n"
        return fallback

# --- PAGE CONFIG & SESSION STATE ---
st.set_page_config(page_title="Analyst Pro Options Suite v2", layout="wide")

state_keys = {
    'price': None, 'trend': None, 'sma20': 0, 'pct_change': 0, 
    'stock_name': None, 'expiries': [], 'current_ticker': "", 
    'credits_used': 0, 'ai_brief': "", 'last_refresh': "Never", 'hist_data': pd.DataFrame(),
    'global_conservative': None, 'global_aggressive': None, 'global_speculative': None,
    'ai_cache': None  # Will be initialized below
}
for key, default in state_keys.items():
    if key not in st.session_state:
        st.session_state[key] = default

# Initialize AI cache
if st.session_state.ai_cache is None:
    st.session_state.ai_cache = SimpleCache()

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
            # First, collect all contracts for this expiry
            all_available_contracts = []
            tier_contracts = []
            
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

            # Calculate the BEST contract for this tier (highest EV)
            if tier_contracts:
                best_contract = max(tier_contracts, key=lambda x: x['ev'])
            else:
                # If no contracts in delta range, use the closest to target delta
                target_delta = (delta_min + delta_max) / 2
                best_contract = min(all_available_contracts, key=lambda x: abs(x['delta'] - target_delta))
            
            # LOCKED RECOMMENDATION SECTION (Never changes)
            st.markdown("### ⭐ RECOMMENDED STRIKE FOR THIS EXPIRY")
            st.markdown(f"*Best structure based on highest Expected Value (EV) for {tier_label} strategy*")
            
            # Calculate exit/stop for recommended contract
            reco_exit = best_contract['mid'] * (1 + profit_target_pct / 100)
            reco_stop = best_contract['mid'] * (1 - stop_loss_pct / 100)
            reco_hold = min(int(days_to_expiry * 0.4), 45)
            reco_date = (datetime.now() + timedelta(days=reco_hold)).strftime('%B %d, %Y')
            
            # Display the locked recommendation
            reco_html = f"""
            <div style="border: 2px solid #4CAF50; padding: 20px; border-radius: 10px; background-color: rgba(76, 175, 80, 0.1); margin-bottom: 25px;">
                <h4 style="margin-top:0; color:#4CAF50;">🎯 ${best_contract['strike']:.2f} Call Option</h4>
                <table style="width:100%; border:none; color:inherit; margin-top:10px;">
                    <tr>
                        <td><b>Composite Score:</b></td>
                        <td>{best_contract['cts']}/100</td>
                        <td><b>Entry Mid Price:</b></td>
                        <td>${best_contract['mid']:.2f}</td>
                    </tr>
                    <tr>
                        <td><b>Take Profit Target:</b></td>
                        <td>${reco_exit:.2f}</td>
                        <td><b>Stop Loss Point:</b></td>
                        <td>${reco_stop:.2f}</td>
                    </tr>
                    <tr>
                        <td><b>Max Hold Limit:</b></td>
                        <td>{reco_hold} Days</td>
                        <td><b>Calendar Cutoff Date:</b></td>
                        <td>{reco_date}</td>
                    </tr>
                </table>
            </div>
            """
            st.markdown(reco_html, unsafe_allow_html=True)
            
            # DIVIDER
            st.divider()
            
            # COMPARISON SECTION (Dropdown only affects this part)
            st.markdown("### 🔍 Compare Other Strikes")
            st.markdown("*Select any strike below to see how its mathematical metrics compare to the recommendation above*")
            
            # Create dropdown with all strikes, default to best contract
            strike_list = sorted([item['strike'] for item in all_available_contracts])
            default_index = strike_list.index(best_contract['strike']) if best_contract['strike'] in strike_list else 0
            
            selected_k = st.selectbox(
                f"Select Strike to Analyze ({tier_label} Comparison):", 
                strike_list, 
                index=default_index, 
                key=f"compare_{tier_label}_{expiry}"
            )
            
            # Find the selected contract data
            selected_contract = next((item for item in all_available_contracts if item['strike'] == selected_k), None)
            
            if selected_contract:
                selected_exit = selected_contract['mid'] * (1 + profit_target_pct / 100)
                selected_stop = selected_contract['mid'] * (1 - stop_loss_pct / 100)
                selected_hold = min(int(days_to_expiry * 0.4), 45)
                selected_date = (datetime.now() + timedelta(days=selected_hold)).strftime('%B %d, %Y')
                
                st.markdown("### 📊 Mathematical Output Summary")
                c1, c2, c3 = st.columns([1.5, 1.5, 2])
                with c1:
                    # Conviction Status Indicators
                    if selected_contract['cts'] >= 55 and selected_contract['ev'] > 0:
                        st.success("✅ STRUCTURAL BUY INSTANCE")
                        st.markdown("""
                        <p style='font-size:0.85rem; color:rgba(255,255,255,0.75);line-height:1.3;'>
                        <b>What this means:</b> The odds are highly in your favor. The combination of healthy upward stock momentum, 
                        a strong mathematical win rate, and fair contract pricing makes this a premier risk-reward setup.
                        </p>
                        """, unsafe_allow_html=True)
                    elif selected_contract['cts'] >= 40 and selected_contract['ev'] > 0:
                        st.warning("⚠️ WEAK EDGE PATTERN")
                        st.markdown("""
                        <p style='font-size:0.85rem; color:rgba(255,255,255,0.75);line-height:1.3;'>
                        <b>What this means:</b> This option has a mathematical edge, but it is thin. Some charts are flashing mixed signals, 
                        meaning you have a decent shot, but you must keep your position size smaller and stay strict with your stop-loss.
                        </p>
                        """, unsafe_allow_html=True)
                    else:
                        st.error("❌ NEGATIVE EXPECTANCY AVOID")
                        st.markdown("""
                        <p style='font-size:0.85rem; color:rgba(255,255,255,0.75);line-height:1.3;'>
                        <b>What this means:</b> Stay away. The pricing math on this strike is either too expensive or too far out of reach. 
                        Statistically, playing these setups results in an outright loss over time.
                        </p>
                        """, unsafe_allow_html=True)
                        
                    st.metric("Composite Score", f"{selected_contract['cts']}/100")
                    st.metric("Entry Target", f"${selected_contract['mid']:.2f}")
                    
                with c2:
                    st.metric("Take Profit Target", f"${selected_exit:.2f}")
                    st.metric("Stop Loss Point", f"${selected_stop:.2f}")
                    st.write(f"⏱️ **Hold Cutoff:** `{selected_hold} days` ({selected_date})")

                with c3:
                    st.write("**Stochastic Engine Outputs**")
                    st.write(f"- Stat Probability ($P_{{\\text{{ITM}}}}$ Delta Proxy): `{selected_contract['delta'] * 100:.1f}%`")
                    st.write(f"- Path Touch Probability ($P_{{\\text{{touch}}}}$): `{selected_contract['p_touch'] * 100:.1f}%`")
                    st.write(f"- Expected Valuation ($E[X]$): `{selected_contract['ev']:.3f}`")
                    st.write(f"- Volatility Index (IV): `{selected_contract['iv']*100:.1f}%` | Daily Theta: `-{abs(selected_contract['theta']):.3f}`")
                    
                    st.markdown("""
                    <div style="background-color: rgba(255,255,255,0.05); padding: 8px 12px; border-radius: 5px; font-size: 0.85rem; border-left: 3px solid #888;">
                    <b>📈 What these numbers mean:</b><br>
                    • <b>Delta Proxy:</b> Estimated chance this contract finishes in the money at expiration.<br>
                    • <b>Touch Probability:</b> Likelihood the stock price touches this strike before expiry.<br>
                    • <b>Expected Value ($E[X]$):</b> Net profit expectancy. Positive = favorable risk-reward.<br>
                    • <b>Daily Theta:</b> Premium value lost each day from time decay.
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.write("")
                    h_chart = yf.Ticker(selected_contract['symbol']).history(period="1mo")
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

    # --- AI RESEARCH TAB (Now with Finnhub news + Groq AI) ---
    with t_ai:
        c1, c2 = st.columns([4, 1])
        with c1: 
            st.subheader(f"📰 News & AI Analysis: {st.session_state.current_ticker}")
            st.caption("Powered by Finnhub news + Groq Llama 3.3 70B (auto-retry on rate limits)")
        with c2:
            if st.button("🔄 Refresh News", use_container_width=True):
                # Clear cache for this ticker on manual refresh
                cache_key = f"news_summary_{st.session_state.current_ticker}"
                if cache_key in st.session_state.ai_cache.cache:
                    del st.session_state.ai_cache.cache[cache_key]
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
