import streamlit as st
import yfinance as yf
import requests
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from scipy.stats import norm
from groq import Groq
import time
import gspread
from google.oauth2.service_account import Credentials
import json

# --- PAGE CONFIG MUST BE FIRST ---
st.set_page_config(page_title="Analyst Pro Options Suite v3.1", layout="wide")

# --- SIMPLE CACHE FOR AI RESPONSES ---
class SimpleCache:
    def __init__(self, ttl_seconds=300):
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

# --- GOOGLE SHEETS CONNECTION ---
@st.cache_resource
def get_google_sheet():
    try:
        creds_json = st.secrets["GOOGLE_SHEETS_CREDENTIALS"]
        creds_dict = json.loads(creds_json)
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds = Credentials.from_service_account_info(creds_dict, scopes=scope)
        client = gspread.authorize(creds)
        sheet_id = st.secrets["SPREADSHEET_ID"]
        sheet = client.open_by_key(sheet_id)
        return sheet
    except Exception as e:
        st.error(f"Failed to connect to Google Sheets: {str(e)}")
        return None

def init_portfolio_sheet():
    sheet = get_google_sheet()
    if not sheet:
        return None
    try:
        worksheet = sheet.worksheet("Portfolio")
    except:
        worksheet = sheet.add_worksheet(title="Portfolio", rows="1000", cols="20")
        headers = [
            "timestamp", "trader_name", "ticker", "strike", "expiry", 
            "contracts", "entry_price", "target_price", "stop_loss", "cutoff_date",
            "entry_iv", "entry_delta", "status", "last_recommendation", "last_alert_sent"
        ]
        worksheet.append_row(headers)
    return worksheet

def add_position_to_sheet(trader_name, ticker, strike, expiry, contracts, entry_price, 
                          entry_iv, entry_delta, target_price, stop_loss, cutoff_date):
    worksheet = init_portfolio_sheet()
    if not worksheet:
        return False
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = [
        now, trader_name, ticker, strike, expiry, contracts, entry_price,
        target_price, stop_loss, cutoff_date, entry_iv, entry_delta,
        "active", "HOLD", ""
    ]
    worksheet.append_row(row)
    return True

def get_portfolio_positions(trader_name=None):
    worksheet = init_portfolio_sheet()
    if not worksheet:
        return []
    records = worksheet.get_all_records()
    positions = []
    for idx, record in enumerate(records):
        if record.get("status") == "active":
            if trader_name and record.get("trader_name") != trader_name:
                continue
            positions.append((idx, record))
    return positions

def close_position(row_index):
    worksheet = init_portfolio_sheet()
    if not worksheet:
        return
    worksheet.update_cell(row_index + 2, 13, "closed")

def get_trader_list():
    worksheet = init_portfolio_sheet()
    if not worksheet:
        return ["Mukul"]
    records = worksheet.get_all_records()
    traders = set()
    for record in records:
        if record.get('trader_name'):
            traders.add(record['trader_name'])
    traders.add("Mukul")
    if not traders:
        return ["Mukul"]
    return sorted(list(traders))

# --- GROQ RETRY LOGIC ---
def call_groq_with_retry(client, prompt, max_retries=3, base_delay=2):
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a financial analyst specializing in stock market news summarization."},
                    {"role": "user", "content": prompt}
                ],
                model="llama-3.3-70b-versatile",
                max_tokens=600,
                temperature=0.3,
            )
            return response.choices[0].message.content
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "rate limit" in error_str.lower() or "quota" in error_str.lower():
                if attempt < max_retries - 1:
                    wait_time = base_delay * (2 ** attempt)
                    st.warning(f"⏳ Groq rate limit hit. Waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                else:
                    return None
            else:
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

def calculate_p_touch(S, K, T, sigma):
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

def get_macd_trend(hist_values, days=3):
    """Determine if MACD histogram is rising or falling."""
    if len(hist_values) < days:
        return "stable", 0
    recent = hist_values[-days:]
    if all(recent[i] > recent[i-1] for i in range(1, len(recent))):
        return "rising", days
    elif all(recent[i] < recent[i-1] for i in range(1, len(recent))):
        return "falling", days
    return "stable", 0

def get_ema_cross(curr_ema8, curr_ema20, prev_ema8, prev_ema20):
    """Determine EMA crossover status."""
    if curr_ema8 > curr_ema20:
        if prev_ema8 <= prev_ema20:
            return "bullish_cross"
        return "bullish"
    elif curr_ema8 < curr_ema20:
        if prev_ema8 >= prev_ema20:
            return "bearish_cross"
        return "bearish"
    return "neutral"

# --- HYBRID RECOMMENDATION ENGINE ---
def get_hybrid_recommendation(option_price, entry_price, target, stop_loss, 
                               days_left, current_delta, current_theta, current_iv,
                               cts, ev, tech_score, ema_status, macd_trend, macd_days,
                               touch_prob, iv_percentile=None):
    """
    Hybrid Quant + Technical Recommendation Engine
    """
    pnl_pct = ((option_price - entry_price) / entry_price) * 100 if entry_price > 0 else 0
    
    # ========== TIER 1: RISK OF RUIN (Exit Now) ==========
    if option_price <= stop_loss:
        return "🔴 EXIT - STOP LOSS", f"Stop loss hit at ${stop_loss:.2f}"
    
    if ev is not None and ev < 0:
        return "🔴 EXIT - NO EDGE", f"Expected Value negative: ${ev:.2f} after spreads"
    
    if cts is not None and cts < 35:
        return "🔴 EXIT - POOR SCORE", f"Composite Score dropped to {cts}/100"
    
    # ========== TIER 2: VOLATILITY SIGNALS ==========
    if iv_percentile is not None and iv_percentile > 90:
        return "🔴 SELL VOL", f"IV at {iv_percentile}th percentile - overpriced. Take profits."
    
    # ========== TIER 3: PROFIT TARGETS ==========
    if option_price >= target:
        return "🟢 TAKE PROFITS", f"Target ${target:.2f} reached (Entry: ${entry_price:.2f})"
    
    if option_price >= target * 0.8:
        if touch_prob > 0.65:
            return "🟡 PARTIAL EXIT", f"80% to target & {touch_prob:.0f}% touch probability"
        else:
            return "🟡 TAKE SOME OFF", f"80% to target - consider taking partial profits"
    
    # ========== TIER 4: TIME & PROBABILITY DECAY ==========
    if days_left < 7:
        theta_pct = abs(current_theta) / option_price * 100 if current_theta and option_price > 0 else 0
        return "🟠 TIME DECAY", f"{days_left} days left | Theta: {theta_pct:.1f}%/day"
    
    if current_delta < 0.25:
        return "🟠 PROBABILITY DECAY", f"Delta fell to {current_delta:.2f} (<0.25 threshold)"
    
    # ========== TIER 5: TECHNICAL DETERIORATION ==========
    if ema_status == "bearish_cross":
        return "🟠 TECHNICAL EXIT", "8 EMA crossed below 20 EMA - momentum turning bearish"
    
    if macd_trend == "falling" and macd_days >= 3:
        return "🟠 TECHNICAL EXIT", f"MACD falling for {macd_days} days - momentum fading"
    
    # ========== TIER 6: OPPORTUNITY SIGNALS ==========
    if (pnl_pct > 0 and pnl_pct < 25 and current_delta > 0.45 and 
        cts is not None and cts > 65 and ema_status in ["bullish", "bullish_cross"] and 
        macd_trend == "rising"):
        kelly_estimate = (pnl_pct / 100) * current_delta * 2
        return "🟢 ADD MORE", f"High conviction | CTS: {cts} | Kelly: {kelly_estimate:.0f}% | Add at ${option_price:.2f}"
    
    # ========== TIER 7: STRONG HOLD CONDITIONS ==========
    if (cts is not None and cts > 55 and ema_status in ["bullish", "bullish_cross"] and 
        macd_trend == "rising" and ev is not None and ev > 0.25):
        return "🔵 STRONG HOLD", f"All metrics aligned | CTS: {cts} | EV: ${ev:.2f}"
    
    # ========== DEFAULT: HOLD ==========
    ev_text = f"EV: ${ev:.2f}" if ev is not None else "EV: N/A"
    return "🔵 HOLD", f"Normal | {ev_text} | Days: {days_left} | Delta: {current_delta:.2f}"

# --- FETCH NEWS FROM FINNHUB ---
def fetch_news_finnhub(ticker):
    api_key = st.secrets.get("FINNHUB_API_KEY")
    if not api_key:
        return None
    try:
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
            return None
        articles = response.json()
        if not articles:
            return None
        formatted_news = []
        for item in articles[:8]:
            formatted_news.append({
                'title': item.get('headline', 'No title'),
                'link': item.get('url', '#'),
                'publisher': item.get('source', 'Unknown'),
                'datetime': datetime.fromtimestamp(item.get('datetime', 0)).strftime('%Y-%m-%d %H:%M'),
                'summary': item.get('summary', '')[:200]
            })
        return formatted_news
    except Exception as e:
        return None

def get_ai_research(ticker):
    groq_api_key = st.secrets.get("GROQ_API_KEY")
    cache_key = f"news_summary_{ticker}"
    cached_response = st.session_state.ai_cache.get(cache_key)
    if cached_response:
        return cached_response
    news_articles = fetch_news_finnhub(ticker)
    if not news_articles:
        return f"ℹ️ No recent news found for {ticker}"
    news_text = "\n\n".join([
        f"**News {i+1}** (Source: {item['publisher']}, Time: {item['datetime']})\n"
        f"Title: {item['title']}\n"
        f"Summary: {item['summary']}"
        for i, item in enumerate(news_articles)
    ])
    prompt = f"""
    You are a financial analyst. Below are the latest {len(news_articles)} news articles for stock {ticker}.
    
    NEWS ARTICLES:
    {news_text}
    
    Based ONLY on these news articles, provide a concise analysis:
    1. **Analyst Consensus**
    2. **Key Catalysts**
    3. **Sentiment Drivers**
    4. **Actionable View** (Bullish/Neutral/Cautious)
    """
    try:
        client = Groq(api_key=groq_api_key)
        response_text = call_groq_with_retry(client, prompt)
        if response_text is None:
            result = f"### 📰 Recent News for {ticker}\n\n"
            for i, item in enumerate(news_articles[:5]):
                result += f"**{i+1}. {item['title']}**  \n📌 {item['publisher']}\n\n"
        else:
            result = f"### 📰 AI Summary for {ticker}\n\n{response_text}"
        st.session_state.ai_cache.set(cache_key, result)
        return result
    except Exception as e:
        return f"News unavailable"

# --- PAGE CONFIG & SESSION STATE ---
state_keys = {
    'price': None, 'trend': None, 'sma20': 0, 'pct_change': 0, 
    'stock_name': None, 'expiries': [], 'current_ticker': "", 
    'credits_used': 0, 'ai_brief': "", 'last_refresh': "Never", 'hist_data': pd.DataFrame(),
    'global_conservative': None, 'global_aggressive': None, 'global_speculative': None,
    'ai_cache': None, 'profit_target_pct': 40, 'stop_loss_pct': 30
}
for key, default in state_keys.items():
    if key not in st.session_state:
        st.session_state[key] = default

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
    st.session_state.profit_target_pct = profit_target_pct
    st.session_state.stop_loss_pct = stop_loss_pct

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
            st.error(f"❌ No valid history found for {ticker_input}")
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

            today = datetime.now().date()
            valid_global_expiries = [exp for exp in stock_obj.options if (pd.to_datetime(exp).date() - today).days >= 60]
            
            cons_candidates = []
            aggr_candidates = []
            spec_candidates = []
            
            with st.spinner("Processing options chain..."):
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
    
    # Tabs (8 tabs including Portfolio)
    t_summary, t_cons, t_aggr, t_spec, t_tech, t_ai, t_portfolio, t_edu = st.tabs([
        "📋 Global Recs", "🛡️ Conservative", "⚡ Aggressive", 
        "🎰 Speculative", "📊 Technical", "🤖 AI Research", 
        "📂 Portfolio", "📖 Strategy Guide"
    ])

    with t_summary:
        st.subheader("🏁 Automated Quantitative Trading Dashboard")
        st.markdown("Mathematically optimal contracts with >= 60 days expiry")
        
        sum_data = []
        profiles = [
            ("🛡️ Conservative", st.session_state.global_conservative, "50-60%"),
            ("⚡ Aggressive", st.session_state.global_aggressive, "40-49%"),
            ("🎰 Speculative", st.session_state.global_speculative, "30-39%")
        ]
        
        for name, profile, target_d in profiles:
            if profile:
                t_exit = profile['mid'] * (1 + profit_target_pct / 100)
                s_loss = profile['mid'] * (1 - stop_loss_pct / 100)
                h_days = min(int(profile['days'] * 0.4), 45)
                h_date = (datetime.now() + timedelta(days=h_days)).strftime('%b %d, %Y')
                
                sum_data.append({
                    "Strategy": name,
                    "Target Delta": target_d,
                    "Strike": f"${profile['strike']:.2f} Call",
                    "Expiry": profile['expiry'],
                    "Entry": f"${profile['mid']:.2f}",
                    "Target": f"${t_exit:.2f}",
                    "Stop": f"${s_loss:.2f}",
                    "Score": f"{profile['cts']}/100"
                })
        
        if sum_data:
            st.dataframe(pd.DataFrame(sum_data), use_container_width=True)
        else:
            st.warning("No contracts met the criteria.")

    st.sidebar.markdown("---")
    st.sidebar.subheader("🔍 Workspace Adjuster")
    expiry = st.sidebar.selectbox("Select Expiry for Individual Tabs Below:", st.session_state.expiries)
    days_to_expiry = (pd.to_datetime(expiry).date() - datetime.now().date()).days
    T_years = max(days_to_expiry, 1) / 365

    if days_to_expiry < 60:
        st.sidebar.warning(f"⚠️ Selected expiry ({days_to_expiry} days) is under the 2+ month framework.")

    chain = yf.Ticker(st.session_state.current_ticker).option_chain(expiry).calls
    
    tech_score = 0
    verdict_reasons = []
    if not st.session_state.hist_data.empty:
        df_tech = st.session_state.hist_data.copy()
        curr, prev = get_technicals(df_tech)
        if curr['ema8'] > curr['ema20']:
            tech_score += 1
            verdict_reasons.append("Short-term momentum (8 EMA) is leading.")
        if curr['hist'] > prev['hist']:
            tech_score += 1
            verdict_reasons.append("MACD histogram is rising.")
        if S > curr['sma20']:
            tech_score += 1
            verdict_reasons.append("Price is above 20-day baseline.")

    def process_tier_strategy(tab_component, delta_min, delta_max, tier_label):
        with tab_component:
            all_available_contracts = []
            tier_contracts = []
            
            for index, row in chain.iterrows():
                mid = (row['bid'] + row['ask']) / 2 if row['bid'] > 0 else row['lastPrice']
                if mid <= 0 or row['impliedVolatility'] <= 0: continue
                
                volume = row.get('volume', 0)
                open_interest = row.get('openInterest', 0)
                bid = row.get('bid', 0)
                ask = row.get('ask', 0)
                spread = (ask - bid) if ask > 0 and bid > 0 else 0
                spread_pct = (spread / mid) * 100 if mid > 0 else 100
                
                d, g, t, v = calculate_greeks(S, row['strike'], T_years, 0.05, row['impliedVolatility'])
                p_touch = calculate_p_touch(S, row['strike'], T_years, row['impliedVolatility'])
                pot_profit = mid * (1 + profit_target_pct / 100)
                pot_loss = mid * (stop_loss_pct / 100)
                ev = (p_touch * pot_profit) - ((1 - p_touch) * pot_loss)
                cts = int(((d * 0.4) + (p_touch * 0.4) + (tech_score / 3.0 * 0.2)) * 100)
                
                if volume < 10:
                    liquidity_status = "🔴 EXTREMELY ILLIQUID"
                    liquidity_warning = "Less than 10 contracts traded today. AVOID."
                elif volume < 50:
                    liquidity_status = "🟠 LOW LIQUIDITY"
                    liquidity_warning = "Low volume. Wide spreads likely."
                elif volume < 200:
                    liquidity_status = "🟡 MODERATE LIQUIDITY"
                    liquidity_warning = "Acceptable for smaller positions."
                else:
                    liquidity_status = "🟢 HIGHLY LIQUID"
                    liquidity_warning = "Tight spreads, easy entry/exit."
                
                item = {
                    'strike': row['strike'], 'mid': mid, 'delta': d, 'theta': t, 'gamma': g, 'vega': v,
                    'iv': row['impliedVolatility'], 'p_touch': p_touch, 'ev': ev, 'cts': cts, 
                    'symbol': row['contractSymbol'], 'volume': volume, 'open_interest': open_interest,
                    'bid': bid, 'ask': ask, 'spread': spread, 'spread_pct': spread_pct,
                    'liquidity_status': liquidity_status, 'liquidity_warning': liquidity_warning
                }
                
                all_available_contracts.append(item)
                if delta_min <= d <= delta_max:
                    tier_contracts.append(item)
            
            if not all_available_contracts:
                st.error("No valid options contracts returned.")
                return

            if tier_contracts:
                best_contract = max(tier_contracts, key=lambda x: x['ev'])
            else:
                target_delta = (delta_min + delta_max) / 2
                best_contract = min(all_available_contracts, key=lambda x: abs(x['delta'] - target_delta))
            
            st.markdown("### ⭐ RECOMMENDED STRIKE FOR THIS EXPIRY")
            if best_contract['volume'] < 50:
                st.warning(f"{best_contract['liquidity_status']}: {best_contract['liquidity_warning']}")
            
            reco_exit = best_contract['mid'] * (1 + profit_target_pct / 100)
            reco_stop = best_contract['mid'] * (1 - stop_loss_pct / 100)
            reco_hold = min(int(days_to_expiry * 0.4), 45)
            reco_date = (datetime.now() + timedelta(days=reco_hold)).strftime('%B %d, %Y')
            
            st.markdown(f"""
            <div style="border: 2px solid #4CAF50; padding: 20px; border-radius: 10px; background-color: rgba(76, 175, 80, 0.1);">
                <h4 style="margin-top:0;">🎯 ${best_contract['strike']:.2f} Call Option</h4>
                <table style="width:100%;">
                    <tr><td><b>Composite Score:</b></td><td>{best_contract['cts']}/100</td>
                        <td><b>Entry:</b></td><td>${best_contract['mid']:.2f}</td></tr>
                    <tr><td><b>Target:</b></td><td>${reco_exit:.2f}</td>
                        <td><b>Stop:</b></td><td>${reco_stop:.2f}</td></tr>
                    <tr><td><b>Hold Limit:</b></td><td>{reco_hold} Days</td>
                        <td><b>Cutoff:</b></td><td>{reco_date}</td></tr>
                    <tr><td><b>Volume:</b></td><td>{best_contract['volume']:,}</td>
                        <td><b>OI:</b></td><td>{best_contract['open_interest']:,}</td></tr>
                </table>
            </div>
            """, unsafe_allow_html=True)
            
            st.divider()
            st.markdown("### 🔍 Compare Other Strikes")
            strike_list = sorted([item['strike'] for item in all_available_contracts])
            default_index = strike_list.index(best_contract['strike'])
            selected_k = st.selectbox(f"Select Strike to Compare:", strike_list, index=default_index, key=f"compare_{tier_label}")
            selected = next((item for item in all_available_contracts if item['strike'] == selected_k), None)
            
            if selected:
                selected_exit = selected['mid'] * (1 + profit_target_pct / 100)
                selected_stop = selected['mid'] * (1 - stop_loss_pct / 100)
                st.markdown("### 📊 Mathematical Output Summary")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Composite Score", f"{selected['cts']}/100")
                    st.metric("Entry", f"${selected['mid']:.2f}")
                with col2:
                    st.metric("Target", f"${selected_exit:.2f}")
                    st.metric("Stop", f"${selected_stop:.2f}")
                with col3:
                    st.metric("Volume", f"{selected['volume']:,}")
                    st.metric("OI", f"{selected['open_interest']:,}")
                st.write(f"Delta: {selected['delta']:.3f} | Theta: {selected['theta']:.3f} | Vega: {selected['vega']:.3f}")
                st.write(f"Touch Prob: {selected['p_touch']*100:.1f}% | EV: {selected['ev']:.3f}")

    process_tier_strategy(t_cons, 0.50, 0.60, "Conservative")
    process_tier_strategy(t_aggr, 0.40, 0.49, "Aggressive")
    process_tier_strategy(t_spec, 0.30, 0.39, "Speculative")

    with t_tech:
        if not st.session_state.hist_data.empty:
            st.subheader("Technical Indicators")
            st.line_chart(st.session_state.hist_data[['Close']])
            if tech_score == 3:
                st.success("✅ BULLISH - All indicators aligned")
            elif tech_score == 2:
                st.warning("⚠️ MIXED - Exercise caution")
            else:
                st.error("❌ BEARISH - Avoid calls")

    with t_ai:
        if st.button("🔄 Refresh AI Analysis"):
            st.session_state.ai_brief = ""
            st.rerun()
        if not st.session_state.ai_brief:
            with st.spinner("Fetching AI analysis..."):
                st.session_state.ai_brief = get_ai_research(st.session_state.current_ticker)
        st.markdown(st.session_state.ai_brief)

        # ========================
    # PORTFOLIO TAB (HYBRID ADVANCED VERSION - CORRECTED)
    # ========================
    with t_portfolio:
        st.header("📂 Options Portfolio Tracker")
        
        if 'sheet_initialized' not in st.session_state:
            init_portfolio_sheet()
            st.session_state.sheet_initialized = True
        
        # Helper function to get conservative recommended strike for a given expiry
        def get_conservative_strike_for_expiry(ticker, expiry_date, profit_target_pct, stop_loss_pct):
            try:
                stock_obj = yf.Ticker(ticker)
                hist = stock_obj.history(period="100d")
                if hist.empty:
                    return None, None
                current_price = hist['Close'].iloc[-1]
                opt_chain = stock_obj.option_chain(expiry_date)
                calls = opt_chain.calls
                df_tech = hist.copy()
                curr, prev = get_technicals(df_tech)
                tech_score = 0
                if curr['ema8'] > curr['ema20']:
                    tech_score += 1
                if curr['hist'] > prev['hist']:
                    tech_score += 1
                if current_price > curr['sma20']:
                    tech_score += 1
                conservative_candidates = []
                days_to_expiry = (pd.to_datetime(expiry_date).date() - datetime.now().date()).days
                t_yrs = max(days_to_expiry, 1) / 365
                for _, row in calls.iterrows():
                    mid_p = (row['bid'] + row['ask']) / 2 if row['bid'] > 0 else row['lastPrice']
                    if mid_p <= 0 or row['impliedVolatility'] <= 0:
                        continue
                    d, _, _, _ = calculate_greeks(current_price, row['strike'], t_yrs, 0.05, row['impliedVolatility'])
                    if 0.50 <= d <= 0.60:
                        p_touch = calculate_p_touch(current_price, row['strike'], t_yrs, row['impliedVolatility'])
                        ev_val = (p_touch * (mid_p * (1 + profit_target_pct / 100))) - ((1 - p_touch) * (mid_p * (stop_loss_pct / 100)))
                        conservative_candidates.append({'strike': row['strike'], 'mid': mid_p, 'ev': ev_val})
                if conservative_candidates:
                    best = max(conservative_candidates, key=lambda x: x['ev'])
                    return best['strike'], best['mid']
                return None, None
            except Exception:
                return None, None
        
        def get_strikes_for_expiry(ticker, expiry_date):
            try:
                stock_obj = yf.Ticker(ticker)
                opt_chain = stock_obj.option_chain(expiry_date)
                calls = opt_chain.calls
                strikes_with_prices = []
                for _, row in calls.iterrows():
                    mid_p = (row['bid'] + row['ask']) / 2 if row['bid'] > 0 else row['lastPrice']
                    if mid_p > 0:
                        strikes_with_prices.append({'strike': row['strike'], 'mid': mid_p})
                return sorted(strikes_with_prices, key=lambda x: x['strike'])
            except Exception:
                return []
        
        def get_current_option_price(ticker, expiry, strike):
            try:
                stock_obj = yf.Ticker(ticker)
                opt_chain = stock_obj.option_chain(expiry)
                calls = opt_chain.calls
                option_row = calls[calls['strike'] == strike]
                if not option_row.empty:
                    row = option_row.iloc[0]
                    mid = (row['bid'] + row['ask']) / 2 if row['bid'] > 0 else row['lastPrice']
                    iv = row['impliedVolatility']
                    return mid, iv
                return None, None
            except Exception:
                return None, None
        
        def calculate_composite_score_for_position(current_price, strike, days_left, iv, tech_score):
            d, _, _, _ = calculate_greeks(current_price, strike, max(days_left, 1)/365, 0.05, iv)
            p_touch = calculate_p_touch(current_price, strike, max(days_left, 1)/365, iv)
            ev = (p_touch * (1 + st.session_state.profit_target_pct/100)) - ((1 - p_touch) * (st.session_state.stop_loss_pct/100))
            cts = int(((d * 0.4) + (p_touch * 0.4) + (tech_score / 3.0 * 0.2)) * 100)
            return cts, ev, d, p_touch
        
        # Helper function to add trader with email
        def add_trader_to_sheet(trader_name, email):
            """Add a new trader with email to the Traders sheet."""
            sheet = get_google_sheet()
            if not sheet:
                return False
            try:
                traders_worksheet = sheet.worksheet("Traders")
            except:
                traders_worksheet = sheet.add_worksheet(title="Traders", rows="100", cols="10")
                traders_worksheet.append_row(["trader_name", "email", "enabled", "created_at"])
            
            # Check if trader already exists
            existing = traders_worksheet.findall(trader_name)
            if existing:
                return False
            
            traders_worksheet.append_row([
                trader_name, email, "TRUE", datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ])
            return True
        
        def get_trader_email(trader_name):
            """Get email for a trader from Traders sheet."""
            sheet = get_google_sheet()
            if not sheet:
                return None
            try:
                traders_worksheet = sheet.worksheet("Traders")
                records = traders_worksheet.get_all_records()
                for record in records:
                    if record.get('trader_name') == trader_name:
                        return record.get('email')
                return None
            except:
                return None
        
        # Trader selection
        trader_options = get_trader_list()
        selected_trader = st.selectbox("Select Trader:", trader_options, key="trader_select")
        
        col_add1, col_add2 = st.columns([3, 1])
        with col_add2:
            if st.button("➕ Add New Trader", key="show_add_trader"):
                st.session_state.show_new_trader = True
        
        if st.session_state.get('show_new_trader', False):
            col_n1, col_n2, col_n3, col_n4 = st.columns([2, 2, 1, 1])
            with col_n1:
                new_trader_name = st.text_input("Trader name:", key="new_trader_input")
            with col_n2:
                new_trader_email = st.text_input("Email address:", key="new_trader_email", 
                                                  placeholder="trader@example.com")
            with col_n3:
                if st.button("Save", key="save_new_trader"):
                    if new_trader_name and new_trader_name not in trader_options:
                        if new_trader_email and "@" in new_trader_email:
                            # Add to Traders sheet
                            success = add_trader_to_sheet(new_trader_name, new_trader_email)
                            if success:
                                # Add placeholder position to Portfolio sheet
                                dummy_worksheet = init_portfolio_sheet()
                                if dummy_worksheet:
                                    dummy_worksheet.append_row([
                                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
                                        new_trader_name, "PLACEHOLDER", 0, "2024-01-01", 
                                        0, 0, 0, 0, "2024-01-01", 0, 0, "inactive", "", ""
                                    ])
                                    st.success(f"Trader '{new_trader_name}' added with email {new_trader_email}!")
                                    st.session_state.show_new_trader = False
                                    st.rerun()
                            else:
                                st.error("Trader already exists")
                        else:
                            st.error("Please enter a valid email address")
                    else:
                        st.error("Please enter a valid trader name")
            with col_n4:
                if st.button("Cancel", key="cancel_new_trader"):
                    st.session_state.show_new_trader = False
                    st.rerun()
        
        st.divider()
        st.subheader("📊 Active Positions")
        positions = get_portfolio_positions(selected_trader)
        
        col_refresh, _ = st.columns([1, 5])
        with col_refresh:
            if st.button("🔄 Refresh", key="refresh_portfolio", use_container_width=True):
                st.rerun()
        
        if positions:
            for idx, (row_idx, pos) in enumerate(positions):
                entry_price = float(pos['entry_price'])
                contracts = int(pos['contracts'])
                strike = float(pos['strike'])
                ticker = pos['ticker']
                expiry_date_str = pos['expiry']
                target = float(pos['target_price'])
                stop = float(pos['stop_loss'])
                
                option_price, current_iv = get_current_option_price(ticker, expiry_date_str, strike)
                
                if option_price and option_price > 0:
                    days_left = max((pd.to_datetime(expiry_date_str).date() - datetime.now().date()).days, 0)
                    pnl = (option_price - entry_price) * contracts * 100
                    pnl_pct = ((option_price - entry_price) / entry_price) * 100
                    
                    # Get technical data for this ticker
                    try:
                        stock = yf.Ticker(ticker)
                        hist = stock.history(period="60d")
                        if not hist.empty:
                            curr, prev = get_technicals(hist)
                            tech_score_pos = 0
                            if curr['ema8'] > curr['ema20']:
                                tech_score_pos += 1
                            if curr['hist'] > prev['hist']:
                                tech_score_pos += 1
                            if stock.history(period="1d")['Close'].iloc[-1] > curr['sma20']:
                                tech_score_pos += 1
                            hist_values = hist['hist'].dropna().values
                            macd_trend, macd_days = get_macd_trend(hist_values, 3)
                            ema_status = get_ema_cross(curr['ema8'], curr['ema20'], prev['ema8'], prev['ema20'])
                        else:
                            tech_score_pos = 1
                            macd_trend, macd_days = "stable", 0
                            ema_status = "neutral"
                    except:
                        tech_score_pos = 1
                        macd_trend, macd_days = "stable", 0
                        ema_status = "neutral"
                    
                    # Calculate CTS, EV, Delta, Touch Prob
                    stock_price = yf.Ticker(ticker).history(period="1d")['Close'].iloc[-1]
                    cts, ev, delta_calc, touch_prob = calculate_composite_score_for_position(
                        stock_price, strike, days_left, current_iv, tech_score_pos
                    )
                    
                    # Get hybrid recommendation
                    iv_percentile = 50
                    rec_icon_full, rec_reason = get_hybrid_recommendation(
                        option_price, entry_price, target, stop, days_left, delta_calc, 0, current_iv,
                        cts, ev, tech_score_pos, ema_status, macd_trend, macd_days, touch_prob, iv_percentile
                    )
                    
                    # Extract short icon for summary
                    rec_icon = rec_icon_full.split()[0]
                    rec_text = " ".join(rec_icon_full.split()[1:])[:30]
                    
                    summary = f"{rec_icon} {ticker} ${strike:.2f} Call | Exp: {expiry_date_str} | ${option_price:.2f} | P&L: {pnl_pct:+.1f}% (${pnl:+.0f})"
                    
                    with st.expander(summary):
                        st.markdown("### 📊 Position Summary")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Current Option Price", f"${option_price:.2f}")
                            # Fix P&L color - red for negative, green for positive
                            pnl_delta_color = "inverse" if pnl < 0 else "normal"
                            st.metric("P&L", f"{pnl_pct:+.1f}%", delta=f"${pnl:+.0f}", delta_color=pnl_delta_color)
                        with col2:
                            st.metric("Days Left", f"{days_left}")
                            st.metric("Delta", f"{delta_calc:.3f}")
                        with col3:
                            st.metric("Entry Price", f"${entry_price:.2f}")
                            st.metric("Contracts", contracts)
                        
                        st.markdown("---")
                        st.markdown("### 💡 Recommendation")
                        st.info(f"**{rec_icon_full}**")
                        st.caption(rec_reason)
                        
                        st.markdown("---")
                        st.markdown("### 📈 Quant Analytics")
                        col_q1, col_q2, col_q3 = st.columns(3)
                        with col_q1:
                            st.metric("Expected Value (EV)", f"${ev:.2f}")
                            st.metric("Composite Score", f"{cts}/100")
                            st.metric("Touch Probability", f"{touch_prob*100:.0f}%")
                        with col_q2:
                            # Calculate Theta decay
                            if option_price > 0:
                                theta_est = (option_price * 0.02) / 365  # Approximate theta
                                theta_pct = (theta_est / option_price) * 100 if option_price > 0 else 0
                                st.metric("Theta Decay", f"{theta_pct:.1f}%/day")
                            else:
                                st.metric("Theta Decay", "N/A")
                            st.metric("IV", f"{current_iv*100:.1f}%")
                        with col_q3:
                            st.metric("Technical Score", f"{tech_score_pos}/3")
                            ema_text = "🟢 Bullish" if ema_status in ["bullish", "bullish_cross"] else ("🔴 Bearish" if ema_status in ["bearish", "bearish_cross"] else "⚪ Neutral")
                            st.metric("8/20 EMA", ema_text)
                            if ema_status == "bullish_cross":
                                st.caption("🔥 Just crossed bullish")
                            elif ema_status == "bearish_cross":
                                st.caption("⚠️ Just crossed bearish")
                        
                        st.markdown("---")
                        st.markdown("### 🎯 Targets")
                        col_target1, col_target2 = st.columns(2)
                        
                        with col_target1:
                            # STOP LOSS (left)
                            st.metric("🛑 Stop Loss", f"${stop:.2f}")
                            if option_price > stop:
                                stop_distance_abs = option_price - stop
                                stop_distance_pct = (stop_distance_abs / stop) * 100
                                st.caption(f"✅ ${stop_distance_abs:.2f} above stop (+{stop_distance_pct:.0f}%)")
                            else:
                                stop_distance_abs = stop - option_price
                                stop_distance_pct = (stop_distance_abs / stop) * 100
                                st.caption(f"⚠️ ${stop_distance_abs:.2f} below stop (-{stop_distance_pct:.0f}%)")
                            # Progress bar showing how close to stop (inverse)
                            stop_progress = max(0, min(1, 1 - ((option_price - stop) / (target - stop)))) if target > stop else 0.5
                            st.progress(stop_progress)
                        
                        with col_target2:
                            # TARGET (right)
                            st.metric("🎯 Target", f"${target:.2f}")
                            if option_price < target:
                                target_distance_abs = target - option_price
                                target_distance_pct = (target_distance_abs / option_price) * 100
                                st.caption(f"📈 Need +${target_distance_abs:.2f} (+{target_distance_pct:.0f}%) to target")
                                progress = option_price / target
                            else:
                                target_distance_abs = option_price - target
                                target_distance_pct = (target_distance_abs / target) * 100
                                st.caption(f"✅ Target exceeded by ${target_distance_abs:.2f} (+{target_distance_pct:.0f}%)")
                                progress = 1.0
                            st.progress(min(progress, 1.0))
                        
                        st.markdown("---")
                        confirm_key = f"confirm_close_{idx}"
                        if st.checkbox("Confirm close position", key=confirm_key):
                            if st.button("❌ Close Position", key=f"close_{idx}", use_container_width=True):
                                close_position(row_idx)
                                st.success(f"✅ Position {ticker} ${strike:.2f} Call closed!")
                                time.sleep(1)
                                st.rerun()
                else:
                    with st.expander(f"⚠️ {ticker} ${strike:.2f} Call | Data unavailable"):
                        st.warning(f"Option price data not available")
                        confirm_key = f"confirm_close_{idx}"
                        if st.checkbox("Confirm close position", key=confirm_key):
                            if st.button("❌ Close Position", key=f"close_{idx}", use_container_width=True):
                                close_position(row_idx)
                                st.success(f"Position {ticker} ${strike:.2f} Call closed!")
                                st.rerun()
        else:
            st.info(f"No active positions for {selected_trader}. Add a position below.")
        
        st.divider()
        
        # --- ADD NEW POSITION FORM ---
        st.subheader("➕ Add New Position")
        
        if not st.session_state.expiries:
            st.warning("Please analyze a ticker first (click 'Analyze Options Structure') before adding positions.")
        else:
            current_ticker = st.session_state.current_ticker if st.session_state.current_ticker else "SHOP"
            expiry_options = st.session_state.expiries
            default_expiry_index = 0
            if 'expiry' in dir() and expiry in expiry_options:
                default_expiry_index = expiry_options.index(expiry)
            elif st.session_state.get('last_selected_expiry') in expiry_options:
                default_expiry_index = expiry_options.index(st.session_state.last_selected_expiry)
            
            selected_expiry_str = st.selectbox("Expiry Date:", options=expiry_options, index=default_expiry_index, key="portfolio_expiry_select")
            st.session_state.last_selected_expiry = selected_expiry_str
            
            cons_strike, cons_mid = get_conservative_strike_for_expiry(
                current_ticker, selected_expiry_str, st.session_state.profit_target_pct, st.session_state.stop_loss_pct
            )
            all_strikes = get_strikes_for_expiry(current_ticker, selected_expiry_str)
            
            if not all_strikes:
                st.warning(f"No option data available for {current_ticker} on {selected_expiry_str}")
            else:
                strike_options = [s['strike'] for s in all_strikes]
                default_strike_index = 0
                if cons_strike and cons_strike in strike_options:
                    default_strike_index = strike_options.index(cons_strike)
                selected_strike = st.selectbox("Strike Price:", options=strike_options, index=default_strike_index, key="portfolio_strike_select")
                selected_mid = next((s['mid'] for s in all_strikes if s['strike'] == selected_strike), None)
                
                if cons_strike and cons_strike == selected_strike:
                    st.caption(f"⭐ Recommended strike - Mid: ${selected_mid:.2f}")
                elif cons_strike:
                    st.caption(f"💡 Conservative recommendation: ${cons_strike:.2f} (Mid: ${cons_mid:.2f})")

                                # Show email reminder if trader has no email
                trader_email = get_trader_email(selected_trader)
                if not trader_email:
                    st.warning(f"⚠️ No email configured for {selected_trader}. Add email when creating trader to receive alerts.")
                else:
                    st.caption(f"📧 Alerts will be sent to: {trader_email}")
                
                st.divider()
              
                with st.form("add_position_form"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        ticker_pos = st.text_input("Ticker:", value=current_ticker).upper()
                    with col2:
                        contracts_pos = st.number_input("Contracts:", min_value=1, step=1)
                    with col3:
                        default_entry = selected_mid if selected_mid else 0.01
                        entry_price_pos = st.number_input("Entry Price:", min_value=0.01, step=0.05, format="%.2f", value=default_entry)
                    
                    target_auto = entry_price_pos * (1 + st.session_state.profit_target_pct / 100)
                    stop_auto = entry_price_pos * (1 - st.session_state.stop_loss_pct / 100)
                    st.info(f"🎯 Target: ${target_auto:.2f} | 🛑 Stop: ${stop_auto:.2f}")
                    
                    expiry_pos = pd.to_datetime(selected_expiry_str).date()
                    cutoff_days = min((expiry_pos - datetime.now().date()).days, 45)
                    cutoff_date = datetime.now().date() + timedelta(days=max(cutoff_days, 1))
                    
                    submitted = st.form_submit_button("💾 Save Position", use_container_width=True, type="primary")
                    if submitted:
                        success = add_position_to_sheet(
                            trader_name=selected_trader, ticker=ticker_pos, strike=selected_strike,
                            expiry=selected_expiry_str, contracts=contracts_pos, entry_price=entry_price_pos,
                            entry_iv=0.35, entry_delta=0.50, target_price=target_auto,
                            stop_loss=stop_auto, cutoff_date=cutoff_date.strftime('%Y-%m-%d')
                        )
                        if success:
                            st.success(f"✅ Position added for {selected_trader}!")
                            st.balloons()
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error("❌ Failed to save. Check Google Sheets connection.")

    # ========================
    # STRATEGY GUIDE
    # ========================
    with t_edu:
        st.header("📖 Complete Strategy Guide & Indicator Dictionary")
        st.markdown("""
        Welcome to the complete trading manual. This guide explains every indicator in the app and provides 
        **actionable guidelines** on how to use them for real trading decisions.
        """)
        
        st.divider()
        
        st.subheader("🎯 Recommendation Types - Quick Reference")
        st.markdown("""
        | Icon | Recommendation | Meaning |
        |------|----------------|---------|
        | 🔴 | EXIT - STOP LOSS | Hit your predefined stop loss |
        | 🔴 | EXIT - NO EDGE | Expected Value turned negative |
        | 🔴 | SELL VOL | IV overpriced (>90th percentile) |
        | 🟢 | TAKE PROFITS | Target reached |
        | 🟡 | PARTIAL EXIT | 80%+ to target with high touch probability |
        | 🟠 | TIME DECAY | <7 days left or delta <0.25 |
        | 🟠 | TECHNICAL EXIT | Bearish crossover or MACD falling |
        | 🟢 | ADD MORE | High conviction opportunity |
        | 🔵 | STRONG HOLD | All metrics aligned |
        | 🔵 | HOLD | Normal, continue monitoring |
        """)
        
        st.divider()
        
        st.subheader("💧 Liquidity Indicators - Your First Filter")
        st.markdown("**Before looking at any other metric, check liquidity first.**")
        
        with st.expander("📊 Volume Today - How to Use", expanded=False):
            st.markdown("""
            **Guidelines for Use:**
            | Volume | Rating | Action |
            |--------|--------|--------|
            | 200+ | 🟢 EXCELLENT | Safe to trade any position size |
            | 100-199 | 🟡 GOOD | Acceptable for positions under 50 contracts |
            | 50-99 | 🟠 CAUTION | Only for positions under 10 contracts |
            | 10-49 | 🔴 DANGER | Avoid unless absolutely necessary |
            | <10 | ⚫ TOXIC | NEVER TRADE - you won't exit |
            """)
        
        with st.expander("💰 Bid-Ask Spread - Your Real Transaction Cost", expanded=False):
            st.markdown("""
            **Spread Percentage Rule of Thumb:**
            | Spread % | Grade | Trading Implication |
            |----------|-------|---------------------|
            | < 2% | 🟢 EXCELLENT | Round-trip cost <4% |
            | 2-5% | 🟡 ACCEPTABLE | Round-trip cost 4-10% |
            | 5-10% | 🟠 WIDE | Cost 10-20% |
            | > 10% | 🔴 TOXIC | Cost >20% - AVOID |
            """)
        
        st.divider()
        
        st.subheader("📊 Greeks - Understanding Your Risk Exposures")
        st.markdown("""
        | Greek | What It Measures | Target Range |
        |-------|------------------|--------------|
        | Delta (Δ) | Probability of profit / Price sensitivity | Conservative: 0.50-0.60 |
        | Theta (θ) | Daily time decay | < 2% of premium/day |
        | Vega (ν) | Volatility exposure | Lower for longer holds |
        | Gamma (Γ) | Delta acceleration | Manageable with 60+ DTE |
        """)
        
        st.divider()
        
        st.subheader("✅ Complete Trade Decision Framework")
        st.markdown("""
        **Step 1: Liquidity Filter (MANDATORY)**
        - [ ] Volume > 50 (preferably > 200)
        - [ ] Open Interest > 500
        - [ ] Spread < 5% (preferably < 3%)
        
        **Step 2: Strategy Match**
        - [ ] Delta matches your risk profile
        - [ ] Days to expiry > 60
        
        **Step 3: Math Confirmation**
        - [ ] Composite Score > 55
        - [ ] Expected Value > 0.25
        
        **Step 4: Technical Confirmation**
        - [ ] 8 EMA > 20 EMA (bullish)
        - [ ] MACD histogram rising
        
        **Step 5: Trade Management Plan**
        - [ ] Take profit at 40% above entry
        - [ ] Stop loss at 30% below entry
        - [ ] Exit by cutoff date
        """)
        
        st.warning("""
        **⚠️ Remember:** No indicator is perfect. Always size positions appropriately 
        (never risk more than 1-2% of account per trade) and follow your stop losses.
        """)
        
        st.success("""
        **💡 Pro Tip:** The best trades happen when ALL four filters pass:
        1. ✅ Liquidity (Volume + OI + Spread)
        2. ✅ Math (CTS + EV)
        3. ✅ Technicals (EMAs + MACD)
        4. ✅ Management (Pre-set exits)
        """)

else:
    st.info("👈 Input a valid trading ticker to trigger the options matrix models.")
