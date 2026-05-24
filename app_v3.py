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
st.set_page_config(page_title="Analyst Pro Options Suite v2", layout="wide")

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

# --- GOOGLE SHEETS CONNECTION ---
@st.cache_resource
def get_google_sheet():
    """Connect to Google Sheets using service account credentials."""
    try:
        # Load credentials from secrets (JSON string)
        creds_json = st.secrets["GOOGLE_SHEETS_CREDENTIALS"]
        creds_dict = json.loads(creds_json)
        
        # Define the scope
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        
        # Create credentials object
        creds = Credentials.from_service_account_info(creds_dict, scopes=scope)
        
        # Connect to Google Sheets
        client = gspread.authorize(creds)
        
        # Open the spreadsheet by ID
        sheet_id = st.secrets["SPREADSHEET_ID"]
        sheet = client.open_by_key(sheet_id)
        
        return sheet
    except Exception as e:
        st.error(f"Failed to connect to Google Sheets: {str(e)}")
        return None

# --- PORTFOLIO FUNCTIONS ---
def init_portfolio_sheet():
    """Initialize the portfolio sheet if it doesn't exist."""
    sheet = get_google_sheet()
    if not sheet:
        return None
    
    try:
        # Try to get the Portfolio worksheet
        worksheet = sheet.worksheet("Portfolio")
    except:
        # Create it if it doesn't exist
        worksheet = sheet.add_worksheet(title="Portfolio", rows="1000", cols="20")
        
        # Add headers
        headers = [
            "timestamp", "trader_name", "ticker", "strike", "expiry", 
            "contracts", "entry_price", "target_price", "stop_loss", "cutoff_date",
            "entry_iv", "entry_delta", "status", "last_recommendation", "last_alert_sent"
        ]
        worksheet.append_row(headers)
    
    return worksheet

def add_position_to_sheet(trader_name, ticker, strike, expiry, contracts, entry_price, 
                          entry_iv, entry_delta, target_price, stop_loss, cutoff_date):
    """Add a new position to Google Sheets."""
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
    """Get all portfolio positions, optionally filtered by trader name."""
    worksheet = init_portfolio_sheet()
    if not worksheet:
        return []
    
    records = worksheet.get_all_records()
    positions = []
    
    for record in records:
        if record.get("status") == "active":
            if trader_name and record.get("trader_name") != trader_name:
                continue
            positions.append(record)
    
    return positions

def update_position_recommendation(row_index, recommendation):
    """Update the last_recommendation for a position."""
    worksheet = init_portfolio_sheet()
    if not worksheet:
        return
    
    # Find the row by index (add 2 because of 1-indexing and header row)
    worksheet.update_cell(row_index + 2, 14, recommendation)  # Column 14 = last_recommendation

def close_position(row_index):
    """Mark a position as closed."""
    worksheet = init_portfolio_sheet()
    if not worksheet:
        return
    
    worksheet.update_cell(row_index + 2, 13, "closed")  # Column 13 = status

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
                    st.warning(f"⏳ Groq rate limit hit. Waiting {wait_time} seconds before retry...")
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

# --- FETCH NEWS FROM FINNHUB ---
def fetch_news_finnhub(ticker):
    """Fetch latest news for a ticker using Finnhub API."""
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

# --- AI RESEARCH ENGINE ---
def get_ai_research(ticker):
    groq_api_key = st.secrets.get("GROQ_API_KEY")
    finnhub_api_key = st.secrets.get("FINNHUB_API_KEY")
    
    if not groq_api_key:
        return "⚠️ Please add GROQ_API_KEY to Streamlit Secrets."
    
    cache_key = f"news_summary_{ticker}"
    cached_response = st.session_state.ai_cache.get(cache_key)
    if cached_response:
        return cached_response
    
    news_articles = fetch_news_finnhub(ticker)
    
    if not news_articles:
        return f"ℹ️ No recent news found for {ticker} in the last 7 days."
    
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
    
    1. **Analyst Consensus**: What are analysts saying?
    2. **Key Catalysts**: Upcoming events, earnings dates mentioned
    3. **Sentiment Drivers**: Top 3 themes from the last 7 days
    4. **Actionable View**: Give a "Bullish", "Neutral", or "Cautious" rating with 1-sentence reasoning
    
    Keep it factual and concise.
    """
    
    try:
        client = Groq(api_key=groq_api_key)
        response_text = call_groq_with_retry(client, prompt)
        
        if response_text is None:
            fallback = f"### 📰 Recent News for {ticker}\n\n"
            for i, item in enumerate(news_articles[:5]):
                fallback += f"**{i+1}. {item['title']}**  \n"
                fallback += f"📌 Source: {item['publisher']} | 🕐 {item['datetime']}  \n"
                fallback += f"🔗 [Read full article]({item['link']})  \n\n"
            result = fallback
        else:
            sources_text = "\n".join([f"- [{item['title']}]({item['link']}) ({item['publisher']})" for item in news_articles[:5]])
            result = f"### 📰 AI Summary for {ticker}\n\n{response_text}\n\n---\n### 🔗 Sources\n{sources_text}"
        
        st.session_state.ai_cache.set(cache_key, result)
        return result
        
    except Exception as e:
        fallback = f"### 📰 Recent News for {ticker}\n\n"
        for i, item in enumerate(news_articles[:5]):
            fallback += f"**{i+1}. {item['title']}**  \n"
            fallback += f"📌 {item['publisher']} | 🕐 {item['datetime']}  \n"
            fallback += f"🔗 [Read full article]({item['link']})  \n\n"
        return fallback

# --- PORTFOLIO RECOMMENDATION ENGINE ---
def get_position_recommendation(position, current_price, current_iv, current_delta, days_left):
    """Generate recommendation for a single position."""
    entry = float(position['entry_price'])
    current = current_price
    target = float(position['target_price'])
    stop = float(position['stop_loss'])
    
    pnl_pct = ((current - entry) / entry) * 100
    
    # Check stop loss first
    if current <= stop:
        return "🔴 SELL IMMEDIATELY", f"Stop loss hit at ${stop:.2f} (Entry: ${entry:.2f})"
    
    # Check profit target
    if current >= target:
        return "🟢 TAKE PROFIT", f"Target reached at ${target:.2f} (Entry: ${entry:.2f})"
    
    # Check partial profit (80% of target)
    if current >= target * 0.8:
        return "🟡 PARTIAL PROFIT", f"80% of target reached. Consider taking partial profits."
    
    # Check days left
    if days_left < 7:
        return "🟠 EXIT SOON", f"Only {days_left} days left. Time decay accelerating."
    
    # Check if edge is still positive
    if current_delta < 0.25:
        return "🟠 EXIT", f"Delta dropped to {current_delta:.2f}. Probability decreased."
    
    # Check if we should add more
    if pnl_pct > 0 and pnl_pct < 20 and current_delta > 0.45:
        return "🟢 ADD MORE", f"Position is working. Consider adding at ${current:.2f}"
    
    # Default hold
    return "🔵 HOLD", f"Position intact. Target: ${target:.2f}, Stop: ${stop:.2f}"

# --- SESSION STATE INITIALIZATION ---
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

# --- DATA FETCHING & GLOBAL SCANS (Same as before) ---
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
    
    # Updated tabs to include Portfolio
    t_summary, t_cons, t_aggr, t_spec, t_tech, t_ai, t_portfolio, t_edu = st.tabs([
        "📋 Global Recs", "🛡️ Conservative", "⚡ Aggressive", 
        "🎰 Speculative", "📊 Technical", "🤖 AI Research", "📂 Portfolio", "📖 Strategy Guide"
    ])

    # --- SUMMARY TAB (Keep your existing code) ---
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

    # --- STRATEGY TABS (Placeholders - keep your existing implementation) ---
    with t_cons:
        st.info("Conservative strategy details - select expiry from sidebar")
        # Your existing conservative logic here
    
    with t_aggr:
        st.info("Aggressive strategy details - select expiry from sidebar")
    
    with t_spec:
        st.info("Speculative strategy details - select expiry from sidebar")
    
    with t_tech:
        if not st.session_state.hist_data.empty:
            st.subheader("Momentum & Volatility Health")
            st.line_chart(st.session_state.hist_data[['Close']])
        else:
            st.info("Technical analysis available after fetching data")
    
    with t_ai:
        if not st.session_state.ai_brief:
            with st.spinner("Fetching AI analysis..."):
                st.session_state.ai_brief = get_ai_research(st.session_state.current_ticker)
        st.markdown(st.session_state.ai_brief)

    # ========================
    # NEW PORTFOLIO TAB
    # ========================
        # --- PORTFOLIO TAB (UPDATED) ---
    with t_portfolio:
        st.header("📂 Options Portfolio Tracker")
        
        if 'sheet_initialized' not in st.session_state:
            init_portfolio_sheet()
            st.session_state.sheet_initialized = True
        
        # --- TRADER MANAGEMENT (Dynamic from Google Sheets) ---
        def get_trader_list():
            """Get unique trader names from the portfolio sheet."""
            worksheet = init_portfolio_sheet()
            if not worksheet:
                return ["Trader 1", "Trader 2", "Trader 3"]
            records = worksheet.get_all_records()
            traders = set()
            for record in records:
                if record.get('trader_name'):
                    traders.add(record['trader_name'])
            # Add default if empty
            if not traders:
                traders = {"Trader 1", "Trader 2", "Trader 3"}
            return sorted(list(traders))
        
        # Option to add new trader on the fly
        col_t1, col_t2 = st.columns([3, 1])
        with col_t1:
            trader_list = get_trader_list()
            selected_trader = st.selectbox("Select Trader:", trader_list, key="trader_select")
        with col_t2:
            st.write("")
            st.write("")
            if st.button("➕ New Trader", key="add_trader_btn"):
                st.session_state.show_new_trader = True
        
        # Show input for new trader name
        if st.session_state.get('show_new_trader', False):
            col_n1, col_n2 = st.columns([2, 1])
            with col_n1:
                new_trader = st.text_input("Enter new trader name:", key="new_trader_name")
            with col_n2:
                st.write("")
                st.write("")
                if st.button("Save Trader", key="save_trader"):
                    if new_trader and new_trader not in trader_list:
                        st.session_state.new_trader_added = new_trader
                        st.success(f"Trader '{new_trader}' added! Select from dropdown.")
                        st.session_state.show_new_trader = False
                        st.rerun()
                    else:
                        st.error("Please enter a valid name")
        
        st.divider()
        
        # --- DISPLAY ACTIVE POSITIONS ---
        st.subheader("📊 Active Positions")
        positions = get_portfolio_positions(selected_trader if selected_trader != "Admin" else None)
        
        col_r1, col_r2 = st.columns([1, 4])
        with col_r1:
            if st.button("🔄 Refresh", key="refresh_portfolio", use_container_width=True):
                st.rerun()
        
        if positions:
            for idx, (row_idx, pos) in enumerate(positions):
                with st.container():
                    st.markdown(f"---")
                    col1, col2, col3 = st.columns([2, 2, 1])
                    
                    with col1:
                        st.markdown(f"**{pos['ticker']} ${float(pos['strike']):.2f} Call**")
                        st.caption(f"Expiry: {pos['expiry']}")
                        st.caption(f"Contracts: {pos['contracts']} @ ${float(pos['entry_price']):.2f}")
                        st.caption(f"Target: ${float(pos['target_price']):.2f} | Stop: ${float(pos['stop_loss']):.2f}")
                    
                    with col2:
                        try:
                            stock = yf.Ticker(pos['ticker'])
                            current_price = stock.history(period="1d")['Close'].iloc[-1]
                            expiry_date = pd.to_datetime(pos['expiry']).date()
                            days_left = max((expiry_date - datetime.now().date()).days, 0)
                            
                            # Get current Greeks
                            try:
                                option_chain = stock.option_chain(pos['expiry'])
                                option_row = option_chain.calls[option_chain.calls['strike'] == float(pos['strike'])]
                                if not option_row.empty:
                                    current_iv = option_row['impliedVolatility'].iloc[0]
                                    d, _, _, _ = calculate_greeks(
                                        current_price, float(pos['strike']), max(days_left, 1)/365, 0.05, current_iv
                                    )
                                    current_delta = d
                                else:
                                    current_iv = 0.35
                                    current_delta = 0.5
                            except:
                                current_iv = 0.35
                                current_delta = 0.5
                            
                            pnl = (current_price - float(pos['entry_price'])) * int(pos['contracts']) * 100
                            pnl_pct = ((current_price - float(pos['entry_price'])) / float(pos['entry_price'])) * 100
                            
                            st.metric("Current Price", f"${current_price:.2f}", 
                                     delta=f"{pnl_pct:+.1f}%", 
                                     delta_color="normal")
                            
                            # Show P&L in dollars
                            if pnl >= 0:
                                st.caption(f"💰 P&L: +${pnl:.0f}")
                            else:
                                st.caption(f"💰 P&L: -${abs(pnl):.0f}")
                            
                            recommendation, reason = get_position_recommendation(
                                pos, current_price, current_iv, current_delta, days_left
                            )
                            
                            # Color-coded recommendation
                            if "SELL" in recommendation or "EXIT" in recommendation:
                                st.error(f"**{recommendation}**")
                            elif "PROFIT" in recommendation:
                                st.success(f"**{recommendation}**")
                            elif "ADD" in recommendation:
                                st.info(f"**{recommendation}**")
                            else:
                                st.info(f"**{recommendation}**")
                            st.caption(reason)
                            st.caption(f"Days left: {days_left} | Delta: {current_delta:.2f}")
                        except Exception as e:
                            st.caption(f"⚠️ Data temporarily unavailable")
                        
                    with col3:
                        if st.button("❌ Close", key=f"close_{idx}", use_container_width=True):
                            close_position(row_idx)
                            st.rerun()
        else:
            st.info(f"No active positions for {selected_trader}")
        
        st.divider()
        
        # --- ADD NEW POSITION (Auto-populated from sidebar!) ---
        st.subheader("➕ Add New Position")
        st.caption(f"Auto-filled from sidebar: Ticker = {st.session_state.current_ticker}, Selected Expiry = {expiry}")
        
        with st.form("add_position_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Auto-populate ticker from sidebar, but allow override
                ticker_pos = st.text_input(
                    "Ticker:", 
                    value=st.session_state.current_ticker if st.session_state.current_ticker else "SHOP",
                    help="Auto-filled from sidebar. Change if needed."
                ).upper()
                
                strike_pos = st.number_input(
                    "Strike Price:", 
                    min_value=1.0, 
                    step=0.5, 
                    format="%.2f",
                    help="The strike price of the option"
                )
            
            with col2:
                # Auto-populate expiry from sidebar
                default_expiry = expiry if 'expiry' in dir() else datetime.now().date() + timedelta(days=60)
                expiry_pos = st.date_input(
                    "Expiry Date:", 
                    value=pd.to_datetime(expiry).date() if 'expiry' in dir() else default_expiry,
                    min_value=datetime.now().date(),
                    help="Auto-filled from sidebar. Change if needed."
                )
                
                contracts_pos = st.number_input(
                    "Number of Contracts:", 
                    min_value=1, 
                    step=1,
                    help="Each contract = 100 shares"
                )
            
            with col3:
                entry_price_pos = st.number_input(
                    "Entry Price (per contract):", 
                    min_value=0.01, 
                    step=0.05, 
                    format="%.2f",
                    help="The price you paid per contract"
                )
                
                # Auto-calculate targets based on current adjusters
                target_auto = entry_price_pos * (1 + st.session_state.profit_target_pct / 100)
                stop_auto = entry_price_pos * (1 - st.session_state.stop_loss_pct / 100)
                
                st.info(f"""
                **Auto-calculated targets:**
                - Target: ${target_auto:.2f} ({st.session_state.profit_target_pct}% above entry)
                - Stop: ${stop_auto:.2f} ({st.session_state.stop_loss_pct}% below entry)
                """)
            
            # Hidden auto-calculations
            cutoff_days = min((expiry_pos - datetime.now().date()).days, 45)
            cutoff_date = datetime.now().date() + timedelta(days=max(cutoff_days, 1))
            
            # Estimate IV and Delta (will be updated when position is saved)
            try:
                stock_temp = yf.Ticker(ticker_pos)
                current_price_temp = stock_temp.history(period="1d")['Close'].iloc[-1]
                option_chain_temp = stock_temp.option_chain(expiry_pos.strftime('%Y-%m-%d'))
                option_row_temp = option_chain_temp.calls[option_chain_temp.calls['strike'] == strike_pos]
                if not option_row_temp.empty:
                    entry_iv = option_row_temp['impliedVolatility'].iloc[0]
                    days_to_exp = (expiry_pos - datetime.now().date()).days
                    d_temp, _, _, _ = calculate_greeks(
                        current_price_temp, strike_pos, max(days_to_exp, 1) / 365, 0.05, entry_iv
                    )
                    entry_delta = d_temp
                else:
                    entry_iv = 0.35
                    entry_delta = 0.50
            except:
                entry_iv = 0.35
                entry_delta = 0.50
            
            submitted = st.form_submit_button("💾 Save Position", use_container_width=True, type="primary")
            
            if submitted:
                if strike_pos <= 0 or entry_price_pos <= 0:
                    st.error("Please enter valid strike price and entry price")
                else:
                    success = add_position_to_sheet(
                        trader_name=selected_trader,
                        ticker=ticker_pos,
                        strike=strike_pos,
                        expiry=expiry_pos.strftime('%Y-%m-%d'),
                        contracts=contracts_pos,
                        entry_price=entry_price_pos,
                        entry_iv=entry_iv,
                        entry_delta=entry_delta,
                        target_price=target_auto,
                        stop_loss=stop_auto,
                        cutoff_date=cutoff_date.strftime('%Y-%m-%d')
                    )
                    if success:
                        st.success(f"✅ Position added for {selected_trader}!")
                        st.balloons()
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("❌ Failed to save. Check Google Sheets connection.")
    
    with t_edu:
        st.header("📖 Strategy Guide")
        st.markdown("""
        **Quick Reference:**
        - 🟢 ADD MORE - Position profitable, add contracts
        - 🔵 HOLD - Position intact, monitor
        - 🟡 PARTIAL PROFIT - Take some off the table
        - 🟢 TAKE PROFIT - Target reached, exit
        - 🔴 SELL IMMEDIATELY - Stop loss hit
        - 🟠 EXIT - Edge deteriorated or time running out
        """)

else:
    st.info("👈 Input a valid trading ticker to trigger the options matrix models.")
