import streamlit as st
import yfinance as yf
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import norm
from groq import Groq
import time
import gspread
from google.oauth2.service_account import Credentials
import json
import warnings
warnings.filterwarnings('ignore')

# --- PAGE CONFIG MUST BE FIRST ---
st.set_page_config(page_title="Analyst Pro Options Suite v5", layout="wide")

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

# --- CACHING FOR YFINANCE DATA (Prevents Rate Limits) ---
@st.cache_data(ttl=300, show_spinner=False)
def get_cached_option_chain(ticker, expiry):
    try:
        stock = yf.Ticker(ticker)
        opt_chain = stock.option_chain(expiry)
        return opt_chain.calls, opt_chain.puts
    except Exception as e:
        st.error(f"Error fetching option chain: {str(e)}")
        return None, None

@st.cache_data(ttl=300, show_spinner=False)
def get_cached_stock_history(ticker, period="100d"):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        return hist
    except Exception as e:
        st.error(f"Error fetching stock history: {str(e)}")
        return None

@st.cache_data(ttl=300, show_spinner=False)
def get_cached_stock_info(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = dict(stock.info)
        return info
    except Exception as e:
        st.error(f"Error fetching stock info: {str(e)}")
        return None

@st.cache_data(ttl=60, show_spinner=False)
def get_cached_current_price(ticker):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1d")
        if not hist.empty:
            return hist['Close'].iloc[-1]
        return None
    except Exception as e:
        return None

# --- NEW: ADVANCED TECHNICAL INDICATORS (Phase 1) ---
def calculate_atr(df, period=14):
    """Calculate Average True Range for volatility measurement."""
    high, low, close = df['High'], df['Low'], df['Close']
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean().iloc[-1]
    atr_pct = (atr / close.iloc[-1]) * 100
    return round(atr, 2), round(atr_pct, 1)

def calculate_rsi(df, period=14):
    """Calculate Relative Strength Index for overbought/oversold detection."""
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return round(rsi.iloc[-1], 1)

def calculate_hv(df, period=20):
    """Calculate Historical Volatility (Realized Vol)."""
    returns = np.log(df['Close'] / df['Close'].shift(1))
    hv = returns.std() * np.sqrt(252)
    return round(hv.iloc[-1] * 100, 1)

def get_earnings_date(ticker):
    """Get next earnings date from yfinance."""
    try:
        stock = yf.Ticker(ticker)
        calendar = stock.calendar
        if not calendar.empty and 'Earnings Date' in calendar.index:
            earnings_date = calendar.loc['Earnings Date']
            if isinstance(earnings_date, pd.Series):
                earnings_date = earnings_date.iloc[0]
            if isinstance(earnings_date, (datetime, pd.Timestamp)):
                return earnings_date.date()
        return None
    except:
        return None

def get_vix():
    """Get current VIX value for market regime."""
    try:
        vix = yf.Ticker("^VIX")
        hist = vix.history(period="1d")
        if not hist.empty:
            return round(hist['Close'].iloc[-1], 1)
        return 15.0
    except:
        return 15.0

def get_market_regime(vix):
    """Determine market regime based on VIX."""
    if vix < 12:
        return "🟢 LOW VOL", "Options cheap, but low probability of large moves. Favorable for selling premium."
    elif vix < 20:
        return "🟢 OPTIMAL", "Normal volatility regime. 60 DTE calls favorable."
    elif vix < 25:
        return "🟡 ELEVATED", "Elevated volatility. Options expensive but higher gamma potential."
    else:
        return "🔴 HIGH VOL", "High volatility. Options overpriced. Caution with long calls."

# --- GOOGLE SHEETS CONNECTION (for Portfolio) ---
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
        worksheet = sheet.add_worksheet(title="Portfolio", rows="1000", cols="25")
        headers = [
            "timestamp", "trader_name", "ticker", "strike", "expiry", 
            "contracts", "entry_price", "target_price", "stop_loss", "cutoff_date",
            "entry_iv", "entry_delta", "status", "last_recommendation", "last_alert_sent",
            "total_cost", "total_contracts_purchased", "sold_contracts", "realized_pnl", "avg_entry_price"
        ]
        worksheet.append_row(headers)
    return worksheet

def add_position_to_sheet(trader_name, ticker, strike, expiry, contracts, entry_price, 
                          entry_iv, entry_delta, target_price, stop_loss, cutoff_date):
    worksheet = init_portfolio_sheet()
    if not worksheet:
        return False
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total_cost = contracts * entry_price
    row = [
        now, trader_name, ticker, strike, expiry, contracts, entry_price,
        target_price, stop_loss, cutoff_date, entry_iv, entry_delta,
        "active", "HOLD", "", total_cost, contracts, 0, 0, entry_price
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

def get_all_positions_for_trader(trader_name):
    worksheet = init_portfolio_sheet()
    if not worksheet:
        return []
    records = worksheet.get_all_records()
    positions = []
    for record in records:
        if record.get("trader_name") == trader_name:
            positions.append(record)
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

def add_trader_to_sheet(trader_name, email):
    sheet = get_google_sheet()
    if not sheet:
        return False
    try:
        traders_worksheet = sheet.worksheet("Traders")
    except:
        traders_worksheet = sheet.add_worksheet(title="Traders", rows="100", cols="10")
        traders_worksheet.append_row(["trader_name", "email", "enabled", "created_at"])
    
    existing = traders_worksheet.findall(trader_name)
    if existing:
        return False
    
    traders_worksheet.append_row([
        trader_name, email, "TRUE", datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ])
    return True

def get_trader_email(trader_name):
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

# --- PORTFOLIO MANAGEMENT FUNCTIONS ---
def update_position_after_add(row_index, additional_contracts, additional_price):
    worksheet = init_portfolio_sheet()
    if not worksheet:
        return False
    
    current_contracts = int(worksheet.cell(row_index + 2, 6).value)
    current_entry = float(worksheet.cell(row_index + 2, 7).value)
    current_total_cost = float(worksheet.cell(row_index + 2, 16).value) if worksheet.cell(row_index + 2, 16).value else current_contracts * current_entry
    current_total_purchased = int(worksheet.cell(row_index + 2, 17).value) if worksheet.cell(row_index + 2, 17).value else current_contracts
    
    new_total_cost = current_total_cost + (additional_contracts * additional_price)
    new_total_purchased = current_total_purchased + additional_contracts
    new_avg_price = new_total_cost / new_total_purchased
    new_current_contracts = new_total_purchased - (int(worksheet.cell(row_index + 2, 18).value) if worksheet.cell(row_index + 2, 18).value else 0)
    
    profit_target_pct = st.session_state.profit_target_pct
    stop_loss_pct = st.session_state.stop_loss_pct
    new_target = new_avg_price * (1 + profit_target_pct / 100)
    new_stop = new_avg_price * (1 - stop_loss_pct / 100)
    
    expiry_str = worksheet.cell(row_index + 2, 5).value
    expiry_date = pd.to_datetime(expiry_str).date()
    days_left = (expiry_date - datetime.now().date()).days
    new_cutoff = expiry_date - timedelta(days=min(int(days_left * 0.4), 45)) if days_left > 0 else expiry_date
    
    worksheet.update_cell(row_index + 2, 6, new_current_contracts)
    worksheet.update_cell(row_index + 2, 7, new_avg_price)
    worksheet.update_cell(row_index + 2, 8, new_target)
    worksheet.update_cell(row_index + 2, 9, new_stop)
    worksheet.update_cell(row_index + 2, 10, new_cutoff.strftime('%Y-%m-%d'))
    worksheet.update_cell(row_index + 2, 16, new_total_cost)
    worksheet.update_cell(row_index + 2, 17, new_total_purchased)
    worksheet.update_cell(row_index + 2, 20, new_avg_price)
    
    return True

def update_position_after_sell(row_index, sell_contracts, sell_price):
    worksheet = init_portfolio_sheet()
    if not worksheet:
        return False
    
    current_contracts = int(worksheet.cell(row_index + 2, 6).value)
    current_avg_price = float(worksheet.cell(row_index + 2, 20).value) if worksheet.cell(row_index + 2, 20).value else float(worksheet.cell(row_index + 2, 7).value)
    current_sold = int(worksheet.cell(row_index + 2, 18).value) if worksheet.cell(row_index + 2, 18).value else 0
    current_realized_pnl = float(worksheet.cell(row_index + 2, 19).value) if worksheet.cell(row_index + 2, 19).value else 0
    
    if sell_contracts > current_contracts:
        return False
    
    pnl_realized = (sell_price - current_avg_price) * sell_contracts * 100
    
    new_sold = current_sold + sell_contracts
    new_contracts = current_contracts - sell_contracts
    new_realized_pnl = current_realized_pnl + pnl_realized
    
    worksheet.update_cell(row_index + 2, 6, new_contracts)
    worksheet.update_cell(row_index + 2, 18, new_sold)
    worksheet.update_cell(row_index + 2, 19, new_realized_pnl)
    
    if new_contracts == 0:
        worksheet.update_cell(row_index + 2, 13, "closed")
    
    return True

def calculate_portfolio_summary(positions_data):
    total_investment = 0
    total_unrealized_pnl = 0
    total_realized_pnl = 0
    
    for pos in positions_data:
        if pos.get('status') != 'active':
            total_realized_pnl += float(pos.get('realized_pnl', 0))
            continue
        contracts = int(pos['contracts'])
        entry_price = float(pos['entry_price'])
        total_investment += contracts * entry_price * 100
        
        option_price, _ = get_current_option_price(pos['ticker'], pos['expiry'], float(pos['strike']))
        if option_price:
            unrealized = (option_price - entry_price) * contracts * 100
            total_unrealized_pnl += unrealized
        
        realized = float(pos.get('realized_pnl', 0))
        total_realized_pnl += realized
    
    return total_investment, total_unrealized_pnl, total_realized_pnl

def calculate_risk_score(pos, current_price, current_delta, days_left, current_iv):
    entry_price = float(pos['entry_price'])
    contracts = int(pos['contracts'])
    position_value = contracts * entry_price * 100
    size_score = min(position_value / 50000, 1.0)
    
    delta_score = 1 - min(max(current_delta, 0), 1)
    time_score = 1 - min(days_left / 365, 1)
    iv_score = min(current_iv * 2, 1) if current_iv else 0.5
    
    try:
        current_stock = yf.Ticker(pos['ticker']).history(period="1d")['Close'].iloc[-1]
        strike = float(pos['strike'])
        moneyness = current_stock / strike if strike > 0 else 1
        if moneyness >= 1:
            moneyness_score = 0
        else:
            moneyness_score = 1 - moneyness
    except:
        moneyness_score = 0.5
    
    risk_score = (size_score * 0.30 + delta_score * 0.25 + time_score * 0.20 + iv_score * 0.15 + moneyness_score * 0.10)
    return risk_score

def get_current_option_price(ticker, expiry, strike):
    try:
        stock_obj = yf.Ticker(ticker)
        opt_chain = stock_obj.option_chain(expiry)
        calls = opt_chain.calls
        option_row = calls[calls['strike'] == float(strike)]
        if not option_row.empty:
            row = option_row.iloc[0]
            mid = (row['bid'] + row['ask']) / 2 if row['bid'] > 0 else row['lastPrice']
            iv = row['impliedVolatility']
            return mid, iv
        return None, None
    except Exception:
        return None, None

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

# --- HYBRID RECOMMENDATION ENGINE (for Portfolio) ---
def get_hybrid_recommendation(option_price, entry_price, target, stop_loss, 
                               days_left, current_delta, current_theta, current_iv,
                               cts, ev, tech_score, ema_status, macd_trend, macd_days,
                               touch_prob, iv_percentile=None):
    pnl_pct = ((option_price - entry_price) / entry_price) * 100 if entry_price > 0 else 0
    
    if option_price <= stop_loss:
        return "🔴 EXIT - STOP LOSS", f"Stop loss hit at ${stop_loss:.2f}"
    if ev is not None and ev < 0:
        return "🔴 EXIT - NO EDGE", f"Expected Value negative: ${ev:.2f}"
    if option_price >= target:
        return "🟢 TAKE PROFITS", f"Target ${target:.2f} reached"
    if option_price >= target * 0.8:
        return "🟡 PARTIAL EXIT", f"80% to target"
    if days_left < 7:
        return "🟠 TIME DECAY", f"{days_left} days left"
    if current_delta < 0.25:
        return "🟠 PROBABILITY DECAY", f"Delta fell to {current_delta:.2f}"
    if ema_status == "bearish_cross":
        return "🟠 TECHNICAL EXIT", "8 EMA crossed below 20 EMA"
    if macd_trend == "falling" and macd_days >= 3:
        return "🟠 TECHNICAL EXIT", f"MACD falling for {macd_days} days"
    if (pnl_pct > 0 and pnl_pct < 25 and current_delta > 0.45 and 
        cts is not None and cts > 65 and ema_status in ["bullish", "bullish_cross"]):
        return "🟢 ADD MORE", f"High conviction | CTS: {cts}"
    if (cts is not None and cts > 55 and ema_status in ["bullish", "bullish_cross"] and 
        macd_trend == "rising" and ev is not None and ev > 0.25):
        return "🔵 STRONG HOLD", f"All metrics aligned"
    return "🔵 HOLD", f"Normal monitoring"

# --- PAGE CONFIG & SESSION STATE ---
state_keys = {
    'price': None, 'trend': None, 'sma20': 0, 'pct_change': 0, 
    'stock_name': None, 'expiries': [], 'current_ticker': "", 
    'credits_used': 0, 'ai_brief': "", 'last_refresh': "Never", 'hist_data': pd.DataFrame(),
    'global_conservative': None, 'global_aggressive': None, 'global_speculative': None,
    'ai_cache': None, 'profit_target_pct': 100, 'stop_loss_pct': 30
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
    profit_target_pct = st.slider("Target Option Profit Booking (%)", 10, 150, 100, step=5)
    stop_loss_pct = st.slider("Max Stop Loss (%)", 10, 100, 30, step=5)
    st.session_state.profit_target_pct = profit_target_pct
    st.session_state.stop_loss_pct = stop_loss_pct
    st.divider()
    if st.button("🗑️ Clear Cache", help="Clear cached data if you're seeing stale information"):
        st.cache_data.clear()
        st.success("Cache cleared! Refresh the page to reload data.")
        st.rerun()

# --- DATA FETCHING & GLOBAL SCANS ---
if fetch_btn:
    st.session_state.current_ticker = ticker_input
    st.session_state.ai_brief = "" 
    st.session_state.global_conservative = None
    st.session_state.global_aggressive = None
    st.session_state.global_speculative = None
    
    try:
        hist = get_cached_stock_history(ticker_input, "100d")
        
        if hist is None or hist.empty or 'Close' not in hist.columns:
            st.error(f"❌ No valid history found for {ticker_input}")
            st.session_state.price = None
        else:
            st.session_state.hist_data = hist
            st.session_state.price = hist['Close'].iloc[-1]
            
            stock_info = get_cached_stock_info(ticker_input)
            st.session_name = stock_info.get('longName', ticker_input) if stock_info else ticker_input
            st.session_state.stock_name = st.session_name
            
            stock_obj = yf.Ticker(ticker_input)
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
            valid_global_expiries = [exp for exp in st.session_state.expiries if (pd.to_datetime(exp).date() - today).days >= 60]
            
            cons_candidates = []
            aggr_candidates = []
            spec_candidates = []
            
            with st.spinner("Processing mathematical matrix across options chain..."):
                for exp_date in valid_global_expiries[:6]:
                    try:
                        calls_df, puts_df = get_cached_option_chain(ticker_input, exp_date)
                        if calls_df is None:
                            continue
                        opt_chain = calls_df
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
                    except Exception as e:
                        st.warning(f"Could not process expiry {exp_date}: {str(e)[:50]}")
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
    
    # Tabs (9 tabs - DASHBOARD is NEW)
    t_dashboard, t_summary, t_cons, t_aggr, t_spec, t_tech, t_ai, t_portfolio, t_edu = st.tabs([
        "📊 Dashboard", "📋 Global Recs", "🛡️ Conservative", "⚡ Aggressive", 
        "🎰 Speculative", "📊 Technical", "🤖 AI Research", 
        "📂 Portfolio", "📖 Strategy Guide"
    ])

    # ========================
    # NEW DASHBOARD TAB (Phase 1)
    # ========================
    with t_dashboard:
        st.header("📊 Trading Dashboard")
        
        # Get market regime
        vix = get_vix()
        regime_text, regime_desc = get_market_regime(vix)
        
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.metric("VIX (Fear Index)", f"{vix:.1f}", delta=regime_text)
            st.caption(regime_desc)
        
        with col_m2:
            # Get ATR and RSI for current ticker
            hist_data = st.session_state.hist_data
            atr_val, atr_pct = calculate_atr(hist_data)
            rsi_val = calculate_rsi(hist_data)
            
            st.metric("ATR (14d)", f"${atr_val:.2f}", delta=f"{atr_pct:.1f}% of price")
            rsi_status = "🟢 Neutral" if 30 <= rsi_val <= 70 else ("🔴 Overbought" if rsi_val > 70 else "🟠 Oversold")
            st.metric("RSI (14d)", f"{rsi_val:.1f}", delta=rsi_status)
        
        with col_m3:
            # Earnings proximity
            earnings_date = get_earnings_date(st.session_state.current_ticker)
            if earnings_date:
                days_to_earnings = (earnings_date - datetime.now().date()).days
                if days_to_earnings < 0:
                    st.metric("Next Earnings", "Recently passed", delta="No immediate risk")
                elif days_to_earnings < 7:
                    st.warning(f"⚠️ Earnings in {days_to_earnings} days - Consider avoiding new positions")
                    st.metric("Earnings Warning", f"In {days_to_earnings} days", delta="HIGH RISK")
                else:
                    st.metric("Next Earnings", earnings_date.strftime('%Y-%m-%d'), delta=f"In {days_to_earnings} days")
            else:
                st.metric("Next Earnings", "Not available", delta="Check manually")
        
        st.divider()
        
        # Active Positions Summary
        st.subheader("📋 Active Positions Summary")
        positions = get_portfolio_positions("Mukul")  # Default trader for dashboard
        if positions:
            pos_data = []
            for idx, (row_idx, pos) in enumerate(positions[:5]):  # Show top 5
                entry = float(pos['entry_price'])
                contracts = int(pos['contracts'])
                option_price, _ = get_current_option_price(pos['ticker'], pos['expiry'], float(pos['strike']))
                if option_price:
                    pnl_pct = ((option_price - entry) / entry) * 100
                    pos_data.append({
                        "Position": f"{pos['ticker']} ${float(pos['strike']):.2f} Call",
                        "P&L": f"{pnl_pct:+.1f}%",
                        "Risk": "🟢 LOW" if pnl_pct > 10 else ("🟡 MEDIUM" if pnl_pct > -10 else "🔴 HIGH")
                    })
            if pos_data:
                st.dataframe(pd.DataFrame(pos_data), use_container_width=True)
            else:
                st.info("No active positions with price data")
        else:
            st.info("No active positions. Add positions in Portfolio tab.")
        
        st.divider()
        
        # Quick Actions
        st.subheader("🚀 Quick Actions")
        col_q1, col_q2, col_q3 = st.columns(3)
        with col_q1:
            if st.button("🔍 Analyze Current Ticker", use_container_width=True):
                st.switch_page("app_v5.py")  # Refresh effectively
        with col_q2:
            st.link_button("📂 Go to Portfolio", "#", use_container_width=True)
        with col_q3:
            if st.button("🗑️ Clear Cache", use_container_width=True):
                st.cache_data.clear()
                st.rerun()

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
    
        calls_df, puts_df = get_cached_option_chain(st.session_state.current_ticker, expiry)
        if calls_df is not None:
            chain = calls_df
        else:
            st.error("Failed to fetch option chain")
            chain = pd.DataFrame()
        
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
            st.markdown(f"*Best structure based on highest Expected Value (EV) for {tier_label} strategy*")
            
            if best_contract['volume'] < 50:
                st.warning(f"{best_contract['liquidity_status']}: {best_contract['liquidity_warning']}")
            
            reco_exit = best_contract['mid'] * (1 + profit_target_pct / 100)
            reco_stop = best_contract['mid'] * (1 - stop_loss_pct / 100)
            reco_hold = min(int(days_to_expiry * 0.4), 45)
            reco_date = (datetime.now() + timedelta(days=reco_hold)).strftime('%B %d, %Y')
            
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
                    <tr>
                        <td><b>Volume Today:</b></td>
                        <td>{best_contract['volume']:,} contracts</td>
                        <td><b>Open Interest:</b></td>
                        <td>{best_contract['open_interest']:,}</td>
                    </tr>
                    <tr>
                        <td><b>Bid-Ask Spread:</b></td>
                        <td>${best_contract['spread']:.2f} ({best_contract['spread_pct']:.1f}%)</td>
                        <td><b>Liquidity:</b></td>
                        <td>{best_contract['liquidity_status']}</td>
                    </tr>
                </table>
            </div>
            """
            st.markdown(reco_html, unsafe_allow_html=True)
            
            st.divider()
            st.markdown("### 🔍 Compare Other Strikes")
            st.markdown("*Select any strike below to see how its mathematical metrics compare to the recommendation above*")
            
            strike_list = sorted([item['strike'] for item in all_available_contracts])
            default_index = strike_list.index(best_contract['strike']) if best_contract['strike'] in strike_list else 0
            
            selected_k = st.selectbox(
                f"Select Strike to Analyze ({tier_label} Comparison):", 
                strike_list, 
                index=default_index, 
                key=f"compare_{tier_label}_{expiry}"
            )
            
            selected_contract = next((item for item in all_available_contracts if item['strike'] == selected_k), None)
            
            if selected_contract:
                selected_exit = selected_contract['mid'] * (1 + profit_target_pct / 100)
                selected_stop = selected_contract['mid'] * (1 - stop_loss_pct / 100)
                selected_hold = min(int(days_to_expiry * 0.4), 45)
                selected_date = (datetime.now() + timedelta(days=selected_hold)).strftime('%B %d, %Y')
                
                if selected_contract['volume'] < 50 and selected_k != best_contract['strike']:
                    st.warning(f"⚠️ {selected_contract['liquidity_status']}: {selected_contract['liquidity_warning']}")
                
                st.markdown("### 📊 Mathematical Output Summary")
                c1, c2, c3 = st.columns([1.5, 1.5, 2])
                with c1:
                    if selected_contract['cts'] >= 55 and selected_contract['ev'] > 0:
                        st.success("✅ STRUCTURAL BUY INSTANCE")
                        st.markdown("""
                        <p style='font-size:0.85rem; color:rgba(255,255,255,0.75);line-height:1.3;'>
                        <b>What this means:</b> The odds are highly in your favor.
                        </p>
                        """, unsafe_allow_html=True)
                    elif selected_contract['cts'] >= 40 and selected_contract['ev'] > 0:
                        st.warning("⚠️ WEAK EDGE PATTERN")
                    else:
                        st.error("❌ NEGATIVE EXPECTANCY AVOID")
                    st.metric("Composite Score", f"{selected_contract['cts']}/100")
                    st.metric("Entry Target", f"${selected_contract['mid']:.2f}")
                    
                with c2:
                    st.metric("Take Profit Target", f"${selected_exit:.2f}")
                    st.metric("Stop Loss Point", f"${selected_stop:.2f}")
                    st.write(f"⏱️ **Hold Cutoff:** `{selected_hold} days` ({selected_date})")
                    st.metric("Volume Today", f"{selected_contract['volume']:,}")
                    st.metric("Open Interest", f"{selected_contract['open_interest']:,}")

                with c3:
                    st.write("**Stochastic Engine Outputs**")
                    st.write(f"- Stat Probability: `{selected_contract['delta'] * 100:.1f}%`")
                    st.write(f"- Touch Probability: `{selected_contract['p_touch'] * 100:.1f}%`")
                    st.write(f"- Expected Value: `{selected_contract['ev']:.3f}`")
                    st.write(f"- IV: `{selected_contract['iv']*100:.1f}%` | Theta: `-{abs(selected_contract['theta']):.3f}`")
                    st.write(f"- Bid-Ask Spread: `${selected_contract['spread']:.2f}` ({selected_contract['spread_pct']:.1f}%)")
                    
                    st.write("")
                    try:
                        h_chart = yf.Ticker(selected_contract['symbol']).history(period="1mo")
                        if not h_chart.empty:
                            st.caption("📈 Contract Price History (Last 30 days)")
                            st.line_chart(h_chart['Close'])
                            if 'Volume' in h_chart.columns and h_chart['Volume'].sum() > 0:
                                st.caption("📊 Daily Trading Volume (Last 30 days)")
                                st.bar_chart(h_chart['Volume'])
                    except:
                        st.caption("Historical chart data unavailable")

    process_tier_strategy(t_cons, 0.50, 0.60, "Conservative")
    process_tier_strategy(t_aggr, 0.40, 0.49, "Aggressive")
    process_tier_strategy(t_spec, 0.30, 0.39, "Speculative")

    with t_tech:
        if not st.session_state.hist_data.empty:
            st.subheader("Momentum & Volatility Health")
            
            # NEW: ATR and RSI display
            atr_val, atr_pct = calculate_atr(st.session_state.hist_data)
            rsi_val = calculate_rsi(st.session_state.hist_data)
            hv_val = calculate_hv(st.session_state.hist_data)
            
            col_a1, col_a2, col_a3 = st.columns(3)
            with col_a1:
                st.metric("ATR (14d)", f"${atr_val:.2f}", delta=f"{atr_pct:.1f}% of price")
                st.caption("Average True Range - Daily volatility measure")
            with col_a2:
                rsi_status = "Overbought" if rsi_val > 70 else ("Oversold" if rsi_val < 30 else "Neutral")
                st.metric("RSI (14d)", f"{rsi_val:.1f}", delta=rsi_status)
                st.caption("Relative Strength Index - Momentum")
            with col_a3:
                st.metric("Historical Vol (20d)", f"{hv_val:.1f}%")
                st.caption("Realized volatility - Compare with IV")
            
            st.divider()
            
            c1, c2, c3 = st.columns(3)
            
            curr, prev = get_technicals(st.session_state.hist_data)
            ema_status = "Bullish Cross" if curr['ema8'] > curr['ema20'] else "Bearish Separation"
            c1.metric("8/20 EMA Status", ema_status, f"{curr['ema8'] - curr['ema20']:.2f} delta")
            if curr['ema8'] > curr['ema20'] and prev['ema8'] <= prev['ema20']:
                c1.success("🔥 JUST CROSSED BULLISH")
            
            macd_dir = "Improving" if curr['hist'] > prev['hist'] else "Fading"
            c2.metric("MACD Momentum", macd_dir, f"{curr['hist']:.3f} hist")
            
            pos = "Upper Half" if S > curr['sma20'] else "Lower Half"
            c3.metric("Bollinger Position", pos, f"{((S - curr['lower'])/(curr['upper'] - curr['lower']))*100:.1f}% Band")
            if S > curr['upper']:
                c3.warning("⚠️ OVEREXTENDED")

            st.divider()
            st.line_chart(st.session_state.hist_data[['Close', 'ema8', 'ema20', 'upper', 'lower']])

            st.subheader("🏁 Final Technical Verdict")
            if tech_score == 3:
                st.success("🎯 **VERDICT: INVEST.** All indicators are aligned.")
            elif tech_score == 2:
                st.warning("⚖️ **VERDICT: CAUTION.** Mixed signals.")
            else:
                st.error("🛑 **VERDICT: STAY AWAY.** Bearish structure.")
            
            with st.expander("View Verdict Logic"):
                for reason in verdict_reasons:
                    st.write(f"- {reason}")
                if tech_score < 2:
                    st.write("- Multiple indicators show declining strength or bearish crossovers.")

    with t_ai:
        if st.button("🔄 Refresh AI Analysis"):
            st.session_state.ai_brief = ""
            st.rerun()
        if not st.session_state.ai_brief:
            with st.spinner("Fetching AI analysis..."):
                st.session_state.ai_brief = get_ai_research(st.session_state.current_ticker)
        st.markdown(st.session_state.ai_brief)

    # ========================
    # PORTFOLIO TAB (FULLY PRESERVED from v4)
    # ========================
    with t_portfolio:
        st.header("📂 Options Portfolio Tracker")
        
        if 'sheet_initialized' not in st.session_state:
            init_portfolio_sheet()
            st.session_state.sheet_initialized = True
        
        def get_conservative_strike_for_expiry(ticker, expiry_date, profit_target_pct, stop_loss_pct):
            try:
                stock_obj = yf.Ticker(ticker)
                hist = stock_obj.history(period="100d")
                if hist.empty:
                    return None, None
                current_price = get_cached_current_price(ticker)
                if current_price is None:
                    current_price = stock.history(period="1d")['Close'].iloc[-1]
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
        
        def calculate_composite_score_for_position(current_price, strike, days_left, iv, tech_score):
            d, _, _, _ = calculate_greeks(current_price, strike, max(days_left, 1)/365, 0.05, iv)
            p_touch = calculate_p_touch(current_price, strike, max(days_left, 1)/365, iv)
            ev = (p_touch * (1 + st.session_state.profit_target_pct/100)) - ((1 - p_touch) * (st.session_state.stop_loss_pct/100))
            cts = int(((d * 0.4) + (p_touch * 0.4) + (tech_score / 3.0 * 0.2)) * 100)
            return cts, ev, d, p_touch
        
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
                            success = add_trader_to_sheet(new_trader_name, new_trader_email)
                            if success:
                                dummy_worksheet = init_portfolio_sheet()
                                if dummy_worksheet:
                                    dummy_worksheet.append_row([
                                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
                                        new_trader_name, "PLACEHOLDER", 0, "2024-01-01", 
                                        0, 0, 0, 0, "2024-01-01", 0, 0, "inactive", "", "", 0, 0, 0, 0, 0
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
        
        all_positions = get_all_positions_for_trader(selected_trader)
        active_positions = get_portfolio_positions(selected_trader)
        
        total_investment, total_unrealized, total_realized = calculate_portfolio_summary(all_positions)
        
        st.markdown("### 📊 Portfolio Summary")
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            st.metric("💰 Total Investment", f"${total_investment:,.2f}")
        with col_s2:
            unrealized_color = "normal" if total_unrealized >= 0 else "inverse"
            st.metric("📈 Unrealized P&L", f"${total_unrealized:+,.2f}", delta_color=unrealized_color)
        with col_s3:
            st.metric("✅ Realized P&L", f"${total_realized:+,.2f}")
        
        st.divider()
        st.subheader("📊 Active Positions")
        
        col_refresh, _ = st.columns([1, 5])
        with col_refresh:
            if st.button("🔄 Refresh", key="refresh_portfolio", use_container_width=True):
                st.rerun()
        
        if active_positions:
            positions_with_risk = []
            for idx, (row_idx, pos) in enumerate(active_positions):
                try:
                    strike = float(pos['strike'])
                    expiry_date = pd.to_datetime(pos['expiry']).date()
                    days_left = max((expiry_date - datetime.now().date()).days, 0)
                    
                    option_price, current_iv = get_current_option_price(pos['ticker'], pos['expiry'], strike)
                    if option_price:
                        stock_price = yf.Ticker(pos['ticker']).history(period="1d")['Close'].iloc[-1]
                        d, _, _, _ = calculate_greeks(stock_price, strike, max(days_left, 1)/365, 0.05, current_iv)
                        current_delta = d
                    else:
                        current_delta = 0.5
                        current_iv = 0.35
                    
                    risk_score = calculate_risk_score(pos, option_price if option_price else 0, current_delta, days_left, current_iv)
                    positions_with_risk.append((risk_score, idx, row_idx, pos))
                except:
                    positions_with_risk.append((0.5, idx, row_idx, pos))
            
            positions_with_risk.sort(key=lambda x: x[0], reverse=True)
            
            for risk_score, idx, row_idx, pos in positions_with_risk:
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
                    
                    if risk_score > 0.7:
                        risk_indicator = "🔴 HIGH"
                    elif risk_score > 0.4:
                        risk_indicator = "🟡 MEDIUM"
                    else:
                        risk_indicator = "🟢 LOW"
                    
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
                            ema_status = "bullish" if curr['ema8'] > curr['ema20'] else "bearish"
                        else:
                            tech_score_pos = 1
                            ema_status = "neutral"
                    except:
                        tech_score_pos = 1
                        ema_status = "neutral"
                    
                    stock_price = yf.Ticker(ticker).history(period="1d")['Close'].iloc[-1]
                    cts, ev, delta_calc, touch_prob = calculate_composite_score_for_position(
                        stock_price, strike, days_left, current_iv, tech_score_pos
                    )
                    
                    rec_icon_full, rec_reason = get_hybrid_recommendation(
                        option_price, entry_price, target, stop, days_left, delta_calc, 0, current_iv,
                        cts, ev, tech_score_pos, ema_status, "stable", 0, touch_prob, 50
                    )
                    
                    rec_icon = rec_icon_full.split()[0]
                    summary = f"{rec_icon} {ticker} ${strike:.2f} Call | Exp: {expiry_date_str} | ${option_price:.2f} | P&L: {pnl_pct:+.1f}% (${pnl:+.0f}) | Risk: {risk_indicator}"
                    
                    with st.expander(summary):
                        st.markdown("### 📊 Position Summary")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Current Option Price", f"${option_price:.2f}")
                            pnl_color = "inverse" if pnl < 0 else "normal"
                            st.metric("P&L", f"{pnl_pct:+.1f}%", delta=f"${pnl:+.0f}", delta_color=pnl_color)
                            st.metric("Risk Score", f"{risk_score:.2f}", help="Higher = Higher Risk")
                        with col2:
                            st.metric("Days Left", f"{days_left}")
                            st.metric("Delta", f"{delta_calc:.3f}")
                        with col3:
                            st.metric("Entry Price (Avg)", f"${entry_price:.2f}")
                            st.metric("Contracts", contracts)
                        
                        st.markdown("---")
                        st.markdown("### 💡 Recommendation")
                        st.info(f"**{rec_icon_full}**")
                        st.caption(rec_reason)
                        
                        st.markdown("---")
                        st.markdown("### 📊 Quant Analytics")
                        q1, q2, q3 = st.columns(3)
                        with q1:
                            st.metric("Expected Value (EV)", f"${ev:.2f}")
                            st.metric("Composite Score", f"{cts}/100")
                            st.metric("Touch Probability", f"{touch_prob*100:.0f}%")
                        with q2:
                            st.metric("IV", f"{current_iv*100:.1f}%")
                        with q3:
                            st.metric("Technical Score", f"{tech_score_pos}/3")
                        
                        st.markdown("---")
                        st.markdown("### 🎯 Targets")
                        col_t1, col_t2 = st.columns(2)
                        with col_t1:
                            st.metric("🛑 Stop Loss", f"${stop:.2f}")
                            if option_price > stop:
                                st.caption(f"✅ ${option_price - stop:.2f} above stop")
                            else:
                                st.caption(f"⚠️ ${stop - option_price:.2f} below stop")
                        with col_t2:
                            st.metric("🎯 Target", f"${target:.2f}")
                            if option_price < target:
                                st.caption(f"📈 Need +${target - option_price:.2f} to target")
                                st.progress(option_price / target)
                            else:
                                st.caption("✅ Target reached")
                                st.progress(1.0)
                        
                        st.markdown("---")
                        st.markdown("### ⚙️ Position Management")
                        
                        col_m1, col_m2 = st.columns(2)
                        
                        with col_m1:
                            st.markdown("**➕ Add More Contracts**")
                            add_qty = st.number_input("Quantity to add:", min_value=1, step=1, key=f"add_qty_{idx}")
                            add_price = st.number_input("Purchase price:", min_value=0.01, step=0.05, format="%.2f", 
                                                        value=option_price, key=f"add_price_{idx}")
                            if st.button("Add More", key=f"add_btn_{idx}"):
                                success = update_position_after_add(row_idx, add_qty, add_price)
                                if success:
                                    st.success(f"Added {add_qty} contracts at ${add_price:.2f}!")
                                    st.rerun()
                                else:
                                    st.error("Failed to add. Check inputs.")
                        
                        with col_m2:
                            st.markdown("**💰 Sell Contracts**")
                            sell_qty = st.number_input("Quantity to sell:", min_value=1, max_value=contracts, step=1, 
                                                       key=f"sell_qty_{idx}")
                            sell_price = st.number_input("Sale price:", min_value=0.01, step=0.05, format="%.2f", 
                                                         value=option_price, key=f"sell_price_{idx}")
                            if st.button("Sell", key=f"sell_btn_{idx}"):
                                if sell_qty > contracts:
                                    st.error(f"Cannot sell more than {contracts} contracts.")
                                else:
                                    success = update_position_after_sell(row_idx, sell_qty, sell_price)
                                    if success:
                                        st.success(f"Sold {sell_qty} contracts at ${sell_price:.2f}!")
                                        st.rerun()
                                    else:
                                        st.error("Failed to sell.")
                else:
                    with st.expander(f"⚠️ {ticker} ${strike:.2f} Call | Data unavailable"):
                        st.warning(f"Option price data not available")
        else:
            st.info(f"No active positions for {selected_trader}.")
        
        st.divider()
        
        # --- ADD NEW POSITION FORM ---
        st.subheader("➕ Add New Position")
        
        if not st.session_state.expiries:
            st.warning("Please analyze a ticker first.")
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
                st.warning(f"No option data available for {current_ticker}")
            else:
                strike_options = [s['strike'] for s in all_strikes]
                default_strike_index = 0
                if cons_strike and cons_strike in strike_options:
                    default_strike_index = strike_options.index(cons_strike)
                selected_strike = st.selectbox("Strike Price:", options=strike_options, index=default_strike_index, key="portfolio_strike_select")
                selected_mid = next((s['mid'] for s in all_strikes if s['strike'] == selected_strike), None)
                
                if cons_strike and cons_strike == selected_strike:
                    st.caption(f"⭐ Recommended strike - Mid: ${selected_mid:.2f}")
                
                trader_email = get_trader_email(selected_trader)
                if not trader_email:
                    st.warning(f"⚠️ No email configured for {selected_trader}. Add email when creating trader.")
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
                    
                    try:
                        stock_temp = yf.Ticker(ticker_pos)
                        current_price_temp = stock_temp.history(period="1d")['Close'].iloc[-1]
                        option_chain_temp = stock_temp.option_chain(selected_expiry_str)
                        option_row_temp = option_chain_temp.calls[option_chain_temp.calls['strike'] == selected_strike]
                        if not option_row_temp.empty:
                            entry_iv = option_row_temp['impliedVolatility'].iloc[0]
                            days_to_exp = (expiry_pos - datetime.now().date()).days
                            d_temp, _, _, _ = calculate_greeks(
                                current_price_temp, selected_strike, max(days_to_exp, 1) / 365, 0.05, entry_iv
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
                        success = add_position_to_sheet(
                            trader_name=selected_trader, ticker=ticker_pos, strike=selected_strike,
                            expiry=selected_expiry_str, contracts=contracts_pos, entry_price=entry_price_pos,
                            entry_iv=entry_iv, entry_delta=entry_delta, target_price=target_auto,
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
    # STRATEGY GUIDE (FULLY PRESERVED from v4)
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
        
        st.subheader("📊 NEW: Advanced Technical Indicators (Phase 1)")
        st.markdown("""
        | Indicator | What It Measures | How To Use |
        |-----------|------------------|------------|
        | **ATR (Average True Range)** | Daily volatility magnitude | Target should be 2-3x ATR for realistic 5-day moves |
        | **RSI (Relative Strength Index)** | Overbought/oversold conditions | RSI > 70 = overbought (avoid entry); RSI < 30 = oversold (potential bounce) |
        | **Historical Volatility (HV)** | Actual past volatility | Compare with IV: IV > HV = options expensive; IV < HV = options cheap |
        | **VIX (Fear Index)** | Market-wide volatility regime | VIX < 15 = low vol (small moves); VIX > 25 = high vol (large moves possible) |
        """)
        
        st.divider()
        
        st.subheader("💧 Liquidity Indicators - Your First Filter")
        st.markdown("**Before looking at any other metric, check liquidity first.**")
        
        with st.expander("📊 Volume Today - How to Use", expanded=False):
            st.markdown("""
            | Volume | Rating | Action |
            |--------|--------|--------|
            | 200+ | 🟢 EXCELLENT | Safe to trade any position size |
            | 100-199 | 🟡 GOOD | Acceptable for positions under 50 contracts |
            | 50-99 | 🟠 CAUTION | Only for positions under 10 contracts |
            | 10-49 | 🔴 DANGER | Avoid unless absolutely necessary |
            | <10 | ⚫ TOXIC | NEVER TRADE |
            """)
        
        with st.expander("💰 Bid-Ask Spread - Your Real Transaction Cost", expanded=False):
            st.markdown("""
            | Spread % | Grade | Implication |
            |----------|-------|-------------|
            | < 2% | 🟢 EXCELLENT | Round-trip cost <4% |
            | 2-5% | 🟡 ACCEPTABLE | Round-trip cost 4-10% |
            | 5-10% | 🟠 WIDE | Cost 10-20% |
            | > 10% | 🔴 TOXIC | AVOID |
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
        **Step 1: Market Regime (NEW)**
        - [ ] VIX between 12-25 (optimal for calls)
        - [ ] Not within 7 days of earnings
        
        **Step 2: Liquidity Filter (MANDATORY)**
        - [ ] Volume > 50 (preferably > 200)
        - [ ] Open Interest > 500
        - [ ] Spread < 5% (preferably < 3%)
        
        **Step 3: Strategy Match**
        - [ ] Delta matches your risk profile
        - [ ] Days to expiry > 60
        
        **Step 4: Technical Confirmation (NEW)**
        - [ ] RSI between 30-70 (not overbought/oversold)
        - [ ] ATR sufficient for target (target move > 2x ATR)
        - [ ] 8 EMA > 20 EMA (bullish)
        
        **Step 5: Math Confirmation**
        - [ ] Composite Score > 55
        - [ ] Expected Value > 0.25
        
        **Step 6: Trade Management Plan**
        - [ ] Take profit at 40% above entry
        - [ ] Stop loss at 30% below entry
        - [ ] Exit by cutoff date
        """)
        
        st.warning("""
        **⚠️ Remember:** No indicator is perfect. Always size positions appropriately 
        (never risk more than 1-2% of account per trade) and follow your stop losses.
        """)

else:
    st.info("👈 Input a valid trading ticker to trigger the options matrix models.")
