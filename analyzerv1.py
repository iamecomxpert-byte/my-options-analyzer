import streamlit as st
import yfinance as yf
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import norm
from groq import Groq
import time
import warnings
warnings.filterwarnings('ignore')

# --- PAGE CONFIG MUST BE FIRST ---
st.set_page_config(page_title="Analyst Pro Options Suite v7", layout="wide")

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

# --- CACHING FOR YFINANCE DATA ---
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
        
        # Validate the data
        if hist is None or hist.empty:
            return pd.DataFrame()
        
        if 'Close' not in hist.columns:
            return pd.DataFrame()
        
        # Check if all Close values are NaN
        if hist['Close'].isna().all():
            return pd.DataFrame()
        
        return hist
    except Exception as e:
        return pd.DataFrame()

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
    """Get current price with multiple fallback methods"""
    try:
        stock = yf.Ticker(ticker)
        
        # Method 1: Get from 1-day history
        hist = stock.history(period="1d")
        if not hist.empty and 'Close' in hist.columns:
            close_val = hist['Close'].iloc[-1]
            if pd.notna(close_val):
                return float(close_val)
        
        # Method 2: Get from 5-day history (if 1-day failed)
        hist_5d = stock.history(period="5d")
        if not hist_5d.empty and 'Close' in hist_5d.columns:
            close_val = hist_5d['Close'].iloc[-1]
            if pd.notna(close_val):
                return float(close_val)
        
        # Method 3: Get from info
        info = stock.info
        if info:
            for key in ['regularMarketPrice', 'currentPrice', 'lastClose', 'previousClose']:
                if key in info and info[key] is not None:
                    val = info[key]
                    if isinstance(val, (int, float)) and pd.notna(val):
                        return float(val)
        
        return None
    except Exception as e:
        return None

# ========================
# ADVANCED QUANT FUNCTIONS
# ========================

def calculate_atr(df, period=14):
    try:
        high, low, close = df['High'], df['Low'], df['Close']
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        last_atr = atr.iloc[-1]
        last_close = close.iloc[-1]
        if pd.notna(last_atr) and pd.notna(last_close) and last_close > 0:
            atr_pct = (last_atr / last_close) * 100
            return round(last_atr, 2), round(atr_pct, 1)
        return 0.0, 0.0
    except Exception:
        return 0.0, 0.0

def calculate_rsi(df, period=14):
    try:
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        last_rsi = rsi.iloc[-1]
        if pd.notna(last_rsi):
            return round(last_rsi, 1)
        return 50.0
    except Exception:
        return 50.0

def calculate_hv(df, period=20):
    try:
        returns = np.log(df['Close'] / df['Close'].shift(1))
        hv = returns.rolling(window=period).std() * np.sqrt(252)
        last_hv = hv.iloc[-1]
        if pd.notna(last_hv):
            return round(last_hv * 100, 1)
        return 0.0
    except Exception:
        return 0.0

def calculate_dynamic_targets(stock_price, option_price, delta, gamma, theta, days=5, target_gain_pct=0.50):
    try:
        target_option = option_price * (1 + target_gain_pct)
        target_gain = target_option - option_price
        
        a = 0.5 * gamma
        b = delta
        c = (theta * days) - target_gain
        
        if a != 0:
            discriminant = b**2 - 4*a*c
            if discriminant >= 0:
                delta_S = (-b + np.sqrt(discriminant)) / (2*a)
                target_stock = stock_price + delta_S
            else:
                target_stock = stock_price + (target_gain / delta) if delta > 0 else stock_price * 1.05
        else:
            target_stock = stock_price + (target_gain / delta) if delta > 0 else stock_price * 1.05
        
        stop_option = option_price * 0.75
        stop_gain = stop_option - option_price
        c_stop = (theta * days) - stop_gain
        
        if a != 0:
            disc_stop = b**2 - 4*a*c_stop
            if disc_stop >= 0:
                delta_S_stop = (-b + np.sqrt(disc_stop)) / (2*a)
                stop_stock = stock_price + delta_S_stop
            else:
                stop_stock = stock_price * 0.97
        else:
            stop_stock = stock_price + (stop_gain / delta) if delta > 0 else stock_price * 0.97
        
        return max(target_stock, stock_price * 1.01), max(stop_stock, stock_price * 0.90)
    except Exception:
        return stock_price * 1.05, stock_price * 0.95

# ========================
# NEW FORECASTING FUNCTIONS (Phase 5.5)
# ========================

def forecast_5day_price(current_option_price, stock_price, strike, delta, gamma, theta, vega, current_iv, forecast_iv_change=0, days=5, expected_stock_move_pct=0.03):
    """
    Calculate expected option price in 5 days using proper options math.
    """
    try:
        # Theta decay (daily decay over 5 days)
        theta_decay = abs(theta) * days
        
        # IV impact (if volatility changes)
        iv_impact = vega * forecast_iv_change
        
        # ========== CORRECT LEVERAGE CALCULATION ==========
        if current_option_price > 0 and stock_price > 0:
            actual_leverage = delta * (stock_price / current_option_price)
            actual_leverage = min(max(actual_leverage, 2.0), 30.0)
        else:
            actual_leverage = 8.0
        
        # FIX: Convert percentage to decimal correctly
        # expected_stock_move_pct is already a decimal (0.03 = 3%)
        expected_stock_move_decimal = expected_stock_move_pct  # Keep as is (0.03)
        
        # Expected option percentage move (as decimal, e.g., 0.117 for 11.7%)
        expected_option_move_decimal = actual_leverage * expected_stock_move_decimal
        
        # Convert to percentage for display (multiply by 100)
        expected_option_move_pct_display = expected_option_move_decimal * 100
        
        # Dollar move
        expected_dollar_move = current_option_price * expected_option_move_decimal
        
        # Gamma convexity
        try:
            moneyness = stock_price / strike if strike > 0 else 1.0
            if moneyness > 0.95:
                gamma_boost = 1.0 + (gamma * stock_price)
                gamma_boost = min(max(gamma_boost, 1.1), 1.5)
            else:
                gamma_boost = 1.1
        except:
            gamma_boost = 1.2
        
        expected_price_change = expected_dollar_move * gamma_boost
        
        # Total expected price
        expected_price = current_option_price - theta_decay + iv_impact + expected_price_change
        expected_price = max(expected_price, 0.05)
        
        # 80% confidence bounds
        uncertainty_factor = abs(expected_price_change) * 0.8
        price_upper = expected_price + uncertainty_factor
        price_lower = max(expected_price - uncertainty_factor, 0.05)
        
        return (round(expected_price, 3), round(price_upper, 3), round(price_lower, 3), 
                round(theta_decay, 3), round(iv_impact, 3), round(actual_leverage, 1))
    except Exception as e:
        return current_option_price, current_option_price, current_option_price, 0, 0, 0

def probability_hit_target(current_option_price, target_price, days, option_iv, num_sims=500):
    """
    Calculate probability of hitting profit target using OPTION implied volatility.
    """
    try:
        # If target is below current price, you've already hit it!
        if target_price <= current_option_price:
            return 0.95  # 95% chance to stay above target (already there)
        
        # Daily option volatility from IV
        daily_vol = option_iv / np.sqrt(252)
        
        # Expected drift (slight bullish bias for calls)
        drift = 0.05 / 252  # 5% annual
        
        # Vectorized Monte Carlo
        np.random.seed(42)
        returns = np.random.normal(drift, daily_vol, (num_sims, days))
        price_paths = current_option_price * np.exp(np.cumsum(returns, axis=1))
        
        hit_target = np.any(price_paths >= target_price, axis=1)
        probability = np.mean(hit_target)
        
        return round(min(probability, 0.95), 3)  # Cap at 95%
    except Exception:
        return 0.45  # Default 45%

def probability_hit_stop(current_option_price, stop_price, days, option_iv, num_sims=500):
    """
    Calculate probability of hitting stop loss using OPTION implied volatility.
    """
    try:
        # If stop is above current price, you're already below stop? No, that's inverted.
        # For a long call: stop should be BELOW current price
        # If stop is above current, you'd have already stopped out
        if stop_price >= current_option_price:
            return 0.85  # High probability of hitting stop (already near it)
        
        daily_vol = option_iv / np.sqrt(252)
        drift = -0.02 / 252  # Slight negative drift for stop probability
        
        np.random.seed(42)
        returns = np.random.normal(drift, daily_vol, (num_sims, days))
        price_paths = current_option_price * np.exp(np.cumsum(returns, axis=1))
        
        hit_stop = np.any(price_paths <= stop_price, axis=1)
        probability = np.mean(hit_stop)
        
        return round(probability, 3)
    except Exception:
        return 0.35  # Default 35%

def estimate_iv_percentile(current_iv, hv):
    """
    Estimate IV percentile using IV/HV spread as proxy.
    """
    try:
        if current_iv is None or hv is None or hv == 0:
            return 50
        
        current_iv_pct = current_iv * 100
        spread = current_iv_pct - hv
        
        # FIX: Normalize spread to percentile (assuming typical IV range 20-80%)
        # If IV is 50% and HV is 30%, spread = 20 → ~80th percentile
        # If IV is 30% and HV is 30%, spread = 0 → 50th percentile
        # If IV is 20% and HV is 40%, spread = -20 → ~20th percentile
        percentile = 50 + spread
        return max(5, min(95, percentile))  # Cap between 5-95%
    except Exception:
        return 50

# ========================
# MACRO & SECTOR CONTEXT FUNCTIONS
# ========================

def get_macro_data():
    """Fetch macro-level data for AI context"""
    macro_data = {}
    
    # VIX
    macro_data['vix'] = get_vix()
    
    # Market regime
    regime_text, regime_desc = get_market_regime(macro_data['vix'])
    macro_data['regime'] = regime_text
    macro_data['regime_description'] = regime_desc
    
    # SPY performance (market benchmark)
    try:
        spy = yf.Ticker("SPY")
        spy_hist = spy.history(period="5d")
        if not spy_hist.empty:
            spy_close = spy_hist['Close']
            macro_data['spy_change_5d'] = round(((spy_close.iloc[-1] / spy_close.iloc[0]) - 1) * 100, 1)
            macro_data['spy_price'] = round(spy_close.iloc[-1], 2)
    except:
        macro_data['spy_change_5d'] = 0
        macro_data['spy_price'] = 0
    
    # Sector ETFs (relevant to quantum computing/semiconductor stocks)
    sector_etfs = {
        'Semiconductors': 'SMH',
        'Technology': 'XLK',
        'Quantum/Computing': 'QTUM',
    }
    
    sector_data = {}
    for sector_name, etf_ticker in sector_etfs.items():
        try:
            etf = yf.Ticker(etf_ticker)
            etf_hist = etf.history(period="5d")
            if not etf_hist.empty:
                etf_close = etf_hist['Close']
                sector_data[sector_name] = {
                    'change_5d': round(((etf_close.iloc[-1] / etf_close.iloc[0]) - 1) * 100, 1),
                    'price': round(etf_close.iloc[-1], 2)
                }
        except:
            pass
    
    macro_data['sector_data'] = sector_data
    
    return macro_data

def get_sector_for_ticker(ticker):
    """Get sector and industry for a ticker"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return info.get('sector', 'Unknown'), info.get('industry', 'Unknown')
    except:
        return 'Unknown', 'Unknown'

def get_market_events_today():
    """Get major market events/news headlines"""
    try:
        api_key = st.secrets.get("FINNHUB_API_KEY")
        if not api_key:
            return []
        
        url = "https://finnhub.io/api/v1/news"
        params = {
            'category': 'general',
            'token': api_key
        }
        response = requests.get(url, params=params)
        if response.status_code == 200:
            articles = response.json()
            return articles[:3]  # Top 3 headlines
        return []
    except:
        return []

def get_stock_news(ticker, limit=5):
    """
    Get recent news for a specific stock using Finnhub
    
    Args:
        ticker: Stock ticker symbol
        limit: Number of news articles to return
    
    Returns:
        List of news articles with headlines and summaries
    """
    try:
        api_key = st.secrets.get("FINNHUB_API_KEY")
        if not api_key:
            return []
        
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
            return []
        
        articles = response.json()
        if not articles:
            return []
        
        # Format and return top articles
        formatted_news = []
        for item in articles[:limit]:
            formatted_news.append({
                'headline': item.get('headline', 'No title'),
                'summary': item.get('summary', '')[:200],
                'source': item.get('source', 'Unknown'),
                'datetime': datetime.fromtimestamp(item.get('datetime', 0)).strftime('%Y-%m-%d %H:%M'),
                'url': item.get('url', '#')
            })
        
        return formatted_news
    except Exception as e:
        return []

# ========================
# PUT/CALL RATIO & BETA ADJUSTMENT
# ========================

def calculate_put_call_ratio(ticker, expiry):
    try:
        calls_df, puts_df = get_cached_option_chain(ticker, expiry)
        if calls_df is None or puts_df is None:
            return None, None, None, None, None
        
        call_volume = calls_df['volume'].sum() if 'volume' in calls_df.columns else 0
        put_volume = puts_df['volume'].sum() if 'volume' in puts_df.columns else 0
        
        if call_volume == 0:
            return None, None, None, None, None
        
        pc_ratio = put_volume / call_volume
        
        if pc_ratio > 1.2:
            sentiment = "🔴 Bearish (High put volume)"
            interpretation = "Market participants are hedging or expecting downside"
        elif pc_ratio < 0.8:
            sentiment = "🟢 Bullish (High call volume)"
            interpretation = "Market participants are optimistic"
        else:
            sentiment = "⚪ Neutral"
            interpretation = "Balanced sentiment between puts and calls"
        
        return round(pc_ratio, 2), sentiment, interpretation, call_volume, put_volume
    except Exception as e:
        return None, None, None, None, None

def calculate_beta(ticker, market_ticker="SPY", period="1y"):
    try:
        stock = yf.Ticker(ticker)
        market = yf.Ticker(market_ticker)
        
        stock_hist = stock.history(period=period)
        market_hist = market.history(period=period)
        
        if stock_hist.empty or market_hist.empty:
            return 1.0, "Insufficient data"
        
        stock_returns = stock_hist['Close'].pct_change().dropna()
        market_returns = market_hist['Close'].pct_change().dropna()
        
        common_dates = stock_returns.index.intersection(market_returns.index)
        if len(common_dates) < 30:
            return 1.0, "Insufficient data"
        
        stock_returns_aligned = stock_returns[common_dates]
        market_returns_aligned = market_returns[common_dates]
        
        covariance = np.cov(stock_returns_aligned, market_returns_aligned)[0][1]
        variance = np.var(market_returns_aligned)
        
        if variance == 0:
            return 1.0, "Calculation error"
        
        beta = covariance / variance
        
        if beta > 1.5:
            interpretation = "🔴 HIGH BETA - Significantly more volatile than market"
        elif beta > 1.2:
            interpretation = "🟡 ELEVATED BETA - More volatile than market"
        elif beta > 0.8:
            interpretation = "🟢 MARKET BETA - Similar volatility to market"
        elif beta > 0.5:
            interpretation = "🟡 LOW BETA - Less volatile than market"
        else:
            interpretation = "🔵 VERY LOW BETA - Defensive stock"
        
        return round(beta, 2), interpretation
    except Exception as e:
        return 1.0, f"Error: {str(e)[:50]}"

def calculate_beta_adjusted_risk(base_risk_score, beta):
    adjustment_factor = 0.7 + (beta * 0.3)
    adjustment_factor = max(0.5, min(1.5, adjustment_factor))
    
    adjusted_risk = base_risk_score * adjustment_factor
    adjusted_risk = max(0, min(100, adjusted_risk))
    
    return round(adjusted_risk, 1), adjustment_factor

def calculate_iv_hv_spread(current_iv, hv):
    if current_iv is None or hv == 0:
        return 0, "N/A"
    spread = (current_iv * 100) - hv
    if spread > 10:
        status = "🔴 Expensive (IV > HV)"
    elif spread < -10:
        status = "🟢 Cheap (IV < HV)"
    else:
        status = "🟡 Fair (IV ≈ HV)"
    return round(spread, 1), status

def calculate_skew(calls_df, puts_df, current_price, at_the_money_strike=None):
    try:
        if at_the_money_strike is None:
            atm_strike = current_price
        else:
            atm_strike = at_the_money_strike
        
        call_atm = calls_df.iloc[(calls_df['strike'] - atm_strike).abs().argsort()[:1]]
        put_atm = puts_df.iloc[(puts_df['strike'] - atm_strike).abs().argsort()[:1]]
        
        call_iv = call_atm['impliedVolatility'].iloc[0] if not call_atm.empty else 0
        put_iv = put_atm['impliedVolatility'].iloc[0] if not put_atm.empty else 0
        
        skew = put_iv - call_iv if put_iv and call_iv else 0
        
        if skew > 0.05:
            status = "⚠️ Negative Skew (Puts expensive - Bearish bias)"
        elif skew < -0.05:
            status = "✅ Positive Skew (Calls expensive - Bullish bias)"
        else:
            status = "⚪ Neutral Skew"
        
        return round(skew * 100, 1), status
    except Exception:
        return 0, "N/A"

def calculate_term_structure(ticker, expiries):
    term_data = []
    for exp in expiries[:5]:
        try:
            calls, puts = get_cached_option_chain(ticker, exp)
            if calls is not None and not calls.empty:
                atm_idx = (calls['strike'] - st.session_state.price).abs().argsort()[:3]
                avg_iv = calls.iloc[atm_idx]['impliedVolatility'].mean()
                days = (pd.to_datetime(exp).date() - datetime.now().date()).days
                term_data.append({'expiry': exp, 'days': days, 'iv': avg_iv})
        except:
            continue
    
    if len(term_data) >= 2:
        if term_data[0]['iv'] < term_data[-1]['iv']:
            structure = "🟢 Contango (Upward sloping - Normal)"
        else:
            structure = "🔴 Backwardation (Downward sloping - Stress)"
        return term_data, structure
    return term_data, "Insufficient data"

def calculate_max_pain(calls_df, puts_df, strikes):
    try:
        max_pain = None
        min_pain_value = float('inf')
        
        for strike in strikes:
            call_pain = 0
            put_pain = 0
            
            call_matches = calls_df[calls_df['strike'] == strike]
            if not call_matches.empty:
                call_oi = call_matches['openInterest'].iloc[0] if 'openInterest' in call_matches.columns else 0
                call_pain = call_oi * max(0, st.session_state.price - strike) / 100
            
            put_matches = puts_df[puts_df['strike'] == strike]
            if not put_matches.empty:
                put_oi = put_matches['openInterest'].iloc[0] if 'openInterest' in put_matches.columns else 0
                put_pain = put_oi * max(0, strike - st.session_state.price) / 100
            
            total_pain = call_pain + put_pain
            if total_pain < min_pain_value:
                min_pain_value = total_pain
                max_pain = strike
        
        return max_pain
    except Exception:
        return None

def get_vix():
    try:
        vix = yf.Ticker("^VIX")
        hist = vix.history(period="1d")
        if not hist.empty:
            return round(hist['Close'].iloc[-1], 1)
        return 15.0
    except:
        return 15.0

def get_market_regime(vix):
    if vix < 12:
        return "🟢 LOW VOL", "Options cheap, low probability of large moves"
    elif vix < 20:
        return "🟢 OPTIMAL", "Normal volatility regime. 60 DTE calls favorable."
    elif vix < 25:
        return "🟡 ELEVATED", "Elevated volatility. Options expensive but higher gamma potential"
    else:
        return "🔴 HIGH VOL", "High volatility. Options overpriced. Caution with long calls"

def get_earnings_date(ticker):
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

# ========================
# PHASE 5: HELPER FUNCTIONS
# ========================

def calculate_gamma_theta_ratio(gamma, theta):
    """Calculate Gamma/Theta ratio for acceleration potential."""
    # Handle None values
    if gamma is None or theta is None:
        return 0.0
    if theta == 0:
        return 0.0
    try:
        gamma = float(gamma)
        theta = float(theta)
        ratio = abs(gamma / theta) if theta != 0 else 0.0
        return min(ratio, 3.0)
    except (TypeError, ValueError, ZeroDivisionError):
        return 0.0
        
def apply_skew_penalty(ev, skew):
    """Apply penalty to EV based on put/call skew."""
    if skew < -0.05:
        return ev * 0.85
    elif skew > 0.05:
        return ev * 1.05
    return ev

def calculate_enhanced_cts(delta, p_touch, gamma_theta_ratio, tech_score):
    """
    Enhanced Composite Score with Gamma/Theta ratio.
    Weights: Delta 30%, Touch Prob 30%, Gamma/Theta 20%, Technical 20%
    """
    # Ensure all values are numeric and within valid ranges
    delta = float(delta) if delta is not None else 0.0
    p_touch = float(p_touch) if p_touch is not None else 0.0
    gamma_theta_ratio = float(gamma_theta_ratio) if gamma_theta_ratio is not None else 0.0
    tech_score = float(tech_score) if tech_score is not None else 0.0
    
    # Clamp values to valid ranges
    delta = max(0.0, min(1.0, delta))
    p_touch = max(0.0, min(1.0, p_touch))
    gamma_theta_ratio = max(0.0, gamma_theta_ratio)
    tech_score = max(0.0, min(3.0, tech_score))
    
    # Normalize gamma_theta_ratio (cap at 1.0 for the score)
    normalized_gt = min(gamma_theta_ratio / 2.0, 1.0)
    
    cts = (delta * 0.30 + 
           p_touch * 0.30 + 
           normalized_gt * 0.20 + 
           (tech_score / 3.0) * 0.20)
    
    return int(cts * 100)

def get_best_contract_for_strategy(ticker, profit_target_pct, stop_loss_pct, current_price, min_dte, max_dte, delta_min, delta_max):
    """Get the best contract for a strategy within a DTE range."""
    try:
        stock_obj = yf.Ticker(ticker)
        all_expiries = list(stock_obj.options)
        
        if not all_expiries:
            return None
        
        today = datetime.now().date()
        
        # Filter expiries by DTE range
        valid_expiries = []
        for exp in all_expiries:
            days = (pd.to_datetime(exp).date() - today).days
            if min_dte <= days <= max_dte:
                valid_expiries.append((exp, days))
        
        if not valid_expiries:
            return None
        
        best_contract = None
        best_ev = -float('inf')
        
        for exp_date, days_exp in valid_expiries:
            calls_df, puts_df = get_cached_option_chain(ticker, exp_date)
            if calls_df is None or calls_df.empty:
                continue
            
            t_yrs = days_exp / 365
            
            for _, row in calls_df.iterrows():
                mid_p = (row['bid'] + row['ask']) / 2 if row['bid'] > 0 else row['lastPrice']
                if mid_p <= 0 or row['impliedVolatility'] <= 0:
                    continue
                
                d, g, t, v = calculate_greeks(current_price, row['strike'], t_yrs, 0.05, row['impliedVolatility'])
                
                if delta_min <= d <= delta_max:
                    p_touch = calculate_p_touch(current_price, row['strike'], t_yrs, row['impliedVolatility'])
                    ev_val = (p_touch * (mid_p * (1 + profit_target_pct / 100))) - ((1 - p_touch) * (mid_p * (stop_loss_pct / 100)))
                    
                    # Apply skew penalty
                    try:
                        if puts_df is not None and not puts_df.empty:
                            skew_val, _ = calculate_skew(calls_df, puts_df, current_price, row['strike'])
                            ev_val = apply_skew_penalty(ev_val, skew_val / 100)
                    except:
                        pass
                    
                    # Calculate CTS with Gamma/Theta
                    gt_ratio = calculate_gamma_theta_ratio(g, t)
                    cts = calculate_enhanced_cts(d, p_touch, gt_ratio, 2)  # Default tech_score = 2
                    
                    if ev_val > best_ev:
                        best_ev = ev_val
                        best_contract = {
                            'strike': row['strike'],
                            'mid': mid_p,
                            'expiry': exp_date,
                            'days': days_exp,
                            'delta': d,
                            'ev': ev_val,
                            'cts': cts,
                            'iv': row['impliedVolatility'],
                            'gamma': g,
                            'theta': t,
                            'gamma_theta_ratio': gt_ratio
                        }
        
        return best_contract
    except Exception as e:
        return None

# --- DASHBOARD HELPER FUNCTIONS ---
def calculate_strict_verdict(vix, rsi, iv_hv_spread, earnings_days, sentiment_score):
    conditions_passed = 0
    
    if 12 <= vix <= 25:
        conditions_passed += 1
    
    if 30 <= rsi <= 70:
        conditions_passed += 1
    
    if -10 <= iv_hv_spread <= 10:
        conditions_passed += 1
    
    if earnings_days is None or earnings_days > 7:
        conditions_passed += 1
    
    if sentiment_score > -0.3:
        conditions_passed += 1
    
    if conditions_passed >= 4:
        return "✅ BUY", conditions_passed, 72 + (conditions_passed * 5)
    elif conditions_passed >= 2:
        return "⏳ WAIT", conditions_passed, 40 + (conditions_passed * 10)
    else:
        return "❌ DROP", conditions_passed, 20 + (conditions_passed * 5)

def calculate_weighted_verdict(vix, rsi, iv_hv_spread, sentiment_score, skew, beta):
    if 12 <= vix <= 25:
        vix_score = 100
    elif vix < 12:
        vix_score = 50
    else:
        vix_score = max(0, 100 - (vix - 25) * 4)
    
    if 30 <= rsi <= 70:
        rsi_score = 100
    elif rsi > 70:
        rsi_score = max(0, 100 - (rsi - 70) * 3)
    else:
        rsi_score = max(0, 100 - (30 - rsi) * 3)
    
    if -10 <= iv_hv_spread <= 10:
        spread_score = 100
    elif iv_hv_spread > 20:
        spread_score = 0
    elif iv_hv_spread < -20:
        spread_score = 50
    else:
        spread_score = 100 - abs(iv_hv_spread) * 2
    
    sentiment_score_normalized = max(0, min(100, (sentiment_score + 1) * 50))
    
    if skew > 0:
        skew_score = 100
    elif skew == 0:
        skew_score = 60
    else:
        skew_score = 20
    
    if 0.8 <= beta <= 1.2:
        beta_score = 100
    elif 1.2 < beta <= 1.5:
        beta_score = 70
    elif beta < 0.8:
        beta_score = 80
    else:
        beta_score = 40
    
    total_weighted = (vix_score * 0.20 + rsi_score * 0.15 + spread_score * 0.20 + 
                      sentiment_score_normalized * 0.20 + skew_score * 0.15 + beta_score * 0.10)
    
    confidence = round(total_weighted, 1)
    
    if confidence >= 65:
        return "✅ BUY", confidence
    elif confidence >= 40:
        return "⏳ WAIT", confidence
    else:
        return "❌ DROP", confidence

def calculate_path_expectation(rsi, sentiment_score, atr_trend, vix_trend):
    if rsi > 55 and sentiment_score > 0.3:
        return "🚀 Straight up", "Strong momentum + bullish sentiment"
    elif 40 <= rsi <= 55 and sentiment_score > 0:
        return "📉📈 Pullback then rise", "Neutral RSI with positive sentiment - expect dip before breakout"
    elif rsi < 40 and sentiment_score < 0:
        return "📉 More downside first", "Weak RSI + bearish sentiment"
    else:
        return "📈 Gradual rise", "Mixed signals - expect slower appreciation"

def get_strategy_for_expiry(ticker, expiry, delta_min, delta_max, profit_target_pct, stop_loss_pct, current_price):
    """Get the best strategy contract for a given expiry and delta range."""
    try:
        calls_df, _ = get_cached_option_chain(ticker, expiry)
        if calls_df is None or calls_df.empty:
            return None
        
        days_to_expiry = (pd.to_datetime(expiry).date() - datetime.now().date()).days
        t_yrs = max(days_to_expiry, 1) / 365
        
        best_contract = None
        best_ev = -float('inf')
        
        for _, row in calls_df.iterrows():
            mid_p = (row['bid'] + row['ask']) / 2 if row['bid'] > 0 else row['lastPrice']
            if mid_p <= 0 or row['impliedVolatility'] <= 0:
                continue
            
            d, g, t, v = calculate_greeks(current_price, row['strike'], t_yrs, 0.05, row['impliedVolatility'])
            
            if delta_min <= d <= delta_max:
                p_touch = calculate_p_touch(current_price, row['strike'], t_yrs, row['impliedVolatility'])
                ev_val = (p_touch * (mid_p * (1 + profit_target_pct / 100))) - ((1 - p_touch) * (mid_p * (stop_loss_pct / 100)))
                cts = int(((d * 0.4) + (p_touch * 0.4) + (0.5 * 0.2)) * 100)
                
                if ev_val > best_ev:
                    best_ev = ev_val
                    best_contract = {
                        'strike': row['strike'],
                        'mid': mid_p,
                        'delta': d,
                        'ev': ev_val,
                        'cts': cts,
                        'days': days_to_expiry,
                        'iv': row['impliedVolatility'],
                        'gamma': g,
                        'theta': t,
                        'gamma_theta_ratio': calculate_gamma_theta_ratio(g, t)
                    }
        
        return best_contract
    except Exception:
        return None

def estimate_pullback_entry(current_price, atr_pct, rsi, sentiment_score, curr_indicators, max_pain=None):
    """
    Estimate optimal entry price during pullback based on multiple technical factors
    """
    # Pullback magnitude based on RSI
    if 40 <= rsi <= 45:  # Lower RSI = deeper pullback likely
        pullback_pct = atr_pct * 1.5  # 1.5x ATR for deeper pullback
        confidence = "Medium-High"
    elif 45 < rsi <= 50:  # Mid-range RSI
        pullback_pct = atr_pct * 1.0  # 1x ATR
        confidence = "High"
    else:  # 50-55 RSI
        pullback_pct = atr_pct * 0.7  # 0.7x ATR for shallow pullback
        confidence = "Medium"
    
    # Sentiment adjustment (stronger sentiment = shallower pullback)
    if sentiment_score > 0.5:
        pullback_pct = pullback_pct * 0.7  # Shallow pullback
        sentiment_effect = "Strong bullish sentiment limiting downside"
    elif sentiment_score > 0.2:
        pullback_pct = pullback_pct * 0.85  # Mild pullback
        sentiment_effect = "Positive sentiment providing support"
    else:
        sentiment_effect = "Neutral sentiment - technical pullback expected"
    
    estimated_entry = current_price * (1 - pullback_pct / 100)
    
    # Get technical levels
    lower_band = curr_indicators['lower']
    middle_band = curr_indicators['sma20']
    
    return {
        'estimated_entry': round(estimated_entry, 2),
        'pullback_percent': round(pullback_pct, 1),
        'confidence': confidence,
        'sentiment_effect': sentiment_effect,
        'technical_levels': {
            'sma20_support': round(middle_band, 2),
            'lower_bb_support': round(lower_band, 2),
            'max_pain_support': round(max_pain, 2) if max_pain and max_pain < current_price else None
        }
    }

def get_pullback_entry_recommendation(current_price, hist_data, max_pain, rsi, sentiment_score, atr_pct):
    """
    Generate complete pullback entry recommendation with trend awareness
    """
    # Get technical indicators
    curr_indicators, _ = get_technicals(hist_data)
    
    # Determine trend strength - how far price is above SMA20
    price_vs_sma20 = ((current_price / curr_indicators['sma20']) - 1) * 100
    
    # Trend factor - reduces pullback expectations in strong uptrends
    if price_vs_sma20 > 15:  # Extended rally (like IREN +39%)
        trend_factor = 0.4
        max_allowed_pullback = 5
        trend_note = "Strong uptrend - limited pullback expected (4-5%)"
    elif price_vs_sma20 > 10:
        trend_factor = 0.55
        max_allowed_pullback = 7
        trend_note = "Moderate uptrend - shallow pullback expected (5-7%)"
    elif price_vs_sma20 > 5:
        trend_factor = 0.7
        max_allowed_pullback = 9
        trend_note = "Mild uptrend - moderate pullback possible (7-9%)"
    elif price_vs_sma20 < -8:
        trend_factor = 1.3
        max_allowed_pullback = 15
        trend_note = "Downtrend - deeper pullback possible"
    else:
        trend_factor = 1.0
        max_allowed_pullback = 10
        trend_note = "Normal range"
    
    # For strong trends, use SMA20 as primary support, not lower BB
    if price_vs_sma20 > 10:
        bollinger_entry = curr_indicators['sma20']  # Use SMA20 instead of lower BB
    else:
        bollinger_entry = curr_indicators['lower']
    
    sma_entry = curr_indicators['sma20']
    max_pain_entry = max_pain if max_pain and max_pain < current_price else None
    
    # Calculate ATR-based entry with trend factor
    atr_entry = current_price * (1 - (atr_pct * trend_factor) / 100)
    
    # Calculate confidence-weighted average entry
    entries = []
    weights = []
    
    # ATR-based entry (weight: rsi-dependent)
    if rsi < 45:
        atr_weight = 0.35
    else:
        atr_weight = 0.25
    entries.append(atr_entry)
    weights.append(atr_weight)
    
    # Bollinger/SMA entry (higher weight)
    entries.append(bollinger_entry)
    weights.append(0.35)
    
    # SMA20 entry (higher weight in strong trends)
    sma_weight = 0.3 if price_vs_sma20 > 10 else 0.2
    entries.append(sma_entry)
    weights.append(sma_weight)
    
    # Max Pain entry (if available)
    if max_pain_entry:
        entries.append(max_pain_entry)
        weights.append(0.2)
    else:
        # Normalize weights if no max pain
        weights = [w/sum(weights) for w in weights]
    
    # Calculate weighted average
    weighted_entry = sum(e * w for e, w in zip(entries, weights))
    
    # Sentiment adjustment (more conservative)
    if sentiment_score > 0.5:
        entry_adjustment = 0.98  # Only 2% pullback for strong bullish
        sentiment_text = "Strongly Bullish - Limited downside expected"
    elif sentiment_score > 0.2:
        entry_adjustment = 0.96  # 4% pullback for bullish
        sentiment_text = "Bullish - Moderate pullback possible"
    elif sentiment_score > 0:
        entry_adjustment = 0.94  # 6% pullback for leaning bullish
        sentiment_text = "Leaning Bullish - Moderate pullback possible"
    else:
        entry_adjustment = 0.91  # 9% pullback for neutral
        sentiment_text = "Neutral - Technical pullback expected"
    
    final_entry = weighted_entry * entry_adjustment
    
    # FINAL SANITY CHECK - Cap based on trend
    actual_pullback_pct = ((current_price - final_entry) / current_price) * 100
    
    if actual_pullback_pct > max_allowed_pullback:
        # Cap the pullback
        final_entry = current_price * (1 - max_allowed_pullback / 100)
        actual_pullback_pct = max_allowed_pullback
    
    # Ensure minimum entry is reasonable (not more than 10% below SMA20)
    min_reasonable_entry = curr_indicators['sma20'] * 0.92
    if final_entry < min_reasonable_entry:
        final_entry = min_reasonable_entry
        actual_pullback_pct = ((current_price - final_entry) / current_price) * 100
    
    # Calculate estimated time to pullback (in days)
    if rsi > 65:
        estimated_days = 1
    elif rsi > 55:
        estimated_days = 2
    elif rsi > 45:
        estimated_days = 3
    elif rsi > 35:
        estimated_days = 4
    else:
        estimated_days = 5
    
    # Generate realistic entry zones
    conservative_entry = final_entry * 1.02  # 2% above optimal
    aggressive_entry = final_entry * 0.98    # 2% below optimal
    
    return {
        'estimated_entry': round(final_entry, 2),
        'entry_range': {
            'aggressive': round(aggressive_entry, 2),
            'conservative': round(conservative_entry, 2),
            'optimal': round(final_entry, 2)
        },
        'support_levels': {
            'first_support': round(sma_entry, 2),
            'second_support': round(curr_indicators['lower'], 2),
            'deep_support': round(max_pain_entry, 2) if max_pain_entry else round(curr_indicators['lower'] * 0.97, 2)
        },
        'estimated_pullback_pct': round(actual_pullback_pct, 1),
        'estimated_days_to_pullback': estimated_days,
        'sentiment_text': sentiment_text,
        'trend_note': trend_note,
        'current_price': current_price,
        'atr_pct': round(atr_pct, 1),
        'price_vs_sma20': round(price_vs_sma20, 1)
    }

# ========================
# GROQ RETRY LOGIC
# ========================
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

# --- AI RESEARCH WITH SENTIMENT ---
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
    cache_key = f"news_sentiment_{ticker}"
    cached_response = st.session_state.ai_cache.get(cache_key)
    if cached_response:
        return cached_response
    
    news_articles = fetch_news_finnhub(ticker)
    if not news_articles:
        return f"ℹ️ No recent news found for {ticker}"
    
    current_price = get_cached_current_price(ticker)
    price_context = f" at ${current_price:.2f}" if current_price else ""
    
    news_text = "\n\n".join([
        f"**News {i+1}** (Source: {item['publisher']}, Time: {item['datetime']})\n"
        f"Title: {item['title']}\n"
        f"Summary: {item['summary']}"
        for i, item in enumerate(news_articles)
    ])
    
    prompt = f"""
    You are a quantitative financial analyst. Analyze the following news articles for stock {ticker}{price_context}.
    
    NEWS ARTICLES:
    {news_text}
    
    Return ONLY valid JSON. Do not include any other text. Use this exact structure:
    {{
        "sentiment_score": float between -1.0 and 1.0,
        "sentiment_label": "Bullish" or "Neutral" or "Bearish",
        "catalyst": "description of upcoming event or null",
        "catalyst_date": "YYYY-MM-DD or null",
        "catalyst_impact": "High/Medium/Low or null",
        "key_themes": ["theme1", "theme2", "theme3"],
        "summary": "brief 1-sentence summary",
        "risk_adjustment": integer between -20 and 20
    }}
    """
    
    try:
        client = Groq(api_key=groq_api_key)
        response_text = call_groq_with_retry(client, prompt)
        
        if response_text is None:
            result = f"### 📰 Recent News for {ticker}\n\n"
            for i, item in enumerate(news_articles[:5]):
                result += f"**{i+1}. {item['title']}**  \n📌 {item['publisher']}\n\n"
            sentiment_data = {"sentiment_score": 0.0, "sentiment_label": "Neutral", "risk_adjustment": 0}
        else:
            import json as json_lib
            try:
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                if json_start != -1 and json_end > json_start:
                    json_str = response_text[json_start:json_end]
                    sentiment_data = json_lib.loads(json_str)
                else:
                    raise ValueError("No JSON found")
            except:
                sentiment_data = {"sentiment_score": 0.0, "sentiment_label": "Neutral", "risk_adjustment": 0}
            
            sentiment_color = "🟢" if sentiment_data.get('sentiment_score', 0) > 0.3 else ("🔴" if sentiment_data.get('sentiment_score', 0) < -0.3 else "⚪")
            result = f"""
### 🤖 AI Sentiment Analysis for {ticker}

{sentiment_color} **Sentiment Score:** {sentiment_data.get('sentiment_score', 0):.2f} ({sentiment_data.get('sentiment_label', 'Neutral')})

**Key Themes:**
"""
            for theme in sentiment_data.get('key_themes', [])[:3]:
                result += f"- {theme}\n"
            
            if sentiment_data.get('catalyst'):
                result += f"""
**📅 Catalyst Detected:** {sentiment_data['catalyst']}
   - Impact: {sentiment_data.get('catalyst_impact', 'Medium')}
"""
            
            result += f"""
**📝 Summary:** {sentiment_data.get('summary', 'No summary available')}

---
### 📰 News Headlines
"""
            for i, item in enumerate(news_articles[:5]):
                result += f"**{i+1}. {item['title']}**  \n📌 {item['publisher']} | 🕐 {item['datetime']}\n\n"
        
        st.session_state.ai_cache.set(cache_key, result)
        st.session_state.current_sentiment = sentiment_data
        
        return result
    except Exception as e:
        fallback = f"### 📰 Recent News for {ticker}\n\n"
        for i, item in enumerate(news_articles[:5]):
            fallback += f"**{i+1}. {item['title']}**  \n📌 {item['publisher']} | 🕐 {item['datetime']}\n\n"
        return fallback

# ========================
# AI CHAT FUNCTIONS
# ========================
def get_ai_chat_response(ticker, user_question, chat_history, stock_metrics):
    """
    Get AI response for follow-up questions with full context
    """
    # Build context string from stock_metrics
    context_str = ""
    for key, value in stock_metrics.items():
        if value is not None:
            context_str += f"- {key}: {value}\n"
    
    # Format chat history
    history_str = ""
    for msg in chat_history[-5:]:  # Last 5 messages for context
        if msg['role'] == 'user':
            history_str += f"User: {msg['content']}\n"
        else:
            history_str += f"AI: {msg['content']}\n"
    
    prompt = f"""
You are a professional options trader and quantitative analyst. You have access to the following data for {ticker}:

STOCK METRICS:
{context_str}

CONVERSATION HISTORY:
{history_str}

USER QUESTION: {user_question}

Provide a professional, concise, and actionable response. Focus on options trading implications. 
If the user asks about anything not related to stocks, options, or trading, politely redirect them to trading-related topics.
Keep your response to 2-4 paragraphs maximum.
"""
    
    try:
        groq_api_key = st.secrets.get("GROQ_API_KEY")
        if not groq_api_key:
            return "⚠️ Groq API key not configured. Please add it to your secrets."
        
        client = Groq(api_key=groq_api_key)
        response = call_groq_with_retry(client, prompt)
        
        if response:
            return response
        else:
            return "⚠️ AI response temporarily unavailable. Please try again later."
    except Exception as e:
        return f"⚠️ Error: {str(e)}"

# --- CORE MATH & OPTIONS QUANT ENGINES ---
def calculate_greeks(S, K, T, r, sigma, type="call"):
    if T <= 0 or sigma <= 0 or S <= 0: return 0.0, 0.0, 0.0, 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    delta = norm.cdf(d1) if type == "call" else norm.cdf(d1) - 1
    
    # FIXED: Gamma formula - remove the division by S that was incorrect
    # The correct formula is: gamma = norm.pdf(d1) / (S * sigma * sqrt(T))
    # This returns values typically 0.02-0.08 for ATM options
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    
    # Theta for call option
    if type == "call":
        theta = (- (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    else:
        theta = (- (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
    
    vega = (S * norm.pdf(d1) * np.sqrt(T)) / 100
    
    return round(delta, 3), round(gamma, 4), round(theta, 3), round(vega, 3)

def calculate_p_touch(S, K, T, sigma):
    if T <= 0 or sigma <= 0 or S <= 0: return 0.0
    d1 = (np.log(S / K) + (0.05 + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    p_itm = norm.cdf(d1) if S < K else 1.0 - norm.cdf(d1)
    p_touch = min(p_itm * 2.0, 0.99)
    return round(p_touch, 3)

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
    'ai_cache': None, 'profit_target_pct': 100, 'stop_loss_pct': 30,
    'current_sentiment': None, 'active_tab': 0, 'last_ai_refresh': None,
    'chat_history': [], 'initial_ai_message_sent': False
}
for key, default in state_keys.items():
    if key not in st.session_state:
        st.session_state[key] = default

if st.session_state.ai_cache is None:
    st.session_state.ai_cache = SimpleCache()

# --- SIDEBAR (SIMPLIFIED) ---
with st.sidebar:
    st.header("🎮 Control Center")
    
    st.divider()
     
    # --- Analysis Section ---
    st.subheader("🔍 Analysis")
    ticker_input = st.text_input("Ticker:", "SHOP").upper()
    fetch_btn = st.button("🚀 Analyze Options Structure", use_container_width=True)
    
    st.divider()
    
    # --- Exit & Hold Adjuster ---
    st.subheader("🧪 Exit & Hold Adjuster")
    profit_target_pct = st.slider("Target Option Profit Booking (%)", 10, 150, 100, step=5)
    stop_loss_pct = st.slider("Max Stop Loss (%)", 10, 100, 30, step=5)
    st.session_state.profit_target_pct = profit_target_pct
    st.session_state.stop_loss_pct = stop_loss_pct
    
    st.divider()
    
    # --- Workspace Adjuster (only shows if expiries exist) ---
    if st.session_state.expiries and len(st.session_state.expiries) > 0:
        st.subheader("🔍 Workspace Adjuster")
        
        current_index = 0
        if st.session_state.last_selected_expiry and st.session_state.last_selected_expiry in st.session_state.expiries:
            current_index = st.session_state.expiries.index(st.session_state.last_selected_expiry)
        
        expiry = st.selectbox(
            "Select Expiry for Analysis Tabs:", 
            st.session_state.expiries,
            index=current_index,
            help="Select any expiry to update the Conservative, Aggressive, and Speculative tabs below"
        )
        
        if expiry != st.session_state.get('last_selected_expiry'):
            st.session_state.last_selected_expiry = expiry
            st.rerun()
    else:
        if fetch_btn:
            st.info("⏳ Loading expiries... Please wait a moment.")
        else:
            st.info("👈 Enter a ticker and click 'Analyze' to see option chains")
    
    st.divider()
    
    # --- Clear Cache ---
    if st.button("🗑️ Clear Cache", help="Clear cached data if you're seeing stale information", use_container_width=True):
        st.cache_data.clear()
        for key in ['expiries', 'last_selected_expiry', 'price', 'hist_data', 'stock_name', 'trend', 'pct_change', 'tech_score', 'verdict_reasons', 'global_conservative', 'global_aggressive', 'global_speculative', 'data_fetched']:
            if key in st.session_state:
                del st.session_state[key]
        st.success("Cache cleared! Refresh the page to reload data.")
        st.rerun()

# --- DATA FETCHING & GLOBAL SCANS (Analysis Tab) ---
if fetch_btn:
    st.session_state.current_ticker = ticker_input
    st.session_state.ai_brief = "" 
    st.session_state.global_conservative = None
    st.session_state.global_aggressive = None
    st.session_state.global_speculative = None
    st.session_state.last_selected_expiry = None
    st.session_state.data_fetched = True
    st.session_state.chat_history = []  # Reset chat when new ticker is analyzed
    st.session_state.initial_ai_message_sent = False
    
    try:
        hist = get_cached_stock_history(ticker_input, "100d")
        
        if hist is None or hist.empty or 'Close' not in hist.columns:
            st.error(f"❌ No valid history found for {ticker_input}")
            st.session_state.price = None
        else:
            st.session_state.hist_data = hist
            
            # Get current price with fallbacks
            current_price = get_cached_current_price(ticker_input)
            
            # If price fetch failed, use last close from history
            if current_price is None or pd.isna(current_price):
                if not hist['Close'].isna().all():
                    current_price = float(hist['Close'].iloc[-1])
                    st.caption("⚠️ Using last closing price (market may be closed)")
                else:
                    st.error(f"❌ Could not determine current price for {ticker_input}")
                    st.session_state.price = None
            
            if current_price is not None and not pd.isna(current_price):
                st.session_state.price = float(current_price)
            else:
                st.session_state.price = None
                st.error(f"❌ No valid price found for {ticker_input}")
            
            stock_info = get_cached_stock_info(ticker_input)
            st.session_name = stock_info.get('longName', ticker_input) if stock_info else ticker_input
            st.session_state.stock_name = st.session_name
            
            stock_obj = yf.Ticker(ticker_input)
            all_expiries = list(stock_obj.options)
            st.session_state.expiries = all_expiries
            
            # Set default expiry to closest to 90 days (Conservative strategy)
            today = datetime.now().date()
            closest_to_90 = None
            closest_diff_90 = float('inf')
            for exp in all_expiries:
                days = (pd.to_datetime(exp).date() - today).days
                diff = abs(days - 90)
                if diff < closest_diff_90:
                    closest_diff_90 = diff
                    closest_to_90 = exp
            
            if closest_to_90:
                st.session_state.last_selected_expiry = closest_to_90
            
            sma20_val = hist['Close'].rolling(window=20).mean().iloc[-1]
            st.session_state.trend = "Bullish" if st.session_state.price > sma20_val else "Bearish"
            st.session_state.pct_change = ((st.session_state.price / hist['Close'].iloc[-20]) - 1) * 100
            
            df_tech_init = hist.copy()
            curr_init, prev_init = get_technicals(df_tech_init)
            
            st.session_state.tech_score = 0
            st.session_state.verdict_reasons = []
            if curr_init['ema8'] > curr_init['ema20']:
                st.session_state.tech_score += 1
                st.session_state.verdict_reasons.append("Short-term momentum (8 EMA) is leading.")
            if curr_init['hist'] > prev_init['hist']:
                st.session_state.tech_score += 1
                st.session_state.verdict_reasons.append("MACD histogram is rising.")
            if st.session_state.price > sma20_val:
                st.session_state.tech_score += 1
                st.session_state.verdict_reasons.append("Price is above 20-day baseline.")

            cons_contract = get_best_contract_for_strategy(
                ticker_input, profit_target_pct, stop_loss_pct, st.session_state.price,
                min_dte=60, max_dte=365, delta_min=0.50, delta_max=0.60
            )
            
            aggr_contract = get_best_contract_for_strategy(
                ticker_input, profit_target_pct, stop_loss_pct, st.session_state.price,
                min_dte=30, max_dte=45, delta_min=0.40, delta_max=0.49
            )
            
            spec_contract = get_best_contract_for_strategy(
                ticker_input, profit_target_pct, stop_loss_pct, st.session_state.price,
                min_dte=15, max_dte=30, delta_min=0.30, delta_max=0.39
            )
            
            if cons_contract:
                st.session_state.global_conservative = cons_contract
            if aggr_contract:
                st.session_state.global_aggressive = aggr_contract
            if spec_contract:
                st.session_state.global_speculative = spec_contract
            
            # Store path expectation values
            vix_value = get_vix()
            atr_val, atr_pct_stored = calculate_atr(hist)
            rsi_val_stored = calculate_rsi(hist)
            
            st.session_state.saved_rsi = rsi_val_stored
            st.session_state.saved_atr_pct = atr_pct_stored
            st.session_state.saved_vix = vix_value
            st.session_state.saved_atr_trend = atr_pct_stored > 1.5
            st.session_state.saved_vix_trend = vix_value > 20
            
            sentiment_data_stored = getattr(st.session_state, 'current_sentiment', None)
            saved_sentiment_score = sentiment_data_stored.get('sentiment_score', 0) if sentiment_data_stored else 0
            st.session_state.saved_sentiment_score = saved_sentiment_score
            
            path_icon, path_text = calculate_path_expectation(
                st.session_state.saved_rsi,
                st.session_state.saved_sentiment_score,
                st.session_state.saved_atr_trend,
                st.session_state.saved_vix_trend
            )
            st.session_state.saved_path_icon = path_icon
            st.session_state.saved_path_text = path_text

    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")

    st.rerun()

# ========================
# TRADING RECOMMENDATION ENGINE (GLOBAL)
# ========================
def get_trading_recommendation(data):
    """
    Contextual trading recommendation based on market regime and stock conditions.
    Uses weighted factors with market regime adjustment.
    
    Returns: dict with recommendation details
    """
    
    # --- Layer 1: Market Regime ---
    vix = data.get('vix', 15)
    
    if vix < 12:
        regime = "low_vol"
        regime_modifier = 0.8  # Lower threshold for BUY
        regime_desc = "Low Volatility - Options cheap, but low premiums"
    elif vix < 20:
        regime = "optimal"
        regime_modifier = 1.0  # Normal threshold
        regime_desc = "Optimal - Normal volatility regime"
    elif vix < 25:
        regime = "elevated"
        regime_modifier = 1.3  # Higher threshold for BUY
        regime_desc = "Elevated Volatility - Options expensive, higher gamma potential"
    else:
        regime = "high_vol"
        regime_modifier = 1.6  # Much higher threshold
        regime_desc = "High Volatility - Options overpriced, caution advised"
    
    # --- Layer 2: Stock-Specific Technicals ---
    rsi = data.get('rsi', 50)
    ema_status = data.get('ema_status', 'neutral')
    bollinger_pos = data.get('bollinger_pos', 50)
    iv_hv_spread = data.get('iv_hv_spread', 0)
    beta = data.get('beta', 1.0)
    pcr = data.get('pcr', 0.5)
    market_verdict = data.get('market_verdict', '')
    market_confidence = data.get('market_confidence', 0)
    price = data.get('price', 100)
    atr_pct = data.get('atr_pct', 3.0)  # Default 3% if not available
    
    # --- Layer 3: Factor Analysis with Weights ---
    bullish_score = 0
    bearish_score = 0
    strong_bullish = 0
    strong_bearish = 0
    factor_details = []
    
    # 1. RSI (Weight: 0.25) - Most important
    if rsi < 30:  # Oversold - Strong BUY signal
        bullish_score += 2.5
        strong_bullish += 1
        factor_details.append("✅ RSI: OVERSOLD (Strong BUY)")
    elif rsi < 35:  # Nearing oversold - BUY signal
        bullish_score += 2.0
        strong_bullish += 1
        factor_details.append("✅ RSI: Nearing oversold (BUY)")
    elif rsi < 40:  # Approaching oversold - LEANING BUY
        bullish_score += 1.0
        factor_details.append("🟡 RSI: Approaching oversold")
    elif rsi < 45:  # Lower neutral - Mild bullish
        bullish_score += 0.5
        factor_details.append("🟡 RSI: Lower neutral")
    elif rsi > 70:  # Overbought - Strong SELL
        bearish_score += 2.5
        strong_bearish += 1
        factor_details.append("❌ RSI: OVERBOUGHT (Strong SELL)")
    elif rsi > 65:  # Nearing overbought
        bearish_score += 1.0
        factor_details.append("❌ RSI: Nearing overbought")
    elif rsi > 60:  # Upper neutral - Mild bearish
        bearish_score += 0.5
        factor_details.append("🟡 RSI: Upper neutral")
    else:
        factor_details.append("⚪ RSI: Neutral")
    
    # 2. IV/HV Spread (Weight: 0.20) - Options pricing
    if iv_hv_spread < -10:  # Cheap options
        bullish_score += 2.0
        strong_bullish += 1
        factor_details.append("✅ IV/HV: CHEAP options (Strong BUY)")
    elif iv_hv_spread < -5:  # Slightly cheap
        bullish_score += 1.0
        factor_details.append("🟡 IV/HV: Slightly cheap")
    elif iv_hv_spread > 10:  # Expensive options
        bearish_score += 1.5
        strong_bearish += 1
        factor_details.append("❌ IV/HV: EXPENSIVE options (Avoid)")
    elif iv_hv_spread > 5:  # Slightly expensive
        bearish_score += 0.5
        factor_details.append("🟡 IV/HV: Slightly expensive")
    else:
        factor_details.append("⚪ IV/HV: Fair value")
    
    # 3. EMA Status (Weight: 0.15)
    if ema_status in ["Bullish Cross", "bullish"]:
        bullish_score += 1.5
        strong_bullish += 1
        factor_details.append("✅ 8/20 EMA: Bullish (Uptrend)")
    elif ema_status in ["Bearish Separation", "bearish"]:
        bearish_score += 1.5
        strong_bearish += 1
        factor_details.append("❌ 8/20 EMA: Bearish (Downtrend)")
    else:
        factor_details.append("⚪ 8/20 EMA: Neutral")
    
    # 4. Market Verdict (Weight: 0.15)
    if "BUY" in market_verdict and market_confidence >= 65:
        bullish_score += 1.5
        strong_bullish += 1
        factor_details.append(f"✅ Market Verdict: BUY ({market_confidence:.0f}%)")
    elif "WAIT" in market_verdict:
        factor_details.append("🟡 Market Verdict: WAIT")
    elif "DROP" in market_verdict:
        bearish_score += 1.5
        strong_bearish += 1
        factor_details.append("❌ Market Verdict: DROP")
    
    # 5. Put/Call Ratio (Weight: 0.10)
    if pcr < 0.6:  # Very bullish
        bullish_score += 1.5
        strong_bullish += 1
        factor_details.append(f"✅ PCR: Very Bullish ({pcr:.2f})")
    elif pcr < 0.8:  # Bullish
        bullish_score += 0.5
        factor_details.append(f"🟡 PCR: Bullish ({pcr:.2f})")
    elif pcr > 1.2:  # Bearish
        bearish_score += 1.0
        strong_bearish += 1
        factor_details.append(f"❌ PCR: Bearish ({pcr:.2f})")
    else:
        factor_details.append(f"⚪ PCR: Neutral ({pcr:.2f})")
    
    # 6. Bollinger Position (Weight: 0.10)
    if bollinger_pos < 20:  # Near lower band - Support
        bullish_score += 1.0
        strong_bullish += 1
        factor_details.append(f"✅ Bollinger: Near lower band ({bollinger_pos:.0f}%)")
    elif bollinger_pos < 30:  # Lower half
        bullish_score += 0.5
        factor_details.append(f"🟡 Bollinger: Lower half ({bollinger_pos:.0f}%)")
    elif bollinger_pos > 80:  # Near upper band - Resistance
        bearish_score += 1.0
        strong_bearish += 1
        factor_details.append(f"❌ Bollinger: Near upper band ({bollinger_pos:.0f}%)")
    elif bollinger_pos > 70:  # Upper half
        bearish_score += 0.5
        factor_details.append(f"🟡 Bollinger: Upper half ({bollinger_pos:.0f}%)")
    else:
        factor_details.append(f"⚪ Bollinger: Middle range ({bollinger_pos:.0f}%)")
    
    # 7. Beta (Risk Penalty - NOT a factor, but a modifier)
    if beta > 1.5:
        risk_penalty = 1.0
        factor_details.append(f"🔴 Risk: HIGH BETA ({beta:.2f}) - Position size reduced")
    elif beta > 1.2:
        risk_penalty = 0.5
        factor_details.append(f"🟡 Risk: Elevated BETA ({beta:.2f}) - Position size reduced")
    else:
        risk_penalty = 0
        factor_details.append(f"🟢 Risk: Normal BETA ({beta:.2f})")
    
    # --- Calculate Net Score ---
    net_score = bullish_score - bearish_score
    
    # Apply risk penalty
    net_score = net_score - risk_penalty
    
    # --- Determine Thresholds Based on Regime ---
    if regime == "optimal":
        buy_threshold = 1.5
        strong_buy_threshold = 3.5
    elif regime == "low_vol":
        buy_threshold = 1.0  # Easier to BUY
        strong_buy_threshold = 3.0
    elif regime == "elevated":
        buy_threshold = 2.5  # Harder to BUY
        strong_buy_threshold = 4.5
    else:  # high_vol
        buy_threshold = 3.5  # Very hard to BUY
        strong_buy_threshold = 5.5
    
    # --- Entry Zone Calculation ---
    if rsi < 30:
        # Oversold - current price may be good entry
        entry_zone_low = price * 0.97
        entry_zone_high = price * 1.02
    elif rsi < 40:
        # Nearing oversold - wait for small pullback
        entry_zone_low = price * 0.92
        entry_zone_high = price * 0.97
    elif ema_status in ["Bearish Separation", "bearish"]:
        # Downtrend - wait for larger pullback
        entry_zone_low = price * 0.88
        entry_zone_high = price * 0.94
    else:
        entry_zone_low = price * 0.95
        entry_zone_high = price * 0.98
    
    # --- Stop Loss (wider for high volatility) ---
    if beta > 1.5:
        stop_pct = 0.18  # 18% stop for high beta
    elif beta > 1.2:
        stop_pct = 0.12  # 12% stop for elevated beta
    else:
        stop_pct = 0.08  # 8% stop for normal beta
    
    stop_loss = price * (1 - stop_pct)
    
    # ============================================================
    # DYNAMIC TARGET CALCULATION (NEW)
    # ============================================================
    
    # 1. ATR-based target (higher volatility = higher target)
    atr_target = atr_pct * 1.2  # 1.2x ATR for conservative target
    
    # 2. RSI-based target (oversold = higher bounce)
    if rsi < 30:
        rsi_adjustment = 2.0  # Strong bounce potential
    elif rsi < 40:
        rsi_adjustment = 1.5  # Moderate bounce
    elif rsi < 50:
        rsi_adjustment = 1.0  # Normal move
    else:
        rsi_adjustment = 0.8  # Limited upside from neutral/overbought
    
    # 3. IV/HV Spread adjustment
    if iv_hv_spread < -10:
        spread_boost = 1.5  # Cheap options - higher target
    elif iv_hv_spread < -5:
        spread_boost = 1.2  # Slightly cheap
    elif iv_hv_spread > 10:
        spread_boost = 0.8  # Expensive options - lower target
    else:
        spread_boost = 1.0  # Fair value
    
    # 4. Beta adjustment (high beta stocks move more)
    if beta > 1.5:
        beta_boost = 1.4  # Very volatile
    elif beta > 1.2:
        beta_boost = 1.2  # Moderately volatile
    elif beta < 0.8:
        beta_boost = 0.7  # Low volatility
    else:
        beta_boost = 1.0  # Market-like
    
    # 5. Market regime adjustment
    if vix < 12:
        regime_boost = 0.7  # Low vol = smaller moves
    elif vix < 20:
        regime_boost = 1.0  # Normal
    elif vix < 25:
        regime_boost = 1.2  # Elevated vol = bigger moves
    else:
        regime_boost = 0.9  # Very high vol = uncertainty
    
    # 6. EMA trend adjustment
    if ema_status in ["Bullish Cross", "bullish"]:
        trend_boost = 1.1  # Uptrend = higher target
    elif ema_status in ["Bearish Separation", "bearish"]:
        trend_boost = 0.9  # Downtrend = lower target
    else:
        trend_boost = 1.0  # Neutral
    
    # 7. Bollinger position adjustment
    if bollinger_pos < 20:
        bollinger_boost = 1.2  # Oversold bounce potential
    elif bollinger_pos > 80:
        bollinger_boost = 0.8  # Overbought resistance
    else:
        bollinger_boost = 1.0  # Middle range
    
    # --- Calculate Final Target Percentage ---
    # Start with ATR-based target
    base_target = atr_target
    
    # Apply all adjustments
    target_pct = base_target * rsi_adjustment * spread_boost * beta_boost * regime_boost * trend_boost * bollinger_boost
    
    # Ensure target is within reasonable bounds (3% to 25%)
    target_pct = max(3.0, min(25.0, target_pct))
    
    # Round to 1 decimal place
    target_pct = round(target_pct, 1)
    
    target_price = price * (1 + target_pct / 100)
    
    # --- Position Size ---
    if beta > 1.5:
        position_size = "Quarter (25%)"
        position_emoji = "🟡"
    elif beta > 1.2:
        position_size = "Half (50%)"
        position_emoji = "🟢"
    elif bearish_score >= 2.0:
        position_size = "Half (50%)"
        position_emoji = "🟢"
    else:
        position_size = "Full (100%)"
        position_emoji = "🟢"
    
    # --- Determine Recommendation ---
    # Check for STRONG BUY
    if strong_bullish >= 2 and net_score >= strong_buy_threshold:
        recommendation = "🟢 STRONG BUY"
        rec_color = "green"
        summary = f"Multiple strong bullish signals + {regime_desc} - EXCELLENT entry opportunity"
        confidence = min(95, 75 + (strong_bullish * 8) + (net_score * 3))
    
    # Check for BUY
    elif net_score >= buy_threshold and strong_bullish >= 1:
        recommendation = "🟢 BUY"
        rec_color = "green"
        summary = f"Favorable conditions in {regime_desc} - consider entry"
        confidence = min(90, 60 + (strong_bullish * 8) + (net_score * 2))
    
    # Check for LEANING BUY
    elif net_score >= buy_threshold - 0.5:
        recommendation = "🟡 LEANING BUY - WAIT"
        rec_color = "orange"
        summary = "Conditions are improving, wait for confirmation"
        confidence = min(75, 50 + (net_score * 8))
    
    # Check for NEUTRAL
    elif strong_bullish == 0 and strong_bearish == 0 and abs(net_score) < 1:
        recommendation = "🟡 NEUTRAL - MONITOR"
        rec_color = "orange"
        summary = "Mixed signals - monitor for clearer direction"
        confidence = 50
    
    # Check for AVOID
    elif strong_bearish >= 2 and net_score < -1:
        recommendation = "🔴 AVOID"
        rec_color = "red"
        summary = f"Strong bearish signals in {regime_desc} - stay away"
        confidence = min(80, 50 + (strong_bearish * 10))
    
    # Everything else - WAIT
    else:
        recommendation = "🟡 WAIT FOR BETTER ENTRY"
        rec_color = "orange"
        summary = f"Not enough bullish confirmation in {regime_desc} - wait for better entry"
        confidence = max(35, 45 + (net_score * 4))
    
    # --- Additional Context ---
    if "BUY" in recommendation:
        if beta > 1.5:
            summary += " - Caution: High beta, use smaller position"
        if iv_hv_spread < -10:
            summary += " - Options are historically cheap"
    elif "WAIT" in recommendation and iv_hv_spread < -10:
        summary += " - Options are cheap but waiting for technical confirmation"
    
    return {
        'recommendation': recommendation,
        'rec_color': rec_color,
        'summary': summary,
        'confidence': min(round(confidence), 95),
        'entry_zone_low': round(entry_zone_low, 2),
        'entry_zone_high': round(entry_zone_high, 2),
        'stop_loss': round(stop_loss, 2),
        'target_price': round(target_price, 2),
        'target_percent': target_pct,
        'position_size': position_size,
        'position_emoji': position_emoji,
        'bullish_score': bullish_score,
        'bearish_score': bearish_score,
        'net_score': net_score,
        'regime': regime_desc,
        'factor_details': factor_details,
        'strong_bullish': strong_bullish,
        'strong_bearish': strong_bearish
    }

# ========================
# MAIN CONTENT - REORDERED TABS (Step 6)
# ========================
t_dashboard, t_analysis, t_quant, t_tech, t_ai, t_edu = st.tabs([
    "📊 Dashboard",
    "🔬 Analysis", 
    "🔬 Quant Analytics",
    "📊 Technical",
    "🤖 AI Research",
    "📖 Strategy Guide"
])

# ========================
# DASHBOARD TAB (UPDATED with historical price chart)
# ========================
with t_dashboard:
    # --- SAFE DEFAULTS FOR ALL VARIABLES ---
    current_price = 0
    current_rsi = 50.0
    current_iv_pct = 0
    current_hv = 0
    current_iv_hv_spread = 0
    current_beta = 1.0
    current_pcr = 0.5
    current_ema_status = "Neutral"
    current_bollinger_pos = 50
    current_term_structure = "Neutral"
    bollinger_pos = 50
    market_verdict = "N/A"
    market_confidence = 0
    pc_ratio = 0.5
    weighted_verdict = "N/A"
    weighted_confidence = 0
    vix = 15
    
    # Only show dashboard if ticker data is available
    if st.session_state.price and st.session_state.expiries:
        S = st.session_state.price
        st.header(f"📊 Trading Dashboard - {st.session_state.stock_name} ({st.session_state.current_ticker})")
        
        col_p, col_t = st.columns(2)
        col_p.metric("Current Underlying Price", f"${S:.2f}")
        col_t.metric("20-Day Baseline Trend", st.session_state.trend, f"{st.session_state.pct_change:.1f}%")
        
        st.divider()
        
        vix = get_vix()
        hist_data = st.session_state.hist_data
        
        try:
            atr_val, atr_pct = calculate_atr(hist_data)
            rsi_val = calculate_rsi(hist_data)
            hv_val = calculate_hv(hist_data)
        except:
            atr_val, atr_pct = 0.0, 0.0
            rsi_val = 50.0
            hv_val = 0.0

        sentiment_data = getattr(st.session_state, 'current_sentiment', None)
        if sentiment_data:
            sentiment_score = sentiment_data.get('sentiment_score', 0)
        else:
            sentiment_score = 0
            st.caption("💡 Run AI Research in the AI tab for sentiment analysis")
        
        current_expiry = st.session_state.last_selected_expiry if st.session_state.last_selected_expiry else (st.session_state.expiries[0] if st.session_state.expiries else None)
        
        if current_expiry:
            calls_df, _ = get_cached_option_chain(st.session_state.current_ticker, current_expiry)
            if calls_df is not None and not calls_df.empty:
                atm_idx = (calls_df['strike'] - S).abs().argsort()[:1]
                current_iv = calls_df.iloc[atm_idx]['impliedVolatility'].iloc[0] if not calls_df.empty else 0.35
            else:
                current_iv = 0.35
        else:
            current_iv = 0.35
        
        iv_hv_spread, _ = calculate_iv_hv_spread(current_iv, hv_val)
        
        earnings_date = get_earnings_date(st.session_state.current_ticker)
        if earnings_date:
            days_to_earnings = (earnings_date - datetime.now().date()).days
        else:
            days_to_earnings = None
        
        beta, _ = calculate_beta(st.session_state.current_ticker)
        try:
            if current_expiry:
                calls_df, puts_df = get_cached_option_chain(st.session_state.current_ticker, current_expiry)
                if calls_df is not None and not calls_df.empty:
                    skew, _ = calculate_skew(calls_df, puts_df if puts_df is not None else pd.DataFrame(), S)
                else:
                    skew = 0
            else:
                skew = 0
        except:
            skew = 0
        
        # --- GET TECHNICALS (DEFINES 'curr' and 'ema_status') ---
        curr, prev = get_technicals(hist_data)
        ema_status = "Bullish Cross" if curr['ema8'] > curr['ema20'] else "Bearish Separation"
        term_structure = "Neutral"  # Default, or get from calculate_term_structure()
        
        # --- GET PUT/CALL RATIO ---
        pc_ratio = 0.5
        try:
            pc_ratio, pc_sentiment, pc_interpretation, call_vol, put_vol = calculate_put_call_ratio(
                st.session_state.current_ticker, current_expiry
            )
            if pc_ratio is None:
                pc_ratio = 0.5
        except:
            pc_ratio = 0.5

        # ============================================================
        # MARKET CONTEXT SECTION
        # ============================================================
        st.divider()
        st.subheader("🌍 Market Context")
        
        # Get macro data
        macro_data = get_macro_data()
        sector, industry = get_sector_for_ticker(st.session_state.current_ticker)
        
        col_m1, col_m2, col_m3 = st.columns(3)
        
        with col_m1:
            st.metric("VIX (Fear Index)", f"{macro_data.get('vix', 15):.1f}", 
                     delta=macro_data.get('regime', 'Neutral'))
            
            if macro_data.get('spy_change_5d') is not None:
                spy_color = "normal" if macro_data['spy_change_5d'] >= 0 else "inverse"
                st.metric("SPY (5-Day)", f"${macro_data.get('spy_price', 0):.2f}", 
                         delta=f"{macro_data.get('spy_change_5d', 0):+.1f}%", 
                         delta_color=spy_color)
        
        with col_m2:
            st.metric("Sector", sector)
            st.metric("Industry", industry)
            
            # Sector performance - show relevant sectors
            if macro_data.get('sector_data'):
                for sector_name, data in macro_data['sector_data'].items():
                    if sector_name in ["Semiconductors", "Technology", "Quantum/Computing"]:
                        delta_color = "normal" if data['change_5d'] >= 0 else "inverse"
                        st.metric(f"{sector_name} (5D)", f"{data['change_5d']:+.1f}%", 
                                 delta_color=delta_color)
        
        with col_m3:
            # Earnings
            if earnings_date:
                if days_to_earnings < 0:
                    st.metric("📅 Next Earnings", "Recently passed")
                elif days_to_earnings < 7:
                    st.warning(f"⚠️ Earnings in {days_to_earnings} days")
                    st.metric("📅 Next Earnings", f"In {days_to_earnings} days", 
                             delta="HIGH RISK")
                else:
                    st.metric("📅 Next Earnings", f"In {days_to_earnings} days", 
                             delta=f"{earnings_date.strftime('%b %d')}")
            else:
                st.metric("📅 Next Earnings", "N/A")
            
            # Quick market events button
            if st.button("📰 Market Events", use_container_width=True):
                events = get_market_events_today()
                if events:
                    with st.expander("📰 Top Market Headlines", expanded=True):
                        for article in events[:3]:
                            headline = article.get('headline', '')
                            if len(headline) > 80:
                                headline = headline[:80] + "..."
                            st.write(f"• {headline}")
                            st.caption(f"  Source: {article.get('source', 'Unknown')}")
                else:
                    st.info("No market events found")

        # --- NOW SHOW MARKET VERDICT ---
        st.divider()
        st.subheader("🎯 Market Verdict")
        
        strict_verdict, strict_passed, strict_confidence = calculate_strict_verdict(
            vix, rsi_val, iv_hv_spread, days_to_earnings, sentiment_score
        )
        weighted_verdict, weighted_confidence = calculate_weighted_verdict(
            vix, rsi_val, iv_hv_spread, sentiment_score, skew, beta
        )
        
        col_v1, col_v2 = st.columns(2)
        with col_v1:
            st.metric("STRICT (4/5 Conditions)", strict_verdict, delta=f"{strict_passed}/5 conditions met")
            st.caption(f"Confidence: {strict_confidence:.0f}%")
        with col_v2:
            st.metric("WEIGHTED (PhD Model)", weighted_verdict, delta=f"{weighted_confidence:.0f}% confidence")
            st.caption(f"Factors: VIX 20%, RSI 15%, IV/HV 20%, Sentiment 20%, Skew 15%, Beta 10%")

        # ============================================================
        # HISTORICAL PRICE CHART (MOVED FROM TECHNICAL TAB)
        # ============================================================
        st.divider()
        st.subheader("📈 Price & Technical Chart")
        
        # Calculate EMAs and Bollinger Bands
        hist_chart = hist_data.copy()
        hist_chart['ema8'] = hist_chart['Close'].ewm(span=8, adjust=False).mean()
        hist_chart['ema20'] = hist_chart['Close'].ewm(span=20, adjust=False).mean()
        hist_chart['sma20'] = hist_chart['Close'].rolling(window=20).mean()
        hist_chart['std20'] = hist_chart['Close'].rolling(window=20).std()
        hist_chart['upper'] = hist_chart['sma20'] + (hist_chart['std20'] * 2)
        hist_chart['lower'] = hist_chart['sma20'] - (hist_chart['std20'] * 2)
        
        # Display the chart
        st.line_chart(hist_chart[['Close', 'ema8', 'ema20', 'upper', 'lower']])
        st.caption("📊 Close price with EMA8, EMA20, and Bollinger Bands (Upper/Lower)")

        # ============================================================
        # TRADING DECISION FRAMEWORK (Enhanced Robust Logic)
        # ============================================================
        st.divider()
        st.subheader("🎯 Trading Decision Framework")
        
        # --- ALL VARIABLES ARE NOW DEFINED ---
        current_price = S
        current_rsi = rsi_val
        current_iv_pct = current_iv * 100 if current_iv else 0
        current_hv = hv_val
        current_iv_hv_spread = iv_hv_spread
        current_beta = beta
        current_pcr = pc_ratio
        current_ema_status = ema_status
        current_bollinger_pos = ((S - curr['lower']) / (curr['upper'] - curr['lower'])) * 100 if 'lower' in curr and 'upper' in curr else 50
        current_term_structure = term_structure if term_structure else "Neutral"
        bollinger_pos = ((S - curr['lower']) / (curr['upper'] - curr['lower'])) * 100 if 'lower' in curr and 'upper' in curr else 50
        market_verdict = weighted_verdict
        market_confidence = weighted_confidence
        
        # Get ATM option info if available
        atm_strike = None
        atm_entry = None
        atm_target = None
        atm_stop = None
        atm_delta = None
        
        if current_expiry:
            calls_df, _ = get_cached_option_chain(st.session_state.current_ticker, current_expiry)
            if calls_df is not None and not calls_df.empty:
                atm_idx = (calls_df['strike'] - S).abs().argsort()[:1]
                atm_row = calls_df.iloc[atm_idx].iloc[0]
                atm_strike = atm_row['strike']
                atm_entry = (atm_row['bid'] + atm_row['ask']) / 2 if atm_row['bid'] > 0 else atm_row['lastPrice']
                atm_target = atm_entry * (1 + profit_target_pct / 100)
                atm_stop = atm_entry * (1 - stop_loss_pct / 100)
                atm_delta = atm_row.get('delta', None)
        
        # --- Prepare data for recommendation ---
        decision_data = {
            'price': S,
            'rsi': rsi_val,
            'iv_hv_spread': iv_hv_spread,
            'pcr': pc_ratio if pc_ratio else 0.5,
            'beta': beta,
            'ema_status': ema_status,
            'bollinger_pos': bollinger_pos,
            'term_structure': term_structure if term_structure else "Neutral",
            'market_verdict': weighted_verdict,
            'market_confidence': weighted_confidence,
            'atm_strike': atm_strike,
            'atm_entry': atm_entry,
            'atm_target': atm_target,
            'atm_stop': atm_stop,
            'vix': vix,
            'atr_pct': atr_pct  # <-- ADD THIS LINE
        }
        
        # --- Generate recommendation ---
        decision = get_trading_recommendation(decision_data)
        
        # --- Display Recommendation ---
        # Color mapping
        color_map = {
            'green': '#28a745',
            'orange': '#ff9800',
            'red': '#dc3545'
        }
        bg_color_map = {
            'green': 'rgba(40, 167, 69, 0.15)',
            'orange': 'rgba(255, 152, 0, 0.15)',
            'red': 'rgba(220, 53, 69, 0.15)'
        }
        
        rec_color = decision['rec_color']
        
        st.markdown(f"""
        <div style="border: 2px solid {color_map[rec_color]}; padding: 20px; border-radius: 10px; background-color: {bg_color_map[rec_color]}; margin-bottom: 15px;">
            <h3 style="margin:0; color:{color_map[rec_color]};">{decision['recommendation']}</h3>
            <p style="margin:5px 0 0 0; font-size:1.1rem;">{decision['summary']}</p>
            <p style="margin:2px 0 0 0; font-size:0.85rem; opacity:0.7;">Confidence: {decision['confidence']}% | Regime: {decision['regime']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # --- Quick Summary Cards ---
        col_q1, col_q2, col_q3, col_q4 = st.columns(4)
        with col_q1:
            st.metric("📊 Market Environment", f"{market_verdict}", delta=f"{market_confidence:.0f}% confidence")
        with col_q2:
            st.metric("🎯 Entry Zone", f"${decision['entry_zone_low']:.2f} - ${decision['entry_zone_high']:.2f}")
        with col_q3:
            st.metric("🛑 Stop Loss", f"${decision['stop_loss']:.2f}")
        with col_q4:
            st.metric("🎯 Target", f"${decision['target_price']:.2f}", 
                     delta=f"{decision['target_percent']:.1f}% upside")

        
        # --- Expandable: Why This Recommendation ---
        with st.expander("📊 Why This Recommendation (Click to expand)", expanded=False):
            st.markdown("**Decision Drivers**")
            
            # Show factor details
            if decision.get('factor_details'):
                for detail in decision['factor_details']:
                    st.write(f"- {detail}")
            
            st.divider()
            
            # Show score breakdown
            col_bull, col_bear, col_net = st.columns(3)
            with col_bull:
                st.metric("Bullish Score", f"{decision['bullish_score']:.1f}", 
                         delta=f"Strong: {decision['strong_bullish']}")
            with col_bear:
                st.metric("Bearish Score", f"{decision['bearish_score']:.1f}", 
                         delta=f"Strong: {decision['strong_bearish']}")
            with col_net:
                net_color = "normal" if decision['net_score'] >= 0 else "inverse"
                st.metric("Net Score", f"{decision['net_score']:+.1f}", 
                         delta="BUY threshold: 1.5", delta_color=net_color)
            
            # Progress bar for net score
            max_score = max(abs(decision['net_score']), 5)
            progress_pct = (decision['net_score'] + max_score) / (2 * max_score)
            st.progress(min(progress_pct, 1.0))
            st.caption(f"Regime: {decision['regime']} | Confidence: {decision['confidence']}%")
        
        # --- Expandable: Execution Plan ---
        with st.expander("📋 Execution Plan (Click to expand)", expanded=False):
            st.markdown("#### 🎯 Recommended Trade Setup")
            
            col_e1, col_e2, col_e3 = st.columns(3)
            with col_e1:
                st.metric("Entry Zone", f"${decision['entry_zone_low']:.2f} - ${decision['entry_zone_high']:.2f}")
                st.caption(f"Current: ${S:.2f} ({((decision['entry_zone_high']/S)-1)*100:.1f}% below current)")
            with col_e2:
                st.metric("Stop Loss", f"${decision['stop_loss']:.2f}")
                st.caption(f"Risk: ${S - decision['stop_loss']:.2f} ({((decision['stop_loss']/S)-1)*100:.1f}%)")
            with col_e3:
                st.metric("Target", f"${decision['target_price']:.2f}")
                st.caption(f"Reward: ${decision['target_price'] - S:.2f} ({((decision['target_price']/S)-1)*100:.1f}%)")
            
            st.divider()
            st.markdown(f"#### 📊 Position Size: {decision['position_emoji']} {decision['position_size']}")
            
            # Risk warning for high beta
            if beta > 1.5:
                st.warning(f"⚠️ High beta stock ({beta:.2f}) - Consider smaller position size and wider stop loss.")
            elif beta > 1.2:
                st.info(f"📊 Elevated beta ({beta:.2f}) - Normal monitoring recommended.")
            
            if atm_entry and atm_strike:
                st.markdown("#### 📈 Option Contract Recommendation")
                col_o1, col_o2, col_o3 = st.columns(3)
                with col_o1:
                    st.metric("Strike", f"${atm_strike:.2f} Call")
                with col_o2:
                    st.metric("Entry Price", f"${atm_entry:.2f}")
                with col_o3:
                    st.metric("Stop Loss", f"${atm_stop:.2f}")
        
        # ============================================================
        # ENHANCED GROQ AI INSIGHT WITH MACRO CONTEXT
        # ============================================================
        with st.expander("🤖 AI Insight (Powered by Groq)", expanded=False):
            with st.spinner("Gathering market context and generating AI insights..."):
                
                # Get sector and macro data for enhanced prompt
                sector, industry = get_sector_for_ticker(st.session_state.current_ticker)
                
                # Build macro context
                macro_context = f"""
MACRO CONTEXT:
- VIX: {macro_data.get('vix', 15):.1f} ({macro_data.get('regime', 'Neutral')})
- SPY 5-Day Change: {macro_data.get('spy_change_5d', 0):+.1f}%
- Market Regime: {macro_data.get('regime_description', 'Normal')}
- Earnings: {f"Next earnings in {days_to_earnings} days" if earnings_date and days_to_earnings > 0 else "No upcoming earnings" if not earnings_date else f"Earnings {abs(days_to_earnings)} days ago"}
"""

                # Build sector performance context
                sector_performance = ""
                if macro_data.get('sector_data'):
                    sector_performance = "SECTOR PERFORMANCE (5-Day):\n"
                    for sector_name, data in macro_data['sector_data'].items():
                        sector_performance += f"- {sector_name}: {data['change_5d']:+.1f}%\n"

                # Build market events context
                events_context = ""
                market_events = get_market_events_today()
                if market_events:
                    events_context = "TOP MARKET EVENTS TODAY:\n"
                    for i, article in enumerate(market_events[:3]):
                        headline = article.get('headline', '')[:100]
                        source = article.get('source', 'Unknown')
                        events_context += f"{i+1}. {headline} ({source})\n"

                # Get stock-specific news
                stock_news_context = ""
                stock_news = get_stock_news(st.session_state.current_ticker, limit=5)
                if stock_news:
                    stock_news_context = f"\n{st.session_state.current_ticker}-SPECIFIC NEWS (Last 7 Days):\n"
                    for i, article in enumerate(stock_news):
                        headline = article.get('headline', 'No title')
                        summary = article.get('summary', '')[:150]
                        source = article.get('source', 'Unknown')
                        date = article.get('datetime', '')
                        stock_news_context += f"{i+1}. {headline}\n"
                        stock_news_context += f"   Summary: {summary}\n"
                        stock_news_context += f"   Source: {source} | Date: {date}\n\n"
                else:
                    stock_news_context = f"\nNo recent news found for {st.session_state.current_ticker} in the last 7 days.\n"
                
                # Build the enhanced AI prompt
                ai_prompt = f"""
You are a professional options trader and quantitative analyst. Based on the following comprehensive data for {st.session_state.current_ticker}, provide a concise trading insight.

TICKER: {st.session_state.current_ticker}
SECTOR: {sector} | INDUSTRY: {industry}

TECHNICAL METRICS:
- Price: ${S:.2f}
- RSI (14d): {rsi_val:.1f}
- 8/20 EMA Status: {ema_status}
- Bollinger Position: {bollinger_pos:.0f}% of band
- Trend: {st.session_state.trend}

VOLATILITY METRICS:
- Implied Volatility (IV): {current_iv_pct:.1f}%
- Historical Volatility (HV): {current_hv:.1f}%
- IV/HV Spread: {iv_hv_spread:.1f}%
- Beta (vs SPY): {beta:.2f}
- Term Structure: {term_structure if term_structure else "Neutral"}

SENTIMENT METRICS:
- Put/Call Ratio: {pc_ratio if pc_ratio else 0.5:.2f}
- Market Verdict: {weighted_verdict} ({weighted_confidence:.0f}% confidence)

{macro_context}
{sector_performance}
{stock_news_context}
{events_context}

MY RECOMMENDATION: {decision['recommendation']}
- Confidence: {decision['confidence']}%
- Entry Zone: ${decision['entry_zone_low']:.2f} - ${decision['entry_zone_high']:.2f}
- Stop Loss: ${decision['stop_loss']:.2f}
- Target: ${decision['target_price']:.2f}
- Position Size: {decision['position_size']}

Provide a 3-4 sentence insight that:
1. Acknowledges the macro environment and how it affects this trade
2. Mentions the sector context and any relevant sector trends
3. INCORPORATES ANY RECENT STOCK-SPECIFIC NEWS OR CATALYSTS (this is critical!)
4. Explains the key reason for the recommendation
5. Gives a clear, actionable takeaway

Keep it professional, concise, and actionable. Focus on the intersection of macro trends, sector performance, stock catalysts, and this specific stock.
"""
                
                try:
                    # Get Groq API key
                    groq_api_key = st.secrets.get("GROQ_API_KEY")
                    if groq_api_key:
                        client = Groq(api_key=groq_api_key)
                        response = call_groq_with_retry(client, ai_prompt)
                        
                        if response:
                            st.markdown(f"**🧠 AI Insight:**\n\n{response}")
                        else:
                            st.info("AI insight temporarily unavailable. Please try again later.")
                    else:
                        st.info("Groq API key not configured. AI insights are unavailable.")
                except Exception as e:
                    st.info("AI insight temporarily unavailable. Please try again later.")

        
        if 'saved_path_icon' in st.session_state and st.session_state.saved_path_icon:
            path_icon = st.session_state.saved_path_icon
            path_text = st.session_state.saved_path_text
        else:
            path_icon, path_text = calculate_path_expectation(rsi_val, sentiment_score, atr_pct > 1.5, vix > 20)
        
        st.info(f"{path_icon} **Path Expectation:** {path_text}")
        
        is_pullback_scenario = (path_icon == "📉📈 Pullback then rise")
        
        if is_pullback_scenario:
            st.divider()
            st.subheader("🎯 Pullback Entry Estimates")
            st.caption("Based on ATR, RSI, Bollinger Bands, and Max Pain analysis")
            
            curr_indicators, _ = get_technicals(hist_data)
            
            max_pain_value = None
            if current_expiry:
                try:
                    calls_df, puts_df = get_cached_option_chain(st.session_state.current_ticker, current_expiry)
                    if calls_df is not None and not calls_df.empty:
                        strikes = sorted(calls_df['strike'].unique())
                        max_pain_value = calculate_max_pain(calls_df, puts_df, strikes)
                except:
                    pass
            
            pullback_data = get_pullback_entry_recommendation(
                S, hist_data, max_pain_value, rsi_val, sentiment_score, atr_pct
            )
            
            col_e1, col_e2, col_e3 = st.columns(3)
            
            with col_e1:
                st.metric(
                    "🎯 Optimal Entry Price", 
                    f"${pullback_data['estimated_entry']:.2f}",
                    delta=f"-{pullback_data['estimated_pullback_pct']:.1f}% from current"
                )
                st.caption(f"⏰ Expected in ~{abs(pullback_data['estimated_days_to_pullback'])} trading days")
                st.caption(f"📊 Confidence: {pullback_data.get('confidence', 'Medium')}")
            
            with col_e2:
                st.write("**📍 Entry Zones**")
                st.write(f"🟢 **Optimal:** ${pullback_data['entry_range']['optimal']:.2f}")
                st.write(f"🟡 **Conservative:** ${pullback_data['entry_range']['conservative']:.2f}")
                st.write(f"🔴 **Aggressive:** ${pullback_data['entry_range']['aggressive']:.2f}")
            
            with col_e3:
                st.write("**🛡️ Support Levels**")
                st.write(f"1st Support (SMA20): ${pullback_data['support_levels']['first_support']:.2f}")
                st.write(f"2nd Support (Lower BB): ${pullback_data['support_levels']['second_support']:.2f}")
                if pullback_data['support_levels']['deep_support']:
                    st.write(f"3rd Support (Max Pain): ${pullback_data['support_levels']['deep_support']:.2f}")
            
            st.caption(f"💭 Sentiment Context: {pullback_data['sentiment_text']}")
            
            st.divider()
            st.subheader("📋 Action Plan")
            
            current_price = pullback_data['current_price']
            optimal_entry = pullback_data['estimated_entry']
            conservative_entry = pullback_data['entry_range']['conservative']
            
            if current_price <= conservative_entry * 1.02:
                st.success("🔔 **PRICE APPROACHING ENTRY ZONE!** Consider placing limit orders NOW.")
                st.info(f"📝 **Recommended Action:** Set limit buy at ${optimal_entry:.2f} (optimal) or market buy if price drops below ${conservative_entry:.2f}")
            elif current_price <= optimal_entry * 1.05:
                st.info("⏳ **Price moving toward target entry.** Getting ready to buy.")
                st.info(f"📝 **Recommended Action:** Set limit order at ${optimal_entry:.2f} and wait for fill")
            else:
                st.warning(f"⏰ **Current price ${current_price:.2f} is above optimal entry.**")
                st.info(f"📝 **Recommended Action:** Wait for pullback to ${optimal_entry:.2f}. Place limit order and be patient.")
            
            col_o1, col_o2 = st.columns(2)
            with col_o1:
                st.write("**📊 Recommended Option Strategy**")
                st.write("- Wait for pullback to complete")
                st.write(f"- Enter when price hits ${optimal_entry:.2f} - ${conservative_entry:.2f}")
                st.write("- Use 30-45 DTE calls for gamma ramp")
            with col_o2:
                st.write("**⚠️ Risk Management**")
                st.write(f"- Stop loss: Below ${pullback_data['support_levels']['second_support']:.2f}")
                st.write("- Position size: 50-70% of normal (save dry powder)")
                st.write("- Add more if price reaches aggressive entry")
        
        st.divider()
        
        st.subheader("📋 Strategy Recommendations")
        st.caption("💡 Click any row to view detailed analysis in the corresponding strategy tab")
        
        if current_expiry:
            strategy_configs = [
                {"name": "🛡️ Conservative", "delta_min": 0.50, "delta_max": 0.60, "tab_index": 6, "tooltip": "Higher probability (50-60%), lower return, slower time to target"},
                {"name": "⚡ Aggressive", "delta_min": 0.40, "delta_max": 0.49, "tab_index": 7, "tooltip": "Medium probability (40-49%), medium return, balanced risk/reward"},
                {"name": "🎰 Speculative", "delta_min": 0.30, "delta_max": 0.39, "tab_index": 8, "tooltip": "Lower probability (30-39%), highest return potential, fastest time to target"}
            ]
            
            table_data = []
            for config in strategy_configs:
                contract = get_strategy_for_expiry(
                    st.session_state.current_ticker, current_expiry,
                    config["delta_min"], config["delta_max"],
                    profit_target_pct, stop_loss_pct, S
                )
                
                if contract:
                    entry = contract['mid']
                    target = entry * (1 + profit_target_pct / 100)
                    stop = entry * (1 - stop_loss_pct / 100)
                    score = contract['cts']
                    
                    if score >= 65:
                        action = "✅ BUY"
                    elif score >= 55:
                        action = "⏳ WAIT"
                    else:
                        action = "❌ DROP"
                    
                    table_data.append({
                        "Strategy": config["name"],
                        "tooltip": config["tooltip"],
                        "Strike": f"${contract['strike']:.2f} Call",
                        "Entry": f"${entry:.2f}",
                        "Target": f"${target:.2f}",
                        "Stop": f"${stop:.2f}",
                        "Score": f"{score}/100",
                        "Action": action,
                        "tab_index": config["tab_index"]
                    })
            
            if table_data:
                col_headers = st.columns([1.5, 1.2, 0.8, 0.8, 0.8, 0.8, 0.8])
                headers = ["Strategy", "Strike", "Entry", "Target", "Stop", "Score", "Action"]
                for col, header in zip(col_headers, headers):
                    col.markdown(f"**{header}**")
                
                for row in table_data:
                    cols = st.columns([1.5, 1.2, 0.8, 0.8, 0.8, 0.8, 0.8])
                    with cols[0]:
                        st.markdown(f"<span title='{row['tooltip']}'>{row['Strategy']}</span>", unsafe_allow_html=True)
                    with cols[1]:
                        st.write(row["Strike"])
                    with cols[2]:
                        st.write(row["Entry"])
                    with cols[3]:
                        st.write(row["Target"])
                    with cols[4]:
                        st.write(row["Stop"])
                    with cols[5]:
                        st.write(row["Score"])
                    with cols[6]:
                        if st.button(row["Action"], key=f"dashboard_strategy_{row['tab_index']}", use_container_width=True):
                            st.session_state.active_tab = row["tab_index"]
                            st.rerun()
            else:
                st.warning(f"No strategy recommendations available for expiry {current_expiry}")
        else:
            st.warning("No expiries available. Analyze a ticker first.")
        
        st.divider()
        
        st.subheader("📊 Key Market Metrics")
        col_k1, col_k2, col_k3, col_k4 = st.columns(4)
        with col_k1:
            regime_text, _ = get_market_regime(vix)
            st.metric("VIX (Fear Index)", f"{vix:.1f}", delta=regime_text)
        with col_k2:
            st.metric("ATR (14d)", f"${atr_val:.2f}", delta=f"{atr_pct:.1f}% of price")
        with col_k3:
            rsi_status = "Overbought" if rsi_val > 70 else ("Oversold" if rsi_val < 30 else "Neutral")
            st.metric("RSI (14d)", f"{rsi_val:.1f}", delta=rsi_status)
        with col_k4:
            st.metric("Beta (vs SPY)", f"{beta:.2f}")
        
        st.divider()
        
        col_k5, col_k6, col_k7, col_k8 = st.columns(4)
        with col_k5:
            spread_status = "Fair" if -10 <= iv_hv_spread <= 10 else ("Expensive" if iv_hv_spread > 10 else "Cheap")
            st.metric("IV/HV Spread", f"{iv_hv_spread:+.1f}%", delta=spread_status)
        with col_k6:
            skew_status = "Bullish" if skew > 0 else ("Bearish" if skew < 0 else "Neutral")
            st.metric("Skew", f"{skew:+.1f}%", delta=skew_status)
        with col_k7:
            if sentiment_data:
                sentiment_color = "Bullish" if sentiment_score > 0.3 else ("Bearish" if sentiment_score < -0.3 else "Neutral")
                st.metric("AI Sentiment", f"{sentiment_score:+.2f}", delta=sentiment_color)
            else:
                st.metric("AI Sentiment", "N/A", delta="Run AI Research")
        with col_k8:
            if days_to_earnings:
                if days_to_earnings < 0:
                    st.metric("Next Earnings", "Recently passed")
                elif days_to_earnings < 7:
                    st.warning(f"⚠️ Earnings in {days_to_earnings} days")
                    st.metric("Earnings", f"In {days_to_earnings} days", delta="HIGH RISK")
                else:
                    st.metric("Next Earnings", f"In {days_to_earnings} days")
            else:
                st.metric("Next Earnings", "N/A")
        
        st.divider()
        
        st.subheader("🤖 AI Market Sentiment")
        sentiment_data = getattr(st.session_state, 'current_sentiment', None)
        if sentiment_data and sentiment_data.get('sentiment_score', 0) != 0:
            sentiment_score = sentiment_data.get('sentiment_score', 0)
            sentiment_label = sentiment_data.get('sentiment_label', 'Neutral')
            
            col_s1, col_s2, col_s3 = st.columns(3)
            with col_s1:
                if sentiment_score > 0.3:
                    st.success(f"**Sentiment: {sentiment_label}**")
                    st.metric("Sentiment Score", f"+{sentiment_score:.2f}", delta="Bullish")
                elif sentiment_score < -0.3:
                    st.error(f"**Sentiment: {sentiment_label}**")
                    st.metric("Sentiment Score", f"{sentiment_score:.2f}", delta="Bearish")
                else:
                    st.info(f"**Sentiment: {sentiment_label}**")
                    st.metric("Sentiment Score", f"{sentiment_score:.2f}", delta="Neutral")
            with col_s2:
                risk_adj = sentiment_data.get('risk_adjustment', 0)
                if risk_adj > 0:
                    st.warning(f"Risk Adjustment: +{risk_adj}")
                elif risk_adj < 0:
                    st.info(f"Risk Adjustment: {risk_adj}")
                else:
                    st.caption("No risk adjustment")
            with col_s3:
                catalyst = sentiment_data.get('catalyst')
                if catalyst:
                    st.info(f"📅 **Catalyst:** {catalyst[:50]}..." if len(catalyst) > 50 else f"📅 **Catalyst:** {catalyst}")
                else:
                    st.caption("No immediate catalysts detected")
        else:
            st.info("Run AI Research in the AI tab to get sentiment analysis")
            if st.button("🔍 Analyze Sentiment Now"):
                st.session_state.ai_brief = ""
                st.rerun()
    else:
        st.info("👈 Analyze a ticker to see the Dashboard")

# ========================
# ANALYSIS TAB (NEW - Replaces old main view)
# ========================
with t_analysis:
    if st.session_state.price and st.session_state.expiries:
        S = st.session_state.price
        st.header(f"🔬 Analysis - {st.session_state.stock_name} ({st.session_state.current_ticker})")
        
        col_p, col_t = st.columns(2)
        col_p.metric("Current Underlying Price", f"${S:.2f}")
        col_t.metric("20-Day Baseline Trend", st.session_state.trend, f"{st.session_state.pct_change:.1f}%")
        
        st.divider()
        
        # Show the strategy tabs within Analysis
        st.subheader("📊 Strategy Analysis")
        
        # Create sub-tabs for strategies within Analysis
        sub_tabs = st.tabs(["🛡️ Conservative", "⚡ Aggressive", "🎰 Speculative"])
        
        # Reuse the existing process_tier_strategy function
        def process_tier_strategy_in_analysis(tab_component, delta_min, delta_max, tier_label, tech_score):
            with tab_component:
                current_expiry = st.session_state.last_selected_expiry if st.session_state.last_selected_expiry else (st.session_state.expiries[0] if st.session_state.expiries else None)
                if not current_expiry:
                    st.warning("No expiry selected. Please select one in the sidebar.")
                    return
                
                days_to_expiry = (pd.to_datetime(current_expiry).date() - datetime.now().date()).days
                T_years = max(days_to_expiry, 1) / 365
                
                calls_df, puts_df = get_cached_option_chain(st.session_state.current_ticker, current_expiry)
                if calls_df is None:
                    st.error("Failed to fetch option chain")
                    return
                
                all_available_contracts = []
                tier_contracts = []
                
                for index, row in calls_df.iterrows():
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
                    
                    try:
                        if puts_df is not None and not puts_df.empty:
                            skew_val, _ = calculate_skew(calls_df, puts_df, S, row['strike'])
                            ev = apply_skew_penalty(ev, skew_val / 100)
                    except:
                        pass
                    
                    gt_ratio = calculate_gamma_theta_ratio(g, t)
                    cts = calculate_enhanced_cts(d, p_touch, gt_ratio, tech_score)
                    
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
                        'liquidity_status': liquidity_status, 'liquidity_warning': liquidity_warning,
                        'gamma_theta_ratio': gt_ratio
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
                
                if best_contract.get('gamma_theta_ratio', 0) > 1.0:
                    st.caption(f"⚡ Gamma/Theta Ratio: {best_contract['gamma_theta_ratio']:.2f} (Excellent acceleration)")
                
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
                    key=f"compare_analysis_{tier_label}_{current_expiry}"
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
                        st.write(f"- Gamma/Theta Ratio: `{selected_contract.get('gamma_theta_ratio', 0):.2f}`")
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
        
        # Call the analysis function for each strategy
        process_tier_strategy_in_analysis(sub_tabs[0], 0.50, 0.60, "Conservative", st.session_state.get('tech_score', 0))
        process_tier_strategy_in_analysis(sub_tabs[1], 0.40, 0.49, "Aggressive", st.session_state.get('tech_score', 0))
        process_tier_strategy_in_analysis(sub_tabs[2], 0.30, 0.39, "Speculative", st.session_state.get('tech_score', 0))
        
    else:
        st.info("👈 Enter a ticker in the sidebar and click 'Analyze Options Structure' to see the analysis.")

# ========================
# QUANT ANALYTICS TAB
# ========================
with t_quant:
    if st.session_state.price and st.session_state.expiries:
        S = st.session_state.price
        st.header("🔬 Quantitative Analytics")
        st.markdown("Advanced metrics for professional traders")
        
        current_expiry = st.session_state.expiries[0] if st.session_state.expiries else None
        
        if current_expiry:
            calls_df, puts_df = get_cached_option_chain(st.session_state.current_ticker, current_expiry)
            
            if calls_df is not None and not calls_df.empty:
                
                st.subheader("📊 Sentiment Indicators")
                
                pc_ratio, pc_sentiment, pc_interpretation, call_vol, put_vol = calculate_put_call_ratio(
                    st.session_state.current_ticker, current_expiry
                )
                
                col_pc1, col_pc2, col_pc3 = st.columns(3)
                with col_pc1:
                    if pc_ratio:
                        st.metric("Put/Call Ratio (Volume)", f"{pc_ratio:.2f}", delta=pc_sentiment)
                        st.caption(pc_interpretation)
                        st.caption(f"📊 Call Volume: {call_vol:,} | Put Volume: {put_vol:,}")
                    else:
                        st.info("Put/Call ratio unavailable - insufficient volume data")
                
                with col_pc2:
                    beta, beta_interpretation = calculate_beta(st.session_state.current_ticker)
                    st.metric("Beta (vs SPY)", f"{beta:.2f}", delta=beta_interpretation[:20])
                    st.caption(beta_interpretation)
                
                with col_pc3:
                    hv = calculate_hv(st.session_state.hist_data)
                    atm_idx = (calls_df['strike'] - S).abs().argsort()[:1]
                    current_iv = calls_df.iloc[atm_idx]['impliedVolatility'].iloc[0] * 100 if not calls_df.empty else 0
                    st.metric("Current IV", f"{current_iv:.1f}%")
                    st.metric("Historical Vol", f"{hv:.1f}%")
                
                st.divider()
                
                st.subheader("🎯 Dynamic Target Calculator (50% in 5 Days)")
                
                atm_idx = (calls_df['strike'] - S).abs().argsort()[:1]
                selected_strike = calls_df.iloc[atm_idx]['strike'].values[0]
                selected_row = calls_df[calls_df['strike'] == selected_strike].iloc[0]
                
                option_price = (selected_row['bid'] + selected_row['ask']) / 2 if selected_row['bid'] > 0 else selected_row['lastPrice']
                days_to_expiry = (pd.to_datetime(current_expiry).date() - datetime.now().date()).days
                T_years = max(days_to_expiry, 1) / 365
                d, g, theta, v = calculate_greeks(S, selected_strike, T_years, 0.05, selected_row['impliedVolatility'])
                
                target_stock, stop_stock = calculate_dynamic_targets(S, option_price, d, g, theta, days=5, target_gain_pct=0.50)
                
                col_d1, col_d2, col_d3 = st.columns(3)
                with col_d1:
                    st.metric("Current Stock Price", f"${S:.2f}")
                    st.metric("ATM Strike", f"${selected_strike:.2f}")
                with col_d2:
                    st.metric("Target Stock Price", f"${target_stock:.2f}", delta=f"${target_stock - S:.2f}")
                    st.metric("Stop Stock Price", f"${stop_stock:.2f}", delta=f"${stop_stock - S:.2f}")
                with col_d3:
                    st.metric("Required Move %", f"{((target_stock - S)/S)*100:.1f}%")
                    st.caption(f"Delta: {d:.3f} | Gamma: {g:.4f}")
                
                st.divider()
                st.subheader("📊 Volatility Analysis")
                
                hv = calculate_hv(st.session_state.hist_data)
                spread, spread_status = calculate_iv_hv_spread(selected_row['impliedVolatility'], hv)
                
                col_v1, col_v2, col_v3 = st.columns(3)
                with col_v1:
                    st.metric("Implied Volatility (IV)", f"{selected_row['impliedVolatility']*100:.1f}%")
                with col_v2:
                    st.metric("Historical Volatility (HV)", f"{hv:.1f}%")
                with col_v3:
                    st.metric("IV - HV Spread", f"{spread:.1f}%", delta=spread_status)
                
                st.divider()
                st.subheader("📐 Skew Analysis")
                
                skew, skew_status = calculate_skew(calls_df, puts_df, S, selected_strike)
                st.metric("Put/Call Volatility Skew", f"{skew:.1f}%", delta=skew_status)
                st.caption("Positive skew = Calls expensive (Bullish) | Negative skew = Puts expensive (Bearish)")
                
                st.divider()
                st.subheader("📈 Term Structure (IV by Expiry)")
                
                term_data, term_structure = calculate_term_structure(st.session_state.current_ticker, st.session_state.expiries)
                if term_data:
                    st.metric("Term Structure", term_structure)
                    term_df = pd.DataFrame(term_data)
                    st.line_chart(term_df.set_index('days')['iv'])
                    st.caption("Upward slope (Contango) = Normal | Downward slope (Backwardation) = Market stress")
                else:
                    st.info("Insufficient data for term structure")
                
                st.divider()
                st.subheader("💀 Max Pain Analysis")
                
                strikes = sorted(calls_df['strike'].unique())
                max_pain = calculate_max_pain(calls_df, puts_df, strikes)
                if max_pain:
                    st.metric("Max Pain Strike", f"${max_pain:.2f}")
                    if max_pain < S:
                        st.caption(f"Max pain is ${S - max_pain:.2f} below current price - Potential downward pull")
                    else:
                        st.caption(f"Max pain is ${max_pain - S:.2f} above current price - Potential upward pull")
                else:
                    st.info("Max pain calculation unavailable")
                
            else:
                st.warning("Option chain data unavailable for quant analysis")
        else:
            st.info("No expiries available. Analyze a ticker first.")
    else:
        st.info("👈 Analyze a ticker to see quant analytics")

# ========================
# TECHNICAL ANALYSIS TAB
# ========================
with t_tech:
    if st.session_state.price and not st.session_state.hist_data.empty:
        S = st.session_state.price
        st.subheader("Momentum & Volatility Health")
        
        try:
            atr_val, atr_pct = calculate_atr(st.session_state.hist_data)
            rsi_val = calculate_rsi(st.session_state.hist_data)
            hv_val = calculate_hv(st.session_state.hist_data)
        except:
            atr_val, atr_pct = 0.0, 0.0
            rsi_val = 50.0
            hv_val = 0.0
        
        col_a1, col_a2, col_a3 = st.columns(3)
        with col_a1:
            st.metric("ATR (14d)", f"${atr_val:.2f}", delta=f"{atr_pct:.1f}% of price")
        with col_a2:
            rsi_status = "Overbought" if rsi_val > 70 else ("Oversold" if rsi_val < 30 else "Neutral")
            st.metric("RSI (14d)", f"{rsi_val:.1f}", delta=rsi_status)
        with col_a3:
            st.metric("Historical Vol (20d)", f"{hv_val:.1f}%")
        
        current_expiry = st.session_state.last_selected_expiry if st.session_state.last_selected_expiry else (st.session_state.expiries[0] if st.session_state.expiries else None)
        if current_expiry:
            calls_df, _ = get_cached_option_chain(st.session_state.current_ticker, current_expiry)
            if calls_df is not None and not calls_df.empty:
                atm_idx = (calls_df['strike'] - S).abs().argsort()[:1]
                current_iv = calls_df.iloc[atm_idx]['impliedVolatility'].iloc[0] if not calls_df.empty else None
                if current_iv:
                    spread, spread_status = calculate_iv_hv_spread(current_iv, hv_val)
                    st.metric("IV/HV Spread", f"{spread:.1f}%", delta=spread_status)
        
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
    else:
        st.info("👈 Analyze a ticker to see technical analysis")

# ========================
# AI RESEARCH TAB (ENHANCED WITH CHAT)
# ========================
with t_ai:
    if st.session_state.current_ticker:
        st.header("🤖 AI Research")
        
        # Initialize chat session state
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        if 'initial_ai_message_sent' not in st.session_state:
            st.session_state.initial_ai_message_sent = False
        
        # Function to generate the comprehensive AI insight (same as Dashboard)
        def generate_comprehensive_ai_insight(ticker):
            """Generate the comprehensive AI insight with macro context"""
            if not st.session_state.price or not st.session_state.expiries:
                return None
            
            S = st.session_state.price
            hist_data = st.session_state.hist_data
            
            # Get all the metrics
            vix = get_vix()
            atr_val, atr_pct = calculate_atr(hist_data)
            rsi_val = calculate_rsi(hist_data)
            hv_val = calculate_hv(hist_data)
            
            current_expiry = st.session_state.last_selected_expiry if st.session_state.last_selected_expiry else (st.session_state.expiries[0] if st.session_state.expiries else None)
            
            if current_expiry:
                calls_df, _ = get_cached_option_chain(ticker, current_expiry)
                if calls_df is not None and not calls_df.empty:
                    atm_idx = (calls_df['strike'] - S).abs().argsort()[:1]
                    current_iv = calls_df.iloc[atm_idx]['impliedVolatility'].iloc[0] if not calls_df.empty else 0.35
                else:
                    current_iv = 0.35
            else:
                current_iv = 0.35
            
            iv_hv_spread, _ = calculate_iv_hv_spread(current_iv, hv_val)
            
            earnings_date = get_earnings_date(ticker)
            if earnings_date:
                days_to_earnings = (earnings_date - datetime.now().date()).days
            else:
                days_to_earnings = None
            
            beta, _ = calculate_beta(ticker)
            
            try:
                if current_expiry:
                    calls_df, puts_df = get_cached_option_chain(ticker, current_expiry)
                    if calls_df is not None and not calls_df.empty:
                        skew, _ = calculate_skew(calls_df, puts_df if puts_df is not None else pd.DataFrame(), S)
                    else:
                        skew = 0
                else:
                    skew = 0
            except:
                skew = 0
            
            curr, prev = get_technicals(hist_data)
            ema_status = "Bullish Cross" if curr['ema8'] > curr['ema20'] else "Bearish Separation"
            term_structure = "Neutral"
            
            pc_ratio = 0.5
            try:
                pc_ratio, pc_sentiment, pc_interpretation, call_vol, put_vol = calculate_put_call_ratio(ticker, current_expiry)
                if pc_ratio is None:
                    pc_ratio = 0.5
            except:
                pc_ratio = 0.5
            
            sentiment_data = getattr(st.session_state, 'current_sentiment', None)
            sentiment_score = sentiment_data.get('sentiment_score', 0) if sentiment_data else 0
            
            weighted_verdict, weighted_confidence = calculate_weighted_verdict(
                vix, rsi_val, iv_hv_spread, sentiment_score, skew, beta
            )
            
            # Get macro data
            macro_data = get_macro_data()
            sector, industry = get_sector_for_ticker(ticker)
            
            # Get the trading recommendation
            decision_data = {
                'price': S,
                'rsi': rsi_val,
                'iv_hv_spread': iv_hv_spread,
                'pcr': pc_ratio if pc_ratio else 0.5,
                'beta': beta,
                'ema_status': ema_status,
                'bollinger_pos': ((S - curr['lower']) / (curr['upper'] - curr['lower'])) * 100 if 'lower' in curr and 'upper' in curr else 50,
                'term_structure': term_structure if term_structure else "Neutral",
                'market_verdict': weighted_verdict,
                'market_confidence': weighted_confidence,
                'vix': vix,
                'atr_pct': atr_pct
            }
            
            decision = get_trading_recommendation(decision_data)
            
            # Build macro context
            macro_context = f"""
MACRO CONTEXT:
- VIX: {macro_data.get('vix', 15):.1f} ({macro_data.get('regime', 'Neutral')})
- SPY 5-Day Change: {macro_data.get('spy_change_5d', 0):+.1f}%
- Market Regime: {macro_data.get('regime_description', 'Normal')}
- Earnings: {f"Next earnings in {days_to_earnings} days" if earnings_date and days_to_earnings > 0 else "No upcoming earnings" if not earnings_date else f"Earnings {abs(days_to_earnings)} days ago"}
"""
            
            # Build sector performance context
            sector_performance = ""
            if macro_data.get('sector_data'):
                sector_performance = "SECTOR PERFORMANCE (5-Day):\n"
                for sector_name, data in macro_data['sector_data'].items():
                    sector_performance += f"- {sector_name}: {data['change_5d']:+.1f}%\n"
            
            # Get stock-specific news
            stock_news_context = ""
            stock_news = get_stock_news(ticker, limit=5)
            if stock_news:
                stock_news_context = f"\n{ticker}-SPECIFIC NEWS (Last 7 Days):\n"
                for i, article in enumerate(stock_news):
                    headline = article.get('headline', 'No title')
                    summary = article.get('summary', '')[:150]
                    source = article.get('source', 'Unknown')
                    date = article.get('datetime', '')
                    stock_news_context += f"{i+1}. {headline}\n"
                    stock_news_context += f"   Summary: {summary}\n"
                    stock_news_context += f"   Source: {source} | Date: {date}\n\n"
            else:
                stock_news_context = f"\nNo recent news found for {ticker} in the last 7 days.\n"
            
            # Build the comprehensive AI prompt (same as Dashboard)
            ai_prompt = f"""
You are a professional options trader and quantitative analyst. Based on the following comprehensive data for {ticker}, provide a concise trading insight.

TICKER: {ticker}
SECTOR: {sector} | INDUSTRY: {industry}

TECHNICAL METRICS:
- Price: ${S:.2f}
- RSI (14d): {rsi_val:.1f}
- 8/20 EMA Status: {ema_status}
- Bollinger Position: {((S - curr['lower']) / (curr['upper'] - curr['lower'])) * 100:.0f}% of band
- Trend: {st.session_state.trend}

VOLATILITY METRICS:
- Implied Volatility (IV): {current_iv * 100:.1f}%
- Historical Volatility (HV): {hv_val:.1f}%
- IV/HV Spread: {iv_hv_spread:.1f}%
- Beta (vs SPY): {beta:.2f}
- Term Structure: {term_structure if term_structure else "Neutral"}

SENTIMENT METRICS:
- Put/Call Ratio: {pc_ratio if pc_ratio else 0.5:.2f}
- Market Verdict: {weighted_verdict} ({weighted_confidence:.0f}% confidence)

{macro_context}
{sector_performance}
{stock_news_context}

MY RECOMMENDATION: {decision['recommendation']}
- Confidence: {decision['confidence']}%
- Entry Zone: ${decision['entry_zone_low']:.2f} - ${decision['entry_zone_high']:.2f}
- Stop Loss: ${decision['stop_loss']:.2f}
- Target: ${decision['target_price']:.2f}
- Position Size: {decision['position_size']}

Provide a 3-4 sentence insight that:
1. Acknowledges the macro environment and how it affects this trade
2. Mentions the sector context and any relevant sector trends
3. INCORPORATES ANY RECENT STOCK-SPECIFIC NEWS OR CATALYSTS (this is critical!)
4. Explains the key reason for the recommendation
5. Gives a clear, actionable takeaway

Keep it professional, concise, and actionable. Focus on the intersection of macro trends, sector performance, stock catalysts, and this specific stock.
"""
            
            try:
                groq_api_key = st.secrets.get("GROQ_API_KEY")
                if groq_api_key:
                    client = Groq(api_key=groq_api_key)
                    response = call_groq_with_retry(client, ai_prompt)
                    if response:
                        return response
            except:
                pass
            
            return None
        
        # Function to send initial AI message
        def send_initial_ai_message():
            if st.session_state.current_ticker and not st.session_state.initial_ai_message_sent:
                # Generate the comprehensive AI insight
                ai_insight = generate_comprehensive_ai_insight(st.session_state.current_ticker)
                
                if ai_insight:
                    # Also fetch the news sentiment for additional context
                    if not st.session_state.ai_brief:
                        with st.spinner("Gathering news sentiment..."):
                            st.session_state.ai_brief = get_ai_research(st.session_state.current_ticker)
                    
                    # Create the initial message with both insights
                    initial_message = f"""🤖 **AI Trading Recommendation for {st.session_state.current_ticker}**

{ai_insight}

---
💡 **Sentiment Summary:**
{st.session_state.ai_brief}

---
*💬 Feel free to ask follow-up questions about this analysis, the options structure, or any other aspects of {st.session_state.current_ticker}.*"""
                else:
                    # Fallback to just the news sentiment if AI insight fails
                    if not st.session_state.ai_brief:
                        with st.spinner("Generating AI insights..."):
                            st.session_state.ai_brief = get_ai_research(st.session_state.current_ticker)
                    
                    initial_message = f"""🤖 **AI Analysis for {st.session_state.current_ticker}**

{st.session_state.ai_brief}

---
*💬 Feel free to ask follow-up questions about this analysis or any other aspects of {st.session_state.current_ticker}.*"""
                
                st.session_state.chat_history.append({"role": "assistant", "content": initial_message})
                st.session_state.initial_ai_message_sent = True
        
        # Send initial AI message if ticker is analyzed and not yet sent
        if st.session_state.price and st.session_state.expiries:
            send_initial_ai_message()
        
        # Chat container
        chat_container = st.container()
        
        with chat_container:
            # Display chat history
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
        
        # Chat input
        if st.session_state.price and st.session_state.expiries:
            # Prepare stock metrics for context
            stock_metrics = {
                "Price": f"${st.session_state.price:.2f}",
                "RSI": st.session_state.get('saved_rsi', 'N/A'),
                "IV/HV Spread": f"{st.session_state.get('iv_hv_spread', 0):.1f}%",
                "Beta": st.session_state.get('beta', 'N/A'),
                "Market Verdict": st.session_state.get('weighted_verdict', 'N/A'),
                "Implied Volatility": f"{st.session_state.get('iv_pct', 0):.1f}%",
                "Historical Volatility": f"{st.session_state.get('hv', 0):.1f}%",
                "Put/Call Ratio": st.session_state.get('pc_ratio', 'N/A'),
                "8/20 EMA Status": st.session_state.get('ema_status', 'Neutral'),
                "Trend": st.session_state.get('trend', 'Neutral')
            }
            
            # Chat input
            user_question = st.chat_input("Ask a follow-up question about this stock or options...")
            
            if user_question:
                # Add user message
                st.session_state.chat_history.append({"role": "user", "content": user_question})
                
                # Get AI response
                with st.spinner("Thinking..."):
                    response = get_ai_chat_response(
                        st.session_state.current_ticker,
                        user_question,
                        st.session_state.chat_history,
                        stock_metrics
                    )
                
                # Add AI response
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                
                # Rerun to update chat display
                st.rerun()
            
            # Buttons below chat
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 Refresh AI Analysis", use_container_width=True):
                    st.session_state.ai_brief = ""
                    st.session_state.chat_history = []
                    st.session_state.initial_ai_message_sent = False
                    if 'current_sentiment' in st.session_state:
                        del st.session_state.current_sentiment
                    # Clear cache for this ticker
                    cache_key = f"news_sentiment_{st.session_state.current_ticker}"
                    if cache_key in st.session_state.ai_cache.cache:
                        del st.session_state.ai_cache.cache[cache_key]
                    st.rerun()
            
            with col2:
                if st.button("🗑️ Clear Chat History", use_container_width=True):
                    st.session_state.chat_history = []
                    st.session_state.initial_ai_message_sent = False
                    st.rerun()
        else:
            st.info("👈 Analyze a ticker to start the AI chat")
    else:
        st.info("👈 Analyze a ticker to get AI research and chat")

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
    
    st.subheader("📊 Global Recommendations - Strategy Ranges")
    st.markdown("""
    | Strategy | Optimal Expiry Range | Delta Range | Goal |
    |----------|---------------------|-------------|------|
    | **Conservative** | 60+ DTE (closest to 90 days) | 0.50 - 0.60 | Higher probability, slower returns |
    | **Aggressive** | 30 - 45 DTE | 0.40 - 0.49 | 50% in 5 days target |
    | **Speculative** | 15 - 30 DTE | 0.30 - 0.39 | Fastest gamma, highest risk |
    
    These recommendations are **fixed** and based on the optimal expiry for each strategy. They do NOT change when you select a different expiry in the Workspace Adjuster.
    """)
    
    st.divider()
    
    st.subheader("📊 Workspace Adjuster")
    st.markdown("""
    The Workspace Adjuster lets you select ANY expiry date. The Conservative, Aggressive, and Speculative tabs will update to show the best strikes for that specific expiry. This allows you to compare different expiries without re-analyzing the ticker.
    """)
    
    st.divider()
    
    st.subheader("📊 Phase 5: Gamma/Theta Ratio & Skew Penalty")
    st.markdown("""
    | Indicator | What It Measures | How To Use |
    |-----------|------------------|------------|
    | **Gamma/Theta Ratio** | Acceleration vs time decay | > 1.0 = Good acceleration potential; > 1.5 = Excellent for 5-day targets |
    | **Skew Penalty** | Adjusts EV based on put/call skew | Negative skew = 15% EV penalty; Positive skew = 5% EV bonus |
    """)
    
    st.divider()
    
    st.subheader("📊 Advanced Quantitative Indicators")
    st.markdown("""
    | Indicator | What It Measures | How To Use |
    |-----------|------------------|------------|
    | **Dynamic Target** | Stock price needed for 50% option gain in 5 days | Uses Gamma/Theta math - more accurate than fixed targets |
    | **IV/HV Spread** | Implied vs Historical Volatility | IV > HV = expensive options; IV < HV = cheap options |
    | **Put/Call Skew** | Difference in put vs call IV | Positive skew = bullish bias; Negative skew = bearish bias |
    | **Term Structure** | IV across different expiries | Contango (upward) = normal; Backwardation (downward) = market stress |
    | **Max Pain** | Strike where option writers profit most | Acts as gravitational price target |
    """)
    
    st.divider()
    
    st.subheader("📊 Put/Call Ratio & Beta Adjustment")
    st.markdown("""
    | Indicator | What It Measures | How To Use |
    |-----------|------------------|------------|
    | **Put/Call Ratio (Volume)** | Ratio of put volume to call volume | > 1.2 = Bearish sentiment; < 0.8 = Bullish sentiment |
    | **Beta** | Stock volatility relative to market (SPY) | Beta > 1.2 = More volatile; Beta < 0.8 = Less volatile |
    | **Beta-Adjusted Risk** | Risk score multiplied by beta factor | Higher beta = higher effective risk |
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
    **Step 1: Check Global Recs** - See optimal trades for each strategy across ALL expiries
    
    **Step 2: Use Workspace Adjuster** - Select any expiry to analyze specific contracts
    
    **Step 3: Liquidity Filter (MANDATORY)**
    - [ ] Volume > 50 (preferably > 200)
    - [ ] Open Interest > 500
    - [ ] Spread < 5% (preferably < 3%)
    
    **Step 4: Strategy Match**
    - [ ] Delta matches your risk profile
    - [ ] Days to expiry matches your target timeframe
    
    **Step 5: Technical Confirmation**
    - [ ] RSI between 30-70 (not overbought/oversold)
    - [ ] ATR sufficient for target (target move > 2x ATR)
    - [ ] 8 EMA > 20 EMA (bullish)
    
    **Step 6: Quant Confirmation**
    - [ ] IV/HV spread not excessive (>10% expensive)
    - [ ] Skew aligns with directional bias
    - [ ] Gamma/Theta ratio > 1.0 (for aggressive targets)
    
    **Step 7: Math Confirmation**
    - [ ] Composite Score > 55
    - [ ] Expected Value > 0.25
    
    **Step 8: Trade Management Plan**
    - [ ] Dynamic target calculated
    - [ ] Stop loss at 25-30% option loss
    - [ ] Exit by cutoff date
    """)
    
    st.warning("""
    **⚠️ Remember:** No indicator is perfect. Always size positions appropriately 
    (never risk more than 1-2% of account per trade) and follow your stop losses.
    """)
