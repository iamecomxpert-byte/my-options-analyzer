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
st.set_page_config(page_title="Analyst Pro Options Suite v6", layout="wide")

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

def get_combined_recommendation(quant_score, ai_score, option_price, stop_loss, days_left):
    """
    Combine quantitative score and AI sentiment for position recommendation.
    
    Returns: (recommendation_icon, reason, combined_score)
    """
    if option_price <= stop_loss:
        return "🔴 EXIT", "Stop loss triggered", 0
    
    # Normalize AI score from -1..1 to 0..100
    ai_score_normalized = (ai_score + 1) * 50
    
    # Weighted combination: 60% quant, 40% AI
    combined_score = (quant_score * 0.6) + (ai_score_normalized * 0.4)
    
    # Time decay urgency
    if days_left < 3:
        combined_score *= 0.7
        time_note = " | URGENT: <3 days left"
    else:
        time_note = ""
    
    if combined_score >= 70 and ai_score >= 0.3:
        return "🟢 STRONG HOLD", f"Quant: {quant_score}/100 | AI: {ai_score:+.2f}{time_note}", combined_score
    elif combined_score >= 55:
        return "🔵 HOLD", f"Maintain position | Combined: {combined_score:.0f}{time_note}", combined_score
    elif combined_score >= 40:
        return "🟡 MONITOR", f"Mixed signals | Tighten stops{time_note}", combined_score
    elif combined_score >= 25:
        return "🟠 CONSIDER EXIT", f"Quant weak ({quant_score}) or AI bearish{time_note}", combined_score
    else:
        return "🔴 EXIT", f"Strong exit signals | Combined: {combined_score:.0f}{time_note}", combined_score

def calculate_portfolio_forecast(positions, st_price=None):
    """
    Aggregate 5-day forecasts for all active positions.
    
    Returns dictionary with aggregated metrics.
    """
    if not positions:
        return None
    
    total_current_value = 0
    total_forecast_value = 0
    total_theta_decay = 0
    positions_improving = 0
    positions_declining = 0
    
    position_forecasts = []
    
    for pos in positions:
        try:
            ticker = pos['ticker']
            expiry = pos['expiry']
            strike = float(pos['strike'])
            contracts = int(pos['contracts'])
            entry_price = float(pos['entry_price'])
            
            # Get current option price and Greeks
            current_price, current_iv, gamma, theta = get_current_option_price(ticker, expiry, strike)
            
            if current_price and current_price > 0:
                # Get stock price
                stock = yf.Ticker(ticker)
                hist = stock.history(period="20d")
                if not hist.empty:
                    current_stock = hist['Close'].iloc[-1]
                    
                    # Calculate delta for this contract
                    days_left = max((pd.to_datetime(expiry).date() - datetime.now().date()).days, 1)
                    d, _, _, _ = calculate_greeks(current_stock, strike, days_left/365, 0.05, current_iv)
                    
                    # ========== FIXED: Correct forecast with all 12 parameters ==========
                    expected_price, _, _, theta_decay, _, _ = forecast_5day_price(
                        current_price,      # current_option_price
                        current_stock,      # stock_price  
                        strike,             # strike
                        d,                  # delta
                        gamma,              # gamma
                        theta,              # theta
                        0,                  # vega (not used in portfolio aggregate)
                        current_iv,         # current_iv
                        0,                  # forecast_iv_change
                        5,                  # days
                        0.03                # expected_stock_move_pct (3%)
                    )
                    # ================================================================
                    
                    position_value = current_price * contracts * 100
                    forecast_value = expected_price * contracts * 100
                    
                    total_current_value += position_value
                    total_forecast_value += forecast_value
                    total_theta_decay += abs(theta_decay) * contracts * 100
                    
                    if forecast_value > position_value:
                        positions_improving += 1
                    else:
                        positions_declining += 1
                    
                    position_forecasts.append({
                        'ticker': ticker,
                        'current': position_value,
                        'forecast': forecast_value,
                        'improving': forecast_value > position_value
                    })
        except Exception as e:
            continue
    
    if total_current_value == 0:
        return None
    
    expected_change = total_forecast_value - total_current_value
    expected_change_pct = (expected_change / total_current_value) * 100
    
    return {
        'total_current_value': total_current_value,
        'total_forecast_value': total_forecast_value,
        'expected_change': expected_change,
        'expected_change_pct': expected_change_pct,
        'total_theta_decay_5d': total_theta_decay,
        'positions_improving': positions_improving,
        'positions_declining': positions_declining,
        'total_positions': len(positions),
        'position_forecasts': position_forecasts
    }

def get_ai_forecast_for_position(ticker, current_price, strike, current_iv):
    """
    Get cached AI sentiment for a specific ticker.
    Returns sentiment dict or None.
    """
    # Check if we have cached AI for this ticker
    cache_key = f"news_sentiment_{ticker}"
    cached_sentiment = st.session_state.ai_cache.get(cache_key)
    
    if cached_sentiment:
        # Try to parse sentiment data from cached response
        try:
            # The cached response is HTML/markdown, need to extract sentiment
            # For now, return the stored sentiment from session state
            if hasattr(st.session_state, 'current_sentiment') and st.session_state.current_sentiment:
                return st.session_state.current_sentiment
        except:
            pass
    
    # Return default neutral sentiment
    return {
        'sentiment_score': 0.0,
        'sentiment_label': 'Neutral',
        'catalyst': None,
        'key_themes': ['No recent news'],
        'risk_adjustment': 0
    }

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
    # Handle None or non-numeric values
    if gamma is None or theta is None:
        return 0.0
    try:
        gamma = float(gamma)
        theta = float(theta)
        if theta == 0:
            return 0.0
        ratio = abs(gamma / theta)
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
    delta = max(0.0, min(1.0, float(delta) if delta is not None else 0.0))
    p_touch = max(0.0, min(1.0, float(p_touch) if p_touch is not None else 0.0))
    gamma_theta_ratio = max(0.0, float(gamma_theta_ratio) if gamma_theta_ratio is not None else 0.0)
    tech_score = float(tech_score) if tech_score is not None else 0.0
    
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
        
        # get_current_option_price returns 4 values (mid, iv, gamma, theta)
        option_price, _, _, _ = get_current_option_price(pos['ticker'], pos['expiry'], float(pos['strike']))
        if option_price:
            unrealized = (option_price - entry_price) * contracts * 100
            total_unrealized_pnl += unrealized
        
        realized = float(pos.get('realized_pnl', 0))
        total_realized_pnl += realized
    
    return total_investment, total_unrealized_pnl, total_realized_pnl

def calculate_risk_score(pos, current_price, current_delta, days_left, current_iv, sentiment_adjustment=0):
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
    
    sentiment_score_normalized = max(-1, min(1, -sentiment_adjustment / 50))
    sentiment_risk = (1 - sentiment_score_normalized) / 2
    
    risk_score = (size_score * 0.20 + delta_score * 0.20 + time_score * 0.15 + iv_score * 0.15 + moneyness_score * 0.10 + sentiment_risk * 0.20)
    
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
            
            # Get current stock price
            stock_price = get_cached_current_price(ticker)
            if stock_price is None:
                stock_price = yf.Ticker(ticker).history(period="1d")['Close'].iloc[-1]
            
            # Calculate days to expiry
            days_to_expiry = (pd.to_datetime(expiry).date() - datetime.now().date()).days
            T_years = max(days_to_expiry, 1) / 365
            
            # FIX: calculate_greeks returns (delta, gamma, theta, vega) in THAT order
            # Your current call might be misaligned
            delta, gamma, theta, vega = calculate_greeks(stock_price, float(strike), T_years, 0.05, iv)
                        
            return mid, iv, gamma, theta
        return None, None, None, None
    except Exception as e:
        st.caption(f"Error in get_current_option_price: {str(e)}")
        return None, None, None, None

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
    'current_sentiment': None, 'active_tab': 0, 'last_ai_refresh': None
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
    
    # --- FIXED: Workspace Adjuster header now INSIDE the conditional ---
    if st.session_state.expiries and len(st.session_state.expiries) > 0:
        st.subheader("🔍 Workspace Adjuster")
        
        # Find index of current selection
        current_index = 0
        if st.session_state.last_selected_expiry and st.session_state.last_selected_expiry in st.session_state.expiries:
            current_index = st.session_state.expiries.index(st.session_state.last_selected_expiry)
        
        expiry = st.selectbox(
            "Select Expiry for Individual Tabs Below:", 
            st.session_state.expiries,
            index=current_index,
            help="Select any expiry to update the Conservative, Aggressive, and Speculative tabs below"
        )
        
        if expiry != st.session_state.get('last_selected_expiry'):
            st.session_state.last_selected_expiry = expiry
            st.rerun()
    else:
        # No expiries loaded yet
        if fetch_btn:
            # Just clicked analyze - show loading message
            st.info("⏳ Loading expiries... Please wait a moment.")
        else:
            # Initial state
            st.info("👈 Enter a ticker and click 'Analyze Options Structure'")
    
    st.divider()
    if st.button("🗑️ Clear Cache", help="Clear cached data if you're seeing stale information"):
        st.cache_data.clear()
        for key in ['expiries', 'last_selected_expiry', 'price', 'hist_data', 'stock_name', 'trend', 'pct_change', 'tech_score', 'verdict_reasons', 'global_conservative', 'global_aggressive', 'global_speculative', 'data_fetched']:
            if key in st.session_state:
                del st.session_state[key]
        st.success("Cache cleared! Refresh the page to reload data.")
        st.rerun()

# --- DATA FETCHING & GLOBAL SCANS ---
if fetch_btn:
    st.session_state.current_ticker = ticker_input
    st.session_state.ai_brief = "" 
    st.session_state.global_conservative = None
    st.session_state.global_aggressive = None
    st.session_state.global_speculative = None
    st.session_state.last_selected_expiry = None
    st.session_state.data_fetched = True
    
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
            
            # Store in session state
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

            # Get Global Recommendations (independent of workspace adjuster)
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
            
            # ============================================================
            # STORE PATH EXPECTATION VALUES ONCE AT FETCH TIME
            # ============================================================
            # Calculate and store technical values that shouldn't change with expiry
            vix_value = get_vix()
            atr_val, atr_pct_stored = calculate_atr(hist)
            rsi_val_stored = calculate_rsi(hist)
            
            # Store in session state for consistent dashboard display
            st.session_state.saved_rsi = rsi_val_stored
            st.session_state.saved_atr_pct = atr_pct_stored
            st.session_state.saved_vix = vix_value
            st.session_state.saved_atr_trend = atr_pct_stored > 1.5
            st.session_state.saved_vix_trend = vix_value > 20
            
            # Get sentiment from session state if available
            sentiment_data_stored = getattr(st.session_state, 'current_sentiment', None)
            saved_sentiment_score = sentiment_data_stored.get('sentiment_score', 0) if sentiment_data_stored else 0
            st.session_state.saved_sentiment_score = saved_sentiment_score
            
            # Calculate and store the path expectation
            path_icon, path_text = calculate_path_expectation(
                st.session_state.saved_rsi,
                st.session_state.saved_sentiment_score,
                st.session_state.saved_atr_trend,
                st.session_state.saved_vix_trend
            )
            st.session_state.saved_path_icon = path_icon
            st.session_state.saved_path_text = path_text
            # ============================================================

    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")

    # FORCE A RERUN AFTER SETTING ALL SESSION STATE VARIABLES
    st.rerun()

# --- MAIN DASHBOARD VIEW ---
if st.session_state.price and st.session_state.expiries:
    S = st.session_state.price
    st.header(f"{st.session_state.stock_name} ({st.session_state.current_ticker})")
    
    col_p, col_t = st.columns(2)
    col_p.metric("Current Underlying Price", f"${S:.2f}")
    col_t.metric("20-Day Baseline Trend", st.session_state.trend, f"{st.session_state.pct_change:.1f}%")

    st.divider()
    
    # Tabs
    t_dashboard, t_quant, t_summary, t_cons, t_aggr, t_spec, t_tech, t_ai, t_portfolio, t_edu = st.tabs([
        "📊 Dashboard", "🔬 Quant Analytics", "📋 Global Recs", "🛡️ Conservative", 
        "⚡ Aggressive", "🎰 Speculative", "📊 Technical", "🤖 AI Research", 
        "📂 Portfolio", "📖 Strategy Guide"
    ])

    # ========================
    # DASHBOARD TAB
    # ========================
    with t_dashboard:
        st.header("📊 Trading Dashboard")
        
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
        
        sentiment_data = getattr(st.session_state, 'current_sentiment', None)
        if sentiment_data:
            sentiment_score = sentiment_data.get('sentiment_score', 0)
        else:
            sentiment_score = 0
        
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
        # USE SAVED PATH EXPECTATION (doesn't change with expiry)
        # ============================================================
        if 'saved_path_icon' in st.session_state and st.session_state.saved_path_icon:
            path_icon = st.session_state.saved_path_icon
            path_text = st.session_state.saved_path_text
        else:
            # Fallback calculation if not saved
            path_icon, path_text = calculate_path_expectation(rsi_val, sentiment_score, atr_pct > 1.5, vix > 20)
        
        st.info(f"{path_icon} **Path Expectation:** {path_text}")
        
        # Also store the current pullback condition for later use
        is_pullback_scenario = (path_icon == "📉📈 Pullback then rise")
        # ============================================================
        
        # ADD THIS RIGHT AFTER THE ABOVE LINE:
        # ============================================================
        # PULLBACK ENTRY ESTIMATES (Only for Pullback Then Rise scenario)
        # ============================================================
        if is_pullback_scenario:
            st.divider()
            st.subheader("🎯 Pullback Entry Estimates")
            st.caption("Based on ATR, RSI, Bollinger Bands, and Max Pain analysis")
            
            # Get current indicators for pullback calculation
            curr_indicators, _ = get_technicals(hist_data)
            
            # Get max pain for current expiry if available
            max_pain_value = None
            if current_expiry:
                try:
                    calls_df, puts_df = get_cached_option_chain(st.session_state.current_ticker, current_expiry)
                    if calls_df is not None and not calls_df.empty:
                        strikes = sorted(calls_df['strike'].unique())
                        max_pain_value = calculate_max_pain(calls_df, puts_df, strikes)
                except:
                    pass
            
            # Calculate pullback recommendations
            pullback_data = get_pullback_entry_recommendation(
                S, hist_data, max_pain_value, rsi_val, sentiment_score, atr_pct
            )
            
            # Display entry estimates in columns
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
            
            # Sentiment context
            st.caption(f"💭 Sentiment Context: {pullback_data['sentiment_text']}")
            
            # Actionable alert based on current price vs entry
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
            
            # Option strategy recommendation based on pullback
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
            
            # Visual gauge showing current price vs entry zones
            st.divider()
            st.subheader("📊 Price Position Gauge")
            
            # Create a simple visual representation
            max_price = current_price * 1.02
            min_price = pullback_data['entry_range']['aggressive'] * 0.98
            
            # Calculate position as percentage
            position_pct = ((current_price - min_price) / (max_price - min_price)) * 100
            position_pct = max(0, min(100, position_pct))
            
            # Display gauge
            st.progress(position_pct / 100)
            st.caption(f"Current price is {position_pct:.0f}% of the way from aggressive entry to 2% above current")
            
            col_g1, col_g2, col_g3 = st.columns(3)
            with col_g1:
                st.caption(f"🔴 Aggressive Entry\n${pullback_data['entry_range']['aggressive']:.2f}")
            with col_g2:
                st.caption(f"🟡 Optimal Entry\n${pullback_data['estimated_entry']:.2f}")
            with col_g3:
                st.caption(f"🟢 Conservative Entry\n${pullback_data['entry_range']['conservative']:.2f}")
        
        # ============================================================
        # END OF PULLBACK ENTRY ESTIMATES CODE
        # ============================================================
        
        st.divider()
        
        st.subheader("📋 Strategy Recommendations")
        st.caption("💡 Click any row to view detailed analysis in the corresponding strategy tab")
        
        current_expiry = st.session_state.last_selected_expiry if st.session_state.last_selected_expiry else (st.session_state.expiries[0] if st.session_state.expiries else None)
        
        if current_expiry:
            strategy_configs = [
                {"name": "🛡️ Conservative", "delta_min": 0.50, "delta_max": 0.60, "tab_index": 3, "tooltip": "Higher probability (50-60%), lower return, slower time to target"},
                {"name": "⚡ Aggressive", "delta_min": 0.40, "delta_max": 0.49, "tab_index": 4, "tooltip": "Medium probability (40-49%), medium return, balanced risk/reward"},
                {"name": "🎰 Speculative", "delta_min": 0.30, "delta_max": 0.39, "tab_index": 5, "tooltip": "Lower probability (30-39%), highest return potential, fastest time to target"}
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

    # ========================
    # QUANT ANALYTICS TAB
    # ========================
    with t_quant:
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

    # ========================
    # GLOBAL RECS TAB (UPDATED with Expiry Column)
    # ========================
    with t_summary:
        st.subheader("🏁 Automated Quantitative Trading Dashboard")
        st.markdown("Mathematically optimal contracts across ALL expiries (independent of sidebar selection)")
        
        sum_data = []
        profiles = [
            ("🛡️ Conservative", st.session_state.global_conservative, "50-60%", "60+ DTE (closest to 90 days)"),
            ("⚡ Aggressive", st.session_state.global_aggressive, "40-49%", "30-45 DTE"),
            ("🎰 Speculative", st.session_state.global_speculative, "30-39%", "15-30 DTE")
        ]
        
        for name, profile, target_d, range_desc in profiles:
            if profile:
                t_exit = profile['mid'] * (1 + profit_target_pct / 100)
                s_loss = profile['mid'] * (1 - stop_loss_pct / 100)
                h_days = min(int(profile['days'] * 0.4), 45)
                h_date = (datetime.now() + timedelta(days=h_days)).strftime('%b %d, %Y')
                
                # Format expiry date
                expiry_display = profile['expiry'] if 'expiry' in profile else 'N/A'
                
                sum_data.append({
                    "Strategy": name,
                    "Optimal Expiry": expiry_display,
                    "Expiry Range": range_desc,
                    "Target Delta": target_d,
                    "Strike": f"${profile['strike']:.2f} Call",
                    "Entry": f"${profile['mid']:.2f}",
                    "Target": f"${t_exit:.2f}",
                    "Stop": f"${s_loss:.2f}",
                    "Score": f"{profile['cts']}/100"
                })
        
        if sum_data:
            st.dataframe(pd.DataFrame(sum_data), use_container_width=True)
            st.caption("💡 These recommendations are FIXED and do not change when you select a different expiry in the sidebar.")
        else:
            st.warning("No contracts met the criteria. Try a different ticker or adjust your exit parameters.")

    # ========================
    # CONSERVATIVE / AGGRESSIVE / SPECULATIVE TABS
    # ========================
    
    def process_tier_strategy(tab_component, delta_min, delta_max, tier_label, tech_score):
        with tab_component:
            # Use the expiry from sidebar (Workspace Adjuster)
            current_expiry = st.session_state.last_selected_expiry if st.session_state.last_selected_expiry else (st.session_state.expiries[0] if st.session_state.expiries else None)
            if not current_expiry:
                st.warning("No expiry selected. Please analyze a ticker first.")
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
                
                # Apply skew penalty
                try:
                    if puts_df is not None and not puts_df.empty:
                        skew_val, _ = calculate_skew(calls_df, puts_df, S, row['strike'])
                        ev = apply_skew_penalty(ev, skew_val / 100)
                except:
                    pass
                
                # Calculate enhanced CTS with Gamma/Theta ratio
                gt_ratio = calculate_gamma_theta_ratio(g, t)
                if gt_ratio is None:
                    gt_ratio = 0.0
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
                key=f"compare_{tier_label}_{current_expiry}"
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

    process_tier_strategy(t_cons, 0.50, 0.60, "Conservative", st.session_state.get('tech_score', 0))
    process_tier_strategy(t_aggr, 0.40, 0.49, "Aggressive", st.session_state.get('tech_score', 0))
    process_tier_strategy(t_spec, 0.30, 0.39, "Speculative", st.session_state.get('tech_score', 0))

    # ========================
    # TECHNICAL ANALYSIS TAB
    # ========================
    with t_tech:
        if not st.session_state.hist_data.empty:
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

            st.divider()
            st.line_chart(st.session_state.hist_data[['Close', 'ema8', 'ema20', 'upper', 'lower']])

            st.subheader("🏁 Final Technical Verdict")
            tech_score_val = st.session_state.get('tech_score', 0)
            if tech_score_val == 3:
                st.success("🎯 **VERDICT: INVEST.** All indicators are aligned.")
            elif tech_score_val == 2:
                st.warning("⚖️ **VERDICT: CAUTION.** Mixed signals.")
            else:
                st.error("🛑 **VERDICT: STAY AWAY.** Bearish structure.")
            
            with st.expander("View Verdict Logic"):
                verdict_reasons_val = st.session_state.get('verdict_reasons', [])
                if verdict_reasons_val:
                    for reason in verdict_reasons_val:
                        st.write(f"- {reason}")
                if tech_score_val < 2:
                    st.write("- Multiple indicators show declining strength or bearish crossovers.")
        else:
            st.warning("⚠️ Technical analysis stream offline.")

    # ========================
    # AI RESEARCH TAB
    # ========================
    with t_ai:
        if st.button("🔄 Refresh AI Analysis", use_container_width=True):
            st.session_state.ai_brief = ""
            if 'current_sentiment' in st.session_state:
                del st.session_state.current_sentiment
            st.rerun()
        
        if not st.session_state.ai_brief:
            with st.spinner("Fetching latest news and generating AI sentiment analysis..."):
                st.session_state.ai_brief = get_ai_research(st.session_state.current_ticker)
        
        st.markdown(st.session_state.ai_brief)

    # ========================
    # PORTFOLIO TAB (UPDATED with Forecasts and Risk Breakdown)
    # ========================
    with t_portfolio:
        st.header("📂 Options Portfolio Tracker")
        
        if 'sheet_initialized' not in st.session_state:
            init_portfolio_sheet()
            st.session_state.sheet_initialized = True
        
        # Refresh AI Forecasts function
        def refresh_all_ai_forecasts(positions):
            """Refresh AI sentiment for all unique tickers in portfolio"""
            if not positions:
                return
            
            unique_tickers = list(set([pos.get('ticker', '') for pos in positions if pos.get('ticker')]))
            if not unique_tickers:
                return
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, ticker in enumerate(unique_tickers):
                status_text.text(f"🔄 Refreshing AI for {ticker}...")
                # Force refresh by clearing cache
                cache_key = f"news_sentiment_{ticker}"
                if cache_key in st.session_state.ai_cache.cache:
                    del st.session_state.ai_cache.cache[cache_key]
                # Regenerate
                get_ai_research(ticker)
                progress_bar.progress((i + 1) / len(unique_tickers))
                time.sleep(0.5)  # Rate limiting
            
            status_text.text("✅ AI refresh complete!")
            st.session_state.last_ai_refresh = datetime.now()
            time.sleep(1)
            status_text.empty()
            progress_bar.empty()
            st.rerun()
        
        def get_risk_factor_explanations():
            """Return explanations for each risk factor"""
            return {
                'Position Size': 'Larger positions relative to $50k baseline increase risk',
                'Delta Risk': 'Higher delta = more directional exposure and risk',
                'Time Left': 'Less time = higher risk of theta decay',
                'IV Risk': 'High IV means expensive options with crash risk',
                'Moneyness': 'OTM options have lower probability of profit',
                'Sentiment': 'Bearish sentiment increases risk for long calls'
            }
        
        trader_options = get_trader_list()
        selected_trader = st.selectbox("Select Trader:", trader_options, key="trader_select")
        
        # Row with Add Trader and Refresh AI buttons
        col_top1, col_top2, col_top3 = st.columns([3, 1, 1])
        with col_top2:
            if st.button("➕ Add New Trader", key="show_add_trader"):
                st.session_state.show_new_trader = True
        
        with col_top3:
            all_positions_for_refresh = get_all_positions_for_trader(selected_trader)
            if st.button("🔄 Refresh All AI Forecasts", key="refresh_ai_all"):
                refresh_all_ai_forecasts(all_positions_for_refresh)
        
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
        
        # ============================================================
        # NEW: Portfolio 5-Day Forecast Section
        # ============================================================
        st.subheader("📈 Portfolio 5-Day Forecast")
        
        portfolio_forecast = calculate_portfolio_forecast([pos for _, pos in active_positions])
        
        if portfolio_forecast and portfolio_forecast['total_current_value'] > 0:
            col_f1, col_f2, col_f3, col_f4 = st.columns(4)
            
            with col_f1:
                st.metric("Current Value", f"${portfolio_forecast['total_current_value']:,.0f}")
            with col_f2:
                expected_change = portfolio_forecast['expected_change']
                expected_color = "normal" if expected_change >= 0 else "inverse"
                st.metric("Expected 5-Day Value", f"${portfolio_forecast['total_forecast_value']:,.0f}", 
                         delta=f"{expected_change:+,.0f} ({portfolio_forecast['expected_change_pct']:+.1f}%)",
                         delta_color=expected_color)
            with col_f3:
                st.metric("Theta Decay (5d)", f"-${portfolio_forecast['total_theta_decay_5d']:,.0f}",
                         delta="Time cost")
            with col_f4:
                improving = portfolio_forecast['positions_improving']
                declining = portfolio_forecast['positions_declining']
                st.metric("Position Health", f"{improving} improving / {declining} declining")
            
            # Mini progress bar for portfolio health
            health_ratio = improving / max(portfolio_forecast['total_positions'], 1)
            st.progress(health_ratio, text=f"📊 {health_ratio*100:.0f}% of positions expected to improve")
        else:
            st.info("Add active positions to see 5-day portfolio forecast")
        
        st.divider()
        st.subheader("📊 Active Positions")
        
        col_refresh, _ = st.columns([1, 5])
        with col_refresh:
            if st.button("🔄 Refresh Prices", key="refresh_portfolio", use_container_width=True):
                st.rerun()
        
        if active_positions:
            positions_with_risk = []
            for idx, (row_idx, pos) in enumerate(active_positions):
                try:
                    strike = float(pos['strike'])
                    expiry_date = pd.to_datetime(pos['expiry']).date()
                    days_left = max((expiry_date - datetime.now().date()).days, 0)
                    
                    option_price, current_iv, gamma, theta = get_current_option_price(pos['ticker'], pos['expiry'], strike)
                    if option_price:
                        stock_price = yf.Ticker(pos['ticker']).history(period="1d")['Close'].iloc[-1]
                        d, _, _, _ = calculate_greeks(stock_price, strike, max(days_left, 1)/365, 0.05, current_iv)
                        current_delta = d
                    else:
                        current_delta = 0.5
                        current_iv = 0.35
                        gamma = 0
                        theta = 0
                    
                    sentiment_adj = st.session_state.current_sentiment.get('risk_adjustment', 0) if st.session_state.current_sentiment else 0
                    base_risk_score = calculate_risk_score(pos, option_price if option_price else 0, current_delta, days_left, current_iv, sentiment_adj)
                    beta, _ = calculate_beta(pos['ticker'])
                    risk_score, beta_factor = calculate_beta_adjusted_risk(base_risk_score * 100, beta)
                    
                    # Get AI sentiment for this ticker
                    ai_sentiment = get_ai_forecast_for_position(pos['ticker'], stock_price, strike, current_iv)
                    ai_score = ai_sentiment.get('sentiment_score', 0)
                    
                    # Calculate factor scores for breakdown
                    factor_scores = {
                        'Position Size': min((int(pos['contracts']) * float(pos['entry_price']) * 100) / 50000, 1.0),
                        'Delta Risk': 1 - min(max(current_delta, 0), 1),
                        'Time Left': 1 - min(days_left / 365, 1),
                        'IV Risk': min(current_iv * 2, 1) if current_iv else 0.5,
                        'Moneyness': max(0, 1 - (stock_price / strike)) if strike > 0 else 0.5,
                        'Sentiment': (1 - max(-1, min(1, -sentiment_adj / 50))) / 2
                    }
                    
                    positions_with_risk.append((risk_score, idx, row_idx, pos, option_price, current_iv, gamma, theta, 
                                               current_delta, days_left, stock_price, ai_score, factor_scores, beta_factor))
                except Exception as e:
                    positions_with_risk.append((50.0, idx, row_idx, pos, None, 0.35, 0, 0, 0.5, 0, None, 0, {}, 1.0))
            
            positions_with_risk.sort(key=lambda x: x[0], reverse=True)
            
            for risk_score, idx, row_idx, pos, option_price, current_iv, gamma, theta, current_delta, days_left, stock_price, ai_score, factor_scores, beta_factor in positions_with_risk:
                entry_price = float(pos['entry_price'])
                contracts = int(pos['contracts'])
                strike = float(pos['strike'])
                ticker = pos['ticker']
                expiry_date_str = pos['expiry']
                
                # Get stored targets from sheet
                stored_target = float(pos['target_price'])
                stored_stop = float(pos['stop_loss'])
                
                # DYNAMIC TARGET: Use current price if it's more realistic
                if option_price and option_price > 0:
                    # Target should be ABOVE current price for a long call
                    if stored_target > option_price:
                        target = stored_target
                    else:
                        # Calculate realistic target based on current price + expected move
                        # Options typically need 30-50% move to be profitable
                        target = option_price * 1.35  # 35% target from current
                                            
                    # Stop should be BELOW current price
                    if stored_stop < option_price:
                        stop = stored_stop
                    else:
                        stop = option_price * 0.70  # 30% stop from current
                        
                else:
                    target = stored_target
                    stop = stored_stop
                
                if option_price is None:
                    option_price, current_iv, gamma, theta = get_current_option_price(ticker, expiry_date_str, strike)
                    if option_price is None:
                        option_price = 0
                        current_iv = 0.35
                        gamma = 0
                        theta = 0
                
                risk_score_display = risk_score
                days_left = max((pd.to_datetime(expiry_date_str).date() - datetime.now().date()).days, 0)
                pnl = (option_price - entry_price) * contracts * 100 if option_price else 0
                pnl_pct = ((option_price - entry_price) / entry_price) * 100 if entry_price > 0 and option_price else 0
                
                tech_score_pos = 1
                ema_status = "neutral"
                cts = 50
                ev = 0
                delta_calc = current_delta if current_delta else 0.5
                touch_prob = 0.5
                rec_icon_full = "🔵 HOLD"
                rec_reason = "Data unavailable"
                
                if option_price and option_price > 0:
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
                    
                    # Calculate CTS and EV
                    if stock_price is None:
                        stock_price = yf.Ticker(ticker).history(period="1d")['Close'].iloc[-1]
                    
                    # Expected Value calculation for existing position
                    pot_profit = option_price * (1 + st.session_state.profit_target_pct / 100)
                    pot_loss = option_price * (st.session_state.stop_loss_pct / 100)
                    ev = (touch_prob * pot_profit) - ((1 - touch_prob) * pot_loss)
                    
                    # Composite Score
                    gt_ratio = calculate_gamma_theta_ratio(gamma, theta)
                    cts = calculate_enhanced_cts(delta_calc, touch_prob, gt_ratio, tech_score_pos)
                    
                    # Get hybrid recommendation
                    rec_icon_full, rec_reason = get_hybrid_recommendation(
                        option_price, entry_price, target, stop, days_left, delta_calc, theta, current_iv,
                        cts, ev, tech_score_pos, ema_status, "stable", 0, touch_prob, 50
                    )
                
                if risk_score_display > 70:
                    risk_indicator = "🔴 HIGH"
                elif risk_score_display > 40:
                    risk_indicator = "🟡 MEDIUM"
                else:
                    risk_indicator = "🟢 LOW"

                rec_icon = rec_icon_full.split()[0]
                # Calculate 5-day forecast for the header
                expected_price_5d = None
                expected_5d_pnl = None
                expected_5d_pnl_pct = None
                
                if option_price and option_price > 0 and stock_price and stock_price > 0:
                    try:
                        expected_price_5d, _, _, _, _, _ = forecast_5day_price(
                            option_price, stock_price, strike, delta_calc, gamma, theta, 0, current_iv, 0, 5, 0.03
                        )
                        expected_5d_pnl = (expected_price_5d - option_price) * contracts * 100
                        expected_5d_pnl_pct = ((expected_price_5d / option_price) - 1) * 100
                    except:
                        pass
                
                # Build the summary with 5-day forecast if available
                if expected_price_5d and expected_5d_pnl is not None:
                    if expected_5d_pnl >= 0:
                        pnl_5d_display = f"🟢 +${expected_5d_pnl:,.0f} (+{expected_5d_pnl_pct:+.1f}%)"
                    else:
                        pnl_5d_display = f"🔴 -${abs(expected_5d_pnl):,.0f} ({expected_5d_pnl_pct:+.1f}%)"
                    
                    summary = f"{rec_icon} {ticker} ${strike:.2f} Call | Exp: {expiry_date_str} | ${option_price:.2f} → ${expected_price_5d:.2f} (5d: {pnl_5d_display}) | P&L: {pnl_pct:+.1f}% (${pnl:+.0f}) | Risk: {risk_indicator}"
                else:
                    summary = f"{rec_icon} {ticker} ${strike:.2f} Call | Exp: {expiry_date_str} | ${option_price:.2f} | P&L: {pnl_pct:+.1f}% (${pnl:+.0f}) | Risk: {risk_indicator}"
                
                with st.expander(summary):
                    st.markdown("### 📊 Position Summary")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Current Option Price", f"${option_price:.2f}")
                        pnl_color = "inverse" if pnl < 0 else "normal"
                        st.metric("P&L", f"{pnl_pct:+.1f}%", delta=f"${pnl:+.0f}", delta_color=pnl_color)
                        st.metric("Risk Score", f"{risk_score_display:.1f}/100", help=f"Beta-adjusted: {beta_factor:.1f}x")
                    with col2:
                        st.metric("Days Left", f"{days_left}")
                        st.metric("Delta", f"{delta_calc:.3f}")
                        if gamma:
                            st.metric("Gamma", f"{gamma:.4f}")
                    with col3:
                        st.metric("Entry Price (Avg)", f"${entry_price:.2f}")
                        st.metric("Contracts", contracts)
                        if theta:
                            st.metric("Theta (daily)", f"-${abs(theta):.3f}")
                    
                    # ============================================================
                    # NEW: Risk Score Breakdown Expander
                    # ============================================================
                    with st.expander("📊 Risk Score Breakdown", expanded=False):
                        st.markdown("**6-Factor Risk Analysis**")
                        
                        risk_factors = [
                            ("Position Size", factor_scores.get('Position Size', 0.5), "Larger positions relative to $50k baseline"),
                            ("Delta Risk", factor_scores.get('Delta Risk', 0.5), "Higher delta = more directional exposure"),
                            ("Time Left", factor_scores.get('Time Left', 0.5), "Less time = higher theta decay risk"),
                            ("IV Risk", factor_scores.get('IV Risk', 0.5), "High IV = expensive options with crash risk"),
                            ("Moneyness", factor_scores.get('Moneyness', 0.5), "OTM options have lower probability"),
                            ("Sentiment", factor_scores.get('Sentiment', 0.5), "Bearish sentiment increases risk")
                        ]
                        
                        for factor_name, factor_score, explanation in risk_factors:
                            if factor_score > 0.7:
                                color = "🔴"
                            elif factor_score > 0.4:
                                color = "🟡"
                            else:
                                color = "🟢"
                            
                            st.markdown(f"**{color} {factor_name}:** {factor_score*100:.0f}/100")
                            st.caption(f"*{explanation}*")
                            st.progress(factor_score)
                        
                        st.caption(f"**Total Risk Score:** {risk_score_display:.1f}/100")
                        if risk_score_display > 70:
                            st.warning("⚠️ High risk position - consider reducing size or tightening stops")
                        elif risk_score_display > 40:
                            st.info("📊 Medium risk - normal monitoring")
                        else:
                            st.success("✅ Low risk position")
                    
                    # ============================================================
                    # NEW: 5-Day Forecast & AI Insights Expander
                    # ============================================================
                    with st.expander("📈 5-Day Forecast & AI Insights", expanded=False):
                        # Get ATR for volatility
                        try:
                            stock_hist = yf.Ticker(ticker).history(period="20d")
                            _, atr_pct = calculate_atr(stock_hist)
                        except:
                            atr_pct = 2.0  # Default 2% ATR
                        
                        # Sub-section A: Quantitative Forecast
                        st.markdown("#### 📊 Quantitative Forecast (5-Day)")
                        
                        # Calculate forecasts
                        expected_price, price_upper, price_lower, theta_decay_5d, iv_impact, leverage = forecast_5day_price(
                            option_price, stock_price, strike, delta_calc, gamma, theta, 0, current_iv, 0, 5, 0.03
                        )
                        
                        prob_target = probability_hit_target(option_price, target, 5, current_iv)
                        prob_stop = probability_hit_stop(option_price, stop, 5, current_iv)
                        
                        iv_percentile = estimate_iv_percentile(current_iv, calculate_hv(stock_hist) if not stock_hist.empty else 20)
                        
                        col_q1, col_q2, col_q3 = st.columns(3)
                        with col_q1:
                            st.metric("Expected Price (5d)", f"${expected_price:.2f}")
                            st.caption(f"80% Range: ${price_lower:.2f} - ${price_upper:.2f}")
                            st.metric("Theta Decay (5d)", f"-${theta_decay_5d:.2f}", delta=f"{-abs(theta_decay_5d/option_price)*100:.1f}% of premium")
                        with col_q2:
                            st.metric("🎯 Probability Hit Target", f"{prob_target*100:.0f}%")
                            st.metric("🛑 Probability Hit Stop", f"{prob_stop*100:.0f}%")
                        with col_q3:
                            st.metric("IV Percentile", f"{iv_percentile:.0f}th")
                            if iv_percentile > 80:
                                st.warning("⚠️ IV is expensive - earnings or event risk")
                            elif iv_percentile < 20:
                                st.success("✅ IV is cheap - good entry")
                            if iv_impact != 0:
                                st.caption(f"IV Change Impact: ${iv_impact:+.2f}")
                        
                        # Sub-section B: AI Sentiment Insights
                        st.markdown("#### 🤖 AI Sentiment Insights")
                        
                        ai_sentiment_data = get_ai_forecast_for_position(ticker, stock_price if stock_price else S, strike, current_iv)
                        ai_sentiment_score = ai_sentiment_data.get('sentiment_score', 0)
                        ai_label = ai_sentiment_data.get('sentiment_label', 'Neutral')
                        ai_themes = ai_sentiment_data.get('key_themes', ['No recent news'])[:3]
                        ai_catalyst = ai_sentiment_data.get('catalyst', None)
                        
                        col_a1, col_a2 = st.columns(2)
                        with col_a1:
                            if ai_sentiment_score > 0.3:
                                st.success(f"**Sentiment: {ai_label}**")
                            elif ai_sentiment_score < -0.3:
                                st.error(f"**Sentiment: {ai_label}**")
                            else:
                                st.info(f"**Sentiment: {ai_label}**")
                            st.metric("Sentiment Score", f"{ai_sentiment_score:+.2f}")
                            
                            if ai_catalyst:
                                st.info(f"📅 **Catalyst:** {ai_catalyst[:50]}...")
                        with col_a2:
                            st.write("**Key Themes:**")
                            for theme in ai_themes:
                                st.write(f"- {theme}")
                        
                        # Sub-section C: Combined Recommendation
                        st.markdown("#### 🎯 Combined Recommendation")
                        
                        # Get combined recommendation
                        combined_rec, combined_reason, combined_score = get_combined_recommendation(
                            cts, ai_sentiment_score, option_price, stop, days_left
                        )
                        
                        col_r1, col_r2 = st.columns([1, 2])
                        with col_r1:
                            st.markdown(f"## {combined_rec}")
                        with col_r2:
                            st.write(combined_reason)
                            st.caption(f"Combined Score: {combined_score:.0f}/100 (60% Quant + 40% AI)")
                        
                        # Expected 5-day P&L
                        expected_5d_pnl = (expected_price - option_price) * contracts * 100
                        if expected_5d_pnl > 0:
                            st.success(f"💰 Expected 5-Day P&L: +${expected_5d_pnl:,.0f}")
                        else:
                            st.warning(f"⚠️ Expected 5-Day P&L: ${expected_5d_pnl:,.0f}")
                    
                    st.markdown("---")
                    st.markdown("### 💡 Recommendation")
                    st.info(f"**{rec_icon_full}**")
                    st.caption(rec_reason)
                    
                    st.markdown("---")
                    st.markdown("### 📊 Quant Analytics")
                    q1, q2, q3 = st.columns(3)
                    with q1:
                        st.metric("Expected Value (EV)", f"${ev:.2f}" if isinstance(ev, (int, float)) else "N/A")
                        st.metric("Composite Score", f"{cts}/100")
                        st.metric("Touch Probability", f"{touch_prob*100:.0f}%")
                    with q2:
                        st.metric("IV", f"{current_iv*100:.1f}%" if current_iv else "N/A")
                        if gamma:
                            st.metric("Gamma/Theta Ratio", f"{calculate_gamma_theta_ratio(gamma, theta):.2f}")
                    with q3:
                        st.metric("Technical Score", f"{tech_score_pos}/3")
                    
                    st.markdown("---")
                    st.markdown("### 🎯 Targets")
                    col_t1, col_t2 = st.columns(2)
                    with col_t1:
                        st.metric("🛑 Stop Loss", f"${stop:.2f}")
                        if option_price and option_price > stop:
                            st.caption(f"✅ ${option_price - stop:.2f} above stop")
                        else:
                            st.caption(f"⚠️ ${stop - (option_price if option_price else 0):.2f} below stop")
                    with col_t2:
                        st.metric("🎯 Target", f"${target:.2f}")
                        if option_price and option_price < target:
                            st.caption(f"📈 Need +${target - option_price:.2f} to target")
                            st.progress(option_price / target if option_price else 0)
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
                                                    value=option_price if option_price else 0.01, key=f"add_price_{idx}")
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
                                                     value=option_price if option_price else 0.01, key=f"sell_price_{idx}")
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
            if st.session_state.last_selected_expiry and st.session_state.last_selected_expiry in expiry_options:
                default_expiry_index = expiry_options.index(st.session_state.last_selected_expiry)
            elif st.session_state.get('last_selected_expiry') in expiry_options:
                default_expiry_index = expiry_options.index(st.session_state.last_selected_expiry)
            
            selected_expiry_str = st.selectbox("Expiry Date:", options=expiry_options, index=default_expiry_index, key="portfolio_expiry_select")
            st.session_state.last_selected_expiry = selected_expiry_str
            
            def get_conservative_strike_for_expiry(ticker, expiry_date, profit_target_pct, stop_loss_pct):
                try:
                    stock_obj = yf.Ticker(ticker)
                    hist = stock_obj.history(period="100d")
                    if hist.empty:
                        return None, None
                    current_price = get_cached_current_price(ticker)
                    if current_price is None:
                        current_price = stock_obj.history(period="1d")['Close'].iloc[-1]
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

else:
    st.info("👈 Input a valid trading ticker to trigger the options matrix models.")
