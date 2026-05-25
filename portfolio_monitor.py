#!/usr/bin/env python3
"""
Options Portfolio Monitor - Runs via GitHub Actions
Checks all active positions and sends email alerts for recommendation changes.
"""

import os
import sys
import json
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from scipy.stats import norm
import yfinance as yf
import gspread
from google.oauth2.service_account import Credentials

# ========== CONFIGURATION ==========
TRADING_START_HOUR = 9
TRADING_END_HOUR = 16
TRADING_TIMEZONE = "US/Eastern"

# ========== CORE FUNCTIONS (copied from app) ==========
def calculate_greeks(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0.5, 0, 0, 0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    delta = norm.cdf(d1)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    theta = (- (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    vega = (S * norm.pdf(d1) * np.sqrt(T)) / 100
    return round(delta, 3), round(gamma, 4), round(theta, 3), round(vega, 3)

def calculate_p_touch(S, K, T, sigma):
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0.0
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
    return df.iloc[-1], df.iloc[-2]

def get_macd_trend(hist_values, days=3):
    if len(hist_values) < days:
        return "stable", 0
    recent = hist_values[-days:]
    if all(recent[i] > recent[i-1] for i in range(1, len(recent))):
        return "rising", days
    elif all(recent[i] < recent[i-1] for i in range(1, len(recent))):
        return "falling", days
    return "stable", 0

def get_ema_cross(curr_ema8, curr_ema20, prev_ema8, prev_ema20):
    if curr_ema8 > curr_ema20:
        if prev_ema8 <= prev_ema20:
            return "bullish_cross"
        return "bullish"
    elif curr_ema8 < curr_ema20:
        if prev_ema8 >= prev_ema20:
            return "bearish_cross"
        return "bearish"
    return "neutral"

def get_hybrid_recommendation(option_price, entry_price, target, stop_loss, 
                               days_left, current_delta, current_theta, current_iv,
                               cts, ev, tech_score, ema_status, macd_trend, macd_days,
                               touch_prob, iv_percentile=None):
    pnl_pct = ((option_price - entry_price) / entry_price) * 100 if entry_price > 0 else 0
    
    if option_price <= stop_loss:
        return "🔴 SELL IMMEDIATELY", f"Stop loss hit at ${stop_loss:.2f}"
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

# ========== GOOGLE SHEETS CONNECTION ==========
def get_google_sheet():
    try:
        creds_json = os.environ.get("GOOGLE_SHEETS_CREDENTIALS")
        if not creds_json:
            print("ERROR: GOOGLE_SHEETS_CREDENTIALS not found in environment")
            return None
        creds_dict = json.loads(creds_json)
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds = Credentials.from_service_account_info(creds_dict, scopes=scope)
        client = gspread.authorize(creds)
        sheet_id = os.environ.get("SPREADSHEET_ID")
        sheet = client.open_by_key(sheet_id)
        return sheet
    except Exception as e:
        print(f"ERROR connecting to Google Sheets: {str(e)}")
        return None

def get_traders_with_email():
    """Get all enabled traders with their emails."""
    sheet = get_google_sheet()
    if not sheet:
        return []
    try:
        traders_worksheet = sheet.worksheet("Traders")
        records = traders_worksheet.get_all_records()
        traders = []
        for record in records:
            if record.get('enabled') == "TRUE" and record.get('email'):
                traders.append({
                    'name': record.get('trader_name'),
                    'email': record.get('email')
                })
        return traders
    except:
        return []

def get_active_positions_for_trader(trader_name):
    """Get all active positions for a specific trader."""
    sheet = get_google_sheet()
    if not sheet:
        return []
    try:
        portfolio = sheet.worksheet("Portfolio")
        records = portfolio.get_all_records()
        positions = []
        for idx, record in enumerate(records):
            if record.get('status') == 'active' and record.get('trader_name') == trader_name:
                positions.append((idx, record))
        return positions
    except:
        return []

def update_last_alert(row_index, recommendation):
    """Update the last_alert_sent and last_recommendation in sheet."""
    sheet = get_google_sheet()
    if not sheet:
        return
    try:
        portfolio = sheet.worksheet("Portfolio")
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        portfolio.update_cell(row_index + 2, 14, recommendation)  # last_recommendation
        portfolio.update_cell(row_index + 2, 15, now)  # last_alert_sent
    except Exception as e:
        print(f"Error updating alert timestamp: {str(e)}")

def get_last_recommendation(row_index):
    """Get the last recommendation sent for a position."""
    sheet = get_google_sheet()
    if not sheet:
        return None
    try:
        portfolio = sheet.worksheet("Portfolio")
        return portfolio.cell(row_index + 2, 14).value
    except:
        return None

# ========== EMAIL FUNCTIONS ==========
def send_digest_email(to_email, trader_name, changes):
    """Send a digest email with all changes for a trader."""
    if not changes:
        return False
    
    # Email configuration
    smtp_user = os.environ.get("GMAIL_USER")
    smtp_password = os.environ.get("GMAIL_APP_PASSWORD")
    
    if not smtp_user or not smtp_password:
        print(f"Email credentials not configured for {trader_name}")
        return False
    
    # Build HTML table
    html_rows = ""
    for change in changes:
        html_rows += f"""
        <tr style="border-bottom: 1px solid #ddd;">
            <td style="padding: 8px; text-align: left;"><b>{change['position']}</b></td>
            <td style="padding: 8px; text-align: center;">{change['old_rec']}</td>
            <td style="padding: 8px; text-align: center;">{change['new_rec']}</td>
            <td style="padding: 8px; text-align: left;">{change['reason']}</td>
        </tr>
        """
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; }}
            .header {{ background-color: #1a73e8; color: white; padding: 10px; text-align: center; }}
            .summary {{ background-color: #f0f0f0; padding: 10px; margin: 10px 0; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th {{ background-color: #4CAF50; color: white; padding: 10px; text-align: center; }}
            td {{ padding: 8px; }}
            .footer {{ font-size: 12px; color: #666; text-align: center; margin-top: 20px; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h2>📊 Options Portfolio Alert - {trader_name}</h2>
        </div>
        
        <div class="summary">
            <b>Alert Summary:</b> {len(changes)} position(s) changed<br>
            <b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S ET')}<br>
            <b>Action Required:</b> Review positions below
        </div>
        
        <table>
            <thead>
                <tr>
                    <th>Position</th>
                    <th>Previous</th>
                    <th>Current</th>
                    <th>Reason</th>
                </tr>
            </thead>
            <tbody>
                {html_rows}
            </tbody>
        </table>
        
        <div class="footer">
            <p>💡 Open your Portfolio Dashboard to take action.</p>
            <p>This is an automated alert from your Options Portfolio Tracker.</p>
        </div>
    </body>
    </html>
    """
    
    # Send email
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"📊 Options Alert - {len(changes)} position(s) changed for {trader_name}"
        msg["From"] = smtp_user
        msg["To"] = to_email
        
        html_part = MIMEText(html_content, "html")
        msg.attach(html_part)
        
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(smtp_user, smtp_password)
            server.sendmail(smtp_user, to_email, msg.as_string())
        
        print(f"Alert sent to {to_email} for {trader_name} ({len(changes)} changes)")
        return True
    except Exception as e:
        print(f"Failed to send email to {to_email}: {str(e)}")
        return False

# ========== MAIN MONITORING FUNCTION ==========
def monitor_positions():
    """Main function to check all positions and send alerts."""
    print(f"{datetime.now()}: Starting portfolio monitoring...")
    
    # Get all traders with emails
    traders = get_traders_with_email()
    if not traders:
        print("No traders with email configured. Exiting.")
        return
    
    total_alerts_sent = 0
    
    for trader in traders:
        trader_name = trader['name']
        trader_email = trader['email']
        
        print(f"Checking positions for {trader_name} ({trader_email})...")
        
        positions = get_active_positions_for_trader(trader_name)
        if not positions:
            print(f"  No active positions for {trader_name}")
            continue
        
        changes = []
        
        for row_idx, pos in positions:
            ticker = pos['ticker']
            strike = float(pos['strike'])
            expiry = pos['expiry']
            entry_price = float(pos['entry_price'])
            target = float(pos['target_price'])
            stop = float(pos['stop_loss'])
            
            # Get current option price
            option_price, current_iv = get_current_option_price(ticker, expiry, strike)
            if not option_price:
                continue
            
            days_left = max((pd.to_datetime(expiry).date() - datetime.now().date()).days, 0)
            
            # Get technicals
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period="60d")
                if not hist.empty:
                    curr, prev = get_technicals(hist)
                    tech_score = 0
                    if curr['ema8'] > curr['ema20']:
                        tech_score += 1
                    if curr['hist'] > prev['hist']:
                        tech_score += 1
                    if stock.history(period="1d")['Close'].iloc[-1] > curr['sma20']:
                        tech_score += 1
                    hist_vals = hist['hist'].dropna().values
                    macd_trend, macd_days = get_macd_trend(hist_vals, 3)
                    ema_status = get_ema_cross(curr['ema8'], curr['ema20'], prev['ema8'], prev['ema20'])
                else:
                    tech_score = 1
                    macd_trend, macd_days = "stable", 0
                    ema_status = "neutral"
            except:
                tech_score = 1
                macd_trend, macd_days = "stable", 0
                ema_status = "neutral"
            
            # Calculate metrics
            current_stock_price = yf.Ticker(ticker).history(period="1d")['Close'].iloc[-1]
            d, _, _, _ = calculate_greeks(current_stock_price, strike, max(days_left, 1)/365, 0.05, current_iv if current_iv else 0.35)
            p_touch = calculate_p_touch(current_stock_price, strike, max(days_left, 1)/365, current_iv if current_iv else 0.35)
            cts = int(((d * 0.4) + (p_touch * 0.4) + (tech_score / 3.0 * 0.2)) * 100)
            ev = (p_touch * (1 + 0.4)) - ((1 - p_touch) * (0.3))
            
            # Get current recommendation
            new_rec, reason = get_hybrid_recommendation(
                option_price, entry_price, target, stop, days_left, d, 0, current_iv,
                cts, ev, tech_score, ema_status, macd_trend, macd_days, p_touch, 50
            )
            
            # Get last recommendation
            old_rec = get_last_recommendation(row_idx)
            
            # Only alert if recommendation changed AND it's an actionable change
            if old_rec and old_rec != new_rec:
                actionable_changes = ["🔴 SELL IMMEDIATELY", "🔴 EXIT - NO EDGE", "🔴 SELL VOL", 
                                       "🟢 TAKE PROFITS", "🟡 PARTIAL EXIT", "🟠 TIME DECAY",
                                       "🟠 TECHNICAL EXIT", "🟢 ADD MORE"]
                
                should_alert = any(change in new_rec for change in actionable_changes)
                
                if should_alert:
                    changes.append({
                        'position': f"{ticker} ${strike:.2f} Call",
                        'old_rec': old_rec,
                        'new_rec': new_rec,
                        'reason': reason[:50] + "..." if len(reason) > 50 else reason
                    })
                    # Update the sheet
                    update_last_alert(row_idx, new_rec)
                    print(f"  Change detected: {ticker} ${strike:.2f} Call: {old_rec} → {new_rec}")
        
        # Send digest email if there are changes
        if changes:
            if send_digest_email(trader_email, trader_name, changes):
                total_alerts_sent += len(changes)
                print(f"  Sent digest to {trader_email} ({len(changes)} changes)")
        else:
            print(f"  No actionable changes for {trader_name}")
    
    print(f"Monitoring complete. Total alerts sent: {total_alerts_sent}")

if __name__ == "__main__":
    # Check if within trading hours
    now = datetime.now()
    current_hour = now.hour
    
    # Simple check - runs Monday-Friday during trading hours
    if TRADING_START_HOUR <= current_hour < TRADING_END_HOUR and now.weekday() < 5:
        print(f"Trading hours detected. Running monitor...")
        monitor_positions()
    else:
        print(f"Outside trading hours ({TRADING_START_HOUR}:00-{TRADING_END_HOUR}:00 ET) or weekend. Skipping.")
