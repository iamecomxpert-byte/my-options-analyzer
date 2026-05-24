
**Real-world example:**
- Bid: $1.00, Ask: $1.10, Mid: $1.05
- Spread % = ($0.10 / $1.05) × 100 = 9.5%
- You lose 9.5% immediately upon entry (buy at $1.10, could only sell at $1.00)

**How to use in trading:**
- **NEVER** enter with spread > 10%
- **ACCEPTABLE** for longer holds (30+ days) if spread < 5%
- **REQUIRE** spread < 3% for short-term trades (<14 days)
- Use limit orders, never market orders on wide spreads
- Calculate your breakeven including spread cost
""")

with st.expander("🎯 Combined Liquidity Filter - The Rule of 3", expanded=False):
st.markdown("""
**The Rule of 3 Liquidity Check:**

Before trading ANY option, verify all three conditions:

1. **Volume > 50** (preferably > 200)
2. **Open Interest > 500** (preferably > 1,000)
3. **Spread < 5%** (preferably < 3%)

**If any condition fails:** Move to the next strike or expiry.

**Why this matters:** 
Illiquid options have hidden costs that destroy edge. A mathematically perfect trade with 10% spread is actually a losing trade.

**Example:**
- Mathematical EV: +$50
- Spread cost (10% on $500 trade): -$50
- Real EV: $0 → No edge, just paying the market maker
""")

st.divider()

# SECTION 2: CONVICTION INDICATORS
st.subheader("🎯 Conviction Indicators - Your Decision Framework")

with st.expander("📊 Composite Score (CTS) - The Single Decision Number", expanded=False):
st.markdown("""
**What it measures:** A weighted combination of Delta (40%), Touch Probability (40%), and Technical Score (20%).

**Why it matters:** One number that summarizes the trade's mathematical and technical strength.

**Guidelines for Use:**
| Score | Rating | Action |
|-------|--------|--------|
| 70-100 | 🟢 STRONG BUY | High conviction. Size position normally. |
| 55-69 | 🟡 MODERATE BUY | Positive edge. Consider smaller position. |
| 40-54 | 🟠 WEAK EDGE | Small or negative edge. Size down significantly. |
| <40 | 🔴 AVOID | Negative expectancy. Skip entirely. |

**How to use in trading:**
- Use CTS as your primary filter
- Combine with liquidity check: CTS > 55 + Liquidity passes = Trade
- For spreads/iron condors, require CTS > 60
""")

with st.expander("💰 Expected Value (EV) - Long-Term Profitability", expanded=False):
st.markdown("""
**What it measures:** (Touch Probability × Take Profit Value) - (Loss Probability × Stop Loss Value)

**Why it matters:** Positive EV means the math favors you over many trades.

**Guidelines for Use:**
- **EV > 0.50** → Strong edge, trade with confidence
- **EV > 0.25** → Moderate edge, trade with normal size
- **EV > 0** → Small edge, size down or monitor
- **EV < 0** → Negative edge, DO NOT TRADE regardless of other metrics

**Note:** EV must be evaluated AFTER accounting for spread costs.
`Real EV = Mathematical EV - Spread Cost`
""")

st.divider()

# SECTION 3: GREEKS
st.subheader("📊 Greeks - Understanding Your Risk Exposures")

with st.expander("🎲 Delta (Δ) - Probability of Profit", expanded=False):
st.markdown("""
**What it measures:** Estimated probability the option expires in-the-money. Also measures price sensitivity ($0.01 move in stock = Δ × $0.01 change in option).

**Guidelines by Strategy:**
| Strategy Type | Target Delta | Risk Level |
|---------------|--------------|------------|
| Conservative | 0.50 - 0.60 | Lower risk, higher probability |
| Aggressive | 0.40 - 0.49 | Moderate risk, leveraged |
| Speculative | 0.30 - 0.39 | High risk, lottery-like |

**How to use in trading:**
- Match delta to your risk tolerance
- Lower delta = cheaper option, less probability
- Higher delta = more expensive, higher probability
- As delta drops below 0.30, options behave like lottery tickets
""")

with st.expander("⏱️ Theta (θ) - Time Decay Cost", expanded=False):
st.markdown("""
**What it measures:** Daily premium erosion from time passing.

**Why it matters:** Theta is your daily cost of holding the option. High theta kills long positions.

**Guidelines for Use:**
- **Daily theta as % of premium:**
    - < 1% per day → Manageable for long holds
    - 1-2% per day → Standard for 30-45 DTE
    - 2-3% per day → Expensive, needs fast move
    - > 3% per day → Too expensive, look for longer expiry

**How to use in trading:**
- Calculate: `Theta % = (Theta × 365) / Premium`
- For longer holds (30+ days), prefer theta < 1.5% of premium
- Set hold duration based on theta: exit before theta accelerates (last 14 days)
""")

with st.expander("📈 Vega (ν) - Volatility Exposure", expanded=False):
st.markdown("""
**What it measures:** Option price change for 1% change in implied volatility (IV).

**Why it matters:** Vega tells you how much you're betting on volatility expansion.

**Guidelines for Use:**
- **High Vega (>0.10)** → Betting on volatility increase (earnings, events)
- **Low Vega (<0.05)** → Volatility movement won't affect you much

**How to use in trading:**
- Before earnings: Expect high Vega, IV is inflated (option expensive)
- Buy Vega when IV is low (30th percentile or lower)
- Avoid buying Vega when IV is high (70th percentile+)
""")

with st.expander("📐 Gamma (Γ) - Acceleration Risk", expanded=False):
st.markdown("""
**What it measures:** Rate of change of Delta. How fast your position's probability changes as stock moves.

**Why it matters:** High gamma means your position can swing dramatically.

**Guidelines for Use:**
- **Near-term options (<21 DTE)** → Very high gamma, risky
- **Longer-term options (>60 DTE)** → Lower gamma, more stable
- **At-the-money options** → Highest gamma

**How to use in trading:**
- Our framework requires 60+ DTE to keep gamma manageable
- Avoid high gamma unless you can monitor constantly
""")

st.divider()

# SECTION 4: PROBABILITY METRICS
st.subheader("🎲 Probability Metrics - Understanding Your Odds")

with st.expander("📐 Path Touch Probability (P_touch)", expanded=False):
st.markdown("""
**What it measures:** Probability the stock touches your strike price at least once before expiration.

**Why it matters:** Touch probability is usually double Delta. It tells you your chance of hitting profit target early.

**Guidelines for Use:**
- **P_touch > 60%** → High chance to hit strike, good for taking profits
- **P_touch 40-60%** → Moderate chance, requires patience
- **P_touch < 40%** → Unlikely to touch, set realistic expectations

**How to use in trading:**
- Use P_touch to set exit strategy: if P_touch is high, plan to exit at touch
- Combine with profit target: If P_touch > 60%, consider taking partial profits at touch
""")

with st.expander("🎯 Delta Proxy (P_ITM)", expanded=False):
st.markdown("""
**What it measures:** Estimated probability option expires in-the-money.

**Why it matters:** Unlike P_touch (touches any time), P_ITM only counts if you hold to expiration.

**How to use in trading:**
- Don't hold to expiration unless P_ITM > 70%
- Exit when Delta drops below 0.25 (probability has deteriorated)
- Our framework's exit is based on time (40% of days left) or profit target, not expiration
""")

st.divider()

# SECTION 5: TECHNICAL INDICATORS
st.subheader("📈 Technical Indicators - Timing Your Entry")

with st.expander("📊 8/20 EMA Crossover", expanded=False):
st.markdown("""
**What it measures:** Short-term (8-day) vs medium-term (20-day) moving averages.

**Why it matters:** Tells you which time frame has control.

**How to use in trading:**
- Only buy calls when 8 EMA > 20 EMA
- Exit when 8 EMA crosses below 20 EMA (momentum loss)
- Look for widening gap between EMAs = strengthening trend
""")

with st.expander("📊 MACD Histogram", expanded=False):
st.markdown("""
**What it measures:** Difference between MACD line and signal line. Shows momentum acceleration/deceleration.

**Why it matters:** Detects whether buying/selling pressure is increasing.

**How to use in trading:**
- Prefer trades when histogram is rising
- Exit if histogram falls significantly (momentum loss)
- Combine with EMA crossover for confirmation
""")

with st.expander("📊 Bollinger Bands", expanded=False):
st.markdown("""
**What it measures:** Volatility boundaries (20-day SMA ± 2 standard deviations).

**Why it matters:** Shows if price is overextended or has room to run.

**How to use in trading:**
- Best entries: Price near or below middle band (SMA 20)
- Avoid chasing price above upper band (overextended)
- Band width: Widening bands = increasing volatility, tightening = decreasing
""")

st.divider()

# SECTION 6: COMPLETE TRADE DECISION FRAMEWORK
st.subheader("✅ Complete Trade Decision Framework - Step by Step")
st.markdown("""
Use this exact checklist before entering ANY trade recommended by the app:

### Step 1: Liquidity Filter (MANDATORY)
- [ ] Volume > 50 (preferably > 200)
- [ ] Open Interest > 500 (preferably > 1,000)
- [ ] Spread < 5% (preferably < 3%)

**If any fail:** Move to next strike.

### Step 2: Strategy Match
- [ ] Does Delta match your risk profile? (Conservative: 0.50-0.60, Aggressive: 0.40-0.49, Speculative: 0.30-0.39)
- [ ] Days to expiry > 60 (our framework requirement)

### Step 3: Math Confirmation
- [ ] Composite Score > 55 (for normal position size)
- [ ] Expected Value > 0.25 (after accounting for spread)

### Step 4: Technical Confirmation
- [ ] 8 EMA > 20 EMA (bullish alignment)
- [ ] MACD histogram rising (momentum building)
- [ ] Price not above upper Bollinger Band (not overextended)

### Step 5: Trade Management Plan
- [ ] Set take profit at 40% of premium price
- [ ] Set stop loss at 30% of premium price
- [ ] Maximum hold = 40% of days to expiry (or 45 days, whichever smaller)
- [ ] Exit date = Calendar date shown in recommendation

**If all checks pass:** Enter the trade with confidence.
**If 2+ checks fail:** Skip the trade, wait for better setup.
""")

st.warning("""
**⚠️ Remember:** No indicator is perfect. The framework increases your odds but cannot guarantee profits.
Always size positions appropriately (never risk more than 1-2% of account per trade) and follow your stop losses.
""")

st.success("""
**💡 Pro Tip:** The best trades happen when ALL four filters pass:
1. ✅ Liquidity (Volume + OI + Spread)
2. ✅ Math (CTS + EV)
3. ✅ Technicals (EMAs + MACD + Bollinger)
4. ✅ Management (Pre-set exits)

When all four align, the probability of success is maximized.
""")
else:
st.info("👈 Input a valid trading ticker to trigger the options matrix models.")
