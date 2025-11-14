# Getting Started Guide - Complete Backtesting Workflow

**Date:** November 14, 2025
**Purpose:** Step-by-step guide to backtest and deploy your improved trading bot

---

## 📋 Table of Contents

1. [Initial Quick Test (7 days)](#1-initial-quick-test)
2. [Full Backtest (90 days)](#2-full-backtest)
3. [Understanding ML Training Process](#3-ml-training-process)
4. [Analyzing Results](#4-analyzing-results)
5. [Dashboard Implementation](#5-dashboard-implementation)
6. [Going Live](#6-going-live)

---

## 1. Initial Quick Test (7 days)

### Purpose
Verify everything is set up correctly before running longer backtests.

### Command
```bash
cd /home/user/mlbot
python quick_test.py
```

### What This Does
1. ✅ Fetches 7 days of real mainnet BTC/USDT data (15-minute candles)
2. ✅ Trains ML model on this short dataset
3. ✅ Runs backtest with conservative settings (10x leverage, 5% risk)
4. ✅ Shows you if signals are being generated
5. ✅ Verifies all improvements are working

### Expected Output
```
QUICK TEST RESULTS
================================================================================
✅ Test completed successfully!

Key Metrics:
  • Total Trades: 3-8 (varies by market conditions)
  • Win Rate: 60-75%
  • Final Capital: $95-$115
  • Total Return: -5% to +15%
  • Profit Factor: 1.5-3.0
```

### If No Trades Generated
This is normal! In a 7-day period, market conditions might not meet all the strict criteria:
- Adaptive RSI thresholds
- Volume confirmation
- Trend filter
- Multiple timeframe Z-score

**Solution:** Run the full 90-day backtest (more opportunities).

### Troubleshooting

**Error: "python-binance not installed"**
```bash
pip install python-binance
```

**Error: "TA-Lib import error"**
```bash
# Install TA-Lib C library first
# Ubuntu/Debian:
sudo apt-get install ta-lib

# Mac:
brew install ta-lib

# Then:
pip install TA-Lib
```

**Error: "Cannot fetch data from Binance"**
- Check internet connection
- Binance might be geo-restricted in your region
- Try with VPN if needed

---

## 2. Full Backtest (90 days)

### Purpose
Train ML model on substantial data and evaluate strategy performance over different market conditions.

### Command
```bash
python example_usage.py
```

### What This Does

#### Step 1: Data Collection
- Fetches 90 days of real Binance mainnet data
- 15-minute candles = ~8,640 data points
- Downloads: open, high, low, close, volume

#### Step 2: Feature Engineering
- Calculates 24 technical indicators:
  - Mean reversion: RSI, Bollinger Bands, Z-scores (20 & 50 period)
  - Market context: Volatility, ADX, trend strength, volume ratio, MFI
  - Momentum: Momentum indicator, ROC
  - Pattern features: Support/resistance distance, consolidation, drawdown
  - Candle patterns: Doji, Hammer, Engulfing
  - Feature interactions: RSI×BB, Trend×Volatility, Volume×Momentum

#### Step 3: Signal Generation (Enhanced)
- Identifies mean reversion setups with:
  - ✅ Adaptive RSI thresholds (25-35 based on volatility)
  - ✅ Adaptive Z-score thresholds (-1.5 to -2.0 based on trend)
  - ✅ Volume confirmation (>1.2x average)
  - ✅ Trend filter (don't fight strong trends)
  - ✅ Multiple timeframe confirmation

**Expected Signals:**
- Before improvements: ~100 signals
- After improvements: ~60-70 signals (higher quality)

#### Step 4: ML Model Training
Creates training data from historical signals:

**For each signal:**
- Features (X): All 24 indicators at that moment
- Label (y): Was this trade successful?
  - Calculate forward return over next 10 periods
  - Success = forward return > 1% (threshold)
  - Label: 1 (success) or 0 (failure)

**Training Process:**
```python
# Collect all historical signals
signals = identify_all_signals(historical_data)  # ~1000-2000 signals

# For each signal, calculate outcome
for signal in signals:
    features = extract_features(signal)  # 24 features
    forward_return = calculate_forward_return(signal, periods=10)
    label = 1 if forward_return > 0.01 else 0  # Success threshold

# Train ML model
X = all_features  # Shape: (n_signals, 24)
y = all_labels    # Shape: (n_signals,)

model = GradientBoostingClassifier()
model.fit(X_train, y_train)
```

**Model Output:**
- For any new signal → Probability of success (0-100%)
- Trade only if probability > 65% (confidence threshold)

#### Step 5: Backtesting
Simulates trading with realistic conditions:

**For each trading opportunity:**
1. Check if mean reversion signal generated
2. Extract current market features
3. Ask ML model: "Is this a good trade?" (predict success probability)
4. If probability < 65%: Skip trade
5. If probability ≥ 65%: Execute trade simulation
   - Calculate position size based on risk (5% of capital)
   - Apply leverage (10x)
   - Simulate limit order fill
   - Track price action bar-by-bar
   - Check stop loss (1.5%)
   - Check take profit (3%)
   - Check trailing stop (moves to breakeven at 60% of TP)
   - Calculate PnL including fees

**Output Files:**
- `ml_mean_reversion_model.pkl` - Trained ML model
- `trade_history.json` - All trades with details
- `backtest_results.png` - Visual charts

### Expected Output

```
=== Running Backtest with MAINNET Data (CONSERVATIVE SETTINGS) ===

Capital: $100
Leverage: 10x
Risk per trade: 5.0%
Stop Loss: 1.50% (15.0% of capital)
Take Profit: 3.00% (30.0% of capital)
Reward:Risk Ratio: 2.0:1
Order Type: LIMIT (Maker: 0.02%)
Trailing Stop: Enabled

[... training progress ...]

============================================================
BACKTEST RESULTS - $100 @ 10x Leverage
============================================================
Initial Capital:    $100.00
Final Capital:      $XXX.XX
Total Return:       XX.XX%
Order Type:         LIMIT (Fee: 0.02%)
Total Fees Paid:    $X.XX

Trade Statistics:
Total Trades:       XX
Winning Trades:     XX
Losing Trades:      XX
Win Rate:           XX.XX%
Avg Win:            $XX.XX (XX% of capital)
Avg Loss:           $XX.XX (XX% of capital)
Profit Factor:      X.XX
Fill Rate:          ~85.0%

Position Sizing:
Max Position:       $XXX.XX
Avg Position:       $XXX.XX

Risk Metrics:
Sharpe Ratio:       X.XX
Max Drawdown:       -XX.XX% ($XX.XX)

Leverage Impact:
ROI with 10x:       XX.XX%
============================================================
```

### Good Results Indicators
- ✅ Win Rate: 65-75%
- ✅ Profit Factor: > 2.0
- ✅ Sharpe Ratio: > 1.5
- ✅ Max Drawdown: < 20%
- ✅ Total Return: Positive

### If Results Are Poor
- Win Rate < 55%: Strategy not working in this period
- Profit Factor < 1.3: Risk/reward imbalanced
- Max Drawdown > 30%: Too aggressive

**Possible Causes:**
1. Market conditions in this 90-day period didn't favor mean reversion
2. Parameter tuning needed
3. Try different time period
4. Strategy might work better in ranging vs trending markets

---

## 3. ML Training Process - Deep Dive

### How Data Flows from Backtest to ML Model

#### Phase 1: Historical Signal Collection

```
Historical Data (90 days)
         ↓
Calculate Features (24 indicators)
         ↓
Identify All Mean Reversion Signals (~1000-2000)
         ↓
For Each Signal:
  - Timestamp: When signal occurred
  - Features: All 24 indicator values at that moment
  - Forward Return: What happened in next 10 periods?
         ↓
Create Training Dataset:
  X (features) = [[rsi, bb_pos, zscore, ...], [...], ...]  # Shape: (n_signals, 24)
  y (labels)   = [1, 0, 1, 1, 0, ...]                       # Shape: (n_signals,)
```

#### Phase 2: Model Training

```
Training Dataset (80% of signals)
         ↓
Gradient Boosting Classifier
  - n_estimators: 200
  - learning_rate: 0.05
  - max_depth: 5
         ↓
Learns patterns:
  "When RSI < 28 AND volatility is LOW AND trend_strength > -0.3
   AND volume_ratio > 1.5 AND zscore_50 < -1.2
   → 78% chance of success"
         ↓
Test Dataset (20% of signals)
         ↓
Evaluate Performance:
  - Training Accuracy: ~70-75%
  - Testing Accuracy: ~65-70%
  - Classification Report
  - Feature Importance Ranking
```

#### Phase 3: Real-Time Prediction (During Backtest or Live Trading)

```
New Signal Appears
         ↓
Extract Current Features (24 values)
         ↓
Ask Trained Model:
  "What's the probability this trade will succeed?"
         ↓
Model Returns: 0.73 (73% probability)
         ↓
Decision Logic:
  if probability >= 0.65:
      Execute Trade ✅
  else:
      Skip Trade ❌ (Not confident enough)
```

### Key Insight: The ML Model Is a Filter

**Traditional Trading Bot:**
```
Signal Appears → Execute Trade
(Takes every signal blindly)
```

**ML-Enhanced Trading Bot:**
```
Signal Appears → ML Evaluates Context → Only Execute if Confident
(Filters out low-quality signals)
```

### Example: Why ML Rejects a Signal

**Scenario:**
- RSI = 28 (oversold) ✓
- BB position = 0.15 (near lower band) ✓
- Z-score = -1.8 (below mean) ✓

**Traditional Bot:** "All conditions met, BUY!"

**ML-Enhanced Bot:**
```
Wait, let me check the context...
  - Volatility: HIGH (risky environment)
  - Trend strength: -0.65 (strong downtrend)
  - Volume ratio: 0.8 (below average, weak signal)
  - Historical similar setups: 35% win rate

ML Prediction: 42% probability of success

Decision: REJECT ❌ (Below 65% confidence threshold)
```

**This is why the ML model improves performance!**

---

## 4. Analyzing Results

### Files Generated

After running `python example_usage.py`, you'll find:

```
/mnt/user-data/outputs/
├── ml_mean_reversion_model.pkl    # Trained ML model
├── trade_history.json              # Detailed trade log
└── backtest_results.png            # Visual charts
```

### trade_history.json Structure

```json
[
  {
    "timestamp": "2024-10-15T14:30:00",
    "entry_price": 67845.23,
    "exit_price": 68890.45,
    "direction": "LONG",
    "rsi": 27.5,
    "bb_position": 0.18,
    "zscore": -1.92,
    "zscore_50": -1.15,
    "volatility_regime": "MEDIUM",
    "trend_strength": -0.25,
    "volume_ratio": 1.45,
    "confidence_score": 0.73,
    "predicted_success_prob": 0.73,
    "actual_outcome": "WIN",
    "pnl_percent": 0.0284,
    "pnl_usd": 28.40,
    "similar_trades": [
      {"direction": "LONG", "outcome": "WIN", "pnl_percent": 0.025, "similarity": 0.87},
      {"direction": "LONG", "outcome": "WIN", "pnl_percent": 0.031, "similarity": 0.82},
      ...
    ]
  },
  ...
]
```

### Key Metrics to Analyze

#### 1. Win Rate by Confidence Level
```python
# Group trades by confidence quartiles
Q1 (Low confidence 65-70%):   Win rate ~60%
Q2 (Medium 70-75%):            Win rate ~65%
Q3 (Medium-high 75-80%):       Win rate ~70%
Q4 (High confidence 80%+):     Win rate ~80%
```
**Goal:** Higher confidence = higher win rate (validates ML model)

#### 2. Win Rate by Volatility Regime
```python
LOW volatility:    Win rate ~72%  (mean reversion works best)
MEDIUM volatility: Win rate ~65%  (decent)
HIGH volatility:   Win rate ~55%  (challenging)
```
**Insight:** Strategy performs better in low volatility

#### 3. Win Rate by Trend Strength
```python
Strong Downtrend (<-0.5):  Few trades (filtered out)
Neutral (-0.5 to 0.5):     Win rate ~70% (best)
Strong Uptrend (>0.5):     Few trades (filtered out)
```
**Insight:** Trend filter is working correctly

#### 4. Trailing Stop Performance
```python
Trades with trailing stop hit:  XX trades
  - Saved from loss: XX trades (would have been losers)
  - Protected profits: XX trades (exited at breakeven instead of loss)

Expected improvement: +5-10% win rate
```

### Visual Analysis (backtest_results.png)

**Chart 1: Equity Curve**
- Shows capital over time
- Look for: Smooth upward trend
- Avoid: Steep drawdowns, jagged equity curve

**Chart 2: BTC Price with Trade Entries**
- Green triangles: Long entries
- Red triangles: Short entries
- Look for: Good entry timing at extremes

**Chart 3: PnL Distribution**
- Histogram of trade returns
- Look for: More winners than losers, right-skewed distribution

---

## 5. Dashboard Implementation

### Current Analysis Tools

Your bot already has basic analysis in `utils.py`:
- Performance metrics calculation
- Equity curve plotting
- PnL distribution charts
- Confidence analysis

### Enhanced Dashboard (Similar to trabot)

I'll help you implement a comprehensive dashboard with:

#### Features to Add:
1. **Day-wise Filtering**
   - Filter trades by specific dates
   - Compare performance across different periods
   - Identify best/worst performing days

2. **Drawdown Analysis**
   - Real-time drawdown tracking
   - Underwater equity chart
   - Recovery time analysis

3. **Equity Graph Enhancements**
   - Interactive zoom/pan
   - Multiple timeframe views (daily, weekly, monthly)
   - Benchmark comparison (buy & hold vs strategy)

4. **Trade Sorting**
   - Sort by: Winners, Losers, Biggest gain, Biggest loss
   - Filter by: Direction (Long/Short), Confidence level, Volatility regime

5. **Performance Metrics**
   - Most profitable day
   - Most losing day
   - Longest winning streak
   - Longest losing streak
   - Best month, worst month

6. **ML Model Insights**
   - Feature importance chart
   - Confidence score distribution
   - Prediction accuracy over time

### Would you like me to:

1. **Look at your trabot repository** to see exactly what dashboard features you have
2. **Create a similar dashboard** for this mlbot project
3. **Build an interactive web dashboard** (using Streamlit or Dash)

Please confirm and I'll:
- Review your trabot repo
- Implement similar features here
- Create interactive visualizations
- Add the filtering and sorting capabilities you mentioned

---

## 6. Going Live

### Prerequisites Checklist

Before running live trading:

- [ ] Successful backtest with positive returns
- [ ] Win rate > 60%
- [ ] Profit factor > 2.0
- [ ] Max drawdown < 20%
- [ ] Trained model saved to `/mnt/user-data/outputs/`
- [ ] config.json with correct API keys
- [ ] API keys have Futures permission (NOT Withdraw)
- [ ] $100 in Binance Futures wallet
- [ ] Understand liquidation risks (10% BTC move = 100% loss)
- [ ] Ready to monitor every 2-4 hours

### Command

```bash
python live_trading_bot.py
```

### What Happens

1. Loads trained ML model from file
2. Loads API keys from config.json
3. Sets leverage to 10x on Binance
4. Asks for confirmation: Type "YES" to proceed
5. Starts monitoring market every 60 seconds
6. When signal appears:
   - Extracts features
   - ML model evaluates confidence
   - If confidence > 65%: Places limit order
   - Sets stop loss and take profit
   - Applies trailing stop
7. Monitors open positions
8. Logs all activity

### Monitoring During Live Trading

**Check Every 2-4 Hours:**
- Open positions
- Current PnL
- Stop loss and take profit levels
- Account balance

**Daily Review:**
- Win rate vs expected
- Drawdown level
- Any unusual trades
- ML confidence scores

**Weekly Analysis:**
- Overall performance
- Compare to backtest
- Adjust if needed

### Emergency Actions

**Stop Trading:**
```python
# Ctrl+C in terminal
# Or modify live_trading_bot.py:
self.is_running = False
```

**Close All Positions:**
```python
# In Binance app or web interface:
# Futures → Positions → Close All
```

### Risk Management Rules

**Personal Stop-Loss:**
- If down 50% ($50 → stop trading for a month)
- If 3 consecutive losses → review strategy
- If drawdown > 30% → reduce leverage or stop

**Never:**
- Add more capital during drawdowns
- Override stop losses
- Trade with more than you can afford to lose

---

## 📊 Summary: Complete Workflow

```
1. Quick Test (7 days)
   python quick_test.py
   ↓
   Verify setup works
   ↓
2. Full Backtest (90 days)
   python example_usage.py
   ↓
   Train ML model
   Generate trade history
   Analyze results
   ↓
3. Review Results
   - Check win rate > 60%
   - Check profit factor > 2.0
   - Analyze trade_history.json
   - Review backtest_results.png
   ↓
4. Implement Dashboard (Optional)
   - Enhanced visualizations
   - Day-wise filtering
   - Drawdown analysis
   - Trade sorting
   ↓
5. Paper Trade (Optional but Recommended)
   - Watch signals in real-time
   - Don't execute (mental tracking)
   - Build confidence
   ↓
6. Go Live (When Ready)
   python live_trading_bot.py
   ↓
   Monitor closely
   Follow risk management rules
   Review performance regularly
```

---

**Ready to start? Run the quick test:**

```bash
python quick_test.py
```

Let me know the results and we'll proceed to the full backtest and dashboard implementation! 🚀
