# CLI Backtest Tool - Quick Reference

The new `run_backtest.py` provides a professional command-line interface with progress indicators and customizable parameters.

## 🚀 Quick Start

### Basic Commands

```bash
# Quick 3-day test (fast, good for testing)
python run_backtest.py --days 3

# Standard 90-day backtest (recommended)
python run_backtest.py --days 90

# One week backtest
python run_backtest.py --days 7
```

### Custom Date Range (dd/mm/yyyy format)

```bash
# Specific date range
python run_backtest.py --start 01/11/2024 --end 14/11/2024

# Different date format (2-digit year)
python run_backtest.py --start 01/11/24 --end 14/11/24
```

## 📊 Advanced Options

### Custom Trading Parameters

```bash
# Lower leverage (safer)
python run_backtest.py --days 30 --leverage 5

# Higher capital
python run_backtest.py --days 90 --capital 500

# More aggressive risk
python run_backtest.py --days 60 --risk 0.10 --leverage 15

# Wider stops
python run_backtest.py --days 30 --stop-loss 0.02 --take-profit 0.04
```

### ML Training Options

```bash
# Skip ML training (use existing model)
python run_backtest.py --days 7 --no-train

# Different forward period for labels
python run_backtest.py --days 90 --forward-periods 15
```

### Order Type Options

```bash
# Use market orders (faster fill, higher fees)
python run_backtest.py --days 30 --use-market

# Disable trailing stop
python run_backtest.py --days 30 --no-trailing
```

### Different Symbol/Interval

```bash
# ETH instead of BTC
python run_backtest.py --days 30 --symbol ETHUSDT

# 5-minute candles
python run_backtest.py --days 7 --interval 5m

# 1-hour candles
python run_backtest.py --days 90 --interval 1h
```

## 📁 Output Control

```bash
# Don't save results
python run_backtest.py --days 7 --no-save

# Custom output directory
python run_backtest.py --days 30 --output-dir /path/to/output
```

## 🎯 Common Use Cases

### 1. Quick Strategy Validation

```bash
# Fast test to see if strategy works at all
python run_backtest.py --days 3
```

**Expected time:** 1-2 minutes
**Use for:** Quick validation before longer backtests

### 2. Standard Production Backtest

```bash
# Recommended for actual deployment
python run_backtest.py --days 90
```

**Expected time:** 5-10 minutes
**Use for:** Full strategy evaluation before going live

### 3. Conservative Settings

```bash
# Safer parameters for cautious traders
python run_backtest.py --days 90 --leverage 5 --risk 0.02 --stop-loss 0.02 --take-profit 0.04
```

**Reward:Risk:** 2:1
**Use for:** Lower risk tolerance

### 4. Aggressive Settings

```bash
# Higher risk/reward
python run_backtest.py --days 90 --leverage 15 --risk 0.08 --stop-loss 0.01 --take-profit 0.03
```

**Warning:** Higher liquidation risk!
**Use for:** Experienced traders only

### 5. Specific Historical Period

```bash
# Test during volatile period (example dates)
python run_backtest.py --start 01/03/2024 --end 15/03/2024
```

**Use for:** Testing strategy in specific market conditions

### 6. Compare Timeframes

```bash
# Test on 5-minute candles
python run_backtest.py --days 30 --interval 5m

# Test on 1-hour candles
python run_backtest.py --days 90 --interval 1h
```

**Use for:** Finding optimal timeframe

## 🎨 Output Features

### Progress Indicators

The tool shows **real-time progress** for:

✅ **Data Fetching**
```
ℹ️  Fetching BTCUSDT data from Binance mainnet...
   Period: 2024-08-16 to 2024-11-14
   Interval: 15m

   Downloading... Done!
✅ Fetched 8,640 candles
   Price Range: $58,234.50 - $93,456.78
```

✅ **ML Training** (with spinner animation)
```
ℹ️  Training Gradient Boosting Classifier...
   └─ Model: GradientBoostingClassifier
   └─ Estimators: 200
   └─ Training set: 1,200 samples
   └─ Test set: 300 samples

   ⠋ Training model...
   ✓ Training complete!

   └─ Training Accuracy: 72.50%
   └─ Testing Accuracy:  68.33%
```

✅ **Feature Importance**
```
ℹ️  Top 5 Most Important Features:
   1. rsi                      ██████████████████░░░░░░░░░░░░ 0.235
   2. zscore                   ████████████████░░░░░░░░░░░░░░ 0.198
   3. trend_strength           ███████████░░░░░░░░░░░░░░░░░░░ 0.145
   4. volatility               ████████░░░░░░░░░░░░░░░░░░░░░░ 0.112
   5. volume_ratio             ██████░░░░░░░░░░░░░░░░░░░░░░░░ 0.089
```

✅ **Backtest Progress**
```
   Running backtest simulation...

   Backtest completed in 3.2 seconds
```

✅ **Colored Results**
```
Overall Performance
────────────────────────────────────────────────────────────────────────────────
Initial Capital:    $100.00
Final Capital:      $143.50  (green if profit, red if loss)
Total Return:       +43.50%
```

### Result Assessment

The tool automatically evaluates performance:

**Excellent** (Green ✅)
- Win Rate ≥ 65%
- Profit Factor ≥ 2.0
- Sharpe Ratio ≥ 1.5

**Good** (Cyan ℹ️)
- Win Rate ≥ 55%
- Profit Factor ≥ 1.5

**Fair** (Yellow ⚠️)
- Win Rate ≥ 50%
- Profit Factor ≥ 1.2

**Poor** (Red ❌)
- Below all thresholds

## 🔧 All Available Options

```
Positional Arguments:
  --days N                Number of days to backtest
  --start DD/MM/YYYY      Start date (requires --end)
  --end DD/MM/YYYY        End date (requires --start)

Symbol Options:
  --symbol SYMBOL         Trading pair (default: BTCUSDT)
  --interval {1m,5m,15m,1h,4h}
                         Candle interval (default: 15m)

Trading Parameters:
  --capital FLOAT         Initial capital in USDT (default: 100)
  --leverage INT          Leverage multiplier (default: 10)
  --risk FLOAT           Risk per trade as decimal (default: 0.05 = 5%)
  --stop-loss FLOAT      Stop loss % (default: 0.015 = 1.5%)
  --take-profit FLOAT    Take profit % (default: 0.03 = 3%)

Order Options:
  --use-market           Use market orders instead of limit (higher fees)
  --no-trailing          Disable trailing stop

ML Options:
  --no-train             Skip ML training (use existing model)
  --forward-periods INT  Forward periods for ML labels (default: 10)

Output Options:
  --output-dir PATH      Output directory for results (default: ./outputs)
  --no-save              Do not save model and results
  --dashboard            Auto-launch dashboard after backtest completes

Help:
  -h, --help             Show help message and exit
```

## 💡 Tips & Best Practices

### 1. Start with Quick Test

```bash
# Always start with a quick 3-day test
python run_backtest.py --days 3

# If it looks good, run full backtest
python run_backtest.py --days 90
```

### 2. Test Parameter Sensitivity

```bash
# Test different stop losses
python run_backtest.py --days 30 --stop-loss 0.01
python run_backtest.py --days 30 --stop-loss 0.015
python run_backtest.py --days 30 --stop-loss 0.02

# Compare results
```

### 3. Validate Across Periods

```bash
# Test Q1 2024
python run_backtest.py --start 01/01/24 --end 31/03/24

# Test Q2 2024
python run_backtest.py --start 01/04/24 --end 30/06/24

# Is performance consistent?
```

### 4. Check Different Market Conditions

```bash
# Trending market
python run_backtest.py --start 01/10/24 --end 15/11/24

# Ranging market
python run_backtest.py --start 01/05/24 --end 31/05/24

# Does strategy adapt?
```

## ⚠️ Common Errors

### "Start date must be before end date"

```bash
# Wrong:
python run_backtest.py --start 14/11/24 --end 01/11/24

# Correct:
python run_backtest.py --start 01/11/24 --end 14/11/24
```

### "Invalid date format"

```bash
# Wrong formats:
--start 2024-11-01      # Use dd/mm/yyyy
--start 11/01/2024      # Month first (American format)

# Correct formats:
--start 01/11/2024      # Day first (dd/mm/yyyy)
--start 01/11/24        # Day first (dd/mm/yy)
```

### "Training failed"

**Cause:** Not enough data or poor internet connection

**Solution:**
```bash
# Use more days
python run_backtest.py --days 90

# Or skip training if model exists
python run_backtest.py --days 7 --no-train
```

## 📊 Understanding the Output

### Win Rate

- **>65%**: Excellent - Strategy very selective
- **55-65%**: Good - Normal for mean reversion
- **50-55%**: Fair - Barely profitable
- **<50%**: Poor - Losing strategy

### Profit Factor

- **>2.0**: Excellent - Winners much larger than losers
- **1.5-2.0**: Good - Profitable with good risk/reward
- **1.2-1.5**: Fair - Marginally profitable
- **<1.2**: Poor - Not worth the risk

### Sharpe Ratio

- **>1.5**: Excellent - Great risk-adjusted returns
- **1.0-1.5**: Good - Decent risk-adjusted returns
- **0.5-1.0**: Fair - Mediocre risk-adjusted returns
- **<0.5**: Poor - Returns don't justify risk

### Max Drawdown

- **<15%**: Excellent - Low risk
- **15-25%**: Good - Acceptable risk
- **25-35%**: Fair - High risk
- **>35%**: Poor - Very high risk

## 🔄 Typical Workflow

```bash
# 1. Quick validation
python run_backtest.py --days 3

# 2. Full backtest
python run_backtest.py --days 90

# 3. Review results

# 4. If good, start dashboard
python dashboard_app.py

# 5. Analyze in detail

# 6. If excellent, deploy live
python live_trading_bot.py
```

## 🆘 Getting Help

```bash
# Show all options
python run_backtest.py --help

# See examples
python run_backtest.py --help | grep -A 20 "Examples:"
```

---

**Happy backtesting! 📈**
