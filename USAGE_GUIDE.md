# Trading Bot Usage Guide - Complete Reference

## Quick Start

```bash
# Mean reversion strategy (default)
python run_backtest.py --days 365 --walk-forward

# Extreme trend following strategy
python run_backtest.py --days 365 --strat extremetrends --walk-forward

# Combined strategy (auto-selects based on market regime)
python run_backtest.py --days 365 --strat combined --walk-forward
```

---

## Strategy Options

### 1. Mean Reversion (`--strat meanrev`)

**Use when**: Markets are ranging, choppy, or sideways
**Logic**: Buy dips, sell rips
**Best for**: Low ADX (<20), ranging markets

```bash
python run_backtest.py --days 365 --strat meanrev --walk-forward
```

**How it works:**
- Looks for oversold conditions (RSI < 30, Z-score < -2)
- Trades AGAINST the current move
- Profits from price returning to mean
- **Fails in strong trends** (gets stopped out repeatedly)

---

### 2. Extreme Trends (`--strat extremetrends`)

**Use when**: You want to catch strong directional moves
**Logic**: Ride breakouts and momentum
**Best for**: High ADX (>30), trending markets

```bash
python run_backtest.py --days 365 --strat extremetrends --walk-forward
```

**How it works:**
- Looks for breakouts with volume confirmation
- Strong ADX + directional movement
- Multi-timeframe alignment (15m, 1H, 4H all agree)
- Trades WITH the trend
- **Fails in ranging markets** (whipsaws)

---

### 3. Combined Strategy (`--strat combined`)

**Use when**: You want intelligent regime-adaptive trading
**Logic**: Auto-selects strategy based on market conditions
**Best for**: All market conditions

```bash
python run_backtest.py --days 365 --strat combined --walk-forward
```

**How it works:**
- Detects market regime (RANGING, TRENDING_UP, TRENDING_DOWN, HIGH_VOLATILITY)
- **RANGING markets** → Uses mean reversion
- **TRENDING markets** → Uses trend following
- **HIGH VOLATILITY** → Stays out (too dangerous)
- Filters out weak signals from both strategies
- **Recommended for live trading**

---

## ML Model Options

### 1. Gradient Boosting (`--model gradientboost`) - DEFAULT

**Speed**: Fast ⚡
**Accuracy**: Good ✓
**Overfitting risk**: Medium

```bash
python run_backtest.py --days 365 --model gradientboost --walk-forward
```

**Use when**: You want fast training and good general performance

---

### 2. Random Forest (`--model randomforest`)

**Speed**: Medium
**Accuracy**: Good ✓
**Overfitting risk**: Low ✓✓

```bash
python run_backtest.py --days 365 --model randomforest --walk-forward
```

**Use when**: You're concerned about overfitting or want more robust predictions

---

### 3. XGBoost (`--model xgboost`)

**Speed**: Medium
**Accuracy**: Excellent ✓✓✓
**Overfitting risk**: Low ✓✓

```bash
python run_backtest.py --days 365 --model xgboost --walk-forward
```

**Requirements**: `pip install xgboost`

**Use when**: You want the best possible performance

---

### 4. Ensemble (`--model ensemble`)

**Speed**: Slow 🐌
**Accuracy**: Excellent ✓✓✓
**Overfitting risk**: Very Low ✓✓✓

```bash
python run_backtest.py --days 365 --model ensemble --walk-forward
```

**Use when**: You want maximum accuracy and have time for training (combines all 3 models)

---

## Complete Examples

### Example 1: Test mean reversion with XGBoost
```bash
python run_backtest.py --days 180 \
    --strat meanrev \
    --model xgboost \
    --walk-forward \
    --train-window 120 \
    --test-window 30 \
    --min-confidence 0.70
```

**Result**: Tests 180 days using mean reversion + XGBoost
**Walk-forward**: Trains on 120 days, tests on 30 days, rolls forward
**Confidence**: Only trades setups with 70%+ ML confidence

---

### Example 2: Test trend following with ensemble
```bash
python run_backtest.py --days 365 \
    --strat extremetrends \
    --model ensemble \
    --walk-forward \
    --min-confidence 0.75
```

**Result**: Full year backtest of trend-following with ensemble model
**Confidence**: Very conservative (75%+ required)

---

### Example 3: Combined strategy comparison
```bash
# Test all three strategies on same data
python run_backtest.py --days 365 --strat meanrev --walk-forward --model xgboost
python run_backtest.py --days 365 --strat extremetrends --walk-forward --model xgboost
python run_backtest.py --days 365 --strat combined --walk-forward --model xgboost
```

**Result**: Compare which strategy performs best over the past year

---

## Advanced Options

### Confidence Threshold

Controls how selective the bot is:

```bash
# Conservative (fewer trades, higher quality)
--min-confidence 0.75

# Balanced (default)
--min-confidence 0.65

# Aggressive (more trades, lower quality)
--min-confidence 0.55
```

**Recommendation**: Start with 0.70, adjust based on results

---

### Walk-Forward Windows

Controls training/testing split:

```bash
# Conservative (more training data)
--train-window 180 --test-window 30

# Balanced (default)
--train-window 120 --test-window 30

# Aggressive (faster adaptation)
--train-window 90 --test-window 20
```

**Rule of thumb**: Train window should be 3-6x test window

---

### Risk Management

```bash
--capital 1000        # Starting capital in USDT
--leverage 10         # Leverage multiplier (1-20)
--risk 0.03           # Risk 3% per trade
--min-confidence 0.70 # Further scales risk based on confidence
```

**Position sizing is dynamic**:
- 80%+ confidence → 5% risk
- 70-80% confidence → 3% risk
- 65-70% confidence → 2% risk
- <65% confidence → Trade rejected

---

## Output Files

### Backtest Archive

```
./backtest/
├── backtest_365days_meanrev_xgboost_14Nov_16-24_walkforward_180x30.json
├── backtest_365days_extremetrends_ensemble_14Nov_17-30_walkforward_180x30.json
└── backtest_365days_combined_randomforest_14Nov_18-45_walkforward_180x30.json
```

**Filename format:**
```
backtest_{days}_{strategy}_{model}_{date}_{time}_{type}.json
```

### Dashboard Data

```
./outputs/
├── latest_backtest_results.json  ← Dashboard reads this
├── ml_mean_reversion_model.pkl   ← Trained model
└── trade_history.json             ← Historical trades
```

---

## Recommended Workflows

### Workflow 1: Initial Testing

```bash
# 1. Quick 30-day test with default settings
python run_backtest.py --days 30 --walk-forward

# 2. If promising, test 90 days
python run_backtest.py --days 90 --walk-forward

# 3. If still good, full 365-day test
python run_backtest.py --days 365 --walk-forward
```

---

### Workflow 2: Strategy Comparison

```bash
# Test all strategies on same period
python run_backtest.py --days 365 --strat meanrev --walk-forward
python run_backtest.py --days 365 --strat extremetrends --walk-forward
python run_backtest.py --days 365 --strat combined --walk-forward

# Compare results in dashboard
python dashboard_app.py
```

---

### Workflow 3: Model Optimization

```bash
# Test each model type
python run_backtest.py --days 365 --model gradientboost --walk-forward
python run_backtest.py --days 365 --model randomforest --walk-forward
python run_backtest.py --days 365 --model xgboost --walk-forward
python run_backtest.py --days 365 --model ensemble --walk-forward

# Use best performer for live trading
```

---

### Workflow 4: Confidence Tuning

```bash
# Test different confidence thresholds
python run_backtest.py --days 365 --min-confidence 0.60 --walk-forward
python run_backtest.py --days 365 --min-confidence 0.65 --walk-forward
python run_backtest.py --days 365 --min-confidence 0.70 --walk-forward
python run_backtest.py --days 365 --min-confidence 0.75 --walk-forward

# Find sweet spot: highest win rate with sufficient trade count
```

---

## Interpreting Results

### Win Rate Targets

| Win Rate | Assessment | Action |
|----------|------------|--------|
| <50% | Poor | Change strategy/model |
| 50-55% | Fair | Adjust confidence threshold |
| 55-60% | Good | Fine-tune parameters |
| 60-70% | Excellent | Ready for paper trading |
| >70% | Suspicious | Check for overfitting |

### Profit Factor Targets

| Profit Factor | Assessment |
|---------------|------------|
| <1.0 | Losing strategy |
| 1.0-1.5 | Marginal |
| 1.5-2.0 | Good |
| 2.0-3.0 | Excellent |
| >3.0 | Check for bugs/overfitting |

### Sharpe Ratio Targets

| Sharpe Ratio | Assessment |
|--------------|------------|
| <0.5 | Poor risk-adjusted returns |
| 0.5-1.0 | Fair |
| 1.0-2.0 | Good |
| >2.0 | Excellent |

---

## Troubleshooting

### "Win rate is low (<50%)"

**Solutions:**
1. Increase `--min-confidence` (try 0.70 or 0.75)
2. Try different strategy (`--strat combined`)
3. Try different model (`--model xgboost` or `--model ensemble`)
4. Check if market regime suits your strategy

### "Model training fails"

**Solutions:**
1. Ensure you have enough data (365+ days recommended)
2. Check for missing dependencies: `pip install xgboost talib`
3. Try simpler model: `--model gradientboost`

### "Not enough trades"

**Solutions:**
1. Lower `--min-confidence` (try 0.60)
2. Use longer backtest period: `--days 365`
3. Check market had sufficient volatility during test period

### "Too many trades, getting whipsawed"

**Solutions:**
1. Increase `--min-confidence` (try 0.75)
2. Use combined strategy: `--strat combined`
3. Increase test window: `--test-window 45`

---

## Feature Improvements Implemented

✅ **Phase 1: Market Regime Detection**
- ADX-based trend detection
- Confidence-based trade filtering

✅ **Phase 2: Dynamic Risk Management**
- ATR-based stop loss/take profit
- Confidence-scaled position sizing

✅ **Phase 3: Enhanced Features**
- Multi-timeframe analysis (15m, 1H, 4H)
- Temporal features (hour, day, session)
- Advanced pattern detection
- 40+ features (vs. original 24)

✅ **Phase 4: Multiple ML Models**
- Gradient Boosting, Random Forest, XGBoost, Ensemble
- CLI-selectable via `--model`

---

## Next Steps

1. **Run initial tests** with default settings
2. **Compare strategies** to find best fit for current market
3. **Optimize model** selection and confidence threshold
4. **Paper trade** best configuration for 30 days
5. **Go live** with conservative position sizing

**Questions?** Check STRATEGY_IMPROVEMENT_PLAN.md for detailed explanations.
