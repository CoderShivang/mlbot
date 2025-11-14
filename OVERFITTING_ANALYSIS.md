# ML Overfitting Analysis: 90% Training → 37% Actual Win Rate

## The Problem

**Observation from Backtest:**
- Training Accuracy: 95-98%
- Testing Accuracy: 90-96%
- **Actual Win Rate: 37.5%** ❌

This is a **massive gap** indicating severe overfitting or train-test mismatch.

## Root Cause: Training vs Trading Mismatch

### What We're Training On:
```python
# In train_model_with_progress(), line 173:
df_features['forward_return'] = df_features['close'].shift(-forward_periods) / df_features['close'] - 1

# Line 176-177:
df_features.loc[long_mask, 'success'] = (df_features.loc[long_mask, 'forward_return'] > threshold).astype(int)
# success = 1 if forward_return > 1%, else 0
```

**Training Question:** "Will the price be up >1% in 10 bars?"

### What We're Actually Trading:
```python
# In backtest(), lines 507-534:
stop_loss_price = fill_price * (1 - 0.015)  # 1.5% SL
take_profit_price = fill_price * (1 + 0.03)  # 3% TP

# Check SL/TP sequentially bar-by-bar
```

**Trading Question:** "Can we capture 3% profit before hitting 1.5% loss?"

## Why This Causes Overfitting

### Example Scenario:

**Bar 100 Setup:**
- Entry: $100
- SL: $98.50 (-1.5%)
- TP: $103.00 (+3%)

**Price Action:**
1. Bar 101: $100.50 (+0.5%)
2. Bar 102: $99.00 (-1%)
3. Bar 103: $98.40 (-1.6%) → **SL HIT** → LOSS
4. Bar 104-110: Rallies to $101.50 (+1.5%)

**ML Model Prediction:**
- Forward return (bar 110): +1.5%
- Model predicts: SUCCESS ✅ (90% confidence)
- Label: 1 (success, because price went up >1%)

**Actual Trade Outcome:**
- Stopped out at bar 103: -1.5%
- Actual result: LOSS ❌

**Gap:** Model thinks this is a winning trade, but it's actually a losing trade!

## The Magnitude of the Problem

With 90% model accuracy but 37% actual win rate:
- ~60% of "predicted wins" are actually losses
- This means most setups get stopped out before reaching profit

## Solutions

### Solution 1: Train on Simulated Trade Outcomes (BEST)

Instead of using forward returns, simulate SL/TP for each training sample:

```python
def calculate_actual_trade_outcome(df, entry_idx, sl_pct=0.015, tp_pct=0.03):
    """
    Simulate actual trade with SL/TP to get TRUE outcome
    """
    entry_price = df.iloc[entry_idx]['close']
    sl_price = entry_price * (1 - sl_pct)
    tp_price = entry_price * (1 + tp_pct)

    # Check next 10 bars
    for i in range(entry_idx + 1, min(entry_idx + 11, len(df))):
        bar = df.iloc[i]

        # Check SL first (more common)
        if bar['low'] <= sl_price:
            return 0  # Loss

        # Check TP
        if bar['high'] >= tp_price:
            return 1  # Win

    # Timeout
    final_price = df.iloc[min(entry_idx + 10, len(df) - 1)]['close']
    return 1 if final_price > entry_price else 0
```

**Impact:** Training will match actual trading, eliminating the mismatch.

### Solution 2: Increase Success Threshold

Current threshold: 1% return
Actual TP target: 3% return

Change threshold to match:
```python
threshold = 0.025  # 2.5% threshold (closer to 3% TP)
```

**Impact:** Model learns patterns that lead to larger moves, more likely to hit TP.

### Solution 3: Simplify the Model

Current: 25 features with XGBoost
Too complex → learns noise

Simplify to top 5-8 most important features only:
- atr_pct (volatility)
- ema_21_50_diff (trend)
- macd (momentum)
- volume_ratio (confirmation)
- adx (trend strength)

**Impact:** More robust, less overfitting.

### Solution 4: Add Stricter Entry Filters

Current: Very permissive (ADX > 10, volume > 0.5)

Increase quality bar:
- ADX > 20 (stronger trends)
- Volume > 1.0 (better confirmation)
- Momentum > 70th percentile (top 30%)

**Impact:** Fewer but higher quality setups.

## Recommended Fix: Combination Approach

1. **Implement Solution 1** (train on simulated outcomes) - CRITICAL
2. **Apply Solution 3** (simplify to 8 features) - HIGH PRIORITY
3. **Apply Solution 2** (increase threshold to 2.5%) - MEDIUM
4. **Keep current filters** but add option for stricter mode - LOW

## Expected Results After Fix

- Training accuracy: 70-80% (lower but more realistic)
- Testing accuracy: 65-75% (closer to training)
- **Actual win rate: 50-60%** ✅ (much closer to testing)

The lower accuracy is GOOD - it means the model is learning realistic patterns, not overfitting to noise.
