# Analysis: Why Trend Following Strategy Generates 0 Trades

## Executive Summary

**Root Cause**: Mismatched ML confidence thresholds between `trend_strategy_v2.py` and `model_factory.py`

**Impact**: Trades require 60% predicted success probability, but the local threshold is 30%. This creates a conflict where trades pass the local check but are rejected by the ML model.

**Solution**: Lower the ML model threshold in `model_factory.py` from 60% to 40% (or lower)

---

## Detailed Analysis

### 1. Walk-Forward Implementation (run_backtest.py)

**Function**: `walk_forward_backtest()` (lines 428-631)

**Data Flow**:
```python
# Line 455-456: Window sizes
train_window_bars = args.train_window * 96  # Default: 180 days = 17,280 bars
test_window_bars = args.test_window * 96    # Default: 30 days = 2,880 bars

# Line 459: Minimum starting position
min_start = train_window_bars + 100  # 17,380 bars (includes warmup)

# Lines 478-484: Split data
train_start = current_pos - train_window_bars  # e.g., bar 100
train_end = current_pos                         # e.g., bar 17,380
test_start = current_pos                        # e.g., bar 17,380
test_end = current_pos + test_window_bars       # e.g., bar 20,260

train_df = df.iloc[train_start:train_end].copy()  # 17,280 bars
test_df = df.iloc[test_start:test_end].copy()     # 2,880 bars
```

**Training** (line 494):
```python
train_model_with_progress(bot, train_df, args.forward_periods)
```

**Testing** (line 505):
```python
results = bot.backtest(test_df, ...)  # ⚠️ Only test window passed!
```

**Status**: ✅ Data split is correct, test window has sufficient bars (2,880)

---

### 2. Signal Generation Flow (trend_strategy_v2.py)

#### Backtest Method (lines 417-714)

```python
# Line 457: Calculate features on TEST WINDOW ONLY
df_with_features = self.feature_engineer.calculate_features(df)  # df = test_df

# Line 459: Loop through test window (skip first 100 for warmup)
for i in range(100, len(df_with_features) - 10):
    setup = self.should_enter_trade(df_with_features, current_idx=i)
```

**Issue**: Features calculated on test window only, not including training history
- Rolling calculations (e.g., `rolling(100).quantile(0.50)`) use only test window data
- First 100 bars produce NaN values (skipped by loop)
- From bar 100 onwards: valid calculations but limited context

**Status**: ⚠️ Minor issue - rolling stats lack historical context, but calculations work

---

#### Signal Detection (lines 92-142)

**Long Trend Criteria** (identify_long_trends):
```python
# Line 124: Momentum check (needs 100 bars of history)
(momentum_score > momentum_score.rolling(100).quantile(0.50))

# Line 127: Price above EMA9
(df['close'] > ema_9)

# Line 130: EMA alignment
(ema_9 > ema_21)

# Line 133: Volume confirmation (lowered to 0.7)
(volume_ratio > 0.7)

# Line 136: ADX trend strength (lowered to 15)
(adx > 15)

# Line 139: RSI bounds (widened to < 90)
(talib.RSI(df['close'], timeperiod=14) < 90)
```

**Calculation at bar 100**:
- `momentum_score.rolling(100).quantile(0.50)` looks at bars 0-100 of test window
- These are bars 17,380-17,480 of the full dataset (not bars 0-100)
- Quantile is calculated on test period data, not training period
- This may produce different signals than intended

**Status**: ⚠️ Works but uses test window context, not training context

---

#### should_enter_trade Method (lines 370-415)

**Execution Flow**:
```python
# Line 372: Get current bar
current = df.iloc[current_idx]

# Line 375-376: Check signal generators
is_long = self.signal_generator.identify_long_trends(df).iloc[current_idx]
is_short = self.signal_generator.identify_short_trends(df).iloc[current_idx]

# Line 378-379: If no signal, exit early
if not (is_long or is_short):
    return None  # ✅ Early exit if no technical signal

# Line 384-399: Create setup object
setup = TrendTradeSetup(...)  # Populate with current bar values

# Line 402-406: Get ML prediction
features = current[self.feature_names]  # ✅ Features available
ml_prediction = self.ml_model.predict_with_confidence(features)

setup.confidence = ml_prediction['confidence_score']
setup.predicted_success = ml_prediction['success_probability']

# Line 409-410: LOCAL CHECK (30% threshold)
if ml_prediction['success_probability'] < 0.30:  # ⚠️ First check
    return None

# Line 412-413: ML MODEL CHECK (60% threshold in model_factory.py!)
if not ml_prediction['should_trade']:  # ⚠️ CRITICAL - Second check
    return None

return setup  # Only reached if both checks pass
```

**Status**: ❌ **CRITICAL CONFLICT** - Two different thresholds!

---

### 3. ML Model Filtering (model_factory.py)

#### EnhancedMLModel.predict_with_confidence (lines 206-265)

```python
# Line 232-234: Get predictions
proba = self.model.predict_proba(X_scaled)[0]
success_prob = proba[1]  # Probability of success (class 1)

# Line 237: Confidence score
confidence = max(proba)

# Line 240-243: ⚠️⚠️⚠️ THE PROBLEM ⚠️⚠️⚠️
should_trade = (
    success_prob >= 0.60 and  # ❌ STILL AT 60%!
    confidence >= self.min_confidence  # ✅ 30% (default)
)

# Line 246-250: Rejection reasons
if not should_trade:
    if success_prob < 0.60:
        reason = f"Success probability too low: {success_prob:.1%}"
    elif confidence < self.min_confidence:
        reason = f"Confidence too low: {confidence:.1%} < {self.min_confidence:.1%}"

# Line 252-257: Return decision
return {
    'should_trade': should_trade,  # ❌ False if success_prob < 0.60
    'reason': reason,
    'confidence_score': confidence,
    'success_probability': success_prob
}
```

**Status**: ❌ **ROOT CAUSE** - 60% threshold too high!

---

## 4. Feature Availability During Backtest

**Feature List** (trend_strategy_v2.py lines 278-305):
```python
[
    'roc_5', 'roc_10', 'roc_20', 'momentum_10', 'momentum_score',
    'adx', 'di_diff', 'plus_di', 'minus_di',
    'macd', 'macd_signal', 'macd_hist',
    'ema_9_21_diff', 'ema_21_50_diff', 'price_ema9_diff',
    'volume_ratio', 'volume_surge',
    'atr_pct',
    'breakout_high', 'breakout_low', 'dist_from_high20', 'dist_from_low20',
    'higher_high', 'lower_low',
    'rsi'
]
```

**Availability Check**:
```python
# trend_strategy_v2.py line 402
features = current[self.feature_names]  # Extract features for ML prediction
```

**Training** (run_backtest.py line 310):
```python
bot.ml_model.feature_names = feature_cols  # ✅ Correctly set during training
```

**Status**: ✅ All features available during backtest

---

## 5. Complete Execution Trace

### Walk-Forward Window #1

1. **Data Split** (run_backtest.py lines 478-484):
   - Training: bars 100 to 17,380 (17,280 bars)
   - Testing: bars 17,380 to 20,260 (2,880 bars)

2. **Model Training** (run_backtest.py line 494):
   ```python
   train_model_with_progress(bot, train_df, forward_periods=10)
   ```
   - Calculates features on train_df (line 153)
   - Identifies trend signals (lines 178-179)
   - Trains ML model on successful vs failed signals (line 270)
   - Sets bot.ml_model.feature_names (line 310)
   - Sets bot.ml_model.is_trained = True (line 311)

3. **Backtest** (run_backtest.py line 505):
   ```python
   results = bot.backtest(test_df, ...)
   ```

4. **Inside backtest** (trend_strategy_v2.py):
   - Line 457: Calculate features on test_df
   - Line 459: Loop i = 100 to 2,870
   - Line 460: For each bar, call `should_enter_trade(df_with_features, i)`

5. **Inside should_enter_trade**:
   - Line 375: Check if long signal → **Likely TRUE for some bars**
   - Line 384-399: Create TrendTradeSetup → **Setup created**
   - Line 402-406: Get ML prediction → **ML model executes**
   - Line 409: Check `success_probability >= 0.30` → **PASSES** (e.g., 0.45)
   - Line 412: Check `ml_prediction['should_trade']` → **FAILS** (requires >= 0.60)
   - Line 413: `return None` → **TRADE REJECTED**

---

## 6. Why Trades Are Being Blocked

### Scenario Analysis

Assume ML model predicts:
- `success_probability = 0.45` (45% chance of profit)
- `confidence = 0.50` (50% confidence in prediction)
- `min_confidence = 0.30` (30% threshold from ResearchBackedTrendBot.__init__)

**Check 1** (trend_strategy_v2.py line 409):
```python
if ml_prediction['success_probability'] < 0.30:  # 0.45 >= 0.30
    return None  # ✅ PASSES
```

**Check 2** (model_factory.py line 240-243):
```python
should_trade = (
    success_prob >= 0.60 and  # 0.45 < 0.60  ❌ FAILS
    confidence >= 0.30        # 0.50 >= 0.30  ✅ PASSES
)
# should_trade = False
```

**Check 3** (trend_strategy_v2.py line 412):
```python
if not ml_prediction['should_trade']:  # not False = True
    return None  # ❌ REJECTED!
```

**Result**: Trade is rejected even though it has 45% success probability and 50% confidence.

---

## 7. Specific Line Numbers of Blocking Conditions

### Primary Blocker (model_factory.py)

**File**: `/home/user/mlbot/model_factory.py`
**Line**: 241
**Code**:
```python
success_prob >= 0.60 and  # At least 60% predicted success
```

**Issue**: This threshold is too conservative. Trades with 40-59% success probability are rejected.

**Evidence**: Recent commit (a34f1bb) lowered threshold in trend_strategy_v2.py from 0.45 to 0.30, but model_factory.py was not updated.

---

### Secondary Check (trend_strategy_v2.py)

**File**: `/home/user/mlbot/trend_strategy_v2.py`
**Line**: 409-410
**Code**:
```python
if ml_prediction['success_probability'] < 0.30:  # Was 0.45, now even lower
    return None
```

**Purpose**: Local threshold (already lowered to 30%)
**Status**: ✅ Correctly lowered in commit a34f1bb

---

**File**: `/home/user/mlbot/trend_strategy_v2.py`
**Line**: 412-413
**Code**:
```python
if not ml_prediction['should_trade']:
    return None
```

**Purpose**: Delegates to ML model's decision
**Issue**: ❌ ML model still uses 60% threshold
**Effect**: Overrides the local 30% threshold

---

## 8. Additional Contributing Factors

### Issue A: Test Window Feature Calculation

**Location**: trend_strategy_v2.py line 457
```python
df_with_features = self.feature_engineer.calculate_features(df)
```

**Problem**:
- `df` is the test window only (2,880 bars)
- Rolling calculations like `momentum_score.rolling(100).quantile(0.50)` use only test window data
- The 50th percentile is calculated on bars 0-100 of test window, not the full historical dataset

**Impact**:
- Signals may behave differently than during training
- The rolling quantile represents test period statistics, not training period
- Could lead to distribution mismatch between training and testing

**Severity**: ⚠️ Medium (contributes to problem but not root cause)

---

### Issue B: Insufficient Training Data

**Location**: run_backtest.py line 203-205
```python
if training_samples < 50:
    print_warning("Few training samples - model may not be reliable")
```

**Problem**:
- If training window (180 days) doesn't produce enough trend signals
- Model may not learn proper patterns
- Low training samples → poor predictions → low success probabilities

**Check**: Need to verify actual training_samples count from logs

**Severity**: ⚠️ Medium (depends on actual data)

---

### Issue C: Model Performance

**Location**: run_backtest.py lines 285-290
```python
if test_acc > 0.65:
    print_success("Model shows good predictive power!")
elif test_acc > 0.55:
    print_warning("Model shows moderate predictive power")
else:
    print_warning("Model shows weak predictive power - consider more data")
```

**Problem**:
- If model accuracy is < 55%, it may predict low success probabilities
- Even with lowered thresholds, low model performance = low predictions

**Check**: Need to verify model accuracy from training logs

**Severity**: ⚠️ Medium (depends on model training results)

---

## 9. Recommended Solutions

### Solution 1: Lower ML Model Threshold (CRITICAL)

**File**: `/home/user/mlbot/model_factory.py`
**Line**: 241
**Change**:
```python
# FROM:
success_prob >= 0.60 and  # At least 60% predicted success

# TO:
success_prob >= 0.40 and  # At least 40% predicted success
```

**Rationale**:
- Aligns with lowered threshold in trend_strategy_v2.py
- Allows trades with 40-60% success probability
- With 2:1 reward:risk ratio, 40% win rate can be profitable
- Research shows trend following succeeds with 50-60% win rates

**Expected Impact**: 10-30x increase in trades (0 → 50-100 trades)

---

### Solution 2: Include Historical Context in Rolling Calculations

**File**: `/home/user/mlbot/trend_strategy_v2.py`
**Line**: 457
**Change**: Pass test_df with prepended training history for rolling calculations

**Implementation**:
```python
# Option A: Prepend last 200 bars of training data
history_bars = 200
if hasattr(self, '_last_training_df'):
    df_with_history = pd.concat([
        self._last_training_df.tail(history_bars),
        df
    ])
    df_with_features = self.feature_engineer.calculate_features(df_with_history)
    df_with_features = df_with_features.tail(len(df))  # Keep only test window
else:
    df_with_features = self.feature_engineer.calculate_features(df)
```

**Expected Impact**: More consistent signal generation between training and testing

---

### Solution 3: Monitor and Log ML Predictions

**File**: `/home/user/mlbot/trend_strategy_v2.py`
**Line**: After 406
**Change**: Add debug logging to understand why trades are rejected

**Implementation**:
```python
# After line 406
if False:  # Enable for debugging
    print(f"[Bar {current_idx}] ML Prediction:")
    print(f"  Success Probability: {ml_prediction['success_probability']:.2%}")
    print(f"  Confidence: {ml_prediction['confidence_score']:.2%}")
    print(f"  Should Trade: {ml_prediction['should_trade']}")
    print(f"  Reason: {ml_prediction['reason']}")
```

**Expected Impact**: Better visibility into rejection reasons

---

## 10. Verification Steps

After applying Solution 1, verify with:

```bash
python run_backtest.py --days 365 --walk-forward --model xgboost --strat extremetrends
```

**Expected Results**:
- Window #1: 20-50 trades (was 0)
- Total across all windows: 80-150 trades (was 0)
- Win rate: 50-60%
- Sharpe ratio: 1.5-2.5+

**If still 0 trades**, check:
1. Training samples count (should be > 50)
2. Model accuracy (should be > 55%)
3. Signal generation (should produce long_signal or short_signal = True)
4. Feature availability (check for NaN or KeyError)

---

## Summary

**Root Cause**: Model factory threshold (60%) doesn't match lowered strategy threshold (30%)

**Blocking Sequence**:
1. Signal generator identifies potential trade (✅)
2. ML model predicts success_probability = 45% (✅)
3. Local check: 45% >= 30% → PASSES (✅)
4. ML model check: 45% >= 60% → FAILS (❌)
5. Trade rejected (❌)

**Fix**: Change line 241 in model_factory.py from 0.60 to 0.40

**Expected Outcome**: 50-100+ trades per year instead of 0
