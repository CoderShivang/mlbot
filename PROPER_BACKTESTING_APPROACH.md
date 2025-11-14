# Proper Backtesting Approach - Walk-Forward Analysis

## ⚠️ Current Problem

The current implementation has **look-ahead bias**: the ML model is trained on the same data it's tested on, allowing it to "see the future."

## ✅ Correct Approach: Walk-Forward Analysis

### Concept
Train the model on PAST data only, then test on FUTURE data you haven't seen yet.

### Example: 365-Day Backtest

```
Total Period: 365 days
Training Window: 180 days
Testing Window: 30 days
Retraining Frequency: Every 30 days

Timeline:
[Day 1 ═══════════════════════════════════════════════════════ Day 365]

Walk-Forward Windows:

Window 1:
├─ Train: Day 1-180   (6 months of history)
└─ Test:  Day 181-210 (1 month forward) ← Model has NEVER seen this data

Window 2:
├─ Train: Day 31-210  (6 months, rolling window)
└─ Test:  Day 211-240 (1 month forward)

Window 3:
├─ Train: Day 61-240
└─ Test:  Day 241-270

...and so on
```

### Implementation Pseudocode

```python
def walk_forward_backtest(df, train_window=180, test_window=30):
    """
    Proper walk-forward backtest without look-ahead bias

    Args:
        df: Full historical data
        train_window: Days to train on (e.g., 180 = 6 months)
        test_window: Days to test on (e.g., 30 = 1 month)
    """
    results = []

    # Start after we have enough training data
    start_idx = train_window

    while start_idx + test_window < len(df):
        # Training period (PAST data only)
        train_start = start_idx - train_window
        train_end = start_idx
        train_df = df.iloc[train_start:train_end]

        # Train model on historical data
        model = train_ml_model(train_df)

        # Testing period (FUTURE data - model has never seen this)
        test_start = start_idx
        test_end = start_idx + test_window
        test_df = df.iloc[test_start:test_end]

        # Backtest on unseen future data
        test_results = run_backtest(test_df, model)
        results.append(test_results)

        # Move forward by test_window days
        start_idx += test_window

    # Combine all results
    return aggregate_results(results)
```

### Key Differences from Current Implementation

| Aspect | Current (Wrong) | Walk-Forward (Correct) |
|--------|----------------|----------------------|
| **Training Data** | All 365 days | Only past 180 days |
| **Testing Data** | Same 365 days | Next 30 unseen days |
| **Model Updates** | Once, on all data | Retrained every 30 days |
| **Look-Ahead Bias** | ❌ Yes (sees future) | ✓ No (only sees past) |
| **Realistic** | ❌ Overly optimistic | ✓ Realistic performance |

### Benefits of Walk-Forward

1. **No Look-Ahead Bias**: Model only uses data available at that point in time
2. **Adapts to Market Changes**: Retrains regularly on recent data
3. **Realistic Performance**: Results reflect what you'd actually get in live trading
4. **Out-of-Sample Testing**: Each test period is truly unseen

### Expected Performance Change

When fixing look-ahead bias, expect:
- **Win rate to drop** (model was cheating before)
- **More realistic Sharpe ratio**
- **Higher variance in results**
- **Better representation of live trading**

If your strategy still performs well with walk-forward, it's truly robust!

## 📊 Example Comparison

### Current Method (Look-Ahead Bias):
```
Backtest Results: 365 days
├─ Win Rate: 68%        ← Unrealistically high
├─ Sharpe: 2.5          ← Too good to be true
└─ Max DD: -5%          ← Suspiciously low
```

### Walk-Forward (Realistic):
```
Backtest Results: 365 days (12 windows)
├─ Win Rate: 52-58%     ← More realistic range
├─ Sharpe: 1.2-1.8      ← Achievable in real trading
└─ Max DD: -12%         ← Expected drawdowns
```

## 🛠️ Implementation Recommendation

To implement walk-forward in the bot:

1. **Add new function**: `walk_forward_backtest()`
2. **Parameters**:
   - `--train-window 180` (days to train on)
   - `--test-window 30` (days to test forward)
   - `--retrain-freq 30` (how often to retrain)

3. **Usage**:
```bash
python run_backtest.py --days 365 --walk-forward \
  --train-window 180 --test-window 30
```

This would give you 12 windows of testing (365/30 ≈ 12 months) with proper separation.

## 📚 Further Reading

- [Quantitative Trading: Walk-Forward Analysis](https://www.quantstart.com)
- [Overfitting in Machine Learning for Trading](https://www.risk.net)
- Research: "The Deflated Sharpe Ratio" by Bailey & López de Prado

---

**Remember**: A strategy that performs well with walk-forward analysis is much more likely to work in live trading than one that only works with look-ahead bias!
