# System Architecture Overview

## ML-Enhanced Mean Reversion Trading Bot

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE                           │
│  (Command Line / Scripts / Configuration Files)                 │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                    MAIN BOT CONTROLLER                           │
│               (MLMeanReversionBot Class)                         │
│                                                                   │
│  • Orchestrates all components                                   │
│  • Manages state and persistence                                 │
│  • Handles data flow between modules                             │
└──┬────────┬──────────┬──────────┬──────────┬────────────────────┘
   │        │          │          │          │
   ▼        ▼          ▼          ▼          ▼
┌──────┐ ┌──────┐  ┌──────┐  ┌──────┐  ┌──────────┐
│Data  │ │Feature│ │Signal│ │  ML  │  │  Risk    │
│Layer │ │Engine │ │Gen.  │ │Model │  │Management│
└──────┘ └──────┘  └──────┘  └──────┘  └──────────┘
   │        │          │          │          │
   └────────┴──────────┴──────────┴──────────┘
                       │
                       ▼
            ┌──────────────────┐
            │  BINANCE API     │
            │  (Execution)     │
            └──────────────────┘
```

## Core Components

### 1. Data Layer (`fetch_historical_data()`)

**Purpose**: Interface with Binance API to fetch market data

**Responsibilities**:
- Fetch OHLCV (Open, High, Low, Close, Volume) data
- Handle different timeframes (1m, 5m, 15m, 1h, etc.)
- Data validation and cleaning
- Timestamp management

**Key Functions**:
```python
fetch_historical_data(symbol, interval, lookback_days)
get_recent_data(lookback_candles)
```

**Outputs**: 
- Pandas DataFrame with OHLCV data
- Indexed by timestamp

---

### 2. Feature Engineering (`FeatureEngineering` class)

**Purpose**: Transform raw price data into meaningful features for ML

**Categories of Features**:

1. **Mean Reversion Indicators**:
   - RSI (Relative Strength Index)
   - Bollinger Bands position and width
   - Z-Score of price

2. **Market Context**:
   - Volatility (ATR, rolling std)
   - Trend strength (ADX, DI)
   - Volume analysis (relative volume, MFI)

3. **Pattern Features**:
   - Momentum (MOM, ROC)
   - Support/Resistance proximity
   - Consolidation detection
   - Drawdown analysis

4. **Candlestick Patterns**:
   - Doji, Hammer, Engulfing patterns

**Key Method**:
```python
calculate_features(df: DataFrame) -> DataFrame
```

**Inputs**: Raw OHLCV DataFrame
**Outputs**: Enhanced DataFrame with 20+ technical features

---

### 3. Signal Generation (`MeanReversionSignals` class)

**Purpose**: Identify potential mean reversion setups using traditional TA

**Long Setup Criteria**:
- RSI < 30 (oversold)
- Price near lower Bollinger Band (bb_position < 0.2)
- Z-score < -1.5 (significantly below mean)

**Short Setup Criteria**:
- RSI > 70 (overbought)
- Price near upper Bollinger Band (bb_position > 0.8)
- Z-score > 1.5 (significantly above mean)

**Key Methods**:
```python
identify_long_setups(df) -> Series
identify_short_setups(df) -> Series
```

---

### 4. ML Pattern Recognizer (`MLPatternRecognizer` class)

**Purpose**: Learn which mean reversion signals lead to successful trades

**ML Model**: Gradient Boosting Classifier
- **Algorithm**: Gradient Boosting Decision Trees
- **Why?**: 
  - Handles non-linear relationships
  - Feature importance analysis
  - Robust to overfitting with proper tuning
  - Good performance on tabular data

**Training Process**:

```
Historical Data → Feature Engineering → Forward Returns
                                              ↓
                                    Label Creation
                                   (Success = 1, Fail = 0)
                                              ↓
                                    Train/Test Split
                                              ↓
                                    Feature Scaling
                                              ↓
                                    Model Training
                                              ↓
                                    Performance Eval
```

**Key Methods**:
```python
train(df, forward_returns)
predict_trade_quality(features) -> Dict
find_similar_trades(features, n_similar) -> List
add_trade_outcome(trade)
```

**Pattern Matching**:
- Uses Euclidean distance in feature space
- Finds K most similar historical trades
- Provides context on past outcomes
- Helps filter bad setups and boost confidence on good ones

---

### 5. Trade Setup (`TradeSetup` dataclass)

**Purpose**: Encapsulate all information about a potential trade

**Contains**:
- Entry conditions (price, direction)
- Mean reversion signals (RSI, BB, Z-score)
- Market context (volatility, trend, volume)
- Pattern features
- ML predictions (confidence, similar trades)
- Outcomes (for learning)

**Data Flow**:
```
Market Data → Features → Signal Check → Create TradeSetup
                                              ↓
                                    ML Analysis
                                    (confidence score)
                                              ↓
                                    Decision: Trade or Skip?
```

---

### 6. Risk Management

**Position Sizing**:
```python
position_size = (capital × risk_per_trade) / stop_loss_pct
position_size *= leverage
quantity = position_size / entry_price
```

**Stop Loss & Take Profit**:
- Fixed percentage from entry
- Automatically placed on Binance
- OCO (One-Cancels-Other) orders

**Safety Mechanisms**:
- Maximum positions limit
- Maximum daily loss check
- Emergency stop all positions
- Balance monitoring

---

### 7. Backtesting Engine (`backtest()`)

**Purpose**: Validate strategy on historical data before live trading

**Simulation Process**:
```
For each candle in history:
    1. Calculate features
    2. Check for signal
    3. If signal → ML evaluation
    4. If ML approves → Simulate trade
    5. Check forward price action
    6. Determine if SL or TP hit
    7. Calculate PnL
    8. Update capital
    9. Record trade outcome
```

**Metrics Calculated**:
- Win rate
- Average win/loss
- Profit factor
- Sharpe ratio
- Maximum drawdown
- Total return

---

### 8. Live Trading (`LiveTradingBot` class)

**Purpose**: Execute trades on Binance in real-time

**Main Loop**:
```python
while is_running:
    1. Fetch recent market data
    2. Calculate features
    3. Check for signals
    4. If signal → ML evaluation
    5. If approved → Execute trade
    6. Monitor open positions
    7. Save state
    8. Wait for next interval
```

**Order Execution**:
1. Calculate position size
2. Place market order (buy/sell)
3. Set stop loss order
4. Set take profit order
5. Monitor position

**Position Monitoring**:
- Track unrealized PnL
- Check if SL/TP hit
- Emergency close if needed
- Update trade outcomes for learning

---

## Data Flow Diagram

### Training Phase

```
Historical Data (Binance)
    ↓
Feature Engineering
    ↓
Signal Identification
    ↓
Forward Returns Calculation
    ↓
ML Model Training
    ↓
Backtesting & Evaluation
    ↓
Trained Model (saved to disk)
```

### Live Trading Phase

```
Real-time Data (Binance)
    ↓
Feature Engineering
    ↓
Signal Check → [No Signal] → Wait
    ↓ [Signal Detected]
ML Evaluation
    ↓
Confidence Score → [Low] → Skip Trade
    ↓ [High Confidence]
Find Similar Trades
    ↓
Execute Trade (if approved)
    ↓
Monitor Position
    ↓
Close Trade (SL/TP hit)
    ↓
Record Outcome → Update ML Model
    ↓
Continue Loop
```

---

## File Structure

```
ml-mean-reversion-bot/
│
├── ml_mean_reversion_bot.py    # Core bot implementation
│   ├── FeatureEngineering
│   ├── MeanReversionSignals
│   ├── MLPatternRecognizer
│   └── MLMeanReversionBot
│
├── example_usage.py             # Training & backtesting example
├── live_trading_bot.py          # Live trading implementation
├── utils.py                     # Analysis & visualization tools
├── setup.py                     # Setup verification script
│
├── requirements.txt             # Python dependencies
├── config_template.json         # Configuration template
│
├── README.md                    # User documentation
└── ARCHITECTURE.md              # This file
```

---

## Key Design Decisions

### 1. Why Mean Reversion?

**Pros**:
- Well-defined entry/exit rules
- Works in ranging/sideways markets
- Clear statistical basis
- High win rate potential

**Cons**:
- Can suffer in strong trends
- Requires good timing
- Needs proper risk management

**ML Mitigation**: Model learns to filter trades in trending conditions

---

### 2. Why Gradient Boosting?

**Alternatives Considered**:
- Random Forest: Good but slightly less accurate
- Neural Networks: Overkill, prone to overfitting on limited data
- SVM: Slower, harder to interpret

**Gradient Boosting Advantages**:
- Best performance on tabular data
- Feature importance insights
- Handles mixed feature types well
- Robust with proper regularization

---

### 3. Why 15-Minute Timeframe?

**Balance**:
- Fast enough for multiple trades per day
- Slow enough to avoid noise
- Good for mean reversion (gives time to revert)
- Reasonable backtesting data size

**Customizable**: Users can change to 5m, 1h, etc.

---

### 4. Feature Selection Philosophy

**Principle**: Diverse feature types for comprehensive market context

**Categories**:
1. **Price-based**: What is happening to price?
2. **Momentum-based**: How fast is it changing?
3. **Volume-based**: Who is participating?
4. **Volatility-based**: How stable is the market?
5. **Trend-based**: What's the bigger picture?

**Avoids**: Highly correlated features (e.g., multiple moving averages)

---

## Security Considerations

### API Key Management

**Never** hardcode API keys:
```python
# ❌ BAD
api_key = "abc123..."

# ✅ GOOD
api_key = os.getenv('BINANCE_API_KEY')
```

### Permissions

API key should only have:
- ✅ Read account data
- ✅ Futures trading
- ❌ Withdraw (NEVER enable)

### Network Security

- Use HTTPS only
- Verify SSL certificates
- Consider IP whitelist on Binance

---

## Performance Optimization

### Data Fetching
- Cache recent data
- Minimize API calls
- Use appropriate rate limits

### Feature Calculation
- Vectorized operations (Pandas/NumPy)
- Avoid loops where possible
- Calculate once, use multiple times

### ML Inference
- Pre-scale features
- Batch predictions if needed
- Model runs in ~ms on modern hardware

---

## Extensibility

### Adding New Features

1. Add calculation in `FeatureEngineering.calculate_features()`
2. Add feature name to `get_feature_list()`
3. Retrain model
4. Test in backtest

### Adding New Signals

1. Create method in `MeanReversionSignals`
2. Integrate in `analyze_setup()`
3. Backtest to validate

### Changing ML Model

1. Import new model in `MLPatternRecognizer`
2. Update `__init__()` with new model
3. Ensure `fit()` and `predict_proba()` compatibility
4. Retrain and evaluate

---

## Common Issues & Solutions

### Issue: Model predicts all trades as unsuccessful

**Cause**: Imbalanced training data or overfitting
**Solution**: 
- Adjust success_threshold
- Use class_weight='balanced' in model
- Collect more training data

### Issue: Poor backtest but model accuracy high

**Cause**: Overfitting to training data
**Solution**:
- Reduce model complexity (lower max_depth)
- Increase regularization
- Use time-based train/test split

### Issue: Trades not executing live

**Cause**: ML confidence threshold too high
**Solution**:
- Lower confidence_threshold (e.g., 0.60 instead of 0.65)
- Review feature calculations for errors
- Check if signals are being generated

---

## Future Enhancements

1. **Multi-timeframe Analysis**: Combine signals from multiple timeframes
2. **Sentiment Integration**: Add crypto fear/greed index
3. **Order Book Analysis**: Use bid/ask imbalance
4. **Funding Rates**: Consider funding rate in perps
5. **Portfolio Optimization**: Multi-asset trading
6. **Reinforcement Learning**: Let agent learn optimal actions
7. **Real-time Model Updates**: Continuous online learning

---

## Conclusion

This architecture balances:
- **Simplicity**: Easy to understand and modify
- **Effectiveness**: Proven trading concepts + ML
- **Robustness**: Comprehensive risk management
- **Extensibility**: Easy to add new features

The key insight: **ML doesn't replace traditional trading logic; it enhances it by learning which setups work in which contexts.**
