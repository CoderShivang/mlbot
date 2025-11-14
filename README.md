# ML-Enhanced Mean Reversion Bot - $100 @ 20x Leverage Edition

A sophisticated algorithmic trading bot for BTC/USDT perpetual futures on Binance, optimized for **$100 starting capital with 20x leverage** using **limit orders** for minimal fees.

## 🎯 Configuration

- **Capital**: $100 USDT
- **Leverage**: 20x (max position: $2,000)
- **Order Type**: LIMIT orders (0.02% maker fee vs 0.05% market)
- **Stop Loss**: 0.8% (16% of capital with 20x)
- **Take Profit**: 1.2% (24% of capital with 20x)
- **Data Source**: Real Binance mainnet (no testnet)
- **Backtesting**: Simulated limit order fills with realistic fees

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train Model with Real Data
```bash
python example_usage.py
```

This fetches real Binance mainnet data and backtests with your exact configuration.

### 3. Deploy Live (Real Money!)
Edit `live_trading_bot.py` with your API credentials, then:
```bash
python live_trading_bot.py
```

## ⚠️ Critical Warnings

🚨 **20x Leverage Risks:**
- 5% BTC move = 100% capital loss (liquidation)
- 1% BTC move = 20% capital impact
- Designed for experienced traders
- Very high risk

🔴 **Real Money Trading:**
- This trades on Binance mainnet (not testnet)
- Every trade uses real USDT
- Start with only $100
- Only risk what you can afford to lose

## 📊 Expected Performance

**Good metrics:**
- Win Rate: 60-70%
- Profit Factor: 1.5-2.5
- Monthly Return: 10-30% (variable!)

**Realistic:**
- Past performance ≠ future results
- Volatility will cause drawdowns
- Monitor closely (check every 2-4 hours)

## 💡 Key Features

### Why Limit Orders?
With $2,000 positions:
- Limit: $0.40 fee (0.02%)
- Market: $1.00 fee (0.05%)
- **Save $0.60 per trade!**

### ML Enhancement
Bot doesn't just follow signals blindly:
- Checks volatility regime
- Analyzes trend strength  
- Finds similar historical trades
- Only trades high-confidence setups (>65%)

## 📋 Complete Documentation

See full guides:
- **README.md** (this file) - Quick reference
- **PROJECT_SUMMARY.md** - Detailed overview
- **ARCHITECTURE.md** - Technical deep-dive
- **INDEX.md** - File navigation

## 🛡️ Risk Management

**Position Sizing:**
```
$100 capital × 20x = $2,000 max position
Risk per trade: $15 (15%)
Stop Loss: 0.8% = ~$16 (16% of capital)
```

**Safety Features:**
- Automatic stop loss on every trade
- Take profit locks in gains
- Max 1 concurrent position
- ML filters low-quality setups

**What Can Go Wrong:**
- Flash crash past stop loss
- API connection issues
- Slippage on limit orders
- Extended drawdowns
- Liquidation if BTC moves 5%

## 🎯 Best Practices

✅ **Do:**
- Monitor bot every 2-4 hours
- Start with exactly $100
- Use limit orders
- Stop if you lose 50%
- Review weekly performance

❌ **Don't:**
- Trade with rent money
- Leave unmonitored for days
- Override stop losses
- Add money during drawdown
- Trade during major news

## 📞 Support

Issues? Check:
1. This README
2. Code comments
3. Test with small amounts
4. Learn from results

## ⚖️ Disclaimer

**THIS IS NOT FINANCIAL ADVICE.**

You can lose your entire investment. Crypto trading is risky. 20x leverage amplifies both gains AND losses. Use at your own risk. Past performance does not indicate future results.

**Trade responsibly! 📈**
