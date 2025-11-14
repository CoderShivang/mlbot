"""
Complete Usage Example: ML-Enhanced Mean Reversion Bot
======================================================

This script demonstrates:
1. Fetching REAL MAINNET data from Binance
2. Training the ML model
3. Backtesting with realistic conditions ($100 @ 20x leverage)
4. Analyzing results
5. Saving the trained model

IMPORTANT: This uses REAL mainnet data but SIMULATES trades (no actual execution)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from ml_mean_reversion_bot import MLMeanReversionBot

def fetch_real_binance_data(symbol='BTCUSDT', interval='15m', days=90):
    """
    Fetch REAL mainnet data from Binance (no API key needed for public data)
    """
    print(f"Fetching REAL mainnet data for {symbol}...")
    
    try:
        from binance.client import Client
        
        # Public client (no API key needed for market data)
        client = Client()
        
        # Calculate start time
        start_time = datetime.now() - timedelta(days=days)
        start_str = start_time.strftime("%d %b %Y %H:%M:%S")
        
        print(f"Downloading {days} days of {interval} candles from Binance mainnet...")
        
        # Fetch klines
        klines = client.get_historical_klines(
            symbol, interval, start_str
        )
        
        # Convert to DataFrame
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        # Convert types
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        
        df.set_index('timestamp', inplace=True)
        df = df[['open', 'high', 'low', 'close', 'volume']]
        
        print(f"✅ Successfully fetched {len(df)} real mainnet candles")
        print(f"   Period: {df.index[0]} to {df.index[-1]}")
        print(f"   BTC Price Range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
        
        return df
        
    except ImportError:
        print("❌ python-binance not installed")
        print("   Install with: pip install python-binance")
        print("   Falling back to sample data...")
        return generate_sample_data(days=days, interval_minutes=15)
    except Exception as e:
        print(f"❌ Error fetching data: {e}")
        print("   Falling back to sample data...")
        return generate_sample_data(days=days, interval_minutes=15)


def generate_sample_data(days=90, interval_minutes=15):
    """
    Generate sample BTC price data for testing
    (In production, use real Binance data)
    """
    print("Generating sample data...")
    
    # Number of candles
    n_candles = int(days * 24 * 60 / interval_minutes)
    
    # Generate timestamps
    start_time = datetime.now() - timedelta(days=days)
    timestamps = pd.date_range(start=start_time, periods=n_candles, freq=f'{interval_minutes}min')
    
    # Generate price with trend and mean-reverting components
    np.random.seed(42)
    
    # Base price
    base_price = 40000
    
    # Trend component
    trend = np.linspace(0, 5000, n_candles)
    
    # Mean-reverting component (Ornstein-Uhlenbeck process)
    mean_reversion_speed = 0.1
    volatility = 500
    price = np.zeros(n_candles)
    price[0] = base_price
    
    for i in range(1, n_candles):
        drift = mean_reversion_speed * (base_price + trend[i] - price[i-1])
        diffusion = volatility * np.random.randn()
        price[i] = price[i-1] + drift + diffusion
    
    # Create OHLCV data
    df = pd.DataFrame(index=timestamps)
    df['close'] = price
    
    # Generate OHLC from close
    noise = 50
    df['open'] = df['close'] + np.random.randn(n_candles) * noise
    df['high'] = df[['open', 'close']].max(axis=1) + abs(np.random.randn(n_candles) * noise)
    df['low'] = df[['open', 'close']].min(axis=1) - abs(np.random.randn(n_candles) * noise)
    
    # Generate volume
    df['volume'] = 100 + 50 * np.random.randn(n_candles)
    df['volume'] = df['volume'].abs()
    
    print(f"Generated {len(df)} candles from {df.index[0]} to {df.index[-1]}")
    
    return df


def plot_backtest_results(results, df):
    """Plot backtest results"""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # Plot 1: Equity curve
    equity = results['equity_curve']
    axes[0].plot(equity, linewidth=2, color='steelblue')
    axes[0].set_title(f'Equity Curve - ${results["initial_capital"]} @ {results["leverage"]}x Leverage', 
                     fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Capital ($)')
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=results['initial_capital'], color='r', linestyle='--', alpha=0.5, label='Initial Capital')
    
    # Add final value annotation
    axes[0].text(len(equity)-1, equity[-1], f"  ${equity[-1]:.2f}", 
                va='center', fontsize=10, fontweight='bold')
    axes[0].legend()
    
    # Plot 2: Price with entry points
    trades_info = results['trades']
    trades = [t['setup'] for t in trades_info]
    
    long_entries = [(i, t['entry_price']) for i, t in enumerate(trades_info) if t['setup'].direction == 'LONG']
    short_entries = [(i, t['entry_price']) for i, t in enumerate(trades_info) if t['setup'].direction == 'SHORT']
    
    axes[1].plot(df.index, df['close'], label='BTC Price', linewidth=1.5, color='black', alpha=0.7)
    
    if long_entries and len(long_entries) > 0:
        long_x, long_y = zip(*long_entries)
        try:
            # Map trade indices to dataframe indices
            long_times = [df.index[100 + x] for x in long_x if 100 + x < len(df)]
            long_prices = [long_y[i] for i, x in enumerate(long_x) if 100 + x < len(df)]
            axes[1].scatter(long_times, long_prices, 
                           color='green', marker='^', s=100, label='Long Entry', zorder=5, alpha=0.7)
        except:
            pass
    
    if short_entries and len(short_entries) > 0:
        short_x, short_y = zip(*short_entries)
        try:
            short_times = [df.index[100 + x] for x in short_x if 100 + x < len(df)]
            short_prices = [short_y[i] for i, x in enumerate(short_x) if 100 + x < len(df)]
            axes[1].scatter(short_times, short_prices,
                           color='red', marker='v', s=100, label='Short Entry', zorder=5, alpha=0.7)
        except:
            pass
    
    axes[1].set_title('BTC Price with Trade Entries', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Price ($)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Trade PnL distribution
    pnls = [t['setup'].pnl_percent * 100 for t in trades_info if t['setup'].pnl_percent is not None]
    
    if pnls:
        axes[2].hist(pnls, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
        axes[2].axvline(x=0, color='red', linestyle='--', linewidth=2, label='Break-even')
        axes[2].axvline(x=np.mean(pnls), color='green', linestyle='--', linewidth=2, label=f'Mean: {np.mean(pnls):.1f}%')
        axes[2].set_title('Trade PnL Distribution (% of Capital)', fontsize=14, fontweight='bold')
        axes[2].set_xlabel('PnL (% of Capital)')
        axes[2].set_ylabel('Frequency')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/backtest_results.png', dpi=150, bbox_inches='tight')
    print("\n📊 Backtest visualization saved to backtest_results.png")
    
    return fig


def analyze_ml_performance(bot, results):
    """Analyze ML model's contribution to performance"""
    trades = [t['setup'] for t in results['trades']]  # Extract setups from trade info
    
    if not trades:
        print("No trades to analyze")
        return
    
    # Group trades by confidence quartiles
    confidences = [t.confidence_score for t in trades]
    quartiles = pd.qcut(confidences, q=4, labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)'], duplicates='drop')
    
    print("\n" + "="*60)
    print("ML MODEL PERFORMANCE ANALYSIS")
    print("="*60)
    
    for q in ['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)']:
        q_trades = [t for t, conf_q in zip(trades, quartiles) if conf_q == q]
        
        if not q_trades:
            continue
        
        wins = len([t for t in q_trades if t.actual_outcome == 'WIN'])
        total = len(q_trades)
        win_rate = wins / total if total > 0 else 0
        avg_pnl = np.mean([t.pnl_percent for t in q_trades if t.pnl_percent is not None])
        
        print(f"\n{q} Confidence Trades:")
        print(f"  Total: {total}")
        print(f"  Win Rate: {win_rate:.2%}")
        print(f"  Avg PnL: {avg_pnl:.2%} of capital")
    
    # Analyze by volatility regime
    print(f"\n{'-'*60}")
    print("Performance by Volatility Regime:")
    print(f"{'-'*60}")
    
    for regime in ['LOW', 'MEDIUM', 'HIGH']:
        regime_trades = [t for t in trades if t.volatility_regime == regime]
        
        if not regime_trades:
            continue
        
        wins = len([t for t in regime_trades if t.actual_outcome == 'WIN'])
        total = len(regime_trades)
        win_rate = wins / total if total > 0 else 0
        avg_pnl = np.mean([t.pnl_percent for t in regime_trades if t.pnl_percent is not None])
        
        print(f"\n{regime} Volatility:")
        print(f"  Total: {total}")
        print(f"  Win Rate: {win_rate:.2%}")
        print(f"  Avg PnL: {avg_pnl:.2%} of capital")
    
    # Analyze by trend strength
    print(f"\n{'-'*60}")
    print("Performance by Trend Strength:")
    print(f"{'-'*60}")
    
    trend_strengths = [t.trend_strength for t in trades]
    trend_labels = ['Strong Down', 'Down', 'Neutral', 'Up', 'Strong Up']
    trend_bins = pd.cut(trend_strengths, bins=5, labels=trend_labels)
    
    for label in trend_labels:
        label_trades = [t for t, trend_l in zip(trades, trend_bins) if trend_l == label]
        
        if not label_trades:
            continue
        
        wins = len([t for t in label_trades if t.actual_outcome == 'WIN'])
        total = len(label_trades)
        win_rate = wins / total if total > 0 else 0
        avg_pnl = np.mean([t.pnl_percent for t in label_trades if t.pnl_percent is not None])
        
        print(f"\n{label} Trend:")
        print(f"  Total: {total}")
        print(f"  Win Rate: {win_rate:.2%}")
        print(f"  Avg PnL: {avg_pnl:.2%}")


def main():
    """Main execution function"""
    
    print("="*80)
    print("ML-ENHANCED MEAN REVERSION BOT - MAINNET BACKTEST (CONSERVATIVE SETTINGS)")
    print("="*80)
    print("\n⚠️  Using REAL Binance mainnet data (no actual trades executed)")
    print("Capital: $100 | Leverage: 10x | Risk: 5% | Order: LIMIT (0.02% fee)")
    print("✅ IMPROVED SETTINGS: Lower leverage, wider stops, 2:1 reward:risk\n")
    
    # Step 1: Initialize the bot
    print("\n📦 Step 1: Initializing bot...")
    bot = MLMeanReversionBot()
    
    # Step 2: Fetch REAL mainnet data
    print("\n📊 Step 2: Fetching real mainnet data from Binance...")
    
    try:
        df = fetch_real_binance_data(symbol='BTCUSDT', interval='15m', days=90)
    except Exception as e:
        print(f"Could not fetch real data: {e}")
        print("Using sample data instead...")
        df = generate_sample_data(days=90, interval_minutes=15)
    
    # Step 3: Train the ML model
    print("\n🤖 Step 3: Training ML model on historical patterns...")
    df_with_features = bot.train_model(df, forward_periods=10)
    
    # Step 4: Run backtest with CONSERVATIVE parameters (IMPROVED SETTINGS)
    print("\n📈 Step 4: Running backtest with $100 @ 10x leverage (CONSERVATIVE)...")
    print("   Risk per trade: 5% (reduced from 15%)")
    print("   Stop Loss: 1.5% (15% of capital with 10x)")
    print("   Take Profit: 3% (30% of capital with 10x)")
    print("   Reward:Risk: 2:1 (improved from 1.5:1)")
    print("   Order Type: LIMIT (Maker fee: 0.02%)")
    print("   Trailing Stop: Enabled (moves to breakeven at 60% of TP)")

    results = bot.backtest(
        df_with_features,
        initial_capital=100,      # $100 starting capital
        leverage=10,              # 10x leverage (REDUCED from 20x for safety)
        risk_per_trade=0.05,      # Risk 5% of capital per trade (REDUCED from 15%)
        stop_loss_pct=0.015,      # 1.5% SL = 15% of capital with 10x (WIDENED from 0.8%)
        take_profit_pct=0.03,     # 3% TP = 30% of capital with 10x (INCREASED from 1.2%)
        use_limit_orders=True,    # Use limit orders (better fees)
        use_trailing_stop=True    # Enable trailing stop (NEW FEATURE)
    )
    
    # Step 5: Analyze results
    print("\n🔍 Step 5: Analyzing ML model performance...")
    analyze_ml_performance(bot, results)
    
    # Step 6: Visualize
    print("\n📊 Step 6: Creating visualizations...")
    plot_backtest_results(results, df_with_features)
    
    # Step 7: Save model
    print("\n💾 Step 7: Saving trained model...")
    bot.save_state(
        model_path='/mnt/user-data/outputs/ml_mean_reversion_model.pkl',
        trades_path='/mnt/user-data/outputs/trade_history.json'
    )
    
    # Additional insights
    print("\n" + "="*80)
    print("💡 KEY INSIGHTS")
    print("="*80)
    
    if results['total_trades'] > 0:
        expected_value = (results['win_rate'] * results['avg_win_pct'] + 
                         (1 - results['win_rate']) * results['avg_loss_pct'])
        
        print(f"\nExpected Value per Trade: {expected_value:.2%} of capital")
        print(f"With {results['win_rate']:.1%} win rate @ {results['avg_win_pct']:.1%} avg win")
        print(f"vs {1-results['win_rate']:.1%} loss rate @ {results['avg_loss_pct']:.1%} avg loss")
        
        print(f"\nWith $100 capital:")
        print(f"  • Expected profit per trade: ${expected_value * 100:.2f}")
        print(f"  • Break-even win rate: {abs(results['avg_loss_pct']) / (results['avg_win_pct'] + abs(results['avg_loss_pct'])):.1%}")
        
        # Compound growth projection
        if expected_value > 0:
            trades_per_week = 5  # Conservative estimate
            weeks = 12  # 3 months
            total_trades = trades_per_week * weeks
            projected_capital = 100 * ((1 + expected_value) ** total_trades)
            
            print(f"\n📊 Projection (if performance continues):")
            print(f"  • {trades_per_week} trades/week for {weeks} weeks = {total_trades} trades")
            print(f"  • Projected capital: ${projected_capital:.2f}")
            print(f"  • Total return: {(projected_capital/100 - 1):.1%}")
            print(f"\n  ⚠️  This is a projection, not a guarantee!")
    
    print("\n" + "="*80)
    print("✅ COMPLETE! Your ML-enhanced trading bot is ready.")
    print("="*80)
    print("\nNext steps:")
    print("1. Review the backtest results above")
    print("2. Check the visualization in backtest_results.png")
    print("3. Understand the win rate and profit factor")
    print("4. If satisfied, deploy for live trading with LIMIT orders")
    print("\n⚠️  Remember:")
    print("  • Start with the configured $100 capital")
    print("  • Monitor closely (check multiple times per day)")
    print("  • Past performance ≠ future results")
    print("  • With 20x leverage, 5% BTC move = 100% capital impact")
    print("  • Be prepared for volatility and drawdowns")


if __name__ == "__main__":
    main()
