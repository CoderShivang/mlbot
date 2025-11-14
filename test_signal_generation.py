"""
Quick test to verify signal generation with balanced criteria
"""

import pandas as pd
import numpy as np
from trend_strategy_v2 import ResearchBackedTrendSignals, TrendFeatureEngineer

def test_signal_generation():
    """Test how many signals are generated with current criteria"""

    print("=" * 80)
    print("TESTING SIGNAL GENERATION")
    print("=" * 80)

    # Generate synthetic Bitcoin-like data
    np.random.seed(42)
    n_bars = 10000  # ~70 days of 15m data

    # Simulate realistic Bitcoin price action
    returns = np.random.normal(0.0001, 0.005, n_bars)  # Slight upward drift, 0.5% volatility
    close_prices = 100 * np.exp(np.cumsum(returns))

    # Generate OHLCV data
    df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=n_bars, freq='15min'),
        'open': close_prices * (1 + np.random.uniform(-0.002, 0.002, n_bars)),
        'high': close_prices * (1 + np.random.uniform(0, 0.005, n_bars)),
        'low': close_prices * (1 - np.random.uniform(0, 0.005, n_bars)),
        'close': close_prices,
        'volume': np.random.uniform(100, 1000, n_bars)
    })

    df.set_index('timestamp', inplace=True)

    print(f"\n📊 Generated {len(df):,} bars of synthetic data")
    print(f"   Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    print(f"   Average price: ${df['close'].mean():.2f}")

    # Calculate features
    print(f"\n🔧 Calculating features...")
    feature_engineer = TrendFeatureEngineer()
    df_features = feature_engineer.calculate_features(df)

    # Generate signals
    print(f"\n🎯 Generating signals with BREAKOUT-FOCUSED criteria...")
    print("   Criteria:")
    print("   - BREAKOUT: Must break 20-bar high/low (NEW!)")
    print("   - Momentum: Top 30% (quantile 0.70)")
    print("   - ADX: > 20")
    print("   - Volume: > 1.2x average")
    print("   - RSI: 50-80 for longs, 20-50 for shorts")

    long_signals = ResearchBackedTrendSignals.identify_long_trends(df_features)
    short_signals = ResearchBackedTrendSignals.identify_short_trends(df_features)

    long_count = long_signals.sum()
    short_count = short_signals.sum()
    total_signals = long_count + short_count

    signal_pct = (total_signals / len(df)) * 100

    print(f"\n✅ RESULTS:")
    print(f"   Long signals:  {long_count:,} ({(long_count/len(df)*100):.1f}% of bars)")
    print(f"   Short signals: {short_count:,} ({(short_count/len(df)*100):.1f}% of bars)")
    print(f"   Total signals: {total_signals:,} ({signal_pct:.1f}% of bars)")
    print(f"\n   Target: 5-10% of bars")

    if signal_pct < 1:
        print(f"\n   ❌ TOO STRICT: Only {signal_pct:.1f}% of bars (need 5-10%)")
        print("      Criteria are too restrictive, will result in 0 trades")
    elif signal_pct < 5:
        print(f"\n   ⚠️  CONSERVATIVE: {signal_pct:.1f}% of bars (target 5-10%)")
        print("      Might work but consider loosening slightly")
    elif signal_pct <= 15:
        print(f"\n   ✅ GOOD: {signal_pct:.1f}% of bars (target 5-10%)")
        print("      Should generate meaningful number of trades")
    elif signal_pct <= 30:
        print(f"\n   ⚠️  LIBERAL: {signal_pct:.1f}% of bars (target 5-10%)")
        print("      Consider tightening criteria for better quality")
    else:
        print(f"\n   ❌ TOO LOOSE: {signal_pct:.1f}% of bars (target 5-10%)")
        print("      Criteria are too permissive, quality will suffer")

    # Estimate annual trades for 15m timeframe
    bars_per_day = 96  # 15-minute bars
    bars_per_year = bars_per_day * 365
    estimated_annual_signals = (signal_pct / 100) * bars_per_year

    # Assume ~30% pass ML filtering and get executed
    estimated_annual_trades = estimated_annual_signals * 0.30

    print(f"\n📈 Estimated Annual Performance (15m timeframe):")
    print(f"   Signals per year: ~{estimated_annual_signals:.0f}")
    print(f"   Trades per year: ~{estimated_annual_trades:.0f} (after ML filtering)")

    if estimated_annual_trades < 20:
        print(f"   ⚠️  WARNING: Too few trades for statistical significance")
    elif estimated_annual_trades < 50:
        print(f"   ✅ Acceptable trade frequency")
    else:
        print(f"   ✅ Good trade frequency for robust statistics")

    print("\n" + "=" * 80)

if __name__ == '__main__':
    test_signal_generation()
