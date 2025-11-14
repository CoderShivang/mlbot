"""
Quick Test Script - Verify bot setup with short backtest
=========================================================

This runs a quick 7-day backtest to verify everything is working correctly
before running longer backtests.
"""

from ml_mean_reversion_bot import MLMeanReversionBot
from binance.client import Client
import pandas as pd
from datetime import datetime, timedelta

def quick_test():
    """Run a quick 7-day backtest to verify setup"""

    print("="*80)
    print("QUICK TEST - 7 Day Backtest")
    print("="*80)
    print("\nThis will verify your setup is working correctly.\n")

    # Step 1: Initialize bot (no API keys needed for public data)
    print("📦 Step 1: Initializing bot...")
    bot = MLMeanReversionBot()

    # Step 2: Fetch 7 days of data
    print("\n📊 Step 2: Fetching 7 days of mainnet data from Binance...")
    try:
        client = Client()  # Public client, no keys needed

        start_time = datetime.now() - timedelta(days=7)
        start_str = start_time.strftime("%d %b %Y %H:%M:%S")

        klines = client.get_historical_klines(
            'BTCUSDT', '15m', start_str
        )

        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])

        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)

        df.set_index('timestamp', inplace=True)
        df = df[['open', 'high', 'low', 'close', 'volume']]

        print(f"✅ Fetched {len(df)} candles")
        print(f"   Period: {df.index[0]} to {df.index[-1]}")
        print(f"   BTC Price Range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")

    except Exception as e:
        print(f"❌ Error fetching data: {e}")
        return

    # Step 3: Train model (quick training on 7 days)
    print("\n🤖 Step 3: Training ML model (quick training)...")
    try:
        df_with_features = bot.train_model(df, forward_periods=10)
        print("✅ Model trained successfully")
    except Exception as e:
        print(f"❌ Error training model: {e}")
        import traceback
        traceback.print_exc()
        return

    # Step 4: Run quick backtest
    print("\n📈 Step 4: Running quick backtest...")
    print("Settings: $100 @ 10x leverage (CONSERVATIVE)")

    try:
        results = bot.backtest(
            df_with_features,
            initial_capital=100,
            leverage=10,
            risk_per_trade=0.05,
            stop_loss_pct=0.015,
            take_profit_pct=0.03,
            use_limit_orders=True,
            use_trailing_stop=True
        )

        # Print summary
        print("\n" + "="*80)
        print("QUICK TEST RESULTS")
        print("="*80)
        print(f"\n✅ Test completed successfully!")
        print(f"\nKey Metrics:")
        print(f"  • Total Trades: {results['total_trades']}")
        print(f"  • Win Rate: {results['win_rate']:.1%}")
        print(f"  • Final Capital: ${results['final_capital']:.2f}")
        print(f"  • Total Return: {results['total_return']:.2%}")
        print(f"  • Profit Factor: {results['profit_factor']:.2f}")

        if results['total_trades'] > 0:
            print(f"\n💰 Trade Performance:")
            print(f"  • Winning Trades: {results['winning_trades']}")
            print(f"  • Losing Trades: {results['losing_trades']}")
            print(f"  • Avg Win: ${results['avg_win_usd']:.2f} ({results['avg_win_pct']:.2%} of capital)")
            print(f"  • Avg Loss: ${results['avg_loss_usd']:.2f} ({results['avg_loss_pct']:.2%} of capital)")

            print(f"\n📊 Risk Metrics:")
            print(f"  • Max Drawdown: {results['max_drawdown']:.2%} (${results['max_drawdown_usd']:.2f})")
            print(f"  • Sharpe Ratio: {results['sharpe_ratio']:.2f}")

            print(f"\n💵 Fees & Execution:")
            print(f"  • Total Fees: ${results['total_fees_paid']:.2f}")
            print(f"  • Order Type: {results['order_type']}")
            print(f"  • Fee Rate: {results['fee_rate']:.2%}")
        else:
            print(f"\n⚠️  No trades generated in this 7-day period.")
            print(f"   This might happen if market conditions didn't meet signal criteria.")
            print(f"   Try running a longer backtest with more data.")

        print("\n" + "="*80)
        print("✅ SETUP VERIFIED - Ready for longer backtests!")
        print("="*80)
        print("\nNext steps:")
        print("1. Run full backtest: python example_usage.py")
        print("2. Review results and ML model performance")
        print("3. If satisfied, proceed to live trading")

        return results

    except Exception as e:
        print(f"❌ Error running backtest: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    quick_test()
