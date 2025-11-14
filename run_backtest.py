#!/usr/bin/env python3
"""
ML Mean Reversion Bot - CLI Backtest Tool
==========================================

Professional command-line interface for running backtests with various options.

Usage Examples:
  # Quick 3-day test
  python run_backtest.py --days 3

  # Standard 90-day backtest
  python run_backtest.py --days 90

  # Custom date range
  python run_backtest.py --start 01/11/2024 --end 14/11/2024

  # With custom parameters
  python run_backtest.py --days 30 --leverage 5 --risk 0.03

  # Disable ML training (use existing model)
  python run_backtest.py --days 7 --no-train
"""

import argparse
import sys
import json
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
from binance.client import Client

# Add src to path if needed
sys.path.insert(0, str(Path(__file__).parent))

from ml_mean_reversion_bot import MLMeanReversionBot
from trend_strategy_v2 import ResearchBackedTrendBot
from combined_strategy import CombinedStrategyBot
from enhanced_features import EnhancedFeatureEngineering
from dynamic_risk_management import DynamicRiskManager
from model_factory import ModelFactory

# ANSI color codes for pretty output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_header(text):
    """Print colored header"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text:^80}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}\n")


def print_success(text):
    """Print success message"""
    print(f"{Colors.GREEN}✅ {text}{Colors.ENDC}")


def print_info(text):
    """Print info message"""
    print(f"{Colors.CYAN}ℹ️  {text}{Colors.ENDC}")


def print_warning(text):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠️  {text}{Colors.ENDC}")


def print_error(text):
    """Print error message"""
    print(f"{Colors.RED}❌ {text}{Colors.ENDC}")


def parse_date(date_str):
    """Parse date string in dd/mm/yyyy format"""
    try:
        return datetime.strptime(date_str, "%d/%m/%Y")
    except ValueError:
        try:
            return datetime.strptime(date_str, "%d/%m/%y")
        except ValueError:
            raise ValueError(f"Invalid date format: {date_str}. Use dd/mm/yyyy or dd/mm/yy")


def fetch_binance_data(symbol, start_date, end_date, interval='15m'):
    """Fetch data from Binance with progress"""
    print_info(f"Fetching {symbol} data from Binance mainnet...")
    print(f"   Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"   Interval: {interval}")

    try:
        client = Client()  # Public client

        start_str = start_date.strftime("%d %b %Y %H:%M:%S")
        end_str = end_date.strftime("%d %b %Y %H:%M:%S")

        print(f"\n   Downloading... ", end='', flush=True)

        klines = client.get_historical_klines(symbol, interval, start_str, end_str)

        print(f"{Colors.GREEN}Done!{Colors.ENDC}")

        # Convert to DataFrame
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

        print_success(f"Fetched {len(df):,} candles")
        print(f"   Price Range: ${df['close'].min():,.2f} - ${df['close'].max():,.2f}")

        return df

    except Exception as e:
        print_error(f"Failed to fetch data: {e}")
        raise


def train_model_with_progress(bot, df, forward_periods=10):
    """Train ML model with progress indicators"""
    print_header("ML MODEL TRAINING")

    # Detect bot type and use appropriate feature engineer
    from trend_strategy_v2 import ResearchBackedTrendBot
    is_trend_bot = isinstance(bot, ResearchBackedTrendBot)

    if is_trend_bot:
        print_info("Calculating trend following features (25 indicators)...")
        print("   └─ Momentum: ROC, Rate of Change, Momentum scores")
        print("   └─ Trend strength: ADX, DI, MACD")
        print("   └─ Breakouts: High/Low breakouts, Distance metrics")
        print("   └─ Volume: Volume ratio, OBV, Volume surge")

        # Use trend feature engineer
        df_features = bot.feature_engineer.calculate_features(df)
        feature_cols = bot.feature_names
    else:
        print_info("Calculating technical features (24 indicators)...")
        print("   └─ Mean reversion: RSI, Bollinger Bands, Z-scores")
        print("   └─ Market context: Volatility, Trend, Volume")
        print("   └─ Pattern features: Momentum, S/R, Consolidation")
        print("   └─ Feature interactions: RSI×BB, Trend×Vol, Volume×Mom")

        # Use mean reversion feature engineer
        from ml_mean_reversion_bot import FeatureEngineering
        df_features = FeatureEngineering.calculate_features(df)
        feature_cols = FeatureEngineering.get_feature_list()

    print_success(f"Features calculated for {len(df_features):,} candles")

    # Generate training labels
    print_info("Generating training labels...")
    print(f"   └─ Checking forward returns over {forward_periods} periods")

    # Calculate forward returns
    df_features['forward_return'] = df_features['close'].shift(-forward_periods) / df_features['close'] - 1

    # Identify signals based on bot type
    if is_trend_bot:
        df_features['long_signal'] = bot.signal_generator.identify_long_trends(df_features)
        df_features['short_signal'] = bot.signal_generator.identify_short_trends(df_features)
    else:
        from ml_mean_reversion_bot import MeanReversionSignals
        signals = MeanReversionSignals()
        df_features['long_signal'] = signals.identify_long_setups(df_features)
        df_features['short_signal'] = signals.identify_short_setups(df_features)

    # Create labels
    df_features['is_setup'] = df_features['long_signal'] | df_features['short_signal']

    # Label success (1) or failure (0)
    threshold = 0.01  # 1% return threshold
    df_features['success'] = 0

    long_mask = df_features['long_signal']
    short_mask = df_features['short_signal']

    df_features.loc[long_mask, 'success'] = (df_features.loc[long_mask, 'forward_return'] > threshold).astype(int)
    df_features.loc[short_mask, 'success'] = (df_features.loc[short_mask, 'forward_return'] < -threshold).astype(int)

    # Count training samples
    training_samples = df_features['is_setup'].sum()
    print_success(f"Generated {training_samples:,} training samples from historical signals")

    if training_samples < 50:
        print_warning("Few training samples - model may not be reliable")
        print("   Consider using more historical data (--days 90)")

    # Get model info
    model_type = bot.ml_model.model_type if hasattr(bot.ml_model, 'model_type') else 'gradientboost'
    model_info = ModelFactory.get_model_info(model_type)

    # Train model
    print_info(f"Training {model_info['name']}...")
    print(f"   └─ Model: {model_info['name']}")
    print(f"   └─ Speed: {model_info['speed']}")
    print(f"   └─ Accuracy: {model_info['accuracy']}")
    print(f"   └─ Overfitting Risk: {model_info['overfitting_risk']}")

    # Filter feature_cols to only include columns that exist in df_features
    # (feature_cols already set above based on bot type)
    feature_cols = [col for col in feature_cols if col in df_features.columns]

    # Prepare training data
    train_data = df_features[df_features['is_setup']].copy()
    train_data = train_data.dropna(subset=['forward_return'] + feature_cols)

    X = train_data[feature_cols]
    y = train_data['success']

    # Train-test split
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"   └─ Training set: {len(X_train):,} samples")
    print(f"   └─ Test set: {len(X_test):,} samples")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create model using factory (respects --model flag)
    from sklearn.metrics import classification_report, accuracy_score

    print("\n   Training in progress...")

    # Create the correct model type
    model = ModelFactory.create_model(model_type)

    # Unfortunately sklearn's GBClassifier doesn't have incremental training
    # But we can show a spinner
    import itertools
    import threading
    import time

    done = False

    def animate():
        for c in itertools.cycle(['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']):
            if done:
                break
            print(f'\r   {Colors.CYAN}{c}{Colors.ENDC} Training model...', end='', flush=True)
            time.sleep(0.1)

    t = threading.Thread(target=animate)
    t.daemon = True
    t.start()

    # Train
    model.fit(X_train_scaled, y_train)

    done = True
    t.join()
    print(f'\r   {Colors.GREEN}✓{Colors.ENDC} Training complete!                    ')

    # Evaluate
    print_info("Evaluating model performance...")

    train_acc = accuracy_score(y_train, model.predict(X_train_scaled))
    test_acc = accuracy_score(y_test, model.predict(X_test_scaled))

    print(f"   └─ Training Accuracy: {train_acc:.2%}")
    print(f"   └─ Testing Accuracy:  {test_acc:.2%}")

    if test_acc > 0.65:
        print_success("Model shows good predictive power!")
    elif test_acc > 0.55:
        print_warning("Model shows moderate predictive power")
    else:
        print_warning("Model shows weak predictive power - consider more data")

    # Feature importance (top 5) - only for models that support it
    if hasattr(model, 'feature_importances_'):
        print_info("Top 5 Most Important Features:")
        importances = model.feature_importances_
        feature_importance = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)

        for i, (feature, importance) in enumerate(feature_importance[:5], 1):
            bar_length = int(importance * 30)
            bar = '█' * bar_length + '░' * (30 - bar_length)
            print(f"   {i}. {feature:25s} {bar} {importance:.3f}")
    elif hasattr(model, 'estimators_'):
        print_info("Ensemble model - feature importance averaged across estimators")
    else:
        print_info("Feature importance not available for this model type")

    # Update bot's ML model
    bot.ml_model.model = model
    bot.ml_model.scaler = scaler
    bot.ml_model.feature_names = feature_cols
    bot.ml_model.is_trained = True

    print_success("Model training complete and ready for backtest!")

    return df_features


def run_backtest_with_progress(bot, df, args):
    """Run backtest with detailed progress"""
    print_header("RUNNING BACKTEST")

    print_info("Backtest Configuration:")
    print(f"   Capital:          ${args.capital:.0f}")
    print(f"   Leverage:         {args.leverage}x")
    print(f"   Risk per trade:   {args.risk:.1%}")
    print(f"   Stop Loss:        {args.stop_loss:.2%} ({args.stop_loss * args.leverage:.1%} of capital)")
    print(f"   Take Profit:      {args.take_profit:.2%} ({args.take_profit * args.leverage:.1%} of capital)")
    print(f"   Reward:Risk:      {args.take_profit/args.stop_loss:.1f}:1")
    print(f"   Order Type:       {'LIMIT (Maker: 0.02%)' if args.use_limit else 'MARKET (Taker: 0.05%)'}")
    print(f"   Trailing Stop:    {'Enabled' if args.trailing_stop else 'Disabled'}")

    print("\n   Running backtest simulation...\n")

    results = bot.backtest(
        df,
        initial_capital=args.capital,
        leverage=args.leverage,
        risk_per_trade=args.risk,
        stop_loss_pct=args.stop_loss,
        take_profit_pct=args.take_profit,
        use_limit_orders=args.use_limit,
        use_trailing_stop=args.trailing_stop
    )

    return results


def display_results(results):
    """Display backtest results in a nice format"""
    print_header("BACKTEST RESULTS")

    # Overall Performance
    print(f"{Colors.BOLD}Overall Performance{Colors.ENDC}")
    print(f"{'─'*80}")

    pnl_color = Colors.GREEN if results['total_return'] > 0 else Colors.RED
    print(f"Initial Capital:    ${results['initial_capital']:.2f}")
    print(f"Final Capital:      {pnl_color}${results['final_capital']:.2f}{Colors.ENDC}")
    print(f"Total Return:       {pnl_color}{results['total_return']:.2%}{Colors.ENDC}")

    # Calculate ROI considering leverage
    roi_with_leverage = results['total_return'] * results['leverage']
    print(f"ROI with {results['leverage']}x:     {pnl_color}{roi_with_leverage:.2%}{Colors.ENDC}")

    # Trade Statistics
    print(f"\n{Colors.BOLD}Trade Statistics{Colors.ENDC}")
    print(f"{'─'*80}")
    print(f"Total Trades:       {results['total_trades']}")

    if results['total_trades'] == 0:
        print_warning("No trades executed!")
        print("   Possible reasons:")
        print("   • ML model rejected all signals (need to train model first)")
        print("   • No signals met entry criteria in this period")
        print("   • Try: --no-train flag or use more historical data")
        return

    print(f"Winning Trades:     {Colors.GREEN}{results['winning_trades']}{Colors.ENDC}")
    print(f"Losing Trades:      {Colors.RED}{results['losing_trades']}{Colors.ENDC}")

    wr_color = Colors.GREEN if results['win_rate'] >= 0.6 else Colors.YELLOW if results['win_rate'] >= 0.5 else Colors.RED
    print(f"Win Rate:           {wr_color}{results['win_rate']:.2%}{Colors.ENDC}")

    print(f"Avg Win:            {Colors.GREEN}${results.get('avg_win_usd', 0):.2f} ({results.get('avg_win_pct', 0):.2%}){Colors.ENDC}")
    print(f"Avg Loss:           {Colors.RED}${results.get('avg_loss_usd', 0):.2f} ({results.get('avg_loss_pct', 0):.2%}){Colors.ENDC}")

    # Risk Metrics
    print(f"\n{Colors.BOLD}Risk Metrics{Colors.ENDC}")
    print(f"{'─'*80}")

    pf_color = Colors.GREEN if results['profit_factor'] > 2.0 else Colors.YELLOW if results['profit_factor'] > 1.5 else Colors.RED
    print(f"Profit Factor:      {pf_color}{results['profit_factor']:.2f}{Colors.ENDC}")

    sr_color = Colors.GREEN if results['sharpe_ratio'] > 1.5 else Colors.YELLOW if results['sharpe_ratio'] > 1.0 else Colors.RED
    print(f"Sharpe Ratio:       {sr_color}{results['sharpe_ratio']:.2f}{Colors.ENDC}")

    # Use 'max_drawdown' instead of 'max_drawdown_pct'
    max_dd = results.get('max_drawdown', 0)
    max_dd_usd = results.get('max_drawdown_usd', 0)
    print(f"Max Drawdown:       {Colors.RED}{max_dd:.2%} (${max_dd_usd:.2f}){Colors.ENDC}")
    print(f"Max Consecutive:    {results.get('max_consecutive_losses', 0)} losses")

    # Fees
    print(f"\n{Colors.BOLD}Execution Details{Colors.ENDC}")
    print(f"{'─'*80}")
    print(f"Total Fees Paid:    ${results['total_fees_paid']:.2f}")
    print(f"Order Type:         {results['order_type']}")
    print(f"Fee Rate:           {results['fee_rate']:.2%}")
    if 'fill_rate' in results:
        print(f"Fill Rate:          {results['fill_rate']:.1%}")

    # Assessment
    print(f"\n{Colors.BOLD}Performance Assessment{Colors.ENDC}")
    print(f"{'─'*80}")

    if results['win_rate'] >= 0.65 and results['profit_factor'] >= 2.0 and results['sharpe_ratio'] >= 1.5:
        print_success("EXCELLENT - Strategy shows strong performance!")
    elif results['win_rate'] >= 0.55 and results['profit_factor'] >= 1.5:
        print_info("GOOD - Strategy shows positive performance")
    elif results['win_rate'] >= 0.50 and results['profit_factor'] >= 1.2:
        print_warning("FAIR - Strategy is marginally profitable")
    else:
        print_error("POOR - Strategy needs improvement")

    print(f"\n{'='*80}\n")


def walk_forward_backtest(bot, df, args):
    """
    Proper walk-forward analysis without look-ahead bias

    Trains on PAST data only, tests on FUTURE unseen data
    Retrains the model periodically as it "walks forward" through time

    Args:
        bot: MLMeanReversionBot instance
        df: Full historical DataFrame
        args: Command-line arguments with walk-forward parameters

    Returns:
        Aggregated results from all windows
    """
    print_header("WALK-FORWARD ANALYSIS")
    print_info(f"Training Window: {args.train_window} days")
    print_info(f"Testing Window: {args.test_window} days")
    print(f"{'─'*80}\n")

    from ml_mean_reversion_bot import FeatureEngineering

    # Calculate features once for entire dataset
    df = FeatureEngineering.calculate_features(df)

    # Determine window boundaries
    total_days = len(df)
    train_window_bars = args.train_window * 96  # ~96 15min bars per day
    test_window_bars = args.test_window * 96

    # Need at least train_window + 100 bars for indicators
    min_start = train_window_bars + 100

    if total_days < min_start + test_window_bars:
        print_error(f"Not enough data! Need at least {(min_start + test_window_bars) / 96:.0f} days")
        print(f"   You have: {total_days / 96:.0f} days")
        print(f"   Try: --days {int((min_start + test_window_bars) / 96) + 10}")
        sys.exit(1)

    # Walk through time
    window_results = []
    window_num = 1
    current_pos = min_start

    while current_pos + test_window_bars <= total_days:
        print(f"\n{Colors.CYAN}{'='*80}{Colors.ENDC}")
        print(f"{Colors.CYAN}{Colors.BOLD}Window #{window_num}{Colors.ENDC}")
        print(f"{Colors.CYAN}{'='*80}{Colors.ENDC}\n")

        # Define train and test periods
        train_start = current_pos - train_window_bars
        train_end = current_pos
        test_start = current_pos
        test_end = current_pos + test_window_bars

        train_df = df.iloc[train_start:train_end].copy()
        test_df = df.iloc[test_start:test_end].copy()

        train_dates = f"{train_df.index[0].strftime('%Y-%m-%d')} to {train_df.index[-1].strftime('%Y-%m-%d')}"
        test_dates = f"{test_df.index[0].strftime('%Y-%m-%d')} to {test_df.index[-1].strftime('%Y-%m-%d')}"

        print_info(f"Training Period: {train_dates} ({len(train_df)} bars)")
        print_info(f"Testing Period:  {test_dates} ({len(test_df)} bars)")

        # Train model on historical data ONLY
        try:
            train_model_with_progress(bot, train_df, args.forward_periods)
        except Exception as e:
            print_warning(f"Training failed for window {window_num}: {e}")
            print("   Skipping this window...")
            current_pos += test_window_bars
            window_num += 1
            continue

        # Test on future unseen data
        print_info(f"\nTesting on unseen future data...")

        results = bot.backtest(
            test_df,
            initial_capital=args.capital,
            leverage=args.leverage,
            risk_per_trade=args.risk,
            stop_loss_pct=args.stop_loss,
            take_profit_pct=args.take_profit,
            use_limit_orders=not args.use_market,
            use_trailing_stop=not args.no_trailing
        )

        # Store window results
        results['window_num'] = window_num
        results['train_start'] = train_dates.split(' to ')[0]
        results['train_end'] = train_dates.split(' to ')[1]
        results['test_start'] = test_dates.split(' to ')[0]
        results['test_end'] = test_dates.split(' to ')[1]
        window_results.append(results)

        # Display window summary
        print(f"\n{Colors.BOLD}Window #{window_num} Results:{Colors.ENDC}")
        pnl_color = Colors.GREEN if results['total_return'] > 0 else Colors.RED
        print(f"   Trades: {results['total_trades']}")
        print(f"   Win Rate: {results['win_rate']:.1%}")
        print(f"   Return: {pnl_color}{results['total_return']:.2%}{Colors.ENDC}")
        print(f"   Sharpe: {results['sharpe_ratio']:.2f}")

        # Move forward
        current_pos += test_window_bars
        window_num += 1

    # Aggregate all windows
    print(f"\n\n{Colors.GREEN}{'='*80}{Colors.ENDC}")
    print(f"{Colors.GREEN}{Colors.BOLD}WALK-FORWARD AGGREGATE RESULTS{Colors.ENDC}")
    print(f"{Colors.GREEN}{'='*80}{Colors.ENDC}\n")

    # Check if we have any results
    if not window_results:
        print_error("All walk-forward windows failed!")
        print_error("This usually means:")
        print("  - The strategy couldn't find any trade setups")
        print("  - Training data was insufficient")
        print("  - There was an error in the bot's logic")
        print("\nTry:")
        print("  1. Use a different strategy: --strat meanrev or --strat combined")
        print("  2. Use more data: --days 365 or more")
        print("  3. Lower confidence threshold: --min-confidence 0.60")
        sys.exit(1)

    # Combine all trades
    all_trades = []
    for wr in window_results:
        all_trades.extend(wr.get('trades', []))

    # Calculate aggregate metrics
    total_windows = len(window_results)
    winning_windows = sum(1 for wr in window_results if wr['total_return'] > 0)

    total_trades = sum(wr['total_trades'] for wr in window_results)
    total_wins = sum(wr['winning_trades'] for wr in window_results)

    avg_return = np.mean([wr['total_return'] for wr in window_results])
    avg_sharpe = np.mean([wr['sharpe_ratio'] for wr in window_results])
    avg_win_rate = np.mean([wr['win_rate'] for wr in window_results])

    # Best and worst windows
    best_window = max(window_results, key=lambda x: x['total_return'])
    worst_window = min(window_results, key=lambda x: x['total_return'])

    print_info(f"Total Windows Tested: {total_windows}")
    print(f"   Profitable Windows: {Colors.GREEN}{winning_windows}{Colors.ENDC} ({winning_windows/total_windows:.1%})")
    print(f"   Unprofitable Windows: {Colors.RED}{total_windows - winning_windows}{Colors.ENDC}\n")

    print_info(f"Aggregate Performance:")
    print(f"   Total Trades: {total_trades}")
    print(f"   Winning Trades: {total_wins}")
    wr_color = Colors.GREEN if avg_win_rate >= 0.55 else Colors.YELLOW if avg_win_rate >= 0.50 else Colors.RED
    print(f"   Average Win Rate: {wr_color}{avg_win_rate:.1%}{Colors.ENDC}")
    ret_color = Colors.GREEN if avg_return > 0 else Colors.RED
    print(f"   Average Return per Window: {ret_color}{avg_return:.2%}{Colors.ENDC}")
    sr_color = Colors.GREEN if avg_sharpe > 1.0 else Colors.YELLOW if avg_sharpe > 0.5 else Colors.RED
    print(f"   Average Sharpe Ratio: {sr_color}{avg_sharpe:.2f}{Colors.ENDC}\n")

    print_info(f"Best Window:")
    print(f"   Window #{best_window['window_num']}: {Colors.GREEN}{best_window['total_return']:+.2%}{Colors.ENDC}")
    print(f"   Period: {best_window['test_start']} to {best_window['test_end']}\n")

    print_info(f"Worst Window:")
    print(f"   Window #{worst_window['window_num']}: {Colors.RED}{worst_window['total_return']:+.2%}{Colors.ENDC}")
    print(f"   Period: {worst_window['test_start']} to {worst_window['test_end']}\n")

    # Calculate cumulative P&L from all trades (CORRECT calculation)
    total_pnl_usd = sum([t.get('net_pnl_usd', 0) for t in all_trades])
    cumulative_return = total_pnl_usd / args.capital  # Actual return based on USD P&L
    final_capital = args.capital + total_pnl_usd

    # Create aggregate results structure
    aggregate_results = {
        'total_windows': total_windows,
        'profitable_windows': winning_windows,
        'total_trades': total_trades,
        'winning_trades': total_wins,
        'win_rate': avg_win_rate,
        'total_return': cumulative_return,  # FIXED: Use cumulative, not average
        'avg_return_per_window': avg_return,  # Keep for reference
        'sharpe_ratio': avg_sharpe,
        'trades': all_trades,
        'window_results': window_results,
        'initial_capital': args.capital,
        'leverage': args.leverage,
        'final_capital': final_capital,  # FIXED: Actual final capital
        'total_pnl_usd': total_pnl_usd,  # Add explicit USD P&L
        'profit_factor': np.mean([wr.get('profit_factor', 1) for wr in window_results]),
        'max_drawdown': max([wr.get('max_drawdown', 0) for wr in window_results]),
        'max_drawdown_usd': max([wr.get('max_drawdown_usd', 0) for wr in window_results]),
        'total_fees_paid': sum([wr.get('total_fees_paid', 0) for wr in window_results]),
        'avg_win_usd': np.mean([wr.get('avg_win_usd', 0) for wr in window_results if wr.get('avg_win_usd')]),
        'avg_loss_usd': np.mean([wr.get('avg_loss_usd', 0) for wr in window_results if wr.get('avg_loss_usd')]),
        'avg_win_pct': np.mean([wr.get('avg_win_pct', 0) for wr in window_results if wr.get('avg_win_pct')]),
        'avg_loss_pct': np.mean([wr.get('avg_loss_pct', 0) for wr in window_results if wr.get('avg_loss_pct')]),
        'max_consecutive_losses': max([wr.get('max_consecutive_losses', 0) for wr in window_results]),
        'order_type': 'LIMIT' if not args.use_market else 'MARKET',
        'fee_rate': 0.0002 if not args.use_market else 0.0005,
        'losing_trades': total_trades - total_wins
    }

    return aggregate_results


def save_detailed_results(results: dict, bot, output_path: Path, args):
    """
    Save detailed backtest results including ML decision data for dashboard

    Args:
        results: Backtest results dictionary
        bot: MLMeanReversionBot instance
        output_path: Path to save JSON file
        args: Command-line arguments
    """
    # Extract trade details with ML decision information
    trades_data = []

    for trade_info in results.get('trades', []):
        setup = trade_info['setup']

        # Serialize TradeSetup data
        trade_detail = {
            'timestamp': setup.timestamp.isoformat() if hasattr(setup.timestamp, 'isoformat') else str(setup.timestamp),
            'direction': setup.direction,
            'entry_price': trade_info['entry_price'],
            'exit_price': trade_info['exit_price'],
            'outcome': setup.actual_outcome,
            'pnl_usd': trade_info['net_pnl_usd'],
            'pnl_pct': setup.pnl_percent,
            'position_size': trade_info['position_size_usd'],
            'fees': trade_info['fees_usd'],
            'hit_type': trade_info['hit_type'],

            # ML Decision Factors
            'ml_confidence': setup.confidence_score,
            'ml_success_prob': setup.predicted_success_prob,

            # Market Features at Entry
            'features': {
                'rsi': float(setup.rsi),
                'bb_position': float(setup.bb_position),
                'zscore': float(setup.zscore),
                'volatility_regime': setup.volatility_regime,
                'trend_strength': float(setup.trend_strength),
                'volume_ratio': float(setup.volume_ratio),
                'recent_drawdown': float(setup.recent_drawdown),
                'pattern_momentum': float(setup.pattern_momentum),
                'support_resistance_proximity': float(setup.support_resistance_proximity),
                'consolidation_duration': int(setup.consolidation_duration)
            },

            # Similar Historical Trades
            'similar_trades': setup.similar_trades if setup.similar_trades else []
        }

        trades_data.append(trade_detail)

    # Prepare summary statistics
    summary = {
        'backtest_date': datetime.now().isoformat(),
        'parameters': {
            'symbol': args.symbol,
            'interval': args.interval,
            'days': getattr(args, 'days', None),
            'start_date': getattr(args, 'start', None),
            'end_date': getattr(args, 'end', None),
            'initial_capital': args.capital,
            'leverage': args.leverage,
            'risk_per_trade': args.risk,
            'stop_loss_pct': args.stop_loss,
            'take_profit_pct': args.take_profit,
            'use_limit_orders': not args.use_market,
            'trailing_stop': not args.no_trailing
        },
        'results': {
            'total_trades': results['total_trades'],
            'winning_trades': results['winning_trades'],
            'losing_trades': results['losing_trades'],
            'win_rate': results['win_rate'],
            'total_return': results['total_return'],
            'final_capital': results['final_capital'],
            'profit_factor': results['profit_factor'],
            'sharpe_ratio': results['sharpe_ratio'],
            'max_drawdown': results.get('max_drawdown', 0),
            'max_drawdown_usd': results.get('max_drawdown_usd', 0),
            'total_fees_paid': results['total_fees_paid'],
            'avg_win_usd': results.get('avg_win_usd', 0),
            'avg_loss_usd': results.get('avg_loss_usd', 0),
            'avg_win_pct': results.get('avg_win_pct', 0),
            'avg_loss_pct': results.get('avg_loss_pct', 0)
        },
        'trades': trades_data
    }

    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)


def initialize_bot_for_strategy(strategy: str, model_type: str, min_confidence: float):
    """
    Initialize the appropriate bot based on strategy choice

    Args:
        strategy: 'meanrev', 'extremetrends', or 'combined'
        model_type: ML model type
        min_confidence: Minimum confidence threshold

    Returns:
        Initialized bot instance
    """
    print_info(f"Strategy: {Colors.CYAN}{strategy.upper()}{Colors.ENDC}")
    print_info(f"ML Model: {Colors.CYAN}{model_type.upper()}{Colors.ENDC}")
    print_info(f"Min Confidence: {Colors.CYAN}{min_confidence:.0%}{Colors.ENDC}")

    if strategy == 'meanrev':
        print_info("Initializing Mean Reversion Bot...")
        bot = MLMeanReversionBot()
        bot.ml_model.model_type = model_type
        bot.ml_model.min_confidence = min_confidence
        # Replace model with enhanced version
        from model_factory import EnhancedMLModel
        bot.ml_model = EnhancedMLModel(model_type, min_confidence)
        bot.ml_model.feature_names = bot.feature_engineer.get_feature_list()
        return bot

    elif strategy == 'extremetrends':
        print_info("Initializing Research-Backed Trend Following Bot...")
        bot = ResearchBackedTrendBot(model_type, min_confidence)
        return bot

    elif strategy == 'combined':
        print_info("Initializing Combined Strategy Bot...")
        # Initialize both sub-bots
        meanrev_bot = MLMeanReversionBot()
        from model_factory import EnhancedMLModel
        meanrev_bot.ml_model = EnhancedMLModel(model_type, min_confidence)
        meanrev_bot.ml_model.feature_names = meanrev_bot.feature_engineer.get_feature_list()

        trend_bot = ResearchBackedTrendBot(model_type, min_confidence)

        # Create combined bot
        bot = CombinedStrategyBot(meanrev_bot, trend_bot, min_confidence)
        return bot

    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def enhance_features_for_strategy(df: pd.DataFrame, strategy: str, interval: str = '15m') -> pd.DataFrame:
    """
    Add enhanced features based on strategy

    Args:
        df: Base OHLCV DataFrame
        strategy: Strategy type
        interval: Candle interval

    Returns:
        DataFrame with enhanced features
    """
    print_info("Calculating enhanced features...")

    # Add market regime detection (ALL strategies need this)
    df = EnhancedFeatureEngineering.detect_market_regime(df)

    # Add multi-timeframe features
    df = EnhancedFeatureEngineering.add_multi_timeframe_features(df, interval)

    # Add temporal features
    df = EnhancedFeatureEngineering.add_temporal_features(df)

    # Add advanced patterns
    df = EnhancedFeatureEngineering.add_advanced_patterns(df)

    # Add volatility percentile
    df = EnhancedFeatureEngineering.add_volatility_percentile(df)

    print_success("Enhanced features calculated")

    return df


def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(
        description='ML Mean Reversion Bot - Backtest Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Quick 3-day test:
    python run_backtest.py --days 3

  Standard backtest:
    python run_backtest.py --days 90

  Custom date range:
    python run_backtest.py --start 01/11/2024 --end 14/11/2024

  With custom parameters:
    python run_backtest.py --days 30 --leverage 5 --risk 0.03 --capital 200

  Skip ML training (use existing model):
    python run_backtest.py --days 7 --no-train

Notes:
  - Date format: dd/mm/yyyy or dd/mm/yy
  - Use either --days OR --start/--end, not both
  - Model is saved to /mnt/user-data/outputs/
  - Dashboard reads from this same location
        """
    )

    # Date options
    date_group = parser.add_mutually_exclusive_group(required=True)
    date_group.add_argument('--days', type=int, metavar='N',
                           help='Number of days to backtest (e.g., --days 90)')
    date_group.add_argument('--start', type=str, metavar='DD/MM/YYYY',
                           help='Start date in dd/mm/yyyy format (requires --end)')

    parser.add_argument('--end', type=str, metavar='DD/MM/YYYY',
                       help='End date in dd/mm/yyyy format (requires --start)')

    # Symbol
    parser.add_argument('--symbol', type=str, default='BTCUSDT',
                       help='Trading pair (default: BTCUSDT)')
    parser.add_argument('--interval', type=str, default='15m',
                       choices=['1m', '5m', '15m', '1h', '4h'],
                       help='Candle interval (default: 15m)')

    # Trading parameters
    parser.add_argument('--capital', type=float, default=100,
                       help='Initial capital in USDT (default: 100)')
    parser.add_argument('--leverage', type=int, default=10,
                       help='Leverage multiplier (default: 10)')
    parser.add_argument('--risk', type=float, default=0.05,
                       help='Risk per trade as decimal (default: 0.05 = 5%%)')
    parser.add_argument('--stop-loss', type=float, default=0.015,
                       help='Stop loss %% (default: 0.015 = 1.5%%)')
    parser.add_argument('--take-profit', type=float, default=0.03,
                       help='Take profit %% (default: 0.03 = 3%%)')

    # Order options
    parser.add_argument('--use-market', action='store_true',
                       help='Use market orders instead of limit orders (higher fees)')
    parser.add_argument('--no-trailing', action='store_true',
                       help='Disable trailing stop')

    # Strategy selection (NEW!)
    parser.add_argument('--strat', '--strategy', type=str, default='meanrev',
                       choices=['meanrev', 'extremetrends', 'combined'],
                       help='''Strategy to use:
  meanrev       = Mean reversion (buy dips, sell rips)
  extremetrends = Trend following (ride strong moves)
  combined      = Both (regime-adaptive)
Default: meanrev''')

    # ML Model selection (NEW!)
    parser.add_argument('--model', type=str, default='gradientboost',
                       choices=['gradientboost', 'randomforest', 'xgboost', 'ensemble'],
                       help='''ML model type:
  gradientboost = Default, fast, good performance
  randomforest  = More robust, less overfitting
  xgboost       = Best performance (requires: pip install xgboost)
  ensemble      = Combines all models (slowest, most accurate)
Default: gradientboost''')

    # ML options
    parser.add_argument('--no-train', action='store_true',
                       help='Skip ML training (use existing model)')
    parser.add_argument('--forward-periods', type=int, default=10,
                       help='Forward periods for ML labels (default: 10)')
    parser.add_argument('--min-confidence', type=float, default=0.65,
                       help='Minimum ML confidence to take trades (default: 0.65 = 65%%)')

    # Walk-forward analysis options
    parser.add_argument('--walk-forward', action='store_true',
                       help='Use walk-forward analysis (proper backtesting without look-ahead bias)')
    parser.add_argument('--train-window', type=int, default=180,
                       help='Days of historical data to train on (default: 180)')
    parser.add_argument('--test-window', type=int, default=30,
                       help='Days to test forward after training (default: 30)')

    # Output options
    parser.add_argument('--output-dir', type=str, default='./outputs',
                       help='Output directory for results (default: ./outputs)')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save model and results')
    parser.add_argument('--dashboard', action='store_true',
                       help='Auto-launch dashboard after backtest completes')

    args = parser.parse_args()

    # Validate date arguments
    if args.start and not args.end:
        parser.error("--start requires --end")
    if args.end and not args.start:
        parser.error("--end requires --start")

    # Parse dates
    if args.days:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=args.days)
    else:
        try:
            start_date = parse_date(args.start)
            end_date = parse_date(args.end)

            if start_date >= end_date:
                print_error("Start date must be before end date")
                sys.exit(1)

        except ValueError as e:
            print_error(str(e))
            sys.exit(1)

    # Print header
    strategy_names = {
        'meanrev': 'MEAN REVERSION',
        'extremetrends': 'EXTREME TREND FOLLOWING',
        'combined': 'COMBINED STRATEGY'
    }
    print_header(f"{strategy_names.get(args.strat, 'TRADING')} BOT - BACKTEST")

    print_info(f"Symbol: {args.symbol}")
    print_info(f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} ({(end_date - start_date).days} days)")
    print_info(f"Interval: {args.interval}")

    # Initialize bot based on strategy choice
    bot = initialize_bot_for_strategy(args.strat, args.model, args.min_confidence)
    print_success("Bot initialized")

    # Fetch data
    try:
        df = fetch_binance_data(args.symbol, start_date, end_date, args.interval)
    except Exception as e:
        print_error(f"Failed to fetch data: {e}")
        sys.exit(1)

    # Add enhanced features (ALL strategies need this)
    try:
        df = enhance_features_for_strategy(df, args.strat, args.interval)
    except Exception as e:
        print_error(f"Feature calculation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Run backtest (walk-forward or standard)
    try:
        if args.walk_forward:
            # Walk-forward analysis (proper backtesting)
            print_info(f"Using walk-forward analysis (train on past, test on future)")
            results = walk_forward_backtest(bot, df, args)
        else:
            # Standard backtest (has look-ahead bias)
            if not args.no_train:
                print_header("ML MODEL TRAINING")

                # Train based on strategy type
                if args.strat == 'combined':
                    bot.train_models(df, args.forward_periods)
                else:
                    # Calculate base features first
                    from ml_mean_reversion_bot import FeatureEngineering
                    df = FeatureEngineering.calculate_features(df)

                    # Train the specific bot
                    bot.train_model(df, args.forward_periods)

                print_success("Model training complete")
            else:
                print_warning("Skipping ML training - using existing model (if available)")

                # Still need to calculate base features
                print_info("Calculating base features for backtest...")
                from ml_mean_reversion_bot import FeatureEngineering
                df = FeatureEngineering.calculate_features(df)
                print_success("Base features calculated")

            results = run_backtest_with_progress(bot, df, argparse.Namespace(
                capital=args.capital,
                leverage=args.leverage,
                risk=args.risk,
                stop_loss=args.stop_loss,
                take_profit=args.take_profit,
                use_limit=not args.use_market,
                trailing_stop=not args.no_trailing
            ))

            print_warning("⚠️  Standard backtest has LOOK-AHEAD BIAS")
            print("   The model was trained on the same data it's being tested on.")
            print(f"   For realistic results, use: {Colors.CYAN}--walk-forward{Colors.ENDC}")

    except Exception as e:
        print_error(f"Backtest failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Display results
    display_results(results)

    # Save results
    if not args.no_save:
        print_info("Saving results...")

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create backtest archive directory
        backtest_dir = Path('./backtest')
        backtest_dir.mkdir(parents=True, exist_ok=True)

        # Generate timestamped filename
        now = datetime.now()

        # Determine backtest type
        if args.walk_forward:
            backtest_type = f"walkforward_{args.train_window}x{args.test_window}"
        else:
            backtest_type = "standard"

        # Get period description
        if args.days:
            period = f"{args.days}days"
        else:
            period = f"{args.start.replace('/', '')}_to_{args.end.replace('/', '')}"

        # Format: backtest_365days_meanrev_xgboost_14Nov_16-24_walkforward.json
        timestamp_str = now.strftime("%d%b_%H-%M")
        archive_filename = f"backtest_{period}_{args.strat}_{args.model}_{timestamp_str}_{backtest_type}.json"

        # Save model (overwrites with latest)
        model_path = output_dir / 'ml_mean_reversion_model.pkl'
        trades_path = output_dir / 'trade_history.json'

        bot.save_state(str(model_path), str(trades_path))

        # Save detailed backtest results
        # 1. Latest results for dashboard (overwrites)
        latest_results_path = output_dir / 'latest_backtest_results.json'
        save_detailed_results(results, bot, latest_results_path, args)

        # 2. Timestamped archive copy (never overwrites)
        archive_path = backtest_dir / archive_filename
        save_detailed_results(results, bot, archive_path, args)

        print_success(f"Model saved to: {model_path}")
        print_success(f"Trades saved to: {trades_path}")
        print_success(f"Latest results: {latest_results_path}")
        print_success(f"Archive saved: {archive_path}")

        # Show archive location
        print(f"\n{Colors.CYAN}📁 Backtest Archive:{Colors.ENDC}")
        print(f"   All your backtests are saved in: ./backtest/")
        print(f"   Latest: {archive_filename}")

    print(f"\n{Colors.GREEN}{Colors.BOLD}{'='*80}{Colors.ENDC}")
    print(f"{Colors.GREEN}{Colors.BOLD}Backtest Complete!{Colors.ENDC}")
    print(f"{Colors.GREEN}{Colors.BOLD}{'='*80}{Colors.ENDC}\n")

    # Auto-launch dashboard if requested
    if args.dashboard and not args.no_save:
        print_info("\nLaunching dashboard...")
        print(f"   Opening at http://localhost:8050")
        print(f"   Press {Colors.CYAN}Ctrl+C{Colors.ENDC} to stop\n")

        import subprocess
        import webbrowser
        import time

        # Launch dashboard in background
        dashboard_process = subprocess.Popen(
            [sys.executable, 'dashboard_app.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        # Wait a moment for server to start
        time.sleep(2)

        # Open browser
        webbrowser.open('http://localhost:8050')

        print_success("Dashboard launched! View your results in the browser.")
        print_info("Press Ctrl+C to stop the dashboard server...")

        try:
            dashboard_process.wait()
        except KeyboardInterrupt:
            print(f"\n{Colors.YELLOW}Stopping dashboard...{Colors.ENDC}")
            dashboard_process.terminate()
            dashboard_process.wait()
    elif not args.no_save:
        print_info("\nNext steps:")
        print("   1. Review results above")
        print(f"   2. Run dashboard: {Colors.CYAN}python dashboard_app.py{Colors.ENDC}")
        print("   3. If satisfied, deploy live: python live_trading_bot.py")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Backtest interrupted by user{Colors.ENDC}")
        sys.exit(0)
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
