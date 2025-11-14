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

    print_info("Calculating technical features (24 indicators)...")
    print("   └─ Mean reversion: RSI, Bollinger Bands, Z-scores")
    print("   └─ Market context: Volatility, Trend, Volume")
    print("   └─ Pattern features: Momentum, S/R, Consolidation")
    print("   └─ Feature interactions: RSI×BB, Trend×Vol, Volume×Mom")

    # Calculate features
    from ml_mean_reversion_bot import FeatureEngineering
    df_features = FeatureEngineering.calculate_features(df)
    print_success(f"Features calculated for {len(df_features):,} candles")

    # Generate training labels
    print_info("Generating training labels...")
    print(f"   └─ Checking forward returns over {forward_periods} periods")

    # Calculate forward returns
    df_features['forward_return'] = df_features['close'].shift(-forward_periods) / df_features['close'] - 1

    # Identify signals
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

    # Train model
    print_info("Training Gradient Boosting Classifier...")
    print(f"   └─ Model: GradientBoostingClassifier")
    print(f"   └─ Estimators: 200")
    print(f"   └─ Learning Rate: 0.05")
    print(f"   └─ Max Depth: 5")

    # Get feature list
    feature_cols = FeatureEngineering.get_feature_list()
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

    # Train with progress
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import classification_report, accuracy_score

    print("\n   Training in progress...")

    # We'll show progress by monitoring estimators (boosting rounds)
    model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        random_state=42,
        verbose=0  # We'll handle our own progress
    )

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

    # Feature importance (top 5)
    print_info("Top 5 Most Important Features:")
    importances = model.feature_importances_
    feature_importance = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)

    for i, (feature, importance) in enumerate(feature_importance[:5], 1):
        bar_length = int(importance * 30)
        bar = '█' * bar_length + '░' * (30 - bar_length)
        print(f"   {i}. {feature:25s} {bar} {importance:.3f}")

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

    # ML options
    parser.add_argument('--no-train', action='store_true',
                       help='Skip ML training (use existing model)')
    parser.add_argument('--forward-periods', type=int, default=10,
                       help='Forward periods for ML labels (default: 10)')

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
    print_header("ML MEAN REVERSION BOT - BACKTEST")

    print_info(f"Symbol: {args.symbol}")
    print_info(f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} ({(end_date - start_date).days} days)")
    print_info(f"Interval: {args.interval}")

    # Initialize bot
    print_info("Initializing bot...")
    bot = MLMeanReversionBot()
    print_success("Bot initialized")

    # Fetch data
    try:
        df = fetch_binance_data(args.symbol, start_date, end_date, args.interval)
    except Exception as e:
        print_error(f"Failed to fetch data: {e}")
        sys.exit(1)

    # Train ML model
    if not args.no_train:
        try:
            df = train_model_with_progress(bot, df, args.forward_periods)
        except Exception as e:
            print_error(f"Training failed: {e}")
            sys.exit(1)
    else:
        print_warning("Skipping ML training - using existing model (if available)")

        # Still need to calculate features
        print_info("Calculating features for backtest...")
        from ml_mean_reversion_bot import FeatureEngineering
        df = FeatureEngineering.calculate_features(df)
        print_success("Features calculated")

    # Run backtest
    try:
        results = run_backtest_with_progress(bot, df, argparse.Namespace(
            capital=args.capital,
            leverage=args.leverage,
            risk=args.risk,
            stop_loss=args.stop_loss,
            take_profit=args.take_profit,
            use_limit=not args.use_market,
            trailing_stop=not args.no_trailing
        ))
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

        # Save model
        model_path = output_dir / 'ml_mean_reversion_model.pkl'
        trades_path = output_dir / 'trade_history.json'

        bot.save_state(str(model_path), str(trades_path))

        # Save detailed backtest results for dashboard
        results_path = output_dir / 'latest_backtest_results.json'
        save_detailed_results(results, bot, results_path, args)

        print_success(f"Model saved to: {model_path}")
        print_success(f"Trades saved to: {trades_path}")
        print_success(f"Detailed results saved to: {results_path}")

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
