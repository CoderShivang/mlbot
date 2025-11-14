"""
Live Trading Bot for Binance Perpetual Futures
==============================================

WARNING: This bot trades with real money. Use at your own risk.
Always test thoroughly with paper trading first.

Features:
- Real-time monitoring of BTC/USDT perpetual futures
- Automated trade execution based on ML predictions
- Risk management (stop loss, take profit, position sizing)
- Continuous learning from trade outcomes
- Emergency stop mechanisms
"""

import time
import json
from datetime import datetime, timedelta
from typing import Optional
import signal
import sys

from ml_mean_reversion_bot import MLMeanReversionBot, TradeSetup
from binance.client import Client
from binance.enums import *
import pandas as pd


class LiveTradingBot:
    """
    Live trading bot for Binance perpetual futures
    """
    
    def __init__(self, api_key: str, api_secret: str, 
                 model_path: str = 'ml_mean_reversion_model.pkl'):
        """
        Initialize live trading bot
        
        Args:
            api_key: Binance API key
            api_secret: Binance API secret
            model_path: Path to trained ML model
        """
        self.bot = MLMeanReversionBot(api_key, api_secret)
        
        # Load trained model
        try:
            self.bot.load_state(model_path)
            print("✅ Loaded trained ML model")
        except:
            print("⚠️  No trained model found. Train a model first using example_usage.py")
            raise
        
        # Trading parameters - CONFIGURED FOR $100 @ 20x LEVERAGE
        self.symbol = 'BTCUSDT'
        self.interval = '15m'  # 15-minute candles
        self.leverage = 20
        self.capital = 100  # Starting capital in USDT
        self.risk_per_trade = 0.15  # Risk 15% of capital per trade
        self.stop_loss_pct = 0.008  # 0.8% stop loss (16% of capital with 20x)
        self.take_profit_pct = 0.012  # 1.2% take profit (24% of capital with 20x)
        self.max_positions = 1  # Max concurrent positions
        self.use_limit_orders = True  # Use limit orders for better fees (0.02% vs 0.05%)
        
        # Fee structure
        self.maker_fee = 0.0002  # 0.02% for limit orders
        self.taker_fee = 0.0005  # 0.05% for market orders
        
        # State
        self.active_positions = []
        self.is_running = False
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.shutdown_handler)
        signal.signal(signal.SIGTERM, self.shutdown_handler)
        
        self.client = self.bot.client
        
        # Set leverage
        try:
            self.client.futures_change_leverage(symbol=self.symbol, leverage=self.leverage)
            print(f"✅ Set leverage to {self.leverage}x")
        except Exception as e:
            print(f"⚠️  Could not set leverage: {e}")
    
    def shutdown_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        print("\n\n🛑 Shutdown signal received. Closing positions...")
        self.is_running = False
        self.close_all_positions()
        self.bot.save_state()
        print("✅ Shutdown complete")
        sys.exit(0)
    
    def get_account_balance(self) -> float:
        """Get current USDT balance"""
        try:
            balance = self.client.futures_account_balance()
            usdt_balance = [b for b in balance if b['asset'] == 'USDT'][0]
            return float(usdt_balance['balance'])
        except Exception as e:
            print(f"Error fetching balance: {e}")
            return 0.0
    
    def get_current_price(self) -> float:
        """Get current BTC price"""
        ticker = self.client.futures_symbol_ticker(symbol=self.symbol)
        return float(ticker['price'])
    
    def get_recent_data(self, lookback_candles: int = 200) -> pd.DataFrame:
        """Fetch recent market data"""
        klines = self.client.futures_klines(
            symbol=self.symbol,
            interval=self.interval,
            limit=lookback_candles
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
        
        return df[['open', 'high', 'low', 'close', 'volume']]
    
    def calculate_position_size(self, entry_price: float) -> float:
        """
        Calculate position size based on risk management
        
        Returns quantity of BTC to trade
        """
        balance = self.get_account_balance()
        risk_amount = balance * self.risk_per_trade
        
        # Position size based on stop loss
        position_value = risk_amount / self.stop_loss_pct
        
        # Apply leverage
        position_value *= self.leverage
        
        # Convert to BTC quantity
        quantity = position_value / entry_price
        
        # Round to appropriate precision (Binance requires specific precision)
        quantity = round(quantity, 3)
        
        return quantity
    
    def place_order(self, direction: str, quantity: float, entry_price: float) -> Optional[dict]:
        """
        Place a LIMIT order for better fees (0.02% vs 0.05%)
        
        Args:
            direction: 'LONG' or 'SHORT'
            quantity: Amount of BTC
            entry_price: Desired entry price
            
        Returns:
            Order response or None if failed
        """
        try:
            side = SIDE_BUY if direction == 'LONG' else SIDE_SELL
            
            if self.use_limit_orders:
                # Place limit order slightly better than current price
                # For LONG: bid slightly below market
                # For SHORT: ask slightly above market
                current_price = self.get_current_price()
                
                if direction == 'LONG':
                    # Buy limit slightly below current price (0.05%)
                    limit_price = current_price * 0.9995
                else:
                    # Sell limit slightly above current price (0.05%)
                    limit_price = current_price * 1.0005
                
                # Round to appropriate precision
                limit_price = round(limit_price, 2)
                
                order = self.client.futures_create_order(
                    symbol=self.symbol,
                    side=side,
                    type=ORDER_TYPE_LIMIT,
                    timeInForce=TIME_IN_FORCE_GTC,  # Good till cancelled
                    quantity=quantity,
                    price=limit_price
                )
                
                print(f"\n✅ LIMIT order placed: {direction} {quantity} BTC at ${limit_price:.2f}")
                print(f"   Order ID: {order['orderId']}")
                print(f"   Fee: {self.maker_fee:.2%} (Maker)")
                
            else:
                # Market order (fallback)
                order = self.client.futures_create_order(
                    symbol=self.symbol,
                    side=side,
                    type=ORDER_TYPE_MARKET,
                    quantity=quantity
                )
                
                print(f"\n✅ MARKET order placed: {direction} {quantity} BTC at ~${entry_price:.2f}")
                print(f"   Order ID: {order['orderId']}")
                print(f"   Fee: {self.taker_fee:.2%} (Taker)")
            
            return order
            
        except Exception as e:
            print(f"\n❌ Error placing order: {e}")
            return None
    
    def wait_for_limit_fill(self, order_id: str, timeout_seconds: int = 180) -> bool:
        """
        Wait for limit order to fill
        
        Args:
            order_id: Order ID to monitor
            timeout_seconds: Max seconds to wait
            
        Returns:
            True if filled, False if timeout or cancelled
        """
        import time
        start_time = time.time()
        
        while time.time() - start_time < timeout_seconds:
            try:
                order_status = self.client.futures_get_order(
                    symbol=self.symbol,
                    orderId=order_id
                )
                
                status = order_status['status']
                
                if status == 'FILLED':
                    filled_price = float(order_status['avgPrice'])
                    print(f"   ✅ Order filled at ${filled_price:.2f}")
                    return True
                elif status in ['CANCELED', 'REJECTED', 'EXPIRED']:
                    print(f"   ❌ Order {status.lower()}")
                    return False
                
                # Still pending, wait a bit
                time.sleep(2)
                
            except Exception as e:
                print(f"   Error checking order status: {e}")
                return False
        
        # Timeout reached
        print(f"   ⏰ Order not filled within {timeout_seconds}s, cancelling...")
        try:
            self.client.futures_cancel_order(symbol=self.symbol, orderId=order_id)
        except:
            pass
        
        return False
    
    def set_stop_loss_take_profit(self, direction: str, quantity: float, 
                                  entry_price: float) -> bool:
        """
        Set stop loss and take profit orders
        
        Args:
            direction: 'LONG' or 'SHORT'
            quantity: Position size
            entry_price: Entry price
            
        Returns:
            True if successful
        """
        try:
            if direction == 'LONG':
                stop_price = entry_price * (1 - self.stop_loss_pct)
                take_profit_price = entry_price * (1 + self.take_profit_pct)
                sl_side = SIDE_SELL
                tp_side = SIDE_SELL
            else:
                stop_price = entry_price * (1 + self.stop_loss_pct)
                take_profit_price = entry_price * (1 - self.take_profit_pct)
                sl_side = SIDE_BUY
                tp_side = SIDE_BUY
            
            # Place stop loss
            sl_order = self.client.futures_create_order(
                symbol=self.symbol,
                side=sl_side,
                type=FUTURE_ORDER_TYPE_STOP_MARKET,
                stopPrice=round(stop_price, 2),
                quantity=quantity
            )
            
            # Place take profit
            tp_order = self.client.futures_create_order(
                symbol=self.symbol,
                side=tp_side,
                type=FUTURE_ORDER_TYPE_TAKE_PROFIT_MARKET,
                stopPrice=round(take_profit_price, 2),
                quantity=quantity
            )
            
            print(f"   Stop Loss: ${stop_price:.2f}")
            print(f"   Take Profit: ${take_profit_price:.2f}")
            
            return True
            
        except Exception as e:
            print(f"   ⚠️  Error setting SL/TP: {e}")
            return False
    
    def get_open_positions(self) -> list:
        """Get current open positions"""
        try:
            positions = self.client.futures_position_information(symbol=self.symbol)
            open_positions = [p for p in positions if float(p['positionAmt']) != 0]
            return open_positions
        except Exception as e:
            print(f"Error fetching positions: {e}")
            return []
    
    def close_position(self, position: dict) -> bool:
        """Close a specific position"""
        try:
            position_amt = float(position['positionAmt'])
            
            if position_amt == 0:
                return True
            
            side = SIDE_SELL if position_amt > 0 else SIDE_BUY
            quantity = abs(position_amt)
            
            order = self.client.futures_create_order(
                symbol=self.symbol,
                side=side,
                type=ORDER_TYPE_MARKET,
                quantity=quantity
            )
            
            print(f"✅ Closed position: {quantity} BTC")
            return True
            
        except Exception as e:
            print(f"❌ Error closing position: {e}")
            return False
    
    def close_all_positions(self):
        """Emergency: Close all open positions"""
        positions = self.get_open_positions()
        
        for position in positions:
            self.close_position(position)
    
    def monitor_positions(self):
        """Monitor and manage active positions"""
        positions = self.get_open_positions()
        
        for position in positions:
            position_amt = float(position['positionAmt'])
            entry_price = float(position['entryPrice'])
            current_price = self.get_current_price()
            
            if position_amt > 0:  # Long position
                pnl_pct = (current_price - entry_price) / entry_price
                direction = 'LONG'
            else:  # Short position
                pnl_pct = (entry_price - current_price) / entry_price
                direction = 'SHORT'
            
            unrealized_pnl = float(position['unrealizedProfit'])
            
            print(f"\n📊 Active {direction} position:")
            print(f"   Entry: ${entry_price:.2f}")
            print(f"   Current: ${current_price:.2f}")
            print(f"   PnL: {pnl_pct:.2%} (${unrealized_pnl:.2f})")
    
    def execute_trade(self, setup: TradeSetup) -> bool:
        """
        Execute a trade based on ML setup (using LIMIT orders)
        
        Args:
            setup: TradeSetup object from ML analysis
            
        Returns:
            True if trade executed successfully
        """
        # Check if we already have max positions
        if len(self.get_open_positions()) >= self.max_positions:
            print("⏸️  Max positions reached, skipping trade")
            return False
        
        # Calculate position size based on capital and leverage
        max_position_notional = self.capital * self.leverage
        risk_amount = self.capital * self.risk_per_trade
        position_notional = min(risk_amount / self.stop_loss_pct, max_position_notional)
        
        # BTC quantity
        quantity = position_notional / setup.entry_price
        quantity = round(quantity, 3)  # Round to 3 decimals
        
        if quantity == 0:
            print("❌ Calculated position size is 0, skipping trade")
            return False
        
        print(f"\n🎯 Executing {setup.direction} trade:")
        print(f"   Confidence: {setup.confidence_score:.1%}")
        print(f"   Position Size: ${position_notional:.2f} ({quantity:.3f} BTC)")
        print(f"   Entry: ${setup.entry_price:.2f}")
        
        # Place limit order
        order = self.place_order(setup.direction, quantity, setup.entry_price)
        
        if order is None:
            return False
        
        # Wait for limit order to fill
        if self.use_limit_orders:
            filled = self.wait_for_limit_fill(order['orderId'], timeout_seconds=180)
            if not filled:
                print("   ⚠️  Limit order not filled, skipping trade")
                return False
        
        # Set stop loss and take profit
        self.set_stop_loss_take_profit(setup.direction, quantity, setup.entry_price)
        
        # Track the setup
        self.active_positions.append({
            'setup': setup,
            'order': order,
            'quantity': quantity,
            'position_size': position_notional
        })
        
        return True
    
    def run(self, check_interval_seconds: int = 60):
        """
        Main trading loop
        
        Args:
            check_interval_seconds: How often to check for new setups
        """
        print("\n" + "="*80)
        print("🚀 LIVE TRADING BOT STARTED - REAL MONEY")
        print("="*80)
        print(f"Symbol: {self.symbol}")
        print(f"Interval: {self.interval}")
        print(f"Capital: ${self.capital}")
        print(f"Leverage: {self.leverage}x")
        print(f"Risk per trade: {self.risk_per_trade:.1%}")
        print(f"Stop Loss: {self.stop_loss_pct:.2%} ({self.stop_loss_pct * self.leverage:.1%} of capital)")
        print(f"Take Profit: {self.take_profit_pct:.2%} ({self.take_profit_pct * self.leverage:.1%} of capital)")
        print(f"Order Type: {'LIMIT (Maker: 0.02%)' if self.use_limit_orders else 'MARKET (Taker: 0.05%)'}")
        print("="*80)
        
        print("\n🔴 LIVE TRADING MODE - Using real money!")
        print("⚠️  With 20x leverage, a 5% BTC move = 100% capital impact")
        print("⚠️  Monitor closely and be prepared for volatility")
        
        response = input("\nAre you sure you want to continue? (type 'YES' to confirm): ")
        if response != 'YES':
            print("Aborted.")
            return
        
        print("\n🔄 Monitoring market...\n")
        
        self.is_running = True
        
        while self.is_running:
            try:
                # Get recent data
                df = self.get_recent_data(lookback_candles=200)
                
                # Add features
                df_features = self.bot.feature_engineer.calculate_features(df)
                
                # Check for signals
                setup = self.bot.analyze_setup(df_features)
                
                if setup is not None:
                    # Execute trade if conditions met
                    self.execute_trade(setup)
                
                # Monitor existing positions
                self.monitor_positions()
                
                # Update and save state periodically
                if len(self.bot.ml_model.trade_history) > 0:
                    self.bot.save_state()
                
                # Display status
                balance = self.get_account_balance()
                current_price = self.get_current_price()
                pnl_pct = (balance - self.capital) / self.capital
                
                print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                      f"Balance: ${balance:.2f} ({pnl_pct:+.1%}) | BTC: ${current_price:.2f}")
                
                # Wait before next check
                time.sleep(check_interval_seconds)
                
            except KeyboardInterrupt:
                print("\n\n🛑 Keyboard interrupt detected")
                break
                
            except Exception as e:
                print(f"\n❌ Error in main loop: {e}")
                print("Continuing...")
                time.sleep(check_interval_seconds)
        
        print("\n🛑 Trading bot stopped")


def main():
    """Main entry point"""
    
    # Configuration
    CONFIG = {
        'api_key': 'YOUR_BINANCE_API_KEY',
        'api_secret': 'YOUR_BINANCE_API_SECRET',
        'model_path': 'ml_mean_reversion_model.pkl',
        'check_interval': 60  # Check every 60 seconds
    }
    
    # Validate config
    if CONFIG['api_key'] == 'YOUR_BINANCE_API_KEY':
        print("❌ Error: Please set your Binance API credentials in the script")
        print("\nTo get API keys:")
        print("1. Log into Binance")
        print("2. Go to API Management")
        print("3. Create a new API key")
        print("4. Enable 'Futures' trading permission")
        print("5. Add your keys to this script")
        print("\n⚠️  IMPORTANT:")
        print("   • Do NOT enable 'Withdraw' permission")
        print("   • Consider IP whitelist for security")
        print("   • Start with minimum capital ($100)")
        return
    
    # Check if model exists
    import os
    if not os.path.exists(CONFIG['model_path']):
        print("❌ Error: Trained model not found")
        print(f"\nPlease train a model first using example_usage.py")
        print(f"Expected model path: {CONFIG['model_path']}")
        return
    
    # Initialize and run bot
    print("Initializing live trading bot...")
    print("Configuration:")
    print(f"  • Capital: $100")
    print(f"  • Leverage: 20x")
    print(f"  • Order Type: LIMIT (0.02% fee)")
    print(f"  • Stop Loss: 0.8% (16% of capital)")
    print(f"  • Take Profit: 1.2% (24% of capital)\n")
    
    try:
        bot = LiveTradingBot(
            api_key=CONFIG['api_key'],
            api_secret=CONFIG['api_secret'],
            model_path=CONFIG['model_path']
        )
        
        bot.run(check_interval_seconds=CONFIG['check_interval'])
        
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
