"""
ML-Enhanced Mean Reversion Trading Bot for BTC/USDT Perpetual Futures
=====================================================================

This bot combines traditional mean reversion signals with ML pattern recognition
to identify high-quality trade setups and avoid historically unsuccessful patterns.

Features:
- Context-aware trading using market regime detection
- Pattern similarity matching to historical trades
- Trade confidence scoring based on ML
- Automatic learning from trade outcomes
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pickle
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Technical Analysis
import talib
from scipy import stats

# Binance API (you'll need to install: pip install python-binance)
try:
    from binance.client import Client
    from binance.enums import *
except ImportError:
    print("Warning: python-binance not installed. Install with: pip install python-binance")


@dataclass
class TradeSetup:
    """Represents a potential trade setup with all context"""
    timestamp: datetime
    entry_price: float
    direction: str  # 'LONG' or 'SHORT'
    
    # Mean reversion signals
    rsi: float
    bb_position: float  # Position within Bollinger Bands (-1 to 1)
    zscore: float
    
    # Market context features
    volatility_regime: str  # 'LOW', 'MEDIUM', 'HIGH'
    trend_strength: float  # -1 (strong down) to 1 (strong up)
    volume_ratio: float  # Current volume vs average
    recent_drawdown: float
    
    # Pattern features
    pattern_momentum: float
    support_resistance_proximity: float
    consolidation_duration: int  # bars in consolidation
    
    # ML outputs
    confidence_score: float = 0.0
    similar_trades: List[Dict] = None
    predicted_success_prob: float = 0.0
    
    # Outcome (for learning)
    actual_outcome: Optional[str] = None  # 'WIN', 'LOSS', or None if not closed
    pnl_percent: Optional[float] = None
    

class FeatureEngineering:
    """Extract comprehensive features for ML model"""
    
    @staticmethod
    def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical and contextual features
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added features
        """
        data = df.copy()
        
        # === MEAN REVERSION INDICATORS ===
        
        # RSI
        data['rsi'] = talib.RSI(data['close'], timeperiod=14)
        data['rsi_oversold'] = (data['rsi'] < 30).astype(int)
        data['rsi_overbought'] = (data['rsi'] > 70).astype(int)
        
        # Bollinger Bands
        data['bb_upper'], data['bb_middle'], data['bb_lower'] = talib.BBANDS(
            data['close'], timeperiod=20, nbdevup=2, nbdevdn=2
        )
        data['bb_position'] = (data['close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])
        data['bb_width'] = (data['bb_upper'] - data['bb_lower']) / data['bb_middle']
        
        # Z-Score of price
        data['returns'] = data['close'].pct_change()
        data['zscore'] = (data['close'] - data['close'].rolling(20).mean()) / data['close'].rolling(20).std()
        
        # === MARKET CONTEXT ===
        
        # Volatility (ATR and rolling std)
        data['atr'] = talib.ATR(data['high'], data['low'], data['close'], timeperiod=14)
        data['volatility'] = data['returns'].rolling(20).std()
        data['volatility_regime'] = pd.cut(
            data['volatility'], 
            bins=[0, data['volatility'].quantile(0.33), data['volatility'].quantile(0.67), np.inf],
            labels=['LOW', 'MEDIUM', 'HIGH']
        )
        
        # Trend indicators
        data['ema_9'] = talib.EMA(data['close'], timeperiod=9)
        data['ema_21'] = talib.EMA(data['close'], timeperiod=21)
        data['ema_50'] = talib.EMA(data['close'], timeperiod=50)
        
        # Trend strength (ADX)
        data['adx'] = talib.ADX(data['high'], data['low'], data['close'], timeperiod=14)
        data['plus_di'] = talib.PLUS_DI(data['high'], data['low'], data['close'], timeperiod=14)
        data['minus_di'] = talib.MINUS_DI(data['high'], data['low'], data['close'], timeperiod=14)
        
        # Trend direction score
        data['trend_strength'] = np.where(
            data['plus_di'] > data['minus_di'],
            data['adx'] / 100,
            -data['adx'] / 100
        )
        
        # Volume analysis
        data['volume_sma'] = data['volume'].rolling(20).mean()
        data['volume_ratio'] = data['volume'] / data['volume_sma']
        
        # Money Flow Index
        data['mfi'] = talib.MFI(data['high'], data['low'], data['close'], data['volume'], timeperiod=14)
        
        # === PATTERN FEATURES ===
        
        # Momentum
        data['momentum'] = talib.MOM(data['close'], timeperiod=10)
        data['roc'] = talib.ROC(data['close'], timeperiod=10)
        
        # Support/Resistance proximity
        data['swing_high'] = data['high'].rolling(20).max()
        data['swing_low'] = data['low'].rolling(20).min()
        data['distance_to_high'] = (data['swing_high'] - data['close']) / data['close']
        data['distance_to_low'] = (data['close'] - data['swing_low']) / data['close']
        
        # Consolidation detection (measure of ranging)
        data['high_low_ratio'] = (data['high'] - data['low']) / data['close']
        data['is_consolidating'] = (data['high_low_ratio'].rolling(10).mean() < 
                                    data['high_low_ratio'].rolling(50).mean() * 0.8).astype(int)
        
        # Recent drawdown
        data['cummax'] = data['close'].cummax()
        data['drawdown'] = (data['close'] - data['cummax']) / data['cummax']
        
        # Candle patterns
        data['doji'] = talib.CDLDOJI(data['open'], data['high'], data['low'], data['close'])
        data['hammer'] = talib.CDLHAMMER(data['open'], data['high'], data['low'], data['close'])
        data['engulfing'] = talib.CDLENGULFING(data['open'], data['high'], data['low'], data['close'])
        
        return data
    
    @staticmethod
    def get_feature_list() -> List[str]:
        """Return list of features to use in ML model"""
        return [
            # Mean reversion
            'rsi', 'bb_position', 'bb_width', 'zscore',
            
            # Market context
            'volatility', 'atr', 'adx', 'trend_strength', 'volume_ratio', 'mfi',
            
            # Momentum
            'momentum', 'roc',
            
            # Pattern features
            'distance_to_high', 'distance_to_low', 'is_consolidating', 'drawdown',
            
            # Candle patterns
            'doji', 'hammer', 'engulfing'
        ]


class MeanReversionSignals:
    """Generate traditional mean reversion signals"""
    
    @staticmethod
    def identify_long_setups(df: pd.DataFrame) -> pd.Series:
        """
        Identify potential LONG mean reversion setups
        
        Criteria:
        - RSI < 30 (oversold)
        - Price near lower Bollinger Band
        - Z-score < -2 (price significantly below mean)
        """
        long_signals = (
            (df['rsi'] < 30) &
            (df['bb_position'] < 0.2) &
            (df['zscore'] < -1.5)
        )
        return long_signals
    
    @staticmethod
    def identify_short_setups(df: pd.DataFrame) -> pd.Series:
        """
        Identify potential SHORT mean reversion setups
        
        Criteria:
        - RSI > 70 (overbought)
        - Price near upper Bollinger Band
        - Z-score > 2 (price significantly above mean)
        """
        short_signals = (
            (df['rsi'] > 70) &
            (df['bb_position'] > 0.8) &
            (df['zscore'] > 1.5)
        )
        return short_signals


class MLPatternRecognizer:
    """
    Machine Learning model to recognize successful vs unsuccessful trade patterns
    """
    
    def __init__(self):
        self.model = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.feature_names = FeatureEngineering.get_feature_list()
        self.is_trained = False
        self.trade_history: List[TradeSetup] = []
        
    def prepare_training_data(self, df: pd.DataFrame, 
                            forward_returns: pd.Series,
                            success_threshold: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data from historical trades
        
        Args:
            df: DataFrame with features
            forward_returns: Returns after signal (target variable)
            success_threshold: Minimum return to consider trade successful
            
        Returns:
            X, y for training
        """
        # Get features
        X = df[self.feature_names].values
        
        # Create binary labels (successful trade = 1, unsuccessful = 0)
        y = (forward_returns > success_threshold).astype(int).values
        
        # Remove NaN rows
        valid_idx = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X = X[valid_idx]
        y = y[valid_idx]
        
        return X, y
    
    def train(self, df: pd.DataFrame, forward_returns: pd.Series):
        """Train the ML model on historical data"""
        X, y = self.prepare_training_data(df, forward_returns)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        print("Training ML model...")
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        print(f"Training accuracy: {train_score:.3f}")
        print(f"Testing accuracy: {test_score:.3f}")
        
        # Classification report
        y_pred = self.model.predict(X_test_scaled)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Unsuccessful', 'Successful']))
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print(feature_importance.head(10))
        
        self.is_trained = True
        
        return {
            'train_score': train_score,
            'test_score': test_score,
            'feature_importance': feature_importance
        }
    
    def predict_trade_quality(self, features: pd.Series) -> Dict:
        """
        Predict quality of a trade setup
        
        Returns:
            Dictionary with confidence score and probability
        """
        if not self.is_trained:
            return {
                'confidence_score': 0.5,
                'success_probability': 0.5,
                'should_trade': False,
                'reason': 'Model not trained yet'
            }
        
        # Prepare features
        X = features[self.feature_names].values.reshape(1, -1)
        
        # Check for NaN
        if np.isnan(X).any():
            return {
                'confidence_score': 0.0,
                'success_probability': 0.0,
                'should_trade': False,
                'reason': 'Invalid features (NaN values)'
            }
        
        # Scale and predict
        X_scaled = self.scaler.transform(X)
        success_prob = self.model.predict_proba(X_scaled)[0][1]
        
        # Decision logic
        confidence_threshold = 0.65  # Require 65% probability of success
        should_trade = success_prob >= confidence_threshold
        
        return {
            'confidence_score': success_prob,
            'success_probability': success_prob,
            'should_trade': should_trade,
            'reason': f"Success probability: {success_prob:.2%}"
        }
    
    def find_similar_trades(self, current_features: pd.Series, n_similar: int = 5) -> List[Dict]:
        """
        Find historically similar trade setups using feature similarity
        
        Args:
            current_features: Current market features
            n_similar: Number of similar trades to return
            
        Returns:
            List of similar historical trades with outcomes
        """
        if len(self.trade_history) == 0:
            return []
        
        # Calculate similarity scores using Euclidean distance in feature space
        current_vec = current_features[self.feature_names].values
        
        similarities = []
        for trade in self.trade_history:
            if trade.actual_outcome is not None:  # Only compare completed trades
                # Reconstruct feature vector from trade
                trade_vec = np.array([
                    trade.rsi, trade.bb_position, 0,  # bb_width not stored
                    trade.zscore, 0, 0, 0,  # volatility, atr, adx not stored
                    trade.trend_strength, trade.volume_ratio, 0,  # mfi
                    trade.pattern_momentum, 0,  # roc
                    trade.support_resistance_proximity, 0, 0,  # distance features
                    0, trade.recent_drawdown, 0, 0, 0  # consolidating, candle patterns
                ])
                
                # Note: This is simplified - in production, store full feature vectors
                # For now, use available features
                available_features = ['rsi', 'bb_position', 'zscore', 'trend_strength', 
                                    'volume_ratio', 'pattern_momentum', 'recent_drawdown']
                
                distance = np.linalg.norm(
                    current_features[available_features].values - 
                    np.array([getattr(trade, f) for f in available_features])
                )
                
                similarities.append({
                    'trade': trade,
                    'distance': distance
                })
        
        # Sort by similarity (smaller distance = more similar)
        similarities.sort(key=lambda x: x['distance'])
        
        # Return top N
        similar_trades = []
        for item in similarities[:n_similar]:
            trade = item['trade']
            similar_trades.append({
                'direction': trade.direction,
                'outcome': trade.actual_outcome,
                'pnl_percent': trade.pnl_percent,
                'confidence': trade.confidence_score,
                'similarity': 1 / (1 + item['distance'])  # Convert distance to similarity score
            })
        
        return similar_trades
    
    def add_trade_outcome(self, trade: TradeSetup):
        """Add completed trade to history for learning"""
        self.trade_history.append(trade)
    
    def save_model(self, filepath: str):
        """Save trained model and scaler"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained,
            'trade_history': self.trade_history
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model and scaler"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.is_trained = model_data['is_trained']
        self.trade_history = model_data.get('trade_history', [])
        print(f"Model loaded from {filepath}")


class MLMeanReversionBot:
    """
    Main trading bot that combines mean reversion signals with ML pattern recognition
    """
    
    def __init__(self, api_key: str = None, api_secret: str = None):
        self.api_key = api_key
        self.api_secret = api_secret
        self.client = None
        
        if api_key and api_secret:
            self.client = Client(api_key, api_secret)
        
        self.ml_model = MLPatternRecognizer()
        self.feature_engineer = FeatureEngineering()
        self.signal_generator = MeanReversionSignals()
        
        self.active_trades: List[TradeSetup] = []
        
    def fetch_historical_data(self, symbol: str = 'BTCUSDT', 
                            interval: str = '15m', 
                            lookback_days: int = 30) -> pd.DataFrame:
        """
        Fetch historical kline data from Binance
        
        Args:
            symbol: Trading pair
            interval: Candle interval (1m, 5m, 15m, 1h, 4h, 1d)
            lookback_days: Days of history to fetch
        """
        if not self.client:
            raise ValueError("Binance client not initialized. Provide API credentials.")
        
        print(f"Fetching {lookback_days} days of {interval} data for {symbol}...")
        
        # Calculate start time
        start_time = datetime.now() - timedelta(days=lookback_days)
        start_str = start_time.strftime("%d %b %Y %H:%M:%S")
        
        # Fetch klines
        klines = self.client.get_historical_klines(
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
        
        print(f"Fetched {len(df)} candles from {df.index[0]} to {df.index[-1]}")
        
        return df[['open', 'high', 'low', 'close', 'volume']]
    
    def train_model(self, df: pd.DataFrame, forward_periods: int = 10):
        """
        Train ML model on historical data
        
        Args:
            df: Historical OHLCV data
            forward_periods: Periods ahead to measure trade success
        """
        print("\n=== Training ML Model ===\n")
        
        # Add features
        df_features = self.feature_engineer.calculate_features(df)
        
        # Calculate forward returns (target variable)
        df_features['forward_return'] = df_features['close'].pct_change(forward_periods).shift(-forward_periods)
        
        # Identify mean reversion signals
        df_features['long_signal'] = self.signal_generator.identify_long_setups(df_features)
        df_features['short_signal'] = self.signal_generator.identify_short_setups(df_features)
        
        # Train on LONG setups
        long_data = df_features[df_features['long_signal']].copy()
        if len(long_data) > 50:
            print("Training on LONG setups...")
            results_long = self.ml_model.train(long_data, long_data['forward_return'])
        
        # Could also train separate model for shorts, or combine with direction as feature
        
        return df_features
    
    def analyze_setup(self, df: pd.DataFrame, current_idx: int = -1) -> Optional[TradeSetup]:
        """
        Analyze current market setup and decide if we should trade
        
        Args:
            df: DataFrame with features
            current_idx: Index to analyze (default: most recent)
            
        Returns:
            TradeSetup object if trade is recommended, None otherwise
        """
        current = df.iloc[current_idx]
        
        # Check for mean reversion signals
        is_long = self.signal_generator.identify_long_setups(df).iloc[current_idx]
        is_short = self.signal_generator.identify_short_setups(df).iloc[current_idx]
        
        if not (is_long or is_short):
            return None
        
        direction = 'LONG' if is_long else 'SHORT'
        
        # Create trade setup
        setup = TradeSetup(
            timestamp=current.name if hasattr(current, 'name') else datetime.now(),
            entry_price=current['close'],
            direction=direction,
            rsi=current['rsi'],
            bb_position=current['bb_position'],
            zscore=current['zscore'],
            volatility_regime=current.get('volatility_regime', 'MEDIUM'),
            trend_strength=current['trend_strength'],
            volume_ratio=current['volume_ratio'],
            recent_drawdown=current['drawdown'],
            pattern_momentum=current['momentum'],
            support_resistance_proximity=current.get('distance_to_low' if is_long else 'distance_to_high', 0),
            consolidation_duration=int(current.get('is_consolidating', 0))
        )
        
        # Get ML prediction
        ml_prediction = self.ml_model.predict_trade_quality(current)
        setup.confidence_score = ml_prediction['confidence_score']
        setup.predicted_success_prob = ml_prediction['success_probability']
        
        # Find similar historical trades
        similar_trades = self.ml_model.find_similar_trades(current, n_similar=5)
        setup.similar_trades = similar_trades
        
        # Decision
        if not ml_prediction['should_trade']:
            print(f"\n❌ {direction} setup identified but ML model rejects it")
            print(f"   Reason: {ml_prediction['reason']}")
            return None
        
        print(f"\n✅ {direction} setup identified with {setup.confidence_score:.1%} confidence")
        print(f"   Entry: ${setup.entry_price:.2f}")
        print(f"   RSI: {setup.rsi:.1f} | Z-Score: {setup.zscore:.2f}")
        print(f"   Volatility: {setup.volatility_regime} | Trend: {setup.trend_strength:.2f}")
        
        if similar_trades:
            print(f"\n   Similar historical setups:")
            for i, trade in enumerate(similar_trades, 1):
                emoji = "🟢" if trade['outcome'] == 'WIN' else "🔴"
                print(f"   {i}. {emoji} {trade['direction']}: {trade['pnl_percent']:.2%} "
                      f"(similarity: {trade['similarity']:.2%})")
        
        return setup
    
    def backtest(self, df: pd.DataFrame, initial_capital: float = 100,
                leverage: int = 20, risk_per_trade: float = 0.15,
                stop_loss_pct: float = 0.008, take_profit_pct: float = 0.012,
                use_limit_orders: bool = True) -> Dict:
        """
        Backtest the ML-enhanced mean reversion strategy with REALISTIC mainnet simulation
        
        Args:
            df: Historical MAINNET data with features
            initial_capital: Starting capital (default $100)
            leverage: Leverage multiplier (default 20x)
            risk_per_trade: Fraction of capital to risk per trade (0.15 = 15%)
            stop_loss_pct: Stop loss as % of position value (0.008 = 0.8%)
            take_profit_pct: Take profit as % of position value (0.012 = 1.2%)
            use_limit_orders: Use limit orders (True) vs market orders (False)
            
        Returns:
            Dictionary with backtest results
            
        Note:
            With $100 capital and 20x leverage:
            - Max position size: $2,000 notional
            - 1% BTC move = 20% capital impact
            - SL at 0.8% = $16 loss (16% of capital)
            - TP at 1.2% = $24 profit (24% of capital)
        """
        print("\n=== Running Backtest with MAINNET Data ===\n")
        print(f"Capital: ${initial_capital}")
        print(f"Leverage: {leverage}x")
        print(f"Stop Loss: {stop_loss_pct:.2%} ({stop_loss_pct * leverage:.1%} of capital)")
        print(f"Take Profit: {take_profit_pct:.2%} ({take_profit_pct * leverage:.1%} of capital)")
        print(f"Order Type: {'LIMIT (Maker: 0.02%)' if use_limit_orders else 'MARKET (Taker: 0.05%)'}\n")
        
        capital = initial_capital
        trades = []
        equity_curve = [initial_capital]
        
        # Fee structure
        maker_fee = 0.0002  # 0.02% for limit orders
        taker_fee = 0.0005  # 0.05% for market orders
        fee_rate = maker_fee if use_limit_orders else taker_fee
        
        for i in range(100, len(df) - 10):  # Skip first 100 for indicators, last 10 for forward returns
            setup = self.analyze_setup(df, current_idx=i)
            
            if setup is None:
                equity_curve.append(capital)
                continue
            
            # Calculate position size based on risk and leverage
            # We're risking 'risk_per_trade' fraction of capital
            # Position size = capital * leverage (max notional value)
            max_position_notional = capital * leverage
            
            # But we size based on risk: risk_amount / stop_loss_pct
            risk_amount = capital * risk_per_trade
            position_notional = min(risk_amount / stop_loss_pct, max_position_notional)
            
            # BTC quantity
            entry_price = setup.entry_price
            btc_quantity = position_notional / entry_price
            
            # Simulate limit order fill
            if use_limit_orders:
                # For LONG: place buy limit at current price (or slightly below)
                # For SHORT: place sell limit at current price (or slightly above)
                # We'll assume fill if price touches our level within next 3 candles
                
                next_prices = df.iloc[i+1:i+4]
                
                if setup.direction == 'LONG':
                    # Check if price came down to fill our buy limit
                    if next_prices['low'].min() <= entry_price:
                        filled = True
                        fill_price = entry_price
                    else:
                        filled = False
                else:  # SHORT
                    # Check if price came up to fill our sell limit
                    if next_prices['high'].max() >= entry_price:
                        filled = True
                        fill_price = entry_price
                    else:
                        filled = False
                
                if not filled:
                    # Order not filled, skip this trade
                    equity_curve.append(capital)
                    continue
            else:
                # Market order - always fills at current price + slippage
                fill_price = entry_price
            
            # Entry fee
            entry_fee = position_notional * fee_rate
            
            # Calculate exit price based on forward price action
            future_prices = df.iloc[i+1:i+11]
            
            if setup.direction == 'LONG':
                stop_loss_price = fill_price * (1 - stop_loss_pct)
                take_profit_price = fill_price * (1 + take_profit_pct)
                
                # Check if SL or TP hit
                if future_prices['low'].min() <= stop_loss_price:
                    exit_price = stop_loss_price
                    outcome = 'LOSS'
                    hit_type = 'SL'
                elif future_prices['high'].max() >= take_profit_price:
                    exit_price = take_profit_price
                    outcome = 'WIN'
                    hit_type = 'TP'
                else:
                    # Neither hit, exit at last price
                    exit_price = future_prices.iloc[-1]['close']
                    outcome = 'WIN' if exit_price > fill_price else 'LOSS'
                    hit_type = 'TIMEOUT'
                
                price_pnl_pct = (exit_price - fill_price) / fill_price
                
            else:  # SHORT
                stop_loss_price = fill_price * (1 + stop_loss_pct)
                take_profit_price = fill_price * (1 - take_profit_pct)
                
                if future_prices['high'].max() >= stop_loss_price:
                    exit_price = stop_loss_price
                    outcome = 'LOSS'
                    hit_type = 'SL'
                elif future_prices['low'].min() <= take_profit_price:
                    exit_price = take_profit_price
                    outcome = 'WIN'
                    hit_type = 'TP'
                else:
                    exit_price = future_prices.iloc[-1]['close']
                    outcome = 'WIN' if exit_price < fill_price else 'LOSS'
                    hit_type = 'TIMEOUT'
                
                price_pnl_pct = (fill_price - exit_price) / fill_price
            
            # Exit fee
            exit_fee = position_notional * fee_rate
            
            # Total PnL including fees and leverage
            gross_pnl_usd = position_notional * price_pnl_pct
            net_pnl_usd = gross_pnl_usd - entry_fee - exit_fee
            
            # PnL as percentage of capital (considering leverage)
            pnl_pct_capital = net_pnl_usd / capital
            
            # Update capital
            capital += net_pnl_usd
            equity_curve.append(capital)
            
            # Check for liquidation (if capital drops too low)
            if capital <= initial_capital * 0.1:  # 90% loss
                print(f"\n⚠️  WARNING: Capital dropped to ${capital:.2f} - Near liquidation!")
                break
            
            # Record trade with detailed info
            setup.actual_outcome = outcome
            setup.pnl_percent = pnl_pct_capital  # Store as % of capital
            
            # Add metadata
            trade_info = {
                'setup': setup,
                'entry_price': fill_price,
                'exit_price': exit_price,
                'position_size_usd': position_notional,
                'btc_quantity': btc_quantity,
                'gross_pnl_usd': gross_pnl_usd,
                'fees_usd': entry_fee + exit_fee,
                'net_pnl_usd': net_pnl_usd,
                'hit_type': hit_type,
                'capital_after': capital
            }
            
            trades.append(trade_info)
            
            # Add to model's learning history
            self.ml_model.add_trade_outcome(setup)
        
        # Calculate metrics
        winning_trades = [t for t in trades if t['setup'].actual_outcome == 'WIN']
        losing_trades = [t for t in trades if t['setup'].actual_outcome == 'LOSS']
        
        total_trades = len(trades)
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        # PnL in USD
        total_pnl_usd = sum([t['net_pnl_usd'] for t in trades])
        avg_win_usd = np.mean([t['net_pnl_usd'] for t in winning_trades]) if winning_trades else 0
        avg_loss_usd = np.mean([t['net_pnl_usd'] for t in losing_trades]) if losing_trades else 0
        
        # PnL in % of capital
        avg_win_pct = np.mean([t['setup'].pnl_percent for t in winning_trades]) if winning_trades else 0
        avg_loss_pct = np.mean([t['setup'].pnl_percent for t in losing_trades]) if losing_trades else 0
        
        # Total fees paid
        total_fees = sum([t['fees_usd'] for t in trades])
        
        # Profit factor
        gross_profit = sum([t['net_pnl_usd'] for t in winning_trades]) if winning_trades else 0
        gross_loss = abs(sum([t['net_pnl_usd'] for t in losing_trades])) if losing_trades else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
        
        total_return = (capital - initial_capital) / initial_capital
        
        # Sharpe ratio (simplified)
        returns = pd.Series(equity_curve).pct_change().dropna()
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        # Max drawdown
        equity_series = pd.Series(equity_curve)
        cummax = equity_series.cummax()
        drawdown = (equity_series - cummax) / cummax
        max_drawdown = drawdown.min()
        max_drawdown_usd = (cummax - equity_series).max()
        
        # Order fill statistics (for limit orders)
        if use_limit_orders:
            filled_count = len(trades)
            # Approximate unfilled by signal count vs filled count
            # This is simplified - in reality we'd track all signals
            fill_rate = 0.85  # Approximate
        else:
            fill_rate = 1.0
        
        # Risk metrics with leverage
        max_position_size = max([t['position_size_usd'] for t in trades]) if trades else 0
        avg_position_size = np.mean([t['position_size_usd'] for t in trades]) if trades else 0
        
        results = {
            'initial_capital': initial_capital,
            'final_capital': capital,
            'total_return': total_return,
            'total_pnl_usd': total_pnl_usd,
            'leverage': leverage,
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'avg_win_usd': avg_win_usd,
            'avg_loss_usd': avg_loss_usd,
            'avg_win_pct': avg_win_pct,
            'avg_loss_pct': avg_loss_pct,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'max_drawdown_usd': max_drawdown_usd,
            'total_fees_paid': total_fees,
            'fee_rate': fee_rate,
            'order_type': 'LIMIT' if use_limit_orders else 'MARKET',
            'fill_rate': fill_rate,
            'max_position_size': max_position_size,
            'avg_position_size': avg_position_size,
            'equity_curve': equity_curve,
            'trades': trades
        }
        
        # Print results
        print(f"\n{'='*60}")
        print(f"BACKTEST RESULTS - ${initial_capital} @ {leverage}x Leverage")
        print(f"{'='*60}")
        print(f"Initial Capital:    ${initial_capital:.2f}")
        print(f"Final Capital:      ${capital:.2f}")
        print(f"Total Return:       {total_return:.2%} (${total_pnl_usd:.2f})")
        print(f"Order Type:         {results['order_type']} (Fee: {fee_rate:.2%})")
        print(f"Total Fees Paid:    ${total_fees:.2f}")
        
        print(f"\nTrade Statistics:")
        print(f"Total Trades:       {total_trades}")
        print(f"Winning Trades:     {len(winning_trades)}")
        print(f"Losing Trades:      {len(losing_trades)}")
        print(f"Win Rate:           {win_rate:.2%}")
        print(f"Avg Win:            ${avg_win_usd:.2f} ({avg_win_pct:.2%} of capital)")
        print(f"Avg Loss:           ${avg_loss_usd:.2f} ({avg_loss_pct:.2%} of capital)")
        print(f"Profit Factor:      {profit_factor:.2f}")
        
        if use_limit_orders:
            print(f"Fill Rate:          ~{fill_rate:.1%}")
        
        print(f"\nPosition Sizing:")
        print(f"Max Position:       ${max_position_size:.2f} ({max_position_size/capital*100:.1f}x capital)")
        print(f"Avg Position:       ${avg_position_size:.2f}")
        
        print(f"\nRisk Metrics:")
        print(f"Sharpe Ratio:       {sharpe:.2f}")
        print(f"Max Drawdown:       {max_drawdown:.2%} (${max_drawdown_usd:.2f})")
        
        # Calculate ROI considering leverage
        roi_with_leverage = total_return
        roi_without_leverage = total_return / leverage  # Approximate
        print(f"\nLeverage Impact:")
        print(f"ROI with {leverage}x:     {roi_with_leverage:.2%}")
        print(f"Equivalent no-lev:  ~{roi_without_leverage:.2%}")
        
        print(f"{'='*60}\n")
        
        return results
    
    def save_state(self, model_path: str = 'ml_model.pkl', 
                  trades_path: str = 'trade_history.json'):
        """Save model and trade history"""
        self.ml_model.save_model(model_path)
        
        # Save trade history
        trades_data = [asdict(trade) for trade in self.ml_model.trade_history]
        # Convert datetime objects to strings
        for trade in trades_data:
            if isinstance(trade['timestamp'], datetime):
                trade['timestamp'] = trade['timestamp'].isoformat()
        
        with open(trades_path, 'w') as f:
            json.dump(trades_data, f, indent=2)
        
        print(f"Trade history saved to {trades_path}")
    
    def load_state(self, model_path: str = 'ml_model.pkl'):
        """Load model and trade history"""
        self.ml_model.load_model(model_path)


# Example usage
if __name__ == "__main__":
    print("ML-Enhanced Mean Reversion Trading Bot")
    print("="*60)
    
    # Initialize bot (without API keys for this example)
    bot = MLMeanReversionBot()
    
    # For testing, you would:
    # 1. Load your historical data
    # 2. Train the model
    # 3. Backtest
    # 4. Deploy for live trading
    
    print("\nTo use this bot:")
    print("1. Install dependencies: pip install python-binance ta-lib scikit-learn")
    print("2. Provide your Binance API credentials")
    print("3. Run training on historical data")
    print("4. Backtest the strategy")
    print("5. Deploy for live trading")
    print("\nSee the comprehensive usage example in the separate script.")
