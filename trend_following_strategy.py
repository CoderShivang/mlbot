"""
Extreme Trend Following Strategy - ML-Trained
=============================================

This strategy identifies and trades EXTREME trends - strong directional moves
where price has clearly "ripped up or down".

Key differences from mean reversion:
- Trades WITH the trend, not against it
- Looks for breakouts and momentum continuation
- Uses different entry/exit logic
- Trained on different features (momentum, volume, breakouts)
"""

import numpy as np
import pandas as pd
import talib
from typing import Dict, Optional, List
from dataclasses import dataclass


@dataclass
class TrendSetup:
    """Represents an extreme trend trade setup"""
    timestamp: pd.Timestamp
    entry_price: float
    direction: str  # 'LONG' or 'SHORT'

    # Trend indicators
    adx: float  # Trend strength
    di_spread: float  # Difference between +DI and -DI
    trend_duration: int  # Bars in current trend

    # Momentum indicators
    momentum: float
    roc: float  # Rate of change
    macd: float
    macd_signal: float

    # Volume confirmation
    volume_ratio: float
    volume_surge: bool

    # Breakout detection
    is_breakout: bool
    breakout_strength: float
    distance_from_breakout: float

    # Multi-timeframe alignment
    trend_1h: float
    trend_4h: float
    trend_alignment: float  # All timeframes aligned?

    # ML outputs
    confidence_score: float = 0.0
    predicted_continuation_prob: float = 0.0

    # Outcome
    actual_outcome: Optional[str] = None
    pnl_percent: Optional[float] = None


class TrendSignalGenerator:
    """Generate extreme trend signals"""

    @staticmethod
    def identify_extreme_long_trends(df: pd.DataFrame) -> pd.Series:
        """
        Identify EXTREME bullish trends - strong upward momentum

        Criteria:
        - ADX > 30 (strong trend)
        - +DI significantly above -DI (bullish direction)
        - Recent breakout above resistance
        - High volume surge
        - Multi-timeframe alignment (15m, 1H, 4H all bullish)
        - Strong momentum (ROC, MACD)
        """
        signals = (
            # Strong trend
            (df['adx'] > 30) &

            # Bullish direction (+DI > -DI by at least 10)
            (df['plus_di'] > df['minus_di'] + 10) &

            # Breakout (price above 20-bar high)
            (df['close'] > df['high'].rolling(20).max().shift(1)) &

            # Volume surge (2x average)
            (df['volume'] > df['volume'].rolling(20).mean() * 2) &

            # Strong momentum
            (df['roc'] > 2.0) &  # 2% rate of change

            # MACD bullish
            (df['macd'] > df['macd_signal']) &

            # Multi-timeframe bullish (if available)
            (df.get('trend_1h', 1.0) > 0) &
            (df.get('trend_4h', 1.0) > 0) &

            # Not overbought yet (RSI < 80)
            (df['rsi'] < 80) &

            # Higher highs structure
            (df['close'] > df['close'].shift(1)) &
            (df['close'].shift(1) > df['close'].shift(2))
        )

        return signals

    @staticmethod
    def identify_extreme_short_trends(df: pd.DataFrame) -> pd.Series:
        """
        Identify EXTREME bearish trends - strong downward momentum

        Criteria:
        - ADX > 30 (strong trend)
        - -DI significantly above +DI (bearish direction)
        - Recent breakdown below support
        - High volume surge
        - Multi-timeframe alignment (15m, 1H, 4H all bearish)
        - Strong downward momentum
        """
        signals = (
            # Strong trend
            (df['adx'] > 30) &

            # Bearish direction (-DI > +DI by at least 10)
            (df['minus_di'] > df['plus_di'] + 10) &

            # Breakdown (price below 20-bar low)
            (df['close'] < df['low'].rolling(20).min().shift(1)) &

            # Volume surge (2x average)
            (df['volume'] > df['volume'].rolling(20).mean() * 2) &

            # Strong downward momentum
            (df['roc'] < -2.0) &  # -2% rate of change

            # MACD bearish
            (df['macd'] < df['macd_signal']) &

            # Multi-timeframe bearish (if available)
            (df.get('trend_1h', -1.0) < 0) &
            (df.get('trend_4h', -1.0) < 0) &

            # Not oversold yet (RSI > 20)
            (df['rsi'] > 20) &

            # Lower lows structure
            (df['close'] < df['close'].shift(1)) &
            (df['close'].shift(1) < df['close'].shift(2))
        )

        return signals


class TrendFeatureEngineering:
    """Feature engineering specific to trend following"""

    @staticmethod
    def calculate_trend_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate features specific to trend following

        These are different from mean reversion features!
        """
        data = df.copy()

        # Basic features (if not already present)
        if 'rsi' not in data.columns:
            data['rsi'] = talib.RSI(data['close'], timeperiod=14)

        if 'adx' not in data.columns:
            data['adx'] = talib.ADX(data['high'], data['low'], data['close'], timeperiod=14)
            data['plus_di'] = talib.PLUS_DI(data['high'], data['low'], data['close'], timeperiod=14)
            data['minus_di'] = talib.MINUS_DI(data['high'], data['low'], data['close'], timeperiod=14)

        # Directional spread
        data['di_spread'] = data['plus_di'] - data['minus_di']

        # MACD
        data['macd'], data['macd_signal'], data['macd_hist'] = talib.MACD(
            data['close'], fastperiod=12, slowperiod=26, signalperiod=9
        )

        # Rate of Change (momentum)
        data['roc'] = talib.ROC(data['close'], timeperiod=10)
        data['roc_20'] = talib.ROC(data['close'], timeperiod=20)

        # Momentum
        if 'momentum' not in data.columns:
            data['momentum'] = talib.MOM(data['close'], timeperiod=10)

        # Trend duration (consecutive bars in same direction)
        data['is_up_bar'] = (data['close'] > data['close'].shift(1)).astype(int)
        data['is_down_bar'] = (data['close'] < data['close'].shift(1)).astype(int)

        # Count consecutive up/down bars
        data['trend_duration'] = 0
        up_count = 0
        down_count = 0

        for i in range(1, len(data)):
            if data['is_up_bar'].iloc[i]:
                up_count += 1
                down_count = 0
                data['trend_duration'].iloc[i] = up_count
            elif data['is_down_bar'].iloc[i]:
                down_count += 1
                up_count = 0
                data['trend_duration'].iloc[i] = -down_count
            else:
                up_count = 0
                down_count = 0

        # Breakout detection
        data['breakout_high'] = (data['close'] > data['high'].rolling(20).max().shift(1)).astype(int)
        data['breakout_low'] = (data['close'] < data['low'].rolling(20).min().shift(1)).astype(int)

        # Breakout strength (how far from breakout level)
        data['breakout_strength'] = np.where(
            data['breakout_high'],
            (data['close'] - data['high'].rolling(20).max().shift(1)) / data['close'],
            np.where(
                data['breakout_low'],
                (data['low'].rolling(20).min().shift(1) - data['close']) / data['close'],
                0
            )
        )

        # Volume analysis
        if 'volume_ratio' not in data.columns:
            data['volume_sma'] = data['volume'].rolling(20).mean()
            data['volume_ratio'] = data['volume'] / data['volume_sma']

        data['volume_surge'] = (data['volume'] > data['volume_sma'] * 2.5).astype(int)

        # Volatility expansion (trends often start with volatility expansion)
        data['atr'] = talib.ATR(data['high'], data['low'], data['close'], timeperiod=14)
        data['atr_ratio'] = data['atr'] / data['atr'].rolling(50).mean()

        # Multi-timeframe alignment score
        if 'trend_1h' in data.columns and 'trend_4h' in data.columns:
            # 15m trend direction
            data['trend_15m'] = np.where(data['di_spread'] > 0, 1.0, -1.0)

            # Alignment: all timeframes pointing same direction
            data['trend_alignment'] = np.where(
                (data['trend_15m'] * data['trend_1h'] * data['trend_4h']) > 0,
                1.0,  # All aligned
                0.0   # Not aligned
            )
        else:
            data['trend_alignment'] = 0.5  # Unknown

        # Price distance from EMAs (trending markets pull away from MAs)
        data['ema_20'] = talib.EMA(data['close'], timeperiod=20)
        data['ema_50'] = talib.EMA(data['close'], timeperiod=50)
        data['distance_from_ema20'] = (data['close'] - data['ema_20']) / data['ema_20']
        data['distance_from_ema50'] = (data['close'] - data['ema_50']) / data['ema_50']

        # EMA slope (trending EMAs have strong slopes)
        data['ema20_slope'] = data['ema_20'].pct_change(5)
        data['ema50_slope'] = data['ema_50'].pct_change(10)

        return data

    @staticmethod
    def get_trend_feature_list() -> List[str]:
        """Features to use for trend following ML model"""
        return [
            # Trend strength
            'adx', 'di_spread', 'trend_duration',

            # Momentum
            'roc', 'roc_20', 'momentum', 'macd', 'macd_hist',

            # Breakouts
            'breakout_high', 'breakout_low', 'breakout_strength',

            # Volume
            'volume_ratio', 'volume_surge',

            # Volatility
            'atr_ratio',

            # Multi-timeframe
            'trend_alignment',

            # Price-MA relationship
            'distance_from_ema20', 'distance_from_ema50',
            'ema20_slope', 'ema50_slope',

            # Context
            'rsi'  # To avoid entering overbought/oversold
        ]


class ExtremeTrendFollowingBot:
    """Main trend following bot"""

    def __init__(self, model_type: str = 'gradientboost', min_confidence: float = 0.70):
        """
        Initialize trend following bot

        Args:
            model_type: ML model type
            min_confidence: Minimum confidence for trading
        """
        from model_factory import EnhancedMLModel
        from ml_mean_reversion_bot import FeatureEngineering

        # Use ml_model for consistency with MLMeanReversionBot
        self.ml_model = EnhancedMLModel(model_type, min_confidence)
        self.signal_generator = TrendSignalGenerator()
        self.feature_engineer = TrendFeatureEngineering()
        self.feature_names = self.feature_engineer.get_trend_feature_list()

        # Also need base feature engineering for compatibility
        self.base_feature_engineer = FeatureEngineering()

    def train_model(self, df: pd.DataFrame, forward_periods: int = 10):
        """Train ML model on extreme trend data"""
        print("\n=== Training Extreme Trend Following Model ===\n")

        # Add features
        df_features = self.feature_engineer.calculate_trend_features(df)

        # Calculate forward returns
        df_features['forward_return'] = df_features['close'].pct_change(forward_periods).shift(-forward_periods)

        # Identify trend signals
        df_features['long_signal'] = self.signal_generator.identify_extreme_long_trends(df_features)
        df_features['short_signal'] = self.signal_generator.identify_extreme_short_trends(df_features)

        # Train on trend setups
        trend_data = df_features[df_features['long_signal'] | df_features['short_signal']].copy()

        if len(trend_data) > 30:
            print(f"Training on {len(trend_data)} extreme trend setups...")
            X = trend_data[self.feature_names].fillna(0)
            y = trend_data['forward_return']
            results = self.ml_model.train(X, y)
            print(f"✅ Model trained: {results['accuracy']:.2%} accuracy")
        else:
            print(f"⚠️  Only {len(trend_data)} trend setups found. Need at least 30 for training.")

        return df_features

    def should_enter_trade(self, df: pd.DataFrame, current_idx: int = -1) -> Optional[TrendSetup]:
        """Check if we should enter an extreme trend trade"""
        current = df.iloc[current_idx]

        # Check for trend signals
        is_long = self.signal_generator.identify_extreme_long_trends(df).iloc[current_idx]
        is_short = self.signal_generator.identify_extreme_short_trends(df).iloc[current_idx]

        if not (is_long or is_short):
            return None

        direction = 'LONG' if is_long else 'SHORT'

        # Create setup
        setup = TrendSetup(
            timestamp=current.name if hasattr(current, 'name') else pd.Timestamp.now(),
            entry_price=current['close'],
            direction=direction,
            adx=current['adx'],
            di_spread=current['di_spread'],
            trend_duration=int(current.get('trend_duration', 0)),
            momentum=current['momentum'],
            roc=current['roc'],
            macd=current['macd'],
            macd_signal=current['macd_signal'],
            volume_ratio=current['volume_ratio'],
            volume_surge=bool(current.get('volume_surge', False)),
            is_breakout=bool(current.get('breakout_high' if is_long else 'breakout_low', False)),
            breakout_strength=current.get('breakout_strength', 0),
            distance_from_breakout=current.get('distance_from_ema20', 0),
            trend_1h=current.get('trend_1h', 0),
            trend_4h=current.get('trend_4h', 0),
            trend_alignment=current.get('trend_alignment', 0)
        )

        # Get ML prediction
        features = current[self.feature_names]
        ml_prediction = self.ml_model.predict_with_confidence(features)

        setup.confidence_score = ml_prediction['confidence_score']
        setup.predicted_continuation_prob = ml_prediction['success_probability']

        if not ml_prediction['should_trade']:
            return None

        return setup

    def backtest(self, *args, **kwargs):
        """
        Backtest method stub - trend following uses same backtesting engine as mean reversion
        For now, raise NotImplementedError
        """
        raise NotImplementedError("Trend following backtest not yet fully implemented. Use meanrev or combined strategy.")

    def save_state(self, model_path: str, trades_path: str):
        """Save bot state"""
        self.ml_model.save_model(model_path)
