"""
Research-Backed Trend Following Strategy for Bitcoin
====================================================

Based on academic research:
- Beluška & Vojtko (2024): "Revisiting Trend-following and Mean-Reversion Strategies in Bitcoin"
- MAX strategy (momentum) significantly outperforms MIN (mean reversion) on Bitcoin 2015-2024
- Alpaca Markets: Successful Bitcoin trend strategies using momentum + breakouts

Key Principles:
1. Follow strong momentum (not fight it)
2. Breakout confirmation with volume
3. Multi-timeframe alignment
4. Minimal filtering (let the model decide)
5. Target: 50-100+ trades per year on 15m timeframe

Performance Target:
- Sharpe Ratio: >2.0 (research shows 3.0+ is achievable)
- Win Rate: 50-60% (with proper R:R this is profitable)
- Trades: 50-100+ per year
"""

import numpy as np
import pandas as pd
import talib
from typing import Dict, Optional, List
from dataclasses import dataclass


@dataclass
class TrendTradeSetup:
    """Trade setup for trend following"""
    timestamp: pd.Timestamp
    entry_price: float
    direction: str  # 'LONG' or 'SHORT'

    # Momentum indicators
    momentum_score: float  # Combined momentum measure
    roc_10: float  # 10-period rate of change
    roc_20: float  # 20-period rate of change

    # Trend indicators
    adx: float
    ema_9: float
    ema_21: float
    ema_50: float

    # Volume
    volume_ratio: float
    volume_surge: bool

    # Breakout
    is_breakout: bool
    bars_since_breakout: int

    # ML output
    confidence: float = 0.5
    predicted_success: float = 0.5

    # Outcome
    actual_outcome: Optional[str] = None
    pnl_percent: Optional[float] = None


class ResearchBackedTrendSignals:
    """
    Trend signals based on academic research

    MAX Strategy (Beluška & Vojtko 2024):
    - Asset with highest recent momentum tends to continue
    - Focus on rate of change (ROC) over multiple periods
    """

    @staticmethod
    def calculate_momentum_score(df: pd.DataFrame) -> pd.Series:
        """
        Calculate composite momentum score

        Based on research: Combine multiple momentum measures
        """
        # Rate of change over different periods
        roc_5 = talib.ROC(df['close'], timeperiod=5)
        roc_10 = talib.ROC(df['close'], timeperiod=10)
        roc_20 = talib.ROC(df['close'], timeperiod=20)

        # Normalize and combine (equal weight)
        momentum_score = (roc_5 + roc_10 + roc_20) / 3

        return momentum_score

    @staticmethod
    def identify_long_trends(df: pd.DataFrame) -> pd.Series:
        """
        Identify bullish trend entries

        Research-backed criteria:
        1. Strong positive momentum (top 30% of recent momentum)
        2. Price above key EMAs (9, 21)
        3. Volume confirmation (above average)
        4. Optional: Breakout above recent high
        """
        # Calculate momentum
        momentum_score = ResearchBackedTrendSignals.calculate_momentum_score(df)

        # EMAs
        ema_9 = talib.EMA(df['close'], timeperiod=9)
        ema_21 = talib.EMA(df['close'], timeperiod=21)
        ema_50 = talib.EMA(df['close'], timeperiod=50)

        # Volume
        volume_sma = df['volume'].rolling(20).mean()
        volume_ratio = df['volume'] / volume_sma

        # Breakout detection (20-bar high)
        high_20 = df['high'].rolling(20).max()
        is_breakout = df['close'] > high_20.shift(1)

        # ADX for trend strength
        adx = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)

        # LONG signal criteria (LESS RESTRICTIVE than before)
        long_signals = (
            # Strong momentum (top 30% of rolling 100-bar momentum)
            (momentum_score > momentum_score.rolling(100).quantile(0.70)) &

            # Price structure: above EMA9 and EMA21
            (df['close'] > ema_9) &
            (df['close'] > ema_21) &

            # EMA alignment (short-term above mid-term)
            (ema_9 > ema_21) &

            # Volume above average (confirms strength)
            (volume_ratio > 1.0) &

            # Moderate trend strength (ADX > 20, not too restrictive)
            (adx > 20) &

            # Not extremely overbought (RSI < 85, very generous)
            (talib.RSI(df['close'], timeperiod=14) < 85)
        )

        return long_signals

    @staticmethod
    def identify_short_trends(df: pd.DataFrame) -> pd.Series:
        """
        Identify bearish trend entries

        Mirror of long criteria
        """
        # Calculate momentum
        momentum_score = ResearchBackedTrendSignals.calculate_momentum_score(df)

        # EMAs
        ema_9 = talib.EMA(df['close'], timeperiod=9)
        ema_21 = talib.EMA(df['close'], timeperiod=21)
        ema_50 = talib.EMA(df['close'], timeperiod=50)

        # Volume
        volume_sma = df['volume'].rolling(20).mean()
        volume_ratio = df['volume'] / volume_sma

        # Breakdown detection (20-bar low)
        low_20 = df['low'].rolling(20).min()
        is_breakdown = df['close'] < low_20.shift(1)

        # ADX for trend strength
        adx = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)

        # SHORT signal criteria (LESS RESTRICTIVE)
        short_signals = (
            # Strong negative momentum (bottom 30% of rolling 100-bar momentum)
            (momentum_score < momentum_score.rolling(100).quantile(0.30)) &

            # Price structure: below EMA9 and EMA21
            (df['close'] < ema_9) &
            (df['close'] < ema_21) &

            # EMA alignment (short-term below mid-term)
            (ema_9 < ema_21) &

            # Volume above average
            (volume_ratio > 1.0) &

            # Moderate trend strength
            (adx > 20) &

            # Not extremely oversold (RSI > 15, very generous)
            (talib.RSI(df['close'], timeperiod=14) > 15)
        )

        return short_signals


class TrendFeatureEngineer:
    """Feature engineering for trend following ML model"""

    @staticmethod
    def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all features for trend following

        Focus on momentum and trend strength, not mean reversion
        """
        data = df.copy()

        # === MOMENTUM FEATURES ===
        data['roc_5'] = talib.ROC(data['close'], timeperiod=5)
        data['roc_10'] = talib.ROC(data['close'], timeperiod=10)
        data['roc_20'] = talib.ROC(data['close'], timeperiod=20)
        data['momentum_10'] = talib.MOM(data['close'], timeperiod=10)
        data['momentum_score'] = (data['roc_5'] + data['roc_10'] + data['roc_20']) / 3

        # === TREND INDICATORS ===
        data['ema_9'] = talib.EMA(data['close'], timeperiod=9)
        data['ema_21'] = talib.EMA(data['close'], timeperiod=21)
        data['ema_50'] = talib.EMA(data['close'], timeperiod=50)
        data['ema_200'] = talib.EMA(data['close'], timeperiod=200)

        # EMA relationships
        data['ema_9_21_diff'] = (data['ema_9'] - data['ema_21']) / data['close']
        data['ema_21_50_diff'] = (data['ema_21'] - data['ema_50']) / data['close']
        data['price_ema9_diff'] = (data['close'] - data['ema_9']) / data['close']

        # === TREND STRENGTH ===
        data['adx'] = talib.ADX(data['high'], data['low'], data['close'], timeperiod=14)
        data['plus_di'] = talib.PLUS_DI(data['high'], data['low'], data['close'], timeperiod=14)
        data['minus_di'] = talib.MINUS_DI(data['high'], data['low'], data['close'], timeperiod=14)
        data['di_diff'] = data['plus_di'] - data['minus_di']

        # === MACD ===
        data['macd'], data['macd_signal'], data['macd_hist'] = talib.MACD(
            data['close'], fastperiod=12, slowperiod=26, signalperiod=9
        )

        # === VOLUME ===
        data['volume_sma_20'] = data['volume'].rolling(20).mean()
        data['volume_ratio'] = data['volume'] / data['volume_sma_20']
        data['volume_surge'] = (data['volume'] > data['volume_sma_20'] * 1.5).astype(int)

        # OBV (On-Balance Volume)
        data['obv'] = talib.OBV(data['close'], data['volume'])
        data['obv_ema'] = talib.EMA(data['obv'], timeperiod=20)

        # === VOLATILITY ===
        data['atr'] = talib.ATR(data['high'], data['low'], data['close'], timeperiod=14)
        data['atr_pct'] = (data['atr'] / data['close']) * 100

        # === BREAKOUTS ===
        data['high_20'] = data['high'].rolling(20).max()
        data['low_20'] = data['low'].rolling(20).min()
        data['breakout_high'] = (data['close'] > data['high_20'].shift(1)).astype(int)
        data['breakout_low'] = (data['close'] < data['low_20'].shift(1)).astype(int)

        # Distance from 20-bar high/low
        data['dist_from_high20'] = (data['high_20'] - data['close']) / data['close']
        data['dist_from_low20'] = (data['close'] - data['low_20']) / data['close']

        # === PRICE ACTION ===
        # Higher highs / lower lows
        data['higher_high'] = (
            (data['high'] > data['high'].shift(1)) &
            (data['high'].shift(1) > data['high'].shift(2))
        ).astype(int)

        data['lower_low'] = (
            (data['low'] < data['low'].shift(1)) &
            (data['low'].shift(1) < data['low'].shift(2))
        ).astype(int)

        # === RSI (for context, not primary signal) ===
        data['rsi'] = talib.RSI(data['close'], timeperiod=14)

        return data

    @staticmethod
    def get_feature_list() -> List[str]:
        """Features for ML model"""
        return [
            # Momentum (most important for trend following)
            'roc_5', 'roc_10', 'roc_20', 'momentum_10', 'momentum_score',

            # Trend strength
            'adx', 'di_diff', 'plus_di', 'minus_di',

            # MACD
            'macd', 'macd_signal', 'macd_hist',

            # EMA relationships
            'ema_9_21_diff', 'ema_21_50_diff', 'price_ema9_diff',

            # Volume
            'volume_ratio', 'volume_surge',

            # Volatility
            'atr_pct',

            # Breakouts
            'breakout_high', 'breakout_low', 'dist_from_high20', 'dist_from_low20',

            # Price action
            'higher_high', 'lower_low',

            # Context
            'rsi'
        ]


class ResearchBackedTrendBot:
    """
    Trend following bot based on 2024 academic research

    Goal: Achieve Sharpe ratio >2.0 with 50-100+ trades per year
    """

    def __init__(self, model_type: str = 'randomforest', min_confidence: float = 0.45):
        """
        Initialize trend bot

        Args:
            model_type: 'randomforest' (recommended), 'gradientboost', 'xgboost', 'ensemble'
            min_confidence: Lower threshold for more trades (default: 0.45 vs old 0.70)
        """
        from model_factory import EnhancedMLModel
        from ml_mean_reversion_bot import FeatureEngineering

        # ML model with LOWER confidence threshold
        self.ml_model = EnhancedMLModel(model_type, min_confidence)

        # Feature engineering
        self.signal_generator = ResearchBackedTrendSignals()
        self.feature_engineer = TrendFeatureEngineer()
        self.feature_names = self.feature_engineer.get_feature_list()

        # Base features for compatibility
        self.base_feature_engineer = FeatureEngineering()

    def train_model(self, df: pd.DataFrame, forward_periods: int = 10):
        """Train ML model on trend data"""
        print("\n=== Training Research-Backed Trend Following Model ===")
        print("Based on: Beluška & Vojtko (2024) - MAX strategy research\n")

        # Calculate trend features
        df_features = self.feature_engineer.calculate_features(df)

        # Calculate forward returns
        df_features['forward_return'] = df_features['close'].pct_change(forward_periods).shift(-forward_periods)

        # Identify trend signals (LESS RESTRICTIVE)
        df_features['long_signal'] = self.signal_generator.identify_long_trends(df_features)
        df_features['short_signal'] = self.signal_generator.identify_short_trends(df_features)

        # Combine signals
        trend_data = df_features[df_features['long_signal'] | df_features['short_signal']].copy()

        print(f"Found {len(trend_data)} trend setups (target: 50-100+ per year)")

        if len(trend_data) > 20:  # Lower threshold than before (was 30)
            X = trend_data[self.feature_names].fillna(0)
            y = trend_data['forward_return']

            results = self.ml_model.train(X, y)
            print(f"✅ Model trained: {results['accuracy']:.2%} accuracy")
            print(f"   Model type: {self.ml_model.model_type}")
            print(f"   Min confidence: {self.ml_model.min_confidence:.0%}")
        else:
            print(f"⚠️  Only {len(trend_data)} setups - need more data or adjust parameters")

        return df_features

    def should_enter_trade(self, df: pd.DataFrame, current_idx: int = -1) -> Optional[TrendTradeSetup]:
        """Check if we should enter a trend trade"""
        current = df.iloc[current_idx]

        # Check signals
        is_long = self.signal_generator.identify_long_trends(df).iloc[current_idx]
        is_short = self.signal_generator.identify_short_trends(df).iloc[current_idx]

        if not (is_long or is_short):
            return None

        direction = 'LONG' if is_long else 'SHORT'

        # Create setup
        setup = TrendTradeSetup(
            timestamp=current.name if hasattr(current, 'name') else pd.Timestamp.now(),
            entry_price=current['close'],
            direction=direction,
            momentum_score=current.get('momentum_score', 0),
            roc_10=current.get('roc_10', 0),
            roc_20=current.get('roc_20', 0),
            adx=current.get('adx', 0),
            ema_9=current.get('ema_9', current['close']),
            ema_21=current.get('ema_21', current['close']),
            ema_50=current.get('ema_50', current['close']),
            volume_ratio=current.get('volume_ratio', 1.0),
            volume_surge=bool(current.get('volume_surge', False)),
            is_breakout=bool(current.get('breakout_high' if is_long else 'breakout_low', False)),
            bars_since_breakout=0  # Placeholder
        )

        # Get ML prediction (with LOWER threshold)
        features = current[self.feature_names]
        ml_prediction = self.ml_model.predict_with_confidence(features)

        setup.confidence = ml_prediction['confidence_score']
        setup.predicted_success = ml_prediction['success_probability']

        # LESS RESTRICTIVE: Lower success probability threshold
        if ml_prediction['success_probability'] < 0.45:  # Was 0.60 before
            return None

        if not ml_prediction['should_trade']:
            return None

        return setup

    def backtest(self, *args, **kwargs):
        """Backtest stub - uses same engine as mean reversion"""
        raise NotImplementedError("Use mean reversion backtest engine with this bot")

    def save_state(self, model_path: str, trades_path: str):
        """Save bot state"""
        self.ml_model.save_model(model_path)
