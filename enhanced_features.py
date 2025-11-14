"""
Enhanced Feature Engineering with Multi-Timeframe Analysis
==========================================================

Adds Phase 1-3 improvements:
- Market regime detection (ADX-based)
- Multi-timeframe features (1H, 4H)
- Temporal features (hour, day, session)
- Advanced pattern detection
"""

import numpy as np
import pandas as pd
import talib
from typing import Dict, Tuple


class EnhancedFeatureEngineering:
    """Enhanced feature engineering with regime detection and multi-timeframe analysis"""

    @staticmethod
    def detect_market_regime(df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect market regime: TRENDING_UP, TRENDING_DOWN, RANGING, HIGH_VOLATILITY

        Uses ADX and Directional Movement Indicators to classify market state.
        This is CRITICAL for mean reversion strategies - don't trade against strong trends!
        """
        data = df.copy()

        # Ensure we have ADX and DI indicators
        if 'adx' not in data.columns:
            data['adx'] = talib.ADX(data['high'], data['low'], data['close'], timeperiod=14)
        if 'plus_di' not in data.columns:
            data['plus_di'] = talib.PLUS_DI(data['high'], data['low'], data['close'], timeperiod=14)
        if 'minus_di' not in data.columns:
            data['minus_di'] = talib.MINUS_DI(data['high'], data['low'], data['close'], timeperiod=14)

        # ATR percentage for volatility classification
        if 'atr' not in data.columns:
            data['atr'] = talib.ATR(data['high'], data['low'], data['close'], timeperiod=14)
        data['atr_pct'] = (data['atr'] / data['close']) * 100

        # Classify regime for each bar
        def classify_regime(row):
            adx = row['adx']
            plus_di = row['plus_di']
            minus_di = row['minus_di']
            atr_pct = row['atr_pct']

            # High volatility check (top 25% of ATR)
            volatility_threshold = data['atr_pct'].quantile(0.75)

            if pd.isna(adx) or pd.isna(plus_di) or pd.isna(minus_di):
                return 'UNKNOWN'

            # Strong trend detection (ADX > 25)
            if adx > 25:
                if plus_di > minus_di + 5:  # Bullish trend
                    return 'TRENDING_UP'
                elif minus_di > plus_di + 5:  # Bearish trend
                    return 'TRENDING_DOWN'
                else:
                    return 'CHOPPY'  # High ADX but unclear direction

            # Ranging market (ADX < 20)
            elif adx < 20:
                if atr_pct > volatility_threshold:
                    return 'HIGH_VOLATILITY'  # Dangerous - skip trading
                else:
                    return 'RANGING'  # IDEAL for mean reversion!

            # Medium ADX (20-25)
            else:
                return 'TRANSITIONAL'  # Market forming direction

        data['market_regime'] = data.apply(classify_regime, axis=1)

        # Add numeric regime scores for ML model
        regime_mapping = {
            'RANGING': 0.0,
            'TRANSITIONAL': 0.3,
            'CHOPPY': 0.5,
            'TRENDING_UP': 0.8,
            'TRENDING_DOWN': -0.8,
            'HIGH_VOLATILITY': 1.0,
            'UNKNOWN': 0.0
        }
        data['regime_score'] = data['market_regime'].map(regime_mapping)

        # Trend strength indicator (for ML)
        data['trend_strength_indicator'] = np.where(
            data['adx'] > 25,
            data['adx'] / 50.0,  # Normalize to 0-1
            0.0
        )

        return data

    @staticmethod
    def add_multi_timeframe_features(df: pd.DataFrame, interval='15m') -> pd.DataFrame:
        """
        Add higher timeframe context (1H, 4H)

        Helps avoid trading against higher timeframe trends
        """
        data = df.copy()

        # Resample to 1H
        if interval == '15m':
            df_1h = data.resample('1H', on=data.index if isinstance(data.index, pd.DatetimeIndex) else data.index).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()

            if len(df_1h) > 50:
                # 1H RSI
                df_1h['rsi_1h'] = talib.RSI(df_1h['close'], timeperiod=14)

                # 1H trend (EMA crossover)
                df_1h['ema_20_1h'] = talib.EMA(df_1h['close'], timeperiod=20)
                df_1h['ema_50_1h'] = talib.EMA(df_1h['close'], timeperiod=50)
                df_1h['trend_1h'] = np.where(
                    df_1h['ema_20_1h'] > df_1h['ema_50_1h'],
                    1.0,  # Bullish
                    -1.0  # Bearish
                )

                # Merge back to 15m data
                data = data.merge(
                    df_1h[['rsi_1h', 'trend_1h']],
                    left_index=True,
                    right_index=True,
                    how='left'
                )
                data['rsi_1h'] = data['rsi_1h'].ffill()
                data['trend_1h'] = data['trend_1h'].ffill()
            else:
                data['rsi_1h'] = 50.0
                data['trend_1h'] = 0.0

            # Resample to 4H
            df_4h = data.resample('4H', on=data.index if isinstance(data.index, pd.DatetimeIndex) else data.index).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()

            if len(df_4h) > 50:
                # 4H trend (EMA crossover)
                df_4h['ema_20_4h'] = talib.EMA(df_4h['close'], timeperiod=20)
                df_4h['ema_50_4h'] = talib.EMA(df_4h['close'], timeperiod=50)
                df_4h['trend_4h'] = np.where(
                    df_4h['ema_20_4h'] > df_4h['ema_50_4h'],
                    1.0,  # Bullish
                    -1.0  # Bearish
                )

                # Merge back
                data = data.merge(
                    df_4h[['trend_4h']],
                    left_index=True,
                    right_index=True,
                    how='left'
                )
                data['trend_4h'] = data['trend_4h'].ffill()
            else:
                data['trend_4h'] = 0.0

        return data

    @staticmethod
    def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add time-based features (hour, day, trading session)

        Crypto markets have different behavior patterns by time of day
        """
        data = df.copy()

        # Ensure index is datetime
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)

        # Hour of day (0-23)
        data['hour'] = data.index.hour

        # Day of week (0=Monday, 6=Sunday)
        data['day_of_week'] = data.index.dayofweek

        # Trading sessions (UTC-based)
        # Asian: 0-8, London: 8-16, NY: 13-21 (overlap exists)
        data['is_asian_session'] = ((data['hour'] >= 0) & (data['hour'] < 8)).astype(int)
        data['is_london_session'] = ((data['hour'] >= 8) & (data['hour'] < 16)).astype(int)
        data['is_ny_session'] = ((data['hour'] >= 13) & (data['hour'] < 21)).astype(int)
        data['is_overlap'] = ((data['hour'] >= 13) & (data['hour'] < 16)).astype(int)  # London-NY overlap

        # Weekend flag (crypto still trades but often lower volume)
        data['is_weekend'] = (data['day_of_week'] >= 5).astype(int)

        # Cyclical encoding for hour (so 23 is close to 0)
        data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
        data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)

        # Cyclical encoding for day of week
        data['day_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
        data['day_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7)

        return data

    @staticmethod
    def add_advanced_patterns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add advanced price action pattern detection
        """
        data = df.copy()

        # Higher highs / lower lows detection (trend structure)
        data['higher_highs'] = (
            (data['high'] > data['high'].shift(1)) &
            (data['high'].shift(1) > data['high'].shift(2))
        ).astype(int)

        data['lower_lows'] = (
            (data['low'] < data['low'].shift(1)) &
            (data['low'].shift(1) < data['low'].shift(2))
        ).astype(int)

        # Double bottom / double top detection (simplified)
        data['double_bottom'] = (
            (data['low'] == data['low'].rolling(20).min()) &
            (data['low'].shift(10) == data['low'].shift(10).rolling(10).min()) &
            (abs(data['low'] - data['low'].shift(10)) / data['low'] < 0.02)
        ).astype(int)

        data['double_top'] = (
            (data['high'] == data['high'].rolling(20).max()) &
            (data['high'].shift(10) == data['high'].shift(10).rolling(10).max()) &
            (abs(data['high'] - data['high'].shift(10)) / data['high'] < 0.02)
        ).astype(int)

        # Breakout detection
        data['breakout_up'] = (
            (data['close'] > data['high'].rolling(20).max().shift(1))
        ).astype(int)

        data['breakout_down'] = (
            (data['close'] < data['low'].rolling(20).min().shift(1))
        ).astype(int)

        # Volume surge (3x average)
        data['volume_surge'] = (
            data['volume'] > data['volume'].rolling(20).mean() * 3
        ).astype(int)

        return data

    @staticmethod
    def add_volatility_percentile(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add volatility percentile (helps with dynamic thresholds)
        """
        data = df.copy()

        if 'atr' not in data.columns:
            data['atr'] = talib.ATR(data['high'], data['low'], data['close'], timeperiod=14)

        # Rolling 7-day percentile
        data['volatility_percentile'] = data['atr'].rolling(96 * 7).rank(pct=True)

        return data

    @staticmethod
    def get_enhanced_feature_list() -> list:
        """Return full list of enhanced features for ML model"""
        return [
            # Original mean reversion features
            'rsi', 'bb_position', 'bb_width', 'zscore', 'zscore_50',

            # Market context
            'volatility', 'atr', 'adx', 'trend_strength', 'volume_ratio', 'mfi',

            # NEW: Market regime
            'regime_score', 'trend_strength_indicator',

            # NEW: Multi-timeframe
            'rsi_1h', 'trend_1h', 'trend_4h',

            # NEW: Temporal
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
            'is_asian_session', 'is_london_session', 'is_ny_session', 'is_overlap', 'is_weekend',

            # Momentum
            'momentum', 'roc',

            # Pattern features
            'distance_to_high', 'distance_to_low', 'is_consolidating', 'drawdown',

            # NEW: Advanced patterns
            'higher_highs', 'lower_lows', 'double_bottom', 'double_top',
            'breakout_up', 'breakout_down', 'volume_surge',

            # NEW: Volatility percentile
            'volatility_percentile',

            # Candle patterns
            'doji', 'hammer', 'engulfing',

            # Feature interactions
            'rsi_bb_interaction', 'trend_volatility_interaction', 'volume_momentum_interaction'
        ]
