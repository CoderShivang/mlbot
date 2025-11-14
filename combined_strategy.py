"""
Combined Strategy - Mean Reversion + Trend Following
====================================================

Intelligently combines both strategies:
- Uses mean reversion in RANGING markets
- Uses trend following in TRENDING markets
- Filters out weak signals from both strategies
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict
from dataclasses import dataclass


@dataclass
class CombinedSetup:
    """Unified trade setup that could come from either strategy"""
    timestamp: pd.Timestamp
    entry_price: float
    direction: str
    strategy_type: str  # 'MEAN_REVERSION' or 'TREND_FOLLOWING'
    market_regime: str
    confidence_score: float
    predicted_success_prob: float
    setup_data: Dict  # Original setup data

    # Outcome
    actual_outcome: Optional[str] = None
    pnl_percent: Optional[float] = None


class CombinedStrategyBot:
    """
    Intelligent strategy selector that uses:
    - Mean reversion in ranging/choppy markets
    - Trend following in trending markets
    - Neither in high volatility / uncertain conditions
    """

    def __init__(
        self,
        meanrev_bot,
        trend_bot,
        min_confidence: float = 0.70
    ):
        """
        Initialize combined strategy

        Args:
            meanrev_bot: Initialized mean reversion bot
            trend_bot: Initialized trend following bot
            min_confidence: Minimum confidence for any trade
        """
        self.meanrev_bot = meanrev_bot
        self.trend_bot = trend_bot
        self.min_confidence = min_confidence

    def select_strategy(self, market_regime: str, adx: float) -> str:
        """
        Select which strategy to use based on market regime

        Args:
            market_regime: Current market regime
            adx: Current ADX value

        Returns:
            'MEAN_REVERSION', 'TREND_FOLLOWING', or 'NONE'
        """
        # RANGING market → Mean Reversion
        if market_regime == 'RANGING':
            return 'MEAN_REVERSION'

        # STRONG TREND → Trend Following
        elif market_regime in ['TRENDING_UP', 'TRENDING_DOWN']:
            if adx > 30:  # Strong trend
                return 'TREND_FOLLOWING'
            else:  # Weak trend, use mean reversion for pullbacks
                return 'MEAN_REVERSION'

        # HIGH VOLATILITY → Skip trading (too dangerous)
        elif market_regime == 'HIGH_VOLATILITY':
            return 'NONE'

        # CHOPPY / TRANSITIONAL → Use caution
        elif market_regime in ['CHOPPY', 'TRANSITIONAL']:
            # Only use mean reversion with high confidence
            return 'MEAN_REVERSION'

        # Unknown → Skip
        else:
            return 'NONE'

    def should_enter_trade(self, df: pd.DataFrame, current_idx: int = -1) -> Optional[CombinedSetup]:
        """
        Decide if and which strategy should take a trade

        Args:
            df: DataFrame with all features
            current_idx: Current bar index

        Returns:
            CombinedSetup if trade is approved, None otherwise
        """
        current = df.iloc[current_idx]

        # Get market regime
        market_regime = current.get('market_regime', 'UNKNOWN')
        adx = current.get('adx', 0)

        # Select strategy
        chosen_strategy = self.select_strategy(market_regime, adx)

        if chosen_strategy == 'NONE':
            return None

        # Try mean reversion strategy
        if chosen_strategy == 'MEAN_REVERSION':
            meanrev_setup = self.meanrev_bot.should_enter_trade(df, current_idx)

            if meanrev_setup:
                # Additional filter: Don't trade mean reversion against strong trends
                if market_regime in ['TRENDING_UP', 'TRENDING_DOWN'] and adx > 25:
                    # Only allow counter-trend if extremely confident
                    if meanrev_setup.confidence_score < 0.75:
                        return None

                # Wrap in combined setup
                return CombinedSetup(
                    timestamp=meanrev_setup.timestamp,
                    entry_price=meanrev_setup.entry_price,
                    direction=meanrev_setup.direction,
                    strategy_type='MEAN_REVERSION',
                    market_regime=market_regime,
                    confidence_score=meanrev_setup.confidence_score,
                    predicted_success_prob=meanrev_setup.predicted_success_prob,
                    setup_data=meanrev_setup.__dict__
                )

        # Try trend following strategy
        elif chosen_strategy == 'TREND_FOLLOWING':
            trend_setup = self.trend_bot.should_enter_trade(df, current_idx)

            if trend_setup:
                # Wrap in combined setup
                return CombinedSetup(
                    timestamp=trend_setup.timestamp,
                    entry_price=trend_setup.entry_price,
                    direction=trend_setup.direction,
                    strategy_type='TREND_FOLLOWING',
                    market_regime=market_regime,
                    confidence_score=trend_setup.confidence_score,
                    predicted_success_prob=trend_setup.predicted_continuation_prob,
                    setup_data=trend_setup.__dict__
                )

        return None

    def train_models(self, df: pd.DataFrame, forward_periods: int = 10):
        """Train both strategies"""
        print("\n" + "="*80)
        print("TRAINING COMBINED STRATEGY MODELS")
        print("="*80)

        # Train mean reversion
        print("\n[1/2] Training Mean Reversion Bot...")
        self.meanrev_bot.train_model(df, forward_periods)

        # Train trend following
        print("\n[2/2] Training Trend Following Bot...")
        self.trend_bot.train_model(df, forward_periods)

        print("\n✅ Combined strategy training complete!")
        print("="*80 + "\n")

    def get_strategy_stats(self, trades: list) -> Dict:
        """Get statistics broken down by strategy type"""
        if not trades:
            return {}

        df = pd.DataFrame(trades)

        stats = {
            'total_trades': len(df),
            'meanrev_trades': len(df[df.get('strategy_type') == 'MEAN_REVERSION']),
            'trend_trades': len(df[df.get('strategy_type') == 'TREND_FOLLOWING']),
        }

        # Win rates by strategy
        if stats['meanrev_trades'] > 0:
            meanrev_df = df[df['strategy_type'] == 'MEAN_REVERSION']
            stats['meanrev_win_rate'] = (meanrev_df['pnl_percent'] > 0).mean()

        if stats['trend_trades'] > 0:
            trend_df = df[df['strategy_type'] == 'TREND_FOLLOWING']
            stats['trend_win_rate'] = (trend_df['pnl_percent'] > 0).mean()

        return stats
