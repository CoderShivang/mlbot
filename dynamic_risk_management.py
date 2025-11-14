"""
Dynamic Risk Management - ATR-Based Stop Loss & Take Profit
===========================================================

Phase 2 improvement: Dynamic SL/TP that adapts to market volatility
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict


class DynamicRiskManager:
    """ATR-based dynamic stop loss and take profit calculator"""

    def __init__(
        self,
        sl_atr_multiplier: float = 1.5,
        tp_atr_multiplier: float = 3.0,
        min_risk_reward: float = 2.0,
        base_risk_pct: float = 0.03
    ):
        """
        Initialize dynamic risk manager

        Args:
            sl_atr_multiplier: ATR multiplier for stop loss (default: 1.5x)
            tp_atr_multiplier: ATR multiplier for take profit (default: 3.0x)
            min_risk_reward: Minimum risk:reward ratio (default: 1:2)
            base_risk_pct: Base risk percentage of capital (default: 3%)
        """
        self.sl_atr_multiplier = sl_atr_multiplier
        self.tp_atr_multiplier = tp_atr_multiplier
        self.min_risk_reward = min_risk_reward
        self.base_risk_pct = base_risk_pct

    def calculate_dynamic_levels(
        self,
        entry_price: float,
        atr: float,
        direction: str,
        volatility_regime: str = 'MEDIUM'
    ) -> Dict[str, float]:
        """
        Calculate dynamic SL/TP based on ATR and volatility regime

        Args:
            entry_price: Entry price
            atr: Current ATR value
            direction: 'LONG' or 'SHORT'
            volatility_regime: 'LOW', 'MEDIUM', or 'HIGH'

        Returns:
            Dictionary with stop_loss, take_profit, risk_amount, reward_amount
        """
        # Adjust multipliers based on volatility regime
        sl_mult = self.sl_atr_multiplier
        tp_mult = self.tp_atr_multiplier

        if volatility_regime == 'HIGH':
            # Wider stops in high volatility
            sl_mult *= 1.3
            tp_mult *= 1.3
        elif volatility_regime == 'LOW':
            # Tighter stops in low volatility
            sl_mult *= 0.8
            tp_mult *= 0.8

        # Calculate levels
        if direction == 'LONG':
            stop_loss = entry_price - (atr * sl_mult)
            take_profit = entry_price + (atr * tp_mult)
        else:  # SHORT
            stop_loss = entry_price + (atr * sl_mult)
            take_profit = entry_price - (atr * tp_mult)

        # Calculate risk and reward
        risk_amount = abs(entry_price - stop_loss)
        reward_amount = abs(take_profit - entry_price)

        # Ensure minimum risk:reward ratio
        actual_rr = reward_amount / risk_amount if risk_amount > 0 else 0

        if actual_rr < self.min_risk_reward:
            # Adjust TP to maintain minimum R:R
            if direction == 'LONG':
                take_profit = entry_price + (risk_amount * self.min_risk_reward)
            else:
                take_profit = entry_price - (risk_amount * self.min_risk_reward)

            reward_amount = abs(take_profit - entry_price)

        # Calculate stop loss and take profit percentages
        sl_pct = abs(stop_loss - entry_price) / entry_price
        tp_pct = abs(take_profit - entry_price) / entry_price

        return {
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'stop_loss_pct': sl_pct,
            'take_profit_pct': tp_pct,
            'risk_amount': risk_amount,
            'reward_amount': reward_amount,
            'risk_reward_ratio': reward_amount / risk_amount if risk_amount > 0 else 0,
            'atr_used': atr,
            'sl_multiplier': sl_mult,
            'tp_multiplier': tp_mult
        }

    def calculate_position_size(
        self,
        capital: float,
        entry_price: float,
        stop_loss: float,
        leverage: int = 10,
        confidence_score: float = 0.70
    ) -> Dict[str, float]:
        """
        Calculate position size with confidence-based scaling

        Args:
            capital: Available capital
            entry_price: Entry price
            stop_loss: Stop loss price
            leverage: Leverage to use
            confidence_score: ML model confidence (0.0-1.0)

        Returns:
            Dictionary with position_size, position_value, risk_amount
        """
        # Base risk percentage
        base_risk = self.base_risk_pct

        # Scale risk by confidence (1% to 5%)
        if confidence_score >= 0.80:
            risk_pct = 0.05  # 5% for very confident trades
        elif confidence_score >= 0.75:
            risk_pct = 0.04  # 4%
        elif confidence_score >= 0.70:
            risk_pct = 0.03  # 3%
        elif confidence_score >= 0.65:
            risk_pct = 0.02  # 2%
        else:
            risk_pct = 0.01  # 1% for marginal trades

        # Calculate risk amount in USD
        risk_usd = capital * risk_pct

        # Calculate stop loss distance
        sl_distance = abs(entry_price - stop_loss) / entry_price

        # Position size calculation
        # risk_usd = position_value * sl_distance
        # position_value = risk_usd / sl_distance
        position_value = risk_usd / sl_distance

        # Apply leverage
        position_value_with_leverage = position_value * leverage

        # Calculate number of contracts (for futures)
        contracts = position_value_with_leverage / entry_price

        return {
            'position_value': position_value,
            'position_value_leveraged': position_value_with_leverage,
            'contracts': contracts,
            'risk_pct': risk_pct,
            'risk_usd': risk_usd,
            'margin_required': position_value_with_leverage / leverage,
            'confidence_score': confidence_score
        }

    def calculate_trailing_stop(
        self,
        entry_price: float,
        current_price: float,
        direction: str,
        atr: float,
        trailing_multiplier: float = 2.0
    ) -> float:
        """
        Calculate trailing stop based on ATR

        Args:
            entry_price: Original entry price
            current_price: Current market price
            direction: 'LONG' or 'SHORT'
            atr: Current ATR
            trailing_multiplier: ATR multiplier for trailing distance

        Returns:
            Trailing stop price
        """
        trailing_distance = atr * trailing_multiplier

        if direction == 'LONG':
            # Trail stop up as price increases
            trailing_stop = current_price - trailing_distance
            # Never move stop loss down
            initial_stop = entry_price - (atr * self.sl_atr_multiplier)
            return max(trailing_stop, initial_stop)
        else:  # SHORT
            # Trail stop down as price decreases
            trailing_stop = current_price + trailing_distance
            # Never move stop loss up
            initial_stop = entry_price + (atr * self.sl_atr_multiplier)
            return min(trailing_stop, initial_stop)

    def get_dynamic_risk_summary(
        self,
        entry_price: float,
        atr: float,
        direction: str,
        volatility_regime: str,
        capital: float,
        leverage: int,
        confidence_score: float
    ) -> Dict:
        """
        Get complete risk management summary for a trade

        Returns all relevant risk metrics in one call
        """
        # Calculate SL/TP levels
        levels = self.calculate_dynamic_levels(
            entry_price, atr, direction, volatility_regime
        )

        # Calculate position size
        position = self.calculate_position_size(
            capital, entry_price, levels['stop_loss'], leverage, confidence_score
        )

        # Combine
        return {
            **levels,
            **position,
            'entry_price': entry_price,
            'direction': direction,
            'volatility_regime': volatility_regime
        }
