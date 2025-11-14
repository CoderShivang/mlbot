"""
Utilities for ML Mean Reversion Bot
===================================

Helper functions for analysis, monitoring, and visualization
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict
from ml_mean_reversion_bot import TradeSetup


class PerformanceAnalyzer:
    """Analyze trading bot performance"""
    
    @staticmethod
    def load_trade_history(filepath: str = 'trade_history.json') -> List[Dict]:
        """Load trade history from JSON file"""
        with open(filepath, 'r') as f:
            trades = json.load(f)
        return trades
    
    @staticmethod
    def calculate_metrics(trades: List[Dict]) -> Dict:
        """Calculate comprehensive performance metrics"""
        
        if not trades:
            return {}
        
        # Filter completed trades
        completed = [t for t in trades if t.get('actual_outcome') is not None]
        
        if not completed:
            return {}
        
        # Basic stats
        total_trades = len(completed)
        winning_trades = [t for t in completed if t['actual_outcome'] == 'WIN']
        losing_trades = [t for t in completed if t['actual_outcome'] == 'LOSS']
        
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        # PnL stats
        pnls = [t['pnl_percent'] for t in completed]
        avg_pnl = np.mean(pnls)
        median_pnl = np.median(pnls)
        
        wins_pnl = [t['pnl_percent'] for t in winning_trades]
        losses_pnl = [t['pnl_percent'] for t in losing_trades]
        
        avg_win = np.mean(wins_pnl) if wins_pnl else 0
        avg_loss = np.mean(losses_pnl) if losses_pnl else 0
        
        # Risk metrics
        max_win = max(pnls) if pnls else 0
        max_loss = min(pnls) if pnls else 0
        
        # Profit factor
        gross_profit = sum([p for p in pnls if p > 0])
        gross_loss = abs(sum([p for p in pnls if p < 0]))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
        
        # Expectancy
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * abs(avg_loss))
        
        # Consecutive wins/losses
        outcomes = [t['actual_outcome'] for t in completed]
        max_consecutive_wins = PerformanceAnalyzer._max_consecutive(outcomes, 'WIN')
        max_consecutive_losses = PerformanceAnalyzer._max_consecutive(outcomes, 'LOSS')
        
        # Direction analysis
        long_trades = [t for t in completed if t['direction'] == 'LONG']
        short_trades = [t for t in completed if t['direction'] == 'SHORT']
        
        long_win_rate = len([t for t in long_trades if t['actual_outcome'] == 'WIN']) / len(long_trades) if long_trades else 0
        short_win_rate = len([t for t in short_trades if t['actual_outcome'] == 'WIN']) / len(short_trades) if short_trades else 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'avg_pnl': avg_pnl,
            'median_pnl': median_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'max_win': max_win,
            'max_loss': max_loss,
            'profit_factor': profit_factor,
            'expectancy': expectancy,
            'max_consecutive_wins': max_consecutive_wins,
            'max_consecutive_losses': max_consecutive_losses,
            'long_trades': len(long_trades),
            'short_trades': len(short_trades),
            'long_win_rate': long_win_rate,
            'short_win_rate': short_win_rate
        }
    
    @staticmethod
    def _max_consecutive(outcomes: List[str], target: str) -> int:
        """Calculate maximum consecutive occurrences"""
        max_count = 0
        current_count = 0
        
        for outcome in outcomes:
            if outcome == target:
                current_count += 1
                max_count = max(max_count, current_count)
            else:
                current_count = 0
        
        return max_count
    
    @staticmethod
    def print_performance_report(trades: List[Dict]):
        """Print comprehensive performance report"""
        metrics = PerformanceAnalyzer.calculate_metrics(trades)
        
        if not metrics:
            print("No completed trades to analyze")
            return
        
        print("\n" + "="*70)
        print("PERFORMANCE REPORT")
        print("="*70)
        
        print(f"\n📊 Trade Statistics:")
        print(f"   Total Trades:           {metrics['total_trades']}")
        print(f"   Winning Trades:         {metrics['winning_trades']}")
        print(f"   Losing Trades:          {metrics['losing_trades']}")
        print(f"   Win Rate:               {metrics['win_rate']:.2%}")
        
        print(f"\n💰 PnL Statistics:")
        print(f"   Average PnL:            {metrics['avg_pnl']:.2%}")
        print(f"   Median PnL:             {metrics['median_pnl']:.2%}")
        print(f"   Average Win:            {metrics['avg_win']:.2%}")
        print(f"   Average Loss:           {metrics['avg_loss']:.2%}")
        print(f"   Max Win:                {metrics['max_win']:.2%}")
        print(f"   Max Loss:               {metrics['max_loss']:.2%}")
        
        print(f"\n📈 Performance Metrics:")
        print(f"   Profit Factor:          {metrics['profit_factor']:.2f}")
        print(f"   Expectancy:             {metrics['expectancy']:.2%}")
        
        print(f"\n🔄 Streak Analysis:")
        print(f"   Max Consecutive Wins:   {metrics['max_consecutive_wins']}")
        print(f"   Max Consecutive Losses: {metrics['max_consecutive_losses']}")
        
        print(f"\n📉 Direction Analysis:")
        print(f"   Long Trades:            {metrics['long_trades']} ({metrics['long_win_rate']:.2%} win rate)")
        print(f"   Short Trades:           {metrics['short_trades']} ({metrics['short_win_rate']:.2%} win rate)")
        
        print("="*70 + "\n")


class Visualizer:
    """Create visualizations for bot performance"""
    
    @staticmethod
    def plot_equity_curve(trades: List[Dict], initial_capital: float = 10000):
        """Plot equity curve from trades"""
        if not trades:
            print("No trades to plot")
            return
        
        # Calculate cumulative returns
        pnls = [t['pnl_percent'] for t in trades if t.get('pnl_percent') is not None]
        
        capital = initial_capital
        equity_curve = [capital]
        
        for pnl in pnls:
            capital *= (1 + pnl)
            equity_curve.append(capital)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(equity_curve, linewidth=2, color='steelblue')
        ax.axhline(y=initial_capital, color='red', linestyle='--', alpha=0.5, label='Initial Capital')
        ax.fill_between(range(len(equity_curve)), initial_capital, equity_curve, 
                        where=[e >= initial_capital for e in equity_curve],
                        alpha=0.3, color='green', label='Profit')
        ax.fill_between(range(len(equity_curve)), initial_capital, equity_curve,
                        where=[e < initial_capital for e in equity_curve],
                        alpha=0.3, color='red', label='Loss')
        
        ax.set_title('Equity Curve', fontsize=14, fontweight='bold')
        ax.set_xlabel('Trade Number')
        ax.set_ylabel('Capital ($)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_pnl_distribution(trades: List[Dict]):
        """Plot PnL distribution"""
        pnls = [t['pnl_percent'] * 100 for t in trades if t.get('pnl_percent') is not None]
        
        if not pnls:
            print("No PnL data to plot")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        ax1.hist(pnls, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
        ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Break-even')
        ax1.axvline(x=np.mean(pnls), color='green', linestyle='--', linewidth=2, label='Mean')
        ax1.set_title('PnL Distribution', fontsize=12, fontweight='bold')
        ax1.set_xlabel('PnL (%)')
        ax1.set_ylabel('Frequency')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Box plot
        ax2.boxplot(pnls, vert=True)
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax2.set_title('PnL Statistics', fontsize=12, fontweight='bold')
        ax2.set_ylabel('PnL (%)')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_confidence_analysis(trades: List[Dict]):
        """Plot performance by confidence levels"""
        confidences = [t['confidence_score'] * 100 for t in trades if t.get('confidence_score') is not None]
        pnls = [t['pnl_percent'] * 100 for t in trades if t.get('pnl_percent') is not None]
        
        if len(confidences) != len(pnls):
            print("Mismatched data")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Scatter plot
        colors = ['green' if p > 0 else 'red' for p in pnls]
        ax.scatter(confidences, pnls, c=colors, alpha=0.6, s=50)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax.set_title('Trade Performance vs ML Confidence', fontsize=14, fontweight='bold')
        ax.set_xlabel('ML Confidence Score (%)')
        ax.set_ylabel('Trade PnL (%)')
        ax.grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(confidences, pnls, 1)
        p = np.poly1d(z)
        ax.plot(sorted(confidences), p(sorted(confidences)), "b--", alpha=0.5, label='Trend')
        ax.legend()
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_feature_importance(model_path: str = 'ml_mean_reversion_model.pkl'):
        """Plot feature importance from trained model"""
        import pickle
        
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            model = model_data['model']
            feature_names = model_data['feature_names']
            
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.barh(range(len(importance_df)), importance_df['importance'])
            ax.set_yticks(range(len(importance_df)))
            ax.set_yticklabels(importance_df['feature'])
            ax.invert_yaxis()
            ax.set_xlabel('Importance')
            ax.set_title('Feature Importance', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x')
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    
    @staticmethod
    def create_dashboard(trades: List[Dict], initial_capital: float = 10000,
                        model_path: str = 'ml_mean_reversion_model.pkl'):
        """Create comprehensive dashboard"""
        fig = plt.figure(figsize=(16, 12))
        
        # Equity curve
        ax1 = plt.subplot(3, 2, 1)
        Visualizer.plot_equity_curve(trades, initial_capital)
        
        # PnL distribution
        ax2 = plt.subplot(3, 2, 2)
        pnls = [t['pnl_percent'] * 100 for t in trades if t.get('pnl_percent') is not None]
        ax2.hist(pnls, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
        ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax2.set_title('PnL Distribution')
        ax2.set_xlabel('PnL (%)')
        ax2.grid(True, alpha=0.3)
        
        # Confidence vs PnL
        ax3 = plt.subplot(3, 2, 3)
        confidences = [t['confidence_score'] * 100 for t in trades if t.get('confidence_score') is not None]
        colors = ['green' if p > 0 else 'red' for p in pnls]
        ax3.scatter(confidences, pnls, c=colors, alpha=0.6)
        ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax3.set_title('Confidence vs PnL')
        ax3.set_xlabel('Confidence (%)')
        ax3.set_ylabel('PnL (%)')
        ax3.grid(True, alpha=0.3)
        
        # Win rate by confidence quartile
        ax4 = plt.subplot(3, 2, 4)
        df = pd.DataFrame(trades)
        if 'confidence_score' in df.columns and 'actual_outcome' in df.columns:
            df['confidence_quartile'] = pd.qcut(df['confidence_score'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
            win_rates = df.groupby('confidence_quartile')['actual_outcome'].apply(lambda x: (x == 'WIN').mean())
            ax4.bar(win_rates.index, win_rates.values, color='steelblue')
            ax4.set_title('Win Rate by Confidence Quartile')
            ax4.set_ylabel('Win Rate')
            ax4.set_ylim([0, 1])
            ax4.grid(True, alpha=0.3, axis='y')
        
        # Trade outcomes over time
        ax5 = plt.subplot(3, 2, 5)
        outcomes = [1 if t['actual_outcome'] == 'WIN' else -1 for t in trades if t.get('actual_outcome')]
        cumulative_outcomes = np.cumsum(outcomes)
        ax5.plot(cumulative_outcomes, linewidth=2)
        ax5.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax5.set_title('Cumulative Win/Loss')
        ax5.set_xlabel('Trade Number')
        ax5.set_ylabel('Net Wins')
        ax5.grid(True, alpha=0.3)
        
        # Direction analysis
        ax6 = plt.subplot(3, 2, 6)
        df = pd.DataFrame(trades)
        if 'direction' in df.columns and 'actual_outcome' in df.columns:
            direction_stats = df.groupby('direction')['actual_outcome'].apply(lambda x: (x == 'WIN').mean())
            ax6.bar(direction_stats.index, direction_stats.values, color=['green', 'red'])
            ax6.set_title('Win Rate by Direction')
            ax6.set_ylabel('Win Rate')
            ax6.set_ylim([0, 1])
            ax6.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        return fig


class TradingJournal:
    """Maintain a detailed trading journal"""
    
    def __init__(self, filepath: str = 'trading_journal.md'):
        self.filepath = filepath
    
    def add_entry(self, trade: Dict, notes: str = ""):
        """Add entry to trading journal"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        entry = f"""
## Trade on {timestamp}

**Direction:** {trade.get('direction', 'N/A')}
**Entry Price:** ${trade.get('entry_price', 0):.2f}
**Confidence:** {trade.get('confidence_score', 0)*100:.1f}%

### Market Context
- RSI: {trade.get('rsi', 0):.1f}
- Z-Score: {trade.get('zscore', 0):.2f}
- Volatility: {trade.get('volatility_regime', 'N/A')}
- Trend: {trade.get('trend_strength', 0):.2f}

### Outcome
- Result: {trade.get('actual_outcome', 'PENDING')}
- PnL: {trade.get('pnl_percent', 0)*100:.2f}%

### Notes
{notes}

---

"""
        
        with open(self.filepath, 'a') as f:
            f.write(entry)
        
        print(f"✅ Added entry to trading journal: {self.filepath}")


# Quick usage functions
def quick_analysis(trade_history_path: str = 'trade_history.json'):
    """Quick performance analysis"""
    analyzer = PerformanceAnalyzer()
    trades = analyzer.load_trade_history(trade_history_path)
    analyzer.print_performance_report(trades)
    return trades


def create_report(trade_history_path: str = 'trade_history.json',
                 output_path: str = '/mnt/user-data/outputs/performance_report.png'):
    """Create visual performance report"""
    analyzer = PerformanceAnalyzer()
    trades = analyzer.load_trade_history(trade_history_path)
    
    visualizer = Visualizer()
    fig = visualizer.create_dashboard(trades)
    
    if fig:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✅ Report saved to: {output_path}")
    
    return fig


if __name__ == "__main__":
    print("ML Mean Reversion Bot - Utilities")
    print("="*60)
    print("\nAvailable functions:")
    print("  quick_analysis(path)  - Print performance metrics")
    print("  create_report(path)   - Generate visual report")
    print("\nExample:")
    print("  python utils.py")
