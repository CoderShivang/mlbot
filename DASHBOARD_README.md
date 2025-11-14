# ML Mean Reversion Bot - Interactive Dashboard

Comprehensive web-based dashboard for analyzing backtest results with interactive charts and filtering.

## Features

✅ **Header Metrics**
- Win Rate, Total Trades, Net P&L at a glance

✅ **Day of Week Analysis**
- Stacked bar chart showing wins/losses by day
- Win rate percentage chart by day

✅ **Equity Curve & Drawdown**
- Real-time equity progression with markers for max profit/drawdown
- Separate underwater equity (drawdown) chart

✅ **Daily P&L Table**
- Interactive table with daily summaries
- Click any row to filter trades to that specific day
- Color-coded: Green for profitable days, Red for losing days

✅ **Trade Filtering**
- Filter by Outcome (All/Winners/Losers)
- Filter by Direction (All/LONG/SHORT)
- Filter by Signal Type
- Combine with date filtering

✅ **Trade List**
- Detailed trade-by-trade breakdown
- Click any trade to view details
- Sortable columns

✅ **Trade Details**
- Individual trade information
- Entry/exit prices, P&L, reasoning

## Installation

### 1. Install Dashboard Dependencies

```bash
pip install -r requirements_dashboard.txt
```

Or manually:

```bash
pip install dash>=2.14.0 plotly>=5.18.0 dash-bootstrap-components>=1.5.0
```

### 2. Run a Backtest First

The dashboard loads data from `/mnt/user-data/outputs/trade_history.json`, so you must run a backtest first:

```bash
python example_usage.py
```

This will:
- Fetch 90 days of mainnet data
- Train the ML model
- Run backtest
- Generate `trade_history.json`

## Usage

### Start the Dashboard

```bash
python dashboard_app.py
```

The dashboard will start on: **http://localhost:8050**

Open your browser and navigate to that URL.

### Dashboard Navigation

1. **View Overall Performance**
   - Check header metrics for quick overview
   - Review day-of-week charts to find best trading days

2. **Analyze Equity Progression**
   - See equity curve with max profit/drawdown markers
   - Check drawdown chart for risk assessment

3. **Daily Breakdown**
   - Review daily P&L table
   - Click any day to filter trades to that date
   - Use "Clear Filter" button to remove date filter

4. **Filter Trades**
   - Use dropdowns to filter by outcome, direction
   - Combine filters to find specific trade patterns
   - Example: "Show me all losing LONG trades"

5. **Individual Trade Analysis**
   - Scroll to trade list
   - Click any trade row to view details
   - See entry/exit prices, P&L, and reasoning

### Example Workflow

```
1. Run backtest:
   $ python example_usage.py

2. Start dashboard:
   $ python dashboard_app.py

3. In browser (http://localhost:8050):
   - Check overall win rate and P&L
   - Notice which day has best win rate
   - Click a losing day in Daily P&L table
   - Filter to "Losers" only
   - Review what went wrong on those trades
   - Click individual trades to see details
```

## Dashboard Sections Explained

### Header Metrics
Shows 3 key numbers:
- **Win Rate**: Percentage of winning trades (green if >60%, yellow if 50-60%, red if <50%)
- **Total Trades**: Number of trades executed
- **Net P&L**: Total profit/loss in $ and % (green if positive, red if negative)

### Day of Week Analysis

**Trades by Day of Week (Stacked Bar)**
- Green bars = Winning trades
- Red bars = Losing trades
- Shows which days are most active

**Win Rate by Day of Week**
- Green bars = Win rate >50%
- Red bars = Win rate <50%
- Dashed line at 50% = breakeven
- Helps identify best/worst trading days

### Equity Curve & Drawdown

**Equity Curve (Top)**
- Blue line = Your capital over time
- Starts at $100
- Shaded area under curve
- Red diamond = Max drawdown point
- Green star = Max profit point
- Dashed line = Initial capital ($100)

**Drawdown Chart (Bottom)**
- Red line = Percentage below peak equity
- Always negative or zero
- Shows risk exposure
- Lower = worse drawdown

### Daily P&L Analysis

Interactive table with columns:
- **Date**: Trading date
- **Total P&L**: Net profit/loss for the day
- **Total Fees**: Estimated fees paid
- **Trades**: Number of trades that day
- **Wins**: Number of winning trades
- **Losses**: Number of losing trades
- **Win Rate**: Percentage of winners

**Features:**
- Click column headers to sort
- Click any row to filter all trades to that day
- Green rows = Profitable days
- Red rows = Losing days

### Filter Trades

Three dropdown filters:

**Outcome:**
- All Trades
- Winners only (P&L > 0)
- Losers only (P&L ≤ 0)

**Direction:**
- All Directions
- LONG only
- SHORT only

**Signal Type:**
- All Types
- Mean Reversion (currently all trades are this type)

Filters combine with date selection from Daily P&L table.

### Trade List

Table showing all trades (filtered):
- **#**: Trade number
- **Direction**: LONG or SHORT
- **Type**: mean_reversion
- **Entry Time**: When trade was entered
- **Entry Price**: BTC price at entry
- **Exit Price**: BTC price at exit
- **P&L %**: Profit/loss as percentage
- **Reason**: Why the trade was taken (RSI, Z-score values)

**Features:**
- Click any row to view full trade details
- Green P&L = Winner
- Red P&L = Loser
- Paginated (10 trades per page)

### Trade Details

When you click a trade, you see:
- Direction and type
- Entry price and timestamp
- Exit price
- P&L (highlighted green/red)
- Full reasoning (RSI, Z-score, trend, volatility)

## Tips for Analysis

### Finding Your Edge

1. **Best Day to Trade**
   - Look at "Win Rate by Day of Week"
   - Trade more on green days, avoid red days

2. **Identify Losing Patterns**
   - Click a red day in Daily P&L
   - Filter to "Losers"
   - Review what conditions led to losses
   - Example: "All losses had RSI < 25 in HIGH volatility"

3. **Validate ML Model**
   - If high-confidence trades (in trade_history.json) have better win rates
   - ML is working correctly

4. **Drawdown Analysis**
   - Check when max drawdown occurred
   - What market conditions caused it?
   - Did you recover? How long did it take?

### Performance Benchmarks

**Good Results:**
- Win Rate: 65-75%
- Profit Factor: >2.0
- Max Drawdown: <20%
- Profitable days: >50%

**Warning Signs:**
- Win Rate: <55%
- Profit Factor: <1.5
- Max Drawdown: >30%
- Many consecutive red days

## Troubleshooting

### "No Backtest Data Found"
**Solution:** Run `python example_usage.py` first

### Dashboard won't start
**Error:** `ModuleNotFoundError: No module named 'dash'`
**Solution:** `pip install -r requirements_dashboard.txt`

### "Empty trade table"
**Cause:** Filters are too restrictive
**Solution:** Reset all filters to "All"

### Date filter stuck
**Solution:** Click "Clear Filter" button or refresh page

### Dashboard is slow
**Cause:** Large number of trades (>1000)
**Solution:** This is normal, charts update after brief delay

## Advanced: Adding Candlestick Charts

The reference dashboard shows candlestick charts with trade entry/exit markers. To add this:

### Option 1: Store Candles During Backtest

Modify `ml_mean_reversion_bot.py`:

```python
# In backtest() function, store candles for each trade
trade_info = {
    ...existing fields...,
    'candles': df.iloc[i-20:i+10].to_dict('records')  # 20 before, 10 after
}
```

### Option 2: Fetch from Binance

Add to `dashboard_app.py`:

```python
from binance.client import Client

def fetch_trade_candles(entry_time, symbol='BTCUSDT', interval='15m'):
    """Fetch candles around trade time"""
    client = Client()  # Public, no API key needed

    start_time = entry_time - timedelta(hours=5)
    end_time = entry_time + timedelta(hours=2)

    klines = client.get_historical_klines(
        symbol, interval,
        start_time.strftime("%d %b %Y %H:%M:%S"),
        end_time.strftime("%d %b %Y %H:%M:%S")
    )

    # Convert to DataFrame and return
    ...
```

Then create candlestick chart:

```python
def create_trade_chart(trade, candles):
    """Create candlestick chart with trade markers"""

    fig = go.Figure()

    # Candlesticks
    fig.add_trace(go.Candlestick(
        x=candles['timestamp'],
        open=candles['open'],
        high=candles['high'],
        low=candles['low'],
        close=candles['close'],
        name='Price'
    ))

    # Entry marker
    fig.add_trace(go.Scatter(
        x=[trade['entry_time']],
        y=[trade['entry_price']],
        mode='markers+text',
        marker=dict(size=15, color='green' if trade['direction']=='LONG' else 'red',
                   symbol='triangle-up' if trade['direction']=='LONG' else 'triangle-down'),
        text=['ENTRY'],
        textposition='top center',
        name='Entry'
    ))

    # Exit marker
    # ... similar for exit

    # Add moving averages, support/resistance, etc.

    return fig
```

## Support

For issues or questions:
1. Check `GETTING_STARTED_GUIDE.md`
2. Review `STRATEGY_ANALYSIS.md`
3. Ensure backtest completed successfully

---

**Happy analyzing! 📊📈**
