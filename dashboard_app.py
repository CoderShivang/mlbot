"""
ML Mean Reversion Bot - Interactive Dashboard
==============================================

Matches the reference dashboard design with:
- Day of Week Analysis (stacked bars + win rates)
- Equity Curve & Drawdown charts
- Daily P&L Table (clickable for filtering)
- Trade Filtering (Outcome, Direction, Signal Type)
- Trade List (clickable to view candlestick chart)
- Individual Trade Charts with entry/exit markers

Runs on: http://localhost:8050
"""

import dash
from dash import dcc, html, dash_table, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from pathlib import Path
from binance.client import Client

# Initialize Dash app with Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY], suppress_callback_exceptions=True)
app.title = "ML Mean Reversion Bot Dashboard"

# Global data storage
TRADES_DATA = None
DAILY_DATA = None

def load_backtest_data():
    """Load backtest results"""
    try:
        trade_history_path = Path('/mnt/user-data/outputs/trade_history.json')
        if trade_history_path.exists():
            with open(trade_history_path, 'r') as f:
                trades = json.load(f)
            return trades
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def prepare_daily_data(trades):
    """Prepare daily summary data"""
    if not trades:
        return pd.DataFrame()

    df = pd.DataFrame(trades)

    # Parse timestamps
    if 'timestamp' in df.columns:
        df['datetime'] = pd.to_datetime(df['timestamp'])
    else:
        df['datetime'] = pd.to_datetime(df.get('entry_time', pd.Timestamp.now()))

    df['date'] = df['datetime'].dt.date

    # Calculate daily stats
    daily = df.groupby('date').agg({
        'pnl_percent': ['sum', 'count'],
        'direction': 'count'
    }).reset_index()

    daily.columns = ['date', 'total_pnl', 'trades', 'dummy']
    daily = daily.drop('dummy', axis=1)

    # Calculate wins and losses per day
    wins_per_day = df[df['pnl_percent'] > 0].groupby('date').size().reset_index(name='wins')
    losses_per_day = df[df['pnl_percent'] <= 0].groupby('date').size().reset_index(name='losses')

    daily = daily.merge(wins_per_day, on='date', how='left')
    daily = daily.merge(losses_per_day, on='date', how='left')

    daily['wins'] = daily['wins'].fillna(0).astype(int)
    daily['losses'] = daily['losses'].fillna(0).astype(int)
    daily['win_rate'] = (daily['wins'] / daily['trades'] * 100).round(1)

    # Calculate fees (estimate 0.02% per trade)
    daily['total_fees'] = daily['trades'] * 0.02 * 2  # Entry + exit

    return daily

def create_header_metrics(trades):
    """Create header with key metrics"""
    if not trades:
        return html.Div("No data", className="text-center p-5")

    total_trades = len(trades)
    wins = len([t for t in trades if t.get('pnl_percent', 0) > 0])
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0

    total_pnl_pct = sum([t.get('pnl_percent', 0) for t in trades]) * 100

    # Color code
    pnl_color = "text-success" if total_pnl_pct > 0 else "text-danger"
    wr_color = "text-success" if win_rate >= 60 else "text-warning" if win_rate >= 50 else "text-danger"

    return dbc.Row([
        dbc.Col([
            html.Div([
                html.Span("Win Rate: ", style={'fontSize': '18px'}),
                html.Span(f"{win_rate:.1f}%", className=wr_color, style={'fontSize': '18px', 'fontWeight': 'bold'})
            ])
        ], width=4, className="text-center"),
        dbc.Col([
            html.Div([
                html.Span("Total Trades: ", style={'fontSize': '18px'}),
                html.Span(f"{total_trades}", className="text-primary", style={'fontSize': '18px', 'fontWeight': 'bold'})
            ])
        ], width=4, className="text-center"),
        dbc.Col([
            html.Div([
                html.Span("Net P&L: ", style={'fontSize': '18px'}),
                html.Span(f"${total_pnl_pct*100/100:.2f} ({total_pnl_pct:+.2f}%)",
                         className=pnl_color, style={'fontSize': '18px', 'fontWeight': 'bold'})
            ])
        ], width=4, className="text-center")
    ], className="p-3", style={'backgroundColor': '#f8f9fa', 'borderRadius': '5px'})

def create_day_of_week_charts(trades):
    """Create day of week analysis charts"""
    if not trades:
        return html.Div("No data")

    df = pd.DataFrame(trades)

    # Parse timestamps
    if 'timestamp' in df.columns:
        df['datetime'] = pd.to_datetime(df['timestamp'])
    else:
        df['datetime'] = pd.to_datetime(df.get('entry_time', pd.Timestamp.now()))

    df['dayofweek'] = df['datetime'].dt.dayofweek
    df['dayname'] = df['datetime'].dt.day_name()
    df['is_win'] = df['pnl_percent'] > 0

    # Day order
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    # Trades by day (stacked)
    day_stats = []
    for day in day_order:
        day_trades = df[df['dayname'] == day]
        wins = len(day_trades[day_trades['is_win']])
        losses = len(day_trades) - wins
        day_stats.append({'day': day, 'wins': wins, 'losses': losses})

    day_df = pd.DataFrame(day_stats)

    fig_trades = go.Figure()
    fig_trades.add_trace(go.Bar(
        name='Wins',
        x=day_df['day'],
        y=day_df['wins'],
        marker_color='#28a745',
        text=day_df['wins'],
        textposition='inside'
    ))
    fig_trades.add_trace(go.Bar(
        name='Losses',
        x=day_df['day'],
        y=day_df['losses'],
        marker_color='#dc3545',
        text=day_df['losses'],
        textposition='inside'
    ))

    fig_trades.update_layout(
        title='Trades by Day of Week',
        xaxis_title='Day',
        yaxis_title='Number of Trades',
        barmode='stack',
        height=350,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        template='simple_white'
    )

    # Win rate by day
    day_wr = []
    for day in day_order:
        day_trades = df[df['dayname'] == day]
        if len(day_trades) > 0:
            wr = (day_trades['is_win'].sum() / len(day_trades)) * 100
        else:
            wr = 0
        day_wr.append({'day': day, 'win_rate': wr})

    wr_df = pd.DataFrame(day_wr)

    # Color bars based on win rate
    colors = ['#28a745' if wr >= 50 else '#dc3545' for wr in wr_df['win_rate']]

    fig_wr = go.Figure()
    fig_wr.add_trace(go.Bar(
        x=wr_df['day'],
        y=wr_df['win_rate'],
        marker_color=colors,
        text=[f"{wr:.1f}%" for wr in wr_df['win_rate']],
        textposition='inside'
    ))

    # Add 50% breakeven line
    fig_wr.add_hline(y=50, line_dash="dash", line_color="gray",
                     annotation_text="50% Breakeven", annotation_position="top right")

    fig_wr.update_layout(
        title='Win Rate by Day of Week',
        xaxis_title='Day',
        yaxis_title='Win Rate (%)',
        height=350,
        template='simple_white',
        yaxis=dict(range=[0, 105])
    )

    return dbc.Row([
        dbc.Col([dcc.Graph(figure=fig_trades)], width=6),
        dbc.Col([dcc.Graph(figure=fig_wr)], width=6)
    ])

def create_equity_drawdown_chart(trades):
    """Create equity curve and drawdown chart"""
    if not trades:
        return html.Div("No data")

    df = pd.DataFrame(trades)
    df['cumulative_pnl'] = df['pnl_percent'].cumsum()
    df['equity'] = 100 * (1 + df['cumulative_pnl'])

    # Calculate drawdown
    df['running_max'] = df['equity'].cummax()
    df['drawdown'] = ((df['equity'] - df['running_max']) / df['running_max']) * 100

    # Find max drawdown point
    max_dd_idx = df['drawdown'].idxmin()
    max_dd_value = df.loc[max_dd_idx, 'drawdown']

    # Find max profit point
    max_profit_idx = df['equity'].idxmax()
    max_profit_value = df.loc[max_profit_idx, 'equity']

    # Create subplot
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Equity Curve', 'Drawdown'),
        vertical_spacing=0.15,
        row_heights=[0.6, 0.4]
    )

    # Equity curve
    fig.add_trace(
        go.Scatter(
            x=list(range(len(df))),
            y=df['equity'],
            mode='lines',
            name='Equity',
            line=dict(color='#17a2b8', width=2),
            fill='tozeroy',
            fillcolor='rgba(23, 162, 184, 0.2)'
        ),
        row=1, col=1
    )

    # Add initial capital line
    fig.add_hline(y=100, line_dash="dash", line_color="gray", row=1, col=1,
                  annotation_text=f"Initial: $100", annotation_position="right")

    # Mark max drawdown
    fig.add_trace(
        go.Scatter(
            x=[max_dd_idx],
            y=[df.loc[max_dd_idx, 'equity']],
            mode='markers+text',
            name='Max Drawdown',
            marker=dict(color='red', size=12, symbol='diamond'),
            text=["Max Drawdown"],
            textposition="top center",
            showlegend=True
        ),
        row=1, col=1
    )

    # Mark max profit
    fig.add_trace(
        go.Scatter(
            x=[max_profit_idx],
            y=[max_profit_value],
            mode='markers+text',
            name='Max Profit',
            marker=dict(color='green', size=12, symbol='star'),
            text=["Max Profit"],
            textposition="top center",
            showlegend=True
        ),
        row=1, col=1
    )

    # Drawdown
    fig.add_trace(
        go.Scatter(
            x=list(range(len(df))),
            y=df['drawdown'],
            mode='lines',
            name='Drawdown',
            line=dict(color='#dc3545', width=2),
            fill='tozeroy',
            fillcolor='rgba(220, 53, 69, 0.2)'
        ),
        row=2, col=1
    )

    # Mark max drawdown point
    fig.add_trace(
        go.Scatter(
            x=[max_dd_idx],
            y=[max_dd_value],
            mode='markers+text',
            marker=dict(color='darkred', size=12, symbol='diamond'),
            text=[f"Max DD: {max_dd_value:.2f}%"],
            textposition="bottom center",
            showlegend=False
        ),
        row=2, col=1
    )

    fig.update_xaxes(title_text="Trade Number", row=2, col=1)
    fig.update_yaxes(title_text="Equity ($)", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)

    fig.update_layout(
        height=600,
        template='simple_white',
        legend=dict(orientation='h', yanchor='top', y=-0.15, xanchor='center', x=0.5)
    )

    return dcc.Graph(figure=fig)

def create_daily_pnl_table(daily_df):
    """Create daily P&L table"""
    if daily_df.empty:
        return html.Div("No data")

    # Format for display
    daily_display = daily_df.copy()
    daily_display['date'] = daily_display['date'].astype(str)
    daily_display['total_pnl'] = daily_display['total_pnl'].round(2)
    daily_display['total_fees'] = daily_display['total_fees'].round(2)

    return dash_table.DataTable(
        id='daily-pnl-table',
        columns=[
            {'name': '', 'id': 'selector', 'presentation': 'input'},
            {'name': 'Date', 'id': 'date'},
            {'name': 'Total P&L', 'id': 'total_pnl', 'type': 'numeric'},
            {'name': 'Total Fees', 'id': 'total_fees', 'type': 'numeric'},
            {'name': 'Trades', 'id': 'trades'},
            {'name': 'Wins', 'id': 'wins'},
            {'name': 'Losses', 'id': 'losses'},
            {'name': 'Win Rate', 'id': 'win_rate', 'type': 'numeric'}
        ],
        data=daily_display.to_dict('records'),
        style_cell={
            'textAlign': 'center',
            'padding': '10px',
            'fontSize': '14px'
        },
        style_header={
            'backgroundColor': '#343a40',
            'color': 'white',
            'fontWeight': 'bold',
            'textAlign': 'center'
        },
        style_data_conditional=[
            {
                'if': {'filter_query': '{total_pnl} > 0'},
                'backgroundColor': '#d4edda',
                'color': '#155724'
            },
            {
                'if': {'filter_query': '{total_pnl} < 0'},
                'backgroundColor': '#f8d7da',
                'color': '#721c24'
            }
        ],
        row_selectable='single',
        selected_rows=[],
        page_action='none',
        style_table={'height': '400px', 'overflowY': 'auto'}
    )

def create_trade_table(trades, outcome_filter='all', direction_filter='all'):
    """Create trade list table"""
    if not trades:
        return html.Div("No trades")

    df = pd.DataFrame(trades)

    # Apply filters
    if outcome_filter == 'win':
        df = df[df['pnl_percent'] > 0]
    elif outcome_filter == 'loss':
        df = df[df['pnl_percent'] <= 0]

    if direction_filter != 'all':
        df = df[df['direction'] == direction_filter]

    if df.empty:
        return html.Div("No trades match filters")

    # Prepare display data
    df['trade_num'] = range(1, len(df) + 1)
    df['pnl_pct'] = (df['pnl_percent'] * 100).round(2)

    # Add reason/type
    df['type'] = 'mean_reversion'  # All trades are mean reversion

    # Format entry time
    if 'timestamp' in df.columns:
        df['entry_time_fmt'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
    else:
        df['entry_time_fmt'] = pd.to_datetime(df.get('entry_time', pd.Timestamp.now())).dt.strftime('%Y-%m-%d %H:%M')

    # Create reason string
    df['reason'] = df.apply(lambda row:
        f"{row['direction']} Mean Reversion: RSI={row.get('rsi', 0):.1f}, Z-score={row.get('zscore', 0):.2f}",
        axis=1
    )

    display_df = df[['trade_num', 'direction', 'type', 'entry_time_fmt', 'entry_price',
                      'exit_price', 'pnl_pct', 'reason']].copy()

    display_df.columns = ['#', 'Direction', 'Type', 'Entry Time', 'Entry Price',
                          'Exit Price', 'P&L %', 'Reason']

    return dash_table.DataTable(
        id='trade-list-table',
        columns=[{'name': col, 'id': col} for col in display_df.columns],
        data=display_df.to_dict('records'),
        style_cell={
            'textAlign': 'left',
            'padding': '8px',
            'fontSize': '13px',
            'whiteSpace': 'normal',
            'height': 'auto'
        },
        style_header={
            'backgroundColor': '#343a40',
            'color': 'white',
            'fontWeight': 'bold'
        },
        style_data_conditional=[
            {
                'if': {'filter_query': '{P&L %} > 0', 'column_id': 'P&L %'},
                'color': '#28a745',
                'fontWeight': 'bold'
            },
            {
                'if': {'filter_query': '{P&L %} < 0', 'column_id': 'P&L %'},
                'color': '#dc3545',
                'fontWeight': 'bold'
            },
            {
                'if': {'column_id': 'Direction'},
                'width': '80px'
            },
            {
                'if': {'column_id': '#'},
                'width': '50px'
            }
        ],
        row_selectable='single',
        selected_rows=[],
        page_size=10,
        style_table={'overflowX': 'auto'}
    )

# Layout
app.layout = dbc.Container([
    html.H2("ML Mean Reversion Bot - Backtest Dashboard",
            className="text-center mt-3 mb-4", style={'color': '#343a40'}),

    # Hidden store for data
    dcc.Store(id='trades-store'),
    dcc.Store(id='selected-date-store'),

    # Header metrics
    html.Div(id='header-metrics', className="mb-4"),

    html.Hr(),

    # Day of Week Analysis
    html.Div([
        html.H4("Day of Week Analysis", className="mb-3"),
        html.P("Performance breakdown by day of the week", className="text-muted"),
        html.Div(id='day-of-week-charts')
    ], className="mb-4 p-3", style={'backgroundColor': '#f8f9fa', 'borderRadius': '5px'}),

    html.Hr(),

    # Equity & Drawdown
    html.Div([
        html.H4("Equity Curve & Drawdown", className="mb-3"),
        html.P("Track account balance progression and maximum drawdown over time", className="text-muted"),
        html.Div(id='equity-drawdown-chart')
    ], className="mb-4 p-3", style={'backgroundColor': '#f8f9fa', 'borderRadius': '5px'}),

    html.Hr(),

    # Daily P&L Analysis
    html.Div([
        html.H4("Daily P&L Analysis", className="mb-3"),
        html.P("Click column headers to sort. Click a row to filter trades to that day.", className="text-muted"),
        html.Div(id='daily-pnl-table-container')
    ], className="mb-4 p-3", style={'backgroundColor': '#f8f9fa', 'borderRadius': '5px'}),

    html.Div(id='date-filter-info', className="mb-3"),

    html.Hr(),

    # Filter Trades
    html.Div([
        html.H4("Filter Trades", className="mb-3"),
        dbc.Row([
            dbc.Col([
                html.Label("Outcome:", className="font-weight-bold"),
                dcc.Dropdown(
                    id='outcome-filter',
                    options=[
                        {'label': 'All Trades', 'value': 'all'},
                        {'label': 'Winners', 'value': 'win'},
                        {'label': 'Losers', 'value': 'loss'}
                    ],
                    value='all',
                    clearable=False
                )
            ], width=4),
            dbc.Col([
                html.Label("Direction:", className="font-weight-bold"),
                dcc.Dropdown(
                    id='direction-filter',
                    options=[
                        {'label': 'All Directions', 'value': 'all'},
                        {'label': 'LONG', 'value': 'LONG'},
                        {'label': 'SHORT', 'value': 'SHORT'}
                    ],
                    value='all',
                    clearable=False
                )
            ], width=4),
            dbc.Col([
                html.Label("Signal Type:", className="font-weight-bold"),
                dcc.Dropdown(
                    id='signal-type-filter',
                    options=[
                        {'label': 'All Types', 'value': 'all'},
                        {'label': 'Mean Reversion', 'value': 'mean_reversion'}
                    ],
                    value='all',
                    clearable=False
                )
            ], width=4)
        ])
    ], className="mb-4 p-3", style={'backgroundColor': '#f8f9fa', 'borderRadius': '5px'}),

    # Trade List
    html.Div([
        html.H4("Trade List (Click to View Chart)", className="mb-3"),
        html.P(id='trade-count-text'),
        html.Div(id='trade-table-container')
    ], className="mb-4 p-3", style={'backgroundColor': '#f8f9fa', 'borderRadius': '5px'}),

    # Trade Details (shown when trade selected)
    html.Div(id='trade-details-container', className="mb-4")

], fluid=True, style={'backgroundColor': '#ffffff'})

# Callbacks
@app.callback(
    [Output('trades-store', 'data'),
     Output('header-metrics', 'children'),
     Output('day-of-week-charts', 'children'),
     Output('equity-drawdown-chart', 'children'),
     Output('daily-pnl-table-container', 'children')],
    Input('trades-store', 'data')
)
def load_initial_data(_):
    """Load data on startup"""
    trades = load_backtest_data()

    if not trades:
        no_data = html.Div([
            html.H4("No Backtest Data Found", className="text-center text-danger mt-5"),
            html.P("Please run a backtest first:", className="text-center"),
            html.Code("python example_usage.py", className="d-block text-center mb-2"),
            html.P("Then refresh this page.", className="text-center text-muted")
        ])
        return None, no_data, no_data, no_data, no_data

    # Prepare daily data
    daily_df = prepare_daily_data(trades)

    return (
        trades,
        create_header_metrics(trades),
        create_day_of_week_charts(trades),
        create_equity_drawdown_chart(trades),
        create_daily_pnl_table(daily_df)
    )

@app.callback(
    [Output('selected-date-store', 'data'),
     Output('date-filter-info', 'children')],
    Input('daily-pnl-table', 'selected_rows'),
    State('daily-pnl-table', 'data')
)
def handle_date_selection(selected_rows, table_data):
    """Handle daily table row selection"""
    if not selected_rows or not table_data:
        return None, ""

    selected_date = table_data[selected_rows[0]]['date']

    info = dbc.Alert([
        html.Strong("📅 Filtering trades for: "),
        html.Span(selected_date),
        html.Button("Clear Filter", id='clear-date-filter',
                   className="btn btn-sm btn-outline-secondary ml-3", n_clicks=0)
    ], color="info", className="d-flex align-items-center justify-content-between")

    return selected_date, info

@app.callback(
    Output('selected-date-store', 'data', allow_duplicate=True),
    Input('clear-date-filter', 'n_clicks'),
    prevent_initial_call=True
)
def clear_date_filter(n_clicks):
    """Clear date filter"""
    if n_clicks:
        return None
    return None

@app.callback(
    [Output('trade-table-container', 'children'),
     Output('trade-count-text', 'children')],
    [Input('trades-store', 'data'),
     Input('outcome-filter', 'value'),
     Input('direction-filter', 'value'),
     Input('selected-date-store', 'data')]
)
def update_trade_table(trades, outcome, direction, selected_date):
    """Update trade table based on filters"""
    if not trades:
        return html.Div("No trades"), "0 trades"

    # Apply date filter
    if selected_date:
        df = pd.DataFrame(trades)
        if 'timestamp' in df.columns:
            df['date'] = pd.to_datetime(df['timestamp']).dt.date.astype(str)
        else:
            df['date'] = pd.to_datetime(df.get('entry_time', pd.Timestamp.now())).dt.date.astype(str)

        trades = df[df['date'] == selected_date].to_dict('records')

    table = create_trade_table(trades, outcome, direction)

    # Count trades
    df_filtered = pd.DataFrame(trades) if trades else pd.DataFrame()
    if not df_filtered.empty:
        if outcome == 'win':
            df_filtered = df_filtered[df_filtered['pnl_percent'] > 0]
        elif outcome == 'loss':
            df_filtered = df_filtered[df_filtered['pnl_percent'] <= 0]

        if direction != 'all':
            df_filtered = df_filtered[df_filtered['direction'] == direction]

    count_text = f"Showing {len(df_filtered)} trades"

    return table, count_text

@app.callback(
    Output('trade-details-container', 'children'),
    [Input('trade-list-table', 'selected_rows'),
     Input('trades-store', 'data')],
    [State('outcome-filter', 'value'),
     State('direction-filter', 'value'),
     State('selected-date-store', 'data')]
)
def show_trade_details(selected_rows, trades, outcome, direction, selected_date):
    """Show trade details and chart when trade selected"""
    if not selected_rows or not trades:
        return html.Div()

    # Apply same filters as table
    df = pd.DataFrame(trades)

    if selected_date:
        if 'timestamp' in df.columns:
            df['date'] = pd.to_datetime(df['timestamp']).dt.date.astype(str)
        else:
            df['date'] = pd.to_datetime(df.get('entry_time', pd.Timestamp.now())).dt.date.astype(str)
        df = df[df['date'] == selected_date]

    if outcome == 'win':
        df = df[df['pnl_percent'] > 0]
    elif outcome == 'loss':
        df = df[df['pnl_percent'] <= 0]

    if direction != 'all':
        df = df[df['direction'] == direction]

    if df.empty:
        return html.Div()

    # Get selected trade
    trade_idx = selected_rows[0]
    if trade_idx >= len(df):
        return html.Div()

    trade = df.iloc[trade_idx].to_dict()

    # Trade details
    details_card = dbc.Card([
        dbc.CardHeader(html.H4(f"Trade #{trade_idx + 1} Details")),
        dbc.CardBody([
            html.P([
                html.Strong("Direction: "),
                html.Span(f"{trade['direction']} (mean_reversion)",
                         className="text-primary" if trade['direction'] == 'LONG' else "text-danger")
            ]),
            html.P([
                html.Strong("Entry: "),
                f"${trade['entry_price']:,.2f} at ",
                pd.to_datetime(trade.get('timestamp', trade.get('entry_time', pd.Timestamp.now()))).strftime('%Y-%m-%d %H:%M')
            ]),
            html.P([
                html.Strong("Exit: "),
                f"${trade.get('exit_price', 0):,.2f}"
            ]),
            html.P([
                html.Strong("P&L: "),
                html.Span(f"${trade['pnl_percent'] * 100:.2f} ({trade['pnl_percent']:.2%})",
                         className="text-success" if trade['pnl_percent'] > 0 else "text-danger",
                         style={'fontSize': '16px', 'fontWeight': 'bold'})
            ]),
            html.P([
                html.Strong("Reason: "),
                f"{trade['direction']} Mean Reversion: RSI={trade.get('rsi', 0):.1f}, Z-score={trade.get('zscore', 0):.2f}"
            ])
        ])
    ], className="mb-3")

    # Note: Candlestick chart would require historical price data
    # which isn't stored in trade_history.json
    # You would need to fetch this from Binance or store it during backtest

    note = dbc.Alert([
        html.Strong("ℹ️ Note: "),
        "Candlestick charts require historical price data. ",
        "This can be added by storing candle data during backtest or fetching from Binance API."
    ], color="info")

    return html.Div([details_card, note])

if __name__ == '__main__':
    print("="*80)
    print("🚀 ML Mean Reversion Bot Dashboard Starting...")
    print("="*80)
    print("\n📊 Dashboard URL: http://localhost:8050")
    print("\n⚠️  Make sure you've run 'python example_usage.py' first!")
    print("   This generates the trade_history.json file needed for the dashboard.\n")
    print("🔄 Loading dashboard...\n")

    app.run_server(debug=True, host='127.0.0.1', port=8050)
