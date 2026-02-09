"""
Model Performance Tab Layout
------------------------------
Dash layout for model comparison, equity curves, rolling Sharpe,
drawdowns, weight allocation, and CVaR risk gauge.
"""

from dash import html, dcc


def create_layout():
    return html.Div([
        html.H2("Model Performance"),

        # Selector row
        html.Div([
            dcc.Dropdown(
                id="mp-model-dropdown",
                options=[],
                value=None,
                multi=True,
                placeholder="Select model versions...",
                className="mp-model-dropdown",
                style={"width": "400px"},
            ),
            dcc.Dropdown(
                id="mp-horizon-dropdown",
                options=[
                    {"label": "1 Month", "value": "1M"},
                    {"label": "3 Months", "value": "3M"},
                    {"label": "6 Months", "value": "6M"},
                ],
                value="1M",
                clearable=False,
                className="period-dropdown",
            ),
            html.Button(
                "Compare",
                id="mp-compare-button",
                n_clicks=0,
                className="btn btn-primary",
            ),
        ], className="input-row"),

        # Loading wrapper
        dcc.Loading(
            id="mp-loading",
            type="circle",
            children=[
                # Regime indicator
                html.Div(id="mp-regime-indicator", className="mp-regime-indicator"),

                # Comparison table
                html.Div(id="mp-comparison-table", className="mp-comparison-table"),

                # Charts row 1: Equity + Rolling Sharpe
                html.Div([
                    html.Div([
                        dcc.Graph(id="mp-equity-chart", style={"display": "none"}),
                    ], className="mp-chart-half"),
                    html.Div([
                        dcc.Graph(id="mp-sharpe-chart", style={"display": "none"}),
                    ], className="mp-chart-half"),
                ], className="mp-chart-row"),

                # Charts row 2: Drawdown + Weight Allocation
                html.Div([
                    html.Div([
                        dcc.Graph(id="mp-drawdown-chart", style={"display": "none"}),
                    ], className="mp-chart-half"),
                    html.Div([
                        dcc.Graph(id="mp-weights-chart", style={"display": "none"}),
                    ], className="mp-chart-half"),
                ], className="mp-chart-row"),

                # CVaR gauge
                html.Div([
                    dcc.Graph(id="mp-cvar-gauge", style={"display": "none"}),
                ], className="mp-gauge-container"),
            ],
        ),
    ], className="tab-content")
