from dash import html, dcc


def create_layout():
    return html.Div([
        html.H2("Technical Analysis"),

        # Input row
        html.Div([
            dcc.Input(
                id="ta-ticker-input",
                type="text",
                placeholder="Enter ticker (e.g. AAPL)",
                className="ticker-input",
                debounce=True,
            ),
            dcc.Dropdown(
                id="ta-period-dropdown",
                options=[
                    {"label": "3 Months", "value": "3mo"},
                    {"label": "6 Months", "value": "6mo"},
                    {"label": "1 Year", "value": "1y"},
                    {"label": "2 Years", "value": "2y"},
                    {"label": "5 Years", "value": "5y"},
                ],
                value="1y",
                clearable=False,
                className="period-dropdown",
            ),
            html.Button("Analyze", id="ta-button", n_clicks=0, className="btn btn-primary"),
        ], className="input-row"),

        # Indicator checkboxes
        html.Div([
            html.Label("Indicators:", className="indicator-label"),
            dcc.Checklist(
                id="ta-indicators",
                options=[
                    {"label": " SMA (20, 50)", "value": "sma"},
                    {"label": " EMA (12, 26)", "value": "ema"},
                    {"label": " RSI (14)", "value": "rsi"},
                    {"label": " MACD (12, 26, 9)", "value": "macd"},
                ],
                value=["sma", "rsi"],
                inline=True,
                className="indicator-checklist",
            ),
        ], className="indicator-row"),

        # Chart
        dcc.Loading(
            id="ta-loading",
            type="circle",
            children=[
                html.Div(id="ta-error-message"),
                dcc.Graph(id="ta-chart", style={"display": "none"}),
            ],
        ),
    ], className="tab-content")
