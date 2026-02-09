from dash import html, dcc


def create_layout():
    return html.Div([
        html.H2("AI Forecast"),

        # Input row
        html.Div([
            dcc.Input(
                id="fc-ticker-input",
                type="text",
                placeholder="Tickers (e.g. AAPL, MSFT)",
                className="ticker-input",
                style={"width": "260px"},
                debounce=True,
            ),
            dcc.Dropdown(
                id="fc-horizon-dropdown",
                options=[
                    {"label": "1 Month", "value": "1M"},
                    {"label": "3 Months", "value": "3M"},
                    {"label": "6 Months", "value": "6M"},
                ],
                value="1M",
                clearable=False,
                className="period-dropdown",
            ),
            dcc.Dropdown(
                id="fc-model-dropdown",
                options=[
                    {"label": "LightGBM", "value": "lightgbm"},
                    {"label": "Elastic Net", "value": "elastic_net"},
                    {"label": "LSTM-Attention", "value": "lstm_attention"},
                    {"label": "Transformer", "value": "transformer"},
                ],
                value="lightgbm",
                clearable=False,
                className="period-dropdown",
            ),
            html.Button(
                "Run Forecast",
                id="fc-run-button",
                n_clicks=0,
                className="btn btn-primary",
            ),
        ], className="input-row"),

        # Loading wrapper for entire output
        dcc.Loading(
            id="fc-loading",
            type="circle",
            children=[
                # Regime card
                html.Div(id="fc-regime-card"),

                # Risk-off banner (hidden by default)
                html.Div(id="fc-risk-banner"),

                # Forecast result cards
                html.Div(id="fc-result-cards"),

                # Confidence chart
                dcc.Graph(id="fc-confidence-chart", style={"display": "none"}),

                # Position table
                html.Div(id="fc-position-table"),

                # Risk metrics
                html.Div(id="fc-risk-metrics"),
            ],
        ),
    ], className="tab-content")
