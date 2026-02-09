from dash import html, dcc


def create_layout():
    return html.Div([
        html.H2("Stock Price Lookup"),

        # Input row
        html.Div([
            dcc.Input(
                id="lookup-ticker-input",
                type="text",
                placeholder="Enter ticker (e.g. AAPL)",
                className="ticker-input",
                debounce=True,
            ),
            dcc.Dropdown(
                id="lookup-period-dropdown",
                options=[
                    {"label": "1 Week", "value": "5d"},
                    {"label": "1 Month", "value": "1mo"},
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
            html.Button("Lookup", id="lookup-button", n_clicks=0, className="btn btn-primary"),
        ], className="input-row"),

        # Loading wrapper
        dcc.Loading(
            id="lookup-loading",
            type="circle",
            children=[
                # Info card
                html.Div(id="stock-info-card", className="info-card"),
                # Price chart
                dcc.Graph(id="price-chart", style={"display": "none"}),
            ],
        ),
    ], className="tab-content")
