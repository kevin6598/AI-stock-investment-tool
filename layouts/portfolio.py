from dash import html, dcc, dash_table


def create_layout():
    return html.Div([
        html.H2("Portfolio Management"),

        # Add holding form
        html.Div([
            html.H4("Add Holding"),
            html.Div([
                dcc.Input(
                    id="port-ticker-input",
                    type="text",
                    placeholder="Ticker",
                    className="ticker-input",
                ),
                dcc.Input(
                    id="port-shares-input",
                    type="number",
                    placeholder="Shares",
                    className="number-input",
                    min=0.01,
                    step=0.01,
                ),
                dcc.Input(
                    id="port-cost-input",
                    type="number",
                    placeholder="Avg Cost ($)",
                    className="number-input",
                    min=0.01,
                    step=0.01,
                ),
                dcc.Input(
                    id="port-date-input",
                    type="text",
                    placeholder="Date (YYYY-MM-DD)",
                    className="date-input",
                ),
                html.Button("Add", id="port-add-button", n_clicks=0, className="btn btn-primary"),
            ], className="input-row"),
            html.Div(id="port-form-message", className="form-message"),
        ], className="form-section"),

        # Summary cards
        dcc.Loading(
            type="circle",
            children=html.Div(id="portfolio-summary", className="summary-cards"),
        ),

        # Holdings table
        dcc.Loading(
            type="circle",
            children=html.Div(id="holdings-table-container"),
        ),

        # Hidden div for delete triggers
        dcc.Store(id="delete-trigger"),
        # Interval for refresh
        dcc.Store(id="portfolio-refresh-trigger"),
    ], className="tab-content")
