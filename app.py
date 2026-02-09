from dash import Dash, html, dcc, callback, Input, Output

from data.database import init_db
from layouts import price_lookup, portfolio, technical, forecast, model_performance

# Initialize database
init_db()

# Create Dash app
app = Dash(__name__, suppress_callback_exceptions=True)
app.title = "Stock Investment Tool"

app.layout = html.Div([
    html.H1("Stock Investment Tool", className="app-title"),
    dcc.Tabs(
        id="main-tabs",
        value="tab-lookup",
        children=[
            dcc.Tab(label="Stock Lookup", value="tab-lookup"),
            dcc.Tab(label="Portfolio", value="tab-portfolio"),
            dcc.Tab(label="Technical Analysis", value="tab-technical"),
            dcc.Tab(label="AI Forecast", value="tab-forecast"),
            dcc.Tab(label="Model Performance", value="tab-model-perf"),
        ],
        className="main-tabs",
    ),
    html.Div(id="tab-content"),
])


@callback(Output("tab-content", "children"), Input("main-tabs", "value"))
def render_tab(tab):
    if tab == "tab-lookup":
        return price_lookup.create_layout()
    elif tab == "tab-portfolio":
        return portfolio.create_layout()
    elif tab == "tab-technical":
        return technical.create_layout()
    elif tab == "tab-forecast":
        return forecast.create_layout()
    elif tab == "tab-model-perf":
        return model_performance.create_layout()


# Import callbacks to register them
import callbacks.price_callbacks
import callbacks.portfolio_callbacks
import callbacks.technical_callbacks
import callbacks.forecast_callbacks
import callbacks.model_performance_callbacks

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=8050)
