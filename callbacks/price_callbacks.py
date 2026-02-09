from dash import callback, Input, Output, State, html, no_update
import plotly.graph_objects as go

from data.stock_api import get_stock_info, get_historical_data


def _format_number(n):
    """Format large numbers with K/M/B suffixes."""
    if n is None:
        return "N/A"
    if n >= 1_000_000_000:
        return f"${n / 1_000_000_000:.2f}B"
    if n >= 1_000_000:
        return f"${n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"${n / 1_000:.2f}K"
    return f"${n:,.2f}"


@callback(
    Output("stock-info-card", "children"),
    Output("price-chart", "figure"),
    Output("price-chart", "style"),
    Input("lookup-button", "n_clicks"),
    State("lookup-ticker-input", "value"),
    State("lookup-period-dropdown", "value"),
    prevent_initial_call=True,
)
def lookup_stock(n_clicks, ticker, period):
    if not ticker:
        return html.Div("Please enter a ticker symbol.", className="error-msg"), {}, {"display": "none"}

    ticker = ticker.strip().upper()
    info = get_stock_info(ticker)

    if not info:
        return html.Div(f"Could not find data for '{ticker}'.", className="error-msg"), {}, {"display": "none"}

    # Build info card
    pe_display = f"{info['pe_ratio']:.2f}" if info["pe_ratio"] else "N/A"
    div_display = f"{info['dividend_yield'] * 100:.2f}%" if info["dividend_yield"] else "N/A"
    w52 = f"${info['week_52_low']:.2f} - ${info['week_52_high']:.2f}" if info["week_52_low"] else "N/A"

    info_card = html.Div([
        html.Div([
            html.H3(f"{info['name']} ({info['ticker']})"),
            html.Span(f"${info['price']:.2f}", className="current-price"),
        ], className="info-header"),
        html.Div([
            html.Div([html.Span("Sector"), html.Span(info["sector"])], className="info-item"),
            html.Div([html.Span("Market Cap"), html.Span(_format_number(info["market_cap"]))], className="info-item"),
            html.Div([html.Span("P/E Ratio"), html.Span(pe_display)], className="info-item"),
            html.Div([html.Span("52-Week Range"), html.Span(w52)], className="info-item"),
            html.Div([html.Span("Volume"), html.Span(f"{info['volume']:,}")], className="info-item"),
            html.Div([html.Span("Dividend Yield"), html.Span(div_display)], className="info-item"),
        ], className="info-grid"),
    ], className="stock-card")

    # Build price chart
    df = get_historical_data(ticker, period=period)
    if df.empty:
        return info_card, {}, {"display": "none"}

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df["Close"],
        mode="lines",
        name="Close",
        line=dict(color="#2196F3", width=2),
        fill="tozeroy",
        fillcolor="rgba(33, 150, 243, 0.1)",
    ))
    fig.update_layout(
        title=f"{ticker} - Price History ({period})",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        template="plotly_white",
        height=450,
        margin=dict(l=50, r=30, t=50, b=50),
        hovermode="x unified",
    )

    return info_card, fig, {"display": "block"}
