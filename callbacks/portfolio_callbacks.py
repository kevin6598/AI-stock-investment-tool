from dash import callback, Input, Output, State, html, no_update, ctx, dash_table, ALL
import json

from data.database import add_holding, delete_holding, get_all_holdings
from data.stock_api import get_current_price


@callback(
    Output("port-form-message", "children"),
    Output("portfolio-refresh-trigger", "data"),
    Input("port-add-button", "n_clicks"),
    State("port-ticker-input", "value"),
    State("port-shares-input", "value"),
    State("port-cost-input", "value"),
    State("port-date-input", "value"),
    prevent_initial_call=True,
)
def add_holding_callback(n_clicks, ticker, shares, cost, date):
    if not ticker or not shares or not cost:
        return html.Span("Please fill in ticker, shares, and cost.", className="error-msg"), no_update

    if shares <= 0 or cost <= 0:
        return html.Span("Shares and cost must be positive.", className="error-msg"), no_update

    try:
        add_holding(ticker, shares, cost, date)
        return (
            html.Span(f"Added {shares} shares of {ticker.upper()} at ${cost:.2f}", className="success-msg"),
            {"ts": n_clicks},
        )
    except Exception as e:
        return html.Span(f"Error: {e}", className="error-msg"), no_update


@callback(
    Output("portfolio-summary", "children"),
    Output("holdings-table-container", "children"),
    Input("main-tabs", "value"),
    Input("portfolio-refresh-trigger", "data"),
    Input({"type": "delete-btn", "index": ALL}, "n_clicks"),
)
def update_portfolio(tab, refresh_trigger, delete_clicks):
    if tab != "tab-portfolio":
        return no_update, no_update

    # Handle delete
    triggered = ctx.triggered_id
    if isinstance(triggered, dict) and triggered.get("type") == "delete-btn":
        holding_id = triggered["index"]
        delete_holding(holding_id)

    # Fetch holdings and current prices
    holdings = get_all_holdings()

    if not holdings:
        summary = html.Div([
            _summary_card("Total Value", "$0.00", ""),
            _summary_card("Total Cost", "$0.00", ""),
            _summary_card("Total P&L", "$0.00", "neutral"),
        ], className="summary-row")
        table = html.P("No holdings yet. Add a stock above to get started.", className="empty-msg")
        return summary, table

    total_value = 0
    total_cost = 0
    rows = []

    for h in holdings:
        current_price = get_current_price(h["ticker"])
        if current_price is None:
            current_price = h["avg_cost"]  # fallback

        market_value = current_price * h["shares"]
        cost_basis = h["avg_cost"] * h["shares"]
        pnl = market_value - cost_basis
        pnl_pct = (pnl / cost_basis * 100) if cost_basis != 0 else 0

        total_value += market_value
        total_cost += cost_basis

        pnl_class = "gain" if pnl >= 0 else "loss"

        rows.append(html.Tr([
            html.Td(h["ticker"], className="ticker-cell"),
            html.Td(f"{h['shares']:.2f}"),
            html.Td(f"${h['avg_cost']:.2f}"),
            html.Td(f"${current_price:.2f}"),
            html.Td(f"${market_value:.2f}"),
            html.Td(f"${pnl:+.2f} ({pnl_pct:+.1f}%)", className=pnl_class),
            html.Td(h["date_added"]),
            html.Td(html.Button(
                "Delete", id={"type": "delete-btn", "index": h["id"]},
                className="btn btn-danger btn-sm",
            )),
        ]))

    total_pnl = total_value - total_cost
    total_pnl_pct = (total_pnl / total_cost * 100) if total_cost != 0 else 0
    pnl_class = "gain" if total_pnl >= 0 else "loss"

    summary = html.Div([
        _summary_card("Total Value", f"${total_value:,.2f}", ""),
        _summary_card("Total Cost", f"${total_cost:,.2f}", ""),
        _summary_card("Total P&L", f"${total_pnl:+,.2f} ({total_pnl_pct:+.1f}%)", pnl_class),
    ], className="summary-row")

    table = html.Table([
        html.Thead(html.Tr([
            html.Th("Ticker"),
            html.Th("Shares"),
            html.Th("Avg Cost"),
            html.Th("Current Price"),
            html.Th("Market Value"),
            html.Th("P&L"),
            html.Th("Date Added"),
            html.Th("Actions"),
        ])),
        html.Tbody(rows),
    ], className="holdings-table")

    return summary, table


def _summary_card(title, value, css_class):
    return html.Div([
        html.Div(title, className="card-title"),
        html.Div(value, className=f"card-value {css_class}"),
    ], className="summary-card")
