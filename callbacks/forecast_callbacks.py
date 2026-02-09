"""Callbacks for the AI Forecast tab."""

from dash import callback, Input, Output, State, html, no_update
import plotly.graph_objects as go
import logging

logger = logging.getLogger(__name__)


def _build_regime_card(regime, confidence):
    """Build the regime indicator card."""
    color_map = {
        "strong_bull": "#2e7d32",
        "normal": "#1565c0",
        "bear": "#e65100",
        "crisis": "#b71c1c",
    }
    label_map = {
        "strong_bull": "Strong Bull",
        "normal": "Normal",
        "bear": "Bear",
        "crisis": "Crisis",
    }
    color = color_map.get(regime, "#666")
    label = label_map.get(regime, regime.title())

    return html.Div([
        html.Div([
            html.Span("Market Regime: ", style={"fontWeight": "600", "color": "#333"}),
            html.Span(
                label,
                className="regime-badge",
                style={
                    "backgroundColor": color,
                    "color": "white",
                    "padding": "4px 12px",
                    "borderRadius": "12px",
                    "fontWeight": "600",
                    "fontSize": "14px",
                },
            ),
            html.Span(
                f"  Confidence: {confidence:.0%}",
                style={"marginLeft": "12px", "color": "#666", "fontSize": "13px"},
            ),
        ]),
    ], className="regime-card")


def _build_result_cards(signals):
    """Build forecast result cards for each ticker."""
    cards = []
    for sig in signals:
        pct = sig.predicted_return * 100
        sign = "+" if pct >= 0 else ""
        color_cls = "positive" if pct >= 0 else "negative"
        ci_low = sig.confidence_interval[0] * 100
        ci_high = sig.confidence_interval[1] * 100

        cards.append(html.Div([
            html.Div(sig.ticker, style={
                "fontWeight": "700", "fontSize": "18px", "color": "#1a237e",
                "marginBottom": "8px",
            }),
            html.Div(f"{sign}{pct:.2f}%", className=f"forecast-value {color_cls}"),
            html.Div(
                f"CI: [{ci_low:+.2f}%, {ci_high:+.2f}%]",
                style={"fontSize": "12px", "color": "#666", "marginTop": "4px"},
            ),
            html.Div([
                html.Div(
                    style={
                        "width": f"{sig.confidence_score * 100:.0f}%",
                        "height": "4px",
                        "backgroundColor": "#2196F3",
                        "borderRadius": "2px",
                    },
                ),
            ], className="confidence-bar"),
            html.Div(
                f"Confidence: {sig.confidence_score:.0%}",
                style={"fontSize": "11px", "color": "#999", "marginTop": "2px"},
            ),
        ], className="forecast-card"))

    return html.Div(cards, className="forecast-cards")


def _build_confidence_chart(signals):
    """Build a bar chart showing predicted returns with confidence intervals."""
    tickers = [s.ticker for s in signals]
    returns_pct = [s.predicted_return * 100 for s in signals]
    ci_low = [(s.predicted_return - s.confidence_interval[0]) * 100 for s in signals]
    ci_high = [(s.confidence_interval[1] - s.predicted_return) * 100 for s in signals]
    colors = ["#2e7d32" if r >= 0 else "#d32f2f" for r in returns_pct]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=tickers,
        y=returns_pct,
        error_y=dict(
            type="data",
            symmetric=False,
            array=ci_high,
            arrayminus=ci_low,
            color="#999",
        ),
        marker_color=colors,
        text=[f"{r:+.2f}%" for r in returns_pct],
        textposition="outside",
    ))
    fig.update_layout(
        title="Predicted Returns with 95% Confidence Interval",
        yaxis_title="Predicted Return (%)",
        xaxis_title="Ticker",
        template="plotly_white",
        height=350,
        margin=dict(t=50, b=40, l=50, r=20),
    )
    return fig


def _build_position_table(signals):
    """Build position sizing table."""
    header = html.Tr([
        html.Th("Ticker"),
        html.Th("Predicted Return"),
        html.Th("Std Dev"),
        html.Th("Raw Weight"),
        html.Th("Risk-Adj Weight"),
        html.Th("CVaR Contrib"),
    ])
    rows = []
    for sig in signals:
        pct = sig.predicted_return * 100
        color_cls = "gain" if pct >= 0 else "loss"
        rows.append(html.Tr([
            html.Td(sig.ticker, className="ticker-cell"),
            html.Td(f"{pct:+.2f}%", className=color_cls),
            html.Td(f"{sig.return_std * 100:.2f}%"),
            html.Td(f"{sig.position_weight:.1%}"),
            html.Td(f"{sig.risk_adjusted_weight:.1%}"),
            html.Td(f"{sig.cvar_contribution * 100:.3f}%"),
        ]))

    return html.Table(
        [html.Thead(header), html.Tbody(rows)],
        className="holdings-table",
    )


def _build_risk_display(result):
    """Build risk metrics display row."""
    metrics = [
        ("Portfolio Vol", f"{result.portfolio_vol:.1%}"),
        ("CVaR (95%)", f"{result.portfolio_cvar_95:.2%}"),
        ("Confidence", f"{result.total_confidence:.0%}"),
        ("Regime", result.regime.replace("_", " ").title()),
    ]

    cards = []
    for label, value in metrics:
        cards.append(html.Div([
            html.Div(label, className="card-title"),
            html.Div(value, className="card-value"),
        ], className="risk-metric-card"))

    return html.Div(cards, className="risk-metrics-row")


@callback(
    Output("fc-regime-card", "children"),
    Output("fc-risk-banner", "children"),
    Output("fc-result-cards", "children"),
    Output("fc-confidence-chart", "figure"),
    Output("fc-confidence-chart", "style"),
    Output("fc-position-table", "children"),
    Output("fc-risk-metrics", "children"),
    Input("fc-run-button", "n_clicks"),
    State("fc-ticker-input", "value"),
    State("fc-horizon-dropdown", "value"),
    State("fc-model-dropdown", "value"),
    prevent_initial_call=True,
)
def run_forecast(n_clicks, ticker_input, horizon, model_type):
    """Main forecast callback: runs inference pipeline on button click."""
    if not n_clicks or not ticker_input:
        return no_update, no_update, no_update, no_update, no_update, no_update, no_update

    # Parse tickers
    tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
    if not tickers:
        error = html.Div("Please enter at least one ticker.", className="error-msg")
        return error, "", "", {}, {"display": "none"}, "", ""

    try:
        from engine.inference import InferencePipeline

        pipeline = InferencePipeline(
            tickers=tickers,
            model_type=model_type,
        )
        result = pipeline.run(horizon)

        # Build all UI components
        regime_card = _build_regime_card(result.regime, result.total_confidence)

        # Risk-off banner
        risk_banner = ""
        if result.risk_off_mode:
            risk_banner = html.Div(
                "RISK-OFF MODE ACTIVE -- Positions reduced",
                className="risk-off-banner",
            )

        result_cards = _build_result_cards(result.signals)
        conf_fig = _build_confidence_chart(result.signals)
        position_table = _build_position_table(result.signals)
        risk_display = _build_risk_display(result)

        return (
            regime_card,
            risk_banner,
            result_cards,
            conf_fig,
            {"display": "block"},
            position_table,
            risk_display,
        )

    except Exception as e:
        logger.error(f"Forecast failed: {e}", exc_info=True)
        error_msg = html.Div(
            f"Forecast error: {str(e)}", className="error-msg",
        )
        return error_msg, "", "", {}, {"display": "none"}, "", ""
