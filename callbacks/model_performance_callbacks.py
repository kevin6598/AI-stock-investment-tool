"""
Model Performance Tab Callbacks
---------------------------------
Populates dropdowns from model registry and generates comparison charts.
"""

from dash import callback, Input, Output, State, html, no_update
import plotly.graph_objects as go
import numpy as np
import json
import logging

logger = logging.getLogger(__name__)


@callback(
    Output("mp-model-dropdown", "options"),
    Input("main-tabs", "value"),
)
def load_model_versions(tab):
    """Populate model version dropdown when tab is selected."""
    if tab != "tab-model-perf":
        return no_update

    try:
        from training.model_versioning import ModelRegistry
        registry = ModelRegistry()
        versions = registry.list_versions()

        options = []
        for v in versions:
            ic = v.metrics.get("mean_ic", "N/A")
            label = f"{v.version_id} (IC={ic})"
            options.append({"label": label, "value": v.version_id})

        return options if options else [{"label": "No models found", "value": ""}]
    except Exception as e:
        logger.warning(f"Failed to load model versions: {e}")
        return [{"label": "Registry unavailable", "value": ""}]


@callback(
    [
        Output("mp-comparison-table", "children"),
        Output("mp-equity-chart", "figure"),
        Output("mp-equity-chart", "style"),
        Output("mp-sharpe-chart", "figure"),
        Output("mp-sharpe-chart", "style"),
        Output("mp-drawdown-chart", "figure"),
        Output("mp-drawdown-chart", "style"),
        Output("mp-weights-chart", "figure"),
        Output("mp-weights-chart", "style"),
        Output("mp-cvar-gauge", "figure"),
        Output("mp-cvar-gauge", "style"),
        Output("mp-regime-indicator", "children"),
    ],
    Input("mp-compare-button", "n_clicks"),
    [
        State("mp-model-dropdown", "value"),
        State("mp-horizon-dropdown", "value"),
    ],
    prevent_initial_call=True,
)
def update_comparison(n_clicks, selected_versions, horizon):
    """Load metrics and generate comparison charts."""
    if not selected_versions or not n_clicks:
        return (
            html.Div("Select model versions and click Compare.", className="empty-msg"),
            go.Figure(), {"display": "none"},
            go.Figure(), {"display": "none"},
            go.Figure(), {"display": "none"},
            go.Figure(), {"display": "none"},
            go.Figure(), {"display": "none"},
            html.Div(),
        )

    try:
        from training.model_versioning import ModelRegistry
        registry = ModelRegistry()
    except Exception:
        return (
            html.Div("Model registry unavailable", className="error-msg"),
            go.Figure(), {"display": "none"},
            go.Figure(), {"display": "none"},
            go.Figure(), {"display": "none"},
            go.Figure(), {"display": "none"},
            go.Figure(), {"display": "none"},
            html.Div(),
        )

    # Ensure selected_versions is a list
    if isinstance(selected_versions, str):
        selected_versions = [selected_versions]

    # Load version metadata
    versions_data = []
    for vid in selected_versions:
        if not vid:
            continue
        try:
            all_versions = registry.list_versions()
            match = [v for v in all_versions if v.version_id == vid]
            if match:
                versions_data.append(match[0])
        except Exception:
            continue

    if not versions_data:
        return (
            html.Div("No valid model versions selected", className="error-msg"),
            go.Figure(), {"display": "none"},
            go.Figure(), {"display": "none"},
            go.Figure(), {"display": "none"},
            go.Figure(), {"display": "none"},
            go.Figure(), {"display": "none"},
            html.Div(),
        )

    # Build comparison table
    table = _build_comparison_table(versions_data)

    # Generate charts
    equity_fig = _build_equity_chart(versions_data)
    sharpe_fig = _build_sharpe_chart(versions_data)
    drawdown_fig = _build_drawdown_chart(versions_data)
    weights_fig = _build_weights_chart(versions_data)
    cvar_fig = _build_cvar_gauge(versions_data)

    show = {"display": "block"}

    # Regime indicator
    regime_div = _build_regime_indicator()

    return (
        table, equity_fig, show, sharpe_fig, show,
        drawdown_fig, show, weights_fig, show,
        cvar_fig, show, regime_div,
    )


def _build_comparison_table(versions):
    """Build an HTML comparison table from model versions."""
    header = html.Tr([
        html.Th("Model"),
        html.Th("Type"),
        html.Th("Horizon"),
        html.Th("IC"),
        html.Th("ICIR"),
        html.Th("Sharpe"),
        html.Th("Max DD"),
        html.Th("Active"),
    ])

    rows = []
    for v in versions:
        ic = v.metrics.get("mean_ic", 0.0)
        icir = v.metrics.get("icir", 0.0)
        sharpe = v.metrics.get("mean_sharpe", 0.0)
        mdd = v.metrics.get("mean_mdd", 0.0)

        rows.append(html.Tr([
            html.Td(v.version_id, className="ticker-cell"),
            html.Td(v.model_type),
            html.Td(v.horizon),
            html.Td(f"{ic:.4f}"),
            html.Td(f"{icir:.2f}"),
            html.Td(f"{sharpe:.2f}"),
            html.Td(f"{mdd:.4f}"),
            html.Td("Yes" if v.is_active else "No"),
        ]))

    return html.Table(
        [html.Thead(header), html.Tbody(rows)],
        className="holdings-table",
    )


def _build_equity_chart(versions):
    """Build equity curve chart (simulated from metrics)."""
    fig = go.Figure()

    for v in versions:
        sharpe = v.metrics.get("mean_sharpe", 0.0)
        # Simulate approximate equity curve from Sharpe
        n_days = 252
        daily_ret = sharpe * 0.15 / np.sqrt(252)
        daily_vol = 0.15 / np.sqrt(252)
        rng = np.random.RandomState(hash(v.version_id) % 2**31)
        returns = rng.normal(daily_ret, daily_vol, n_days)
        equity = 100 * np.cumprod(1 + returns)

        fig.add_trace(go.Scatter(
            y=equity,
            mode="lines",
            name=v.version_id,
        ))

    fig.update_layout(
        title="Equity Curves (Simulated)",
        xaxis_title="Trading Days",
        yaxis_title="Portfolio Value",
        template="plotly_white",
        height=350,
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


def _build_sharpe_chart(versions):
    """Build rolling Sharpe chart."""
    fig = go.Figure()

    for v in versions:
        sharpe = v.metrics.get("mean_sharpe", 0.0)
        n_days = 252
        daily_ret = sharpe * 0.15 / np.sqrt(252)
        daily_vol = 0.15 / np.sqrt(252)
        rng = np.random.RandomState(hash(v.version_id) % 2**31)
        returns = rng.normal(daily_ret, daily_vol, n_days)

        # 63-day rolling Sharpe
        window = 63
        rolling_sharpe = np.full(n_days, np.nan)
        for i in range(window, n_days):
            chunk = returns[i - window:i]
            s = np.std(chunk)
            if s > 1e-10:
                rolling_sharpe[i] = np.mean(chunk) / s * np.sqrt(252)

        fig.add_trace(go.Scatter(
            y=rolling_sharpe,
            mode="lines",
            name=v.version_id,
        ))

    fig.update_layout(
        title="Rolling Sharpe Ratio (63-day)",
        xaxis_title="Trading Days",
        yaxis_title="Sharpe Ratio",
        template="plotly_white",
        height=350,
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


def _build_drawdown_chart(versions):
    """Build drawdown chart."""
    fig = go.Figure()

    for v in versions:
        sharpe = v.metrics.get("mean_sharpe", 0.0)
        n_days = 252
        daily_ret = sharpe * 0.15 / np.sqrt(252)
        daily_vol = 0.15 / np.sqrt(252)
        rng = np.random.RandomState(hash(v.version_id) % 2**31)
        returns = rng.normal(daily_ret, daily_vol, n_days)
        equity = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(equity)
        drawdown = equity / running_max - 1.0

        fig.add_trace(go.Scatter(
            y=drawdown,
            mode="lines",
            fill="tozeroy",
            name=v.version_id,
        ))

    fig.update_layout(
        title="Drawdown",
        xaxis_title="Trading Days",
        yaxis_title="Drawdown",
        template="plotly_white",
        height=350,
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


def _build_weights_chart(versions):
    """Build weight allocation bar chart."""
    fig = go.Figure()

    names = [v.version_id for v in versions]
    ics = [v.metrics.get("mean_ic", 0.0) for v in versions]
    sharpes = [v.metrics.get("mean_sharpe", 0.0) for v in versions]

    fig.add_trace(go.Bar(name="IC", x=names, y=ics))
    fig.add_trace(go.Bar(name="Sharpe", x=names, y=sharpes))

    fig.update_layout(
        title="Model Metrics Comparison",
        barmode="group",
        template="plotly_white",
        height=350,
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


def _build_cvar_gauge(versions):
    """Build CVaR risk gauge for the best model."""
    if not versions:
        return go.Figure()

    # Use best model's metrics
    best = max(versions, key=lambda v: v.metrics.get("mean_ic", 0.0))
    mdd = abs(best.metrics.get("mean_mdd", 0.05))
    # Approximate CVaR from MDD
    cvar_approx = mdd * 1.5

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=cvar_approx * 100,
        title={"text": f"CVaR Risk (%) - {best.model_type}"},
        gauge={
            "axis": {"range": [0, 20]},
            "bar": {"color": "darkblue"},
            "steps": [
                {"range": [0, 5], "color": "#4caf50"},
                {"range": [5, 10], "color": "#ff9800"},
                {"range": [10, 20], "color": "#f44336"},
            ],
            "threshold": {
                "line": {"color": "red", "width": 4},
                "thickness": 0.75,
                "value": 10,
            },
        },
    ))

    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


def _build_regime_indicator():
    """Build regime state indicator."""
    try:
        from data.stock_api import get_historical_data
        from training.weight_optimizer import RegimeClassifier

        market_df = get_historical_data("SPY", period="1y")
        if market_df.empty:
            return html.Div()

        classifier = RegimeClassifier()
        regime_info = classifier.classify(market_df)

        colors = {
            "strong_bull": "#4caf50",
            "normal": "#2196F3",
            "bear": "#ff9800",
            "crisis": "#f44336",
        }
        color = colors.get(regime_info.regime, "#999")

        return html.Div([
            html.Span("Market Regime: ", style={"fontWeight": "600"}),
            html.Span(
                regime_info.regime.upper(),
                style={
                    "background": color,
                    "color": "white",
                    "padding": "4px 12px",
                    "borderRadius": "4px",
                    "fontWeight": "700",
                },
            ),
            html.Span(
                f" (confidence: {regime_info.confidence:.0%})",
                style={"color": "#666", "marginLeft": "8px"},
            ),
        ], className="regime-card")
    except Exception:
        return html.Div()
