from dash import callback, Input, Output, State, html, no_update
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data.stock_api import get_historical_data
from data.indicators import add_sma, add_ema, add_rsi, add_macd


@callback(
    Output("ta-error-message", "children"),
    Output("ta-chart", "figure"),
    Output("ta-chart", "style"),
    Input("ta-button", "n_clicks"),
    State("ta-ticker-input", "value"),
    State("ta-period-dropdown", "value"),
    State("ta-indicators", "value"),
    prevent_initial_call=True,
)
def render_technical_chart(n_clicks, ticker, period, indicators):
    if not ticker:
        return html.Div("Please enter a ticker symbol.", className="error-msg"), {}, {"display": "none"}

    ticker = ticker.strip().upper()
    indicators = indicators or []

    df = get_historical_data(ticker, period=period)
    if df.empty:
        return html.Div(f"No data found for '{ticker}'.", className="error-msg"), {}, {"display": "none"}

    # Compute indicators
    if "sma" in indicators:
        add_sma(df)
    if "ema" in indicators:
        add_ema(df)
    if "rsi" in indicators:
        add_rsi(df)
    if "macd" in indicators:
        add_macd(df)

    # Determine subplot layout
    show_rsi = "rsi" in indicators
    show_macd = "macd" in indicators

    num_rows = 2  # candlestick + volume always shown
    row_heights = [0.5, 0.15]
    if show_rsi:
        num_rows += 1
        row_heights.append(0.15)
    if show_macd:
        num_rows += 1
        row_heights.append(0.2)

    # Normalize heights
    total = sum(row_heights)
    row_heights = [h / total for h in row_heights]

    subplot_titles = [f"{ticker} Price", "Volume"]
    if show_rsi:
        subplot_titles.append("RSI (14)")
    if show_macd:
        subplot_titles.append("MACD")

    fig = make_subplots(
        rows=num_rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_heights,
        subplot_titles=subplot_titles,
    )

    # Row 1: Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        name="OHLC",
        increasing_line_color="#26a69a",
        decreasing_line_color="#ef5350",
    ), row=1, col=1)

    # Overlay SMA
    if "sma" in indicators:
        for col_name, color in [("SMA_20", "#FF9800"), ("SMA_50", "#9C27B0")]:
            if col_name in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.index, y=df[col_name],
                    mode="lines", name=col_name,
                    line=dict(color=color, width=1.5),
                ), row=1, col=1)

    # Overlay EMA
    if "ema" in indicators:
        for col_name, color in [("EMA_12", "#2196F3"), ("EMA_26", "#F44336")]:
            if col_name in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.index, y=df[col_name],
                    mode="lines", name=col_name,
                    line=dict(color=color, width=1.5, dash="dash"),
                ), row=1, col=1)

    # Row 2: Volume
    colors = ["#26a69a" if c >= o else "#ef5350" for c, o in zip(df["Close"], df["Open"])]
    fig.add_trace(go.Bar(
        x=df.index, y=df["Volume"],
        name="Volume",
        marker_color=colors,
        showlegend=False,
    ), row=2, col=1)

    current_row = 3

    # RSI panel
    if show_rsi and "RSI" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["RSI"],
            mode="lines", name="RSI",
            line=dict(color="#7E57C2", width=1.5),
        ), row=current_row, col=1)

        # Overbought/Oversold lines
        fig.add_hline(y=70, line_dash="dash", line_color="red",
                      annotation_text="70", row=current_row, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green",
                      annotation_text="30", row=current_row, col=1)
        fig.update_yaxes(range=[0, 100], row=current_row, col=1)
        current_row += 1

    # MACD panel
    if show_macd and "MACD" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["MACD"],
            mode="lines", name="MACD",
            line=dict(color="#2196F3", width=1.5),
        ), row=current_row, col=1)

        fig.add_trace(go.Scatter(
            x=df.index, y=df["MACD_Signal"],
            mode="lines", name="Signal",
            line=dict(color="#FF9800", width=1.5),
        ), row=current_row, col=1)

        hist_colors = ["#26a69a" if v >= 0 else "#ef5350" for v in df["MACD_Hist"]]
        fig.add_trace(go.Bar(
            x=df.index, y=df["MACD_Hist"],
            name="Histogram",
            marker_color=hist_colors,
        ), row=current_row, col=1)

    # Layout
    fig.update_layout(
        template="plotly_white",
        height=200 + num_rows * 200,
        margin=dict(l=50, r=30, t=50, b=50),
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        showlegend=True,
    )

    return "", fig, {"display": "block"}
