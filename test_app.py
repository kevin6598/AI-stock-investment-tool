"""Quick functional test for the Dash app."""
import requests
import json
import sys

BASE = "http://127.0.0.1:8050"
CALLBACK = f"{BASE}/_dash-update-component"
HEADERS = {"Content-Type": "application/json"}

results = []

def test(name, passed, detail=""):
    status = "PASS" if passed else "FAIL"
    results.append((name, passed))
    print(f"  [{status}] {name}" + (f" - {detail}" if detail else ""))


print("=== Stock Investment Tool - Tests ===\n")

# 1. Main page loads
print("1. Main page")
r = requests.get(BASE)
test("Page loads (HTTP 200)", r.status_code == 200)
test("Title present", "Stock Investment Tool" in r.text)
test("CSS loaded", "style.css" in r.text)

# 2. Tab switching
print("\n2. Tab switching")
for tab_name, tab_val, expected_text in [
    ("Stock Lookup", "tab-lookup", "Stock Price Lookup"),
    ("Portfolio", "tab-portfolio", "Portfolio Management"),
    ("Technical Analysis", "tab-technical", "Technical Analysis"),
]:
    payload = {
        "output": "tab-content.children",
        "outputs": {"id": "tab-content", "property": "children"},
        "inputs": [{"id": "main-tabs", "property": "value", "value": tab_val}],
        "changedPropIds": ["main-tabs.value"],
    }
    r = requests.post(CALLBACK, json=payload)
    test(f"{tab_name} tab renders", r.status_code == 200 and expected_text in r.text)

# 3. Stock Lookup callback (AAPL)
print("\n3. Stock Lookup - AAPL")
payload = {
    "output": "..stock-info-card.children...price-chart.figure...price-chart.style..",
    "outputs": [
        {"id": "stock-info-card", "property": "children"},
        {"id": "price-chart", "property": "figure"},
        {"id": "price-chart", "property": "style"},
    ],
    "inputs": [{"id": "lookup-button", "property": "n_clicks", "value": 1}],
    "state": [
        {"id": "lookup-ticker-input", "property": "value", "value": "AAPL"},
        {"id": "lookup-period-dropdown", "property": "value", "value": "1y"},
    ],
    "changedPropIds": ["lookup-button.n_clicks"],
}
r = requests.post(CALLBACK, json=payload)
test("Callback returns 200", r.status_code == 200)
if r.status_code == 200:
    data = r.json()
    resp = data.get("response", {})
    # Check info card has Apple data
    info_text = json.dumps(resp.get("stock-info-card", {}))
    test("Apple info returned", "AAPL" in info_text, f"contains ticker: {'AAPL' in info_text}")
    test("Price chart visible", resp.get("price-chart", {}).get("style", {}).get("display") == "block")
    fig = resp.get("price-chart", {}).get("figure", {})
    test("Chart has data", "data" in fig and len(fig.get("data", [])) > 0)

# 4. Stock Lookup - invalid ticker
print("\n4. Stock Lookup - Invalid ticker")
payload["state"][0]["value"] = "XYZXYZXYZ123"
payload["inputs"][0]["value"] = 2
r = requests.post(CALLBACK, json=payload)
test("Invalid ticker handled", r.status_code == 200)
if r.status_code == 200:
    data = r.json()
    info_text = json.dumps(data.get("response", {}).get("stock-info-card", {}))
    test("Error message shown", "Could not find" in info_text or "error" in info_text.lower())

# 5. Portfolio - Add holding
print("\n5. Portfolio - Add holding")
payload = {
    "output": "..port-form-message.children...portfolio-refresh-trigger.data..",
    "outputs": [
        {"id": "port-form-message", "property": "children"},
        {"id": "portfolio-refresh-trigger", "property": "data"},
    ],
    "inputs": [{"id": "port-add-button", "property": "n_clicks", "value": 1}],
    "state": [
        {"id": "port-ticker-input", "property": "value", "value": "MSFT"},
        {"id": "port-shares-input", "property": "value", "value": 10},
        {"id": "port-cost-input", "property": "value", "value": 350.0},
        {"id": "port-date-input", "property": "value", "value": "2024-06-01"},
    ],
    "changedPropIds": ["port-add-button.n_clicks"],
}
r = requests.post(CALLBACK, json=payload)
test("Add holding returns 200", r.status_code == 200)
if r.status_code == 200:
    data = r.json()
    msg_text = json.dumps(data.get("response", {}).get("port-form-message", {}))
    test("Success message", "Added" in msg_text and "MSFT" in msg_text, msg_text[:100])

# 6. Portfolio - View holdings
print("\n6. Portfolio - View holdings table")
payload = {
    "output": "..portfolio-summary.children...holdings-table-container.children..",
    "outputs": [
        {"id": "portfolio-summary", "property": "children"},
        {"id": "holdings-table-container", "property": "children"},
    ],
    "inputs": [
        {"id": "main-tabs", "property": "value", "value": "tab-portfolio"},
        {"id": "portfolio-refresh-trigger", "property": "data", "value": {"ts": 1}},
        {"id": {"type": "delete-btn", "index": ""}, "property": "n_clicks", "value": []},
    ],
    "changedPropIds": ["portfolio-refresh-trigger.data"],
}
r = requests.post(CALLBACK, json=payload)
test("Portfolio view returns 200", r.status_code == 200)
if r.status_code == 200:
    data = r.json()
    summary_text = json.dumps(data.get("response", {}).get("portfolio-summary", {}))
    table_text = json.dumps(data.get("response", {}).get("holdings-table-container", {}))
    test("Summary cards present", "Total Value" in summary_text)
    test("MSFT in holdings table", "MSFT" in table_text)
    test("P&L calculated", "$" in table_text and ("+" in table_text or "-" in table_text))

# 7. Technical Analysis
print("\n7. Technical Analysis - AAPL with indicators")
payload = {
    "output": "..ta-error-message.children...ta-chart.figure...ta-chart.style..",
    "outputs": [
        {"id": "ta-error-message", "property": "children"},
        {"id": "ta-chart", "property": "figure"},
        {"id": "ta-chart", "property": "style"},
    ],
    "inputs": [{"id": "ta-button", "property": "n_clicks", "value": 1}],
    "state": [
        {"id": "ta-ticker-input", "property": "value", "value": "AAPL"},
        {"id": "ta-period-dropdown", "property": "value", "value": "6mo"},
        {"id": "ta-indicators", "property": "value", "value": ["sma", "ema", "rsi", "macd"]},
    ],
    "changedPropIds": ["ta-button.n_clicks"],
}
r = requests.post(CALLBACK, json=payload)
test("TA callback returns 200", r.status_code == 200)
if r.status_code == 200:
    data = r.json()
    resp = data.get("response", {})
    test("Chart visible", resp.get("ta-chart", {}).get("style", {}).get("display") == "block")
    fig = resp.get("ta-chart", {}).get("figure", {})
    traces = fig.get("data", [])
    test("Multiple traces (OHLC + indicators)", len(traces) >= 5, f"got {len(traces)} traces")
    trace_names = [t.get("name", "") for t in traces]
    test("Has candlestick (OHLC)", "OHLC" in trace_names)
    test("Has SMA", any("SMA" in n for n in trace_names))
    test("Has EMA", any("EMA" in n for n in trace_names))
    test("Has RSI", "RSI" in trace_names)
    test("Has MACD", "MACD" in trace_names)

# 8. AI Forecast tab
print("\n8. AI Forecast tab")
payload = {
    "output": "tab-content.children",
    "outputs": {"id": "tab-content", "property": "children"},
    "inputs": [{"id": "main-tabs", "property": "value", "value": "tab-forecast"}],
    "changedPropIds": ["main-tabs.value"],
}
r = requests.post(CALLBACK, json=payload)
test("AI Forecast tab renders", r.status_code == 200 and "AI Forecast" in r.text)

# Summary
print("\n" + "=" * 40)
passed = sum(1 for _, p in results if p)
total = len(results)
print(f"Results: {passed}/{total} tests passed")
if passed < total:
    print("Failed tests:")
    for name, p in results:
        if not p:
            print(f"  - {name}")
    sys.exit(1)
else:
    print("All tests passed!")
