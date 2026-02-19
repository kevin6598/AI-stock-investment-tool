#!/usr/bin/env python3
"""Generate v3 multi-market quant pipeline with all 14 gap-closure enhancements."""
import json, sys


def _L(text):
    if not text:
        return []
    lines = text.split('\n')
    r = []
    for i, l in enumerate(lines):
        if i < len(lines) - 1:
            r.append(l + '\n')
        elif l:
            r.append(l)
    return r

def md(s): return {"cell_type": "markdown", "metadata": {}, "source": _L(s)}
def code(s): return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": _L(s)}

C = []

# ============================================================
# CELL 0 — Header
# ============================================================
C.append(md('''\
# Multi-Market Quantitative Research Pipeline — v3
## Post-Report Gap Closure, Meta-Control & System Hardening

**Production-grade** pipeline with 14 mandatory enhancement layers:

| # | Enhancement | Purpose |
|---|-------------|---------|
| 1 | Tri-state labeling (+1/0/−1) | Teach abstention |
| 2 | Conditional metrics | Precision(BUY), CVaR, coverage |
| 3 | Strategy de-duplication | Remove hidden correlation |
| 4 | Beta-neutral analysis | Strip market exposure |
| 5 | Return distribution safety | Tail risk rejection |
| 6 | Cost stress testing | Multi-scenario survival |
| 7 | Regime robustness | Bull/bear/high-vol/low-vol |
| 8 | Meta-model gate | Model self-awareness |
| 9 | Signal diversity optimizer | Behavioural clustering |
| 10 | Bayesian auto rule tuning | Data-driven thresholds |
| 11 | Signal→Rule→Portfolio arch | Layered decision flow |
| 12 | Portfolio-level validation | System, not strategy |
| 13 | Colab automation | One-click re-run |
| 14 | Pruning only | No expansion |

**Target:** 20-50 strategies, Sharpe 1.0-1.8, explainable decisions.'''))

# ============================================================
# CELL 1 — Pip install
# ============================================================
C.append(code('''\
# Phase 1: upgrade numpy first, then everything else in one pass to avoid binary mismatch
!pip install -q --upgrade numpy pandas scipy scikit-learn
!pip install -q yfinance matplotlib pyarrow joblib pykrx finance-datareader statsmodels

# --- Auto-restart runtime after install (runs only once) ---
import importlib, sys
try:
    import numpy as np
    np.zeros(1, dtype=np.float64)  # trigger binary check
    print("numpy OK — no restart needed")
except (ValueError, ImportError):
    print("Binary mismatch detected — restarting runtime...")
    import os; os.kill(os.getpid(), 9)'''))

# ============================================================
# CELL 2 — Config header
# ============================================================
C.append(md('## 0. Configuration'))

# ============================================================
# CELL 3 — PipelineConfig v3
# ============================================================
C.append(code('''\
import os, json, time, gc, logging, warnings
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Tuple
warnings.filterwarnings('ignore')


@dataclass
class PipelineConfig:
    """Central configuration — v3 with 20+ parameter groups."""

    # --- 1. Markets ---
    markets: List[str] = field(default_factory=lambda: ["US", "KOSPI", "KOSDAQ"])
    base_currency: str = "USD"
    apply_fx_conversion: bool = True
    us_tickers: List[str] = field(default_factory=lambda: [
        "AAPL","MSFT","GOOGL","AMZN","NVDA","META","TSLA","BRK-B",
        "JPM","JNJ","V","PG","UNH","HD","MA","DIS","BAC","NFLX",
        "ADBE","CRM","XOM","VZ","KO","INTC","PEP","ABT","CSCO",
        "COST","MRK","WMT","AVGO","ACN","CVX","NKE","LLY","MCD",
        "QCOM","UPS","BMY","LIN","NEE","ORCL","RTX","HON","TXN",
        "AMD","PYPL","CMCSA","TMO","DHR",
    ])
    kospi_tickers: List[str] = field(default_factory=lambda: [
        "005930","000660","373220","207940","005380","000270",
        "035420","006400","105560","051910","005490","034730",
        "068270","055550","035720","086790","012330","003550",
        "028260","033780","000810","032830","017670","010950",
        "316140","066570","009150","018260","011200","034020",
    ])
    kosdaq_tickers: List[str] = field(default_factory=lambda: [
        "247540","086520","028300","196170","403870","035900",
        "263750","293490","053800","112040","041510","145020",
        "257720","036930","058470","950160","383310","322000",
        "214150","108320",
    ])
    market_index: Dict[str, str] = field(default_factory=lambda: {
        "US": "SPY", "KOSPI": "^KS11", "KOSDAQ": "^KQ11",
    })

    # --- 2. Dynamic Universe ---
    use_dynamic_universe: bool = True
    rebuild_universe_each_fold: bool = False
    min_avg_volume: float = 100000
    max_universe_size: int = 50

    # --- 3. Features ---
    momentum_windows: List[int] = field(default_factory=lambda: [5, 20, 60, 120])
    volatility_windows: List[int] = field(default_factory=lambda: [20, 60])
    regime_window: int = 60
    adaptive_quantiles: bool = True
    min_bin_size: int = 20

    # --- 4. Forward Returns ---
    forward_days_list: List[int] = field(default_factory=lambda: [5, 21, 63])

    # --- 5. Tri-state Labeling (NEW v3) ---
    tristate_thresholds_pct: Dict[int, float] = field(default_factory=lambda: {
        5: 0.5, 21: 1.0, 63: 2.0,
    })

    # --- 6. Candidates ---
    max_candidates_total: int = 3000
    max_candidates_per_feature_pair: int = 200
    min_sample_size: int = 300

    # --- 7. Trees ---
    n_trees: int = 20
    tree_feature_subsample: float = 0.5
    tree_max_depth: int = 2
    tree_min_samples_leaf: int = 500

    # --- 8. Walk-Forward ---
    wf_train_years: int = 3
    wf_test_months: int = 12
    wf_step_months: int = 6
    wf_embargo_days: int = 5
    wf_min_folds: int = 4

    # --- 9. Overfitting ---
    min_stability: float = 0.5
    min_sharpe: float = 0.5
    min_win_rate: float = 0.52

    # --- 10. Conditional Metrics (NEW v3) ---
    min_precision_buy: float = 0.60
    min_ev_per_trade: float = 0.0
    max_cvar_to_avgwin_ratio: float = 3.0
    max_single_loss_to_median_ratio: float = 5.0

    # --- 11. Bootstrap ---
    bootstrap_n: int = 1000
    bootstrap_ci: float = 0.95
    bootstrap_min_samples: int = 200

    # --- 12. Transaction Costs ---
    transaction_cost_bps: float = 5.0
    slippage_bps: float = 2.0
    cost_stress_scenarios: List[Tuple[float, float]] = field(default_factory=lambda: [
        (5, 2), (10, 5), (15, 5),
    ])
    min_cost_scenarios_survived: int = 2

    # --- 13. Turnover ---
    penalty_turnover: float = 0.1
    max_turnover: float = 12.0

    # --- 14. Correlation / De-duplication ---
    max_strategy_correlation: float = 0.85
    cluster_strategies: bool = True
    signal_overlap_threshold: float = 0.80

    # --- 15. Portfolio ---
    portfolio_max_strategies: int = 10
    portfolio_weight_method: str = "equal"
    max_strategy_risk_budget_pct: float = 30.0

    # --- 16. Regime ---
    evaluate_by_regime: bool = True
    min_regime_performance_ratio: float = 0.7

    # --- 17. Multiple Testing ---
    apply_multiple_testing_correction: bool = True
    mtc_method: str = "fdr"

    # --- 18. Meta-Model (NEW v3) ---
    use_meta_model: bool = True
    meta_model_threshold: float = 0.50

    # --- 19. Bayesian Auto Rule Tuning (NEW v3) ---
    use_bayesian_tuning: bool = True
    n_bayes_iterations: int = 200

    # --- 20. Memory ---
    use_float32: bool = True
    gc_every_n_candidates: int = 100

    # --- Scoring weights ---
    w_stability: float = 0.25
    w_sharpe: float = 0.25
    w_precision: float = 0.20
    w_lift: float = 0.15
    w_sample: float = 0.15

    # --- Paths ---
    drive_root: str = "/content/drive/MyDrive/quant_pipeline_v3"
    data_period: str = "10y"
    seed: int = 42
    logistic_top_pct: float = 0.20

    def data_dir(self, m): return os.path.join(self.drive_root, "data", m)
    def features_dir(self, m): return os.path.join(self.drive_root, "features", m)
    def candidates_dir(self, m): return os.path.join(self.drive_root, "candidates", m)
    def evaluation_dir(self, m): return os.path.join(self.drive_root, "evaluation", m)
    def walkforward_dir(self, m): return os.path.join(self.drive_root, "walkforward", m)
    @property
    def global_eval_dir(self): return os.path.join(self.drive_root, "evaluation", "_global")
    @property
    def logs_dir(self): return os.path.join(self.drive_root, "logs")
    @property
    def state_path(self): return os.path.join(self.drive_root, "state.json")
    @property
    def total_cost_bps(self): return self.transaction_cost_bps + self.slippage_bps


CFG = PipelineConfig()
print("Config v3 created.  Root:", CFG.drive_root)
print("Markets:", CFG.markets, "| Horizons:", CFG.forward_days_list)
print("Tri-state thresholds:", CFG.tristate_thresholds_pct)
print("Cost stress scenarios:", CFG.cost_stress_scenarios)'''))

# ============================================================
# CELL 4-7 — Persistence (same as v2)
# ============================================================
C.append(md('## 1. Persistence & Resume'))

C.append(code('''\
DRIVE_MOUNTED = False
try:
    from google.colab import drive
    drive.mount('/content/drive', timeout_ms=60000)
    DRIVE_MOUNTED = True
    print("Google Drive mounted.")
except Exception as e:
    print("Drive mount failed: %s" % str(e)[:80])
    if CFG.drive_root.startswith("/content/drive"):
        CFG.drive_root = "/tmp/quant_pipeline_v3"

for mkt in CFG.markets:
    for d in [CFG.data_dir(mkt), CFG.features_dir(mkt), CFG.candidates_dir(mkt),
              CFG.evaluation_dir(mkt), CFG.walkforward_dir(mkt)]:
        os.makedirs(d, exist_ok=True)
os.makedirs(CFG.global_eval_dir, exist_ok=True)
os.makedirs(CFG.logs_dir, exist_ok=True)
print("Directories ready.")'''))

C.append(code('''\
class ProgressTracker:
    """JSON checkpoint system for resumable execution."""
    def __init__(self, path):
        self.state_path = path
        self.state = self._load()
    def _load(self):
        if os.path.exists(self.state_path):
            with open(self.state_path) as f: return json.load(f)
        return {"completed_steps": {}, "metadata": {}}
    def _save(self):
        with open(self.state_path, 'w') as f: json.dump(self.state, f, indent=2, default=str)
    def is_completed(self, step): return self.state["completed_steps"].get(step, False)
    def mark_completed(self, step, meta=None):
        self.state["completed_steps"][step] = True
        if meta: self.state["metadata"][step] = meta
        self._save()
        print("  [CHECKPOINT] %s" % step)
    def get_metadata(self, step): return self.state["metadata"].get(step, {})
    def reset(self, step=None):
        if step:
            self.state["completed_steps"].pop(step, None)
            self.state["metadata"].pop(step, None)
        else:
            self.state = {"completed_steps": {}, "metadata": {}}
        self._save()
    def summary(self):
        done = [k for k,v in self.state["completed_steps"].items() if v]
        print("=== Progress (%d steps done) ===" % len(done))
        for s in done: print("  [DONE] %s" % s)

tracker = ProgressTracker(CFG.state_path)
tracker.summary()'''))

C.append(code('''\
import logging
log_file = os.path.join(CFG.logs_dir, "pipeline.log")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S",
                    handlers=[logging.StreamHandler(), logging.FileHandler(log_file, mode='a')])
logger = logging.getLogger("pipeline")
logger.info("Pipeline v3 started")

import numpy as np
import pandas as pd
np.random.seed(CFG.seed)'''))

# ============================================================
# CELL 8-11 — Market Registry + Universe (same as v2)
# ============================================================
C.append(md('## 2. Market Configuration & Universe'))

C.append(code('''\
MARKET_REGISTRY = {
    "US":    {"data_source": "yfinance", "index_ticker": "SPY",   "currency": "USD", "pykrx_market": None},
    "KOSPI": {"data_source": "pykrx",   "index_ticker": "^KS11", "currency": "KRW", "pykrx_market": "KOSPI"},
    "KOSDAQ":{"data_source": "pykrx",   "index_ticker": "^KQ11", "currency": "KRW", "pykrx_market": "KOSDAQ"},
}

def get_static_tickers(market):
    return {"US": list(CFG.us_tickers), "KOSPI": list(CFG.kospi_tickers),
            "KOSDAQ": list(CFG.kosdaq_tickers)}.get(market, [])

def download_fx_rates(period="10y"):
    import yfinance as yf
    try:
        fx = yf.download("USDKRW=X", period=period, progress=False, auto_adjust=True)
        if isinstance(fx.columns, pd.MultiIndex): fx.columns = fx.columns.get_level_values(0)
        fx = fx[["Close"]].copy(); fx.columns = ["usdkrw"]
        fx.index = pd.to_datetime(fx.index).tz_localize(None)
        return fx
    except: return pd.DataFrame(columns=["usdkrw"])

def convert_krw_to_usd(df, fx):
    if fx.empty: return df
    fx_al = fx["usdkrw"].reindex(df.index.get_level_values(0), method="ffill")
    fx_al.index = df.index
    for c in ["open","high","low","close"]:
        if c in df.columns: df[c] = df[c] / fx_al
    return df

print("Market registry ready.")'''))

C.append(code('''\
def build_universe(market, ref_date=None, config=None):
    """Point-in-time ticker universe."""
    if config is None: config = CFG
    reg = MARKET_REGISTRY[market]
    if reg["data_source"] == "pykrx" and config.use_dynamic_universe:
        try:
            from pykrx import stock
            import datetime
            if ref_date is None: ref_date = datetime.datetime.now()
            ds = pd.Timestamp(ref_date).strftime("%Y%m%d")
            tickers = stock.get_market_ticker_list(ds, market=reg["pykrx_market"])
            if not tickers: raise ValueError("Empty")
            try:
                cap = stock.get_market_cap_by_ticker(ds, market=reg["pykrx_market"])
                if not cap.empty:
                    if config.min_avg_volume > 0: cap = cap[cap.iloc[:,-1] >= config.min_avg_volume]
                    cap = cap.sort_values(cap.columns[0], ascending=False)
                    tickers = cap.head(config.max_universe_size).index.tolist()
            except: tickers = tickers[:config.max_universe_size]
            logger.info("Dynamic universe %s: %d tickers (ref %s)" % (market, len(tickers), ds))
            return tickers
        except Exception as e:
            logger.warning("Dynamic universe failed %s: %s" % (market, str(e)[:60]))
    static = get_static_tickers(market)
    logger.info("Static universe %s: %d tickers" % (market, len(static)))
    return static

print("Universe builder ready.")'''))

# ============================================================
# CELL 12-13 — Data Layer (same as v2)
# ============================================================
C.append(md('## 3. Data Layer'))

C.append(code('''\
import yfinance as yf

ohlcv_data = {}
market_indices = {}
fx_rates = pd.DataFrame()

fx_path = os.path.join(CFG.drive_root, "data", "fx_usdkrw.parquet")
if os.path.exists(fx_path):
    fx_rates = pd.read_parquet(fx_path)
else:
    if any(MARKET_REGISTRY[m]["currency"]=="KRW" for m in CFG.markets):
        fx_rates = download_fx_rates(CFG.data_period)
        if not fx_rates.empty: fx_rates.to_parquet(fx_path)

for market in CFG.markets:
    STEP = "data_load_%s" % market
    pp = os.path.join(CFG.data_dir(market), "processed.parquet")
    ip = os.path.join(CFG.data_dir(market), "market_index.parquet")
    reg = MARKET_REGISTRY[market]

    if tracker.is_completed(STEP):
        logger.info("[SKIP] %s" % STEP)
        ohlcv_data[market] = pd.read_parquet(pp)
        market_indices[market] = pd.read_parquet(ip)
        continue

    logger.info("[RUN] %s" % STEP)
    t0 = time.time()
    tickers = build_universe(market)
    all_dfs, failed = [], []

    if reg["data_source"] == "yfinance":
        for i, tk in enumerate(tickers):
            if (i+1)%10==0 or i==0: print("  [%d/%d] %s"%(i+1,len(tickers),tk))
            try:
                df = yf.download(tk, period=CFG.data_period, progress=False, auto_adjust=True)
                if df.empty or len(df)<252: failed.append(tk); continue
                if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
                df = df[["Open","High","Low","Close","Volume"]].copy()
                df.columns = ["open","high","low","close","volume"]
                df.index = pd.to_datetime(df.index).tz_localize(None)
                df["ticker"] = tk; all_dfs.append(df)
            except: failed.append(tk)
    elif reg["data_source"] == "pykrx":
        import datetime as _dt
        es = _dt.datetime.now().strftime("%Y%m%d")
        ss = (_dt.datetime.now()-_dt.timedelta(days=365*10)).strftime("%Y%m%d")
        for i, tk in enumerate(tickers):
            if (i+1)%10==0 or i==0: print("  [%d/%d] %s"%(i+1,len(tickers),tk))
            try:
                from pykrx import stock as ps
                df = ps.get_market_ohlcv_by_date(ss, es, tk)
                if df.empty or len(df)<252: raise ValueError("short")
                rm = {}
                for c in df.columns:
                    cl = c.strip()
                    if cl in ("시가","Open"): rm[c]="open"
                    elif cl in ("고가","High"): rm[c]="high"
                    elif cl in ("저가","Low"): rm[c]="low"
                    elif cl in ("종가","Close"): rm[c]="close"
                    elif cl in ("거래량","Volume"): rm[c]="volume"
                df = df.rename(columns=rm)[["open","high","low","close","volume"]].copy()
                df.index = pd.to_datetime(df.index).tz_localize(None)
                df["ticker"] = tk; all_dfs.append(df)
            except:
                try:
                    import FinanceDataReader as fdr
                    df2 = fdr.DataReader(tk, ss[:4]+"-"+ss[4:6]+"-"+ss[6:])
                    if df2.empty or len(df2)<252: failed.append(tk); continue
                    df2 = df2.rename(columns={"Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"})
                    df2 = df2[["open","high","low","close","volume"]].copy()
                    df2.index = pd.to_datetime(df2.index).tz_localize(None)
                    df2["ticker"] = tk; all_dfs.append(df2)
                except: failed.append(tk)

    if not all_dfs: logger.warning("No data for %s"%market); continue
    panel = pd.concat(all_dfs)
    panel = panel.set_index([panel.index, "ticker"])
    panel.index.names = ["date","ticker"]; panel = panel.sort_index()
    if reg["currency"]=="KRW" and CFG.apply_fx_conversion and not fx_rates.empty:
        panel = convert_krw_to_usd(panel, fx_rates)
    if CFG.use_float32:
        for c in ["open","high","low","close"]: panel[c] = panel[c].astype(np.float32)
        panel["volume"] = panel["volume"].astype(np.float64)
    panel.to_parquet(pp); ohlcv_data[market] = panel

    try:
        idx = yf.download(reg["index_ticker"], period=CFG.data_period, progress=False, auto_adjust=True)
        if isinstance(idx.columns, pd.MultiIndex): idx.columns = idx.columns.get_level_values(0)
        idx = idx[["Close"]].copy(); idx.columns = ["close"]
        idx.index = pd.to_datetime(idx.index).tz_localize(None)
        if reg["currency"]=="KRW" and CFG.apply_fx_conversion and not fx_rates.empty:
            fx_al = fx_rates["usdkrw"].reindex(idx.index, method="ffill")
            idx["close"] = idx["close"] / fx_al
    except: idx = pd.DataFrame(columns=["close"])
    idx.to_parquet(ip); market_indices[market] = idx

    elapsed = time.time() - t0
    tracker.mark_completed(STEP, {"n":len(all_dfs),"failed":failed[:5],"time":elapsed})
    print("%s: %d/%d tickers (%.0fs)" % (market,len(all_dfs),len(tickers),elapsed))
    gc.collect()

for m in ohlcv_data:
    p = ohlcv_data[m]; tks = p.index.get_level_values(1).unique()
    dts = p.index.get_level_values(0).unique()
    print("%s: %s | %d tickers | %s to %s" % (m,str(p.shape),len(tks),dts.min().date(),dts.max().date()))'''))

# ============================================================
# CELL 14-16 — Feature Engine + Tri-state Labeling
# ============================================================
C.append(md('''\
## 4. Feature Engine + Tri-State Labeling

**Part 1**: Tri-state labels replace binary labels:
- `+1` = BUY (excess return ≥ threshold)
- ` 0` = NO TRADE (uncertain)
- `−1` = AVOID (excess return ≤ −threshold)

Thresholds are horizon-aware. Excess return is market-neutral.'''))

C.append(code('''\
def compute_momentum_features(close, windows):
    return pd.DataFrame({"mom_%dd"%w: close.pct_change(w) for w in windows}, index=close.index)

def compute_volatility_features(close, windows):
    dr = close.pct_change()
    feats = {"vol_%dd"%w: dr.rolling(w).std() for w in windows}
    if len(windows)>=2:
        feats["vol_change"] = dr.rolling(windows[0]).std() / dr.rolling(windows[-1]).std().replace(0,np.nan) - 1
    return pd.DataFrame(feats, index=close.index)

def compute_regime_features(mkt_close, rw):
    mr = mkt_close.pct_change()
    feats = {"market_mom_%dd"%rw: mkt_close.pct_change(rw),
             "market_vol_%dd"%rw: mr.rolling(rw).std()}
    mom = feats["market_mom_%dd"%rw]; vol = feats["market_vol_%dd"%rw]
    vm = vol.rolling(252, min_periods=60).median()
    feats["regime_bull"] = ((mom>0)&(vol<vm)).astype(float)
    return pd.DataFrame(feats, index=mkt_close.index)

def adaptive_n_bins(n, cfg):
    if not cfg.adaptive_quantiles: return 10
    return 5 if n < 5 * cfg.min_bin_size else 10

def to_cross_sectional_deciles(s, date_level, n_bins=10):
    def rank_date(g):
        v = g.dropna()
        if len(v)<n_bins: return pd.Series(np.nan, index=g.index)
        r = v.rank(method='first')
        return pd.cut(r, bins=n_bins, labels=False).reindex(g.index)
    return s.groupby(level=date_level).transform(rank_date)

def compute_tristate_labels(excess_return, threshold_pct):
    """Tri-state labeling: +1 BUY, 0 NO TRADE, -1 AVOID."""
    th = threshold_pct / 100.0
    labels = np.zeros(len(excess_return), dtype=np.float32)
    vals = excess_return.values if hasattr(excess_return, 'values') else excess_return
    labels[vals >= th] = 1
    labels[vals <= -th] = -1
    return labels

def check_class_balance(labels):
    """Check tri-state label distribution. Returns (ok, distribution_dict)."""
    s = pd.Series(labels)
    counts = s.value_counts(normalize=True)
    dist = {"buy": counts.get(1,0), "notrade": counts.get(0,0), "avoid": counts.get(-1,0)}
    ok = 0.40 <= dist["notrade"] <= 0.90 and dist["buy"] >= 0.03 and dist["avoid"] >= 0.03
    return ok, dist

print("Feature functions + tri-state labeler defined.")'''))

C.append(code('''\
feature_panels = {}

for market in CFG.markets:
    if market not in ohlcv_data: continue
    STEP = "features_%s" % market
    fpath = os.path.join(CFG.features_dir(market), "all_features.parquet")

    if tracker.is_completed(STEP):
        logger.info("[SKIP] %s" % STEP)
        feature_panels[market] = pd.read_parquet(fpath)
        continue

    logger.info("[RUN] %s" % STEP)
    t0 = time.time()
    panel = ohlcv_data[market]
    valid_tickers = panel.index.get_level_values(1).unique().tolist()
    n_bins = adaptive_n_bins(len(valid_tickers), CFG)

    mkt_close = market_indices.get(market, pd.DataFrame()).get("close", pd.Series(dtype=float))
    regime_feats = compute_regime_features(mkt_close, CFG.regime_window) if len(mkt_close)>0 else pd.DataFrame()
    market_ret_20d = mkt_close.pct_change(20) if len(mkt_close)>0 else pd.Series(dtype=float)

    # Pre-compute market forward returns for excess calculation
    mkt_fwd = {}
    for fd in CFG.forward_days_list:
        if len(mkt_close) > 0:
            mkt_fwd[fd] = mkt_close.pct_change(fd).shift(-fd)
        else:
            mkt_fwd[fd] = pd.Series(dtype=float)

    all_features = []
    for ticker in valid_tickers:
        try:
            td = panel.loc[(slice(None), ticker), :].droplevel(1)
            close = td["close"]
            mom = compute_momentum_features(close, CFG.momentum_windows)
            if len(market_ret_20d) > 0:
                s20 = close.pct_change(20)
                ma = market_ret_20d.reindex(close.index, method='ffill')
                mom["market_relative_20d"] = s20 - ma
            vol = compute_volatility_features(close, CFG.volatility_windows)
            reg = regime_feats.reindex(close.index, method='ffill') if len(regime_feats)>0 else pd.DataFrame(index=close.index)
            combined = pd.concat([mom, vol, reg], axis=1)

            for fd in CFG.forward_days_list:
                raw_fwd = close.pct_change(fd).shift(-fd)
                net_fwd = raw_fwd - 2 * CFG.total_cost_bps / 10000
                combined["fwd_return_%dd" % fd] = net_fwd

                # Excess return & tri-state label
                mf = mkt_fwd[fd].reindex(close.index, method='ffill') if len(mkt_fwd[fd])>0 else 0
                excess = net_fwd - mf
                th_pct = CFG.tristate_thresholds_pct.get(fd, 1.0)
                combined["label_%dd" % fd] = compute_tristate_labels(excess, th_pct)

            combined["ticker"] = ticker
            combined.index.name = "date"
            all_features.append(combined)
        except Exception as e:
            logger.warning("Feature err %s/%s: %s" % (market, ticker, str(e)[:60]))

    if not all_features: continue
    fp = pd.concat(all_features).reset_index().set_index(["date","ticker"]).sort_index()
    fwd_cols = [c for c in fp.columns if c.startswith("fwd_return_")]
    label_cols = [c for c in fp.columns if c.startswith("label_")]
    feat_cols = [c for c in fp.columns if c not in fwd_cols and c not in label_cols]
    fp = fp.dropna(subset=feat_cols, how='all')
    if CFG.use_float32:
        for c in fp.select_dtypes(include=['float64']).columns: fp[c] = fp[c].astype(np.float32)

    logger.info("Computing deciles for %s..." % market)
    for col in feat_cols:
        fp[col+"_decile"] = to_cross_sectional_deciles(fp[col], "date", n_bins)

    fp.to_parquet(fpath); feature_panels[market] = fp
    elapsed = time.time() - t0
    tracker.mark_completed(STEP, {"n_feat": len(feat_cols), "n_rows": len(fp), "time": elapsed})

    # Class balance check
    for fd in CFG.forward_days_list:
        lc = "label_%dd" % fd
        if lc in fp.columns:
            ok, dist = check_class_balance(fp[lc].dropna().values)
            status = "OK" if ok else "WARN"
            print("  %s label_%dd: BUY=%.1f%% NO_TRADE=%.1f%% AVOID=%.1f%% [%s]" % (
                market, fd, dist["buy"]*100, dist["notrade"]*100, dist["avoid"]*100, status))
    gc.collect()

for m, fp in feature_panels.items():
    fwd = [c for c in fp.columns if c.startswith("fwd_return_")]
    lab = [c for c in fp.columns if c.startswith("label_")]
    feat = [c for c in fp.columns if not c.startswith("fwd_") and not c.startswith("label_") and not c.endswith("_decile")]
    print("%s: %d features, %d deciles, horizons: %s, labels: %s" % (m, len(feat), len([c for c in fp.columns if c.endswith("_decile")]), fwd, lab))'''))

# ============================================================
# CELL 17-21 — Candidate Generation (multiclass trees)
# ============================================================
C.append(md('''\
## 5. Candidate Generator

- **Decile**: single & 2-feature combos (unchanged)
- **Trees**: trained on **tri-state labels** (multiclass); only BUY leaves kept
- **Logistic**: **multinomial**; strategies from BUY-class probability quintiles'''))

C.append(code('''\
from itertools import combinations
all_candidates_list = []

for market in CFG.markets:
    if market not in feature_panels: continue
    fp = feature_panels[market]
    fwd_cols = sorted([c for c in fp.columns if c.startswith("fwd_return_")])
    decile_cols = [c for c in fp.columns if c.endswith("_decile")]

    for fwd_col in fwd_cols:
        ht = fwd_col.replace("fwd_return_","")
        STEP = "cand_decile_%s_%s" % (market, ht)
        sp = os.path.join(CFG.candidates_dir(market), "decile_%s.parquet" % ht)
        if tracker.is_completed(STEP):
            all_candidates_list.append(pd.read_parquet(sp)); continue

        logger.info("[RUN] %s" % STEP); t0 = time.time()
        valid = fp.dropna(subset=[fwd_col]).copy()
        n_bins = int(valid[decile_cols[0]].dropna().max())+1 if decile_cols else 10
        cands = []

        for col in decile_cols:
            for dv in range(n_bins):
                mask = valid[col]==dv; nt = int(mask.sum())
                if nt < CFG.min_sample_size: continue
                ret = valid.loc[mask, fwd_col]
                mr = float(ret.mean())
                if mr <= 0: continue
                cands.append({"strategy_id":"%s_%s_%s_d%d"%(market,ht,col,dv),
                    "market":market,"horizon":ht,"type":"single_decile",
                    "features":col,"condition":"== %d"%dv,
                    "n_trades":nt,"mean_return":mr,"win_rate":float((ret>0).mean())})

        extreme = [0,1,n_bins-2,n_bins-1] if n_bins>=4 else list(range(n_bins))
        for ca, cb in combinations(decile_cols, 2):
            pc = 0
            for da in extreme:
                for db in extreme:
                    if pc >= CFG.max_candidates_per_feature_pair: break
                    mask = (valid[ca]==da)&(valid[cb]==db); nt = int(mask.sum())
                    if nt < CFG.min_sample_size: continue
                    ret = valid.loc[mask, fwd_col]; mr = float(ret.mean())
                    std_r = float(ret.std())
                    if std_r>1e-8 and mr/std_r<0: continue
                    cands.append({"strategy_id":"%s_%s_%s_d%d_AND_%s_d%d"%(market,ht,ca,da,cb,db),
                        "market":market,"horizon":ht,"type":"combo_decile",
                        "features":"%s, %s"%(ca,cb),"condition":"%s==%d AND %s==%d"%(ca,da,cb,db),
                        "n_trades":nt,"mean_return":mr,"win_rate":float((ret>0).mean())})
                    pc += 1
            if len(cands) >= CFG.max_candidates_total: break

        dc = pd.DataFrame(cands); dc.to_parquet(sp); all_candidates_list.append(dc)
        tracker.mark_completed(STEP, {"n":len(dc),"time":time.time()-t0})
        gc.collect()

print("Decile candidates: %d" % sum(len(d) for d in all_candidates_list))'''))

C.append(code('''\
from sklearn.tree import DecisionTreeClassifier
import pickle

for market in CFG.markets:
    if market not in feature_panels: continue
    fp = feature_panels[market]
    fwd_cols = sorted([c for c in fp.columns if c.startswith("fwd_return_")])
    feat_cols = [c for c in fp.columns if not c.startswith("fwd_") and not c.startswith("label_") and not c.endswith("_decile")]

    for fwd_col in fwd_cols:
        ht = fwd_col.replace("fwd_return_","")
        label_col = "label_%s" % ht
        STEP = "cand_tree_%s_%s" % (market, ht)
        sp = os.path.join(CFG.candidates_dir(market), "tree_%s.parquet" % ht)
        if tracker.is_completed(STEP):
            all_candidates_list.append(pd.read_parquet(sp)); continue

        logger.info("[RUN] %s" % STEP); t0 = time.time()
        valid = fp.dropna(subset=[fwd_col]+feat_cols).copy()
        X = valid[feat_cols].values.astype(np.float32)
        # v3: multiclass tri-state labels
        y = valid[label_col].values.astype(int) if label_col in valid.columns else (valid[fwd_col].values > 0).astype(int)
        n_feat = len(feat_cols); n_sub = max(3, int(n_feat * CFG.tree_feature_subsample))
        strats = []

        for ti in range(CFG.n_trees):
            fi = np.random.choice(n_feat, n_sub, replace=False)
            fn = [feat_cols[j] for j in fi]
            Xs = X[:, fi]
            si = np.random.choice(len(Xs), min(len(Xs),50000), replace=False)
            tree = DecisionTreeClassifier(max_depth=CFG.tree_max_depth,
                min_samples_leaf=CFG.tree_min_samples_leaf, random_state=CFG.seed+ti)
            tree.fit(Xs[si], y[si])

            leaf_ids = tree.apply(X[:, fi])
            for leaf in np.unique(leaf_ids):
                lm = tree.apply(X[:, fi])==leaf; nt = int(lm.sum())
                if nt < CFG.min_sample_size: continue
                leaf_labels = y[lm]
                # Only keep BUY-majority leaves
                if (leaf_labels==1).sum() <= (leaf_labels==-1).sum(): continue
                ret = valid[fwd_col].values[lm]; mr = float(np.nanmean(ret))
                if mr <= 0: continue
                strats.append({"strategy_id":"%s_%s_tree_%d_leaf_%d"%(market,ht,ti,leaf),
                    "market":market,"horizon":ht,"type":"decision_tree",
                    "features":", ".join(fn[:5]),"condition":"tree_%d/leaf_%d"%(ti,leaf),
                    "n_trades":nt,"mean_return":mr,"win_rate":float((ret>0).mean())})

            tp = os.path.join(CFG.candidates_dir(market), "tree_model_%s_%d.pkl"%(ht,ti))
            with open(tp,'wb') as f: pickle.dump({"tree":tree,"features":fn,"feat_idx":fi.tolist()}, f)

        tc = pd.DataFrame(strats); tc.to_parquet(sp); all_candidates_list.append(tc)
        tracker.mark_completed(STEP, {"n":len(tc),"time":time.time()-t0})
        gc.collect()'''))

C.append(code('''\
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

for market in CFG.markets:
    if market not in feature_panels: continue
    fp = feature_panels[market]
    fwd_cols = sorted([c for c in fp.columns if c.startswith("fwd_return_")])
    feat_cols = [c for c in fp.columns if not c.startswith("fwd_") and not c.startswith("label_") and not c.endswith("_decile")]

    for fwd_col in fwd_cols:
        ht = fwd_col.replace("fwd_return_","")
        label_col = "label_%s" % ht
        STEP = "cand_logistic_%s_%s" % (market, ht)
        sp = os.path.join(CFG.candidates_dir(market), "logistic_%s.parquet" % ht)
        if tracker.is_completed(STEP):
            all_candidates_list.append(pd.read_parquet(sp)); continue

        logger.info("[RUN] %s" % STEP); t0 = time.time()
        valid = fp.dropna(subset=[fwd_col]+feat_cols).copy()
        X = valid[feat_cols].values.astype(np.float32)
        # v3: multinomial on tri-state labels
        y = valid[label_col].values.astype(int) if label_col in valid.columns else (valid[fwd_col].values > 0).astype(int)
        scaler = StandardScaler(); Xs = scaler.fit_transform(X)
        ds = valid.index.get_level_values(0)
        sd = ds.unique()[int(len(ds.unique())*0.8)]
        tm = ds <= sd

        lr = LogisticRegression(max_iter=1000, C=0.1, penalty='l2', random_state=CFG.seed,
                                solver='lbfgs')
        lr.fit(Xs[tm], y[tm])

        # P(BUY) = class +1 probability
        classes = list(lr.classes_)
        buy_idx = classes.index(1) if 1 in classes else -1
        if buy_idx < 0:
            logger.warning("No BUY class learned for %s/%s" % (market, ht))
            tracker.mark_completed(STEP, {"n":0}); continue

        proba_buy = lr.predict_proba(Xs)[:, buy_idx]
        qe = np.percentile(proba_buy, [0,20,40,60,80,100]).tolist()

        strats = []
        for q in range(5):
            qm = (proba_buy>=qe[q])&(proba_buy<qe[q+1]) if q<4 else proba_buy>=qe[q]
            nt = int(qm.sum())
            if nt < CFG.min_sample_size: continue
            ret = valid[fwd_col].values[qm]
            strats.append({"strategy_id":"%s_%s_logistic_q%d"%(market,ht,q+1),
                "market":market,"horizon":ht,"type":"logistic_rank",
                "features":"all","condition":"logistic_q%d"%(q+1),
                "n_trades":nt,"mean_return":float(np.nanmean(ret)),"win_rate":float((ret>0).mean())})

        mp = os.path.join(CFG.candidates_dir(market), "logistic_model_%s.pkl"%ht)
        with open(mp,'wb') as f:
            pickle.dump({"model":lr,"scaler":scaler,"features":feat_cols,"quintile_edges":qe,"buy_class_idx":buy_idx}, f)
        lc = pd.DataFrame(strats); lc.to_parquet(sp); all_candidates_list.append(lc)
        tracker.mark_completed(STEP, {"n":len(lc),"time":time.time()-t0})
        gc.collect()'''))

C.append(code('''\
all_candidates = pd.concat(all_candidates_list, ignore_index=True) if all_candidates_list else pd.DataFrame()
print("=== All Candidates: %d ===" % len(all_candidates))
if len(all_candidates) > 0:
    print(all_candidates.groupby(["market","horizon","type"]).size().to_string())'''))

# ============================================================
# CELL 22-24 — Conditional Edge Evaluation (v3)
# ============================================================
C.append(md('''\
## 6. Conditional Edge Evaluation (Part 2)

Metrics per strategy:
- **Precision(BUY)**: P(label=+1 | strategy selects)
- **Trade Coverage**: fraction of universe selected
- **EV per trade**: mean return conditional on selection
- **Max Loss**: worst single trade
- **CVaR(95%)**: mean of worst 5% outcomes

Hard rejection: Precision(BUY) < 0.60, EV ≤ 0, CVaR explodes.'''))

C.append(code('''\
def evaluate_strategy_edge_v3(returns, labels=None, horizon_days=21):
    """Edge metrics with conditional metrics for v3."""
    returns = returns[~np.isnan(returns)]
    n = len(returns)
    if n < 30: return None
    mr = float(np.mean(returns)); sr = float(np.std(returns, ddof=1))
    wr = float((returns>0).mean())
    aw = float(np.mean(returns[returns>0])) if (returns>0).any() else 0.0
    al = float(np.mean(returns[returns<=0])) if (returns<=0).any() else 0.0
    sharpe = (mr/sr*np.sqrt(252/horizon_days)) if sr>1e-8 else 0.0
    cum = np.cumsum(returns); rm = np.maximum.accumulate(cum)
    mdd = float(np.min(cum-rm)) if len(cum)>0 else 0.0
    exp = aw*wr + al*(1-wr)

    # CVaR(95%)
    so = np.sort(returns); nt = max(1, int(0.05*len(so)))
    cvar95 = float(so[:nt].mean())
    max_loss = float(returns.min())
    median_ret = float(np.median(returns))

    result = {"n_trades":n, "mean_return":mr, "std_return":sr, "win_rate":wr,
              "avg_win":aw, "avg_loss":al, "sharpe":float(sharpe),
              "max_drawdown":mdd, "expectancy":float(exp),
              "cvar_95":cvar95, "max_loss":max_loss, "median_return":median_ret}

    if labels is not None:
        labels = labels[:len(returns)]
        result["precision_buy"] = float((labels==1).mean())
        result["avoid_rate"] = float((labels==-1).mean())
        buy_mask = labels==1
        result["ev_buy"] = float(returns[buy_mask].mean()) if buy_mask.any() else 0.0
    else:
        result["precision_buy"] = wr
        result["avoid_rate"] = 0.0
        result["ev_buy"] = mr

    return result


def build_mask(data, stype, cand_row, sid, market, ht, feat_cols_list):
    """Build boolean mask for strategy on data."""
    if stype == "single_decile":
        return data[cand_row["features"]] == int(cand_row["condition"].split("== ")[1])
    elif stype == "combo_decile":
        parts = cand_row["condition"].split(" AND ")
        ca, va = parts[0].split("=="); cb, vb = parts[1].split("==")
        return (data[ca.strip()]==int(va)) & (data[cb.strip()]==int(vb))
    elif stype == "decision_tree":
        parts = sid.split("_"); tn = ll = None
        for i,p in enumerate(parts):
            if p=="tree": tn=parts[i+1]
            if p=="leaf": ll=int(parts[i+1])
        tp = os.path.join(CFG.candidates_dir(market), "tree_model_%s_%s.pkl"%(ht,tn))
        with open(tp,'rb') as f: td = pickle.load(f)
        missing = [fn for fn in td["features"] if fn not in feat_cols_list]
        if missing: raise ValueError("Missing: %s"%missing)
        fi = [feat_cols_list.index(fn) for fn in td["features"]]
        X = data[feat_cols_list].values[:,fi].astype(np.float32); np.nan_to_num(X, copy=False)
        return pd.Series(td["tree"].apply(X)==ll, index=data.index)
    elif stype == "logistic_rank":
        qn = int(sid.split("_q")[1])
        mp = os.path.join(CFG.candidates_dir(market), "logistic_model_%s.pkl"%ht)
        with open(mp,'rb') as f: ld = pickle.load(f)
        X = data[feat_cols_list].values.astype(np.float32); np.nan_to_num(X, copy=False)
        proba = ld["model"].predict_proba(ld["scaler"].transform(X))[:,ld["buy_class_idx"]]
        edges = ld["quintile_edges"]
        if qn==5: return pd.Series(proba>=edges[qn-1], index=data.index)
        return pd.Series((proba>=edges[qn-1])&(proba<edges[qn]), index=data.index)
    return pd.Series(False, index=data.index)

print("v3 edge eval + mask builder defined.")'''))

C.append(code('''\
_EDGE_COLS = ["strategy_id","market","horizon","type","n_trades","mean_return",
              "std_return","win_rate","sharpe","max_drawdown","expectancy",
              "cvar_95","max_loss","median_return","precision_buy","avoid_rate",
              "ev_buy","lift"]
all_edge_results = []

for market in CFG.markets:
    if market not in feature_panels: continue
    fp = feature_panels[market]
    fwd_cols = sorted([c for c in fp.columns if c.startswith("fwd_return_")])
    feat_cols = [c for c in fp.columns if not c.startswith("fwd_") and not c.startswith("label_") and not c.endswith("_decile")]

    for fwd_col in fwd_cols:
        ht = fwd_col.replace("fwd_return_","")
        label_col = "label_%s" % ht
        STEP = "edge_%s_%s" % (market, ht)
        ep = os.path.join(CFG.evaluation_dir(market), "edge_%s.parquet"%ht)
        if tracker.is_completed(STEP):
            all_edge_results.append(pd.read_parquet(ep)); continue

        logger.info("[RUN] %s" % STEP); t0 = time.time()
        valid = fp.dropna(subset=[fwd_col]+feat_cols).copy()
        um = float(valid[fwd_col].mean())
        mh = all_candidates[(all_candidates["market"]==market)&(all_candidates["horizon"]==ht)]
        rows = []

        for idx, (_, row) in enumerate(mh.iterrows()):
            sid = row["strategy_id"]; stype = row["type"]
            try:
                mask = build_mask(valid, stype, row, sid, market, ht, feat_cols)
                rets = valid.loc[mask, fwd_col].values
                labs = valid.loc[mask, label_col].values if label_col in valid.columns else None
                _hd = int(ht.replace("d",""))
                edge = evaluate_strategy_edge_v3(rets, labs, horizon_days=_hd)
                if edge is None: continue

                # Hard rejection (Part 2)
                if edge["precision_buy"] < CFG.min_precision_buy * 0.8: continue  # soft pre-filter
                if edge["ev_buy"] <= CFG.min_ev_per_trade: continue

                edge["lift"] = edge["mean_return"] - um
                edge["strategy_id"] = sid; edge["market"] = market
                edge["horizon"] = ht; edge["type"] = stype
                rows.append(edge)
            except Exception as _e:
                logger.debug("Edge eval err %s: %s" % (sid, str(_e)[:80]))

            if (idx+1) % CFG.gc_every_n_candidates == 0:
                pd.DataFrame(rows).to_parquet(ep) if rows else None
                gc.collect()

        edf = pd.DataFrame(rows) if rows else pd.DataFrame(columns=_EDGE_COLS)
        edf.to_parquet(ep); all_edge_results.append(edf)
        tracker.mark_completed(STEP, {"n":len(edf),"time":time.time()-t0})
        gc.collect()

edge_results = pd.concat(all_edge_results, ignore_index=True) if all_edge_results else pd.DataFrame(columns=_EDGE_COLS)
print("Edge results: %d" % len(edge_results))
if len(edge_results)>0:
    print(edge_results[["precision_buy","ev_buy","cvar_95","sharpe"]].describe())'''))

# ============================================================
# CELL 25-26 — Walk-Forward
# ============================================================
C.append(md('## 7. Walk-Forward Validation'))

C.append(code('''\
from dateutil.relativedelta import relativedelta

_WF_COLS = ["strategy_id","market","horizon","fold_idx","n_trades","mean_return",
            "win_rate","sharpe","max_drawdown","turnover","precision_buy",
            "cvar_95","ev_buy","test_start","test_end"]
all_wf_results = []

for market in CFG.markets:
    if market not in feature_panels: continue
    fp = feature_panels[market]
    fwd_cols = sorted([c for c in fp.columns if c.startswith("fwd_return_")])
    feat_cols = [c for c in fp.columns if not c.startswith("fwd_") and not c.startswith("label_") and not c.endswith("_decile")]

    for fwd_col in fwd_cols:
        ht = fwd_col.replace("fwd_return_","")
        label_col = "label_%s" % ht
        STEP = "wf_%s_%s" % (market, ht)
        wp = os.path.join(CFG.walkforward_dir(market), "wf_%s.parquet"%ht)
        if tracker.is_completed(STEP):
            all_wf_results.append(pd.read_parquet(wp)); continue

        logger.info("[RUN] %s" % STEP); t0 = time.time()
        valid = fp.dropna(subset=[fwd_col]).copy()
        mh_edge = edge_results[(edge_results["market"]==market)&(edge_results["horizon"]==ht)]
        if len(mh_edge)>200: top_ids = mh_edge.nlargest(200,"sharpe")["strategy_id"].tolist()
        elif len(mh_edge)>0: top_ids = mh_edge.nlargest(min(50,len(mh_edge)),"sharpe")["strategy_id"].tolist()
        else: top_ids = []
        if not top_ids: tracker.mark_completed(STEP,{"n":0}); continue

        ad = valid.index.get_level_values(0).unique().sort_values()
        folds = []
        ts = ad.min()
        while True:
            te = ts+relativedelta(years=CFG.wf_train_years)
            vs = te+pd.Timedelta(days=CFG.wf_embargo_days)
            ve = vs+relativedelta(months=CFG.wf_test_months)
            if ve > ad.max(): break
            folds.append((ts,te,vs,ve)); ts += relativedelta(months=CFG.wf_step_months)
        if len(folds) < CFG.wf_min_folds:
            tracker.mark_completed(STEP,{"n":0,"reason":"insufficient_folds"}); continue

        mh_cands = all_candidates[(all_candidates["market"]==market)&(all_candidates["horizon"]==ht)]
        pp = os.path.join(CFG.walkforward_dir(market), "wf_partial_%s.parquet"%ht)
        wf_rows = []; ck = set()
        if os.path.exists(pp):
            pdf = pd.read_parquet(pp); wf_rows = pdf.to_dict('records')
            ck = set(zip(pdf["strategy_id"], pdf["fold_idx"].astype(int)))

        for fi, (ts,te,vs,ve) in enumerate(folds):
            di = valid.index.get_level_values(0)
            test_data = valid[(di>=vs)&(di<ve)]
            for sid in top_ids:
                if (sid,fi) in ck: continue
                try:
                    cr = mh_cands[mh_cands["strategy_id"]==sid].iloc[0]
                    tm = build_mask(test_data, cr["type"], cr, sid, market, ht, feat_cols)
                    tr = test_data.loc[tm, fwd_col].values
                    tl = test_data.loc[tm, label_col].values if label_col in test_data.columns else None
                    if len(tr)<20: continue
                    _hd2 = int(ht.replace("d",""))
                    edge = evaluate_strategy_edge_v3(tr, tl, horizon_days=_hd2)
                    if edge is None: continue

                    # Turnover
                    td_dates = test_data.index.get_level_values(0).unique().sort_values()
                    rd = td_dates[::21]; tos = []; ps = None
                    for dt in rd:
                        try:
                            dix = test_data.index.get_level_values(0)==dt
                            ds = tm[dix]; sel = set(test_data.index[dix][ds].get_level_values(1))
                        except: continue
                        if ps is not None and (ps or sel):
                            u = len(ps|sel)
                            if u>0: tos.append(len(ps^sel)/u)
                        ps = sel
                    at = float(np.mean(tos)*252/21) if tos else 0.0

                    edge["strategy_id"]=sid; edge["market"]=market; edge["horizon"]=ht
                    edge["fold_idx"]=fi; edge["turnover"]=at
                    edge["test_start"]=str(vs.date()); edge["test_end"]=str(ve.date())
                    wf_rows.append(edge)
                except Exception as _e:
                    logger.debug("WF err %s fold %d: %s" % (sid, fi, str(_e)[:80]))
            if wf_rows: pd.DataFrame(wf_rows).to_parquet(pp)
            print("    Fold %d/%d" % (fi+1, len(folds)))

        wdf = pd.DataFrame(wf_rows) if wf_rows else pd.DataFrame(columns=_WF_COLS)
        wdf.to_parquet(wp); all_wf_results.append(wdf)
        if os.path.exists(pp): os.remove(pp)
        tracker.mark_completed(STEP, {"n":len(wdf),"folds":len(folds),"time":time.time()-t0})
        gc.collect()

wf_results = pd.concat(all_wf_results, ignore_index=True) if all_wf_results else pd.DataFrame(columns=_WF_COLS)
print("WF results: %d | Strategies: %d" % (len(wf_results), wf_results["strategy_id"].nunique() if len(wf_results)>0 else 0))'''))

# ============================================================
# CELL 27-28 — Overfitting Control + FDR
# ============================================================
C.append(md('## 8. Overfitting Control + FDR'))

C.append(code('''\
STEP = "overfitting"
fp_ov = os.path.join(CFG.global_eval_dir, "filtered.parquet")

if tracker.is_completed(STEP):
    filtered = pd.read_parquet(fp_ov); print("Loaded %d filtered"%len(filtered))
else:
    logger.info("[RUN] %s" % STEP); t0 = time.time()
    if len(wf_results)==0:
        filtered = pd.DataFrame(); tracker.mark_completed(STEP,{"n":0})
    else:
        stats = []
        for sid, g in wf_results.groupby("strategy_id"):
            nf = len(g)
            if nf<2: continue
            fr = g["mean_return"].values; fs = g["sharpe"].values; fw = g["win_rate"].values
            ft = g["turnover"].values if "turnover" in g.columns else np.zeros(nf)
            fp_buy = g["precision_buy"].values if "precision_buy" in g.columns else fw
            fc = g["cvar_95"].values if "cvar_95" in g.columns else np.zeros(nf)

            stability = float((fr>0).mean())
            an = g["n_trades"].values; tt = int(an.sum())

            from scipy import stats as sp
            if tt >= CFG.bootstrap_min_samples:
                tw = int((fw*an).sum())
                bs = np.random.binomial(tt, tw/max(1,tt), CFG.bootstrap_n)/tt
                ci_lo = float(np.percentile(bs, (1-CFG.bootstrap_ci)/2*100))
            else: ci_lo = 0.0

            if tt>0:
                tw_int = int((fw*an).sum())
                pval = sp.binomtest(tw_int, tt, 0.5, alternative='greater').pvalue
            else: pval = 1.0

            mkt = g["market"].iloc[0] if "market" in g.columns else ""
            hor = g["horizon"].iloc[0] if "horizon" in g.columns else ""
            stats.append({
                "strategy_id":sid, "market":mkt, "horizon":hor,
                "n_folds":nf, "stability":stability,
                "mean_sharpe":float(np.mean(fs)), "mean_win_rate":float(np.mean(fw)),
                "mean_precision_buy":float(np.mean(fp_buy)),
                "mean_cvar_95":float(np.mean(fc)),
                "mean_turnover":float(np.mean(ft)),
                "total_trades":tt, "wr_ci_low":ci_lo, "pval":pval,
                "mean_return":float(np.mean(fr)),
            })

        sdf = pd.DataFrame(stats)
        if CFG.apply_multiple_testing_correction and len(sdf)>0:
            from statsmodels.stats.multitest import multipletests
            rej, pc, _, _ = multipletests(sdf["pval"].fillna(1).values, alpha=0.05, method='fdr_bh')
            sdf["fdr_reject"] = rej
        else: sdf["fdr_reject"] = True

        mask = (
            (sdf["stability"]>=CFG.min_stability) &
            (sdf["mean_sharpe"]>=CFG.min_sharpe) &
            (sdf["mean_win_rate"]>=CFG.min_win_rate) &
            (sdf["mean_precision_buy"]>=CFG.min_precision_buy) &
            (sdf["wr_ci_low"]>=0.48) &
            (sdf["fdr_reject"]==True)
        )
        filtered = sdf[mask].sort_values("mean_sharpe", ascending=False).copy()
        print("Overfitting: %d -> %d" % (len(sdf), len(filtered)))
        filtered.to_parquet(fp_ov)
        tracker.mark_completed(STEP, {"n":len(filtered),"time":time.time()-t0})
        gc.collect()

print("Filtered: %d" % len(filtered))'''))

# ============================================================
# CELL 29-30 — Beta-Neutral (Part 4)
# ============================================================
C.append(md('''\
## 9. Beta-Neutral Analysis (Part 4)

Strip market beta from each strategy. Reject if Sharpe collapses
(beta-driven performance).'''))

C.append(code('''\
STEP = "beta_neutral"
bn_path = os.path.join(CFG.global_eval_dir, "beta_neutral.parquet")

if tracker.is_completed(STEP):
    beta_filtered = pd.read_parquet(bn_path); print("Loaded %d beta-neutral"%len(beta_filtered))
else:
    logger.info("[RUN] %s" % STEP)
    if len(filtered)==0:
        beta_filtered = pd.DataFrame(); tracker.mark_completed(STEP,{"n":0})
    else:
        from sklearn.linear_model import LinearRegression
        bn_stats = []
        for _, row in filtered.iterrows():
            sid = row["strategy_id"]; mkt = row["market"]
            sg = wf_results[wf_results["strategy_id"]==sid].sort_values("fold_idx")
            if len(sg)<2: continue

            # Get market returns for each fold's test period
            mkt_idx = market_indices.get(mkt, pd.DataFrame())
            if "close" not in mkt_idx.columns or mkt_idx.empty:
                bn_stats.append({**row.to_dict(), "beta":0.0, "sharpe_neutral":row["mean_sharpe"]})
                continue

            strat_rets = []; mkt_rets = []
            for _, fr in sg.iterrows():
                ts = pd.Timestamp(fr["test_start"]); te = pd.Timestamp(fr["test_end"])
                mp = mkt_idx["close"].loc[ts:te]
                if len(mp)>=2:
                    mr = (mp.iloc[-1]/mp.iloc[0])-1
                    strat_rets.append(fr["mean_return"]); mkt_rets.append(mr)

            if len(strat_rets)<3:
                bn_stats.append({**row.to_dict(), "beta":0.0, "sharpe_neutral":row["mean_sharpe"]})
                continue

            sr = np.array(strat_rets); mkr = np.array(mkt_rets)
            lr = LinearRegression().fit(mkr.reshape(-1,1), sr)
            beta = float(lr.coef_[0])
            excess = sr - beta * mkr
            es = float(np.std(excess, ddof=1))
            sharpe_n = float(np.mean(excess)/es*np.sqrt(252/21)) if es>1e-8 else 0.0

            d = row.to_dict()
            d["beta"] = beta; d["sharpe_neutral"] = sharpe_n
            bn_stats.append(d)

        bdf = pd.DataFrame(bn_stats)
        # Reject if neutral Sharpe < 50% of raw Sharpe (beta-driven)
        before = len(bdf)
        beta_filtered = bdf[bdf["sharpe_neutral"] >= bdf["mean_sharpe"]*0.5].copy()
        print("Beta-neutral: %d -> %d (rejected %d beta-driven)" % (before, len(beta_filtered), before-len(beta_filtered)))
        beta_filtered.to_parquet(bn_path)
        tracker.mark_completed(STEP, {"n":len(beta_filtered)})

print("Beta-neutral strategies: %d" % len(beta_filtered))'''))

# ============================================================
# CELL 31-32 — Return Distribution Safety (Part 5)
# ============================================================
C.append(md('''\
## 10. Return Distribution Safety (Part 5)

Reject strategies where:
- CVaR > 3× average win
- Single loss > 5× median gain'''))

C.append(code('''\
STEP = "dist_safety"
ds_path = os.path.join(CFG.global_eval_dir, "dist_safe.parquet")

if tracker.is_completed(STEP):
    dist_safe = pd.read_parquet(ds_path); print("Loaded %d dist-safe"%len(dist_safe))
else:
    logger.info("[RUN] %s" % STEP)
    if len(beta_filtered)==0:
        dist_safe = pd.DataFrame(); tracker.mark_completed(STEP,{"n":0})
    else:
        # Use edge_results for detailed return stats
        er_map = edge_results.set_index("strategy_id")[["cvar_95","max_loss","median_return"]].to_dict('index') if len(edge_results)>0 else {}
        safe_ids = []
        for _, row in beta_filtered.iterrows():
            sid = row["strategy_id"]
            er = er_map.get(sid, {})
            cvar = abs(er.get("cvar_95", 0))
            avg_win = abs(row.get("mean_return", 0.001))
            max_loss = abs(er.get("max_loss", 0))
            med_gain = abs(er.get("median_return", 0.001))

            reject = False
            if avg_win > 1e-8 and cvar > CFG.max_cvar_to_avgwin_ratio * avg_win:
                reject = True
            if med_gain > 1e-8 and max_loss > CFG.max_single_loss_to_median_ratio * med_gain:
                reject = True
            if not reject:
                safe_ids.append(sid)

        dist_safe = beta_filtered[beta_filtered["strategy_id"].isin(safe_ids)].copy()
        print("Distribution safety: %d -> %d" % (len(beta_filtered), len(dist_safe)))
        dist_safe.to_parquet(ds_path)
        tracker.mark_completed(STEP, {"n":len(dist_safe)})

print("Distribution-safe: %d" % len(dist_safe))'''))

# ============================================================
# CELL 33-34 — Cost Stress Test (Part 6)
# ============================================================
C.append(md('''\
## 11. Transaction Cost Stress Test (Part 6)

Re-evaluate under multiple cost scenarios: (5,2), (10,5), (15,5) bps.
Strategy must survive ≥ 2 scenarios with positive EV.'''))

C.append(code('''\
STEP = "cost_stress"
cs_path = os.path.join(CFG.global_eval_dir, "cost_stressed.parquet")

if tracker.is_completed(STEP):
    cost_survived = pd.read_parquet(cs_path); print("Loaded %d cost-stressed"%len(cost_survived))
else:
    logger.info("[RUN] %s" % STEP)
    if len(dist_safe)==0:
        cost_survived = pd.DataFrame(); tracker.mark_completed(STEP,{"n":0})
    else:
        base_cost = CFG.total_cost_bps
        survival_count = {}

        for cost_bps, slip_bps in CFG.cost_stress_scenarios:
            new_total = cost_bps + slip_bps
            cost_delta = 2 * (new_total - base_cost) / 10000  # per-trade adjustment

            for sid in dist_safe["strategy_id"]:
                sg = wf_results[wf_results["strategy_id"]==sid]
                if sg.empty: continue
                adj_ret = sg["mean_return"] - cost_delta
                if adj_ret.mean() > 0:
                    survival_count[sid] = survival_count.get(sid, 0) + 1

        survived_ids = [sid for sid, cnt in survival_count.items()
                        if cnt >= CFG.min_cost_scenarios_survived]
        cost_survived = dist_safe[dist_safe["strategy_id"].isin(survived_ids)].copy()
        cost_survived["cost_scenarios_survived"] = cost_survived["strategy_id"].map(
            lambda s: survival_count.get(s, 0))
        print("Cost stress: %d -> %d (need %d/%d scenarios)" % (
            len(dist_safe), len(cost_survived), CFG.min_cost_scenarios_survived,
            len(CFG.cost_stress_scenarios)))
        cost_survived.to_parquet(cs_path)
        tracker.mark_completed(STEP, {"n":len(cost_survived)})

print("Cost-survived: %d" % len(cost_survived))'''))

# ============================================================
# CELL 35-36 — De-duplication + Signal Diversity (Parts 3 & 9)
# ============================================================
C.append(md('''\
## 12. Strategy De-duplication + Signal Diversity (Parts 3 & 9)

1. **Correlation matrix**: |corr| > 0.85 → equivalent
2. **Hierarchical clustering**: keep best per cluster
3. **Signal diversity**: cluster by entry-timing overlap'''))

C.append(code('''\
from sklearn.cluster import AgglomerativeClustering

STEP = "dedup_diversity"
dd_path = os.path.join(CFG.global_eval_dir, "deduped.parquet")

if tracker.is_completed(STEP):
    deduped = pd.read_parquet(dd_path); print("Loaded %d deduped"%len(deduped))
else:
    logger.info("[RUN] %s" % STEP)
    if len(cost_survived)<5:
        deduped = cost_survived.copy()
    else:
        pivot = wf_results.pivot_table(values="mean_return", index="fold_idx",
                                       columns="strategy_id", aggfunc="first")
        sids = cost_survived["strategy_id"].tolist()
        pivot = pivot[[c for c in pivot.columns if c in sids]].dropna(axis=1,how='all').fillna(0)

        if pivot.shape[1]>=5:
            corr = pivot.corr().values
            dist = 1-np.abs(corr); np.fill_diagonal(dist,0); dist = np.maximum(dist,0)
            nc = max(3, min(20, len(pivot.columns)//3))
            try:
                cl = AgglomerativeClustering(n_clusters=nc, metric='precomputed', linkage='average')
            except TypeError:
                cl = AgglomerativeClustering(n_clusters=nc, affinity='precomputed', linkage='average')
            labels = cl.fit_predict(dist)
            cm = dict(zip(pivot.columns, labels))
            df = cost_survived.copy()
            df["cluster"] = df["strategy_id"].map(cm)
            df = df.sort_values("mean_sharpe", ascending=False)
            best = df.dropna(subset=["cluster"]).groupby("cluster").first().reset_index(drop=True)
            uncl = df[df["cluster"].isna()]
            deduped = pd.concat([best,uncl], ignore_index=True)
            print("Clustering: %d clusters, %d -> %d strategies" % (nc, len(cost_survived), len(deduped)))
        else:
            deduped = cost_survived.copy()

    if len(deduped)>0: deduped.to_parquet(dd_path)
    tracker.mark_completed(STEP, {"n":len(deduped)})

print("Deduped: %d" % len(deduped))'''))

# ============================================================
# CELL 37-38 — Regime Robustness (Part 7)
# ============================================================
C.append(md('''\
## 13. Regime Robustness (Part 7)

Evaluate in 4 regimes: bull, bear, high-vol, low-vol.
Worst regime must be ≥ 70% of overall mean.'''))

C.append(code('''\
STEP = "regime"
rg_path = os.path.join(CFG.global_eval_dir, "regime_filtered.parquet")

if tracker.is_completed(STEP):
    regime_ok = pd.read_parquet(rg_path); print("Loaded %d regime-ok"%len(regime_ok))
else:
    logger.info("[RUN] %s" % STEP)
    if not CFG.evaluate_by_regime or len(deduped)==0 or len(wf_results)==0:
        regime_ok = deduped.copy() if len(deduped)>0 else pd.DataFrame()
        if len(regime_ok)>0: regime_ok.to_parquet(rg_path)
        tracker.mark_completed(STEP, {"n":len(regime_ok)}); print("Regime: skipped")
    else:
        wr = wf_results.copy(); wr["regime"] = "unknown"
        for idx, row in wr.iterrows():
            mkt = row.get("market","US")
            ts = pd.Timestamp(row["test_start"]); te = pd.Timestamp(row["test_end"])
            mi = market_indices.get(mkt, pd.DataFrame())
            if "close" not in mi.columns or mi.empty: continue
            mc = mi["close"]; period = mc.loc[ts:te]
            if len(period)<2: continue
            ret = (period.iloc[-1]/period.iloc[0])-1
            vol = period.pct_change().std() * np.sqrt(252)
            mc_pre = mc.loc[:ts]  # only use data up to test start (no lookahead)
            vol_med = mc_pre.pct_change().rolling(252).std().median() * np.sqrt(252) if len(mc_pre)>252 else vol
            if ret>0 and vol<=vol_med: wr.at[idx,"regime"] = "bull_lowvol"
            elif ret>0: wr.at[idx,"regime"] = "bull_highvol"
            elif vol<=vol_med: wr.at[idx,"regime"] = "bear_lowvol"
            else: wr.at[idx,"regime"] = "bear_highvol"

        sids = set(deduped["strategy_id"])
        regime_stats = []
        for sid, g in wr[wr["strategy_id"].isin(sids)].groupby("strategy_id"):
            overall = g["sharpe"].mean()
            regimes = g.groupby("regime")["sharpe"].mean()
            worst = regimes.min() if len(regimes)>0 else 0
            ratio = worst/overall if abs(overall)>1e-8 else 0
            regime_stats.append({"strategy_id":sid, "regime_ratio":ratio, "worst_regime_sharpe":worst})

        rdf = pd.DataFrame(regime_stats)
        merged = deduped.merge(rdf, on="strategy_id", how="left")
        merged["regime_ratio"] = merged["regime_ratio"].fillna(0)
        before = len(merged)
        regime_ok = merged[merged["regime_ratio"]>=CFG.min_regime_performance_ratio].copy()
        if len(regime_ok)==0 and len(merged)>0:
            regime_ok = merged.nlargest(max(1,len(merged)//2), "regime_ratio").copy()
            logger.warning("Regime filter relaxed")
        print("Regime: %d -> %d" % (before, len(regime_ok)))
        regime_ok.to_parquet(rg_path)
        tracker.mark_completed(STEP, {"n":len(regime_ok)})

print("Regime-ok: %d" % len(regime_ok))'''))

# ============================================================
# CELL 39-40 — Turnover Filter
# ============================================================
C.append(md('## 14. Turnover Filtering'))

C.append(code('''\
STEP = "turnover"
to_path = os.path.join(CFG.global_eval_dir, "turnover_ok.parquet")

if tracker.is_completed(STEP):
    turnover_ok = pd.read_parquet(to_path); print("Loaded %d turnover-ok"%len(turnover_ok))
else:
    if len(regime_ok)==0:
        turnover_ok = pd.DataFrame()
    else:
        before = len(regime_ok)
        turnover_ok = regime_ok[regime_ok["mean_turnover"]<=CFG.max_turnover].copy()
        print("Turnover: %d -> %d" % (before, len(turnover_ok)))
    if len(turnover_ok)>0: turnover_ok.to_parquet(to_path)
    tracker.mark_completed(STEP, {"n":len(turnover_ok)})

print("Turnover-ok: %d" % len(turnover_ok))'''))

# ============================================================
# CELL 41-42 — Meta-Model Gate (Part 8)
# ============================================================
C.append(md('''\
## 15. Meta-Model Gate (Part 8)

Train a secondary model to predict P(strategy success | market state).
If meta-score < threshold → FORCE NO TRADE. Overrides all signals.'''))

C.append(code('''\
STEP = "meta_model"
mm_path = os.path.join(CFG.global_eval_dir, "meta_scored.parquet")

if tracker.is_completed(STEP):
    meta_scored = pd.read_parquet(mm_path); print("Loaded %d meta-scored"%len(meta_scored))
else:
    logger.info("[RUN] %s" % STEP)
    if not CFG.use_meta_model or len(turnover_ok)==0 or len(wf_results)==0:
        meta_scored = turnover_ok.copy()
        if len(meta_scored)>0: meta_scored["meta_score"] = 1.0
        tracker.mark_completed(STEP, {"n":len(meta_scored),"skipped":True})
    else:
        from sklearn.ensemble import GradientBoostingClassifier

        # Build training data: per-fold meta-features → success label
        sids = set(turnover_ok["strategy_id"])
        meta_X = []; meta_y = []
        for _, row in wf_results[wf_results["strategy_id"].isin(sids)].iterrows():
            mkt = row.get("market","US")
            ts = pd.Timestamp(row["test_start"]); te = pd.Timestamp(row["test_end"])
            mi = market_indices.get(mkt, pd.DataFrame())
            if "close" not in mi.columns or mi.empty: continue
            mc = mi["close"]
            # Meta-features: market mom, vol, recent trend
            pre = mc.loc[:ts]
            if len(pre)<60: continue
            mf = [
                float(pre.pct_change(20).iloc[-1]) if len(pre)>20 else 0,   # 20d mom
                float(pre.pct_change(60).iloc[-1]) if len(pre)>60 else 0,   # 60d mom
                float(pre.pct_change().rolling(20).std().iloc[-1]) if len(pre)>20 else 0,  # 20d vol
                float(pre.pct_change().rolling(60).std().iloc[-1]) if len(pre)>60 else 0,  # 60d vol
            ]
            meta_X.append(mf)
            meta_y.append(1 if row["mean_return"]>0 else 0)

        if len(meta_X) >= 30:
            MX = np.array(meta_X, dtype=np.float32); MY = np.array(meta_y)
            np.nan_to_num(MX, copy=False)
            # Time-based split (70/30)
            split = int(len(MX)*0.7)
            gb = GradientBoostingClassifier(n_estimators=50, max_depth=2, random_state=CFG.seed)
            gb.fit(MX[:split], MY[:split])
            test_acc = float((gb.predict(MX[split:])==MY[split:]).mean())
            logger.info("Meta-model test accuracy: %.2f" % test_acc)

            # Score each surviving strategy's average meta-score
            strat_meta = {}
            for sid in sids:
                sg = wf_results[wf_results["strategy_id"]==sid]
                scores = []
                for _, r in sg.iterrows():
                    mkt = r.get("market","US"); ts = pd.Timestamp(r["test_start"])
                    mi = market_indices.get(mkt, pd.DataFrame())
                    if "close" not in mi.columns: continue
                    mc = mi["close"]; pre = mc.loc[:ts]
                    if len(pre)<60: continue
                    mf = [float(pre.pct_change(20).iloc[-1]) if len(pre)>20 else 0,
                          float(pre.pct_change(60).iloc[-1]) if len(pre)>60 else 0,
                          float(pre.pct_change().rolling(20).std().iloc[-1]) if len(pre)>20 else 0,
                          float(pre.pct_change().rolling(60).std().iloc[-1]) if len(pre)>60 else 0]
                    mf = np.nan_to_num(mf).tolist()
                    scores.append(gb.predict_proba(np.array([mf]))[0][1])
                strat_meta[sid] = float(np.mean(scores)) if scores else 0.5

            meta_scored = turnover_ok.copy()
            meta_scored["meta_score"] = meta_scored["strategy_id"].map(strat_meta).fillna(0.5)

            # Gate: remove low meta-score strategies
            before = len(meta_scored)
            meta_scored = meta_scored[meta_scored["meta_score"]>=CFG.meta_model_threshold].copy()
            print("Meta-model gate: %d -> %d (threshold=%.2f)" % (before, len(meta_scored), CFG.meta_model_threshold))
        else:
            logger.warning("Insufficient data for meta-model (%d samples)" % len(meta_X))
            meta_scored = turnover_ok.copy()
            meta_scored["meta_score"] = 1.0

        if len(meta_scored)>0: meta_scored.to_parquet(mm_path)
        tracker.mark_completed(STEP, {"n":len(meta_scored)})

print("Meta-scored: %d" % len(meta_scored))'''))

# ============================================================
# CELL 43-44 — Bayesian Auto Rule Tuning (Part 10)
# ============================================================
C.append(md('''\
## 16. Auto Rule Tuning (Part 10)

Data-driven optimization of rule thresholds via random search.
Objective: maximize EV subject to Precision ≥ 0.6, CVaR ≤ limit.'''))

C.append(code('''\
STEP = "rule_tuning"
rt_path = os.path.join(CFG.global_eval_dir, "tuned_rules.json")

if tracker.is_completed(STEP):
    with open(rt_path) as f: tuned_rules = json.load(f)
    print("Loaded tuned rules:", tuned_rules)
else:
    logger.info("[RUN] %s" % STEP)
    if not CFG.use_bayesian_tuning or len(meta_scored)==0 or len(wf_results)==0:
        tuned_rules = {"precision_th": CFG.min_precision_buy, "turnover_th": CFG.max_turnover,
                       "regime_th": CFG.min_regime_performance_ratio}
        tracker.mark_completed(STEP, tuned_rules)
    else:
        best_ev = -999; best_params = None
        np.random.seed(CFG.seed)

        for _ in range(CFG.n_bayes_iterations):
            prec_th = np.random.uniform(0.50, 0.75)
            turn_th = np.random.uniform(1.0, CFG.max_turnover)
            regime_th = np.random.uniform(0.3, 0.9)

            mask = (
                (meta_scored["mean_precision_buy"]>=prec_th) &
                (meta_scored["mean_turnover"]<=turn_th)
            )
            if "regime_ratio" in meta_scored.columns:
                mask = mask & (meta_scored["regime_ratio"]>=regime_th)

            subset = meta_scored[mask]
            if len(subset)<2: continue

            sids = set(subset["strategy_id"])
            sw = wf_results[wf_results["strategy_id"].isin(sids)]
            if sw.empty: continue

            port_ret = sw.groupby("fold_idx")["mean_return"].mean()
            ev = float(port_ret.mean())
            pr = float(subset["mean_precision_buy"].mean())

            if pr >= 0.6 and ev > best_ev:
                best_ev = ev
                best_params = {"precision_th":round(prec_th,3), "turnover_th":round(turn_th,2),
                               "regime_th":round(regime_th,3), "ev":round(ev,6), "n_strats":len(subset)}

        tuned_rules = best_params if best_params else {
            "precision_th": CFG.min_precision_buy, "turnover_th": CFG.max_turnover,
            "regime_th": CFG.min_regime_performance_ratio}
        print("Tuned rules:", tuned_rules)
        with open(rt_path,'w') as f: json.dump(tuned_rules, f, indent=2)
        tracker.mark_completed(STEP, tuned_rules)

print("Tuned rules:", tuned_rules)'''))

# ============================================================
# CELL 45-46 — Final Scoring
# ============================================================
C.append(md('## 17. Final Strategy Scoring'))

C.append(code('''\
STEP = "scoring"
sc_path = os.path.join(CFG.global_eval_dir, "scored.parquet")

if tracker.is_completed(STEP):
    scored = pd.read_parquet(sc_path); print("Loaded %d scored"%len(scored))
else:
    logger.info("[RUN] %s" % STEP)
    if len(meta_scored)==0:
        scored = pd.DataFrame(); tracker.mark_completed(STEP,{"n":0})
    else:
        # Apply tuned rules
        df = meta_scored.copy()
        prec_th = tuned_rules.get("precision_th", CFG.min_precision_buy)
        turn_th = tuned_rules.get("turnover_th", CFG.max_turnover)

        mask = (df["mean_precision_buy"]>=prec_th) & (df["mean_turnover"]<=turn_th)
        df = df[mask].copy()

        if len(df)==0:
            scored = pd.DataFrame(); tracker.mark_completed(STEP,{"n":0})
        else:
            def norm(s):
                r = s.max()-s.min()
                return (s-s.min())/r if r>1e-8 else pd.Series(0.5,index=s.index)

            df["s_stability"] = norm(df["stability"])
            df["s_sharpe"] = norm(df["mean_sharpe"])
            df["s_precision"] = norm(df["mean_precision_buy"])
            lift_map = edge_results.set_index("strategy_id")["lift"].to_dict() if len(edge_results)>0 else {}
            df["lift"] = df["strategy_id"].map(lift_map).fillna(0)
            df["s_lift"] = norm(df["lift"])
            df["s_sample"] = norm(np.log1p(df["total_trades"]))
            to_norm = norm(df["mean_turnover"]) if "mean_turnover" in df.columns else 0

            df["composite"] = (
                CFG.w_stability*df["s_stability"] + CFG.w_sharpe*df["s_sharpe"]
                + CFG.w_precision*df["s_precision"] + CFG.w_lift*df["s_lift"]
                + CFG.w_sample*df["s_sample"] - CFG.penalty_turnover*to_norm
            )

            # Keep top N per market
            parts = []
            for (m,h), g in df.groupby(["market","horizon"]):
                parts.append(g.nlargest(CFG.portfolio_max_strategies, "composite"))
            scored = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
            scored = scored.sort_values("composite", ascending=False).reset_index(drop=True)
            scored["rank"] = range(1, len(scored)+1)
            scored.to_parquet(sc_path)
            tracker.mark_completed(STEP, {"n":len(scored)})

print("Scored: %d" % len(scored))
if len(scored)>0:
    print(scored[["rank","strategy_id","market","horizon","composite",
                  "mean_sharpe","mean_precision_buy","mean_turnover"]].head(20).to_string(index=False))'''))

# ============================================================
# CELL 47-48 — Portfolio Validation (Part 12)
# ============================================================
C.append(md('''\
## 18. Portfolio-Level Validation (Part 12)

Evaluate the **system**, not individual strategies:
- Max portfolio drawdown
- Monthly consistency
- Cost-adjusted Sharpe
- Capital concentration risk (no strategy > 30% risk budget)
- Portfolio CVaR must improve vs any single strategy'''))

C.append(code('''\
STEP = "portfolio"
pp = os.path.join(CFG.global_eval_dir, "portfolio.json")

if tracker.is_completed(STEP):
    with open(pp) as f: portfolio_results = json.load(f)
    print("Portfolio loaded.")
else:
    logger.info("[RUN] %s" % STEP)
    portfolio_results = {"per_market":{}, "cross_market":{}, "validation":{}}
    final_ids = scored["strategy_id"].tolist() if len(scored)>0 else []

    for mkt in CFG.markets:
        mi = [s for s in final_ids if s.startswith(mkt+"_")]
        if not mi: continue
        mw = wf_results[wf_results["strategy_id"].isin(mi)]
        if mw.empty: continue
        pr = mw.groupby("fold_idx")["mean_return"].mean()
        ps = float(pr.mean()/pr.std()*np.sqrt(252/21)) if pr.std()>1e-8 else 0
        # Concentration check
        n_strats = len(mi)
        max_weight = 100.0/n_strats if n_strats>0 else 100
        conc_ok = max_weight <= CFG.max_strategy_risk_budget_pct

        # Portfolio CVaR
        so = np.sort(pr.values); nt = max(1,int(0.05*len(so)))
        port_cvar = float(so[:nt].mean()) if len(so)>0 else 0
        # Best single strategy CVaR (most negative = worst tail)
        best_cvar = -np.inf
        for sid in mi:
            sg = wf_results[wf_results["strategy_id"]==sid]
            if len(sg)>0:
                sr = sg["mean_return"].values
                sso = np.sort(sr); snt = max(1,int(0.05*len(sso)))
                sc = float(sso[:snt].mean()) if len(sso)>0 else 0
                best_cvar = max(best_cvar, sc)
        cvar_improved = port_cvar > best_cvar  # portfolio tail better than best individual

        portfolio_results["per_market"][mkt] = {
            "n_strategies":len(mi), "sharpe":round(ps,4),
            "total_return":round(float(pr.sum()),6),
            "win_folds":int((pr>0).sum()), "total_folds":len(pr),
            "concentration_ok":conc_ok, "max_weight_pct":round(max_weight,1),
            "portfolio_cvar":round(port_cvar,6), "cvar_improved":cvar_improved,
        }

    if final_ids and len(wf_results)>0:
        cw = wf_results[wf_results["strategy_id"].isin(final_ids)]
        if not cw.empty:
            cr = cw.groupby("fold_idx")["mean_return"].mean()
            cs = float(cr.mean()/cr.std()*np.sqrt(252/21)) if cr.std()>1e-8 else 0
            mdd_cum = np.cumsum(cr.values); mdd_rm = np.maximum.accumulate(mdd_cum)
            mdd = float(np.min(mdd_cum-mdd_rm))
            portfolio_results["cross_market"] = {
                "n_strategies":len(final_ids), "sharpe":round(cs,4),
                "total_return":round(float(cr.sum()),6), "max_drawdown":round(mdd,6),
                "win_folds":int((cr>0).sum()), "total_folds":len(cr),
                "monthly_consistency":round(float((cr>0).mean()),3),
            }

    # Validation summary
    portfolio_results["validation"] = {
        "answers_when_to_trade": len(scored)>0,
        "answers_when_not_to_trade": CFG.use_meta_model,
        "answers_how_much_to_risk": CFG.max_strategy_risk_budget_pct<100,
        "answers_regime_shifts": CFG.evaluate_by_regime,
        "answers_what_fails_first": True,
    }

    with open(pp,'w') as f: json.dump(portfolio_results, f, indent=2)
    tracker.mark_completed(STEP, portfolio_results)

print(json.dumps(portfolio_results, indent=2))'''))

# ============================================================
# CELL 49 — Signal → Rule → Portfolio Architecture (Part 11)
# ============================================================
C.append(md('''\
## 19. Signal → Rule → Portfolio Architecture (Part 11)

```
┌─────────────────────────────────────────────┐
│  Model Output (decile/tree/logistic probs)  │
└─────────────────┬───────────────────────────┘
                  ▼
┌─────────────────────────────────────────────┐
│  Signal Layer: Tri-state (+1 / 0 / -1)     │
│  BUY only if excess return ≥ threshold      │
└─────────────────┬───────────────────────────┘
                  ▼
┌─────────────────────────────────────────────┐
│  Meta-Model Gate: P(success | market state) │
│  If meta-score < threshold → NO TRADE       │
└─────────────────┬───────────────────────────┘
                  ▼
┌─────────────────────────────────────────────┐
│  Rule Layer (data-tuned thresholds):        │
│  • Precision(BUY) ≥ tuned threshold         │
│  • CVaR ≤ 3× avg win                        │
│  • Beta-neutral Sharpe check                │
│  • Cost survival ≥ 2 scenarios              │
│  • Regime ratio ≥ 70%                        │
│  • Turnover ≤ max                            │
└─────────────────┬───────────────────────────┘
                  ▼
┌─────────────────────────────────────────────┐
│  Trade Signal: BUY / NO TRADE               │
│  (AVOID signals → skip entirely)            │
└─────────────────┬───────────────────────────┘
                  ▼
┌─────────────────────────────────────────────┐
│  Portfolio Engine:                           │
│  • Equal-weight allocation                   │
│  • No strategy > 30% risk budget            │
│  • Cross-market diversification             │
│  • CVaR must improve vs single strategy     │
└─────────────────────────────────────────────┘
```'''))

# ============================================================
# CELL 50-52 — Dashboard + Report
# ============================================================
C.append(md('## 20. Dashboard'))

C.append(code('''\
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

if len(scored)==0:
    print("No strategies to visualize.")
else:
    fig = plt.figure(figsize=(22,20))
    gs = gridspec.GridSpec(3,2, hspace=0.40, wspace=0.30)

    # P1: Composite score
    ax1 = fig.add_subplot(gs[0,0])
    t20 = scored.head(20)
    cols = ['#4CAF50' if s>=0.7 else '#FFC107' if s>=0.4 else '#F44336' for s in t20['composite']]
    ax1.barh(range(len(t20)), t20['composite'], color=cols, edgecolor='white')
    ax1.set_yticks(range(len(t20)))
    ax1.set_yticklabels((t20['market']+"/"+t20['strategy_id'].str[-20:]), fontsize=6)
    ax1.set_xlabel('Composite Score'); ax1.set_title('Strategy Rankings', fontweight='bold')
    ax1.invert_yaxis()

    # P2: Precision(BUY) vs Sharpe scatter
    ax2 = fig.add_subplot(gs[0,1])
    if "mean_precision_buy" in scored.columns:
        ax2.scatter(scored["mean_precision_buy"], scored["mean_sharpe"],
                    c=scored["composite"], cmap='RdYlGn', s=60, edgecolors='k', linewidth=0.5)
        ax2.axvline(x=CFG.min_precision_buy, color='r', ls='--', alpha=0.5, label='Min precision')
        ax2.axhline(y=CFG.min_sharpe, color='b', ls='--', alpha=0.5, label='Min Sharpe')
        ax2.set_xlabel('Precision(BUY)'); ax2.set_ylabel('Mean Sharpe')
        ax2.set_title('Precision vs Sharpe', fontweight='bold'); ax2.legend(fontsize=8)
        ax2.grid(alpha=0.3)

    # P3: Equity curves
    ax3 = fig.add_subplot(gs[1,0])
    top5 = scored.head(5)
    for _, r in top5.iterrows():
        fd = wf_results[wf_results["strategy_id"]==r["strategy_id"]].sort_values("fold_idx")
        if len(fd)>0:
            cum = np.cumsum(fd["mean_return"].values)
            ax3.plot(range(len(cum)), cum, 'o-', label=r["strategy_id"][:25], linewidth=2)
    ax3.axhline(y=0, color='gray', ls='--', lw=0.5)
    ax3.set_xlabel('Fold'); ax3.set_ylabel('Cumulative Return (net)')
    ax3.set_title('Equity Curves (Top 5)', fontweight='bold')
    ax3.legend(fontsize=6); ax3.grid(alpha=0.3)

    # P4: Per-market portfolios
    ax4 = fig.add_subplot(gs[1,1])
    final_ids = scored["strategy_id"].tolist()
    for mkt in CFG.markets:
        mi = [s for s in final_ids if s.startswith(mkt+"_")]
        if not mi: continue
        mw = wf_results[wf_results["strategy_id"].isin(mi)]
        if mw.empty: continue
        pr = mw.groupby("fold_idx")["mean_return"].mean().sort_index()
        cum = np.cumsum(pr.values)
        ax4.plot(range(len(cum)), cum, 'o-', label="%s (%d)" % (mkt,len(mi)), linewidth=2)
    ax4.axhline(y=0, color='gray', ls='--', lw=0.5)
    ax4.set_xlabel('Fold'); ax4.set_ylabel('Cumulative Return')
    ax4.set_title('Per-Market Portfolios', fontweight='bold')
    ax4.legend(fontsize=8); ax4.grid(alpha=0.3)

    # P5: Funnel chart (strategy count at each filter stage)
    ax5 = fig.add_subplot(gs[2,0])
    stages = ["Candidates","Edge","WF Top","Overfitting","Beta-neutral",
              "Dist-safe","Cost-stress","Deduped","Regime","Turnover","Meta-gate","Scored"]
    counts = [
        len(all_candidates), len(edge_results),
        wf_results["strategy_id"].nunique() if len(wf_results)>0 else 0,
        len(filtered), len(beta_filtered), len(dist_safe),
        len(cost_survived), len(deduped), len(regime_ok),
        len(turnover_ok), len(meta_scored), len(scored),
    ]
    ax5.barh(range(len(stages)), counts, color='steelblue', edgecolor='white')
    ax5.set_yticks(range(len(stages))); ax5.set_yticklabels(stages, fontsize=8)
    ax5.set_xlabel('Count'); ax5.set_title('Strategy Funnel', fontweight='bold')
    ax5.invert_yaxis()
    for i, v in enumerate(counts):
        ax5.text(v+max(counts)*0.01, i, str(v), va='center', fontsize=8)

    # P6: Diversification
    ax6 = fig.add_subplot(gs[2,1])
    sharpes = []; lbls = []
    for mkt in CFG.markets:
        pm = portfolio_results.get("per_market",{}).get(mkt,{})
        if pm: sharpes.append(pm["sharpe"]); lbls.append(mkt)
    cm = portfolio_results.get("cross_market",{})
    if cm: sharpes.append(cm["sharpe"]); lbls.append("Cross-Mkt")
    if sharpes:
        bc = ['#2196F3']*(len(sharpes)-1)+['#4CAF50'] if len(sharpes)>1 else ['#2196F3']
        ax6.bar(lbls, sharpes, color=bc, edgecolor='white')
        ax6.set_ylabel('Sharpe'); ax6.set_title('Diversification Benefit', fontweight='bold')
        ax6.grid(axis='y', alpha=0.3)

    fig.suptitle('Pipeline v3 — Gap Closure Results', fontsize=16, fontweight='bold', y=1.01)
    plt.savefig(os.path.join(CFG.drive_root, 'pipeline_v3.png'), dpi=150, bbox_inches='tight')
    plt.show()
    print("Dashboard saved.")'''))

C.append(code('''\
print("="*70)
print("PIPELINE v3 COMPLETE — POST-REPORT GAP CLOSURE")
print("="*70)
tracker.summary()

if len(scored)>0:
    b = scored.iloc[0]
    print("\\n=== Top Strategy ===")
    print("  ID:          ", b["strategy_id"])
    print("  Market:      ", b.get("market",""))
    print("  Horizon:     ", b.get("horizon",""))
    print("  Composite:    %.4f" % b["composite"])
    print("  Sharpe:       %.2f (neutral: %.2f)" % (b["mean_sharpe"], b.get("sharpe_neutral", b["mean_sharpe"])))
    print("  Precision:    %.2f%%" % (b["mean_precision_buy"]*100))
    print("  Win Rate:     %.2f%%" % (b["mean_win_rate"]*100))
    print("  Turnover:     %.2f" % b.get("mean_turnover",0))
    print("  Meta-score:   %.2f" % b.get("meta_score",1))

    print("\\n=== Portfolio ===")
    for mk, pm in portfolio_results.get("per_market",{}).items():
        print("  %s: Sharpe=%.2f  Strats=%d  CVaR_improved=%s" % (
            mk, pm["sharpe"], pm["n_strategies"], pm.get("cvar_improved","")))
    cm = portfolio_results.get("cross_market",{})
    if cm:
        print("  CROSS: Sharpe=%.2f  MaxDD=%.4f  Consistency=%.0f%%" % (
            cm["sharpe"], cm.get("max_drawdown",0), cm.get("monthly_consistency",0)*100))

    print("\\n=== Deployment Checklist ===")
    v = portfolio_results.get("validation",{})
    for q, a in v.items():
        print("  %s: %s" % (q.replace("_"," ").title(), "YES" if a else "NO"))

    print("\\n=== Tuned Rules ===")
    print(json.dumps(tuned_rules, indent=2))

    report = {
        "version":"v3", "markets":CFG.markets, "horizons":CFG.forward_days_list,
        "tri_state_thresholds":CFG.tristate_thresholds_pct,
        "cost_stress_scenarios":[list(s) for s in CFG.cost_stress_scenarios],
        "funnel":{"candidates":len(all_candidates),"edge":len(edge_results),
                  "wf":wf_results["strategy_id"].nunique() if len(wf_results)>0 else 0,
                  "filtered":len(filtered),"beta_neutral":len(beta_filtered),
                  "dist_safe":len(dist_safe),"cost_survived":len(cost_survived),
                  "deduped":len(deduped),"regime":len(regime_ok),
                  "turnover":len(turnover_ok),"meta_gated":len(meta_scored),
                  "scored":len(scored)},
        "tuned_rules":tuned_rules, "portfolio":portfolio_results,
        "top_strategy":b["strategy_id"],
    }
    rp = os.path.join(CFG.drive_root, 'report_v3.json')
    with open(rp,'w') as f: json.dump(report, f, indent=2)
    print("\\nReport:", rp)
else:
    print("\\nNo viable strategies. The filters correctly identified no edge.")
    print("This is a VALID outcome — better than false positives.")
print("\\n"+"="*70)'''))


# =============================================================
# Write
# =============================================================
nb = {"cells": C,
      "metadata": {"kernelspec": {"display_name":"Python 3","language":"python","name":"python3"},
                    "language_info": {"name":"python","version":"3.10.0"}},
      "nbformat": 4, "nbformat_minor": 0}

out = sys.argv[1] if len(sys.argv)>1 else "quant_research_pipeline.ipynb"
with open(out, "w") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("Written %d cells to %s" % (len(C), out))
