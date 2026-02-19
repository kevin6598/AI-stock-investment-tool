#!/usr/bin/env python3
"""Generate v4 multi-strategy survivability pipeline with failure-aware intelligence."""
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
# Multi-Market Quantitative Research Pipeline — v4
## Multi-Strategy Survivability & Failure-Aware Intelligence

**V4 enhancements** over v3.1: orthogonal strategy classes, portfolio diversification,
failure-triggered allocation, market-specific adaptation, and governance outputs.

| # | Enhancement | Purpose |
|---|-------------|---------|
| 1-14 | All v3.1 layers | Tri-state, conditional metrics, de-dup, beta-neutral, etc. |
| 15 | Mean reversion strategies | Oversold z-score, Bollinger bands |
| 16 | Volatility/shock strategies | Vol expansion, acceleration |
| 17 | Cross-market rotation | Relative momentum/volatility |
| 18 | Portfolio diversity optimizer | Greedy forward selection |
| 19 | Meta-model v2 (30-dim) | Interaction effects, marginal CVaR |
| 20 | Failure-triggered allocation | Core decay → capital shift |
| 21 | Market-specific thresholds | KOSPI/KOSDAQ adaptation |
| 22 | Trading governance report | WHEN/WHEN NOT/RISK/TRUST |
| 23 | Feedback loop | Cross-run learning |

**Target:** Multi-strategy portfolio, Sharpe 1.0-1.8, structural diversification.'''))

# ============================================================
# CELL 1 — Pip install
# ============================================================
C.append(code('''\
import os

_RESTART_FLAG = "/tmp/_quant_v3_restarted"

if not os.path.exists(_RESTART_FLAG):
    # First run: install everything, then force restart
    os.system("pip install -q --upgrade numpy pandas scipy scikit-learn")
    os.system("pip install -q yfinance matplotlib pyarrow joblib pykrx finance-datareader statsmodels")
    open(_RESTART_FLAG, "w").write("done")
    print("Packages installed. Restarting runtime to pick up new binaries...")
    os.kill(os.getpid(), 9)  # force kernel restart
else:
    print("Packages already installed, runtime already restarted. Continuing...")'''))

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
    """Central configuration — v4 with 39 parameter groups."""

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

    # --- 21. Data Module Flags ---
    enable_liquidity_features: bool = True
    enable_fundamental_features: bool = True
    enable_macro_features: bool = True
    enable_sector_features: bool = True
    enable_sentiment_proxy: bool = False  # optional, off by default

    # --- 22. Axis A: Liquidity ---
    liquidity_dollar_vol_windows: List[int] = field(default_factory=lambda: [20, 60])
    amihud_window: int = 60

    # --- 23. Axis C: Macro ---
    macro_tickers: Dict[str, str] = field(default_factory=lambda: {
        "tnx": "^TNX", "irx": "^IRX", "vix": "^VIX",
        "gold": "GC=F", "oil": "CL=F", "dxy": "DX-Y.NYB",
    })
    macro_veto_vix_multiple: float = 2.0

    # --- 24. Axis B: Fundamentals ---
    fundamental_fields: List[str] = field(default_factory=lambda: [
        "marketCap", "trailingPE", "priceToBook",
        "returnOnEquity", "returnOnAssets", "revenueGrowth", "earningsGrowth",
    ])

    # --- 25. Sector-Neutral Labeling ---
    enable_sector_neutral_labels: bool = True

    # --- 26. Sector Rotation ---
    enable_sector_rotation: bool = True
    sector_rotation_min_constituents: int = 3
    sector_alpha_threshold: float = 0.0

    # --- 27. Theme Crash Early Warning ---
    enable_theme_crash: bool = True
    theme_crash_veto_threshold: float = 0.7
    theme_crash_corr_window: int = 60
    theme_crash_volume_window: int = 20

    # --- 28. Trade Abstention ---
    enable_trade_abstention: bool = True
    trade_abstention_min_ev: float = 0.001

    # --- 29. Cross-Market Consistency ---
    enable_cross_market_consistency: bool = True
    cross_market_contagion_lag_days: int = 5

    # --- 30. Regime-Conditional Thresholds ---
    enable_regime_adaptive_thresholds: bool = True

    # --- 31. Strategy Mortality ---
    enable_strategy_mortality: bool = True
    mortality_rolling_sharpe_window: int = 4
    mortality_sharpe_kill_threshold: float = 0.0
    mortality_consecutive_negative_folds: int = 3

    # --- 32. Strategy Types (v4) ---
    enable_mean_reversion: bool = True
    enable_volatility_shock: bool = True
    enable_cross_market_rotation: bool = True
    enable_sn_candidates: bool = True

    # --- 33. Mean Reversion ---
    mr_zscore_windows: List[int] = field(default_factory=lambda: [10, 20, 40])
    mr_holding_days: List[int] = field(default_factory=lambda: [5, 10])
    mr_zscore_entry_threshold: float = 2.0
    mr_vol_overshoot_multiple: float = 1.5

    # --- 34. Volatility/Shock ---
    vs_activation_ew_threshold: float = 0.6
    vs_vol_expansion_windows: List[int] = field(default_factory=lambda: [5, 10, 20])
    vs_holding_days: List[int] = field(default_factory=lambda: [5, 21])
    vs_vol_spike_multiple: float = 2.0

    # --- 35. Cross-Market Rotation ---
    cmr_rebalance_days: int = 21
    cmr_lookback_windows: List[int] = field(default_factory=lambda: [20, 60])

    # --- 36. Market-Specific Thresholds ---
    market_thresholds: Dict = field(default_factory=lambda: {
        "US":     {"min_precision_buy": 0.60, "min_sharpe": 0.5, "max_turnover": 12.0, "min_stability": 0.5, "min_win_rate": 0.52},
        "KOSPI":  {"min_precision_buy": 0.55, "min_sharpe": 0.4, "max_turnover": 15.0, "min_stability": 0.4, "min_win_rate": 0.50},
        "KOSDAQ": {"min_precision_buy": 0.55, "min_sharpe": 0.3, "max_turnover": 18.0, "min_stability": 0.4, "min_win_rate": 0.50},
    })

    # --- 37. Meta-Model v2 ---
    meta_v2_interaction_effects: bool = True
    meta_v2_signal_diversity_penalty: float = 0.3
    meta_v2_use_scipy_optimize: bool = True
    meta_v2_bayes_n_calls: int = 50
    meta_v2_marginal_cvar_weight: float = 0.2

    # --- 38. Failure-Aware Allocation ---
    enable_failure_allocation: bool = True
    failure_confidence_threshold: float = 0.4
    failure_capital_shift_pct: float = 30.0
    failure_decay_lookback_folds: int = 3

    # --- 39. Portfolio Diversification ---
    enable_diversity_optimizer: bool = True
    diversity_min_strategies: int = 3
    diversity_max_correlation: float = 0.6
    diversity_reward_opposite_convexity: float = 0.2

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

def get_market_threshold(market, param, default=None):
    """Get market-specific threshold, falling back to global CFG value."""
    mkt_th = CFG.market_thresholds.get(market, {})
    return mkt_th.get(param, getattr(CFG, param, default))

print("Config v4 created.  Root:", CFG.drive_root)
print("Markets:", CFG.markets, "| Horizons:", CFG.forward_days_list)
print("Tri-state thresholds:", CFG.tristate_thresholds_pct)
print("Cost stress scenarios:", CFG.cost_stress_scenarios)
print("V4 strategy types: MeanRev=%s VolShock=%s CMR=%s SN=%s" % (
    CFG.enable_mean_reversion, CFG.enable_volatility_shock,
    CFG.enable_cross_market_rotation, CFG.enable_sn_candidates))'''))

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
# CELL — Fundamental + Sector Data Loading (Axis B + D)
# ============================================================
C.append(md('''\
## 3a. Fundamental & Sector Data (Axes B + D)

**Axis B** — Fundamentals: log_market_cap, PE, PB, ROE from yfinance / pykrx (cached).
**Axis D** — Cross-Sectional: sector-relative 20d returns.'''))

C.append(code('''\
fundamental_cache = {}  # {market: {ticker: {field: value}}}
sector_mean_returns = {}  # {market: {sector: mean_20d_return}}

US_SECTOR_MAP = {
    # Technology (39)
    "AAPL":"Technology","MSFT":"Technology","GOOGL":"Technology","NVDA":"Technology",
    "META":"Technology","ADBE":"Technology","CRM":"Technology","INTC":"Technology",
    "CSCO":"Technology","AVGO":"Technology","ACN":"Technology","QCOM":"Technology",
    "TXN":"Technology","AMD":"Technology","PYPL":"Technology","ORCL":"Technology",
    "IBM":"Technology","NOW":"Technology","INTU":"Technology","AMAT":"Technology",
    "MU":"Technology","LRCX":"Technology","ADI":"Technology","KLAC":"Technology",
    "SNPS":"Technology","CDNS":"Technology","MCHP":"Technology","FTNT":"Technology",
    "PANW":"Technology","CRWD":"Technology","WDAY":"Technology","TEAM":"Technology",
    "ADSK":"Technology","ANSS":"Technology","NXPI":"Technology","MRVL":"Technology",
    "ON":"Technology","GEN":"Technology","MPWR":"Technology",
    # CommunicationServices (24)
    "GOOG":"CommunicationServices","DIS":"CommunicationServices",
    "NFLX":"CommunicationServices","CMCSA":"CommunicationServices",
    "VZ":"CommunicationServices","T":"CommunicationServices",
    "TMUS":"CommunicationServices","CHTR":"CommunicationServices",
    "EA":"CommunicationServices","TTWO":"CommunicationServices",
    "WBD":"CommunicationServices","PARA":"CommunicationServices",
    "LYV":"CommunicationServices","MTCH":"CommunicationServices",
    "IPG":"CommunicationServices","OMC":"CommunicationServices",
    "NWSA":"CommunicationServices","NWS":"CommunicationServices",
    "FOXA":"CommunicationServices","FOX":"CommunicationServices",
    "LUMN":"CommunicationServices","PINS":"CommunicationServices",
    "SNAP":"CommunicationServices","RBLX":"CommunicationServices",
    # Financials (38)
    "BRK-B":"Financials","JPM":"Financials","V":"Financials","MA":"Financials",
    "BAC":"Financials","WFC":"Financials","GS":"Financials","MS":"Financials",
    "SCHW":"Financials","C":"Financials","BLK":"Financials","SPGI":"Financials",
    "AXP":"Financials","CB":"Financials","MMC":"Financials","PGR":"Financials",
    "AON":"Financials","CME":"Financials","ICE":"Financials","MET":"Financials",
    "AIG":"Financials","TRV":"Financials","ALL":"Financials","AFL":"Financials",
    "PRU":"Financials","USB":"Financials","PNC":"Financials","TFC":"Financials",
    "BK":"Financials","STT":"Financials","FITB":"Financials","HBAN":"Financials",
    "CFG":"Financials","KEY":"Financials","RF":"Financials","NTRS":"Financials",
    "CINF":"Financials","FDS":"Financials",
    # Healthcare (40)
    "JNJ":"Healthcare","UNH":"Healthcare","ABT":"Healthcare","MRK":"Healthcare",
    "LLY":"Healthcare","BMY":"Healthcare","TMO":"Healthcare","DHR":"Healthcare",
    "PFE":"Healthcare","ABBV":"Healthcare","AMGN":"Healthcare","GILD":"Healthcare",
    "ISRG":"Healthcare","MDT":"Healthcare","SYK":"Healthcare","BDX":"Healthcare",
    "CI":"Healthcare","CVS":"Healthcare","HUM":"Healthcare","ELV":"Healthcare",
    "ZTS":"Healthcare","VRTX":"Healthcare","REGN":"Healthcare","BSX":"Healthcare",
    "EW":"Healthcare","IDXX":"Healthcare","IQV":"Healthcare","MTD":"Healthcare",
    "A":"Healthcare","DXCM":"Healthcare","ALGN":"Healthcare","BAX":"Healthcare",
    "HOLX":"Healthcare","RMD":"Healthcare","WST":"Healthcare","MOH":"Healthcare",
    "CNC":"Healthcare","BIIB":"Healthcare","MRNA":"Healthcare","ILMN":"Healthcare",
    # Industrials (40)
    "UPS":"Industrials","RTX":"Industrials","HON":"Industrials","CAT":"Industrials",
    "DE":"Industrials","GE":"Industrials","BA":"Industrials","LMT":"Industrials",
    "NOC":"Industrials","GD":"Industrials","MMM":"Industrials","EMR":"Industrials",
    "ETN":"Industrials","ITW":"Industrials","WM":"Industrials","RSG":"Industrials",
    "CSX":"Industrials","NSC":"Industrials","UNP":"Industrials","FDX":"Industrials",
    "DAL":"Industrials","LUV":"Industrials","UAL":"Industrials","AAL":"Industrials",
    "CARR":"Industrials","OTIS":"Industrials","PCAR":"Industrials","CMI":"Industrials",
    "ROK":"Industrials","FAST":"Industrials","CTAS":"Industrials","PAYX":"Industrials",
    "SWK":"Industrials","IR":"Industrials","TT":"Industrials","VRSK":"Industrials",
    "CPRT":"Industrials","WAB":"Industrials","PWR":"Industrials","AME":"Industrials",
    # ConsumerDiscretionary (29)
    "AMZN":"ConsumerDiscretionary","TSLA":"ConsumerDiscretionary",
    "HD":"ConsumerDiscretionary","NKE":"ConsumerDiscretionary",
    "MCD":"ConsumerDiscretionary","SBUX":"ConsumerDiscretionary",
    "LOW":"ConsumerDiscretionary","TJX":"ConsumerDiscretionary",
    "BKNG":"ConsumerDiscretionary","CMG":"ConsumerDiscretionary",
    "MAR":"ConsumerDiscretionary","HLT":"ConsumerDiscretionary",
    "ORLY":"ConsumerDiscretionary","AZO":"ConsumerDiscretionary",
    "ROST":"ConsumerDiscretionary","DHI":"ConsumerDiscretionary",
    "LEN":"ConsumerDiscretionary","PHM":"ConsumerDiscretionary",
    "NVR":"ConsumerDiscretionary","GRMN":"ConsumerDiscretionary",
    "POOL":"ConsumerDiscretionary","BBY":"ConsumerDiscretionary",
    "EBAY":"ConsumerDiscretionary","APTV":"ConsumerDiscretionary",
    "GM":"ConsumerDiscretionary","F":"ConsumerDiscretionary",
    "YUM":"ConsumerDiscretionary","DPZ":"ConsumerDiscretionary",
    "ULTA":"ConsumerDiscretionary",
    # ConsumerStaples (24)
    "PG":"ConsumerStaples","KO":"ConsumerStaples","PEP":"ConsumerStaples",
    "COST":"ConsumerStaples","WMT":"ConsumerStaples","PM":"ConsumerStaples",
    "MO":"ConsumerStaples","CL":"ConsumerStaples","MDLZ":"ConsumerStaples",
    "KMB":"ConsumerStaples","GIS":"ConsumerStaples","K":"ConsumerStaples",
    "HSY":"ConsumerStaples","SJM":"ConsumerStaples","MKC":"ConsumerStaples",
    "CPB":"ConsumerStaples","HRL":"ConsumerStaples","SYY":"ConsumerStaples",
    "ADM":"ConsumerStaples","STZ":"ConsumerStaples","BF-B":"ConsumerStaples",
    "TAP":"ConsumerStaples","TSN":"ConsumerStaples","KHC":"ConsumerStaples",
    # Energy (25)
    "XOM":"Energy","CVX":"Energy","COP":"Energy","EOG":"Energy","SLB":"Energy",
    "MPC":"Energy","PSX":"Energy","VLO":"Energy","OXY":"Energy","PXD":"Energy",
    "DVN":"Energy","HES":"Energy","HAL":"Energy","BKR":"Energy","FANG":"Energy",
    "MRO":"Energy","APA":"Energy","CTRA":"Energy","OKE":"Energy","WMB":"Energy",
    "KMI":"Energy","ET":"Energy","TRGP":"Energy","LNG":"Energy","EQT":"Energy",
    # Utilities (14)
    "NEE":"Utilities","DUK":"Utilities","SO":"Utilities","D":"Utilities",
    "AEP":"Utilities","SRE":"Utilities","XEL":"Utilities","EXC":"Utilities",
    "ED":"Utilities","WEC":"Utilities","ES":"Utilities","AWK":"Utilities",
    "AEE":"Utilities","CMS":"Utilities",
    # Materials (19)
    "LIN":"Materials","APD":"Materials","SHW":"Materials","ECL":"Materials",
    "NEM":"Materials","FCX":"Materials","NUE":"Materials","DD":"Materials",
    "DOW":"Materials","PPG":"Materials","VMC":"Materials","MLM":"Materials",
    "ALB":"Materials","CF":"Materials","MOS":"Materials","IFF":"Materials",
    "CE":"Materials","EMN":"Materials","PKG":"Materials",
    # RealEstate (14)
    "AMT":"RealEstate","PLD":"RealEstate","CCI":"RealEstate","EQIX":"RealEstate",
    "PSA":"RealEstate","SPG":"RealEstate","O":"RealEstate","DLR":"RealEstate",
    "WELL":"RealEstate","AVB":"RealEstate","EQR":"RealEstate","VTR":"RealEstate",
    "ARE":"RealEstate","MAA":"RealEstate",
}

for market in CFG.markets:
    if market not in ohlcv_data:
        continue
    STEP = "fundamental_data_%s" % market
    fund_path = os.path.join(CFG.data_dir(market), "fundamentals.parquet")

    _fund_loaded_from_cache = False
    if tracker.is_completed(STEP) and os.path.exists(fund_path):
        logger.info("[SKIP] %s" % STEP)
        _fdf = pd.read_parquet(fund_path)
        fundamental_cache[market] = {
            tk: _fdf.loc[tk].to_dict() for tk in _fdf.index if tk in _fdf.index
        }
        _fund_loaded_from_cache = True

    if not _fund_loaded_from_cache and not CFG.enable_fundamental_features:
        fundamental_cache[market] = {}
        tracker.mark_completed(STEP, {"skipped": True})
        continue

    if not _fund_loaded_from_cache:
        logger.info("[RUN] %s" % STEP)
        t0 = time.time()

    reg = MARKET_REGISTRY[market]
    panel = ohlcv_data[market]
    tickers = panel.index.get_level_values(1).unique().tolist()
    fund_rows = fundamental_cache.get(market, {}) if _fund_loaded_from_cache else {}

    if not _fund_loaded_from_cache:
        if reg["data_source"] == "yfinance":
            for i, tk in enumerate(tickers):
                if (i + 1) % 10 == 0:
                    print("  Fundamental [%d/%d] %s" % (i + 1, len(tickers), tk))
                try:
                    info = yf.Ticker(tk).info
                    row = {}
                    for fld in CFG.fundamental_fields:
                        v = info.get(fld)
                        row[fld] = float(v) if v is not None else np.nan
                    row["sector"] = info.get("sector", US_SECTOR_MAP.get(tk, "Other"))
                    fund_rows[tk] = row
                except Exception as _e:
                    logger.debug("Fund fetch fail %s: %s" % (tk, str(_e)[:60]))
                    fund_rows[tk] = {fld: np.nan for fld in CFG.fundamental_fields}
                    fund_rows[tk]["sector"] = US_SECTOR_MAP.get(tk, "Other")
                time.sleep(0.1)  # rate-limit
        elif reg["data_source"] == "pykrx":
            try:
                from pykrx import stock as _pks
                import datetime as _dt
                ds = _dt.datetime.now().strftime("%Y%m%d")
                fund_df = _pks.get_market_fundamental_by_ticker(ds, market=reg.get("pykrx_market", market))
                if fund_df is not None and not fund_df.empty:
                    col_map = {}
                    for c in fund_df.columns:
                        cl = c.strip()
                        if cl in ("PER", "trailingPE"):
                            col_map[c] = "trailingPE"
                        elif cl in ("PBR", "priceToBook"):
                            col_map[c] = "priceToBook"
                        elif cl in ("DIV", "dividendYield"):
                            col_map[c] = "dividendYield"
                        elif cl in ("EPS",):
                            col_map[c] = "EPS"
                    fund_df = fund_df.rename(columns=col_map)
                    for tk in tickers:
                        if tk in fund_df.index:
                            row = {}
                            for fld in CFG.fundamental_fields:
                                v = fund_df.loc[tk].get(fld, np.nan)
                                row[fld] = float(v) if v is not None and not (isinstance(v, float) and np.isnan(v)) else np.nan
                            row["sector"] = "KR_" + market
                            fund_rows[tk] = row
                        else:
                            fund_rows[tk] = {fld: np.nan for fld in CFG.fundamental_fields}
                            fund_rows[tk]["sector"] = "KR_" + market
            except Exception as _e:
                logger.warning("pykrx fundamental fetch failed: %s" % str(_e)[:80])
                for tk in tickers:
                    fund_rows[tk] = {fld: np.nan for fld in CFG.fundamental_fields}
                    fund_rows[tk]["sector"] = "KR_" + market

        fundamental_cache[market] = fund_rows
        if fund_rows:
            fdf = pd.DataFrame.from_dict(fund_rows, orient="index")
            fdf.to_parquet(fund_path)

    # Compute time-series sector median returns for Axis D + sector-neutral labeling (runs always)
    if CFG.enable_sector_features and fund_rows:
        ticker_sectors = {tk: fund_rows.get(tk, {}).get("sector", "Other") for tk in tickers}

        # Build per-ticker 20d return series
        ticker_r20 = {}
        for tk in tickers:
            try:
                td = panel.loc[(slice(None), tk), :].droplevel(1)
                ticker_r20[tk] = td["close"].pct_change(20)
            except Exception:
                pass

        # Cross-sectional median per sector per date (rolling 20d return)
        if ticker_r20:
            r20_df = pd.DataFrame(ticker_r20)
            sector_cols = {}
            for tk, sec in ticker_sectors.items():
                sector_cols.setdefault(sec, []).append(tk)
            ts_median = pd.DataFrame(index=r20_df.index)
            for sec, cols in sector_cols.items():
                valid = [c for c in cols if c in r20_df.columns]
                if valid:
                    ts_median[sec] = r20_df[valid].median(axis=1)

            # Also compute sector median forward returns per horizon for SN labeling
            sector_median_fwd = {}
            for fd in CFG.forward_days_list:
                ticker_fwd = {}
                for tk in tickers:
                    try:
                        td = panel.loc[(slice(None), tk), :].droplevel(1)
                        raw_fwd = td["close"].pct_change(fd).shift(-fd)
                        net_fwd = raw_fwd - 2 * CFG.total_cost_bps / 10000
                        ticker_fwd[tk] = net_fwd
                    except Exception:
                        pass
                if ticker_fwd:
                    fwd_df = pd.DataFrame(ticker_fwd)
                    med_fwd = pd.DataFrame(index=fwd_df.index)
                    for sec, cols in sector_cols.items():
                        valid = [c for c in cols if c in fwd_df.columns]
                        if valid:
                            med_fwd[sec] = fwd_df[valid].median(axis=1)
                    sector_median_fwd[fd] = med_fwd

            sector_mean_returns[market] = {
                "ts": ts_median,
                "ticker_sectors": ticker_sectors,
                "sector_median_fwd": sector_median_fwd,
            }
        else:
            sector_mean_returns[market] = {
                "ts": pd.DataFrame(),
                "ticker_sectors": ticker_sectors,
                "sector_median_fwd": {},
            }

    if not _fund_loaded_from_cache:
        elapsed = time.time() - t0
        tracker.mark_completed(STEP, {"n_tickers": len(fund_rows), "time": elapsed})
        print("%s fundamentals: %d tickers (%.0fs)" % (market, len(fund_rows), elapsed))

print("Fundamental cache:", {m: len(v) for m, v in fundamental_cache.items()})
print("Sector data:", {m: len(v.get("ticker_sectors", {})) if isinstance(v, dict) else 0 for m, v in sector_mean_returns.items()})'''))

# ============================================================
# CELL — Macro Data Loading (Axis C)
# ============================================================
C.append(md('''\
## 3b. Macro Data (Axis C)

Download 6 macro series: Treasury yields, VIX, Gold, Oil, DXY.
Derived features: yield_curve_slope, vix_regime, commodity/DXY momentum.
Used in **meta-model** and **macro veto** only (Tier 2 — no candidate generation).'''))

C.append(code('''\
macro_data = pd.DataFrame()
STEP = "macro_data"
macro_path = os.path.join(CFG.drive_root, "data", "macro.parquet")

if tracker.is_completed(STEP) and os.path.exists(macro_path):
    logger.info("[SKIP] %s" % STEP)
    macro_data = pd.read_parquet(macro_path)
elif not CFG.enable_macro_features:
    logger.info("[SKIP] macro (disabled)")
    tracker.mark_completed(STEP, {"skipped": True})
else:
    logger.info("[RUN] %s" % STEP)
    t0 = time.time()
    raw_macro = {}
    for name, ticker in CFG.macro_tickers.items():
        try:
            df = yf.download(ticker, period=CFG.data_period, progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if not df.empty and "Close" in df.columns:
                s = df["Close"].copy()
                s.index = pd.to_datetime(s.index).tz_localize(None)
                raw_macro[name] = s
                logger.info("Macro %s: %d rows" % (name, len(s)))
        except Exception as _e:
            logger.warning("Macro download fail %s: %s" % (name, str(_e)[:60]))

    if raw_macro:
        macro_data = pd.DataFrame(raw_macro)
        macro_data = macro_data.sort_index().ffill()

        # Derived features
        if "tnx" in macro_data.columns and "irx" in macro_data.columns:
            macro_data["yield_curve_slope"] = macro_data["tnx"] - macro_data["irx"]
            macro_data["yield_curve_inverted"] = (macro_data["yield_curve_slope"] < 0).astype(float)
        else:
            macro_data["yield_curve_slope"] = 0.0
            macro_data["yield_curve_inverted"] = 0.0

        if "vix" in macro_data.columns:
            vix_median = macro_data["vix"].rolling(252, min_periods=60).median()
            macro_data["vix_regime"] = macro_data["vix"] / vix_median.replace(0, np.nan)
        else:
            macro_data["vix_regime"] = 1.0

        for asset in ["dxy", "gold", "oil"]:
            if asset in macro_data.columns:
                macro_data["%s_mom_60d" % asset] = macro_data[asset].pct_change(60)
            else:
                macro_data["%s_mom_60d" % asset] = 0.0

        macro_data.to_parquet(macro_path)

    elapsed = time.time() - t0
    tracker.mark_completed(STEP, {"n_series": len(raw_macro), "time": elapsed})

print("Macro data:", macro_data.shape if len(macro_data) > 0 else "empty")
if len(macro_data) > 0:
    print("  Columns:", list(macro_data.columns))
    print("  Latest yield_curve_slope:", macro_data["yield_curve_slope"].iloc[-1] if "yield_curve_slope" in macro_data.columns else "N/A")
    print("  Latest vix_regime:", macro_data["vix_regime"].iloc[-1] if "vix_regime" in macro_data.columns else "N/A")'''))

# ============================================================
# CELL — Sentiment Proxy (Axis E, optional)
# ============================================================
C.append(md('''\
## 3c. Sentiment Proxy (Axis E — Optional)

VIX term structure = VIX / VIX3M − 1.
Positive = backwardation (fear). Used in **meta-model** only (Tier 2).'''))

C.append(code('''\
sentiment_data = pd.DataFrame()
STEP = "sentiment_proxy_data"

if tracker.is_completed(STEP):
    sp = os.path.join(CFG.drive_root, "data", "sentiment_proxy.parquet")
    if os.path.exists(sp):
        sentiment_data = pd.read_parquet(sp)
    logger.info("[SKIP] %s" % STEP)
elif not CFG.enable_sentiment_proxy:
    logger.info("[SKIP] sentiment proxy (disabled)")
    tracker.mark_completed(STEP, {"skipped": True})
else:
    logger.info("[RUN] %s" % STEP)
    t0 = time.time()
    try:
        vix3m = yf.download("^VIX3M", period=CFG.data_period, progress=False, auto_adjust=True)
        if isinstance(vix3m.columns, pd.MultiIndex):
            vix3m.columns = vix3m.columns.get_level_values(0)
        if not vix3m.empty and "Close" in vix3m.columns:
            vix3m_close = vix3m["Close"].copy()
            vix3m_close.index = pd.to_datetime(vix3m_close.index).tz_localize(None)

            if len(macro_data) > 0 and "vix" in macro_data.columns:
                vix_close = macro_data["vix"]
            else:
                _vix_raw = yf.download("^VIX", period=CFG.data_period, progress=False, auto_adjust=True)
                if isinstance(_vix_raw.columns, pd.MultiIndex):
                    _vix_raw.columns = _vix_raw.columns.get_level_values(0)
                vix_close = _vix_raw["Close"].copy()
                vix_close.index = pd.to_datetime(vix_close.index).tz_localize(None)

            aligned = pd.DataFrame({"vix": vix_close, "vix3m": vix3m_close}).ffill().dropna()
            if len(aligned) > 0:
                sentiment_data = pd.DataFrame(index=aligned.index)
                sentiment_data["vix_term_structure"] = aligned["vix"] / aligned["vix3m"].replace(0, np.nan) - 1
                sp = os.path.join(CFG.drive_root, "data", "sentiment_proxy.parquet")
                sentiment_data.to_parquet(sp)
    except Exception as _e:
        logger.warning("Sentiment proxy failed: %s" % str(_e)[:80])

    elapsed = time.time() - t0
    tracker.mark_completed(STEP, {"n_rows": len(sentiment_data), "time": elapsed})

print("Sentiment proxy:", sentiment_data.shape if len(sentiment_data) > 0 else "empty")'''))

# ============================================================
# CELL — Theme Crash Early Warning Function
# ============================================================
C.append(md('''\
## 3d. Theme Crash Early Warning

5-indicator fragility score per sector:
1. Concentration ratio (top-3 weight)
2. Drawdown vs mean return
3. Rising CVaR (recent 60d vs prior 60d)
4. Cross-stock correlation (herding detection)
5. Volume climax (spike + negative returns)'''))

C.append(code('''\
def compute_theme_crash_score(panel, ticker_sectors, sector, cfg):
    """Compute theme crash fragility score [0,1] for a sector.

    5 fragility indicators (weighted average):
      1. Concentration ratio (0.15): top-3 stock weight in sector
      2. Drawdown vs mean return (0.25): current DD / annualized mean
      3. Rising CVaR (0.25): recent 60d CVaR worse than prior 60d
      4. Cross-stock correlation (0.20): avg pairwise corr (>0.3 = herding)
      5. Volume climax (0.15): volume spike + negative returns
    """
    sector_tickers = [tk for tk, sec in ticker_sectors.items() if sec == sector]
    if len(sector_tickers) < cfg.sector_rotation_min_constituents:
        return 0.0

    # Collect per-ticker close and volume series
    closes = {}
    volumes = {}
    for tk in sector_tickers:
        try:
            td = panel.loc[(slice(None), tk), :].droplevel(1)
            closes[tk] = td["close"]
            volumes[tk] = td["volume"]
        except Exception:
            pass

    if len(closes) < 2:
        return 0.0

    close_df = pd.DataFrame(closes).dropna(how="all")
    vol_df = pd.DataFrame(volumes).reindex(close_df.index).fillna(0)
    if len(close_df) < cfg.theme_crash_corr_window:
        return 0.0

    scores = []

    # 1. Concentration ratio (0.15): top-3 market cap proxy (latest close * volume)
    latest_mcap = (close_df.iloc[-1] * vol_df.iloc[-20:].mean()).dropna()
    if len(latest_mcap) >= 3:
        total = latest_mcap.sum()
        top3 = latest_mcap.nlargest(3).sum()
        conc = float(top3 / total) if total > 0 else 0.5
        scores.append(0.15 * min(1.0, conc))
    else:
        scores.append(0.15 * 0.5)

    # 2. Drawdown vs mean return (0.25)
    sector_avg = close_df.mean(axis=1)
    if len(sector_avg) > 60:
        peak = sector_avg.rolling(252, min_periods=60).max()
        dd = (sector_avg - peak) / peak.replace(0, np.nan)
        current_dd = abs(float(dd.iloc[-1])) if not np.isnan(dd.iloc[-1]) else 0
        ann_mean = float(sector_avg.pct_change().mean() * 252)
        dd_ratio = min(1.0, current_dd / max(0.01, abs(ann_mean)))
        scores.append(0.25 * dd_ratio)
    else:
        scores.append(0.25 * 0.0)

    # 3. Rising CVaR (0.25): recent 60d CVaR worse than prior 60d
    rets = sector_avg.pct_change().dropna()
    w = cfg.theme_crash_corr_window
    if len(rets) > 2 * w:
        recent = rets.iloc[-w:]
        prior = rets.iloc[-2*w:-w]
        cvar_recent = float(np.sort(recent.values)[:max(1, int(0.05 * w))].mean())
        cvar_prior = float(np.sort(prior.values)[:max(1, int(0.05 * w))].mean())
        cvar_worsened = 1.0 if cvar_recent < cvar_prior else 0.0
        scores.append(0.25 * cvar_worsened)
    else:
        scores.append(0.25 * 0.0)

    # 4. Cross-stock correlation (0.20): avg pairwise corr in sector
    if close_df.shape[1] >= 2:
        ret_df = close_df.pct_change().iloc[-w:]
        corr_mat = ret_df.corr()
        n = len(corr_mat)
        if n >= 2:
            upper = corr_mat.values[np.triu_indices(n, k=1)]
            avg_corr = float(np.nanmean(upper))
            herding = min(1.0, max(0.0, (avg_corr - 0.1) / 0.5))
            scores.append(0.20 * herding)
        else:
            scores.append(0.20 * 0.0)
    else:
        scores.append(0.20 * 0.0)

    # 5. Volume climax (0.15): volume spike + negative returns
    vw = cfg.theme_crash_volume_window
    sector_vol = vol_df.sum(axis=1)
    if len(sector_vol) > vw:
        vol_ma = sector_vol.rolling(60, min_periods=20).mean()
        vol_ratio = sector_vol.iloc[-1] / vol_ma.iloc[-1] if vol_ma.iloc[-1] > 0 else 1.0
        recent_ret = float(sector_avg.pct_change(vw).iloc[-1]) if len(sector_avg) > vw else 0
        climax = 1.0 if (vol_ratio > 2.0 and recent_ret < -0.05) else 0.0
        scores.append(0.15 * climax)
    else:
        scores.append(0.15 * 0.0)

    return min(1.0, sum(scores))

print("Theme crash function defined.")'''))

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

def compute_sector_neutral_excess(net_fwd, sector_median_fwd):
    """Stock fwd return minus sector median fwd return."""
    return net_fwd - sector_median_fwd

def check_class_balance(labels):
    """Check tri-state label distribution. Returns (ok, distribution_dict)."""
    s = pd.Series(labels)
    counts = s.value_counts(normalize=True)
    dist = {"buy": counts.get(1,0), "notrade": counts.get(0,0), "avoid": counts.get(-1,0)}
    ok = 0.40 <= dist["notrade"] <= 0.90 and dist["buy"] >= 0.03 and dist["avoid"] >= 0.03
    return ok, dist

print("Feature functions + tri-state labeler defined.")'''))

C.append(code('''\
def compute_liquidity_features(ohlcv_df, cfg):
    """Axis A: Liquidity features from existing OHLCV — zero new downloads.

    Returns DataFrame with:
      - log_dollar_vol_{w}d: log of rolling dollar volume
      - amihud: Amihud illiquidity ratio (|return| / dollar_volume)
      - vol_imbalance: (up_vol - down_vol) / total_vol rolling
      - turnover_ratio: volume / rolling mean volume
      - gap_freq: frequency of overnight gaps > 1%
    """
    close = ohlcv_df["close"]
    volume = ohlcv_df["volume"]
    feats = pd.DataFrame(index=ohlcv_df.index)

    dollar_vol = close * volume
    for w in cfg.liquidity_dollar_vol_windows:
        rdv = dollar_vol.rolling(w, min_periods=max(1, w // 2)).mean()
        feats["log_dollar_vol_%dd" % w] = np.log1p(rdv)

    # Amihud illiquidity: mean(|ret| / dollar_vol) over window
    abs_ret = close.pct_change().abs()
    dv_safe = dollar_vol.replace(0, np.nan)
    amihud_raw = abs_ret / dv_safe
    feats["amihud"] = amihud_raw.rolling(cfg.amihud_window, min_periods=20).mean()

    # Volume imbalance: (up_vol - down_vol) / total_vol, 20d rolling
    ret = close.pct_change()
    up_vol = volume.where(ret > 0, 0).rolling(20).sum()
    dn_vol = volume.where(ret <= 0, 0).rolling(20).sum()
    total = (up_vol + dn_vol).replace(0, np.nan)
    feats["vol_imbalance"] = (up_vol - dn_vol) / total

    # Turnover ratio: current volume / 60d mean volume
    mean_vol = volume.rolling(60, min_periods=20).mean().replace(0, np.nan)
    feats["turnover_ratio"] = volume / mean_vol

    # Gap frequency: fraction of days with overnight gap > 1% in past 60 days
    if "open" in ohlcv_df.columns:
        gap = (ohlcv_df["open"] / close.shift(1) - 1).abs()
        feats["gap_freq"] = (gap > 0.01).astype(float).rolling(60, min_periods=20).mean()

    return feats

print("Liquidity feature function defined.")'''))

# ============================================================
# CELL — Mean Reversion Feature Function (v4)
# ============================================================
C.append(code('''\
def compute_mean_reversion_features(close, volume, cfg):
    """Mean reversion features: z-scores, Bollinger %B, vol overshoot, RSI.

    Returns DataFrame with:
      - zscore_{w}d: (close - SMA) / rolling_std
      - bollinger_pct_{w}d: position within Bollinger bands [0=lower, 1=upper]
      - vol_overshoot_{w}d: recent vol / trailing vol
      - rsi_14d: RSI continuous [0-100]
      - mean_revert_signal: average z-score inverted
    """
    feats = pd.DataFrame(index=close.index)
    ret = close.pct_change()

    for w in cfg.mr_zscore_windows:
        sma = close.rolling(w, min_periods=max(1, w // 2)).mean()
        std = close.rolling(w, min_periods=max(1, w // 2)).std()
        feats["zscore_%dd" % w] = (close - sma) / std.replace(0, np.nan)

        # Bollinger %B: (close - lower) / (upper - lower)
        upper = sma + 2 * std
        lower = sma - 2 * std
        band_width = (upper - lower).replace(0, np.nan)
        feats["bollinger_pct_%dd" % w] = (close - lower) / band_width

        # Vol overshoot: short vol / long vol
        short_vol = ret.rolling(w, min_periods=max(1, w // 2)).std()
        long_vol = ret.rolling(min(w * 3, 120), min_periods=w).std()
        feats["vol_overshoot_%dd" % w] = short_vol / long_vol.replace(0, np.nan)

    # RSI 14d
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14, min_periods=7).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14, min_periods=7).mean()
    rs = gain / loss.replace(0, np.nan)
    feats["rsi_14d"] = 100 - (100 / (1 + rs))

    # Mean revert signal: average z-score inverted (negative z -> positive signal)
    z_cols = [c for c in feats.columns if c.startswith("zscore_")]
    if z_cols:
        feats["mean_revert_signal"] = -feats[z_cols].mean(axis=1)

    return feats

print("Mean reversion feature function defined.")'''))

# ============================================================
# CELL — Volatility/Shock Feature Function (v4)
# ============================================================
C.append(code('''\
def compute_volatility_shock_features(close, volume, mkt_close, cfg):
    """Volatility/shock features: vol expansion, acceleration, volume shock.

    Returns DataFrame with:
      - vol_expansion_{w}d: short_vol / long_vol
      - vol_acceleration: second derivative of 20d volatility
      - volume_shock_{w}d: current volume / trailing volume mean
      - realized_vs_implied_proxy: vol_20d / vol_60d
      - drawdown_speed: 5-day change in drawdown from peak
    """
    feats = pd.DataFrame(index=close.index)
    ret = close.pct_change()

    for w in cfg.vs_vol_expansion_windows:
        short_vol = ret.rolling(w, min_periods=max(1, w // 2)).std()
        long_vol = ret.rolling(60, min_periods=20).std()
        feats["vol_expansion_%dd" % w] = short_vol / long_vol.replace(0, np.nan)

    # Vol acceleration: second derivative of 20d vol
    vol_20 = ret.rolling(20, min_periods=10).std()
    vol_diff = vol_20.diff()
    feats["vol_acceleration"] = vol_diff.diff()

    # Volume shock
    for w in cfg.vs_vol_expansion_windows:
        vol_mean = volume.rolling(max(w * 3, 60), min_periods=w).mean()
        feats["volume_shock_%dd" % w] = volume / vol_mean.replace(0, np.nan)

    # Realized vs implied proxy
    vol_20d = ret.rolling(20, min_periods=10).std()
    vol_60d = ret.rolling(60, min_periods=20).std()
    feats["realized_vs_implied_proxy"] = vol_20d / vol_60d.replace(0, np.nan)

    # Drawdown speed: 5-day change in drawdown from peak
    peak = close.rolling(252, min_periods=60).max()
    dd = (close - peak) / peak.replace(0, np.nan)
    feats["drawdown_speed"] = dd.diff(5)

    return feats

print("Volatility/shock feature function defined.")'''))

# ============================================================
# CELL — Cross-Market Rotation Feature Function (v4)
# ============================================================
C.append(code('''\
def compute_cross_market_rotation_features(market, market_indices_dict, cfg):
    """Cross-market rotation features broadcast to all tickers.

    Returns DataFrame with:
      - cmr_relmom_{other}_{w}d: self market mom minus other market mom
      - cmr_relvol_{other}_{w}d: self market vol / other market vol
      - cmr_divergence_{other}: momentum divergence flag
    """
    self_idx = market_indices_dict.get(market, pd.DataFrame())
    if "close" not in self_idx.columns or self_idx.empty:
        return pd.DataFrame()

    self_close = self_idx["close"]
    feats = pd.DataFrame(index=self_close.index)

    for other_mkt, other_idx in market_indices_dict.items():
        if other_mkt == market:
            continue
        if "close" not in other_idx.columns or other_idx.empty:
            continue
        other_close = other_idx["close"].reindex(self_close.index, method="ffill")

        for w in cfg.cmr_lookback_windows:
            self_mom = self_close.pct_change(w)
            other_mom = other_close.pct_change(w)
            feats["cmr_relmom_%s_%dd" % (other_mkt.lower(), w)] = self_mom - other_mom

            self_vol = self_close.pct_change().rolling(w, min_periods=max(1, w // 2)).std()
            other_vol = other_close.pct_change().rolling(w, min_periods=max(1, w // 2)).std()
            feats["cmr_relvol_%s_%dd" % (other_mkt.lower(), w)] = self_vol / other_vol.replace(0, np.nan)

        # Divergence: one market up, other down over 20d
        self_mom_20 = self_close.pct_change(20)
        other_mom_20 = other_close.pct_change(20)
        feats["cmr_divergence_%s" % other_mkt.lower()] = (
            ((self_mom_20 > 0) & (other_mom_20 < 0)) |
            ((self_mom_20 < 0) & (other_mom_20 > 0))
        ).astype(float)

    return feats

print("Cross-market rotation feature function defined.")'''))

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

            # --- Axis A: Liquidity features ---
            liq = compute_liquidity_features(td, CFG) if CFG.enable_liquidity_features else pd.DataFrame(index=close.index)

            # --- Axis B: Fundamental features ---
            fund = pd.DataFrame(index=close.index)
            if CFG.enable_fundamental_features:
                # high_52w_pct: current price as % of 52-week high
                high_52w = close.rolling(252, min_periods=60).max()
                fund["high_52w_pct"] = close / high_52w.replace(0, np.nan)
                # Static fundamentals from cache
                fund_info = fundamental_cache.get(market, {}).get(ticker, {})
                if fund_info:
                    mc = fund_info.get("marketCap", np.nan)
                    fund["log_market_cap"] = np.log1p(mc) if not (isinstance(mc, float) and np.isnan(mc)) else np.nan
                    for fld in ["trailingPE", "priceToBook", "returnOnEquity"]:
                        fund[fld] = fund_info.get(fld, np.nan)

            # --- Axis D: Sector relative (time-series) ---
            sect = pd.DataFrame(index=close.index)
            if CFG.enable_sector_features:
                fund_info_s = fundamental_cache.get(market, {}).get(ticker, {})
                ticker_sector = fund_info_s.get("sector", "Other") if fund_info_s else "Other"
                smr = sector_mean_returns.get(market, {})
                s20 = close.pct_change(20)
                if isinstance(smr, dict) and "ts" in smr:
                    ts_df = smr["ts"]
                    if ticker_sector in ts_df.columns:
                        sector_med_ts = ts_df[ticker_sector].reindex(close.index, method="ffill")
                        sect["sector_relative_20d"] = s20 - sector_med_ts
                    else:
                        sect["sector_relative_20d"] = s20
                else:
                    sect["sector_relative_20d"] = s20

            # --- V4: Mean Reversion features ---
            mr_feats = compute_mean_reversion_features(close, td["volume"], CFG) if CFG.enable_mean_reversion else pd.DataFrame(index=close.index)

            # --- V4: Volatility/Shock features ---
            vs_feats = compute_volatility_shock_features(close, td["volume"], mkt_close, CFG) if CFG.enable_volatility_shock else pd.DataFrame(index=close.index)

            combined = pd.concat([mom, vol, reg, liq, fund, sect, mr_feats, vs_feats], axis=1)

            for fd in CFG.forward_days_list:
                raw_fwd = close.pct_change(fd).shift(-fd)
                net_fwd = raw_fwd - 2 * CFG.total_cost_bps / 10000
                combined["fwd_return_%dd" % fd] = net_fwd

                # Excess return & tri-state label
                mf = mkt_fwd[fd].reindex(close.index, method='ffill') if len(mkt_fwd[fd])>0 else 0
                excess = net_fwd - mf
                th_pct = CFG.tristate_thresholds_pct.get(fd, 1.0)
                combined["label_%dd" % fd] = compute_tristate_labels(excess, th_pct)

                # Sector-neutral excess return and label
                # V4: Skip SN labels when market has trivial sectors (< 3 unique)
                _unique_sectors = set()
                _smr_check = sector_mean_returns.get(market, {})
                if isinstance(_smr_check, dict) and "ticker_sectors" in _smr_check:
                    _unique_sectors = set(_smr_check["ticker_sectors"].values())
                _sn_ok = CFG.enable_sector_neutral_labels and len(_unique_sectors) >= 3
                if _sn_ok:
                    smr = sector_mean_returns.get(market, {})
                    if isinstance(smr, dict) and "sector_median_fwd" in smr:
                        smf = smr["sector_median_fwd"].get(fd, pd.DataFrame())
                        if ticker_sector in smf.columns:
                            sector_med = smf[ticker_sector].reindex(close.index, method="ffill").fillna(0)
                        else:
                            sector_med = 0
                    else:
                        sector_med = 0
                    sn_excess = compute_sector_neutral_excess(net_fwd, sector_med)
                    combined["fwd_return_sn_%dd" % fd] = sn_excess
                    combined["label_sn_%dd" % fd] = compute_tristate_labels(sn_excess, th_pct)

            combined["ticker"] = ticker
            combined.index.name = "date"
            all_features.append(combined)
        except Exception as e:
            logger.warning("Feature err %s/%s: %s" % (market, ticker, str(e)[:60]))

    if not all_features: continue
    fp = pd.concat(all_features).reset_index().set_index(["date","ticker"]).sort_index()

    # --- V4: Add cross-market rotation features (market-level, broadcast) ---
    if CFG.enable_cross_market_rotation and len(market_indices) > 1:
        cmr_feats = compute_cross_market_rotation_features(market, market_indices, CFG)
        if len(cmr_feats) > 0:
            dates = fp.index.get_level_values(0)
            for col in cmr_feats.columns:
                cmr_aligned = cmr_feats[col].reindex(dates, method="ffill")
                cmr_aligned.index = fp.index
                fp[col] = cmr_aligned.values
            logger.info("Added %d CMR features for %s" % (len(cmr_feats.columns), market))

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
        lc_sn = "label_sn_%dd" % fd
        if lc_sn in fp.columns:
            ok_sn, dist_sn = check_class_balance(fp[lc_sn].dropna().values)
            status_sn = "OK" if ok_sn else "WARN"
            print("  %s label_sn_%dd: BUY=%.1f%% NO_TRADE=%.1f%% AVOID=%.1f%% [%s]" % (
                market, fd, dist_sn["buy"]*100, dist_sn["notrade"]*100, dist_sn["avoid"]*100, status_sn))
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
    fwd_cols = sorted([c for c in fp.columns if c.startswith("fwd_return_") and ("_sn_" not in c or CFG.enable_sn_candidates)])
    decile_cols = [c for c in fp.columns if c.endswith("_decile")]

    for fwd_col in fwd_cols:
        ht = fwd_col.replace("fwd_return_","")
        STEP = "cand_decile_%s_%s" % (market, ht)
        sp = os.path.join(CFG.candidates_dir(market), "decile_%s.parquet" % ht)
        if tracker.is_completed(STEP):
            if os.path.exists(sp): all_candidates_list.append(pd.read_parquet(sp))
            continue

        logger.info("[RUN] %s" % STEP); t0 = time.time()
        valid = fp.dropna(subset=[fwd_col]).copy()
        if len(valid) < CFG.min_sample_size:
            logger.warning("Skipping %s: only %d valid rows" % (STEP, len(valid)))
            dc = pd.DataFrame(); dc.to_parquet(sp); all_candidates_list.append(dc)
            tracker.mark_completed(STEP, {"n":0,"reason":"insufficient_data"}); continue
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
    fwd_cols = sorted([c for c in fp.columns if c.startswith("fwd_return_") and ("_sn_" not in c or CFG.enable_sn_candidates)])
    feat_cols = [c for c in fp.columns if not c.startswith("fwd_") and not c.startswith("label_") and not c.endswith("_decile")]

    for fwd_col in fwd_cols:
        ht = fwd_col.replace("fwd_return_","")
        label_col = "label_%s" % ht
        STEP = "cand_tree_%s_%s" % (market, ht)
        sp = os.path.join(CFG.candidates_dir(market), "tree_%s.parquet" % ht)
        if tracker.is_completed(STEP):
            if os.path.exists(sp): all_candidates_list.append(pd.read_parquet(sp))
            continue

        logger.info("[RUN] %s" % STEP); t0 = time.time()
        valid = fp.dropna(subset=[fwd_col]+feat_cols).copy()
        if len(valid) < CFG.min_sample_size:
            logger.warning("Skipping %s: only %d valid rows" % (STEP, len(valid)))
            tc = pd.DataFrame(); tc.to_parquet(sp); all_candidates_list.append(tc)
            tracker.mark_completed(STEP, {"n":0,"reason":"insufficient_data"}); continue
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
    fwd_cols = sorted([c for c in fp.columns if c.startswith("fwd_return_") and ("_sn_" not in c or CFG.enable_sn_candidates)])
    feat_cols = [c for c in fp.columns if not c.startswith("fwd_") and not c.startswith("label_") and not c.endswith("_decile")]

    for fwd_col in fwd_cols:
        ht = fwd_col.replace("fwd_return_","")
        label_col = "label_%s" % ht
        STEP = "cand_logistic_%s_%s" % (market, ht)
        sp = os.path.join(CFG.candidates_dir(market), "logistic_%s.parquet" % ht)
        if tracker.is_completed(STEP):
            if os.path.exists(sp): all_candidates_list.append(pd.read_parquet(sp))
            continue

        logger.info("[RUN] %s" % STEP); t0 = time.time()
        valid = fp.dropna(subset=[fwd_col]+feat_cols).copy()
        if len(valid) < CFG.min_sample_size:
            logger.warning("Skipping %s: only %d valid rows" % (STEP, len(valid)))
            lc = pd.DataFrame(); lc.to_parquet(sp); all_candidates_list.append(lc)
            tracker.mark_completed(STEP, {"n":0,"reason":"insufficient_data"}); continue
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
# --- V4: Mean Reversion Candidate Generation ---
if CFG.enable_mean_reversion:
    for market in CFG.markets:
        if market not in feature_panels: continue
        fp = feature_panels[market]
        mr_decile_cols = [c for c in fp.columns if c.endswith("_decile") and
                          any(c.startswith(p) for p in ["zscore_", "bollinger_pct_", "vol_overshoot_", "mean_revert_signal_"])]
        if not mr_decile_cols:
            logger.info("No MR decile columns for %s, skipping MR candidates" % market)
            continue

        for horizon in CFG.mr_holding_days:
            # Find matching forward return column
            fwd_col = None
            for fd in CFG.forward_days_list:
                if fd == horizon or (horizon <= 10 and fd == 5) or (horizon > 10 and fd == 21):
                    fwd_col = "fwd_return_%dd" % fd
                    break
            if fwd_col is None or fwd_col not in fp.columns:
                fwd_col = "fwd_return_%dd" % CFG.forward_days_list[0]
            if fwd_col not in fp.columns: continue
            ht = fwd_col.replace("fwd_return_", "")

            STEP = "cand_meanrev_%s_%s" % (market, ht)
            sp = os.path.join(CFG.candidates_dir(market), "meanrev_%s.parquet" % ht)
            if tracker.is_completed(STEP):
                if os.path.exists(sp): all_candidates_list.append(pd.read_parquet(sp))
                continue

            logger.info("[RUN] %s" % STEP); t0 = time.time()
            valid = fp.dropna(subset=[fwd_col]).copy()
            if len(valid) < CFG.min_sample_size:
                pd.DataFrame().to_parquet(sp); tracker.mark_completed(STEP, {"n": 0}); continue

            n_bins = int(valid[mr_decile_cols[0]].dropna().max()) + 1 if mr_decile_cols else 10
            cands = []
            cap = CFG.max_candidates_total // 3

            # Single oversold decile (low z-score/bollinger = oversold = buy for mean reversion)
            for col in mr_decile_cols:
                for dv in range(min(3, n_bins)):  # Low deciles = oversold
                    mask = valid[col] == dv; nt = int(mask.sum())
                    if nt < CFG.min_sample_size: continue
                    ret = valid.loc[mask, fwd_col]
                    mr = float(ret.mean())
                    if mr <= 0: continue
                    cands.append({
                        "strategy_id": "%s_%s_meanrev_%s_d%d" % (market, ht, col.replace("_decile", ""), dv),
                        "market": market, "horizon": ht, "type": "meanrev_single",
                        "features": col, "condition": "== %d" % dv,
                        "n_trades": nt, "mean_return": mr, "win_rate": float((ret > 0).mean())})
                    if len(cands) >= cap: break
                if len(cands) >= cap: break

            # Combo: oversold z-score + high vol_overshoot
            z_cols = [c for c in mr_decile_cols if "zscore" in c or "bollinger" in c]
            v_cols = [c for c in mr_decile_cols if "vol_overshoot" in c]
            for zc in z_cols[:3]:
                for vc in v_cols[:3]:
                    for zd in range(min(3, n_bins)):
                        for vd in range(max(0, n_bins - 3), n_bins):
                            if len(cands) >= cap: break
                            mask = (valid[zc] == zd) & (valid[vc] == vd)
                            nt = int(mask.sum())
                            if nt < CFG.min_sample_size: continue
                            ret = valid.loc[mask, fwd_col]; mr = float(ret.mean())
                            if mr <= 0: continue
                            cands.append({
                                "strategy_id": "%s_%s_meanrev_%s_d%d_AND_%s_d%d" % (
                                    market, ht, zc.replace("_decile", ""), zd, vc.replace("_decile", ""), vd),
                                "market": market, "horizon": ht, "type": "meanrev_combo",
                                "features": "%s, %s" % (zc, vc),
                                "condition": "%s==%d AND %s==%d" % (zc, zd, vc, vd),
                                "n_trades": nt, "mean_return": mr, "win_rate": float((ret > 0).mean())})

            dc = pd.DataFrame(cands); dc.to_parquet(sp); all_candidates_list.append(dc)
            tracker.mark_completed(STEP, {"n": len(dc), "time": time.time() - t0})
            print("MR candidates %s/%s: %d" % (market, ht, len(dc)))
            gc.collect()

print("After MR: %d total candidates" % sum(len(d) for d in all_candidates_list))'''))

C.append(code('''\
# --- V4: Volatility/Shock Candidate Generation ---
if CFG.enable_volatility_shock:
    for market in CFG.markets:
        if market not in feature_panels: continue
        fp = feature_panels[market]
        vs_decile_cols = [c for c in fp.columns if c.endswith("_decile") and
                          any(c.startswith(p) for p in ["vol_expansion_", "vol_acceleration_",
                                                         "volume_shock_", "drawdown_speed_"])]
        if not vs_decile_cols:
            logger.info("No VS decile columns for %s, skipping VS candidates" % market)
            continue

        for horizon in CFG.vs_holding_days:
            fwd_col = None
            for fd in CFG.forward_days_list:
                if fd == horizon or (horizon <= 10 and fd == 5) or (horizon > 10 and fd == 21):
                    fwd_col = "fwd_return_%dd" % fd
                    break
            if fwd_col is None or fwd_col not in fp.columns:
                fwd_col = "fwd_return_%dd" % CFG.forward_days_list[0]
            if fwd_col not in fp.columns: continue
            ht = fwd_col.replace("fwd_return_", "")

            STEP = "cand_volshock_%s_%s" % (market, ht)
            sp = os.path.join(CFG.candidates_dir(market), "volshock_%s.parquet" % ht)
            if tracker.is_completed(STEP):
                if os.path.exists(sp): all_candidates_list.append(pd.read_parquet(sp))
                continue

            logger.info("[RUN] %s" % STEP); t0 = time.time()
            valid = fp.dropna(subset=[fwd_col]).copy()
            if len(valid) < CFG.min_sample_size:
                pd.DataFrame().to_parquet(sp); tracker.mark_completed(STEP, {"n": 0}); continue

            n_bins = int(valid[vs_decile_cols[0]].dropna().max()) + 1 if vs_decile_cols else 10
            cands = []
            cap = CFG.max_candidates_total // 3

            # High deciles of vol_expansion/vol_acceleration (activate during stress)
            for col in vs_decile_cols:
                for dv in range(max(0, n_bins - 3), n_bins):
                    mask = valid[col] == dv; nt = int(mask.sum())
                    if nt < CFG.min_sample_size: continue
                    ret = valid.loc[mask, fwd_col]
                    mr = float(ret.mean())
                    if mr <= 0: continue
                    cands.append({
                        "strategy_id": "%s_%s_volshock_%s_d%d" % (market, ht, col.replace("_decile", ""), dv),
                        "market": market, "horizon": ht, "type": "volshock_single",
                        "features": col, "condition": "== %d" % dv,
                        "n_trades": nt, "mean_return": mr, "win_rate": float((ret > 0).mean())})
                    if len(cands) >= cap: break
                if len(cands) >= cap: break

            dc = pd.DataFrame(cands); dc.to_parquet(sp); all_candidates_list.append(dc)
            tracker.mark_completed(STEP, {"n": len(dc), "time": time.time() - t0})
            print("VS candidates %s/%s: %d" % (market, ht, len(dc)))
            gc.collect()

print("After VS: %d total candidates" % sum(len(d) for d in all_candidates_list))'''))

C.append(code('''\
# --- V4: Cross-Market Rotation Candidate Generation ---
if CFG.enable_cross_market_rotation:
    for market in CFG.markets:
        if market not in feature_panels: continue
        fp = feature_panels[market]
        cmr_decile_cols = [c for c in fp.columns if c.endswith("_decile") and c.startswith("cmr_")]
        if not cmr_decile_cols:
            logger.info("No CMR decile columns for %s, skipping CMR candidates" % market)
            continue

        # CMR uses rebalance_days as horizon
        fwd_col = None
        for fd in CFG.forward_days_list:
            if fd >= CFG.cmr_rebalance_days:
                fwd_col = "fwd_return_%dd" % fd
                break
        if fwd_col is None:
            fwd_col = "fwd_return_%dd" % CFG.forward_days_list[-1]
        if fwd_col not in fp.columns: continue
        ht = fwd_col.replace("fwd_return_", "")

        STEP = "cand_cmr_%s_%s" % (market, ht)
        sp = os.path.join(CFG.candidates_dir(market), "cmr_%s.parquet" % ht)
        if tracker.is_completed(STEP):
            if os.path.exists(sp): all_candidates_list.append(pd.read_parquet(sp))
            continue

        logger.info("[RUN] %s" % STEP); t0 = time.time()
        valid = fp.dropna(subset=[fwd_col]).copy()
        if len(valid) < CFG.min_sample_size:
            pd.DataFrame().to_parquet(sp); tracker.mark_completed(STEP, {"n": 0}); continue

        n_bins = int(valid[cmr_decile_cols[0]].dropna().max()) + 1 if cmr_decile_cols else 10
        cands = []
        cap = CFG.max_candidates_total // 3

        # Extreme deciles of relative momentum/volatility
        for col in cmr_decile_cols:
            for dv in list(range(min(2, n_bins))) + list(range(max(0, n_bins - 2), n_bins)):
                mask = valid[col] == dv; nt = int(mask.sum())
                if nt < CFG.min_sample_size: continue
                ret = valid.loc[mask, fwd_col]
                mr = float(ret.mean())
                if mr <= 0: continue
                cands.append({
                    "strategy_id": "%s_%s_cmr_%s_d%d" % (market, ht, col.replace("_decile", ""), dv),
                    "market": market, "horizon": ht, "type": "cmr_single",
                    "features": col, "condition": "== %d" % dv,
                    "n_trades": nt, "mean_return": mr, "win_rate": float((ret > 0).mean())})
                if len(cands) >= cap: break
            if len(cands) >= cap: break

        dc = pd.DataFrame(cands); dc.to_parquet(sp); all_candidates_list.append(dc)
        tracker.mark_completed(STEP, {"n": len(dc), "time": time.time() - t0})
        print("CMR candidates %s/%s: %d" % (market, ht, len(dc)))
        gc.collect()

print("After CMR: %d total candidates" % sum(len(d) for d in all_candidates_list))'''))

C.append(code('''\
all_candidates = pd.concat(all_candidates_list, ignore_index=True) if all_candidates_list else pd.DataFrame()
print("=== All Candidates: %d ===" % len(all_candidates))
if len(all_candidates) > 0:
    print(all_candidates.groupby(["market","horizon","type"]).size().to_string())'''))

# ============================================================
# CELL — Sector Rotation Model
# ============================================================
C.append(md('''\
## 5b. Sector Rotation Model

Train `GradientBoostingClassifier` per market to predict P(sector outperforms market).
Input: 5 macro features. Output: sector alpha scores for meta-model and portfolio gating.'''))

C.append(code('''\
sector_alpha_scores = {}  # {market: {sector: float}}
strategy_primary_sector = {}  # {strategy_id: sector}

STEP = "sector_rotation"
if tracker.is_completed(STEP):
    sr_path = os.path.join(CFG.global_eval_dir, "sector_rotation.json")
    if os.path.exists(sr_path):
        with open(sr_path) as f: sector_alpha_scores = json.load(f)
    logger.info("[SKIP] %s" % STEP)
elif not CFG.enable_sector_rotation or len(macro_data) == 0:
    logger.info("[SKIP] sector rotation (disabled or no macro)")
    tracker.mark_completed(STEP, {"skipped": True})
else:
    logger.info("[RUN] %s" % STEP)
    from sklearn.ensemble import GradientBoostingClassifier as _GBC

    for market in CFG.markets:
        if market not in ohlcv_data:
            continue
        smr = sector_mean_returns.get(market, {})
        if not isinstance(smr, dict) or "ts" not in smr:
            continue
        ts_median = smr["ts"]
        ticker_sectors = smr.get("ticker_sectors", {})
        sectors = list(ts_median.columns)
        if len(sectors) < 2:
            continue

        # Build training data: (date, sector) -> label: sector outperforms market
        mkt_idx = market_indices.get(market, pd.DataFrame())
        if "close" not in mkt_idx.columns or mkt_idx.empty:
            continue
        mkt_ret_20d = mkt_idx["close"].pct_change(20)

        # Align macro features with dates
        macro_aligned = macro_data.reindex(ts_median.index, method="ffill")
        macro_feats = ["yield_curve_slope", "vix_regime", "dxy_mom_60d", "gold_mom_60d", "oil_mom_60d"]
        missing_feats = [f for f in macro_feats if f not in macro_aligned.columns]
        for mf in missing_feats:
            macro_aligned[mf] = 0.0

        train_X, train_y = [], []
        valid_dates = ts_median.dropna(how="all").index
        mkt_ret_aligned = mkt_ret_20d.reindex(valid_dates, method="ffill")

        for dt in valid_dates:
            if dt not in macro_aligned.index:
                continue
            mrow = macro_aligned.loc[dt, macro_feats]
            if mrow.isna().all():
                continue
            mkt_r = mkt_ret_aligned.get(dt, 0)
            if np.isnan(mkt_r):
                continue
            for sec in sectors:
                sec_r = ts_median.loc[dt, sec]
                if np.isnan(sec_r):
                    continue
                feat_vec = list(mrow.fillna(0).values)
                train_X.append(feat_vec)
                train_y.append(1 if sec_r > mkt_r else 0)

        if len(train_X) < 50:
            sector_alpha_scores[market] = {s: 0.5 for s in sectors}
            continue

        MX = np.array(train_X, dtype=np.float32)
        MY = np.array(train_y)
        np.nan_to_num(MX, copy=False)

        split = int(len(MX) * 0.7)
        gb = _GBC(n_estimators=30, max_depth=2, random_state=CFG.seed)
        gb.fit(MX[:split], MY[:split])
        test_acc = float((gb.predict(MX[split:]) == MY[split:]).mean())
        logger.info("Sector rotation %s: test acc=%.2f (%d samples)" % (market, test_acc, len(MX)))

        # Score each sector using latest macro state
        latest_macro = macro_aligned.iloc[-1][macro_feats].fillna(0).values.reshape(1, -1).astype(np.float32)
        np.nan_to_num(latest_macro, copy=False)
        scores = {}
        for sec in sectors:
            proba = gb.predict_proba(latest_macro)
            scores[sec] = float(proba[0][1]) if proba.shape[1] > 1 else 0.5
        sector_alpha_scores[market] = scores

    # Save
    sr_path = os.path.join(CFG.global_eval_dir, "sector_rotation.json")
    os.makedirs(os.path.dirname(sr_path), exist_ok=True)
    with open(sr_path, "w") as f:
        json.dump(sector_alpha_scores, f, indent=2)
    tracker.mark_completed(STEP, {"n_markets": len(sector_alpha_scores)})

# Determine primary sector for each candidate
if len(all_candidates) > 0 and sector_mean_returns:
    for _, cand in all_candidates.iterrows():
        sid = cand["strategy_id"]
        mkt = cand.get("market", "")
        smr = sector_mean_returns.get(mkt, {})
        if isinstance(smr, dict) and "ticker_sectors" in smr:
            tk_secs = smr["ticker_sectors"]
            # Heuristic: assign based on most common sector in the market
            # For tree/logistic, use the majority sector of selected tickers
            if tk_secs:
                from collections import Counter
                sec_counts = Counter(tk_secs.values())
                strategy_primary_sector[sid] = sec_counts.most_common(1)[0][0]

print("Sector alpha scores:", {m: len(v) for m, v in sector_alpha_scores.items()})
print("Gated sectors (alpha < %.2f):" % CFG.sector_alpha_threshold,
      {m: [s for s, v in secs.items() if v < CFG.sector_alpha_threshold]
       for m, secs in sector_alpha_scores.items()})'''))

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
    if stype in ("single_decile", "meanrev_single", "volshock_single", "cmr_single"):
        return data[cand_row["features"]] == int(cand_row["condition"].split("== ")[1])
    elif stype in ("combo_decile", "meanrev_combo"):
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
    fwd_cols = sorted([c for c in fp.columns if c.startswith("fwd_return_") and ("_sn_" not in c or CFG.enable_sn_candidates)])
    feat_cols = [c for c in fp.columns if not c.startswith("fwd_") and not c.startswith("label_") and not c.endswith("_decile")]

    for fwd_col in fwd_cols:
        ht = fwd_col.replace("fwd_return_","")
        label_col = "label_%s" % ht
        STEP = "edge_%s_%s" % (market, ht)
        ep = os.path.join(CFG.evaluation_dir(market), "edge_%s.parquet"%ht)
        if tracker.is_completed(STEP):
            if os.path.exists(ep): all_edge_results.append(pd.read_parquet(ep))
            continue

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
                import re as _re; _hd = int(_re.search(r'(\\d+)', ht).group(1))
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
    fwd_cols = sorted([c for c in fp.columns if c.startswith("fwd_return_") and ("_sn_" not in c or CFG.enable_sn_candidates)])
    feat_cols = [c for c in fp.columns if not c.startswith("fwd_") and not c.startswith("label_") and not c.endswith("_decile")]

    for fwd_col in fwd_cols:
        ht = fwd_col.replace("fwd_return_","")
        label_col = "label_%s" % ht
        STEP = "wf_%s_%s" % (market, ht)
        wp = os.path.join(CFG.walkforward_dir(market), "wf_%s.parquet"%ht)
        if tracker.is_completed(STEP):
            if os.path.exists(wp): all_wf_results.append(pd.read_parquet(wp))
            continue

        logger.info("[RUN] %s" % STEP); t0 = time.time()
        valid = fp.dropna(subset=[fwd_col]).copy()
        mh_edge = edge_results[(edge_results["market"]==market)&(edge_results["horizon"]==ht)]
        if len(mh_edge)>200: top_ids = mh_edge.nlargest(200,"sharpe")["strategy_id"].tolist()
        elif len(mh_edge)>0: top_ids = mh_edge.nlargest(min(50,len(mh_edge)),"sharpe")["strategy_id"].tolist()
        else: top_ids = []
        if not top_ids:
            pd.DataFrame(columns=_WF_COLS).to_parquet(wp)
            tracker.mark_completed(STEP,{"n":0}); continue

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
                    import re as _re; _hd2 = int(_re.search(r'(\\d+)', ht).group(1))
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

        # V4: Per-market overfitting thresholds
        mask_parts = []
        for _, row in sdf.iterrows():
            mkt = row.get("market", "US")
            ok = (
                row["stability"] >= get_market_threshold(mkt, "min_stability") and
                row["mean_sharpe"] >= get_market_threshold(mkt, "min_sharpe") and
                row["mean_win_rate"] >= get_market_threshold(mkt, "min_win_rate") and
                row["mean_precision_buy"] >= get_market_threshold(mkt, "min_precision_buy") and
                row["wr_ci_low"] >= 0.48 and
                row["fdr_reject"] == True
            )
            mask_parts.append(ok)
        mask = pd.Series(mask_parts, index=sdf.index)
        filtered = sdf[mask].sort_values("mean_sharpe", ascending=False).copy()
        print("Overfitting: %d -> %d (market-specific thresholds)" % (len(sdf), len(filtered)))
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
# CELL — Portfolio Diversification Optimizer (v4)
# ============================================================
C.append(md('''\
## 12b. Portfolio Diversification Optimizer (v4)

Greedy forward selection maximizing diversity score:
1. Start with highest-Sharpe strategy
2. Score = sharpe × (1 - avg_corr) + type_bonus + convexity_bonus
3. Penalty if avg_corr > max_correlation threshold'''))

C.append(code('''\
STEP = "diversity_optimizer"
do_path = os.path.join(CFG.global_eval_dir, "diversity_optimized.parquet")

if tracker.is_completed(STEP):
    deduped = pd.read_parquet(do_path); print("Loaded %d diversity-optimized" % len(deduped))
elif not CFG.enable_diversity_optimizer or len(deduped) < CFG.diversity_min_strategies:
    logger.info("[SKIP] diversity optimizer (disabled or too few strategies: %d)" % len(deduped))
    tracker.mark_completed(STEP, {"n": len(deduped), "skipped": True})
else:
    logger.info("[RUN] %s" % STEP)
    t0 = time.time()

    # Build return correlation matrix from WF results
    pivot = wf_results.pivot_table(values="mean_return", index="fold_idx",
                                   columns="strategy_id", aggfunc="first")
    sids = deduped["strategy_id"].tolist()
    pivot = pivot[[c for c in pivot.columns if c in sids]].dropna(axis=1, how="all").fillna(0)

    if pivot.shape[1] >= CFG.diversity_min_strategies:
        corr_matrix = pivot.corr()

        # Get strategy types for type bonus
        type_map = deduped.set_index("strategy_id")["type"].to_dict() if "type" in deduped.columns else {}
        sharpe_map = deduped.set_index("strategy_id")["mean_sharpe"].to_dict()

        # Greedy forward selection
        max_select = CFG.portfolio_max_strategies * len(CFG.markets)
        available = [s for s in sids if s in corr_matrix.columns]
        if not available:
            logger.warning("No strategies in correlation matrix")
        else:
            # Start with highest Sharpe
            available.sort(key=lambda s: sharpe_map.get(s, 0), reverse=True)
            selected = [available[0]]
            available = available[1:]
            selected_types = {type_map.get(selected[0], "unknown")}

            while available and len(selected) < max_select:
                best_score = -np.inf
                best_sid = None

                for sid in available:
                    s_sharpe = sharpe_map.get(sid, 0)
                    s_type = type_map.get(sid, "unknown")

                    # Average correlation with already selected
                    corrs = [abs(corr_matrix.loc[sid, sel]) for sel in selected
                             if sid in corr_matrix.index and sel in corr_matrix.columns]
                    avg_corr = float(np.mean(corrs)) if corrs else 0

                    # Type bonus: reward new strategy type
                    type_bonus = 0.3 if s_type not in selected_types else 0

                    # Convexity bonus: reward negative correlation
                    neg_corrs = [corr_matrix.loc[sid, sel] for sel in selected
                                 if sid in corr_matrix.index and sel in corr_matrix.columns]
                    avg_raw_corr = float(np.mean(neg_corrs)) if neg_corrs else 0
                    convexity_bonus = CFG.diversity_reward_opposite_convexity if avg_raw_corr < 0 else 0

                    # Penalty for high correlation
                    corr_penalty = 1.0
                    if avg_corr > CFG.diversity_max_correlation:
                        corr_penalty = CFG.meta_v2_signal_diversity_penalty

                    score = s_sharpe * (1 - avg_corr) * corr_penalty + type_bonus + convexity_bonus

                    if score > best_score:
                        best_score = score
                        best_sid = sid

                if best_sid is None:
                    break
                selected.append(best_sid)
                selected_types.add(type_map.get(best_sid, "unknown"))
                available.remove(best_sid)

            before = len(deduped)
            deduped = deduped[deduped["strategy_id"].isin(selected)].copy()
            print("Diversity optimizer: %d -> %d (types: %s)" % (before, len(deduped), selected_types))
    else:
        logger.info("Too few strategies in corr matrix for diversity optimization")

    if len(deduped) > 0:
        deduped.to_parquet(do_path)
    tracker.mark_completed(STEP, {"n": len(deduped), "time": time.time() - t0})

print("After diversity optimization: %d" % len(deduped))'''))

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
        # V4: per-market turnover thresholds
        turnover_mask = regime_ok.apply(
            lambda r: r["mean_turnover"] <= get_market_threshold(r.get("market", "US"), "max_turnover"), axis=1)
        turnover_ok = regime_ok[turnover_mask].copy()
        print("Turnover: %d -> %d (market-specific)" % (before, len(turnover_ok)))
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
# Initialize variables referenced by _build_meta_features (computed in later cells)
if "theme_crash_scores" not in dir(): theme_crash_scores = {}
if "cross_market_flags" not in dir(): cross_market_flags = {}
if "abstention_count" not in dir(): abstention_count = 0
if "mortality_killed" not in dir(): mortality_killed = []
if "meta_scored" not in dir(): meta_scored = pd.DataFrame()
if "turnover_ok" not in dir(): turnover_ok = pd.DataFrame()

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

        def _build_meta_features(ts_date):
            """Build 30-dim meta-feature vector at a given timestamp (v4).

            [0-3]   Market: 20d mom, 60d mom, 20d vol, 60d vol
            [4-8]   Macro: yield_curve_slope, vix_regime, dxy/gold/oil momentum
            [9]     Sentiment: vix_term_structure
            [10]    Sector alpha: median sector rotation score
            [11]    Theme crash: max ThemeCrashScore
            [12]    Absolute alpha strength: mean fold Sharpe abs
            [13]    SN alpha strength: mean fold Sharpe sn
            [14]    Regime alignment: regime_ratio
            [15]    Cross-market contagion flag
            [16]    Rolling Sharpe (mortality signal)
            [17]    Trade abstention EV score
            [18-19] Reserved padding (zeros)
            [20]    Strategy type diversity (v4)
            [21]    Avg correlation with portfolio (v4)
            [22]    Marginal CVaR contribution proxy (v4)
            [23]    Regime-conditional Sharpe (v4)
            [24]    Shared failure mode indicator (v4)
            [25]    Opposite convexity score (v4)
            [26-29] Reserved (v4)
            """
            mf = [0.0] * 30
            # [0-3] Market features
            for _mkt_name in CFG.markets:
                _mi = market_indices.get(_mkt_name, pd.DataFrame())
                if "close" not in _mi.columns or _mi.empty:
                    continue
                _mc = _mi["close"]
                _pre = _mc.loc[:ts_date]
                if len(_pre) < 60:
                    continue
                mf[0] = float(_pre.pct_change(20).iloc[-1]) if len(_pre) > 20 else 0
                mf[1] = float(_pre.pct_change(60).iloc[-1]) if len(_pre) > 60 else 0
                mf[2] = float(_pre.pct_change().rolling(20).std().iloc[-1]) if len(_pre) > 20 else 0
                mf[3] = float(_pre.pct_change().rolling(60).std().iloc[-1]) if len(_pre) > 60 else 0
                break  # use first available market

            # [4-8] Macro features
            if len(macro_data) > 0:
                _md = macro_data.loc[:ts_date]
                if len(_md) > 0:
                    last = _md.iloc[-1]
                    mf[4] = float(last.get("yield_curve_slope", 0)) if not np.isnan(last.get("yield_curve_slope", 0)) else 0
                    mf[5] = float(last.get("vix_regime", 1)) if not np.isnan(last.get("vix_regime", 1)) else 1
                    mf[6] = float(last.get("dxy_mom_60d", 0)) if not np.isnan(last.get("dxy_mom_60d", 0)) else 0
                    mf[7] = float(last.get("gold_mom_60d", 0)) if not np.isnan(last.get("gold_mom_60d", 0)) else 0
                    mf[8] = float(last.get("oil_mom_60d", 0)) if not np.isnan(last.get("oil_mom_60d", 0)) else 0

            # [9] Sentiment feature
            if len(sentiment_data) > 0:
                _sd = sentiment_data.loc[:ts_date]
                if len(_sd) > 0:
                    mf[9] = float(_sd["vix_term_structure"].iloc[-1]) if not np.isnan(_sd["vix_term_structure"].iloc[-1]) else 0

            # [10] Sector alpha: median sector rotation score
            if sector_alpha_scores:
                all_scores = []
                for _mkt, _secs in sector_alpha_scores.items():
                    if isinstance(_secs, dict):
                        all_scores.extend(_secs.values())
                mf[10] = float(np.median(all_scores)) if all_scores else 0.5

            # [11] Theme crash: max score across all sectors
            if theme_crash_scores:
                max_tc = 0.0
                for _mkt, _secs in theme_crash_scores.items():
                    if isinstance(_secs, dict):
                        for _s, _v in _secs.items():
                            max_tc = max(max_tc, _v)
                mf[11] = max_tc

            # [12] Absolute alpha strength: mean fold Sharpe (abs horizons)
            if len(wf_results) > 0:
                abs_wf = wf_results[~wf_results["horizon"].str.startswith("sn_")]
                if len(abs_wf) > 0:
                    mf[12] = float(abs_wf["sharpe"].mean())

            # [13] SN alpha strength: mean fold Sharpe (sn horizons)
            if len(wf_results) > 0:
                sn_wf = wf_results[wf_results["horizon"].str.startswith("sn_")]
                if len(sn_wf) > 0:
                    mf[13] = float(sn_wf["sharpe"].mean())

            # [14] Regime alignment: median regime_ratio from filtered
            if len(turnover_ok) > 0 and "regime_ratio" in turnover_ok.columns:
                mf[14] = float(turnover_ok["regime_ratio"].median())

            # [15] Cross-market contagion flag
            if cross_market_flags:
                any_contagion = any(v.get("contagion", False) for v in cross_market_flags.values())
                mf[15] = 1.0 if any_contagion else 0.0

            # [16] Rolling Sharpe (mortality signal) - avg recent fold Sharpe
            if len(wf_results) > 0:
                recent_folds = wf_results.sort_values("fold_idx").groupby("strategy_id").tail(CFG.mortality_rolling_sharpe_window)
                mf[16] = float(recent_folds["sharpe"].mean())

            # [17] Trade abstention EV score - proportion surviving
            mf[17] = float(len(meta_scored)) / max(1, float(len(turnover_ok))) if len(turnover_ok) > 0 else 1.0

            # [18-19] reserved padding = 0

            # --- V4 Meta Features [20-29] ---
            # [20] Strategy type diversity
            if len(turnover_ok) > 0 and "type" in turnover_ok.columns:
                n_types = turnover_ok["type"].nunique()
                total = len(turnover_ok)
                mf[20] = float(n_types) / max(1, total) if total > 0 else 0

            # [21] Avg correlation with portfolio (from diversity optimizer)
            if len(wf_results) > 0 and len(turnover_ok) > 0:
                try:
                    _piv = wf_results.pivot_table(values="mean_return", index="fold_idx",
                                                   columns="strategy_id", aggfunc="first")
                    _sids = turnover_ok["strategy_id"].tolist()
                    _piv = _piv[[c for c in _piv.columns if c in _sids]].dropna(axis=1, how="all").fillna(0)
                    if _piv.shape[1] >= 2:
                        _corr = _piv.corr().values
                        _n = len(_corr)
                        _upper = _corr[np.triu_indices(_n, k=1)]
                        mf[21] = float(np.nanmean(np.abs(_upper)))
                except Exception:
                    pass

            # [22] Marginal CVaR contribution proxy
            if len(wf_results) > 0 and len(turnover_ok) > 0:
                try:
                    _port_ret = wf_results[wf_results["strategy_id"].isin(turnover_ok["strategy_id"])].groupby("fold_idx")["mean_return"].mean()
                    _so = np.sort(_port_ret.values)
                    _nt = max(1, int(0.05 * len(_so)))
                    mf[22] = float(_so[:_nt].mean()) if len(_so) > 0 else 0
                except Exception:
                    pass

            # [23] Regime-conditional Sharpe: Sharpe in current VIX regime
            if len(wf_results) > 0:
                try:
                    _recent_wf = wf_results.sort_values("fold_idx").groupby("strategy_id").tail(2)
                    mf[23] = float(_recent_wf["sharpe"].mean())
                except Exception:
                    pass

            # [24] Shared failure: fraction of strategies failing same folds
            if len(wf_results) > 0 and len(turnover_ok) > 0:
                try:
                    _neg_folds = {}
                    for _sid in turnover_ok["strategy_id"]:
                        _sg = wf_results[wf_results["strategy_id"] == _sid]
                        _neg_folds[_sid] = set(_sg[_sg["mean_return"] < 0]["fold_idx"])
                    if _neg_folds:
                        _all_neg = set()
                        for _s in _neg_folds.values():
                            _all_neg |= _s
                        if _all_neg:
                            _overlap = sum(len(s & _all_neg) for s in _neg_folds.values()) / max(1, len(_neg_folds) * len(_all_neg))
                            mf[24] = float(_overlap)
                except Exception:
                    pass

            # [25] Opposite convexity: avg performance in worst portfolio quintile
            if len(wf_results) > 0 and len(turnover_ok) > 0:
                try:
                    _port_ret = wf_results[wf_results["strategy_id"].isin(turnover_ok["strategy_id"])].groupby("fold_idx")["mean_return"].mean()
                    if len(_port_ret) >= 5:
                        _worst_q = _port_ret.nsmallest(max(1, len(_port_ret) // 5))
                        _worst_folds = set(_worst_q.index)
                        _worst_perf = wf_results[wf_results["fold_idx"].isin(_worst_folds)]["mean_return"].mean()
                        mf[25] = float(_worst_perf) if not np.isnan(_worst_perf) else 0
                except Exception:
                    pass

            # [26-29] reserved = 0
            return np.nan_to_num(mf).tolist()

        # Build training data: per-fold meta-features → success label
        sids = set(turnover_ok["strategy_id"])
        meta_X = []; meta_y = []
        for _, row in wf_results[wf_results["strategy_id"].isin(sids)].iterrows():
            ts = pd.Timestamp(row["test_start"])
            mf = _build_meta_features(ts)
            # Check that at least the market features are available
            if all(v == 0 for v in mf[:4]):
                mkt = row.get("market","US")
                mi = market_indices.get(mkt, pd.DataFrame())
                if "close" not in mi.columns or mi.empty: continue
                mc = mi["close"]; pre = mc.loc[:ts]
                if len(pre)<60: continue
                mf[0] = float(pre.pct_change(20).iloc[-1]) if len(pre)>20 else 0
                mf[1] = float(pre.pct_change(60).iloc[-1]) if len(pre)>60 else 0
                mf[2] = float(pre.pct_change().rolling(20).std().iloc[-1]) if len(pre)>20 else 0
                mf[3] = float(pre.pct_change().rolling(60).std().iloc[-1]) if len(pre)>60 else 0
                mf = np.nan_to_num(mf).tolist()
            meta_X.append(mf)
            meta_y.append(1 if row["mean_return"]>0 else 0)

        if len(meta_X) >= 30:
            MX = np.array(meta_X, dtype=np.float32); MY = np.array(meta_y)
            np.nan_to_num(MX, copy=False)
            print("Meta-model v2 input dim: %d features x %d samples" % (MX.shape[1], MX.shape[0]))
            # Time-based split (70/30)
            split = int(len(MX)*0.7)
            gb = GradientBoostingClassifier(n_estimators=50, max_depth=2, random_state=CFG.seed)
            gb.fit(MX[:split], MY[:split])
            test_acc = float((gb.predict(MX[split:])==MY[split:]).mean())
            logger.info("Meta-model v2 test accuracy: %.2f (30-dim features)" % test_acc)

            # Score each surviving strategy's average meta-score
            strat_meta = {}
            for sid in sids:
                sg = wf_results[wf_results["strategy_id"]==sid]
                scores = []
                for _, r in sg.iterrows():
                    mkt = r.get("market","US"); ts = pd.Timestamp(r["test_start"])
                    mf = _build_meta_features(ts)
                    # Fallback: fill market features from per-market index
                    if all(v == 0 for v in mf[:4]):
                        mi = market_indices.get(mkt, pd.DataFrame())
                        if "close" not in mi.columns: continue
                        mc = mi["close"]; pre = mc.loc[:ts]
                        if len(pre)<60: continue
                        mf[0] = float(pre.pct_change(20).iloc[-1]) if len(pre)>20 else 0
                        mf[1] = float(pre.pct_change(60).iloc[-1]) if len(pre)>60 else 0
                        mf[2] = float(pre.pct_change().rolling(20).std().iloc[-1]) if len(pre)>20 else 0
                        mf[3] = float(pre.pct_change().rolling(60).std().iloc[-1]) if len(pre)>60 else 0
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
# CELL — Theme Crash Scoring + Hard Veto (Step 9)
# ============================================================
C.append(md('''\
## 15b. Theme Crash Scoring + Hard Veto

Per market × sector: compute theme crash score. If score > threshold → hard veto,
removing all strategies with matching primary sector from meta_scored.'''))

C.append(code('''\
theme_crash_scores = {}  # {market: {sector: float}}
theme_vetoed_strategies = []

STEP = "theme_crash_scoring"
if tracker.is_completed(STEP):
    tc_path = os.path.join(CFG.global_eval_dir, "theme_crash.json")
    if os.path.exists(tc_path):
        with open(tc_path) as f: theme_crash_scores = json.load(f)
    logger.info("[SKIP] %s" % STEP)
elif not CFG.enable_theme_crash or len(meta_scored) == 0:
    logger.info("[SKIP] theme crash (disabled or no strategies)")
    tracker.mark_completed(STEP, {"skipped": True})
else:
    logger.info("[RUN] %s" % STEP)
    for market in CFG.markets:
        if market not in ohlcv_data:
            continue
        smr = sector_mean_returns.get(market, {})
        if not isinstance(smr, dict) or "ticker_sectors" not in smr:
            continue
        ticker_sectors = smr["ticker_sectors"]
        sectors = list(set(ticker_sectors.values()))
        panel = ohlcv_data[market]
        scores = {}
        for sector in sectors:
            score = compute_theme_crash_score(panel, ticker_sectors, sector, CFG)
            scores[sector] = round(score, 4)
            if score > CFG.theme_crash_veto_threshold:
                logger.warning("THEME CRASH VETO: %s/%s score=%.3f > %.2f" % (
                    market, sector, score, CFG.theme_crash_veto_threshold))
        theme_crash_scores[market] = scores

    # Hard veto: remove strategies whose primary sector is vetoed
    vetoed_sectors = {}
    for mkt, scores in theme_crash_scores.items():
        for sec, score in scores.items():
            if score > CFG.theme_crash_veto_threshold:
                vetoed_sectors.setdefault(mkt, []).append(sec)

    before = len(meta_scored)
    if vetoed_sectors:
        def _is_vetoed(row):
            mkt = row.get("market", "")
            sid = row.get("strategy_id", "")
            sec = strategy_primary_sector.get(sid, "")
            return sec in vetoed_sectors.get(mkt, [])
        veto_mask = meta_scored.apply(_is_vetoed, axis=1)
        theme_vetoed_strategies = meta_scored[veto_mask]["strategy_id"].tolist()
        meta_scored = meta_scored[~veto_mask].copy()
        print("Theme crash hard veto: %d -> %d (vetoed %d strategies)" % (
            before, len(meta_scored), len(theme_vetoed_strategies)))
    else:
        print("Theme crash: no sectors vetoed")

    tc_path = os.path.join(CFG.global_eval_dir, "theme_crash.json")
    with open(tc_path, "w") as f:
        json.dump(theme_crash_scores, f, indent=2)
    tracker.mark_completed(STEP, {
        "scores": theme_crash_scores,
        "vetoed": theme_vetoed_strategies[:10],
        "n_vetoed": len(theme_vetoed_strategies),
    })

print("Theme crash scores:", {m: len(v) for m, v in theme_crash_scores.items()})
print("Theme-vetoed strategies: %d" % len(theme_vetoed_strategies))'''))

# ============================================================
# CELL — Trade Abstention (Step 10)
# ============================================================
C.append(md('''\
## 15c. Trade Abstention

Remove strategies with insufficient expected value after adjusting for
theme risk and regime conditions. Maximizes EV, not win-rate.'''))

C.append(code('''\
abstention_count = 0
STEP = "trade_abstention"

if tracker.is_completed(STEP):
    logger.info("[SKIP] %s" % STEP)
elif not CFG.enable_trade_abstention or len(meta_scored) == 0:
    logger.info("[SKIP] trade abstention (disabled or no strategies)")
    tracker.mark_completed(STEP, {"skipped": True})
else:
    logger.info("[RUN] %s" % STEP)
    before = len(meta_scored)
    keep_mask = []
    for _, row in meta_scored.iterrows():
        sid = row["strategy_id"]
        mkt = row.get("market", "")
        mean_ret = row.get("mean_return", 0)

        # Theme risk from crash scores
        sec = strategy_primary_sector.get(sid, "")
        theme_risk = theme_crash_scores.get(mkt, {}).get(sec, 0)

        # Regime ratio
        regime_ratio = row.get("regime_ratio", 1.0) if "regime_ratio" in row.index else 1.0

        # Adjusted EV
        adjusted_ev = mean_ret * (1 - theme_risk * 0.5) * max(0.5, regime_ratio)

        if adjusted_ev >= CFG.trade_abstention_min_ev:
            keep_mask.append(True)
        else:
            keep_mask.append(False)

    meta_scored = meta_scored[keep_mask].copy()
    abstention_count = before - len(meta_scored)
    print("Trade abstention: %d -> %d (abstained %d)" % (before, len(meta_scored), abstention_count))
    tracker.mark_completed(STEP, {"abstained": abstention_count})

print("After abstention: %d strategies" % len(meta_scored))'''))

# ============================================================
# CELL — Cross-Market Consistency Check (Step 11)
# ============================================================
C.append(md('''\
## 15d. Cross-Market Consistency

Compare US vs KR markets for contagion signals. If US dropped >3%
in past N days while KR is flat → contagion flag. Feeds meta-model.'''))

C.append(code('''\
cross_market_flags = {}
STEP = "cross_market"

if tracker.is_completed(STEP):
    cm_path = os.path.join(CFG.global_eval_dir, "cross_market.json")
    if os.path.exists(cm_path):
        with open(cm_path) as f: cross_market_flags = json.load(f)
    logger.info("[SKIP] %s" % STEP)
elif not CFG.enable_cross_market_consistency or len(market_indices) < 2:
    logger.info("[SKIP] cross-market (disabled or single market)")
    tracker.mark_completed(STEP, {"skipped": True})
else:
    logger.info("[RUN] %s" % STEP)
    lag = CFG.cross_market_contagion_lag_days

    for mkt_a in CFG.markets:
        for mkt_b in CFG.markets:
            if mkt_a >= mkt_b:
                continue
            mi_a = market_indices.get(mkt_a, pd.DataFrame())
            mi_b = market_indices.get(mkt_b, pd.DataFrame())
            if "close" not in mi_a.columns or "close" not in mi_b.columns:
                continue
            if mi_a.empty or mi_b.empty:
                continue

            # Check if mkt_a dropped >3% in past N days
            ret_a = mi_a["close"].pct_change(lag)
            ret_b = mi_b["close"].pct_change(lag)
            if len(ret_a) < lag or len(ret_b) < lag:
                continue

            latest_a = float(ret_a.iloc[-1]) if not np.isnan(ret_a.iloc[-1]) else 0
            latest_b = float(ret_b.iloc[-1]) if not np.isnan(ret_b.iloc[-1]) else 0

            flag_key = "%s_to_%s" % (mkt_a, mkt_b)
            contagion = False
            if latest_a < -0.03 and latest_b > -0.01:
                contagion = True
                print("CONTAGION WARNING: %s dropped %.1f%% but %s only %.1f%% (lag=%dd)" % (
                    mkt_a, latest_a * 100, mkt_b, latest_b * 100, lag))
            elif latest_b < -0.03 and latest_a > -0.01:
                contagion = True
                flag_key = "%s_to_%s" % (mkt_b, mkt_a)
                print("CONTAGION WARNING: %s dropped %.1f%% but %s only %.1f%% (lag=%dd)" % (
                    mkt_b, latest_b * 100, mkt_a, latest_a * 100, lag))

            cross_market_flags[flag_key] = {
                "contagion": contagion,
                "ret_%s" % mkt_a: round(latest_a, 4),
                "ret_%s" % mkt_b: round(latest_b, 4),
            }

    cm_path = os.path.join(CFG.global_eval_dir, "cross_market.json")
    with open(cm_path, "w") as f:
        json.dump(cross_market_flags, f, indent=2)
    tracker.mark_completed(STEP, cross_market_flags)

print("Cross-market flags:", cross_market_flags)'''))

# ============================================================
# CELL — Regime-Conditional Threshold Adaptation (Step 12)
# ============================================================
C.append(md('''\
## 15e. Regime-Conditional Thresholds

Adapt thresholds based on current VIX regime:
- High vol (>1.5): widen tri-state thresholds, raise meta-model threshold
- Low vol (<0.7): narrow thresholds, lower meta-model threshold
These inform meta-model and report; labeling uses adapted thresholds in next run.'''))

C.append(code('''\
adapted_thresholds = {}
STEP = "regime_thresholds"

if tracker.is_completed(STEP):
    at_path = os.path.join(CFG.global_eval_dir, "adapted_thresholds.json")
    if os.path.exists(at_path):
        with open(at_path) as f: adapted_thresholds = json.load(f)
    logger.info("[SKIP] %s" % STEP)
elif not CFG.enable_regime_adaptive_thresholds:
    logger.info("[SKIP] regime thresholds (disabled)")
    tracker.mark_completed(STEP, {"skipped": True})
else:
    logger.info("[RUN] %s" % STEP)
    current_vix_regime = 1.0
    if len(macro_data) > 0 and "vix_regime" in macro_data.columns:
        vr = macro_data["vix_regime"].iloc[-1]
        if not np.isnan(vr):
            current_vix_regime = float(vr)

    # Adapt tristate thresholds
    adapted_tristate = {}
    for fd, th in CFG.tristate_thresholds_pct.items():
        if current_vix_regime > 1.5:
            # High vol: widen thresholds
            adapted_tristate[str(fd)] = round(th * current_vix_regime, 2)
        elif current_vix_regime < 0.7:
            # Low vol: narrow thresholds
            adapted_tristate[str(fd)] = round(th * current_vix_regime, 2)
        else:
            adapted_tristate[str(fd)] = th

    # Adapt meta-model threshold
    if current_vix_regime > 1.5:
        adapted_meta_th = round(CFG.meta_model_threshold * 1.2, 3)
    elif current_vix_regime < 0.7:
        adapted_meta_th = round(CFG.meta_model_threshold * 0.8, 3)
    else:
        adapted_meta_th = CFG.meta_model_threshold

    adapted_thresholds = {
        "current_vix_regime": round(current_vix_regime, 3),
        "adapted_tristate_pct": adapted_tristate,
        "adapted_meta_threshold": adapted_meta_th,
        "original_meta_threshold": CFG.meta_model_threshold,
        "note": "Adapted thresholds apply to next run labeling; current run uses original",
    }

    at_path = os.path.join(CFG.global_eval_dir, "adapted_thresholds.json")
    with open(at_path, "w") as f:
        json.dump(adapted_thresholds, f, indent=2)
    tracker.mark_completed(STEP, adapted_thresholds)

print("Adapted thresholds:", json.dumps(adapted_thresholds, indent=2))'''))

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

        # Objective function for threshold optimization
        def _tune_objective(params):
            prec_th, turn_th, regime_th = params
            mask = (
                (meta_scored["mean_precision_buy"] >= prec_th) &
                (meta_scored["mean_turnover"] <= turn_th)
            )
            if "regime_ratio" in meta_scored.columns:
                mask = mask & (meta_scored["regime_ratio"] >= regime_th)
            subset = meta_scored[mask]
            if len(subset) < 2:
                return 999.0  # infeasible
            sids = set(subset["strategy_id"])
            sw = wf_results[wf_results["strategy_id"].isin(sids)]
            if sw.empty:
                return 999.0
            port_ret = sw.groupby("fold_idx")["mean_return"].mean()
            ev = float(port_ret.mean())
            pr = float(subset["mean_precision_buy"].mean())
            if pr < 0.6:
                return 999.0  # constraint violation

            # Diversity bonus: reward unique strategy types
            n_types = subset["type"].nunique() if "type" in subset.columns else 1
            diversity_bonus = 0.01 * n_types

            return -(ev + diversity_bonus)  # negative for minimization

        if CFG.meta_v2_use_scipy_optimize:
            try:
                from scipy.optimize import minimize
                bounds = [(0.50, 0.75), (1.0, CFG.max_turnover), (0.3, 0.9)]
                for _ in range(CFG.meta_v2_bayes_n_calls):
                    x0 = [np.random.uniform(b[0], b[1]) for b in bounds]
                    res = minimize(_tune_objective, x0, method="Nelder-Mead",
                                   options={"maxiter": 50, "xatol": 0.01, "fatol": 1e-6})
                    if res.fun < 999 and -res.fun > best_ev:
                        best_ev = -res.fun
                        best_params = {
                            "precision_th": round(float(res.x[0]), 3),
                            "turnover_th": round(float(res.x[1]), 2),
                            "regime_th": round(float(res.x[2]), 3),
                            "ev": round(best_ev, 6),
                            "n_strats": int(len(meta_scored[
                                (meta_scored["mean_precision_buy"] >= res.x[0]) &
                                (meta_scored["mean_turnover"] <= res.x[1])
                            ])),
                            "optimizer": "scipy_nelder_mead",
                        }
                print("scipy optimize: best EV=%.6f" % best_ev if best_ev > -999 else "scipy optimize: no feasible solution")
            except ImportError:
                logger.warning("scipy not available, falling back to random search")
                CFG.meta_v2_use_scipy_optimize = False

        if not CFG.meta_v2_use_scipy_optimize:
            # Fallback: random search
            for _ in range(CFG.n_bayes_iterations):
                prec_th = np.random.uniform(0.50, 0.75)
                turn_th = np.random.uniform(1.0, CFG.max_turnover)
                regime_th = np.random.uniform(0.3, 0.9)
                obj = _tune_objective([prec_th, turn_th, regime_th])
                if obj < 999 and -obj > best_ev:
                    best_ev = -obj
                    best_params = {"precision_th": round(prec_th, 3), "turnover_th": round(turn_th, 2),
                                   "regime_th": round(regime_th, 3), "ev": round(best_ev, 6),
                                   "n_strats": int(len(meta_scored[
                                       (meta_scored["mean_precision_buy"] >= prec_th) &
                                       (meta_scored["mean_turnover"] <= turn_th)
                                   ])), "optimizer": "random_search"}

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

        # --- Strategy Mortality: kill degraded strategies ---
        mortality_killed = []
        if CFG.enable_strategy_mortality and len(df) > 0 and len(wf_results) > 0:
            kill_ids = set()
            for sid in df["strategy_id"]:
                sg = wf_results[wf_results["strategy_id"] == sid].sort_values("fold_idx")
                if len(sg) < 2:
                    continue
                # Rolling Sharpe over last N folds
                n_window = min(CFG.mortality_rolling_sharpe_window, len(sg))
                recent = sg.tail(n_window)
                rolling_sharpe = float(recent["sharpe"].mean())
                # Consecutive negative folds
                rets = sg["mean_return"].values
                consec_neg = 0
                max_consec_neg = 0
                for r in reversed(rets):
                    if r < 0:
                        consec_neg += 1
                        max_consec_neg = max(max_consec_neg, consec_neg)
                    else:
                        break
                if rolling_sharpe < CFG.mortality_sharpe_kill_threshold:
                    kill_ids.add(sid)
                    mortality_killed.append(sid)
                elif max_consec_neg >= CFG.mortality_consecutive_negative_folds:
                    kill_ids.add(sid)
                    mortality_killed.append(sid)
            if kill_ids:
                before_mort = len(df)
                df = df[~df["strategy_id"].isin(kill_ids)].copy()
                print("Strategy mortality: %d -> %d (killed %d)" % (before_mort, len(df), len(kill_ids)))

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

            # --- Macro Veto: halve scores during extreme macro stress ---
            if CFG.enable_macro_features and len(macro_data) > 0:
                try:
                    latest_macro = macro_data.iloc[-1]
                    vr = latest_macro.get("vix_regime", 1.0)
                    yc_inv = latest_macro.get("yield_curve_inverted", 0.0)
                    if not np.isnan(vr) and not np.isnan(yc_inv):
                        if vr > CFG.macro_veto_vix_multiple and yc_inv > 0.5:
                            before_veto = df["composite"].mean()
                            df["composite"] = df["composite"] * 0.5
                            print("MACRO VETO: VIX regime=%.2f (>%.1fx), yield curve inverted -> composite halved (%.4f -> %.4f)" % (
                                vr, CFG.macro_veto_vix_multiple, before_veto, df["composite"].mean()))
                        else:
                            print("Macro check: VIX regime=%.2f, yield_inverted=%s -> no veto" % (vr, bool(yc_inv > 0.5)))
                except Exception as _e:
                    logger.debug("Macro veto check error: %s" % str(_e)[:60])

            # --- Sector Alpha Gate: penalize low-alpha sectors ---
            if CFG.enable_sector_rotation and sector_alpha_scores:
                for idx, row in df.iterrows():
                    sid = row["strategy_id"]
                    mkt = row.get("market", "")
                    sec = strategy_primary_sector.get(sid, "")
                    sas = sector_alpha_scores.get(mkt, {})
                    if isinstance(sas, dict) and sec in sas:
                        if sas[sec] < CFG.sector_alpha_threshold:
                            df.at[idx, "composite"] = df.at[idx, "composite"] * 0.7

            # Keep top N per market
            parts = []
            for (m,h), g in df.groupby(["market","horizon"]):
                parts.append(g.nlargest(CFG.portfolio_max_strategies, "composite"))
            scored = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
            scored = scored.sort_values("composite", ascending=False).reset_index(drop=True)
            scored["rank"] = range(1, len(scored)+1)
            scored.to_parquet(sc_path)
            tracker.mark_completed(STEP, {"n":len(scored), "mortality_killed": mortality_killed[:10]})

print("Scored: %d" % len(scored))
if len(scored)>0:
    print(scored[["rank","strategy_id","market","horizon","composite",
                  "mean_sharpe","mean_precision_buy","mean_turnover"]].head(20).to_string(index=False))'''))

# ============================================================
# CELL — Failure-Triggered Allocation (v4)
# ============================================================
C.append(md('''\
## 17b. Failure-Triggered Allocation (v4)

Detect core strategy decay and shift capital to complementary strategies:
1. Top 3 = "core", rest = "complementary"
2. Confidence = recent_sharpe / overall_sharpe
3. If confidence < threshold → shift capital to negatively correlated complements'''))

C.append(code('''\
failure_allocation = {"mode": "normal", "adjustments": {}}
STEP = "failure_allocation"
fa_path = os.path.join(CFG.global_eval_dir, "failure_allocation.json")

if tracker.is_completed(STEP):
    if os.path.exists(fa_path):
        with open(fa_path) as f: failure_allocation = json.load(f)
    logger.info("[SKIP] %s" % STEP)
elif not CFG.enable_failure_allocation or len(scored) < 4 or len(wf_results) == 0:
    logger.info("[SKIP] failure allocation (disabled or too few strategies)")
    tracker.mark_completed(STEP, {"skipped": True})
else:
    logger.info("[RUN] %s" % STEP)

    # Identify core (top 3) and complementary strategies
    core_ids = scored.head(3)["strategy_id"].tolist()
    comp_ids = scored.iloc[3:]["strategy_id"].tolist()

    # Build correlation matrix
    pivot = wf_results.pivot_table(values="mean_return", index="fold_idx",
                                   columns="strategy_id", aggfunc="first")
    all_sids = core_ids + comp_ids
    pivot = pivot[[c for c in pivot.columns if c in all_sids]].dropna(axis=1, how="all").fillna(0)
    corr_matrix = pivot.corr() if pivot.shape[1] >= 2 else pd.DataFrame()

    adjustments = {}
    any_failure = False

    for sid in core_ids:
        sg = wf_results[wf_results["strategy_id"] == sid].sort_values("fold_idx")
        if len(sg) < 2:
            continue

        overall_sharpe = float(sg["sharpe"].mean())
        n_recent = min(CFG.failure_decay_lookback_folds, len(sg))
        recent_sharpe = float(sg.tail(n_recent)["sharpe"].mean())

        confidence = recent_sharpe / overall_sharpe if abs(overall_sharpe) > 1e-8 else 1.0

        if confidence < CFG.failure_confidence_threshold:
            any_failure = True
            # Find complementary strategies negatively correlated with failing core
            best_comp = None
            best_neg_corr = 0
            for comp_sid in comp_ids:
                if sid in corr_matrix.columns and comp_sid in corr_matrix.columns:
                    c = float(corr_matrix.loc[sid, comp_sid])
                    if c < best_neg_corr:
                        best_neg_corr = c
                        best_comp = comp_sid

            shift_pct = min(CFG.failure_capital_shift_pct, abs(best_neg_corr) * CFG.failure_capital_shift_pct) if best_comp else 0

            adjustments[sid] = {
                "action": "reduce",
                "confidence": round(confidence, 3),
                "shift_pct": round(shift_pct, 1),
                "shift_to": best_comp,
                "reason": "core decay (confidence=%.2f < %.2f)" % (confidence, CFG.failure_confidence_threshold),
            }
            if best_comp:
                adjustments[best_comp] = {
                    "action": "increase",
                    "shift_pct": round(shift_pct, 1),
                    "from_sid": sid,
                    "reason": "complement to failing core (corr=%.2f)" % best_neg_corr,
                }
            print("FAILURE SHIFT: %s confidence=%.2f -> shift %.1f%% to %s" % (
                sid[:30], confidence, shift_pct, best_comp or "none"))

    failure_allocation = {
        "mode": "failure_shift" if any_failure else "normal",
        "adjustments": adjustments,
    }

    with open(fa_path, "w") as f:
        json.dump(failure_allocation, f, indent=2)
    tracker.mark_completed(STEP, {"mode": failure_allocation["mode"], "n_adjustments": len(adjustments)})

print("Failure allocation mode:", failure_allocation["mode"])
if failure_allocation["adjustments"]:
    for sid, adj in failure_allocation["adjustments"].items():
        print("  %s: %s (%.1f%%)" % (sid[:30], adj["action"], adj.get("shift_pct", 0)))'''))

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
## 19. Signal → Rule → Portfolio Architecture (v4)

**Veto Priority Hierarchy (v4):**
```
1. THEME CRASH HARD VETO (score > 0.7)       → removes strategies entirely
2. MACRO VETO (VIX > 2x + yield inverted)    → halves composite scores
3. TRADE ABSTENTION (adjusted_ev < min)       → removes low-EV strategies
4. META-MODEL v2 GATE (30-dim, interactions)  → removes low meta-score
5. FAILURE ALLOCATION (core confidence < 0.4) → shifts capital to complements
6. SECTOR ALPHA GATE (rotation model)         → penalizes composite (x0.7)
7. STRATEGY MORTALITY (rolling Sharpe < 0)    → kills degraded strategies
8. DIVERSITY GATE (max_correlation > 0.6)     → penalizes redundant strategies
```

```
┌─────────────────────────────────────────────┐
│  Model Output (decile/tree/logistic probs)  │
│  V4: + meanrev, volshock, CMR candidates    │
│  Dual labels: absolute + sector-neutral     │
└─────────────────┬───────────────────────────┘
                  ▼
┌─────────────────────────────────────────────┐
│  Signal Layer: Tri-state (+1 / 0 / -1)     │
│  BUY only if excess return ≥ threshold      │
└─────────────────┬───────────────────────────┘
                  ▼
┌─────────────────────────────────────────────┐
│  Meta-Model v2: P(success | market state)   │
│  30-dim features + interactions → NO TRADE  │
└─────────────────┬───────────────────────────┘
                  ▼
┌─────────────────────────────────────────────┐
│  Theme Crash + Trade Abstention             │
│  Hard veto sectors, remove low-EV strategies│
└─────────────────┬───────────────────────────┘
                  ▼
┌─────────────────────────────────────────────┐
│  Rule Layer (data-tuned, scipy-optimized):  │
│  • Market-specific precision/Sharpe/turnover│
│  • CVaR ≤ 3× avg win                        │
│  • Beta-neutral Sharpe check                │
│  • Cost survival ≥ 2 scenarios              │
│  • Regime ratio ≥ 70%                        │
│  • Sector alpha gate + Strategy mortality   │
└─────────────────┬───────────────────────────┘
                  ▼
┌─────────────────────────────────────────────┐
│  Portfolio Engine (v4):                      │
│  • Diversity optimizer (greedy selection)    │
│  • Failure-triggered allocation              │
│  • No strategy > 30% risk budget            │
│  • Cross-market + cross-type diversification│
│  • CVaR must improve vs single strategy     │
│  • Trading Governance Report                 │
└─────────────────────────────────────────────┘
```'''))

# ============================================================
# CELL — Ablation & Module Impact Analysis
# ============================================================
C.append(md('''\
## 19b. Ablation & Module Impact Analysis

Audit which data modules are active and their contribution to the feature set.
Saves baseline on first run for future comparison.'''))

C.append(code('''\
print("="*70)
print("MODULE IMPACT ANALYSIS")
print("="*70)

active_modules = {
    "axis_a_liquidity": CFG.enable_liquidity_features,
    "axis_b_fundamentals": CFG.enable_fundamental_features,
    "axis_c_macro": CFG.enable_macro_features,
    "axis_d_sectors": CFG.enable_sector_features,
    "axis_e_sentiment": CFG.enable_sentiment_proxy,
    "sector_neutral_labels": CFG.enable_sector_neutral_labels,
    "sector_rotation": CFG.enable_sector_rotation,
    "theme_crash": CFG.enable_theme_crash,
    "trade_abstention": CFG.enable_trade_abstention,
    "cross_market_consistency": CFG.enable_cross_market_consistency,
    "regime_adaptive_thresholds": CFG.enable_regime_adaptive_thresholds,
    "strategy_mortality": CFG.enable_strategy_mortality,
    "v4_mean_reversion": CFG.enable_mean_reversion,
    "v4_volatility_shock": CFG.enable_volatility_shock,
    "v4_cross_market_rotation": CFG.enable_cross_market_rotation,
    "v4_diversity_optimizer": CFG.enable_diversity_optimizer,
    "v4_failure_allocation": CFG.enable_failure_allocation,
    "v4_sn_candidates": CFG.enable_sn_candidates,
}
print("\\nActive modules:")
for mod, on in active_modules.items():
    print("  %s: %s" % (mod, "ON" if on else "OFF"))

# Count features by axis in the feature panels
axis_counts = {"price_volume": 0, "liquidity": 0, "fundamental": 0, "sector": 0, "mean_reversion": 0, "vol_shock": 0, "cross_market": 0, "other": 0}
sample_market = list(feature_panels.keys())[0] if feature_panels else None
if sample_market:
    fp = feature_panels[sample_market]
    feat_cols = [c for c in fp.columns if not c.startswith("fwd_") and not c.startswith("label_") and not c.endswith("_decile")]
    for col in feat_cols:
        if any(col.startswith(p) for p in ["log_dollar_vol", "amihud", "vol_imbalance", "turnover_ratio", "gap_freq"]):
            axis_counts["liquidity"] += 1
        elif any(col.startswith(p) for p in ["high_52w_pct", "log_market_cap", "trailingPE", "priceToBook", "returnOnEquity"]):
            axis_counts["fundamental"] += 1
        elif col.startswith("sector_relative"):
            axis_counts["sector"] += 1
        elif any(col.startswith(p) for p in ["zscore_", "bollinger_pct_", "vol_overshoot_", "rsi_", "mean_revert_"]):
            axis_counts["mean_reversion"] += 1
        elif any(col.startswith(p) for p in ["vol_expansion_", "vol_acceleration", "volume_shock_", "realized_vs_implied", "drawdown_speed"]):
            axis_counts["vol_shock"] += 1
        elif col.startswith("cmr_"):
            axis_counts["cross_market"] += 1
        elif any(col.startswith(p) for p in ["mom_", "vol_", "vol_change", "market_mom", "market_vol", "regime_", "market_relative"]):
            axis_counts["price_volume"] += 1
        else:
            axis_counts["other"] += 1
    print("\\nFeature count by axis (base features, excl. deciles):")
    for axis, cnt in axis_counts.items():
        print("  %-20s: %d" % (axis, cnt))
    print("  %-20s: %d" % ("TOTAL", sum(axis_counts.values())))
    print("  %-20s: %d" % ("+ decile versions", len([c for c in fp.columns if c.endswith("_decile")])))

# Meta-model dimension check
print("\\nMeta-model v2 feature vector: 30 dimensions")
print("  [0-3]   Market: 20d mom, 60d mom, 20d vol, 60d vol")
print("  [4-8]   Macro: yield_curve_slope, vix_regime, dxy/gold/oil momentum")
print("  [9]     Sentiment: vix_term_structure")
print("  [10]    Sector alpha: median sector rotation score")
print("  [11]    Theme crash: max ThemeCrashScore")
print("  [12]    Absolute alpha strength")
print("  [13]    SN alpha strength")
print("  [14]    Regime alignment")
print("  [15]    Cross-market contagion flag")
print("  [16]    Rolling Sharpe (mortality)")
print("  [17]    Trade abstention EV")
print("  [18-19] Reserved padding")
print("  [20]    Strategy type diversity (v4)")
print("  [21]    Avg portfolio correlation (v4)")
print("  [22]    Marginal CVaR contribution (v4)")
print("  [23]    Regime-conditional Sharpe (v4)")
print("  [24]    Shared failure mode (v4)")
print("  [25]    Opposite convexity score (v4)")
print("  [26-29] Reserved (v4)")

# Macro veto status
if CFG.enable_macro_features and len(macro_data) > 0:
    latest = macro_data.iloc[-1]
    print("\\nLatest macro state:")
    print("  yield_curve_slope: %.4f" % latest.get("yield_curve_slope", 0))
    print("  yield_curve_inverted: %s" % bool(latest.get("yield_curve_inverted", 0) > 0.5))
    print("  vix_regime: %.2f" % latest.get("vix_regime", 1))

# Save/compare baseline
baseline_path = os.path.join(CFG.drive_root, "ablation_baseline.json")
current_report = {
    "active_modules": active_modules,
    "feature_counts": axis_counts,
    "n_scored": len(scored),
    "mean_composite": float(scored["composite"].mean()) if len(scored) > 0 else 0,
    "mean_sharpe": float(scored["mean_sharpe"].mean()) if len(scored) > 0 and "mean_sharpe" in scored.columns else 0,
}
if os.path.exists(baseline_path):
    with open(baseline_path) as f:
        baseline = json.load(f)
    print("\\nComparison vs baseline:")
    for k in ["n_scored", "mean_composite", "mean_sharpe"]:
        bv = baseline.get(k, 0); cv = current_report.get(k, 0)
        delta = cv - bv
        print("  %s: %.4f -> %.4f (%+.4f)" % (k, bv, cv, delta))
else:
    with open(baseline_path, "w") as f:
        json.dump(current_report, f, indent=2)
    print("\\nBaseline saved to %s (first run)" % baseline_path)

print("="*70)'''))

# ============================================================
# CELL — Trading Governance Report (v4)
# ============================================================
C.append(md('''\
## 19c. Trading Governance Report (v4)

Actionable output: WHEN TO TRADE, WHEN NOT TO TRADE, RISK SIZING, WHAT IS FAILING,
and SYSTEM TRUST LEVEL.'''))

C.append(code('''\
print("=" * 70)
print("TRADING GOVERNANCE REPORT (v4)")
print("=" * 70)

# 1. WHEN TO TRADE
print("\\n--- WHEN TO TRADE ---")
if len(scored) > 0:
    for _, row in scored.iterrows():
        ms = row.get("meta_score", 1.0)
        if ms >= 0.7:
            conf = "HIGH"
        elif ms >= 0.5:
            conf = "MEDIUM"
        else:
            conf = "LOW"
        print("  [%s] %s | Sharpe=%.2f | Precision=%.0f%% | Composite=%.3f" % (
            conf, row["strategy_id"][:40], row["mean_sharpe"],
            row["mean_precision_buy"] * 100, row["composite"]))
else:
    print("  NO strategies survived. Do not trade.")

# 2. WHEN NOT TO TRADE
print("\\n--- WHEN NOT TO TRADE ---")
active_vetoes = []
if CFG.enable_macro_features and len(macro_data) > 0:
    latest = macro_data.iloc[-1]
    vr = latest.get("vix_regime", 1.0)
    yci = latest.get("yield_curve_inverted", 0.0)
    if not np.isnan(vr) and vr > CFG.macro_veto_vix_multiple:
        active_vetoes.append("MACRO: VIX regime=%.2f (>%.1fx)" % (vr, CFG.macro_veto_vix_multiple))
    if not np.isnan(yci) and yci > 0.5:
        active_vetoes.append("MACRO: Yield curve inverted")
if theme_vetoed_strategies:
    active_vetoes.append("THEME CRASH: %d strategies vetoed" % len(theme_vetoed_strategies))
if cross_market_flags:
    for k, v in cross_market_flags.items():
        if v.get("contagion", False):
            active_vetoes.append("CONTAGION: %s" % k)
if failure_allocation.get("mode") == "failure_shift":
    active_vetoes.append("CORE DECAY: %d strategies shifting capital" % len(failure_allocation.get("adjustments", {})))
if active_vetoes:
    for v in active_vetoes:
        print("  [VETO] %s" % v)
else:
    print("  No active vetoes.")

# 3. RISK SIZING
print("\\n--- RISK SIZING ---")
n_strats = len(scored) if len(scored) > 0 else 0
if n_strats > 0:
    base_weight = 100.0 / n_strats
    print("  Base: equal-weight %.1f%% per strategy (%d strategies)" % (base_weight, n_strats))
    if failure_allocation.get("adjustments"):
        print("  Failure adjustments active:")
        for sid, adj in failure_allocation["adjustments"].items():
            print("    %s: %s %.1f%%" % (sid[:30], adj["action"], adj.get("shift_pct", 0)))
else:
    print("  No strategies to allocate.")

# 4. WHAT IS FAILING FIRST
print("\\n--- WHAT IS FAILING FIRST ---")
if 'mortality_killed' in dir() and mortality_killed:
    print("  Mortality kills: %d strategies" % len(mortality_killed))
    for mk in mortality_killed[:5]:
        print("    - %s" % mk)
if theme_vetoed_strategies:
    print("  Theme vetoes: %d strategies" % len(theme_vetoed_strategies))
if abstention_count > 0:
    print("  Trade abstention: %d strategies removed" % abstention_count)
if not (mortality_killed if 'mortality_killed' in dir() else []) and not theme_vetoed_strategies and abstention_count == 0:
    print("  Nothing failing currently.")

# 5. SYSTEM TRUST LEVEL
print("\\n--- SYSTEM TRUST LEVEL ---")
trust_score = 1.0
trust_reasons = []
if n_strats <= 1:
    trust_score *= 0.6
    trust_reasons.append("monoculture (%d strategy)" % n_strats)
if active_vetoes:
    trust_score *= 0.7
    trust_reasons.append("%d active vetoes" % len(active_vetoes))
if failure_allocation.get("mode") == "failure_shift":
    trust_score *= 0.8
    trust_reasons.append("failure_shift mode active")

# Check strategy type diversity
if len(scored) > 0 and "type" in scored.columns:
    n_types = scored["type"].nunique()
    if n_types <= 1:
        trust_score *= 0.7
        trust_reasons.append("single strategy type")
    elif n_types >= 3:
        trust_score = min(1.0, trust_score * 1.1)  # diversity bonus

trust_level = "HIGH" if trust_score >= 0.7 else "MEDIUM" if trust_score >= 0.4 else "LOW"
recommendation = "TRADE" if trust_score >= 0.7 else "REDUCE SIZE" if trust_score >= 0.4 else "DO NOT TRADE"

print("  Trust Score: %.2f" % trust_score)
print("  Trust Level: %s" % trust_level)
if trust_reasons:
    for r in trust_reasons:
        print("    - %s" % r)
print("  Recommendation: %s" % recommendation)
print("=" * 70)'''))

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
              "Dist-safe","Cost-stress","Deduped","Diversity","Regime","Turnover","Meta-gate",
              "Theme veto","Abstention","Failure alloc","Scored"]
    _post_meta = len(meta_scored) + len(theme_vetoed_strategies) + abstention_count
    _post_veto = _post_meta - len(theme_vetoed_strategies)
    _post_abstain = _post_veto - abstention_count
    _n_fa = len(failure_allocation.get("adjustments", {})) if "failure_allocation" in dir() else 0
    counts = [
        len(all_candidates), len(edge_results),
        wf_results["strategy_id"].nunique() if len(wf_results)>0 else 0,
        len(filtered), len(beta_filtered), len(dist_safe),
        len(cost_survived), len(deduped), len(deduped), len(regime_ok),
        len(turnover_ok), _post_meta,
        _post_veto, _post_abstain,
        len(scored) + _n_fa,
        len(scored),
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
print("PIPELINE v4 COMPLETE — Multi-Strategy Survivability & Failure-Aware Intelligence")
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

    # Count SN strategies
    _sn_strats = [s for s in scored["strategy_id"] if "_sn_" in s] if len(scored) > 0 else []

    # V4: Strategy type distribution
    _type_dist = scored["type"].value_counts().to_dict() if len(scored) > 0 and "type" in scored.columns else {}

    report = {
        "version":"v4", "markets":CFG.markets, "horizons":CFG.forward_days_list,
        "tri_state_thresholds":CFG.tristate_thresholds_pct,
        "cost_stress_scenarios":[list(s) for s in CFG.cost_stress_scenarios],
        "funnel":{"candidates":len(all_candidates),"edge":len(edge_results),
                  "wf":wf_results["strategy_id"].nunique() if len(wf_results)>0 else 0,
                  "filtered":len(filtered),"beta_neutral":len(beta_filtered),
                  "dist_safe":len(dist_safe),"cost_survived":len(cost_survived),
                  "deduped":len(deduped),"regime":len(regime_ok),
                  "turnover":len(turnover_ok),"meta_gated":len(meta_scored),
                  "theme_vetoed":len(theme_vetoed_strategies),
                  "abstained":abstention_count,
                  "scored":len(scored)},
        "tuned_rules":tuned_rules, "portfolio":portfolio_results,
        "top_strategy":b["strategy_id"],
        "active_modules": active_modules,
        "sector_neutral": {
            "enabled": CFG.enable_sector_neutral_labels,
            "n_sn_strategies": len(_sn_strats),
        },
        "sector_rotation": {
            "enabled": CFG.enable_sector_rotation,
            "sector_scores": sector_alpha_scores,
            "gated_sectors": {m: [s for s, v in secs.items() if v < CFG.sector_alpha_threshold]
                              for m, secs in sector_alpha_scores.items()} if sector_alpha_scores else {},
        },
        "theme_crash": {
            "enabled": CFG.enable_theme_crash,
            "scores": theme_crash_scores,
            "vetoed": theme_vetoed_strategies[:20],
        },
        "trade_abstention": {
            "enabled": CFG.enable_trade_abstention,
            "abstention_count": abstention_count,
        },
        "cross_market": {
            "flags": cross_market_flags,
        },
        "regime_adaptive": {
            "adapted_thresholds": adapted_thresholds,
        },
        "strategy_mortality": {
            "enabled": CFG.enable_strategy_mortality,
            "killed": mortality_killed[:20] if 'mortality_killed' in dir() else [],
            "n_killed": len(mortality_killed) if 'mortality_killed' in dir() else 0,
        },
        "governance": {
            "trust_score": round(trust_score, 3),
            "trust_level": trust_level,
            "active_vetoes": active_vetoes,
            "failure_mode": failure_allocation.get("mode", "normal"),
            "failing_first": {
                "mortality_kills": len(mortality_killed) if 'mortality_killed' in dir() else 0,
                "theme_vetoes": len(theme_vetoed_strategies),
                "abstentions": abstention_count,
            },
            "recommendation": recommendation,
        },
        "strategy_diversity": {
            "n_types": len(_type_dist),
            "type_distribution": _type_dist,
            "n_strategies": len(scored),
        },
        "failure_allocation": failure_allocation,
    }
    rp = os.path.join(CFG.drive_root, 'report_v4.json')
    with open(rp,'w') as f: json.dump(report, f, indent=2)
    print("\\nReport:", rp)
else:
    print("\\nNo viable strategies. The filters correctly identified no edge.")
    print("This is a VALID outcome — better than false positives.")
print("\\n"+"="*70)'''))


# ============================================================
# CELL — Feedback Loop Integration (v4)
# ============================================================
C.append(md('''\
## 21. Feedback Loop (v4)

Saves current run state and compares with previous runs.
Detects degradation, overfitting drift, and strategy stability.'''))

C.append(code('''\
STEP = "feedback_loop"
fb_path = os.path.join(CFG.drive_root, "feedback_state.json")

if tracker.is_completed(STEP):
    logger.info("[SKIP] %s" % STEP)
else:
    logger.info("[RUN] %s" % STEP)

    # Current state
    current_state = {
        "n_scored": len(scored),
        "mean_composite": float(scored["composite"].mean()) if len(scored) > 0 else 0,
        "mean_sharpe": float(scored["mean_sharpe"].mean()) if len(scored) > 0 and "mean_sharpe" in scored.columns else 0,
        "strategy_types": list(scored["type"].unique()) if len(scored) > 0 and "type" in scored.columns else [],
        "markets_represented": list(scored["market"].unique()) if len(scored) > 0 and "market" in scored.columns else [],
        "adapted_thresholds": adapted_thresholds,
        "failure_allocation_mode": failure_allocation.get("mode", "normal"),
        "active_vetoes": active_vetoes if "active_vetoes" in dir() else [],
        "trust_score": trust_score if "trust_score" in dir() else 1.0,
    }

    # Compare with previous state
    if os.path.exists(fb_path):
        with open(fb_path) as f:
            prev_state = json.load(f)

        print("\\n--- FEEDBACK LOOP: Cross-Run Comparison ---")

        # All strategies died
        if current_state["n_scored"] == 0 and prev_state.get("n_scored", 0) > 0:
            print("  [WARNING] All strategies died! Previous run had %d." % prev_state["n_scored"])
            print("  [RECOMMEND] Consider relaxing thresholds (precision, Sharpe, turnover).")

        # Sharpe degraded >30%
        prev_sharpe = prev_state.get("mean_sharpe", 0)
        curr_sharpe = current_state["mean_sharpe"]
        if prev_sharpe > 0 and curr_sharpe < prev_sharpe * 0.7:
            print("  [WARNING] Sharpe degraded %.0f%%: %.3f -> %.3f" % (
                (1 - curr_sharpe / prev_sharpe) * 100, prev_sharpe, curr_sharpe))
            print("  [RECOMMEND] Investigate regime change or strategy decay.")

        # Strategy count increased >50%
        prev_n = prev_state.get("n_scored", 0)
        if prev_n > 0 and current_state["n_scored"] > prev_n * 1.5:
            print("  [WARNING] Strategy count increased %.0f%%: %d -> %d" % (
                (current_state["n_scored"] / prev_n - 1) * 100, prev_n, current_state["n_scored"]))
            print("  [RECOMMEND] Check for potential overfitting (loosened filters?).")

        # Stable
        if (current_state["n_scored"] > 0 and
            (prev_n == 0 or abs(current_state["n_scored"] - prev_n) / max(1, prev_n) < 0.3) and
            (prev_sharpe <= 0 or curr_sharpe >= prev_sharpe * 0.7)):
            print("  [STABLE] Pipeline metrics are consistent with previous run.")

        # New strategy types
        prev_types = set(prev_state.get("strategy_types", []))
        curr_types = set(current_state["strategy_types"])
        new_types = curr_types - prev_types
        if new_types:
            print("  [INFO] New strategy types: %s" % list(new_types))
        lost_types = prev_types - curr_types
        if lost_types:
            print("  [INFO] Lost strategy types: %s" % list(lost_types))
    else:
        print("\\n--- FEEDBACK LOOP: First run (baseline saved) ---")

    # Save current state
    with open(fb_path, "w") as f:
        json.dump(current_state, f, indent=2)

    tracker.mark_completed(STEP, {"n_scored": current_state["n_scored"]})
    print("Feedback state saved to %s" % fb_path)'''))

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
