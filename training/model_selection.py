"""
Model Selection & Walk-Forward Validation
------------------------------------------
- WalkForwardValidator: temporal train/val/test splits with embargo gap
- Evaluation metrics: IC, ICIR, Sharpe, Drawdown, Calmar, Hit Ratio, etc.
- Statistical significance: paired t-test, block bootstrap, SPA test
- Model comparison dashboard output
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from scipy import stats as sp_stats


# ---------------------------------------------------------------------------
# Walk-Forward Validator
# ---------------------------------------------------------------------------

@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward validation."""
    train_start: str                    # e.g., "2015-01-01"
    test_end: str                       # e.g., "2025-01-01"
    train_min_months: int = 36          # minimum training window
    val_months: int = 6                 # validation window
    test_months: int = 6                # test window
    step_months: int = 6               # step size between folds
    embargo_days: int = 21             # gap to prevent target leakage
    expanding: bool = True             # expanding vs rolling window


@dataclass
class FoldSplit:
    """A single train/val/test fold."""
    fold_idx: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


class WalkForwardValidator:
    """Generates temporal walk-forward folds with embargo."""

    def __init__(self, config: WalkForwardConfig):
        self.config = config

    def generate_folds(self, dates: pd.DatetimeIndex) -> List[FoldSplit]:
        """Generate walk-forward folds from available dates.

        Args:
            dates: Sorted unique dates in the dataset.

        Returns:
            List of FoldSplit objects.
        """
        cfg = self.config
        overall_start = pd.Timestamp(cfg.train_start)
        overall_end = pd.Timestamp(cfg.test_end)

        folds = []
        fold_idx = 0

        # First possible test start
        test_start = overall_start + pd.DateOffset(months=cfg.train_min_months + cfg.val_months)

        while test_start + pd.DateOffset(months=cfg.test_months) <= overall_end:
            test_end = test_start + pd.DateOffset(months=cfg.test_months)
            val_end = test_start - pd.DateOffset(days=cfg.embargo_days)
            val_start = val_end - pd.DateOffset(months=cfg.val_months)

            if cfg.expanding:
                train_start = overall_start
            else:
                train_start = val_start - pd.DateOffset(months=cfg.train_min_months)

            train_end = val_start - pd.DateOffset(days=cfg.embargo_days)

            # Validate that we have dates in each window
            train_dates = dates[(dates >= train_start) & (dates <= train_end)]
            val_dates = dates[(dates >= val_start) & (dates <= val_end)]
            test_dates = dates[(dates >= test_start) & (dates <= test_end)]

            if len(train_dates) > 50 and len(val_dates) > 10 and len(test_dates) > 10:
                folds.append(FoldSplit(
                    fold_idx=fold_idx,
                    train_start=train_start,
                    train_end=train_end,
                    val_start=val_start,
                    val_end=val_end,
                    test_start=test_start,
                    test_end=test_end,
                ))
                fold_idx += 1

            test_start += pd.DateOffset(months=cfg.step_months)

        return folds

    def split_data(self, panel: pd.DataFrame, fold: FoldSplit
                   ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split panel data according to a fold specification.

        Args:
            panel: DataFrame with DatetimeIndex (or MultiIndex with date as level 0).
            fold: FoldSplit specification.

        Returns:
            (train_df, val_df, test_df)
        """
        if isinstance(panel.index, pd.MultiIndex):
            dates = panel.index.get_level_values(0)
        else:
            dates = panel.index

        train_mask = (dates >= fold.train_start) & (dates <= fold.train_end)
        val_mask = (dates >= fold.val_start) & (dates <= fold.val_end)
        test_mask = (dates >= fold.test_start) & (dates <= fold.test_end)

        return panel[train_mask], panel[val_mask], panel[test_mask]


# ---------------------------------------------------------------------------
# Evaluation Metrics
# ---------------------------------------------------------------------------

@dataclass
class PredictionMetrics:
    """Prediction quality metrics for a single fold."""
    ic: float = 0.0                     # Spearman rank correlation
    rmse: float = 0.0
    mae: float = 0.0
    hit_ratio: float = 0.0             # directional accuracy
    n_samples: int = 0


@dataclass
class InvestmentMetrics:
    """Investment performance metrics from paper portfolio."""
    annualized_return: float = 0.0
    annualized_volatility: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    calmar_ratio: float = 0.0
    sortino_ratio: float = 0.0
    profit_factor: float = 0.0
    information_ratio: float = 0.0
    turnover: float = 0.0
    daily_returns: Optional[np.ndarray] = None


@dataclass
class FoldResult:
    """Complete results for a single fold."""
    fold_idx: int
    prediction: PredictionMetrics
    investment: InvestmentMetrics
    train_ic: float = 0.0              # for overfit ratio


@dataclass
class ModelEvaluation:
    """Aggregated evaluation across all folds."""
    model_name: str
    horizon: str
    fold_results: List[FoldResult] = field(default_factory=list)

    @property
    def mean_ic(self) -> float:
        return np.mean([f.prediction.ic for f in self.fold_results])

    @property
    def icir(self) -> float:
        ics = [f.prediction.ic for f in self.fold_results]
        std = np.std(ics)
        return np.mean(ics) / std if std > 0 else 0.0

    @property
    def mean_sharpe(self) -> float:
        return np.mean([f.investment.sharpe_ratio for f in self.fold_results])

    @property
    def mean_mdd(self) -> float:
        return np.mean([f.investment.max_drawdown for f in self.fold_results])

    @property
    def mean_calmar(self) -> float:
        return np.mean([f.investment.calmar_ratio for f in self.fold_results])

    @property
    def mean_hit_ratio(self) -> float:
        return np.mean([f.prediction.hit_ratio for f in self.fold_results])

    @property
    def overfit_ratio(self) -> float:
        """Train IC / Test IC. Flag if > 3.0."""
        train_ics = [f.train_ic for f in self.fold_results if f.train_ic > 0]
        test_ics = [f.prediction.ic for f in self.fold_results if f.prediction.ic > 0]
        if not train_ics or not test_ics:
            return float("inf")
        return np.mean(train_ics) / np.mean(test_ics)


def compute_prediction_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> PredictionMetrics:
    """Compute prediction quality metrics."""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_t = y_true[mask]
    y_p = y_pred[mask]
    n = len(y_t)

    if n < 5:
        return PredictionMetrics(n_samples=n)

    if np.std(y_p) < 1e-8 or np.std(y_t) < 1e-8:
        ic = 0.0
    else:
        ic, _ = sp_stats.spearmanr(y_p, y_t)
        # Guard against NaN from constant or near-constant predictions
        if np.isnan(ic):
            ic = 0.0
    rmse = np.sqrt(np.mean((y_t - y_p) ** 2))
    mae = np.mean(np.abs(y_t - y_p))
    hit_ratio = np.mean(np.sign(y_t) == np.sign(y_p))

    return PredictionMetrics(
        ic=float(ic),
        rmse=float(rmse),
        mae=float(mae),
        hit_ratio=float(hit_ratio),
        n_samples=n,
    )


def compute_investment_metrics(
    y_pred: np.ndarray,
    forward_returns: np.ndarray,
    tickers: Optional[np.ndarray] = None,
    n_quantiles: int = 5,
) -> InvestmentMetrics:
    """Compute investment metrics from a long-short paper portfolio.

    Strategy: at each rebalance date, long top quintile, short bottom quintile.

    Args:
        y_pred: Model predictions, shape (n_dates * n_stocks,).
        forward_returns: Realized forward returns, same shape.
        tickers: Ticker labels for cross-sectional ranking.
        n_quantiles: Number of quantile buckets.
    """
    mask = ~(np.isnan(y_pred) | np.isnan(forward_returns))
    if mask.sum() < 20:
        return InvestmentMetrics()

    y_p = y_pred[mask]
    y_r = forward_returns[mask]

    # Simple approach: rank predictions, long top quintile, short bottom
    n = len(y_p)
    ranks = sp_stats.rankdata(y_p)
    quintile_size = n // n_quantiles

    if quintile_size < 1:
        return InvestmentMetrics()

    long_mask = ranks > (n - quintile_size)
    short_mask = ranks <= quintile_size

    # Portfolio return = mean(long returns) - mean(short returns)
    long_ret = np.mean(y_r[long_mask]) if long_mask.sum() > 0 else 0
    short_ret = np.mean(y_r[short_mask]) if short_mask.sum() > 0 else 0
    portfolio_return = long_ret - short_ret

    # For proper Sharpe, we need a time series of returns
    # With panel data, approximate using the single-period return
    # In production, this would be computed per rebalance date
    ann_ret = portfolio_return * (252 / 21)  # scale to annual (assuming ~monthly)
    ann_vol = np.std(y_r) * np.sqrt(252 / 21)  # approximate
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    # Simplified max drawdown (approximate from cross-sectional data)
    mdd = min(portfolio_return, 0)
    calmar = ann_ret / abs(mdd) if mdd < 0 else abs(ann_ret) * 10

    # Sortino
    downside = y_r[y_r < 0]
    downside_std = np.std(downside) * np.sqrt(252 / 21) if len(downside) > 0 else ann_vol
    sortino = ann_ret / downside_std if downside_std > 0 else 0

    # Profit factor
    gains = y_r[y_r > 0].sum() if (y_r > 0).any() else 0
    losses = abs(y_r[y_r < 0].sum()) if (y_r < 0).any() else 1e-8
    profit_factor = gains / losses

    return InvestmentMetrics(
        annualized_return=float(ann_ret),
        annualized_volatility=float(ann_vol),
        sharpe_ratio=float(sharpe),
        max_drawdown=float(mdd),
        calmar_ratio=float(calmar),
        sortino_ratio=float(sortino),
        profit_factor=float(profit_factor),
    )


# ---------------------------------------------------------------------------
# Statistical Significance Tests
# ---------------------------------------------------------------------------

def paired_ttest(eval_a: ModelEvaluation, eval_b: ModelEvaluation,
                 metric: str = "sharpe_ratio") -> Tuple[float, float]:
    """Paired t-test comparing two models across folds.

    Returns:
        (t_statistic, p_value)
    """
    if metric == "sharpe_ratio":
        vals_a = [f.investment.sharpe_ratio for f in eval_a.fold_results]
        vals_b = [f.investment.sharpe_ratio for f in eval_b.fold_results]
    elif metric == "ic":
        vals_a = [f.prediction.ic for f in eval_a.fold_results]
        vals_b = [f.prediction.ic for f in eval_b.fold_results]
    else:
        raise ValueError(f"Unknown metric: {metric}")

    n = min(len(vals_a), len(vals_b))
    if n < 3:
        return 0.0, 1.0

    diffs = np.array(vals_a[:n]) - np.array(vals_b[:n])
    t_stat, p_val = sp_stats.ttest_1samp(diffs, 0)
    return float(t_stat), float(p_val)


def block_bootstrap_sharpe(
    eval_a: ModelEvaluation,
    eval_b: ModelEvaluation,
    n_bootstrap: int = 5000,
    block_size: int = 21,
    seed: int = 42,
) -> Tuple[float, Tuple[float, float]]:
    """Block bootstrap test for Sharpe ratio difference.

    Returns:
        (mean_diff, (ci_lower, ci_upper))
        If CI excludes 0, the difference is significant at 95%.
    """
    rng = np.random.RandomState(seed)

    sharpes_a = np.array([f.investment.sharpe_ratio for f in eval_a.fold_results])
    sharpes_b = np.array([f.investment.sharpe_ratio for f in eval_b.fold_results])
    n = min(len(sharpes_a), len(sharpes_b))

    if n < 3:
        return 0.0, (-1.0, 1.0)

    diffs_bootstrap = []
    for _ in range(n_bootstrap):
        # Block bootstrap: sample blocks of indices
        n_blocks = max(1, n // block_size + 1)
        block_starts = rng.randint(0, n, size=n_blocks)
        indices = []
        for start in block_starts:
            indices.extend(range(start, min(start + block_size, n)))
        indices = np.array(indices[:n])

        diff = np.mean(sharpes_a[indices]) - np.mean(sharpes_b[indices])
        diffs_bootstrap.append(diff)

    diffs_bootstrap = np.array(diffs_bootstrap)
    mean_diff = float(np.mean(diffs_bootstrap))
    ci = (float(np.percentile(diffs_bootstrap, 2.5)),
          float(np.percentile(diffs_bootstrap, 97.5)))

    return mean_diff, ci


# ---------------------------------------------------------------------------
# Model Comparison
# ---------------------------------------------------------------------------

def compare_models(evaluations: List[ModelEvaluation]) -> Dict:
    """Generate model comparison report.

    Args:
        evaluations: List of ModelEvaluation for each model.

    Returns:
        Dict with comparison table and statistical tests.
    """
    # Comparison table
    table = {}
    for ev in evaluations:
        table[ev.model_name] = {
            "horizon": ev.horizon,
            "mean_ic": round(ev.mean_ic, 4),
            "icir": round(ev.icir, 2),
            "mean_hit_ratio": round(ev.mean_hit_ratio, 4),
            "mean_sharpe": round(ev.mean_sharpe, 2),
            "mean_mdd": round(ev.mean_mdd, 4),
            "mean_calmar": round(ev.mean_calmar, 2),
            "overfit_ratio": round(ev.overfit_ratio, 2),
            "n_folds": len(ev.fold_results),
        }

    # Statistical tests (all vs baseline = first model, typically Elastic Net)
    baseline = evaluations[0]
    significance = {}
    for ev in evaluations[1:]:
        t_stat, p_val = paired_ttest(ev, baseline, metric="ic")
        bootstrap_diff, bootstrap_ci = block_bootstrap_sharpe(ev, baseline)
        significance[f"{ev.model_name}_vs_{baseline.model_name}"] = {
            "ttest_p_value": round(p_val, 4),
            "ttest_significant_5pct": p_val < 0.05,
            "bootstrap_sharpe_diff": round(bootstrap_diff, 4),
            "bootstrap_95ci": (round(bootstrap_ci[0], 4), round(bootstrap_ci[1], 4)),
            "bootstrap_significant": bootstrap_ci[0] > 0 or bootstrap_ci[1] < 0,
        }

    # Best model selection
    best_calmar = max(evaluations, key=lambda e: e.mean_calmar)
    best_sharpe = max(evaluations, key=lambda e: e.mean_sharpe)
    most_robust = min(evaluations, key=lambda e: e.overfit_ratio)

    return {
        "comparison_table": table,
        "significance_tests": significance,
        "best_risk_adjusted": best_calmar.model_name,
        "best_sharpe": best_sharpe.model_name,
        "most_robust": most_robust.model_name,
    }


def evaluate_model_on_fold(
    model,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
    rank_target: bool = False,
) -> FoldResult:
    """Run a single walk-forward fold evaluation.

    Args:
        model: Model instance with fit/predict interface.
        train_df: Training data.
        val_df: Validation data.
        test_df: Test data.
        target_col: Name of the target column.
        feature_cols: List of feature column names.
        rank_target: If True, compute cross-sectional ranks separately
            per partition to prevent leakage.

    Returns:
        FoldResult with prediction and investment metrics.
    """
    # If rank_target is True, compute ranks within each partition separately
    if rank_target and target_col.startswith("ranked_target_"):
        # Extract the residual column name from the ranked target name
        # e.g., "ranked_target_21d" -> "residual_return_21d"
        horizon = target_col.replace("ranked_target_", "")
        residual_col = "residual_return_{}".format(horizon)
        if residual_col in train_df.columns:
            from training.feature_engineering import rank_within_partition
            train_df = train_df.copy()
            val_df = val_df.copy()
            test_df = test_df.copy()
            train_df[target_col] = rank_within_partition(train_df, residual_col)
            val_df[target_col] = rank_within_partition(val_df, residual_col)
            test_df[target_col] = rank_within_partition(test_df, residual_col)

    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df[target_col].values.astype(np.float32)
    X_val = val_df[feature_cols].values.astype(np.float32)
    y_val = val_df[target_col].values.astype(np.float32)
    X_test = test_df[feature_cols].values.astype(np.float32)
    y_test = test_df[target_col].values.astype(np.float32)

    # Replace NaN/Inf in features
    for arr in [X_train, X_val, X_test]:
        np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    for arr in [y_train, y_val, y_test]:
        np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    # Train
    model.fit(X_train, y_train, X_val, y_val, feature_names=feature_cols)

    # Predict
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    # Handle NaN predictions (LSTM may produce NaN for initial sequence positions)
    test_valid = ~np.isnan(test_pred)
    if test_valid.sum() == 0:
        return FoldResult(
            fold_idx=0,
            prediction=PredictionMetrics(),
            investment=InvestmentMetrics(),
        )

    # Compute metrics
    train_metrics = compute_prediction_metrics(y_train, train_pred)
    test_metrics = compute_prediction_metrics(
        y_test[test_valid], test_pred[test_valid],
    )
    investment_metrics = compute_investment_metrics(
        test_pred[test_valid], y_test[test_valid],
    )

    return FoldResult(
        fold_idx=0,
        prediction=test_metrics,
        investment=investment_metrics,
        train_ic=train_metrics.ic,
    )
