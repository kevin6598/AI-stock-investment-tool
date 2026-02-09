"""
Model Comparison Framework
---------------------------
Comprehensive model comparison with multi-horizon metrics, ranking,
and stability scoring.

Components:
  - ComparisonMetrics: per-model metric summary
  - ComparisonReport: full comparison output
  - ModelComparisonEngine: compare, rank, score models
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class ComparisonMetrics:
    """Comprehensive metrics for a single model."""
    sharpe: float = 0.0
    sortino: float = 0.0
    max_drawdown: float = 0.0
    calmar: float = 0.0
    cvar_95: float = 0.0
    hit_ratio: float = 0.0
    rmse: float = 0.0
    mae: float = 0.0
    turnover: float = 0.0
    ic: float = 0.0
    icir: float = 0.0


@dataclass
class ComparisonReport:
    """Full model comparison output."""
    comparison_table: Dict[str, Dict[str, float]]
    rankings: Dict[str, List[Tuple[str, float]]]
    stability_scores: Dict[str, float]
    significance_tests: Dict[str, Dict[str, Any]]
    best_per_horizon: Dict[str, str]


class ModelComparisonEngine:
    """Comprehensive model comparison with ranking and stability scoring.

    Reuses statistical tests from training.model_selection:
      - compare_models()
      - paired_ttest()
      - block_bootstrap_sharpe()

    Usage:
        engine = ModelComparisonEngine()
        report = engine.compare(evaluations)
        rankings = engine.rank_models("mean_ic")
    """

    def __init__(self):
        self._evaluations = []  # type: List[Any]
        self._metrics_cache = {}  # type: Dict[str, ComparisonMetrics]

    def compare(self, evaluations: List[Any]) -> ComparisonReport:
        """Run full model comparison.

        Args:
            evaluations: list of ModelEvaluation objects.

        Returns:
            ComparisonReport with tables, rankings, and tests.
        """
        from training.model_selection import (
            compare_models, paired_ttest, block_bootstrap_sharpe,
        )

        self._evaluations = evaluations
        self._metrics_cache = {}

        # Build comparison table
        comparison_table = {}
        for ev in evaluations:
            metrics = self._extract_metrics(ev)
            self._metrics_cache[ev.model_name] = metrics
            comparison_table[ev.model_name] = {
                "horizon": ev.horizon,
                "ic": round(metrics.ic, 4),
                "icir": round(metrics.icir, 2),
                "sharpe": round(metrics.sharpe, 2),
                "sortino": round(metrics.sortino, 2),
                "max_drawdown": round(metrics.max_drawdown, 4),
                "calmar": round(metrics.calmar, 2),
                "hit_ratio": round(metrics.hit_ratio, 4),
                "rmse": round(metrics.rmse, 4),
                "mae": round(metrics.mae, 4),
                "turnover": round(metrics.turnover, 4),
                "n_folds": len(ev.fold_results),
            }

        # Rankings by multiple metrics
        rankings = {}
        for metric_name in ["ic", "sharpe", "calmar", "hit_ratio", "sortino"]:
            rankings[metric_name] = self.rank_models(metric_name)

        # Stability scores
        stability_scores = {}
        for ev in evaluations:
            stability_scores[ev.model_name] = self.stability_score(ev)

        # Significance tests (from compare_models)
        significance_tests = {}
        if len(evaluations) >= 2:
            baseline = evaluations[0]
            for ev in evaluations[1:]:
                key = f"{ev.model_name}_vs_{baseline.model_name}"
                t_stat, p_val = paired_ttest(ev, baseline, metric="ic")
                bs_diff, bs_ci = block_bootstrap_sharpe(ev, baseline)
                significance_tests[key] = {
                    "ttest_t_statistic": round(t_stat, 4),
                    "ttest_p_value": round(p_val, 4),
                    "ttest_significant_5pct": p_val < 0.05,
                    "bootstrap_sharpe_diff": round(bs_diff, 4),
                    "bootstrap_95ci": (round(bs_ci[0], 4), round(bs_ci[1], 4)),
                    "bootstrap_significant": bs_ci[0] > 0 or bs_ci[1] < 0,
                }

        # Best per horizon
        best_per_horizon = {}
        horizons_seen = set()
        for ev in evaluations:
            horizons_seen.add(ev.horizon)
        for h in horizons_seen:
            h_evals = [e for e in evaluations if e.horizon == h]
            if h_evals:
                best = max(h_evals, key=lambda e: e.mean_ic)
                best_per_horizon[h] = best.model_name

        return ComparisonReport(
            comparison_table=comparison_table,
            rankings=rankings,
            stability_scores=stability_scores,
            significance_tests=significance_tests,
            best_per_horizon=best_per_horizon,
        )

    def rank_models(self, metric: str) -> List[Tuple[str, float]]:
        """Rank models by a specific metric (descending).

        Args:
            metric: one of ic, sharpe, calmar, hit_ratio, sortino, etc.

        Returns:
            Sorted list of (model_name, metric_value) tuples.
        """
        if not self._metrics_cache:
            return []

        items = []
        for name, m in self._metrics_cache.items():
            val = getattr(m, metric, 0.0)
            items.append((name, val))

        # For max_drawdown, lower (closer to 0) is better
        reverse = metric != "max_drawdown"
        items.sort(key=lambda x: x[1], reverse=reverse)
        return items

    def weighted_score(
        self,
        weights_dict: Dict[str, float],
    ) -> Dict[str, float]:
        """Compute composite weighted score for each model.

        Args:
            weights_dict: metric_name -> weight. e.g. {"ic": 0.3, "sharpe": 0.4, "calmar": 0.3}

        Returns:
            Dict of model_name -> composite score.
        """
        scores = {}
        for name, m in self._metrics_cache.items():
            score = 0.0
            for metric, weight in weights_dict.items():
                val = getattr(m, metric, 0.0)
                score += weight * val
            scores[name] = round(score, 4)
        return scores

    def stability_score(self, evaluation: Any) -> float:
        """Compute stability score: lower std of metrics across folds is better.

        Returns:
            Score in [0, 1] where 1 = perfectly stable, 0 = highly unstable.
        """
        if not evaluation.fold_results:
            return 0.0

        ics = [f.prediction.ic for f in evaluation.fold_results]
        sharpes = [f.investment.sharpe_ratio for f in evaluation.fold_results]

        if len(ics) < 2:
            return 0.5

        ic_std = np.std(ics)
        sharpe_std = np.std(sharpes)

        # Combine: lower std -> higher stability
        # Use sigmoid-like transform: stability = 1 / (1 + combined_std)
        combined_std = (ic_std + sharpe_std * 0.1) / 2.0
        stability = 1.0 / (1.0 + combined_std * 10.0)

        return round(float(stability), 4)

    @staticmethod
    def _extract_metrics(evaluation: Any) -> ComparisonMetrics:
        """Extract ComparisonMetrics from a ModelEvaluation."""
        folds = evaluation.fold_results
        if not folds:
            return ComparisonMetrics()

        ics = [f.prediction.ic for f in folds]
        sharpes = [f.investment.sharpe_ratio for f in folds]
        sortinos = [f.investment.sortino_ratio for f in folds]
        mdds = [f.investment.max_drawdown for f in folds]
        calmars = [f.investment.calmar_ratio for f in folds]
        hit_ratios = [f.prediction.hit_ratio for f in folds]
        rmses = [f.prediction.rmse for f in folds]
        maes = [f.prediction.mae for f in folds]
        turnovers = [f.investment.turnover for f in folds]

        ic_std = np.std(ics) if len(ics) > 1 else 1e-8

        return ComparisonMetrics(
            sharpe=float(np.mean(sharpes)),
            sortino=float(np.mean(sortinos)),
            max_drawdown=float(np.mean(mdds)),
            calmar=float(np.mean(calmars)),
            cvar_95=0.0,  # not available from fold results
            hit_ratio=float(np.mean(hit_ratios)),
            rmse=float(np.mean(rmses)),
            mae=float(np.mean(maes)),
            turnover=float(np.mean(turnovers)),
            ic=float(np.mean(ics)),
            icir=float(np.mean(ics) / ic_std) if ic_std > 0 else 0.0,
        )
