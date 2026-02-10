"""
Forecast Pipeline
-----------------
Main orchestrator that ties together all components:
  1. Data fetching and feature engineering
  2. Model training and evaluation across horizons
  3. Weight optimization across prediction components
  4. Structured output generation (comparison table, weights, confidence, distribution)

Usage:
    from training.forecast import ForecastPipeline
    pipeline = ForecastPipeline(tickers=["AAPL", "MSFT", "GOOGL"])
    results = pipeline.run()
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import json
import logging
from datetime import datetime

from data.stock_api import get_historical_data, get_stock_info
from training.feature_engineering import (
    build_feature_matrix, build_panel_dataset,
    cross_sectional_normalize, get_feature_groups,
)
from training.models import create_model, BaseModel
from training.model_selection import (
    WalkForwardValidator, WalkForwardConfig, ModelEvaluation,
    evaluate_model_on_fold, compare_models, compute_prediction_metrics,
    compute_investment_metrics,
)
from training.weight_optimizer import (
    DynamicWeightEngine, COMPONENT_NAMES, RegimeClassifier,
    ICBasedModelEnsemble,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Confidence Score Calculator
# ---------------------------------------------------------------------------

def compute_confidence(
    model_predictions: Dict[str, np.ndarray],
    model_evaluations: Dict[str, ModelEvaluation],
    regime_stability: float = 0.7,
    data_quality: float = 0.9,
) -> Dict[str, float]:
    """Compute confidence score for predictions.

    Components:
      - model_agreement: do models agree on direction?
      - historical_accuracy: average hit ratio from evaluation
      - regime_stability: how well-represented is current regime in training
      - data_quality: feature completeness

    Returns:
        Dict with overall confidence and decomposition.
    """
    # Model agreement: check direction consistency
    directions = []
    for name, preds in model_predictions.items():
        if len(preds) > 0:
            mean_pred = np.nanmean(preds)
            directions.append(1 if mean_pred > 0 else -1)

    if len(directions) >= 2:
        agreement = sum(d == directions[0] for d in directions) / len(directions)
    else:
        agreement = 0.5

    # Historical accuracy: average hit ratio
    hit_ratios = []
    for ev in model_evaluations.values():
        if ev.mean_hit_ratio > 0:
            hit_ratios.append(ev.mean_hit_ratio)
    avg_hit = np.mean(hit_ratios) if hit_ratios else 0.5

    # Overall confidence (weighted average)
    confidence = (
        0.30 * agreement +
        0.30 * avg_hit +
        0.20 * regime_stability +
        0.20 * data_quality
    )

    return {
        "confidence": round(float(confidence), 3),
        "model_agreement": round(float(agreement), 3),
        "historical_accuracy": round(float(avg_hit), 3),
        "regime_stability": round(float(regime_stability), 3),
        "data_quality": round(float(data_quality), 3),
    }


# ---------------------------------------------------------------------------
# Forecast Pipeline
# ---------------------------------------------------------------------------

class ForecastPipeline:
    """End-to-end forecasting pipeline.

    Orchestrates data fetching, feature engineering, model training,
    weight optimization, and structured output generation.
    """

    def __init__(
        self,
        tickers: List[str],
        market_ticker: str = "SPY",
        horizons: Optional[Dict[str, int]] = None,
        model_types: Optional[List[str]] = None,
        data_period: str = "10y",
        walk_forward_config: Optional[WalkForwardConfig] = None,
    ):
        """
        Args:
            tickers: List of stock tickers to forecast.
            market_ticker: Market index ticker for macro features / regime.
            horizons: Dict of horizon_name → trading days (e.g., {"1M": 21}).
            model_types: Which models to train. Default: all three.
            data_period: How much historical data to fetch.
            walk_forward_config: Configuration for walk-forward validation.
        """
        self.tickers = [t.upper() for t in tickers]
        self.market_ticker = market_ticker.upper()
        self.horizons = horizons or {"1M": 21, "3M": 63, "6M": 126}
        self.model_types = model_types or ["elastic_net", "lightgbm", "lstm_attention"]
        self.data_period = data_period

        if walk_forward_config is None:
            self.wf_config = WalkForwardConfig(
                train_start="2016-01-01",
                test_end=datetime.now().strftime("%Y-%m-%d"),
                train_min_months=36,
                val_months=6,
                test_months=6,
                step_months=6,
                embargo_days=21,
                expanding=True,
            )
        else:
            self.wf_config = walk_forward_config

        self.panel = None
        self.market_df = None
        self.feature_groups = None
        self.results = {}

    def fetch_data(self) -> None:
        """Fetch historical data for all tickers and market index."""
        logger.info(f"Fetching data for {len(self.tickers)} stocks + {self.market_ticker}...")

        stock_dfs = {}
        stock_infos = {}

        for ticker in self.tickers:
            logger.info(f"  Fetching {ticker}...")
            df = get_historical_data(ticker, period=self.data_period)
            if df.empty:
                logger.warning(f"  No data for {ticker}, skipping.")
                continue
            stock_dfs[ticker] = df
            info = get_stock_info(ticker) or {}
            stock_infos[ticker] = info

        # Market index
        self.market_df = get_historical_data(self.market_ticker, period=self.data_period)
        if self.market_df.empty:
            logger.warning(f"No market data for {self.market_ticker}")
            self.market_df = None

        if not stock_dfs:
            raise ValueError("No stock data could be fetched.")

        # Build panel dataset
        horizon_days = list(self.horizons.values())
        self.panel = build_panel_dataset(
            stock_dfs, stock_infos, self.market_df, horizon_days,
        )
        self.panel = cross_sectional_normalize(self.panel)
        self.feature_groups = get_feature_groups(self.panel)

        logger.info(f"Panel shape: {self.panel.shape}, "
                     f"Features: {len([c for c in self.panel.columns if not c.startswith('fwd_')])},"
                     f" Stocks: {len(stock_dfs)}")

    def _get_feature_cols(self) -> List[str]:
        """Get all feature columns (excluding targets and metadata)."""
        return [c for c in self.panel.columns
                if not c.startswith("fwd_return_")
                and not c.startswith("residual_return_")
                and not c.startswith("ranked_target_")
                and c != "_close"]

    def train_and_evaluate(self) -> Dict:
        """Train all models across all horizons with walk-forward validation.

        Returns:
            Dict of horizon → model_name → ModelEvaluation.
        """
        if self.panel is None:
            raise RuntimeError("Call fetch_data() first.")

        feature_cols = self._get_feature_cols()
        dates = self.panel.index.get_level_values(0).unique().sort_values()
        validator = WalkForwardValidator(self.wf_config)
        folds = validator.generate_folds(dates)

        logger.info(f"Walk-forward: {len(folds)} folds")

        all_evaluations = {}

        for horizon_name, horizon_days in self.horizons.items():
            raw_target_col = "fwd_return_{0}d".format(horizon_days)
            ranked_target_col = "ranked_target_{0}d".format(horizon_days)
            # Use ranked target for training if available, else raw
            train_target = ranked_target_col if ranked_target_col in self.panel.columns else raw_target_col
            if raw_target_col not in self.panel.columns:
                logger.warning("Target {0} not found, skipping horizon {1}".format(raw_target_col, horizon_name))
                continue

            logger.info("\n--- Horizon: {0} ({1}d) ---".format(horizon_name, horizon_days))
            horizon_evals = {}

            for model_type in self.model_types:
                logger.info("  Model: {0}".format(model_type))
                fold_results = []

                for fold in folds:
                    try:
                        train_df, val_df, test_df = validator.split_data(self.panel, fold)

                        if len(train_df) < 50 or len(test_df) < 10:
                            continue

                        model = create_model(model_type)
                        result = evaluate_model_on_fold(
                            model, train_df, val_df, test_df,
                            train_target, feature_cols,
                        )
                        result.fold_idx = fold.fold_idx
                        fold_results.append(result)
                    except Exception as e:
                        logger.warning("    Fold {0} failed: {1}".format(fold.fold_idx, e))
                        continue

                evaluation = ModelEvaluation(
                    model_name=model_type,
                    horizon=horizon_name,
                    fold_results=fold_results,
                )
                horizon_evals[model_type] = evaluation

                if fold_results:
                    logger.info(f"    IC: {evaluation.mean_ic:.4f}, "
                                 f"ICIR: {evaluation.icir:.2f}, "
                                 f"Sharpe: {evaluation.mean_sharpe:.2f}, "
                                 f"Overfit: {evaluation.overfit_ratio:.1f}")

            all_evaluations[horizon_name] = horizon_evals

        self.results["evaluations"] = all_evaluations
        return all_evaluations

    def optimize_weights(self) -> Dict:
        """Optimize component weights for each horizon.

        Uses feature groups as proxy for component predictions.
        """
        if self.panel is None or self.feature_groups is None:
            raise RuntimeError("Call fetch_data() and train_and_evaluate() first.")

        weight_results = {}

        for horizon_name, horizon_days in self.horizons.items():
            target_col = f"fwd_return_{horizon_days}d"
            if target_col not in self.panel.columns:
                continue

            # Compute component predictions as average of group features
            y = self.panel[target_col].values
            component_preds = {}
            for comp_name, cols in self.feature_groups.items():
                if cols:
                    valid_cols = [c for c in cols if c in self.panel.columns]
                    if valid_cols:
                        component_preds[comp_name] = self.panel[valid_cols].mean(axis=1).values

            # Run weight optimizer
            engine = DynamicWeightEngine(
                primary_method="ridge",
                secondary_method="inverse_ic",
                use_regime=(self.market_df is not None),
            )

            weights = engine.optimize(
                component_preds, y, self.market_df,
            )

            weight_results[horizon_name] = {
                "weights": weights,
                "regime": engine.get_current_regime().regime if engine.get_current_regime() else "unknown",
            }

            logger.info(f"  {horizon_name} weights: {weights}")

        self.results["weights"] = weight_results
        return weight_results

    def generate_forecasts(self) -> Dict:
        """Generate final forecasts for each ticker and horizon.

        Trains all model types, uses IC-based ensemble for combining,
        then produces point estimates and distributional forecasts.
        """
        if "evaluations" not in self.results:
            raise RuntimeError("Call train_and_evaluate() first.")

        feature_cols = self._get_feature_cols()
        forecasts = {}

        for horizon_name, horizon_days in self.horizons.items():
            target_col = "ranked_target_{0}d".format(horizon_days)
            raw_target_col = "fwd_return_{0}d".format(horizon_days)
            # Fall back to raw target if ranked not available
            if target_col not in self.panel.columns:
                target_col = raw_target_col
            if target_col not in self.panel.columns:
                continue

            evals = self.results["evaluations"].get(horizon_name, {})
            if not evals:
                continue

            X_all = self.panel[feature_cols].values.astype(np.float32)
            y_all = self.panel[target_col].values.astype(np.float32)
            np.nan_to_num(X_all, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            np.nan_to_num(y_all, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

            split_idx = max(len(X_all) - horizon_days * 2, int(len(X_all) * 0.8))
            X_train = X_all[:split_idx]
            y_train = y_all[:split_idx]
            X_recent = X_all[split_idx:]

            # Train all model types and collect OOS predictions for IC ensemble
            model_predictions = {}
            model_objects = {}
            for model_type in self.model_types:
                try:
                    model = create_model(model_type)
                    val_split = int(len(X_train) * 0.85)
                    model.fit(
                        X_train[:val_split], y_train[:val_split],
                        X_train[val_split:], y_train[val_split:],
                        feature_names=feature_cols,
                    )
                    preds = model.predict(X_recent)
                    model_predictions[model_type] = preds
                    model_objects[model_type] = model
                except Exception as e:
                    logger.warning("  Failed to train {0}: {1}".format(model_type, e))
                    continue

            if not model_predictions:
                continue

            # IC-based ensemble
            y_recent = y_all[split_idx:]
            ensemble = ICBasedModelEnsemble()
            ensemble_weights = ensemble.fit(model_predictions, y_recent)
            combined_preds = ensemble.combine(model_predictions)

            logger.info("  {0} ensemble weights: {1}".format(
                horizon_name,
                {k: round(v, 3) for k, v in ensemble_weights.items()},
            ))

            # Get quantiles from best model
            best_model_name = max(ensemble_weights, key=lambda k: ensemble_weights[k])
            best_model = model_objects[best_model_name]
            quantile_preds = best_model.predict_quantiles(X_recent)

            # Per-ticker forecasts
            recent_panel = self.panel.iloc[split_idx:]
            horizon_forecasts = {}

            for ticker in self.tickers:
                ticker_mask = recent_panel.index.get_level_values("ticker") == ticker
                if not ticker_mask.any():
                    continue

                ticker_idx = np.where(ticker_mask)[0]
                last_idx = ticker_idx[-1] if len(ticker_idx) > 0 else 0

                point = float(combined_preds[last_idx]) if last_idx < len(combined_preds) else 0.0
                distribution = {}
                for q, vals in quantile_preds.items():
                    if last_idx < len(vals):
                        distribution["p{0}".format(int(q * 100))] = round(float(vals[last_idx]), 5)

                direction_prob = 0.5
                if point > 0:
                    p10 = distribution.get("p10", -0.05)
                    p50 = distribution.get("p50", point)
                    if p10 < 0 < p50:
                        direction_prob = 0.5 + 0.4 * (p50 / (p50 - p10))
                    else:
                        direction_prob = 0.7 if point > 0 else 0.3
                else:
                    direction_prob = 0.3

                horizon_forecasts[ticker] = {
                    "point_estimate": round(point, 5),
                    "distribution": distribution,
                    "direction_probability": round(min(max(direction_prob, 0.05), 0.95), 3),
                    "model_weights": {k: round(v, 4) for k, v in ensemble_weights.items()},
                }

            forecasts[horizon_name] = {
                "best_model": best_model_name,
                "ensemble_weights": ensemble_weights,
                "ticker_forecasts": horizon_forecasts,
            }

        self.results["forecasts"] = forecasts
        return forecasts

    def run(self) -> Dict:
        """Execute the full pipeline.

        Returns:
            Complete results dict with:
              - comparison_table: model comparison across horizons
              - best_model_per_horizon: best model for each horizon
              - optimal_weights: component weights per horizon
              - forecasts: per-ticker distributional forecasts
              - confidence: per-ticker confidence scores
        """
        logger.info("=" * 60)
        logger.info("STOCK FORECASTING PIPELINE")
        logger.info(f"Tickers: {self.tickers}")
        logger.info(f"Horizons: {self.horizons}")
        logger.info(f"Models: {self.model_types}")
        logger.info("=" * 60)

        # Step 1: Fetch data and build features
        logger.info("\n[1/5] Fetching data and building features...")
        self.fetch_data()

        # Step 2: Train and evaluate models
        logger.info("\n[2/5] Training and evaluating models...")
        evaluations = self.train_and_evaluate()

        # Step 3: Compare models
        logger.info("\n[3/5] Comparing models...")
        comparison = {}
        best_per_horizon = {}
        for horizon_name, evals in evaluations.items():
            eval_list = list(evals.values())
            if eval_list:
                comparison[horizon_name] = compare_models(eval_list)
                best_per_horizon[horizon_name] = comparison[horizon_name]["best_risk_adjusted"]

        # Step 4: Optimize weights
        logger.info("\n[4/5] Optimizing component weights...")
        weights = self.optimize_weights()

        # Step 5: Generate forecasts
        logger.info("\n[5/5] Generating forecasts...")
        forecasts = self.generate_forecasts()

        # Compute confidence scores
        confidence_scores = {}
        for horizon_name, horizon_data in forecasts.items():
            evals = evaluations.get(horizon_name, {})
            model_preds = {}
            for model_name, ev in evals.items():
                # Use mean IC as proxy for prediction quality
                model_preds[model_name] = np.array([ev.mean_ic])

            conf = compute_confidence(model_preds, evals)
            confidence_scores[horizon_name] = conf

        # Compile final output
        output = {
            "evaluation_date": datetime.now().strftime("%Y-%m-%d"),
            "tickers": self.tickers,
            "comparison_table": comparison,
            "best_model_per_horizon": best_per_horizon,
            "optimal_weights": weights,
            "forecasts": forecasts,
            "confidence": confidence_scores,
        }

        self.results = output
        return output

    def print_summary(self) -> None:
        """Print a formatted summary of results."""
        if not self.results:
            print("No results. Run pipeline first.")
            return

        print("\n" + "=" * 70)
        print("  STOCK FORECASTING PIPELINE - RESULTS")
        print("=" * 70)

        print(f"\nEvaluation Date: {self.results.get('evaluation_date', 'N/A')}")
        print(f"Tickers: {', '.join(self.results.get('tickers', []))}")

        # Model comparison
        print("\n--- MODEL COMPARISON ---")
        for horizon, comp in self.results.get("comparison_table", {}).items():
            print(f"\n  Horizon: {horizon}")
            table = comp.get("comparison_table", {})
            print(f"  {'Model':<20} {'IC':>8} {'ICIR':>8} {'Sharpe':>8} "
                  f"{'Calmar':>8} {'Overfit':>8}")
            print(f"  {'-'*68}")
            for model, metrics in table.items():
                print(f"  {model:<20} {metrics['mean_ic']:>8.4f} {metrics['icir']:>8.2f} "
                      f"{metrics['mean_sharpe']:>8.2f} {metrics['mean_calmar']:>8.2f} "
                      f"{metrics['overfit_ratio']:>8.1f}")
            print(f"  Best (risk-adjusted): {comp.get('best_risk_adjusted', 'N/A')}")

        # Optimal weights
        print("\n--- OPTIMAL WEIGHTS ---")
        for horizon, w_info in self.results.get("optimal_weights", {}).items():
            weights = w_info.get("weights", {})
            regime = w_info.get("regime", "unknown")
            print(f"\n  {horizon} (regime: {regime}):")
            for comp, val in weights.items():
                bar = "#" * int(val * 40)
                print(f"    {comp:<15} {val:.3f}  {bar}")

        # Forecasts
        print("\n--- FORECASTS ---")
        for horizon, f_data in self.results.get("forecasts", {}).items():
            print(f"\n  {horizon} (model: {f_data.get('best_model', 'N/A')}):")
            for ticker, forecast in f_data.get("ticker_forecasts", {}).items():
                pt = forecast["point_estimate"]
                dp = forecast["direction_probability"]
                dist = forecast.get("distribution", {})
                print(f"    {ticker}: {pt:+.4f} (P(up)={dp:.1%})"
                      f"  [{dist.get('p10', '?')}, {dist.get('p50', '?')}, {dist.get('p90', '?')}]")

        # Confidence
        print("\n--- CONFIDENCE ---")
        for horizon, conf in self.results.get("confidence", {}).items():
            c = conf["confidence"]
            print(f"  {horizon}: {c:.1%} "
                  f"(agreement={conf['model_agreement']:.1%}, "
                  f"accuracy={conf['historical_accuracy']:.1%})")

        print("\n" + "=" * 70)
