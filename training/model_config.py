"""
Model Configuration System
---------------------------
Config-driven training that supports running multiple model configurations.

Components:
  - ModelConfig: dataclass for model hyperparameters
  - ConfigGrid: generates configs from parameter grids
  - MultiConfigRunner: trains + evaluates all configs via walk-forward
  - ConfigResult: stores training results per config
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import itertools
import math
import time
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for a single model training run."""
    model_type: str = "lightgbm"
    sequence_length: int = 60
    dropout: float = 0.2
    learning_rate: float = 0.05
    batch_size: int = 64
    epochs: int = 100
    extra_params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to a flat dict for model creation."""
        d = {
            "sequence_length": self.sequence_length,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
        }
        d.update(self.extra_params)
        return d

    def to_json(self) -> Dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        return {
            "model_type": self.model_type,
            "sequence_length": self.sequence_length,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "extra_params": self.extra_params,
        }

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "ModelConfig":
        """Deserialize from JSON dict."""
        return cls(
            model_type=data.get("model_type", "lightgbm"),
            sequence_length=data.get("sequence_length", 60),
            dropout=data.get("dropout", 0.2),
            learning_rate=data.get("learning_rate", 0.05),
            batch_size=data.get("batch_size", 64),
            epochs=data.get("epochs", 100),
            extra_params=data.get("extra_params", {}),
        )


@dataclass
class ConfigResult:
    """Result from training a single model configuration."""
    config: ModelConfig
    evaluation: Any  # ModelEvaluation from model_selection
    training_time: float  # seconds
    best_params: Dict[str, Any] = field(default_factory=dict)


class ConfigGrid:
    """Generates model configs from parameter grids.

    Usage:
        grid = ConfigGrid.from_grid({
            "model_type": ["lightgbm", "elastic_net"],
            "learning_rate": [0.01, 0.05, 0.1],
        })  # -> 6 configs

        random_configs = ConfigGrid.from_random({
            "learning_rate": [0.001, 0.01, 0.05, 0.1],
            "dropout": [0.1, 0.2, 0.3],
        }, n_samples=5)  # -> 5 random configs
    """

    @staticmethod
    def from_grid(param_dict: Dict[str, List[Any]]) -> List[ModelConfig]:
        """Generate all combinations from parameter grid.

        Args:
            param_dict: mapping of parameter name -> list of values.
                        Recognized keys: model_type, sequence_length, dropout,
                        learning_rate, batch_size, epochs.
                        Unrecognized keys go into extra_params.

        Returns:
            List of ModelConfig covering all combinations.
        """
        direct_keys = {
            "model_type", "sequence_length", "dropout",
            "learning_rate", "batch_size", "epochs",
        }

        keys = list(param_dict.keys())
        values = list(param_dict.values())

        configs = []
        for combo in itertools.product(*values):
            kwargs = {}  # type: Dict[str, Any]
            extra = {}  # type: Dict[str, Any]
            for k, v in zip(keys, combo):
                if k in direct_keys:
                    kwargs[k] = v
                else:
                    extra[k] = v
            kwargs["extra_params"] = extra
            configs.append(ModelConfig(**kwargs))

        return configs

    @staticmethod
    def from_random(
        param_dict: Dict[str, List[Any]],
        n_samples: int = 10,
        seed: int = 42,
    ) -> List[ModelConfig]:
        """Random sampling from parameter grid.

        Args:
            param_dict: mapping of parameter name -> list of candidate values.
            n_samples: how many random configs to generate.
            seed: random seed for reproducibility.

        Returns:
            List of n_samples randomly sampled ModelConfig instances.
        """
        direct_keys = {
            "model_type", "sequence_length", "dropout",
            "learning_rate", "batch_size", "epochs",
        }

        rng = np.random.RandomState(seed)
        keys = list(param_dict.keys())
        configs = []

        for _ in range(n_samples):
            kwargs = {}  # type: Dict[str, Any]
            extra = {}  # type: Dict[str, Any]
            for k in keys:
                val = param_dict[k][rng.randint(len(param_dict[k]))]
                if k in direct_keys:
                    kwargs[k] = val
                else:
                    extra[k] = val
            kwargs["extra_params"] = extra
            configs.append(ModelConfig(**kwargs))

        return configs


class MultiConfigRunner:
    """Trains and evaluates multiple model configurations via walk-forward.

    Uses the existing walk-forward validation infrastructure from
    training.model_selection and stores results in the model registry.

    Usage:
        runner = MultiConfigRunner()
        results = runner.run(
            configs=configs,
            panel=panel_df,
            target_col="fwd_return_21d",
            feature_cols=feature_cols,
            wf_config=wf_config,
        )
    """

    def __init__(self, save_to_registry: bool = True):
        self.save_to_registry = save_to_registry

    def run(
        self,
        configs: List[ModelConfig],
        panel: pd.DataFrame,
        target_col: str,
        feature_cols: List[str],
        wf_config: Optional[Any] = None,
        horizon: str = "1M",
    ) -> List[ConfigResult]:
        """Train and evaluate all configs.

        Args:
            configs: list of ModelConfig to evaluate.
            panel: DataFrame with features and targets.
            target_col: name of the target column.
            feature_cols: list of feature column names.
            wf_config: WalkForwardConfig (if None, uses defaults).
            horizon: horizon label for registry.

        Returns:
            List of ConfigResult sorted by mean IC descending.
        """
        from training.model_selection import (
            WalkForwardValidator, WalkForwardConfig,
            evaluate_model_on_fold, ModelEvaluation, FoldResult,
        )
        from training.models import create_model

        if wf_config is None:
            wf_config = WalkForwardConfig(
                train_start="2015-01-01",
                test_end="2025-01-01",
            )

        # Get dates for fold generation
        if isinstance(panel.index, pd.MultiIndex):
            dates = panel.index.get_level_values(0).unique().sort_values()
        else:
            dates = panel.index.unique().sort_values()

        validator = WalkForwardValidator(wf_config)
        folds = validator.generate_folds(pd.DatetimeIndex(dates))

        results = []

        for i, config in enumerate(configs):
            logger.info(
                f"Config {i + 1}/{len(configs)}: {config.model_type} "
                f"lr={config.learning_rate} dropout={config.dropout}"
            )
            t0 = time.time()

            model = create_model(config.model_type, config.to_dict())
            fold_results = []  # type: List[FoldResult]

            for fold in folds:
                try:
                    train_df, val_df, test_df = validator.split_data(panel, fold)
                    fold_result = evaluate_model_on_fold(
                        model=model,
                        train_df=train_df,
                        val_df=val_df,
                        test_df=test_df,
                        target_col=target_col,
                        feature_cols=feature_cols,
                    )
                    fold_result.fold_idx = fold.fold_idx
                    fold_results.append(fold_result)
                except Exception as e:
                    logger.warning(
                        f"Fold {fold.fold_idx} failed for config {i}: {e}"
                    )
                    continue

            evaluation = ModelEvaluation(
                model_name=f"{config.model_type}_cfg{i}",
                horizon=horizon,
                fold_results=fold_results,
            )

            training_time = time.time() - t0

            # Save to registry
            if self.save_to_registry and fold_results:
                try:
                    from training.model_versioning import ModelRegistry
                    registry = ModelRegistry()
                    metrics = {
                        "mean_ic": evaluation.mean_ic,
                        "icir": evaluation.icir,
                        "mean_sharpe": evaluation.mean_sharpe,
                        "mean_mdd": evaluation.mean_mdd,
                        "n_folds": len(fold_results),
                    }
                    registry.save_model(
                        model=model,
                        model_type=config.model_type,
                        horizon=horizon,
                        params=config.to_dict(),
                        metrics=metrics,
                        config=config,
                    )
                except Exception as e:
                    logger.warning(f"Failed to save to registry: {e}")

            results.append(ConfigResult(
                config=config,
                evaluation=evaluation,
                training_time=training_time,
                best_params=config.to_dict(),
            ))

        # Filter out results with NaN IC, then sort descending
        valid_results = [
            r for r in results
            if not math.isnan(r.evaluation.mean_ic)
        ]
        nan_results = [
            r for r in results
            if math.isnan(r.evaluation.mean_ic)
        ]
        valid_results.sort(
            key=lambda r: r.evaluation.mean_ic, reverse=True,
        )
        # Append NaN results at the end so they never get selected as best
        return valid_results + nan_results


def run_light_comparison(
    panel: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
    models: Optional[List[str]] = None,
    horizons: Optional[Dict[str, int]] = None,
    n_splits: int = 2,
) -> List[Dict]:
    """Run a lightweight model comparison with minimal walk-forward splits.

    No HP search -- uses default parameters for speed.

    Args:
        panel: Panel DataFrame with MultiIndex (date, ticker).
        target_col: Name of the target column.
        feature_cols: List of feature column names.
        models: Model types to compare. Default: gated_hybrid, lstm_attention, lightgbm.
        horizons: Dict of horizon_name -> trading_days. Default: 5D + 1M.
        n_splits: Number of walk-forward splits (default 2).

    Returns:
        List of dicts with model comparison results, sorted by composite score.
    """
    from training.model_selection import (
        WalkForwardValidator, WalkForwardConfig,
        evaluate_model_on_fold, ModelEvaluation, FoldResult,
    )
    from training.models import create_model

    if models is None:
        models = ["gated_hybrid", "lstm_attention", "lightgbm"]
    if horizons is None:
        horizons = {"5D": 5, "1M": 21}

    if isinstance(panel.index, pd.MultiIndex):
        dates = panel.index.get_level_values(0).unique().sort_values()
    else:
        dates = panel.index.unique().sort_values()

    wf_config = WalkForwardConfig(
        train_start=str(dates[0].date()),
        test_end=str(dates[-1].date()),
        train_min_months=24,
        val_months=6,
        test_months=6,
        step_months=12,
        embargo_days=26,
        expanding=True,
    )
    validator = WalkForwardValidator(wf_config)
    folds = validator.generate_folds(pd.DatetimeIndex(dates))[:n_splits]

    comparison = []

    for model_type in models:
        for horizon_name, horizon_days in horizons.items():
            h_target = "fwd_return_{}d".format(horizon_days)
            if h_target not in panel.columns:
                h_target = target_col

            fold_results = []
            for fold in folds:
                try:
                    train_df, val_df, test_df = validator.split_data(panel, fold)
                    model = create_model(model_type)
                    fr = evaluate_model_on_fold(
                        model=model,
                        train_df=train_df,
                        val_df=val_df,
                        test_df=test_df,
                        target_col=h_target,
                        feature_cols=feature_cols,
                    )
                    fr.fold_idx = fold.fold_idx
                    fold_results.append(fr)
                except Exception as e:
                    logger.warning("Light comparison fold failed: %s %s: %s",
                                   model_type, horizon_name, e)
                    continue

            if not fold_results:
                continue

            evaluation = ModelEvaluation(
                model_name=model_type,
                horizon=horizon_name,
                fold_results=fold_results,
            )

            # Composite score
            ic = evaluation.mean_ic
            icir = evaluation.icir
            sharpe = evaluation.mean_sharpe
            ic_std = float(np.std([f.prediction.ic for f in fold_results]))
            overfit = evaluation.overfit_ratio if evaluation.overfit_ratio != float("inf") else 5.0

            if math.isnan(ic):
                ic = 0.0
            if math.isnan(icir):
                icir = 0.0

            composite = 1.0 * ic + 0.5 * icir + 0.3 * sharpe - 0.3 * ic_std
            if overfit > 3.0:
                composite -= 0.5

            comparison.append({
                "model_type": model_type,
                "horizon": horizon_name,
                "ic": ic,
                "icir": icir,
                "sharpe": sharpe,
                "ic_std": ic_std,
                "overfit_ratio": overfit,
                "composite": composite,
                "n_folds": len(fold_results),
            })
            logger.info("Light comparison %s/%s: IC=%.4f, composite=%.4f",
                         model_type, horizon_name, ic, composite)

    comparison.sort(key=lambda x: x["composite"], reverse=True)
    return comparison
