"""
Hyperparameter Search
---------------------
Bayesian optimization via Optuna with nested walk-forward cross-validation.
Prevents data leakage via temporal inner/outer fold separation.

Structure:
  - Outer loop: walk-forward folds for final performance estimation
  - Inner loop: walk-forward folds for hyperparameter selection
  - Optuna study: Bayesian optimization within each outer fold
"""

from typing import Dict, List, Optional, Callable
import numpy as np
import pandas as pd
import optuna
import logging

from training.models import create_model
from training.model_selection import (
    WalkForwardValidator, WalkForwardConfig, FoldSplit,
    evaluate_model_on_fold, ModelEvaluation, FoldResult,
    compute_prediction_metrics,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Search Spaces
# ---------------------------------------------------------------------------

def elastic_net_search_space(trial: optuna.Trial) -> Dict:
    return {
        "alpha": trial.suggest_float("alpha", 1e-4, 10.0, log=True),
        "l1_ratio": trial.suggest_float("l1_ratio", 0.01, 0.99),
    }


def lightgbm_search_space(trial: optuna.Trial) -> Dict:
    return {
        "num_leaves": trial.suggest_int("num_leaves", 15, 63),
        "max_depth": trial.suggest_int("max_depth", 4, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
        "min_child_samples": trial.suggest_int("min_child_samples", 50, 300, step=50),
        "subsample": trial.suggest_float("subsample", 0.6, 0.9),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.9),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 10.0, log=True),
    }


def lstm_attention_search_space(trial: optuna.Trial) -> Dict:
    return {
        "hidden_size": trial.suggest_categorical("hidden_size", [32, 64, 128]),
        "num_layers": trial.suggest_int("num_layers", 1, 3),
        "attention_heads": trial.suggest_categorical("attention_heads", [1, 2, 4]),
        "dropout": trial.suggest_float("dropout", 0.05, 0.4),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        "sequence_length": trial.suggest_categorical("sequence_length", [30, 60, 120]),
        "epochs": 100,
        "patience": 15,
    }


def transformer_search_space(trial: optuna.Trial) -> Dict:
    return {
        "d_model": trial.suggest_categorical("d_model", [64, 128, 256]),
        "n_heads": trial.suggest_categorical("n_heads", [2, 4, 8]),
        "n_layers": trial.suggest_int("n_layers", 2, 6),
        "d_ff": trial.suggest_categorical("d_ff", [128, 256, 512]),
        "dropout": trial.suggest_float("dropout", 0.05, 0.4),
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        "sequence_length": trial.suggest_categorical("sequence_length", [30, 60, 120]),
        "epochs": 100,
        "patience": 15,
    }


def hybrid_multimodal_search_space(trial: optuna.Trial) -> Dict:
    return {
        "hidden_dim": trial.suggest_categorical("hidden_dim", [64, 128, 256]),
        "fusion_dim": trial.suggest_categorical("fusion_dim", [64, 128, 256]),
        "vae_latent_dim": trial.suggest_categorical("vae_latent_dim", [8, 16, 32]),
        "dropout": trial.suggest_float("dropout", 0.05, 0.4),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        "sequence_length": trial.suggest_categorical("sequence_length", [30, 60]),
        "epochs": 50,
        "patience": 10,
    }


SEARCH_SPACES = {
    "elastic_net": elastic_net_search_space,
    "lightgbm": lightgbm_search_space,
    "lstm_attention": lstm_attention_search_space,
    "transformer": transformer_search_space,
    "hybrid_multimodal": hybrid_multimodal_search_space,
}


# ---------------------------------------------------------------------------
# IC Composite Score
# ---------------------------------------------------------------------------

def _compute_ic_composite(ics: List[float]) -> float:
    """Compute composite IC score: ic_mean + 0.5 * ICIR - 0.3 * ic_std.

    This rewards both high mean IC and stability (ICIR), while penalizing
    high variance in IC across folds.

    Args:
        ics: List of per-fold IC values.

    Returns:
        Composite IC score.
    """
    if not ics:
        return -1.0
    ic_mean = float(np.mean(ics))
    ic_std = float(np.std(ics))
    if ic_std < 1e-8:
        ic_ir = 0.0
    else:
        ic_ir = ic_mean / ic_std
    return ic_mean + 0.5 * ic_ir - 0.3 * ic_std


# ---------------------------------------------------------------------------
# Inner Walk-Forward Objective (for hyperparameter tuning)
# ---------------------------------------------------------------------------

def _inner_objective(
    trial: optuna.Trial,
    model_type: str,
    panel: pd.DataFrame,
    inner_folds: List[FoldSplit],
    target_col: str,
    feature_cols: List[str],
    validator: WalkForwardValidator,
) -> float:
    """Optuna objective: mean IC across inner walk-forward folds."""
    search_fn = SEARCH_SPACES[model_type]
    params = search_fn(trial)

    ics = []
    for fold in inner_folds:
        try:
            train_df, val_df, test_df = validator.split_data(panel, fold)

            if len(train_df) < 50 or len(val_df) < 10:
                continue

            model = create_model(model_type, params)

            X_train = train_df[feature_cols].values.astype(np.float32)
            y_train = train_df[target_col].values.astype(np.float32)
            X_val = val_df[feature_cols].values.astype(np.float32)
            y_val = val_df[target_col].values.astype(np.float32)

            np.nan_to_num(X_train, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            np.nan_to_num(X_val, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            np.nan_to_num(y_train, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            np.nan_to_num(y_val, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

            model.fit(X_train, y_train, X_val, y_val, feature_names=feature_cols)
            val_pred = model.predict(X_val)

            valid = ~np.isnan(val_pred)
            if valid.sum() > 5:
                metrics = compute_prediction_metrics(y_val[valid], val_pred[valid])
                ics.append(metrics.ic)
        except Exception as e:
            logger.warning(f"Trial {trial.number} fold {fold.fold_idx} failed: {e}")
            continue

    if not ics:
        return -1.0  # worst possible

    return _compute_ic_composite(ics)


# ---------------------------------------------------------------------------
# Nested Walk-Forward Hyperparameter Search
# ---------------------------------------------------------------------------

class HyperparameterSearcher:
    """Nested walk-forward hyperparameter search using Optuna."""

    def __init__(
        self,
        model_type: str,
        panel: pd.DataFrame,
        target_col: str,
        feature_cols: List[str],
        outer_config: WalkForwardConfig,
        n_trials: int = 50,
        inner_folds_count: int = 3,
    ):
        """
        Args:
            model_type: "elastic_net", "lightgbm", or "lstm_attention".
            panel: Panel DataFrame with MultiIndex (date, ticker).
            target_col: Target column name.
            feature_cols: List of feature column names.
            outer_config: Walk-forward config for outer evaluation folds.
            n_trials: Number of Optuna trials per outer fold.
            inner_folds_count: Number of inner walk-forward folds for HP tuning.
        """
        self.model_type = model_type
        self.panel = panel
        self.target_col = target_col
        self.feature_cols = feature_cols
        self.outer_config = outer_config
        self.n_trials = n_trials
        self.inner_folds_count = inner_folds_count

        self.outer_validator = WalkForwardValidator(outer_config)

    def _create_inner_folds(self, outer_fold: FoldSplit) -> List[FoldSplit]:
        """Create inner walk-forward folds within an outer fold's training data."""
        inner_dates = self.panel.index.get_level_values(0)
        inner_dates = inner_dates[
            (inner_dates >= outer_fold.train_start) &
            (inner_dates <= outer_fold.val_end)
        ].unique().sort_values()

        if len(inner_dates) < 100:
            return []

        # Divide available time into inner folds
        total_months = (outer_fold.val_end - outer_fold.train_start).days // 30
        inner_test_months = max(3, total_months // (self.inner_folds_count + 2))
        inner_val_months = inner_test_months

        inner_folds = []
        for i in range(self.inner_folds_count):
            offset = i * inner_test_months
            test_start = outer_fold.val_end - pd.DateOffset(
                months=(self.inner_folds_count - i) * inner_test_months
            )
            test_end = test_start + pd.DateOffset(months=inner_test_months)
            val_end = test_start - pd.DateOffset(days=self.outer_config.embargo_days)
            val_start = val_end - pd.DateOffset(months=inner_val_months)
            train_end = val_start - pd.DateOffset(days=self.outer_config.embargo_days)
            train_start = outer_fold.train_start

            if train_end > train_start:
                inner_folds.append(FoldSplit(
                    fold_idx=i,
                    train_start=train_start,
                    train_end=train_end,
                    val_start=val_start,
                    val_end=val_end,
                    test_start=test_start,
                    test_end=test_end,
                ))

        return inner_folds

    def search(self) -> Dict:
        """Run full nested walk-forward hyperparameter search.

        Returns:
            Dict with:
              - best_params_per_fold: best hyperparameters for each outer fold
              - evaluation: ModelEvaluation with test-set results
              - all_fold_results: detailed per-fold results
        """
        dates = self.panel.index.get_level_values(0).unique().sort_values()
        outer_folds = self.outer_validator.generate_folds(dates)

        logger.info(f"Running {self.model_type} hyperparameter search: "
                     f"{len(outer_folds)} outer folds, {self.n_trials} trials each")

        fold_results = []
        best_params_per_fold = []

        for outer_fold in outer_folds:
            logger.info(f"Outer fold {outer_fold.fold_idx}: "
                         f"test [{outer_fold.test_start.date()} → {outer_fold.test_end.date()}]")

            # Create inner folds for HP tuning
            inner_folds = self._create_inner_folds(outer_fold)

            if not inner_folds:
                logger.warning(f"  Skipping fold {outer_fold.fold_idx}: insufficient inner data")
                continue

            # Optuna study for this outer fold
            inner_validator = self.outer_validator

            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=42),
            )
            study.optimize(
                lambda trial: _inner_objective(
                    trial, self.model_type, self.panel, inner_folds,
                    self.target_col, self.feature_cols, inner_validator,
                ),
                n_trials=self.n_trials,
                show_progress_bar=False,
            )

            best_params = study.best_params
            best_params_per_fold.append({
                "fold": outer_fold.fold_idx,
                "params": best_params,
                "inner_ic": study.best_value,
            })
            logger.info(f"  Best inner IC: {study.best_value:.4f}")

            # Retrain on full train+val with best params, evaluate on test
            train_df, val_df, test_df = self.outer_validator.split_data(
                self.panel, outer_fold,
            )

            # Combine train + val for final training
            full_train = pd.concat([train_df, val_df])
            model = create_model(self.model_type, best_params)

            fold_result = evaluate_model_on_fold(
                model, full_train, val_df, test_df,
                self.target_col, self.feature_cols,
            )
            fold_result.fold_idx = outer_fold.fold_idx
            fold_results.append(fold_result)

            logger.info(f"  Test IC: {fold_result.prediction.ic:.4f}, "
                         f"Sharpe: {fold_result.investment.sharpe_ratio:.2f}")

        evaluation = ModelEvaluation(
            model_name=self.model_type,
            horizon=self.target_col,
            fold_results=fold_results,
        )

        return {
            "best_params_per_fold": best_params_per_fold,
            "evaluation": evaluation,
            "summary": {
                "model": self.model_type,
                "mean_ic": evaluation.mean_ic,
                "icir": evaluation.icir,
                "mean_sharpe": evaluation.mean_sharpe,
                "mean_calmar": evaluation.mean_calmar,
                "overfit_ratio": evaluation.overfit_ratio,
                "n_folds": len(fold_results),
            },
        }


# ---------------------------------------------------------------------------
# Quick search (no nesting, for rapid prototyping)
# ---------------------------------------------------------------------------

def quick_search(
    model_type: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: List[str],
    n_trials: int = 30,
) -> Dict:
    """Quick hyperparameter search on a single train/val split.

    Use for rapid prototyping. For production, use HyperparameterSearcher.
    """
    search_fn = SEARCH_SPACES[model_type]

    def objective(trial):
        params = search_fn(trial)
        model = create_model(model_type, params)

        np.nan_to_num(X_train, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        np.nan_to_num(X_val, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        try:
            model.fit(X_train, y_train, X_val, y_val, feature_names=feature_names)
            val_pred = model.predict(X_val)
            valid = ~np.isnan(val_pred)
            if valid.sum() < 5:
                return -1.0
            metrics = compute_prediction_metrics(y_val[valid], val_pred[valid])
            return metrics.ic
        except Exception:
            return -1.0

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    return {
        "best_params": study.best_params,
        "best_ic": study.best_value,
        "best_ic_composite": study.best_value,
        "n_trials": n_trials,
    }
