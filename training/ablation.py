"""
Ablation Study Utilities
-------------------------
Tools for systematically disabling model branches and measuring impact.

Usage:
    config = AblationConfig(disable_vae=True)
    results = run_ablation_study(panel, target, features, [config], params)
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


@dataclass
class AblationConfig:
    """Configuration for ablation study."""
    disable_vae: bool = False
    disable_sentiment: bool = False
    disable_ti_embedding: bool = False
    disable_temporal: bool = False
    disable_ticker_embedding: bool = False
    name: str = ""

    def __post_init__(self):
        if not self.name:
            disabled = []
            if self.disable_vae:
                disabled.append("no_vae")
            if self.disable_sentiment:
                disabled.append("no_sentiment")
            if self.disable_ti_embedding:
                disabled.append("no_ti")
            if self.disable_temporal:
                disabled.append("no_temporal")
            if self.disable_ticker_embedding:
                disabled.append("no_ticker")
            self.name = "_".join(disabled) if disabled else "full_model"

    def to_dict(self) -> Dict[str, bool]:
        """Convert to dict for passing to HybridMultiModalNet."""
        return {
            "disable_vae": self.disable_vae,
            "disable_sentiment": self.disable_sentiment,
            "disable_ti_embedding": self.disable_ti_embedding,
            "disable_temporal": self.disable_temporal,
            "disable_ticker_embedding": self.disable_ticker_embedding,
        }


def log_gradient_norms(model) -> Dict[str, float]:
    """Log gradient norms for each named parameter group.

    Args:
        model: PyTorch nn.Module.

    Returns:
        Dict of param_group_name -> gradient L2 norm.
    """
    try:
        import torch
    except ImportError:
        return {}

    norms = {}  # type: Dict[str, float]
    group_norms = {}  # type: Dict[str, List[float]]

    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = float(param.grad.data.norm(2).item())
            # Group by first part of name (e.g., "temporal_encoder", "vae")
            group = name.split(".")[0]
            if group not in group_norms:
                group_norms[group] = []
            group_norms[group].append(grad_norm)

    for group, grad_list in group_norms.items():
        norms[group] = float(np.sqrt(sum(g ** 2 for g in grad_list)))

    return norms


def run_ablation_study(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_cols: List[str],
    configs: Optional[List[AblationConfig]] = None,
    base_model_params: Optional[Dict] = None,
) -> List[Dict]:
    """Run ablation study with multiple configurations.

    Args:
        X_train, y_train: Training data.
        X_val, y_val: Validation data.
        X_test, y_test: Test data.
        feature_cols: Feature column names.
        configs: List of AblationConfig. If None, uses default ablation suite.
        base_model_params: Base parameters for HybridMultiModalModel.

    Returns:
        List of dicts with {name, ic, rmse, config}.
    """
    from training.models import create_model
    from training.model_selection import compute_prediction_metrics

    if configs is None:
        configs = [
            AblationConfig(),  # full model
            AblationConfig(disable_vae=True),
            AblationConfig(disable_sentiment=True),
            AblationConfig(disable_ti_embedding=True),
            AblationConfig(disable_temporal=True),
            AblationConfig(disable_ticker_embedding=True),
        ]

    if base_model_params is None:
        base_model_params = {
            "epochs": 10,
            "sequence_length": 20,
            "patience": 5,
            "hidden_dim": 32,
            "fusion_dim": 32,
            "vae_latent_dim": 4,
            "batch_size": 32,
        }

    results = []
    for cfg in configs:
        params = dict(base_model_params)
        params["ablation_config"] = cfg.to_dict()

        try:
            model = create_model("hybrid_multimodal", params)
            model.fit(X_train, y_train, X_val, y_val, feature_names=feature_cols)
            preds = model.predict(X_test)

            # Filter NaN
            valid = ~np.isnan(preds)
            metrics = compute_prediction_metrics(
                y_test[valid], preds[valid],
            )

            results.append({
                "name": cfg.name,
                "ic": metrics.ic,
                "rmse": metrics.rmse,
                "hit_ratio": metrics.hit_ratio,
                "config": cfg,
            })
            logger.info("Ablation '%s': IC=%.4f, RMSE=%.4f",
                        cfg.name, metrics.ic, metrics.rmse)
        except Exception as e:
            logger.warning("Ablation '%s' failed: %s", cfg.name, e)
            results.append({
                "name": cfg.name,
                "ic": 0.0,
                "rmse": float("inf"),
                "hit_ratio": 0.0,
                "config": cfg,
                "error": str(e),
            })

    return results


# ---------------------------------------------------------------------------
# 3-Way Ablation for GatedHybridModel
# ---------------------------------------------------------------------------

def run_3way_ablation(
    panel: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    n_splits: int = 2,
    model_params: Optional[Dict] = None,
) -> Dict[str, Dict[str, float]]:
    """Run 3-way ablation: price-only, price+nlp, nlp-only.

    Each configuration trains a GatedHybridModel with walk-forward splits
    and reports IC, ICIR, Sharpe.

    Decision rule:
      If B.IC > A.IC + 0.005 AND B.IC > C.IC -> keep dual
      Else -> fallback to price-only

    Args:
        panel: Panel DataFrame with MultiIndex (date, ticker).
        feature_cols: List of feature column names.
        target_col: Target column name.
        n_splits: Number of walk-forward splits.
        model_params: Optional model parameters override.

    Returns:
        Dict with keys 'price_only', 'price_nlp', 'nlp_only', 'decision'.
        Each inner dict has 'ic', 'icir', 'sharpe'.
    """
    from training.models import create_model
    from training.model_selection import (
        WalkForwardValidator, WalkForwardConfig,
        compute_prediction_metrics, compute_investment_metrics,
    )

    nlp_cols = [c for c in feature_cols if c.startswith("nlp_")]
    price_cols = [c for c in feature_cols if not c.startswith("nlp_")]

    if model_params is None:
        model_params = {
            "epochs": 15,
            "sequence_length": 20,
            "patience": 5,
            "temporal_hidden": 32,
            "nlp_embed": 16,
            "batch_size": 32,
        }

    configs = {
        "price_only": {"feature_cols": price_cols, "zero_cols": nlp_cols},
        "price_nlp": {"feature_cols": feature_cols, "zero_cols": []},
        "nlp_only": {"feature_cols": feature_cols, "zero_cols": price_cols},
    }

    # Get dates for splits
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

    results = {}

    for config_name, cfg in configs.items():
        fold_ics = []
        fold_sharpes = []

        for fold in folds:
            try:
                train_df, val_df, test_df = validator.split_data(panel, fold)

                # Prepare data -- zero out specified columns
                all_cols = feature_cols
                X_train = train_df[all_cols].values.astype(np.float32)
                y_train = train_df[target_col].values.astype(np.float32)
                X_val = val_df[all_cols].values.astype(np.float32)
                y_val = val_df[target_col].values.astype(np.float32)
                X_test = test_df[all_cols].values.astype(np.float32)
                y_test = test_df[target_col].values.astype(np.float32)

                np.nan_to_num(X_train, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
                np.nan_to_num(X_val, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
                np.nan_to_num(X_test, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
                np.nan_to_num(y_train, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
                np.nan_to_num(y_val, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
                np.nan_to_num(y_test, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

                # Zero out columns per config
                if cfg["zero_cols"]:
                    zero_idx = [all_cols.index(c) for c in cfg["zero_cols"]
                                if c in all_cols]
                    X_train[:, zero_idx] = 0.0
                    X_val[:, zero_idx] = 0.0
                    X_test[:, zero_idx] = 0.0

                model = create_model("gated_hybrid", model_params)
                model.fit(X_train, y_train, X_val, y_val,
                          feature_names=all_cols)
                preds = model.predict(X_test)

                valid = ~np.isnan(preds)
                if valid.sum() < 5:
                    continue
                metrics = compute_prediction_metrics(y_test[valid], preds[valid])
                inv_metrics = compute_investment_metrics(preds[valid], y_test[valid])
                fold_ics.append(metrics.ic)
                fold_sharpes.append(inv_metrics.sharpe_ratio)
            except Exception as e:
                logger.warning("3-way ablation fold failed for %s: %s",
                               config_name, e)
                continue

        mean_ic = float(np.mean(fold_ics)) if fold_ics else 0.0
        ic_std = float(np.std(fold_ics)) if len(fold_ics) > 1 else 0.0
        icir = mean_ic / ic_std if ic_std > 1e-8 else 0.0
        mean_sharpe = float(np.mean(fold_sharpes)) if fold_sharpes else 0.0

        results[config_name] = {
            "ic": mean_ic,
            "icir": icir,
            "sharpe": mean_sharpe,
            "fold_ics": fold_ics,
        }
        logger.info("3-way ablation '%s': IC=%.4f, ICIR=%.2f, Sharpe=%.2f",
                     config_name, mean_ic, icir, mean_sharpe)

    # Decision
    a_ic = results.get("price_only", {}).get("ic", 0.0)
    b_ic = results.get("price_nlp", {}).get("ic", 0.0)
    c_ic = results.get("nlp_only", {}).get("ic", 0.0)

    if b_ic > a_ic + 0.005 and b_ic > c_ic:
        decision = "dual"
        logger.info("Ablation decision: DUAL (price+nlp)")
    else:
        decision = "price_only"
        logger.info("Ablation decision: PRICE-ONLY fallback")

    results["decision"] = {"mode": decision}
    return results


def compute_nlp_permutation_importance(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    n_repeats: int = 5,
) -> Dict[str, float]:
    """Compute permutation importance for nlp_* features.

    Args:
        model: Fitted model with predict() method.
        X_test: Test features.
        y_test: Test targets.
        feature_names: Feature column names.
        n_repeats: Number of permutation repeats.

    Returns:
        Dict of nlp_feature_name -> importance score.
    """
    from training.model_selection import compute_prediction_metrics

    # Baseline IC
    preds = model.predict(X_test)
    valid = ~np.isnan(preds)
    if valid.sum() < 5:
        return {}
    baseline_ic = compute_prediction_metrics(y_test[valid], preds[valid]).ic

    nlp_idx = [i for i, n in enumerate(feature_names) if n.startswith("nlp_")]
    importance = {}

    for idx in nlp_idx:
        drops = []
        for _ in range(n_repeats):
            X_perm = X_test.copy()
            X_perm[:, idx] = np.random.permutation(X_perm[:, idx])
            preds_perm = model.predict(X_perm)
            valid_p = ~np.isnan(preds_perm)
            if valid_p.sum() < 5:
                continue
            ic_perm = compute_prediction_metrics(
                y_test[valid_p], preds_perm[valid_p],
            ).ic
            drops.append(baseline_ic - ic_perm)
        importance[feature_names[idx]] = float(np.mean(drops)) if drops else 0.0

    # Log pruning candidates
    pruned = [k for k, v in importance.items() if v < 0.001]
    if pruned:
        logger.info("NLP features below importance 0.001: %s", pruned)

    return importance
