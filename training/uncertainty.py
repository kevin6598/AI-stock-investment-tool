"""
Uncertainty-Aware Scaling
--------------------------
Monte Carlo dropout and fallback uncertainty estimation for scaling alpha.

  1. MC dropout: N forward passes with dropout enabled -> mean & variance
  2. Fallback: quantile spread for non-neural models
  3. Uncertainty-aware alpha scaling: alpha * meta_prob / (1 + uncertainty)
"""

from typing import Optional, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)


def mc_dropout_predict(
    model,
    X: np.ndarray,
    n_forward_passes: int = 30,
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate prediction uncertainty via Monte Carlo dropout.

    Enables training mode (for dropout) and runs N forward passes
    to compute mean predictions and variance.

    Args:
        model: A PyTorch model (nn.Module) with dropout layers.
        X: Input tensor as numpy array, shape (n, features) or (n, seq, features).
        n_forward_passes: Number of stochastic forward passes.
        device: Device string.

    Returns:
        (mean_predictions, variance) each of shape (n,).
    """
    try:
        import torch
    except ImportError:
        logger.warning("torch not available; falling back to zero variance")
        mean = np.zeros(X.shape[0])
        variance = np.ones(X.shape[0]) * 0.01
        return mean, variance

    model.train()  # Enable dropout

    X_tensor = torch.from_numpy(X.astype(np.float32)).to(device)

    all_preds = []
    with torch.no_grad():
        for _ in range(n_forward_passes):
            # Handle sequential models (3D input)
            if X_tensor.dim() == 3:
                x_static = X_tensor[:, -1, :]
                ticker_ids = torch.zeros(
                    X_tensor.size(0), dtype=torch.long, device=device,
                )
                out = model(X_tensor, x_static, ticker_ids)
                if isinstance(out, dict):
                    # Hybrid model: use median quantile
                    if "quantiles" in out:
                        preds = out["quantiles"][:, 3].cpu().numpy()
                    elif "p_up" in out:
                        preds = out["p_up"].cpu().numpy()
                    else:
                        preds = list(out.values())[0].cpu().numpy()
                        if preds.ndim > 1:
                            preds = preds[:, 0]
                else:
                    preds = out.cpu().numpy()
                    if preds.ndim > 1:
                        preds = preds[:, 0]
            else:
                out = model(X_tensor)
                if isinstance(out, dict):
                    if "quantiles" in out:
                        preds = out["quantiles"][:, 3].cpu().numpy()
                    else:
                        preds = list(out.values())[0].cpu().numpy()
                        if preds.ndim > 1:
                            preds = preds[:, 0]
                elif isinstance(out, torch.Tensor):
                    preds = out.cpu().numpy()
                    if preds.ndim > 1:
                        preds = preds[:, 0]
                else:
                    preds = np.array(out)

            all_preds.append(preds.flatten())

    model.eval()  # Restore eval mode

    all_preds_arr = np.array(all_preds)  # (n_passes, n)
    mean_pred = np.mean(all_preds_arr, axis=0)
    variance = np.var(all_preds_arr, axis=0)

    return mean_pred, variance


def compute_uncertainty_fallback(
    model,
    X: np.ndarray,
) -> np.ndarray:
    """Compute uncertainty for non-neural models using quantile spread.

    Uncertainty = (q90 - q10) / 2.56  (approximate std from 80% CI)

    Args:
        model: A model with predict_quantiles() method.
        X: Input features, shape (n, features).

    Returns:
        Uncertainty array, shape (n,).
    """
    try:
        q_preds = model.predict_quantiles(X, [0.10, 0.90])
        q10 = q_preds.get(0.10, np.zeros(X.shape[0]))
        q90 = q_preds.get(0.90, np.zeros(X.shape[0]))

        # Handle NaN
        q10 = np.nan_to_num(q10, nan=0.0)
        q90 = np.nan_to_num(q90, nan=0.0)

        uncertainty = (q90 - q10) / 2.56
        uncertainty = np.maximum(uncertainty, 1e-6)
        return uncertainty
    except Exception as e:
        logger.warning("Uncertainty fallback failed: %s", e)
        return np.full(X.shape[0], 0.01)


def scale_alpha_with_uncertainty(
    base_alpha: np.ndarray,
    meta_trade_probability: np.ndarray,
    uncertainty: np.ndarray,
) -> np.ndarray:
    """Scale alpha predictions by meta-probability and uncertainty.

    scaled_alpha = alpha * meta_prob / (1 + uncertainty)

    Higher uncertainty reduces the alpha; higher meta-trade probability
    amplifies it.

    Args:
        base_alpha: Raw alpha predictions, shape (n,).
        meta_trade_probability: P(profitable trade), shape (n,).
        uncertainty: Prediction uncertainty, shape (n,).

    Returns:
        Scaled alpha array, shape (n,).
    """
    uncertainty = np.maximum(uncertainty, 0.0)
    meta_trade_probability = np.clip(meta_trade_probability, 0.0, 1.0)

    scaled = base_alpha * meta_trade_probability / (1.0 + uncertainty)
    return scaled
