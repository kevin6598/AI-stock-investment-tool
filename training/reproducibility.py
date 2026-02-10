"""
Reproducibility utilities for ML training.

Provides seed management and training configuration persistence
to ensure experiments can be reliably reproduced.
"""

import json
import os
import random
from datetime import datetime
from typing import Dict, List, Optional


def set_all_seeds(seed: int = 42) -> None:
    """Set random seeds across all relevant libraries for reproducibility.

    Sets seeds for:
      - Python's built-in random module
      - numpy
      - torch (CPU and CUDA, if available)
      - PYTHONHASHSEED environment variable
      - cudnn deterministic / benchmark flags

    Args:
        seed: The seed value to use. Defaults to 42.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def save_training_config(
    output_path: str,
    training_dates: Dict[str, str],
    tickers: List[str],
    model_version: str,
    feature_list: List[str],
    latent_dim: int,
    seq_length: int,
    horizons: List[str],
    extra_params: Optional[Dict] = None,
) -> str:
    """Save a training configuration to a JSON file.

    Persists all parameters that define a training run so the experiment
    can be reproduced later.

    Args:
        output_path: Directory or full file path where the config will be saved.
            If a directory is given, the file will be named
            ``training_config_<timestamp>.json``.
        training_dates: Mapping with date range info, e.g.
            ``{"start": "2020-01-01", "end": "2024-12-31"}``.
        tickers: List of ticker symbols used for training.
        model_version: Identifier for the model version.
        feature_list: Names of features used during training.
        latent_dim: Latent dimension size (e.g. for VAE / embeddings).
        seq_length: Sequence length fed to temporal models.
        horizons: Forecast horizons, e.g. ``["1M", "3M"]``.
        extra_params: Any additional parameters to record. Defaults to None.

    Returns:
        The absolute path to the saved JSON config file.
    """
    config: Dict = {
        "timestamp": datetime.utcnow().isoformat(),
        "training_dates": training_dates,
        "tickers": tickers,
        "model_version": model_version,
        "feature_list": feature_list,
        "latent_dim": latent_dim,
        "seq_length": seq_length,
        "horizons": horizons,
    }

    if extra_params is not None:
        config["extra_params"] = extra_params

    # Determine the final file path
    if os.path.isdir(output_path):
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(output_path, "training_config_{}.json".format(ts))
    else:
        parent_dir = os.path.dirname(output_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        file_path = output_path

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    return os.path.abspath(file_path)


def load_training_config(path: str) -> Dict:
    """Load a previously saved training configuration.

    Args:
        path: Path to the JSON config file.

    Returns:
        A dictionary containing all stored training parameters.

    Raises:
        FileNotFoundError: If the config file does not exist.
        json.JSONDecodeError: If the file contains invalid JSON.
    """
    with open(path, "r", encoding="utf-8") as f:
        config: Dict = json.load(f)
    return config
