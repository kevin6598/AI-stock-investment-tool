"""
Colab Training Script
----------------------
Structured training script for Google Colab.
Can be converted to .ipynb or run as a Python script.

Steps:
  1. Mount Google Drive, load data from parquet
  2. Build features (reuses training.feature_engineering)
  3. Walk-forward validation loop
  4. Multi-config training (uses MultiConfigRunner)
  5. HP search via Optuna (reuses training.hyperparameter_search)
  6. Save best models to models_registry/ folder
  7. Export results JSON for local import

Usage in Colab:
  !pip install yfinance lightgbm torch optuna pyarrow
  %run train_notebook.py --data_path /content/drive/MyDrive/data.parquet
"""

import argparse
import json
import os
import sys
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def mount_drive():
    """Mount Google Drive if running in Colab."""
    try:
        from google.colab import drive
        drive.mount("/content/drive")
        logger.info("Google Drive mounted")
        return True
    except ImportError:
        logger.info("Not running in Colab, skipping drive mount")
        return False


def load_data(data_path):
    """Load panel dataset from parquet."""
    import pandas as pd
    panel = pd.read_parquet(data_path)
    logger.info(f"Loaded dataset: {panel.shape}")
    return panel


def load_configs(config_path):
    """Load model configs from JSON."""
    from training.model_config import ModelConfig
    with open(config_path, "r") as f:
        data = json.load(f)
    configs = [ModelConfig.from_json(d) for d in data]
    logger.info(f"Loaded {len(configs)} configs")
    return configs


def run_training(panel, configs, target_col, feature_cols, horizon, output_dir):
    """Run multi-config training with walk-forward validation."""
    from training.model_config import MultiConfigRunner
    from training.model_selection import WalkForwardConfig

    wf_config = WalkForwardConfig(
        train_start="2015-01-01",
        test_end="2025-01-01",
        train_min_months=36,
        val_months=6,
        test_months=6,
        step_months=6,
    )

    runner = MultiConfigRunner(save_to_registry=False)
    results = runner.run(
        configs=configs,
        panel=panel,
        target_col=target_col,
        feature_cols=feature_cols,
        wf_config=wf_config,
        horizon=horizon,
    )

    logger.info(f"Training complete: {len(results)} configs evaluated")
    return results


def save_results(results, output_dir):
    """Save training results to models_registry/ folder structure."""
    os.makedirs(output_dir, exist_ok=True)

    results_data = []
    for i, result in enumerate(results):
        # Save model artifacts
        model_dir = os.path.join(
            output_dir,
            f"{result.config.model_type}_v{i}",
        )
        os.makedirs(model_dir, exist_ok=True)

        # Save config
        config_path = os.path.join(model_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(result.config.to_json(), f, indent=2)

        # Save metrics
        metrics = {}
        if result.evaluation is not None:
            metrics = {
                "mean_ic": result.evaluation.mean_ic,
                "icir": result.evaluation.icir,
                "mean_sharpe": result.evaluation.mean_sharpe,
                "mean_mdd": result.evaluation.mean_mdd,
                "n_folds": len(result.evaluation.fold_results),
            }
        metrics_path = os.path.join(model_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        results_data.append({
            "config": result.config.to_json(),
            "training_time": result.training_time,
            "best_params": result.best_params,
            "metrics": metrics,
        })

    # Save combined results JSON
    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results_data, f, indent=2)

    logger.info(f"Results saved to {output_dir}")
    return results_path


def main():
    parser = argparse.ArgumentParser(description="Colab Training Script")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to parquet dataset")
    parser.add_argument("--config_path", type=str, default=None,
                        help="Path to configs JSON (optional)")
    parser.add_argument("--output_dir", type=str, default="models_registry",
                        help="Output directory for models and results")
    parser.add_argument("--horizon", type=str, default="1M",
                        help="Forecast horizon (1M, 3M, 6M)")
    parser.add_argument("--target_col", type=str, default="fwd_return_21d",
                        help="Target column name")
    args = parser.parse_args()

    # Step 1: Mount drive
    mount_drive()

    # Step 2: Load data
    panel = load_data(args.data_path)

    # Determine feature columns
    feature_cols = [
        c for c in panel.columns
        if not c.startswith("fwd_return_") and c != "_close"
    ]

    # Step 3: Load or generate configs
    if args.config_path and os.path.exists(args.config_path):
        configs = load_configs(args.config_path)
    else:
        # Default configs: one per model type
        from training.model_config import ModelConfig
        configs = [
            ModelConfig(model_type="elastic_net", learning_rate=0.1, epochs=1),
            ModelConfig(model_type="lightgbm", learning_rate=0.05, epochs=500),
            ModelConfig(model_type="lstm_attention", learning_rate=1e-3, epochs=100),
        ]

    # Step 4: Train
    results = run_training(
        panel, configs, args.target_col, feature_cols, args.horizon, args.output_dir,
    )

    # Step 5: Save results
    save_results(results, args.output_dir)

    # Print summary
    logger.info("\n=== Training Summary ===")
    for r in results:
        ic = r.evaluation.mean_ic if r.evaluation else 0.0
        logger.info(
            f"  {r.config.model_type}: IC={ic:.4f}, "
            f"time={r.training_time:.1f}s"
        )


if __name__ == "__main__":
    main()
