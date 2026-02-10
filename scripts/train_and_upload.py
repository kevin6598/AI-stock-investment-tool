"""
Training and Upload CLI
-----------------------
Automated retraining pipeline that:
  1. Fetches data for a universe of tickers
  2. Trains the hybrid multi-modal model
  3. Exports artifacts (model, scaler, config, metadata)
  4. Optionally uploads to cloud storage (GCS/S3)

Usage:
    python -m scripts.train_and_upload \
        --tickers AAPL MSFT GOOGL \
        --horizons 1M 3M \
        --model-type hybrid_multimodal \
        --output-dir ./artifacts \
        --upload gcs --bucket my-model-bucket
"""

import sys
import os
import argparse
import json
import pickle
import logging
from datetime import datetime
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Default universe of 50 tickers
DEFAULT_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "BRK-B", "JPM", "JNJ",
    "V", "PG", "UNH", "HD", "MA", "DIS", "PYPL", "BAC", "NFLX", "ADBE",
    "CRM", "CMCSA", "XOM", "VZ", "KO", "INTC", "PEP", "ABT", "CSCO", "TMO",
    "COST", "MRK", "WMT", "AVGO", "ACN", "CVX", "NKE", "LLY", "MCD", "TXN",
    "QCOM", "DHR", "UPS", "BMY", "PM", "LIN", "NEE", "ORCL", "RTX", "HON",
]


def train_model(
    tickers: List[str],
    horizons: List[str],
    model_type: str = "hybrid_multimodal",
    period: str = "5y",
    output_dir: str = "./artifacts",
) -> Dict:
    """Train model and export artifacts.

    Returns:
        Dict with training metadata and file paths.
    """
    from data.stock_api import get_historical_data, get_stock_info
    from training.feature_engineering import (
        build_panel_dataset, cross_sectional_normalize,
        add_ticker_embedding_column,
    )
    from training.models import create_model
    from training.reproducibility import set_all_seeds, save_training_config
    from training.model_selection import WalkForwardValidator, WalkForwardConfig

    set_all_seeds(42)

    horizon_map = {"1M": 21, "3M": 63, "6M": 126, "1Y": 252}
    horizon_days = [horizon_map[h] for h in horizons]

    logger.info("Fetching data for %d tickers...", len(tickers))
    stock_dfs = {}
    stock_infos = {}
    for ticker in tickers:
        try:
            df = get_historical_data(ticker, period=period)
            if not df.empty and len(df) > 300:
                stock_dfs[ticker] = df
                stock_infos[ticker] = get_stock_info(ticker) or {}
        except Exception as e:
            logger.warning("Failed to fetch %s: %s", ticker, e)

    if len(stock_dfs) < 2:
        raise ValueError("Only %d tickers fetched; need at least 2" % len(stock_dfs))

    valid_tickers = sorted(stock_dfs.keys())
    logger.info("Building panel for %d tickers...", len(valid_tickers))

    market_df = get_historical_data("SPY", period=period)
    if market_df.empty:
        market_df = None

    panel = build_panel_dataset(stock_dfs, stock_infos, market_df, horizon_days)
    panel = cross_sectional_normalize(panel)
    panel, ticker_to_id = add_ticker_embedding_column(panel, valid_tickers)

    target_col = "fwd_return_%dd" % horizon_days[0]
    feature_cols = [c for c in panel.columns
                    if not c.startswith("fwd_return_")
                    and not c.startswith("residual_return_")
                    and not c.startswith("ranked_target_")
                    and c not in ("_close", "ticker_id")]

    X = panel[feature_cols].values.astype(np.float32)
    y = panel[target_col].values.astype(np.float32)
    np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    np.nan_to_num(y, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    # Walk-forward split
    dates = panel.index.get_level_values(0).unique().sort_values()
    wf_config = WalkForwardConfig(
        train_start=str(dates[0].date()),
        test_end=str(dates[-1].date()),
        train_min_months=36,
        val_months=6,
        test_months=6,
        step_months=6,
        embargo_days=21,
        expanding=True,
    )
    validator = WalkForwardValidator(wf_config)
    folds = validator.generate_folds(dates)

    # Use last fold for final train/val/test split
    if folds:
        last_fold = folds[-1]
        train_mask = dates <= last_fold.train_end
        val_mask = (dates > last_fold.train_end) & (dates <= last_fold.val_end)
        train_dates = dates[train_mask]
        val_dates = dates[val_mask]
        split = len(train_dates) * len(valid_tickers)
        val_split = split + len(val_dates) * len(valid_tickers)
    else:
        split = int(len(X) * 0.8)
        val_split = int(len(X) * 0.9)

    logger.info("Training %s model (%d features, %d samples)...", model_type, X.shape[1], len(X))
    model = create_model(model_type, {
        "epochs": 50,
        "patience": 10,
        "n_tickers": len(valid_tickers),
    })
    model.fit(
        X[:split], y[:split],
        X[split:val_split], y[split:val_split],
        feature_names=feature_cols,
    )

    # Export artifacts
    os.makedirs(output_dir, exist_ok=True)

    # Save model using state_dict for PyTorch models
    if hasattr(model, 'net'):
        model.net.eval()
        model_path = os.path.join(output_dir, "model.pt")
        torch.save(model.net.state_dict(), model_path)
        logger.info("Model state_dict saved to %s", model_path)
        # Also save full model as pickle for backward compatibility
        pkl_path = os.path.join(output_dir, "model.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(model, f)
    else:
        model_path = os.path.join(output_dir, "model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        logger.info("Model saved to %s", model_path)

    # Save training config for reproducibility
    save_training_config(
        output_path=output_dir,
        training_dates=(str(dates[0].date()), str(dates[-1].date())),
        tickers=valid_tickers,
        model_version="%s_v%s" % (model_type, datetime.now().strftime("%Y%m%d_%H%M%S")),
        feature_list=feature_cols,
        latent_dim=32,
        seq_length=60,
        horizons=horizons,
    )

    # Save feature scaler
    if hasattr(model, 'scaler'):
        scaler_path = os.path.join(output_dir, "feature_scaler.pkl")
        with open(scaler_path, "wb") as f:
            pickle.dump(model.scaler, f)

    # Save config
    config = {
        "model_type": model_type,
        "horizons": horizons,
        "horizon_days": horizon_days,
        "n_features": len(feature_cols),
        "n_tickers": len(valid_tickers),
    }
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Save feature columns
    with open(os.path.join(output_dir, "feature_columns.json"), "w") as f:
        json.dump(feature_cols, f)

    # Save ticker list
    with open(os.path.join(output_dir, "ticker_list.json"), "w") as f:
        json.dump(valid_tickers, f)

    # Save metadata
    metadata = {
        "version": f"{model_type}_v{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_tickers": len(valid_tickers),
        "n_features": len(feature_cols),
        "n_samples": len(X),
        "train_size": split,
        "val_size": val_split - split,
        "test_size": len(X) - val_split,
        "tickers": valid_tickers,
        "horizons": horizons,
        "period": period,
    }
    with open(os.path.join(output_dir, "training_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"All artifacts exported to {output_dir}")
    return metadata


def upload_to_gcs(artifact_dir: str, bucket: str, prefix: str = "models/latest"):
    """Upload artifacts to Google Cloud Storage."""
    try:
        from google.cloud import storage
        client = storage.Client()
        bucket_obj = client.bucket(bucket)

        for filename in os.listdir(artifact_dir):
            filepath = os.path.join(artifact_dir, filename)
            if os.path.isfile(filepath):
                blob = bucket_obj.blob(f"{prefix}/{filename}")
                blob.upload_from_filename(filepath)
                logger.info(f"Uploaded {filename} to gs://{bucket}/{prefix}/{filename}")
    except ImportError:
        logger.error("google-cloud-storage not installed. Install with: pip install google-cloud-storage")
    except Exception as e:
        logger.error(f"GCS upload failed: {e}")


def upload_to_s3(artifact_dir: str, bucket: str, prefix: str = "models/latest"):
    """Upload artifacts to AWS S3."""
    try:
        import boto3
        s3 = boto3.client("s3")

        for filename in os.listdir(artifact_dir):
            filepath = os.path.join(artifact_dir, filename)
            if os.path.isfile(filepath):
                key = f"{prefix}/{filename}"
                s3.upload_file(filepath, bucket, key)
                logger.info(f"Uploaded {filename} to s3://{bucket}/{key}")
    except ImportError:
        logger.error("boto3 not installed. Install with: pip install boto3")
    except Exception as e:
        logger.error(f"S3 upload failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="Train model and upload artifacts")
    parser.add_argument("--tickers", nargs="+", default=None,
                        help="Ticker symbols (default: 50-stock universe)")
    parser.add_argument("--horizons", nargs="+", default=["1M", "3M"],
                        help="Forecast horizons")
    parser.add_argument("--model-type", default="hybrid_multimodal",
                        choices=["elastic_net", "lightgbm", "lstm_attention",
                                 "transformer", "hybrid_multimodal"])
    parser.add_argument("--period", default="5y", help="Historical data period")
    parser.add_argument("--output-dir", default="./artifacts")
    parser.add_argument("--upload", choices=["gcs", "s3", "none"], default="none")
    parser.add_argument("--bucket", default=None, help="Cloud storage bucket name")
    parser.add_argument("--prefix", default="models/latest", help="Upload prefix")

    args = parser.parse_args()

    tickers = args.tickers or DEFAULT_TICKERS

    metadata = train_model(
        tickers=tickers,
        horizons=args.horizons,
        model_type=args.model_type,
        period=args.period,
        output_dir=args.output_dir,
    )

    print(f"\nTraining complete: {metadata['version']}")
    print(f"  Tickers: {metadata['n_tickers']}")
    print(f"  Features: {metadata['n_features']}")
    print(f"  Samples: {metadata['n_samples']}")

    if args.upload == "gcs" and args.bucket:
        upload_to_gcs(args.output_dir, args.bucket, args.prefix)
    elif args.upload == "s3" and args.bucket:
        upload_to_s3(args.output_dir, args.bucket, args.prefix)


if __name__ == "__main__":
    main()
