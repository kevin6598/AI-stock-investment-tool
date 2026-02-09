"""
CLI Entry Point for Stock Forecasting Pipeline
-----------------------------------------------
Usage:
    python -m training.run_forecast --tickers AAPL MSFT GOOGL --horizons 1M 3M 6M
    python -m training.run_forecast --tickers AAPL --models elastic_net lightgbm
"""

import argparse
import logging
import json
import sys
import os

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.forecast import ForecastPipeline
from training.model_selection import WalkForwardConfig


def main():
    parser = argparse.ArgumentParser(description="AI Stock Forecasting Pipeline")
    parser.add_argument(
        "--tickers", nargs="+", required=True,
        help="Stock tickers to forecast (e.g., AAPL MSFT GOOGL)",
    )
    parser.add_argument(
        "--market", default="SPY",
        help="Market index ticker (default: SPY)",
    )
    parser.add_argument(
        "--horizons", nargs="+", default=["1M", "3M", "6M"],
        help="Forecast horizons (default: 1M 3M 6M)",
    )
    parser.add_argument(
        "--models", nargs="+", default=["elastic_net", "lightgbm"],
        help="Model types to train (default: elastic_net lightgbm). "
             "Options: elastic_net, lightgbm, lstm_attention, transformer",
    )
    parser.add_argument(
        "--period", default="5y",
        help="Historical data period (default: 5y)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Path to save JSON results (optional)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--save-model", action="store_true",
        help="Save trained models to model registry after training",
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Map horizon names to trading days
    horizon_map = {
        "1M": 21, "3M": 63, "6M": 126, "1Y": 252, "3Y": 756,
    }
    horizons = {}
    for h in args.horizons:
        if h in horizon_map:
            horizons[h] = horizon_map[h]
        else:
            print(f"Unknown horizon: {h}. Use: {list(horizon_map.keys())}")
            sys.exit(1)

    # Walk-forward config adapted to data period
    wf_config = WalkForwardConfig(
        train_start="2016-01-01",
        test_end="2026-02-08",
        train_min_months=24,
        val_months=6,
        test_months=6,
        step_months=6,
        embargo_days=max(horizons.values()) + 5,
        expanding=True,
    )

    print(f"\nStock Forecasting Pipeline")
    print(f"  Tickers: {args.tickers}")
    print(f"  Horizons: {list(horizons.keys())}")
    print(f"  Models: {args.models}")
    print(f"  Data period: {args.period}")
    print()

    # Run pipeline
    pipeline = ForecastPipeline(
        tickers=args.tickers,
        market_ticker=args.market,
        horizons=horizons,
        model_types=args.models,
        data_period=args.period,
        walk_forward_config=wf_config,
    )

    try:
        results = pipeline.run()
        pipeline.print_summary()

        # Save models to registry
        if args.save_model:
            try:
                from training.model_versioning import ModelRegistry
                registry = ModelRegistry()
                print("\nSaving models to registry...")
                for horizon_name, evals in results.get("evaluations", {}).items():
                    for model_name, ev in evals.items():
                        metrics = {
                            "mean_ic": ev.mean_ic,
                            "icir": ev.icir,
                            "mean_sharpe": ev.mean_sharpe,
                        }
                        # Re-create and train a model for saving
                        from training.models import create_model as cm
                        model = cm(model_name)
                        vid = registry.save_model(
                            model, model_name, horizon_name,
                            params=model.params, metrics=metrics,
                        )
                        registry.activate_version(vid)
                        print(f"  Saved: {vid}")
            except Exception as save_err:
                print(f"  Warning: model save failed: {save_err}")

        # Save JSON output
        if args.output:
            # Make JSON-serializable (strip numpy types)
            serializable = _make_serializable(results)
            with open(args.output, "w") as f:
                json.dump(serializable, f, indent=2)
            print(f"\nResults saved to: {args.output}")

    except Exception as e:
        logging.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


def _make_serializable(obj):
    """Recursively convert numpy types to Python types for JSON serialization."""
    import numpy as np

    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, '__dict__') and not isinstance(obj, type):
        # Dataclass or custom object — try to convert to dict
        try:
            return _make_serializable(obj.__dict__)
        except Exception:
            return str(obj)
    else:
        return obj


if __name__ == "__main__":
    main()
