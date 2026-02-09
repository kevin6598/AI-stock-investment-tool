"""
Data Export / Import for Colab Training
-----------------------------------------
Export datasets and configs for training on Google Colab,
and import results back to local.

Components:
  - DataExporter: export panel datasets, configs, and import results
"""

from typing import List, Optional, Any
import json
import os
import logging

logger = logging.getLogger(__name__)


class DataExporter:
    """Export data for Colab training and import results back.

    Usage:
        exporter = DataExporter()
        path = exporter.export_dataset(["AAPL", "MSFT"], "5y", "data.parquet")
        exporter.export_config(configs, "configs.json")
        results = exporter.import_results("results.json")
    """

    def export_dataset(
        self,
        tickers: List[str],
        period: str = "5y",
        output_path: str = "dataset.parquet",
        forward_horizons: Optional[List[int]] = None,
    ) -> str:
        """Build panel dataset and save as parquet.

        Args:
            tickers: list of ticker symbols.
            period: data period for yfinance (e.g. "5y", "10y").
            output_path: path for output parquet file.
            forward_horizons: forward return horizons in trading days.

        Returns:
            Absolute path to the saved parquet file.
        """
        import pandas as pd
        from data.stock_api import get_historical_data, get_stock_info
        from training.feature_engineering import (
            build_panel_dataset, cross_sectional_normalize,
        )

        if forward_horizons is None:
            forward_horizons = [21, 63, 126]

        # Fetch data
        stock_dfs = {}
        stock_infos = {}
        for ticker in tickers:
            df = get_historical_data(ticker.upper(), period=period)
            if not df.empty:
                stock_dfs[ticker.upper()] = df
                info = get_stock_info(ticker.upper()) or {}
                stock_infos[ticker.upper()] = info

        if not stock_dfs:
            raise ValueError("No stock data fetched for any ticker")

        # Market data for macro features
        market_df = get_historical_data("SPY", period=period)
        if market_df.empty:
            market_df = None

        # Build panel
        panel = build_panel_dataset(
            stock_dfs, stock_infos, market_df, forward_horizons,
        )
        panel = cross_sectional_normalize(panel)

        # Save
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        panel.to_parquet(output_path)
        logger.info(f"Exported dataset to {output_path}: {panel.shape}")

        return os.path.abspath(output_path)

    @staticmethod
    def export_config(
        configs: List[Any],
        output_path: str = "configs.json",
    ) -> str:
        """Save ModelConfig list as JSON.

        Args:
            configs: list of ModelConfig instances.
            output_path: path for output JSON file.

        Returns:
            Absolute path to the saved JSON file.
        """
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        data = []
        for c in configs:
            if hasattr(c, "to_json"):
                data.append(c.to_json())
            elif isinstance(c, dict):
                data.append(c)

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Exported {len(data)} configs to {output_path}")
        return os.path.abspath(output_path)

    @staticmethod
    def import_results(results_path: str) -> List[Any]:
        """Load training results from Colab.

        Args:
            results_path: path to results JSON file.

        Returns:
            List of ConfigResult-like dicts.
        """
        from training.model_config import ModelConfig, ConfigResult

        with open(results_path, "r") as f:
            data = json.load(f)

        results = []
        for item in data:
            config = ModelConfig.from_json(item.get("config", {}))
            # Create a lightweight result object
            result = ConfigResult(
                config=config,
                evaluation=None,
                training_time=item.get("training_time", 0.0),
                best_params=item.get("best_params", {}),
            )
            results.append(result)

        logger.info(f"Imported {len(results)} results from {results_path}")
        return results
