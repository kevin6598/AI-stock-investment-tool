"""
Institutional Risk Engine
--------------------------
Multi-layered risk management system inspired by institutional portfolio
management practices.

Pipeline:
  1. Drawdown check -- trigger risk-off mode if drawdown exceeds threshold
  2. Tail risk detection -- z-score based detection of abnormal returns
  3. Exposure caps -- per-asset and per-sector position limits
  4. Correlation clustering -- identify correlated asset groups
  5. Volatility scaling -- scale portfolio to target volatility
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np


@dataclass
class RiskLimits:
    """Risk policy limits."""
    target_vol: float = 0.15
    max_drawdown: float = 0.10
    max_single_asset: float = 0.25
    max_sector: float = 0.40
    tail_risk_z: float = 2.5
    risk_off_scale: float = 0.3
    corr_spike_threshold: float = 0.3
    min_liquidity_ratio: float = 0.01
    hedge_allocation: float = 0.05


@dataclass
class RiskReport:
    """Output of the risk evaluation pipeline."""
    adjusted_weights: Dict[str, float]
    risk_off_mode: bool
    drawdown: float
    vol: float
    tail_risk_flag: bool
    clusters: List[List[str]]
    violations: List[str]
    scaling_factor: float


class RiskEngine:
    """Multi-layered institutional risk management engine."""

    def __init__(self, limits: Optional[RiskLimits] = None):
        self.limits = limits or RiskLimits()
        self._peak_value = None  # type: Optional[float]

    def volatility_scale(
        self,
        weights: Dict[str, float],
        covariance: np.ndarray,
        tickers: List[str],
    ) -> Tuple[Dict[str, float], float]:
        """Scale portfolio weights to achieve target volatility.

        Args:
            weights: current weights dict.
            covariance: covariance matrix.
            tickers: ordered ticker list matching covariance rows.

        Returns:
            Tuple of (scaled_weights, scaling_factor).
        """
        w = np.array([weights.get(t, 0.0) for t in tickers])
        port_vol = np.sqrt(max(w @ covariance @ w, 1e-12))
        scaling = self.limits.target_vol / port_vol if port_vol > 0 else 1.0
        scaling = min(scaling, 1.0)  # never lever up

        scaled_w = w * scaling
        # Re-normalize to sum to 1
        total = scaled_w.sum()
        if total > 0:
            scaled_w = scaled_w / total

        return {t: float(sw) for t, sw in zip(tickers, scaled_w)}, float(scaling)

    def check_drawdown(self, current_value: float) -> Tuple[bool, float]:
        """Check if portfolio drawdown exceeds threshold.

        Args:
            current_value: current portfolio value.

        Returns:
            Tuple of (risk_off_triggered, current_drawdown).
        """
        if self._peak_value is None:
            self._peak_value = current_value
        else:
            self._peak_value = max(self._peak_value, current_value)

        drawdown = (self._peak_value - current_value) / self._peak_value if self._peak_value > 0 else 0.0
        triggered = drawdown >= self.limits.max_drawdown
        return triggered, drawdown

    def cluster_correlations(
        self,
        corr_matrix: np.ndarray,
        tickers: List[str],
        threshold: float = 0.7,
    ) -> List[List[str]]:
        """Group assets with high pairwise correlation.

        Simple greedy clustering: start from each unvisited asset,
        group all assets with correlation above threshold.

        Args:
            corr_matrix: correlation matrix.
            tickers: ticker names matching matrix rows.
            threshold: correlation threshold for grouping.

        Returns:
            List of correlated asset groups (each group is a list of tickers).
        """
        n = len(tickers)
        visited = set()  # type: set
        clusters = []

        for i in range(n):
            if i in visited:
                continue
            cluster = [tickers[i]]
            visited.add(i)
            for j in range(i + 1, n):
                if j not in visited and abs(corr_matrix[i, j]) >= threshold:
                    cluster.append(tickers[j])
                    visited.add(j)
            if len(cluster) > 1:
                clusters.append(cluster)

        return clusters

    def detect_tail_risk(
        self,
        recent_returns: np.ndarray,
        window: int = 21,
    ) -> bool:
        """Detect tail risk events using z-score of recent returns.

        Args:
            recent_returns: array of recent daily returns.
            window: lookback window.

        Returns:
            True if tail risk detected.
        """
        if len(recent_returns) < window:
            return False

        windowed = recent_returns[-window:]
        mean = np.mean(windowed)
        std = np.std(windowed)
        if std < 1e-10:
            return False

        z_score = abs(mean / std) * np.sqrt(window)
        return z_score > self.limits.tail_risk_z

    def detect_correlation_spike(
        self,
        current_corr: np.ndarray,
        baseline_corr: np.ndarray,
    ) -> bool:
        """Detect spike in average pairwise correlation.

        A sudden increase in correlation suggests systemic risk is rising
        (assets moving together = less diversification benefit).

        Args:
            current_corr: current correlation matrix.
            baseline_corr: historical baseline correlation matrix.

        Returns:
            True if correlation spike detected.
        """
        n = current_corr.shape[0]
        if n < 2:
            return False

        # Compute mean off-diagonal correlation
        mask = ~np.eye(n, dtype=bool)
        current_mean = np.mean(np.abs(current_corr[mask]))
        baseline_mean = np.mean(np.abs(baseline_corr[mask]))

        increase = current_mean - baseline_mean
        return increase > self.limits.corr_spike_threshold

    def check_liquidity_stress(
        self,
        volumes: np.ndarray,
        avg_volumes: np.ndarray,
        tickers: List[str],
    ) -> List[str]:
        """Flag tickers with abnormally low volume (liquidity stress).

        Args:
            volumes: recent volume per ticker.
            avg_volumes: average volume per ticker.
            tickers: ticker names.

        Returns:
            List of tickers with liquidity stress.
        """
        stressed = []
        for i, t in enumerate(tickers):
            if i < len(volumes) and i < len(avg_volumes):
                avg = avg_volumes[i]
                if avg > 0 and volumes[i] < self.limits.min_liquidity_ratio * avg:
                    stressed.append(t)
        return stressed

    def compute_hedge_allocation(
        self,
        regime: str,
        risk_off: bool,
    ) -> float:
        """Compute fraction of portfolio to reserve for hedging/cash.

        In crisis/bear regime, allocate hedge_allocation to defensive assets.

        Args:
            regime: current regime string ("crisis", "bear", "normal", "strong_bull").
            risk_off: whether risk-off mode is active.

        Returns:
            Hedge weight (0 to hedge_allocation), deducted from risk assets.
        """
        if risk_off or regime == "crisis":
            return self.limits.hedge_allocation
        elif regime == "bear":
            return self.limits.hedge_allocation * 0.5
        return 0.0

    def apply_exposure_caps(
        self,
        weights: Dict[str, float],
        sector_map: Optional[Dict[str, str]] = None,
    ) -> Tuple[Dict[str, float], List[str]]:
        """Enforce per-asset and per-sector position limits.

        Args:
            weights: proposed weights.
            sector_map: ticker -> sector mapping.

        Returns:
            Tuple of (capped_weights, list of violation messages).
        """
        violations = []
        capped = dict(weights)

        # Per-asset caps
        for t, w in capped.items():
            if w > self.limits.max_single_asset:
                violations.append(f"{t}: weight {w:.2%} exceeds max {self.limits.max_single_asset:.2%}")
                capped[t] = self.limits.max_single_asset

        # Re-normalize after per-asset caps
        total = sum(capped.values())
        if total > 0:
            capped = {t: v / total for t, v in capped.items()}

        # Sector caps
        if sector_map is not None:
            sector_totals = {}  # type: Dict[str, float]
            for t, w in capped.items():
                sec = sector_map.get(t, "Other")
                sector_totals[sec] = sector_totals.get(sec, 0.0) + w

            for sec, sw in sector_totals.items():
                if sw > self.limits.max_sector:
                    violations.append(f"Sector {sec}: {sw:.2%} exceeds max {self.limits.max_sector:.2%}")
                    # Scale down all tickers in this sector
                    scale = self.limits.max_sector / sw
                    for t in capped:
                        if sector_map.get(t, "Other") == sec:
                            capped[t] *= scale

            # Re-normalize
            total = sum(capped.values())
            if total > 0:
                capped = {t: v / total for t, v in capped.items()}

        return capped, violations

    def evaluate(
        self,
        proposed_weights: Dict[str, float],
        covariance: np.ndarray,
        tickers: List[str],
        current_value: float = 1.0,
        recent_returns: Optional[np.ndarray] = None,
        sector_map: Optional[Dict[str, str]] = None,
        corr_matrix: Optional[np.ndarray] = None,
        baseline_corr: Optional[np.ndarray] = None,
        volumes: Optional[np.ndarray] = None,
        avg_volumes: Optional[np.ndarray] = None,
        regime: str = "normal",
    ) -> RiskReport:
        """Run the full risk evaluation pipeline.

        Steps:
          1. Drawdown check -> risk-off mode
          2. Tail risk detection -> risk-off mode
          2.5. Correlation spike check -> warning
          3. Exposure caps -> enforce limits
          3.5. Liquidity stress check -> warning
          4. Correlation clusters -> informational
          5. Volatility scaling -> target vol
          5.5. Hedge allocation -> reserve for defensive assets

        Args:
            proposed_weights: initial portfolio weights.
            covariance: covariance matrix.
            tickers: ticker list matching covariance rows.
            current_value: current portfolio value for drawdown tracking.
            recent_returns: recent daily portfolio returns.
            sector_map: ticker -> sector mapping.
            corr_matrix: correlation matrix (if None, derived from covariance).
            baseline_corr: historical baseline correlation for spike detection.
            volumes: recent volume per ticker for liquidity check.
            avg_volumes: average volume per ticker.
            regime: current market regime string.

        Returns:
            RiskReport with adjusted weights and diagnostics.
        """
        risk_off = False
        violations = []

        # 1. Drawdown check
        dd_triggered, drawdown = self.check_drawdown(current_value)
        if dd_triggered:
            risk_off = True
            violations.append(f"Drawdown {drawdown:.2%} exceeds limit {self.limits.max_drawdown:.2%}")

        # 2. Tail risk detection
        tail_flag = False
        if recent_returns is not None:
            tail_flag = self.detect_tail_risk(recent_returns)
            if tail_flag:
                risk_off = True
                violations.append("Tail risk detected in recent returns")

        # 2.5. Correlation spike check
        derived_corr = corr_matrix
        if derived_corr is None and covariance is not None:
            stds = np.sqrt(np.diag(covariance))
            stds = np.maximum(stds, 1e-10)
            derived_corr = covariance / np.outer(stds, stds)

        if derived_corr is not None and baseline_corr is not None:
            if self.detect_correlation_spike(derived_corr, baseline_corr):
                violations.append("Correlation spike detected: diversification benefit reduced")

        # 3. Exposure caps
        adjusted, cap_violations = self.apply_exposure_caps(proposed_weights, sector_map)
        violations.extend(cap_violations)

        # 3.5. Liquidity stress check
        if volumes is not None and avg_volumes is not None:
            stressed_tickers = self.check_liquidity_stress(volumes, avg_volumes, tickers)
            for t in stressed_tickers:
                violations.append(f"{t}: liquidity stress (volume below threshold)")

        # 4. Correlation clusters
        clusters = []  # type: List[List[str]]
        if derived_corr is not None:
            clusters = self.cluster_correlations(derived_corr, tickers)

        # 5. Volatility scaling
        adjusted, scaling = self.volatility_scale(adjusted, covariance, tickers)

        # Risk-off mode: reduce all positions
        if risk_off:
            scale = self.limits.risk_off_scale
            adjusted = {t: w * scale for t, w in adjusted.items()}
            # Put remainder in cash (implicit)
            scaling *= scale

        # 5.5. Hedge allocation
        hedge_weight = self.compute_hedge_allocation(regime, risk_off)
        if hedge_weight > 0:
            # Scale down all positions to make room for hedge/cash
            risk_scale = 1.0 - hedge_weight
            adjusted = {t: w * risk_scale for t, w in adjusted.items()}

        # Compute final portfolio vol
        w_arr = np.array([adjusted.get(t, 0.0) for t in tickers])
        final_vol = float(np.sqrt(max(w_arr @ covariance @ w_arr, 0.0)))

        return RiskReport(
            adjusted_weights=adjusted,
            risk_off_mode=risk_off,
            drawdown=drawdown,
            vol=final_vol,
            tail_risk_flag=tail_flag,
            clusters=clusters,
            violations=violations,
            scaling_factor=scaling,
        )
