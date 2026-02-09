"""
Hierarchical Risk Parity (HRP) Optimizer
------------------------------------------
Implements the HRP algorithm by Marcos Lopez de Prado:
  1. Hierarchical clustering of asset correlation matrix
  2. Quasi-diagonalization to reorder assets
  3. Recursive bisection for inverse-variance weight allocation

HRP avoids matrix inversion (unlike Markowitz), making it robust to
estimation error in covariance matrices.

Reference:
  Lopez de Prado, M. (2016). Building Diversified Portfolios that
  Outperform Out-of-Sample. Journal of Portfolio Management.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
import logging

logger = logging.getLogger(__name__)


class HRPOptimizer:
    """Hierarchical Risk Parity portfolio optimizer.

    Usage:
        hrp = HRPOptimizer()
        weights = hrp.optimize(tickers, covariance, correlation)
    """

    def __init__(self, linkage_method: str = "single"):
        """
        Args:
            linkage_method: hierarchical clustering method.
                Options: 'single', 'complete', 'average', 'ward'.
        """
        self.linkage_method = linkage_method

    def _correlation_distance(self, corr: np.ndarray) -> np.ndarray:
        """Convert correlation matrix to distance matrix.

        d(i,j) = sqrt(0.5 * (1 - corr(i,j)))
        """
        # Clip correlation to [-1, 1] to avoid NaN from floating point
        corr_clipped = np.clip(corr, -1.0, 1.0)
        dist = np.sqrt(np.maximum(0.5 * (1.0 - corr_clipped), 0.0))
        np.fill_diagonal(dist, 0.0)
        return dist

    def _cluster_assets(self, corr: np.ndarray) -> np.ndarray:
        """Hierarchical clustering on correlation distance.

        Args:
            corr: correlation matrix (n, n).

        Returns:
            Linkage matrix from scipy.
        """
        dist = self._correlation_distance(corr)
        # Convert to condensed distance matrix
        condensed = squareform(dist, checks=False)
        return linkage(condensed, method=self.linkage_method)

    @staticmethod
    def _quasi_diag(link: np.ndarray) -> List[int]:
        """Quasi-diagonalization: reorder assets so similar ones are adjacent.

        Recursively builds a sorted list of original leaf indices from the
        linkage tree.

        Args:
            link: linkage matrix from scipy.

        Returns:
            Sorted list of asset indices.
        """
        return leaves_list(link).tolist()

    @staticmethod
    def _recursive_bisection(
        cov: np.ndarray,
        sorted_idx: List[int],
    ) -> np.ndarray:
        """Recursive bisection to allocate inverse-variance weights.

        Splits the sorted asset list in half, allocates weight proportional
        to inverse cluster variance, then recurses on each half.

        Args:
            cov: covariance matrix (n, n).
            sorted_idx: quasi-diagonalized asset indices.

        Returns:
            Weight array (n,) summing to 1.
        """
        n = cov.shape[0]
        weights = np.ones(n)

        # Stack of (index_subset,) to process
        clusters = [sorted_idx]

        while clusters:
            next_clusters = []
            for cluster in clusters:
                if len(cluster) <= 1:
                    continue

                mid = len(cluster) // 2
                left = cluster[:mid]
                right = cluster[mid:]

                # Cluster variance = w' * Cov * w with equal weights
                def cluster_var(indices):
                    sub_cov = cov[np.ix_(indices, indices)]
                    n_sub = len(indices)
                    w_eq = np.ones(n_sub) / n_sub
                    return float(w_eq @ sub_cov @ w_eq)

                v_left = cluster_var(left)
                v_right = cluster_var(right)

                # Inverse variance allocation
                total_inv = 1.0 / max(v_left, 1e-12) + 1.0 / max(v_right, 1e-12)
                alpha_left = (1.0 / max(v_left, 1e-12)) / total_inv
                alpha_right = 1.0 - alpha_left

                # Scale weights
                for idx in left:
                    weights[idx] *= alpha_left
                for idx in right:
                    weights[idx] *= alpha_right

                if len(left) > 1:
                    next_clusters.append(left)
                if len(right) > 1:
                    next_clusters.append(right)

            clusters = next_clusters

        # Normalize
        total = weights.sum()
        if total > 0:
            weights = weights / total

        return weights

    def optimize(
        self,
        tickers: List[str],
        covariance: np.ndarray,
        correlation: np.ndarray,
    ) -> Dict[str, float]:
        """Run full HRP optimization.

        Args:
            tickers: list of ticker symbols.
            covariance: covariance matrix (n, n).
            correlation: correlation matrix (n, n).

        Returns:
            Dict of ticker -> weight, summing to 1.0.
        """
        n = len(tickers)
        if n == 0:
            return {}
        if n == 1:
            return {tickers[0]: 1.0}

        # Step 1: Cluster
        link = self._cluster_assets(correlation)

        # Step 2: Quasi-diagonalize
        sorted_idx = self._quasi_diag(link)

        # Step 3: Recursive bisection
        weights = self._recursive_bisection(covariance, sorted_idx)

        return {t: float(weights[i]) for i, t in enumerate(tickers)}
