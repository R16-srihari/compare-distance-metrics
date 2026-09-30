"""Data classes and constants for GMM benchmark."""

from __future__ import annotations

import dataclasses
import numpy as np

# Module-level constants
BETA_A = 2.0
BETA_B = 5.0
SUPPORT = (0.0, 1.0)
DEFAULT_CHUNK = 64
CELL_FRACTION = 0.5
SUPPORTED_K = (100, 200, 500, 1000)
REPEATS = 5


@dataclasses.dataclass(frozen=True)
class GMM:
    """Frozen GMM representation.

    Attributes
    ----------
    weights : np.ndarray
        Shape (K,), mixture weights, sum to 1, all > 0.
    means : np.ndarray
        Shape (K, n), component means.
    covariances : np.ndarray
        Shape (K, n, n), component covariances (symmetric positive-definite).
    """

    weights: np.ndarray
    means: np.ndarray
    covariances: np.ndarray

    def __post_init__(self) -> None:
        # Cheap structural checks; full validation via validate()
        if self.weights.ndim != 1:
            raise ValueError("weights must be 1D")
        if self.means.ndim != 2:
            raise ValueError("means must be 2D")
        if self.covariances.ndim != 3:
            raise ValueError("covariances must be 3D")
        K = self.weights.shape[0]
        if self.means.shape[0] != K or self.covariances.shape[0] != K:
            raise ValueError("leading dimension mismatch")
        n = self.means.shape[1]
        if self.covariances.shape[1:] != (n, n):
            raise ValueError("covariance shape mismatch")

    @property
    def k(self) -> int:
        return int(self.weights.shape[0])

    @property
    def dim(self) -> int:
        return int(self.means.shape[1])

    def validate(self, tol: float = 1e-12) -> None:
        """Validate the GMM instance.

        Parameters
        ----------
        tol : float, optional
            Tolerance for numerical checks, by default 1e-12.

        Raises
        ------
        ValueError
            If any validation fails.
        """
        # Non-finite values
        if not (np.all(np.isfinite(self.weights)) and
                np.all(np.isfinite(self.means)) and
                np.all(np.isfinite(self.covariances))):
            raise ValueError("non-finite values in GMM")

        # Weights positive and sum to 1
        if np.any(self.weights <= 0):
            raise ValueError("non-positive weight encountered")
        if abs(self.weights.sum() - 1.0) > tol:
            raise ValueError(f"weights sum to {self.weights.sum()}, expected 1.0")

        # Check symmetry and positive-definiteness of covariances
        for k in range(self.k):
            cov = self.covariances[k]
            # Symmetry
            if not np.allclose(cov, cov.T, atol=tol, rtol=0):
                raise ValueError(f"covariance matrix {k} is not symmetric")
            # Positive-definiteness via eigenvalues
            evals = np.linalg.eigvalsh(cov)
            if np.any(evals <= 0):
                raise ValueError(f"covariance matrix {k} is not positive-definite")


@dataclasses.dataclass
class TimingRecord:
    """Record of a timed distance computation.

    Attributes
    ----------
    k : int
        Number of components.
    metric : str
        Metric name ('hellinger' or 'airm').
    run_index : int
        Index of the repetition (starting at 1).
    total_s : float
        Total elapsed time in seconds.
    stage_times : dict[str, float]
        Time spent in each stage (e.g., {'cost_matrix': 0.1, 'transport_lp': 0.2}).
    distance : float
        Computed distance value.
    """

    k: int
    metric: str
    run_index: int
    total_s: float
    stage_times: dict[str, float]
    distance: float

    def to_row(self, stage_order: list[str]) -> dict:
        """Flatten to a dictionary for CSV writing.

        Parameters
        ----------
        stage_order : list[str]
            Ordered list of stage names (e.g., ['cost_matrix', 'transport_lp']).

        Returns
        -------
        dict
            Flattened record with keys: k, metric, run_index, total_s,
            stage_1_name, stage_1_s, stage_2_name, stage_2_s, distance.
        """
        row = {
            "k": self.k,
            "metric": self.metric,
            "run_index": self.run_index,
            "total_s": self.total_s,
            "distance": self.distance,
        }
        # Ensure exactly two stages for backward compatibility with the plan
        # (but we can handle more if needed by ignoring extra or using empty)
        stage_names = stage_order[:2] + [""] * max(0, 2 - len(stage_order))
        stage_values = [self.stage_times.get(name, 0.0) for name in stage_order[:2]]
        # Pad to two stages
        while len(stage_names) < 2:
            stage_names.append("")
            stage_values.append(0.0)
        row.update({
            "stage_1_name": stage_names[0],
            "stage_1_s": stage_values[0],
            "stage_2_name": stage_names[1],
            "stage_2_s": stage_values[1],
        })
        return row


@dataclasses.dataclass
class HellingerResult:
    """Result of Hellinger distance computation.

    Attributes
    ----------
    distance : float
        The Hellinger distance.
    stage_times : dict[str, float]
        Timing for each stage.
    cost_matrix_shape : tuple[int, int]
        Shape of the cost matrix (K1, K2).
    transport_sum_marginal_error : float
        Maximum absolute deviation of the optimal plan's row/column sums
        from the prescribed weights.
    plan : np.ndarray | None
        The optimal coupling matrix (retained for validation).
    """

    distance: float
    stage_times: dict[str, float]
    cost_matrix_shape: tuple[int, int]
    transport_sum_marginal_error: float
    plan: np.ndarray | None = None


@dataclasses.dataclass
class AIRMResult:
    """Result of AIRM distance computation.

    Attributes
    ----------
    distance : float
        The AIRM distance.
    stage_times : dict[str, float]
        Timing for each stage.
    method : str
        Either 'block' (fast batched) or 'dense' (reference).
    spd_size : int
        The size of the SPD matrix (K*(n+1)).
    """

    distance: float
    stage_times: dict[str, float]
    method: str
    spd_size: int