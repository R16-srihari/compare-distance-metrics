"""LP-based GMM Hellinger distance (Mercurio & Singla).

This module provides the Hellinger distance computation for GMMs as described in
the Mercurio & Singla paper, using a cost matrix of Gaussian Hellinger distances
and solving the optimal transport problem via Earth Mover's Distance (EMD).

It can be used as a library or run as a script (see main()).
"""

from __future__ import annotations

import sys
import time
from typing import Any

import numpy as np
import ot  # POT

from .gmm_types import (
    DEFAULT_CHUNK,
    GMM,
    HellingerResult,
)


def gaussian_hellinger_block(
    mu_p: np.ndarray,
    cov_p: np.ndarray,
    mu_q: np.ndarray,
    cov_q: np.ndarray,
) -> np.ndarray:
    """Vectorized Gaussian Hellinger distance matrix.

    Computes D[i,j] = sqrt(max(0, 1 - B_ij)) where B_ij is the Bhattacharyya
    coefficient between two Gaussians.

    Parameters
    ----------
    mu_p : np.ndarray
        Shape (C, n), means of first set of Gaussians.
    cov_p : np.ndarray
        Shape (C, n, n), covariances of first set. A single (n, n) matrix
        is also accepted and treated as a set of one Gaussian.
    mu_q : np.ndarray
        Shape (M, n), means of second set.
    cov_q : np.ndarray
        Shape (M, n, n), covariances of second set. A single (n, n) matrix
        is also accepted and treated as a set of one Gaussian.

    Returns
    -------
    np.ndarray
        Shape (C, M), distance matrix.
    """
    # Normalize inputs: accept a single mean as 1D (n,) and a single
    # covariance as 2D (n, n), promoting them to batched shapes.
    if mu_p.ndim == 1:
        mu_p = mu_p[np.newaxis, :]
    if mu_q.ndim == 1:
        mu_q = mu_q[np.newaxis, :]
    if cov_p.ndim == 2:
        cov_p = cov_p[np.newaxis, :, :]
    if cov_q.ndim == 2:
        cov_q = cov_q[np.newaxis, :, :]

    C = mu_p.shape[0]
    M = mu_q.shape[0]
    n = mu_p.shape[1]

    # We'll compute in chunks to avoid too large intermediate arrays if needed,
    # but for simplicity we do the whole block at once (the caller already chunks).
    # Expand dimensions for broadcasting: (C,1,n) and (1,M,n)
    mu_p_exp = mu_p[:, np.newaxis, :]  # (C,1,n)
    mu_q_exp = mu_q[np.newaxis, :, :]  # (1,M,n)
    dmu = mu_p_exp - mu_q_exp  # (C,M,n)

    # Average covariance: (C,1,n,n) and (1,M,n,n) -> (C,M,n,n)
    cov_p_exp = cov_p[:, np.newaxis, :, :]  # (C,1,n,n)
    cov_q_exp = cov_q[np.newaxis, :, :, :]  # (1,M,n,n)
    Sigma = 0.5 * (cov_p_exp + cov_q_exp)  # (C,M,n,n)

    # Compute logdet of Sigma and the individual covariances
    # We use slogdet for stability and to get the sign.
    # For each pair we need logdet(Sigma), logdet(cov_p), logdet(cov_q)
    # We can compute the logdet of each covariance matrix once and then expand.
    logdet_p = np.linalg.slogdet(cov_p)[1]  # (C,)
    logdet_q = np.linalg.slogdet(cov_q)[1]  # (M,)
    logdet_Sigma = np.linalg.slogdet(Sigma)[1]  # (C,M)

    # Compute the quadratic form: dmu^T Sigma^{-1} dmu
    # We solve Sigma * x = dmu for each pair, then compute dmu^T * x.
    # Reshape dmu to (C*M, n) and Sigma to (C*M, n, n)
    dmu_flat = dmu.reshape(-1, n)  # (C*M, n)
    Sigma_flat = Sigma.reshape(-1, n, n)  # (C*M, n, n)
    # Solve for each pair
    x_flat = np.linalg.solve(Sigma_flat, dmu_flat[..., np.newaxis]).squeeze(-1)  # (C*M, n)
    quad_flat = np.sum(dmu_flat * x_flat, axis=1)  # (C*M,)
    quad = quad_flat.reshape(C, M)  # (C, M)

    # Now compute log B_ij
    # log B_ij = 0.25*logdet(cov_p[i]) + 0.25*logdet(cov_q[j])
    #          - 0.5*logdet(Sigma_ij) - 0.5 * (dmu^T Sigma^{-1} dmu)
    logB = (
        0.25 * logdet_p[:, np.newaxis]
        + 0.25 * logdet_q[np.newaxis, :]
        - 0.5 * logdet_Sigma
        - 0.5 * quad
    )
    # B_ij = exp(logB), but we clip to [0,1] for numerical stability
    B = np.exp(logB)
    B = np.clip(B, 0.0, 1.0)
    # Hellinger distance: sqrt(1 - B)
    D = np.sqrt(1.0 - B)
    return D


def component_hellinger_matrix(
    g1: GMM,
    g2: GMM,
    chunk: int = DEFAULT_CHUNK,
    return_stage_time: bool = False,
) -> np.ndarray | tuple[np.ndarray, float]:
    """Compute the Hellinger cost matrix in chunks.

    Parameters
    ----------
    g1 : GMM
        First GMM.
    g2 : GMM
        Second GMM.
    chunk : int, optional
        Number of rows to process at a time, by default DEFAULT_CHUNK.
    return_stage_time : bool, optional
        If True, return (matrix, elapsed_time) for the cost matrix computation.

    Returns
    -------
    np.ndarray or tuple[np.ndarray, float]
        The cost matrix (shape (g1.k, g2.k)) or (matrix, time) if
        return_stage_time is True.
    """
    if g1.dim != g2.dim:
        raise ValueError("GMMs must have the same dimension")

    K1, K2 = g1.k, g2.k
    n = g1.dim
    D = np.empty((K1, K2), dtype=np.float64)

    start = time.perf_counter()
    for i1 in range(0, K1, chunk):
        i2 = min(i1 + chunk, K1)
        # Chunk of g1
        mu_p = g1.means[i1:i2]
        cov_p = g1.covariances[i1:i2]
        # Compute block against all of g2
        block = gaussian_hellinger_block(mu_p, cov_p, g2.means, g2.covariances)
        D[i1:i2, :] = block
    elapsed = time.perf_counter() - start

    if return_stage_time:
        return D, elapsed
    return D


def transport_cost(
    cost: np.ndarray,
    weights_p: np.ndarray,
    weights_q: np.ndarray,
    solver: str = "pot",
) -> tuple[float, np.ndarray]:
    """Solve the optimal transport problem.

    Parameters
    ----------
    cost : np.ndarray
        Shape (K1, K2), cost matrix.
    weights_p : np.ndarray
        Shape (K1,), weights of first distribution.
    weights_q : np.ndarray
        Shape (K2,), weights of second distribution.
    solver : str, optional
        Either 'pot' (uses POT's emd) or 'scipy' (uses scipy.optimize.linprog).

    Returns
    -------
    tuple[float, np.ndarray]
        (total_cost, plan) where total_cost is the optimal transport cost
        and plan is the optimal coupling matrix.

    Raises
    ------
    ImportError
        If solver is 'pot' and POT is not installed.
    ValueError
        If an unknown solver is provided.
    """
    if solver == "pot":
        try:
            import ot
        except ImportError as e:
            raise ImportError(
                "POT (Python Optimal Transport) is not installed. "
                "Install it with: pip install POT"
            ) from e
        # POT's emd returns the optimal plan
        plan = ot.emd(weights_p, weights_q, cost)
        total_cost = np.sum(plan * cost)
        return total_cost, plan
    elif solver == "scipy":
        from scipy.optimize import linprog
        import scipy.sparse as sparse

        K1, K2 = cost.shape
        # Flatten the cost matrix in row-major order
        c = cost.ravel()
        # Equality constraints: row sums = weights_p, column sums = weights_q
        A_eq = []
        b_eq = []
        # Row sums
        for i in range(K1):
            row = np.zeros(K1 * K2)
            row[i * K2 : (i + 1) * K2] = 1
            A_eq.append(row)
            b_eq.append(weights_p[i])
        # Column sums
        for j in range(K2):
            col = np.zeros(K1 * K2)
            col[j::K2] = 1  # every K2-th element starting at j
            A_eq.append(col)
            b_eq.append(weights_q[j])
        A_eq = np.array(A_eq)
        b_eq = np.array(b_eq)
        # Bounds: all variables >= 0
        bounds = [(0, None)] * (K1 * K2)
        # Solve
        res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
        if not res.success:
            raise RuntimeError(f"LP solver failed: {res.message}")
        plan = res.x.reshape(K1, K2)
        total_cost = np.sum(plan * cost)
        return total_cost, plan
    else:
        raise ValueError(f"Unknown solver: {solver}")


def gmm_hellinger(
    g1: GMM,
    g2: GMM,
    chunk: int = DEFAULT_CHUNK,
    solver: str = "pot",
) -> HellingerResult:
    """Compute the Hellinger distance between two GMMs.

    Parameters
    ----------
    g1 : GMM
        First GMM.
    g2 : GMM
        Second GMM.
    chunk : int, optional
        Chunk size for cost matrix computation, by default DEFAULT_CHUNK.
    solver : str, optional
        Solver for the transport LP, by default "pot".

    Returns
    -------
    HellingerResult
        The result containing distance, timing, and diagnostics.
    """
    # Time the cost matrix computation
    cost, t_cost = component_hellinger_matrix(
        g1, g2, chunk=chunk, return_stage_time=True
    )
    # Time the transport LP
    t0 = time.perf_counter()
    distance, plan = transport_cost(cost, g1.weights, g2.weights, solver=solver)
    t_lp = time.perf_counter() - t0

    # Compute the marginal error of the plan
    if plan is not None:
        row_sums = plan.sum(axis=1)
        col_sums = plan.sum(axis=0)
        row_err = np.max(np.abs(row_sums - g1.weights))
        col_err = np.max(np.abs(col_sums - g2.weights))
        transport_sum_marginal_error = max(row_err, col_err)
    else:
        transport_sum_marginal_error = float("inf")

    return HellingerResult(
        distance=distance,
        stage_times={"cost_matrix": t_cost, "transport_lp": t_lp},
        cost_matrix_shape=cost.shape,
        transport_sum_marginal_error=transport_sum_marginal_error,
        plan=plan,
    )


def main(argv: list[str] | None = None) -> int:
    """Command-line interface for the Hellinger distance.

    Parameters
    ----------
    argv : list[str] | None
        Command-line arguments. If None, uses sys.argv[1:].

    Returns
    -------
    int
        Exit code (0 for success, non-zero for failure).
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute GMM Hellinger distance (Mercurio & Singla)"
    )
    parser.add_argument(
        "--k", type=int, default=1000, help="Number of components (default: 1000)"
    )
    parser.add_argument(
        "--chunk",
        type=int,
        default=DEFAULT_CHUNK,
        help=f"Chunk size for cost matrix (default: {DEFAULT_CHUNK})",
    )
    parser.add_argument(
        "--solver",
        choices=["pot", "scipy"],
        default="pot",
        help="Solver for the transport LP (default: pot)",
    )
    parser.add_argument(
        "--gmm",
        choices=["b", "shifted"],
        default="b",
        help="Which GMM to use for the second argument: 'b' for Beta product GMM, "
        "'shifted' for a shifted copy (default: b)",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Number of repetitions (default: 1)",
    )
    args = parser.parse_args(argv)

    # Build the GMM
    gmm = build_beta_product_gmm(k=args.k)
    if args.gmm == "b":
        gmm2 = gmm
    else:  # shifted
        # Shift by 0.1 in each dimension
        delta = np.full(gmm.dim, 0.1)
        gmm2 = shifted_gmm(gmm, delta)

    # Run the distance computation the specified number of times
    for i in range(args.repeat):
        result = gmm_hellinger(gmm, gmm2, chunk=args.chunk, solver=args.solver)
        print(f"Run {i+1}:")
        print(f"  Distance: {result.distance}")
        print(f"  Cost matrix time: {result.stage_times['cost_matrix']:.4f} s")
        print(f"  Transport LP time: {result.stage_times['transport_lp']:.4f} s")
        print(f"  Marginal error: {result.transport_sum_marginal_error}")
        if i < args.repeat - 1:
            print()  # blank line between runs

    # Non-zero exit if validation fails (but we assume the GMM is valid)
    return 0


if __name__ == "__main__":
    sys.exit(main())