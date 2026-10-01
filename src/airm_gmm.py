"""SPD-embedding AIRM GMM distance.

This module provides the AIRM distance computation for GMMs via the
symmetric positive-definite embedding as described in the reference algorithm.
All K-valued distribution AIRM distances are routed through the dense path.

It can be used as a library or run as a script (see main()).
"""

from __future__ import annotations

import sys
import time
from typing import Any

import numpy as np
import scipy.linalg

from .gmm_types import GMM, AIRMResult


def component_spd_blocks(g: GMM) -> np.ndarray:
    """Compute the SPD blocks for each component.

    For each component k:
        A_k = Sigma_k + pi_k * mu_k mu_k^T
        X_k = pi_k * mu_k
        B_k = pi_k

    Parameters
    ----------
    g : GMM
        Input GMM.

    Returns
    -------
    np.ndarray
        Shape (K, n+1, n+1) array of SPD blocks.
    """
    K, n = g.k, g.dim
    # We'll compute each block and stack
    blocks = np.empty((K, n + 1, n + 1), dtype=np.float64)
    for k in range(K):
        pi = g.weights[k]
        mu = g.means[k]  # (n,)
        Sigma = g.covariances[k]  # (n, n)
        # A_k = Sigma + pi * (mu mu^T)
        A = Sigma + pi * np.outer(mu, mu)
        # X_k = pi * mu
        X = pi * mu
        # B_k = pi
        B = pi
        # Assemble the block
        block = np.zeros((n + 1, n + 1), dtype=np.float64)
        block[:n, :n] = A
        block[:n, n] = X
        block[n, :n] = X
        block[n, n] = B
        blocks[k] = block
    return blocks


def spd_embedding(g: GMM, allow_large: bool = False) -> np.ndarray:
    """Assemble the dense SPD embedding matrix S = [[A, X], [X^T, B]].

    Parameters
    ----------
    g : GMM
        Input GMM.
    allow_large : bool, optional
        If False, raise ValueError for large embeddings (default: False).

    Returns
    -------
    np.ndarray
        Shape (K*(n+1), K*(n+1)) block-diagonal matrix.

    Raises
    ------
    ValueError
        If the embedding size exceeds 1000 and allow_large is False.
    """
    K, n = g.k, g.dim
    total_size = K * (n + 1)
    if not allow_large and total_size > 1000:
        raise ValueError(
            f"SPD embedding size {total_size} exceeds 1000. "
            "Set allow_large=True to override (not recommended for timing)."
        )
    # Since the embedding is block-diagonal with blocks S_k, we can
    # construct it directly as a block-diagonal matrix.
    blocks = component_spd_blocks(g)  # (K, n+1, n+1)
    # Use scipy.linalg.block_diag to create the block-diagonal matrix
    return scipy.linalg.block_diag(*blocks)


def airm_dense(g1: GMM, g2: GMM, use_logm: bool = True) -> AIRMResult:
    """Compute AIRM distance via the dense embedding (reference path).

    Parameters
    ----------
    g1 : GMM
        First GMM.
    g2 : GMM
        Second GMM.
    use_logm : bool, optional
        If True, use scipy.linalg.logm; if False, use eigendecomposition
        for the matrix logarithm (default: True).

    Returns
    -------
    AIRMResult
        The result containing distance, timing, and method.
    """
    if g1.k != g2.k:
        raise ValueError("GMMs must have the same number of components")
    if g1.dim != g2.dim:
        raise ValueError("GMMs must have the same dimension")

    start_emb = time.perf_counter()
    S1 = spd_embedding(g1, allow_large=True)
    S2 = spd_embedding(g2, allow_large=True)
    t_emb = time.perf_counter() - start_emb

    # Compute S1^{-1/2}
    start_sqrtm = time.perf_counter()
    S1_inv_sqrt = scipy.linalg.fractional_matrix_power(S1, -0.5)
    t_sqrtm = time.perf_counter() - start_sqrtm

    # Compute M = S1^{-1/2} S2 S1^{-1/2}
    start_mf = time.perf_counter()
    M = S1_inv_sqrt @ S2 @ S1_inv_sqrt
    if use_logm:
        # Matrix logarithm
        logM = scipy.linalg.logm(M)
        # Frobenius norm of logM
        distance = np.linalg.norm(logM, "fro")
    else:
        # Use eigendecomposition for symmetric M
        evalsM = np.linalg.eigvalsh(M)
        evalsM = np.maximum(evalsM, 1e-12)  # avoid zeros
        log_evals = np.log(evalsM)
        distance = np.sqrt(np.sum(log_evals**2))
    t_mf = time.perf_counter() - start_mf

    return AIRMResult(
        distance=distance,
        stage_times={"embedding": t_emb, "sqrtm": t_sqrtm, "logm": t_mf},
        method="dense",
        spd_size=S1.shape[0],
    )


def gmm_airm(
    g1: GMM,
    g2: GMM,
    method: str = "auto",
) -> AIRMResult:
    """Dispatcher for AIRM distance computation.

    Parameters
    ----------
    g1 : GMM
        First GMM.
    g2 : GMM
        Second GMM.
    method : str, optional
        One of "auto" or "dense" (default: "auto").
        - "auto": route through the dense path.
        - "dense": force dense path.

    Returns
    -------
    AIRMResult
        The result from the selected method.

    Raises
    ------
    ValueError
        If an unknown method is provided.
    """
    if method == "auto" or method == "dense":
        return airm_dense(g1, g2, use_logm=True)
    else:
        raise ValueError(f"Unknown method: {method}")


def main(argv: list[str] | None = None) -> int:
    """Command-line interface for the AIRM distance.

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
        description="Compute GMM AIRM distance (SPD embedding)"
    )
    parser.add_argument(
        "--k", type=int, default=1000, help="Number of components (default: 1000)"
    )
    parser.add_argument(
        "--method",
        choices=["auto", "dense"],
        default="auto",
        help="Method to use (default: auto)",
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
        result = gmm_airm(gmm, gmm2, method=args.method)
        print(f"Run {i+1}:")
        print(f"  Distance: {result.distance}")
        print(f"  Method: {result.method}")
        print(f"  SPD size: {result.spd_size}")
        for stage, t in result.stage_times.items():
            print(f"  {stage} time: {t:.4f} s")
        if i < args.repeat - 1:
            print()  # blank line between runs

    # Non-zero exit if validation fails (but we assume the GMM is valid)
    return 0


if __name__ == "__main__":
    sys.exit(main())
