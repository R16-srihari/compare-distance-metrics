"""Beta(2,5) product GMM construction."""

from __future__ import annotations

import numpy as np
from scipy.stats import beta

from .gmm_types import BETA_A, BETA_B, CELL_FRACTION, GMM, SUPPORT


def beta_pdf(x: np.ndarray, a: float = BETA_A, b: float = BETA_B) -> np.ndarray:
    """Wrapper for scipy.stats.beta.pdf returning float64.

    Parameters
    ----------
    x : np.ndarray
        Input values (can be scalar or array).
    a : float, optional
        Beta shape parameter a, by default BETA_A.
    b : float, optional
        Beta shape parameter b, by default BETA_B.

    Returns
    -------
    np.ndarray
        Beta PDF values as float64 array of same shape as x.
    """
    return np.asarray(beta.pdf(x, a, b), dtype=np.float64)


def factorize_components(k: int) -> tuple[int, int, int]:
    """Factorize K into (m1, m2, m3) with m1*m2*m3 = K and m1 <= m2 <= m3,
    minimizing (m3 - m1, m3).

    Parameters
    ----------
    k : int
        Number of components (must be >= 1).

    Returns
    -------
    tuple[int, int, int]
        The factor triple (m1, m2, m3).

    Raises
    ------
    ValueError
        If k < 1.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    best = None
    best_key = None
    # Enumerate divisors
    for m1 in range(1, int(k ** (1/3)) + 2):
        if k % m1 != 0:
            continue
        k2 = k // m1
        for m2 in range(m1, int(k2**0.5) + 2):
            if k2 % m2 != 0:
                continue
            m3 = k2 // m2
            if m3 < m2:
                continue
            # Now m1 <= m2 <= m3
            key = (m3 - m1, m3)  # minimize (m3-m1, then m3)
            if best_key is None or key < best_key:
                best_key = key
                best = (m1, m2, m3)
    if best is None:
        # Fallback (should not happen for k>=1)
        best = (1, 1, k)
    return best


def axis_grid(m: int, cell_fraction: float = CELL_FRACTION) -> tuple[np.ndarray, float]:
    """Generate midpoints and half-width for m cells on [0,1].

    Parameters
    ----------
    m : int
        Number of cells (>=1).
    cell_fraction : float, optional
        Fraction of cell width for sigma, by default CELL_FRACTION.

    Returns
    -------
    midpoints : np.ndarray
        Shape (m,), cell midpoints.
    sigma : float
        Half-width = cell_fraction / m.
    """
    if m < 1:
        raise ValueError("m must be >= 1")
    if cell_fraction <= 0:
        raise ValueError("cell_fraction must be > 0")
    midpoints = (np.arange(m) + 0.5) / m
    sigma = cell_fraction / m
    return midpoints.astype(np.float64), np.float64(sigma)


def build_beta_product_gmm(
    k: int = 1000,
    a: float = BETA_A,
    b: float = BETA_B,
    cell_fraction: float = CELL_FRACTION,
    min_weight: float = 1e-300,
) -> GMM:
    """Construct a 3D product-Beta GMM with exactly K components.

    Parameters
    ----------
    k : int, optional
        Number of components, by default 1000.
    a : float, optional
        Beta shape parameter a, by default BETA_A.
    b : float, optional
        Beta shape parameter b, by default BETA_B.
    cell_fraction : float, optional
        Fraction of cell width for sigma, by default CELL_FRACTION.
    min_weight : float, optional
        Floor for weights before renormalization, by default 1e-300.

    Returns
    -------
    GMM
        The constructed GMM instance.

    Raises
    ------
    ValueError
        If k <= 0 or cell_fraction <= 0.
    """
    if k <= 0:
        raise ValueError("k must be > 0")
    if cell_fraction <= 0:
        raise ValueError("cell_fraction must be > 0")

    # Factorize k
    m1, m2, m3 = factorize_components(k)

    # Build axes
    xs, sx = axis_grid(m1, cell_fraction)
    ys, sy = axis_grid(m2, cell_fraction)
    zs, sz = axis_grid(m3, cell_fraction)

    # Cartesian product of midpoints
    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
    means = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)  # (K, 3)

    # Weights: product of marginal Beta PDFs
    wx = beta_pdf(xs, a, b)
    wy = beta_pdf(ys, a, b)
    wz = beta_pdf(zs, a, b)
    # Outer product
    weights = wx[:, None, None] * wy[None, :, None] * wz[None, None, :]
    weights = weights.ravel()  # (K,)
    # Floor and renormalize
    weights = np.maximum(weights, min_weight)
    weights /= weights.sum()

    # Covariances: diagonal with sigma_d^2
    variances = np.array([sx**2, sy**2, sz**2], dtype=np.float64)  # (3,)
    covariances = np.zeros((k, 3, 3), dtype=np.float64)
    for i in range(3):
        covariances[:, i, i] = variances[i]
    # Off-diagonals remain zero

    gmm = GMM(weights=weights, means=means, covariances=covariances)
    gmm.validate()
    return gmm


def build_beta_product_gmm_1d(
    k: int = 1000,
    a: float = BETA_A,
    b: float = BETA_B,
    cell_fraction: float = CELL_FRACTION,
    min_weight: float = 1e-300,
) -> GMM:
    """Convenience for n=1: Beta(2,5) quantized into k cells on [0,1].

    Parameters
    ----------
    k : int, optional
        Number of components, by default 1000.
    a : float, optional
        Beta shape parameter a, by default BETA_A.
    b : float, optional
        Beta shape parameter b, by default BETA_B.
    cell_fraction : float, optional
        Fraction of cell width for sigma, by default CELL_FRACTION.
    min_weight : float, optional
        Floor for weights before renormalization, by default 1e-300.

    Returns
    -------
    GMM
        The constructed 1D GMM instance.
    """
    if k <= 0:
        raise ValueError("k must be > 0")
    if cell_fraction <= 0:
        raise ValueError("cell_fraction must be > 0")

    # For 1D, we still use factorize_components but we only need one dimension.
    # We'll set m1 = k, m2 = 1, m3 = 1 to get k cells in one dimension.
    # However, the product-Beta GMM in 1D is just a linear grid.
    # We can reuse the same logic by setting m1=k, m2=1, m3=1.
    m1, m2, m3 = factorize_components(k)
    # But we want exactly k cells in 1D, so we ignore m2 and m3 and set them to 1.
    # Actually, factorize_components(k) for 1D should return (k,1,1) because
    # k = k*1*1 and that minimizes (m3-m1, m3) -> (1-k, k) which is not minimal?
    # Let's just compute the grid directly for 1D.
    xs, sx = axis_grid(k, cell_fraction)
    means = xs.reshape(-1, 1)  # (K, 1)
    weights = beta_pdf(xs, a, b)
    weights = np.maximum(weights, min_weight)
    weights /= weights.sum()
    variances = np.array([sx**2], dtype=np.float64)
    covariances = np.zeros((k, 1, 1), dtype=np.float64)
    covariances[:, 0, 0] = variances[0]
    gmm = GMM(weights=weights, means=means, covariances=covariances)
    gmm.validate()
    return gmm


def shifted_gmm(gmm: GMM, delta: np.ndarray) -> GMM:
    """Return a copy of gmm with means shifted by delta.

    Parameters
    ----------
    gmm : GMM
        The GMM to shift.
    delta : np.ndarray
        Shape (n,), shift to add to means.

    Returns
    -------
    GMM
        New GMM with shifted means, same weights and covariances.
    """
    if delta.shape != (gmm.dim,):
        raise ValueError(f"delta must be shape ({gmm.dim},)")
    shifted_means = gmm.means + delta
    return GMM(
        weights=gmm.weights.copy(),
        means=shifted_means,
        covariances=gmm.covariances.copy(),
    )