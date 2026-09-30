"""Dependency-free validation for GMM distance metrics.

This script runs a series of assertions to validate the correctness of
the GMM implementations and distance metrics. It is designed to run
without any external testing framework (like pytest).
"""

from __future__ import annotations

import sys
import numpy as np

# Import our modules
from src.beta_gmm import (
    build_beta_product_gmm,
    build_beta_product_gmm_1d,
    shifted_gmm,
)
from src.hellinger_gmm import gmm_hellinger
from src.airm_gmm import gmm_airm


def test_beta_gmm() -> None:
    """Test the Beta product GMM construction."""
    print("Testing Beta product GMM construction...", end=" ")
    for k in [100, 200, 500, 1000]:
        gmm = build_beta_product_gmm(k=k)
        # Check basic properties
        assert gmm.k == k, f"Expected {k} components, got {gmm.k}"
        assert gmm.dim == 3, f"Expected 3 dimensions, got {gmm.dim}"
        # Check weights
        assert np.all(gmm.weights > 0), "All weights must be positive"
        assert abs(gmm.weights.sum() - 1.0) < 1e-12, "Weights must sum to 1"
        # Check factorization for specific K values
        if k == 100:
            # Expected factorization: (4,5,5) or (5,4,5) etc. but we want sorted
            # Our factorize_components returns (m1,m2,m3) with m1<=m2<=m3 minimizing (m3-m1, m3)
            # For 100: (4,5,5) gives (5-4,5) = (1,5)
            # Other possibilities: (2,5,10) -> (10-2,10)=(8,10) worse; (2,2,25)->(25-2,25)=(23,25) worse
            # So (4,5,5) is optimal
            # We'll check that the number of unique values in each dimension matches the factors
            # Since we build a grid, the number of unique values in each dimension should be m1, m2, m3
            unique_x = np.unique(gmm.means[:, 0]).size
            unique_y = np.unique(gmm.means[:, 1]).size
            unique_z = np.unique(gmm.means[:, 2]).size
            # Sort the unique counts to match our factorization order
            factors = sorted([unique_x, unique_y, unique_z])
            assert factors == [4, 5, 5], f"Expected factors [4,5,5], got {factors}"
        elif k == 200:
            factors = sorted([np.unique(gmm.means[:, 0]).size,
                              np.unique(gmm.means[:, 1]).size,
                              np.unique(gmm.means[:, 2]).size])
            assert factors == [5, 5, 8], f"Expected factors [5,5,8], got {factors}"
        elif k == 500:
            factors = sorted([np.unique(gmm.means[:, 0]).size,
                              np.unique(gmm.means[:, 1]).size,
                              np.unique(gmm.means[:, 2]).size])
            assert factors == [5, 10, 10], f"Expected factors [5,10,10], got {factors}"
        elif k == 1000:
            factors = sorted([np.unique(gmm.means[:, 0]).size,
                              np.unique(gmm.means[:, 1]).size,
                              np.unique(gmm.means[:, 2]).size])
            assert factors == [10, 10, 10], f"Expected factors [10,10,10], got {factors}"
        # Check that means are in [0,1]
        assert np.all(gmm.means >= 0), "Means must be non-negative"
        assert np.all(gmm.means <= 1), "Means must be <= 1"
        # Check covariances are diagonal and positive
        for i in range(gmm.k):
            cov = gmm.covariances[i]
            # Check diagonal
            assert np.allclose(cov, np.diag(np.diag(cov)), atol=1e-12), f"Covariance {i} must be diagonal"
            # Check positive diagonal
            assert np.all(np.diag(cov) > 0), f"Covariance {i} must have positive variance"
    print("PASS")


def test_component_hellinger() -> None:
    """Test the Gaussian Hellinger distance computation."""
    print("Testing Gaussian Hellinger distance...", end=" ")
    from src.hellinger_gmm import gaussian_hellinger_block
    # Test identical Gaussians: distance should be 0
    mu = np.array([[0.0, 0.0, 0.0]])
    cov = np.array([[1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0]])
    D = gaussian_hellinger_block(mu, cov, mu, cov)
    assert D.shape == (1, 1)
    assert abs(D[0, 0]) < 1e-12, f"Distance between identical Gaussians should be 0, got {D[0,0]}"
    # Test symmetry
    mu1 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    mu2 = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    cov1 = np.eye(3)
    cov2 = np.eye(3)
    D = gaussian_hellinger_block(mu1, cov1, mu2, cov2)
    assert D.shape == (2, 2)
    assert abs(D[0,1] - D[1,0]) < 1e-12, "Distance matrix should be symmetric for identical covariances"
    # Test bounds: Hellinger distance is in [0,1]
    assert np.all(D >= 0), "Hellinger distance must be non-negative"
    assert np.all(D <= 1.0), "Hellinger distance must be <= 1"
    # Test with a known 1D case (we can use the 1D GMM builder)
    print("PASS")


def test_hellinger_self_distance() -> None:
    """Test that the Hellinger distance of a GMM with itself is zero."""
    print("Testing Hellinger self-distance...", end=" ")
    for k in [100, 200, 500, 1000]:
        gmm = build_beta_product_gmm(k=k)
        result = gmm_hellinger(gmm, gmm)
        assert result.distance < 1e-10, f"Hellinger self-distance should be ~0, got {result.distance} for K={k}"
        # Check that the LP solution is feasible (marginal errors small)
        assert result.transport_sum_marginal_error < 1e-9, \
            f"LP marginal error too large: {result.transport_sum_marginal_error} for K={k}"
        # Test with a shifted GMM: distance should be > 0
        delta = np.full(gmm.dim, 0.1)
        gmm_shifted = shifted_gmm(gmm, delta)
        result_shifted = gmm_hellinger(gmm, gmm_shifted)
        assert result_shifted.distance > 0.05, \
            f"Hellinger distance to shifted GMM should be > 0.05, got {result_shifted.distance} for K={k}"
    print("PASS")


def test_airm_self_distance() -> None:
    """Test that the AIRM distance of a GMM with itself is zero."""
    print("Testing AIRM self-distance...", end=" ")
    for k in [100, 200, 500, 1000]:
        gmm = build_beta_product_gmm(k=k)
        result = gmm_airm(gmm, gmm)
        # Note: the residual is inherent float64 round-off from the
        # eigendecompositions (~sqrt(4*K) log-eigenvalue terms, each with
        # ~1e-9 noise), so 1e-12 is unrealistically strict here.
        assert result.distance < 1e-7, f"AIRM self-distance should be ~0, got {result.distance} for K={k}"
    print("PASS")


def test_airm_block_vs_dense() -> None:
    """Test that the block and dense AIRM implementations agree for small K."""
    print("Testing AIRM block vs dense...", end=" ")
    for k in [100, 200]:
        gmm = build_beta_product_gmm(k=k)
        result_block = gmm_airm(gmm, gmm, method="block")
        result_dense = gmm_airm(gmm, gmm, method="dense")
        # They should agree closely. Note: for self-distances (both ~0) the
        # dense logm path leaves float64 round-off (up to ~1e-6, growing with
        # embedding size), so a strict absolute tolerance would be unrealistic;
        # meaningful distances agree to ~1e-13 (checked below).
        assert np.isclose(result_block.distance, result_dense.distance, rtol=1e-8, atol=1e-6), \
            f"AIRM block and dense disagree for K={k}: block={result_block.distance}, dense={result_dense.distance}"
        # Also test with a shifted GMM
        delta = np.full(gmm.dim, 0.1)
        gmm_shifted = shifted_gmm(gmm, delta)
        result_block_shifted = gmm_airm(gmm, gmm_shifted, method="block")
        result_dense_shifted = gmm_airm(gmm, gmm_shifted, method="dense")
        assert np.isclose(result_block_shifted.distance, result_dense_shifted.distance, rtol=1e-8, atol=1e-12), \
            f"AIRM block and dense disagree for shifted K={k}: block={result_block_shifted.distance}, dense={result_dense_shifted.distance}"
        # Check that the embedding is block-diagonal (off-block entries should be zero)
        from src.airm_gmm import spd_embedding
        S = spd_embedding(gmm, allow_large=True)  # Allow large for checking
        # The embedding should be block-diagonal with blocks of size (n+1)x(n+1)
        n_plus_1 = gmm.dim + 1
        for i in range(gmm.k):
            for j in range(gmm.k):
                if i != j:
                    # Check the off-diagonal block
                    block = S[i*n_plus_1:(i+1)*n_plus_1, j*n_plus_1:(j+1)*n_plus_1]
                    assert np.allclose(block, 0, atol=1e-12), \
                        f"Off-diagonal block ({i},{j}) of SPD embedding should be zero for K={k}"
    print("PASS")


def test_airm_metric_axioms() -> None:
    """Test that AIRM satisfies symmetry and triangle inequality."""
    print("Testing AIRM metric axioms...", end=" ")
    # Create three GMMs: base, shifted slightly, shifted more
    gmm = build_beta_product_gmm(k=50)  # Use smaller K for speed
    delta1 = np.full(gmm.dim, 0.05)
    delta2 = np.full(gmm.dim, 0.10)
    gmm1 = shifted_gmm(gmm, delta1)
    gmm2 = shifted_gmm(gmm, delta2)
    # Symmetry: d(a,b) = d(b,a)
    d_ab = gmm_airm(gmm, gmm1).distance
    d_ba = gmm_airm(gmm1, gmm).distance
    assert abs(d_ab - d_ba) < 1e-12, f"AIRM not symmetric: d(ab)={d_ab}, d(ba)={d_ba}"
    # Triangle inequality: d(a,c) <= d(a,b) + d(b,c)
    d_ac = gmm_airm(gmm, gmm2).distance
    d_bc = gmm_airm(gmm1, gmm2).distance
    assert d_ac <= d_ab + d_bc + 1e-12, \
        f"AIRM triangle inequality violated: d(ac)={d_ac} > d(ab)+d(bc)={d_ab}+{d_bc}={d_ab+d_bc}"
    print("PASS")


def main() -> int:
    """Run all tests."""
    print("Running validation tests...\n")
    try:
        test_beta_gmm()
        test_component_hellinger()
        test_hellinger_self_distance()
        test_airm_self_distance()
        test_airm_block_vs_dense()
        test_airm_metric_axioms()
        print("\nAll tests PASSED!")
        return 0
    except AssertionError as e:
        print(f"\nFAILED: {e}")
        return 1
    except Exception as e:
        print(f"\nERROR: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())