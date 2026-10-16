"""Check whether rearranging (permuting) the components of a GMM changes the
AIRM distance computed against an identical, unpermuted GMM.

This program builds a Beta-product GMM, creates a copy in which the 1st and
last components are exchanged (weights, means, and covariances swapped
consistently), and compares the AIRM distance of the pair
(original, permuted) against the baseline pair (original, original).

The AIRM distance here is the SPD-embedding variant: the embedding is a
block-diagonal matrix with one (n+1, n+1) SPD block per component, and the
distance is

    d(G1, G2) = sqrt( sum_k || log(S1_k^{-1/2} S2_k S1_k^{-1/2}) ||_F^2 ).

Because the sum runs over positionally matched component blocks, a one-sided
rearrangement of components is expected to leave the distance unchanged; this
program verifies that numerically via both the fast "block" path and the
reference "dense" path.

Usage
-----
    python check_airm_permutation.py [--k 100] [--method {block,dense,both}]

Exit code 0 if all checks pass, 1 otherwise.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from src.airm_gmm import gmm_airm
from src.beta_gmm import build_beta_product_gmm
from src.gmm_types import GMM


def swap_first_last(g: GMM) -> GMM:
    """Return a copy of ``g`` with the 1st and last components exchanged.

    The weights, means, and covariances of component 0 and component K-1 are
    swapped together so the permuted GMM represents exactly the same mixture
    distribution, just with components listed in a different order.

    Parameters
    ----------
    g : GMM
        Input GMM (must have at least 2 components).

    Returns
    -------
    GMM
        New GMM with the first and last components exchanged.
    """
    if g.k < 2:
        raise ValueError("GMM must have at least 2 components to swap")
    weights = g.weights.copy()
    means = g.means.copy()
    covariances = g.covariances.copy()
    weights[[0, -1]] = weights[[-1, 0]]
    means[[0, -1]] = means[[-1, 0]]
    covariances[[0, -1]] = covariances[[-1, 0]]
    return GMM(weights=weights, means=means, covariances=covariances)


def sort_components(g: GMM) -> GMM:
    """Return a copy of ``g`` with components sorted by weight (descending),
    breaking ties by the Euclidean norm of the mean vector (ascending).

    This produces a deterministic ordering that is invariant to permutations
    of components with identical weight/mean-norm pairs.

    Parameters
    ----------
    g : GMM
        Input GMM.

    Returns
    -------
    GMM
        New GMM with components sorted as described.
    """
    # Compute sorting keys: (-weight, norm(mean))
    weights = g.weights
    means = g.means
    norms = np.linalg.norm(means, axis=1)
    # argsort returns indices that would sort the array
    order = np.lexsort((norms, -weights))  # primary: -weight, secondary: norm
    sorted_weights = weights[order]
    sorted_means = means[order]
    sorted_covariances = g.covariances[order]
    return GMM(weights=sorted_weights, means=sorted_means, covariances=sorted_covariances)


def _report(label: str, d_ref: float, d_test: float, rtol: float, atol: float) -> bool:
    """Print a comparison row and return whether it passed."""
    diff = abs(d_test - d_ref)
    ok = bool(np.isclose(d_test, d_ref, rtol=rtol, atol=atol))
    status = "PASS" if ok else "FAIL"
    print(
        f"  {label:<38} ref={d_ref:.12e} test={d_test:.12e} "
        f"|diff|={diff:.3e} [{status}]"
    )
    return ok


def main(argv: list[str] | None = None) -> int:
    """Run the permutation-consistency check for the AIRM distance."""
    parser = argparse.ArgumentParser(
        description=(
            "Check whether exchanging the 1st and last components of a GMM "
            "changes its AIRM distance against the identical unpermuted GMM."
        )
    )
    parser.add_argument(
        "--k",
        type=int,
        default=100,
        help="Number of GMM components (default: 100)",
    )
    parser.add_argument(
        "--method",
        choices=["block", "dense", "both"],
        default="both",
        help="AIRM computation method(s) to use (default: both)",
    )
    parser.add_argument(
        "--sort",
        action="store_true",
        help="Sort components by weight then mean norm before computing distance",
    )
    args = parser.parse_args(argv)

    methods = ["block", "dense"] if args.method == "both" else [args.method]

    gmm_raw = build_beta_product_gmm(k=args.k)
    if args.sort:
        gmm = sort_components(gmm_raw)
        gmm_perm = sort_components(swap_first_last(gmm_raw))
    else:
        gmm = gmm_raw
        gmm_perm = swap_first_last(gmm_raw)

    print(f"GMM with K={gmm.k} components, dim={gmm.dim}")
    print("Permutation: exchange 1st and last component (one-sided)")
    print()

    all_ok = True
    for method in methods:
        print(f"Method: {method}")
        # Tolerances: very tight for the same method; looser when comparing
        # block vs dense paths, which use different numerical routes.
        rtol_same = 1e-10
        rtol_cross = 1e-8

        base = {m: gmm_airm(gmm, gmm, method=m).distance for m in methods}
        perm = {m: gmm_airm(gmm, gmm_perm, method=m).distance for m in methods}

        ok = True
        ok &= _report(
            f"baseline d(gmm, gmm) [{method}]",
            base[method],
            base[method],
            rtol_same,
            0.0,
        )
        ok &= _report(
            f"permuted d(gmm, gmm_perm) [{method}] vs baseline",
            base[method],
            perm[method],
            rtol_same,
            0.0,
        )
        if args.method == "both":
            for m in methods:
                ok &= _report(
                    f"permuted [{m}] vs baseline [other]",
                    base["dense" if m == "block" else "block"],
                    perm[m],
                    rtol_cross,
                    0.0,
                )
        all_ok &= bool(ok)
        print()

    print("RESULT:", "PASS - AIRM distance is invariant to component exchange" if all_ok
          else "FAIL - AIRM distance changed under component exchange")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
