# Implementation Plan

> Target plan file: `plan/plan-compare-distance-metrics.prompt.md`. Algorithm references:
> `plan/Hellinger_algorithm.md` (Mercurio & Singla LP-based GMM Hellinger) and
> `plan/dAIRM_algorithm.md` (SPD-embedding AIRM / Riemannian GMM distance).

## [Overview]

Build a small, self-contained Python benchmark package that (a) constructs a 3D
Gaussian mixture model with exactly K components from the product distribution of
three independent `Beta(2, 5)` marginals on `[0,1]^3`, (b) computes the
distance of that GMM from **itself** with two different GMM dissimilarity
metrics — the LP-based Hellinger distance (Gaussian closed-form component
matrix + optimal-transport LP) and the AIRM Riemannian distance via the
SPD block embedding — and (c) times **only the distance calculations**, in two
independent timing-experiment programs:
1. `benchmark_single_run.py` — one single timed run per metric for
   `K ∈ {100, 200, 500, 1000}`.
2. `benchmark_repeat_runs.py` — mean (plus std/min/median) of **5** repeat runs
   per metric for `K ∈ {100, 200, 500, 1000}`.

Scope and context. The repository is currently empty except for `papers/`; this
is greenfield code. Environment: Windows, PowerShell, Python 3.13.14,
`numpy 2.5.2` and `scipy 1.18.0` already installed; **POT is not installed**
and must be added (`pip install POT`) for `ot.emd`. All code is plain
`numpy`/`scipy`/`ot` — no new numerical framework.

High-level approach.
* State dimension `n = 3`. Covariances are diagonal (`Sigma_k =
  diag(sigma_1^2, sigma_2^2, sigma_3^2)`) because the product-Beta grid is
  axis-aligned; the Hellinger implementation nevertheless keeps a **general
  full-covariance path** so the code is a real metric implementation, not a
  special case hack.
* The product-Beta GMM uses a per-axis lattice of cell midpoints. K is factored
  exactly into `(m1, m2, m3)` with `m1*m2*m3 = K` and `max(m)-min(m)` minimal,
  giving `K=100 → (5,5,4)`, `K=200 → (5,5,8)`, `K=500 → (5,10,10)`,
  `K=1000 → (10,10,10)`. No truncation or moment-matching is needed, so the
  same code path works for all four K values.
* Component weights are the product-Beta density at the cell midpoints
  (cell volumes are constant and cancel under renormalization).
* Each component's variance is the half-cell heuristic
  `sigma_d = 0.5 / m_d`, i.e. `2*sigma_d` spans one cell. Boundary cells keep a
  strictly positive weight for `Beta(2,5)`, so the LP is non-degenerate.
* Hellinger cost matrix `D` is built in row-chunks with fully batched
  `numpy` linear algebra (`slogdet` + `solve`), then the transport LP is solved
  with `ot.emd` (network simplex) — the exact optimum of the Mercurio–Singla LP,
  but far faster than a general-purpose LP at `K=1000` (1,000,000 variables).
* AIRM exploits a structural fact that is *derived* from the reference
  algorithm and is **mathematically exact**, not an approximation: with the
  embedding `S = [[A, X], [X^T, B]]` and `A, X, B` block-diagonal in the
  component index, `S` is **block-diagonal with K blocks of size (n+1)×(n+1)
  = 4×4**, one per mixture component. Therefore
  `S1^{-1/2} S2 S1^{-1/2}` is also block-diagonal, the matrix log is
  block-diagonal, and
  `d_AIRM(G1,G2)^2 = sum_k || log( S1_k^{-1/2} S2_k S1_k^{-1/2} ) ||_F^2`.
  The default fast path evaluates the K independent 4×4 problems with batched
  `np.linalg.eigh`. A **naive dense path** (`scipy.linalg.logm` on a
  `K(n+1) × K(n+1)` matrix, exactly as written in
  `papers/dAIRM_algorithm.md`) is retained and used as a cross-check for
  `K ≤ 200` (dense size ≤ 800×800; the 2000×2000 and 4000×4000 cases are
  infeasible for `logm`, which is itself a reportable finding).
* Timing hygiene: GMM construction, imports, and CSV I/O happen **outside** the
  timed region. Only `gmm_hellinger(G, G)` / `gmm_airm(G, G)` are timed with
  `time.perf_counter()`, and each metric additionally reports sub-timings
  (Hellinger: cost-matrix build vs transport solve; AIRM: embedding build vs
  matrix-function core) so the bottleneck is visible.
* Expected, checkable outputs: both self-distances are ≈ 0 (Hellinger exactly 0
  because component 1 can be coupled to component 1 at zero cost; AIRM exactly 0
  because `S1 = S2 ⇒ M = I ⇒ log M = 0`).

## [Types]

All types live in `src/gmm_types.py` (no third-party typing dependencies).

1. `GMM` — frozen `@dataclass` representing a GMM with component count K and
   state dimension n.
   * `weights: np.ndarray` — shape `(K,)`, float64, `weights.sum() == 1`,
     every entry `> 0`.
   * `means: np.ndarray` — shape `(K, n)`, float64.
   * `covariances: np.ndarray` — shape `(K, n, n)`, float64, symmetric
     positive-definite.
   * Derived properties: `k` (int, number of components), `dim` (int, n).
   * `validate(tol: float = 1e-12) -> None` — raises `ValueError` on shape
     mismatch, non-finite values, non-positive weights, weight sum ≠ 1, or
     non-SPD covariance (checked via eigenvalues of the symmetric part).
2. `TimingRecord` — `@dataclass` with
   `k: int`, `metric: str`, `run_index: int`, `total_s: float`,
   `stage_times: dict[str, float]`, `distance: float`.
   Used by both benchmark programs and serialized to CSV.
3. `AIRMResult` — `@dataclass` with `distance: float`,
   `stage_times: dict[str, float]`, `method: str` (`"block"` or `"dense"`),
   `spd_size: int`.
4. `HellingerResult` — `@dataclass` with `distance: float`,
   `stage_times: dict[str, float]`, `cost_matrix_shape: tuple[int, int]`,
   `transport_sum_marginal_error: float` (max abs deviation of the optimal
   plan's row/column sums from the prescribed weights; a feasibility check).
5. Module-level constants: `BETA_A = 2.0`, `BETA_B = 5.0`, `SUPPORT = (0.0, 1.0)`,
   `DEFAULT_CHUNK = 64`, `CELL_FRACTION = 0.5`, `SUPPORTED_K = (100, 200, 500,
   1000)`, `REPEATS = 5`.

## [Files]

New files (all paths relative to
`c:\Users\sriha\OneDrive - Indian Institute of Space Science and Technology\compare-distance-metrics`):

| File | Purpose |
| --- | --- |
| `.prompt.md` | This plan. |
| `requirements.txt` | Pinned dependency list: `numpy>=2.0`, `scipy>=1.13`, `POT>=0.9.4`. |
| `README.md` | Short usage note: install, run each of the four entry points, expected outputs, and how to pin BLAS threads (`$env:OMP_NUM_THREADS=1`) for reproducible timing. |
| `src/__init__.py` | Marks the package; exposes `build_beta_product_gmm`. |
| `src/gmm_types.py` | `GMM`, `TimingRecord`, `AIRMResult`, `HellingerResult`, constants. |
| `src/beta_gmm.py` | Beta(2,5) → 3D product-Beta GMM with exactly K components; 10×10×10 lattice at K=1000. |
| `src/hellinger_gmm.py` | **Program 1**: LP-based GMM Hellinger distance (component cost matrix + `ot.emd`), importable + CLI. |
| `src/airm_gmm.py` | **Program 2**: SPD-embedding AIRM GMM distance (block fast path + dense reference path), importable + CLI. |
| `src/timing_utils.py` | `perf_counter` helpers, CSV writers, console table formatter. |
| `benchmark_single_run.py` | Timing experiment protocol 1: single timed run per metric per K ∈ {100,200,500,1000}. |
| `benchmark_repeat_runs.py` | Timing experiment protocol 2: 5 repeats per metric per K, mean/std/min/median. |
| `tests/validate_metrics.py` | Dependency-free assertion script (run as `python tests/validate_metrics.py`; pytest is not installed and is deliberately not required). |
| `results/` | Created at runtime by the benchmark scripts (contains the CSVs). |

File modifications to existing content: none — the repository currently contains
only `papers/` (four reference PDFs and the two algorithm markdown files), which
are read-only inputs and are not modified.

Files to delete or move: none.

Configuration updates: add `requirements.txt` and a minimal `.gitignore`
(`__pycache__/`, `results/`, `.venv/`) since no version-control ignore file
exists yet. No `pyproject.toml`/`setup.cfg` is needed because every entry point
is run from the repo root with `python -m src.<module>` or
`python <script>.py`, so the repo root is already on `sys.path`.

Output artifacts (generated, not hand-written):
* `results/benchmark_single.csv` — columns `k, metric, total_s,
  stage_1_name, stage_1_s, stage_2_name, stage_2_s, distance`.
* `results/benchmark_repeat.csv` — same columns plus `run_index` (1..5).
* `results/benchmark_repeat_summary.csv` — columns `k, metric, mean_s, std_s,
  min_s, median_s, max_s, distance`.

## [Functions]

### `src/beta_gmm.py`

1. `beta_pdf(x: np.ndarray, a: float = BETA_A, b: float = BETA_B) -> np.ndarray`
   — `scipy.stats.beta.pdf` wrapper returning a float64 array of the same shape;
   accepts scalars/arrays; used only for weight computation.
2. `factorize_components(k: int) -> tuple[int, int, int]`
   — enumerate all divisor triples `(m1,m2,m3)` with `m1*m2*m3 == k` and
   `m1 <= m2 <= m3`, return the triple minimizing `(m3 - m1, m3)`.
   Yields `100 -> (4,5,5)`, `200 -> (5,5,8)`, `500 -> (5,10,10)`,
   `1000 -> (10,10,10)`. Raises `ValueError` if `k < 1`.
3. `axis_grid(m: int, cell_fraction: float = CELL_FRACTION) -> tuple[np.ndarray, float]`
   — for `m` cells on `[0,1]`: midpoints `(i + 0.5)/m` and half-width
   `sigma = cell_fraction / m`. Returns `(midpoints, sigma)`.
4. `build_beta_product_gmm(k: int = 1000, a: float = BETA_A, b: float = BETA_B,
   cell_fraction: float = CELL_FRACTION, min_weight: float = 1e-300) -> GMM`
   — core constructor. Steps: factorize `k` into `(m1,m2,m3)`; build the three
   midpoint axes and their `sigma_d`; form the Cartesian product with
   `np.meshgrid(..., indexing="ij")` → means `(K, 3)`; weights =
   product of the three marginal Beta PDFs at the midpoints, floored at
   `min_weight` and renormalized to sum to 1; covariances =
   `diag(sigma_1^2, sigma_2^2, sigma_3^2)` broadcast to `(K, 3, 3)`.
   Calls `GMM.validate()`. Raises `ValueError` for `k <= 0` or
   `cell_fraction <= 0`. Deterministic (no RNG).
5. `build_beta_product_gmm_1d(k: int = 1000, ...) -> GMM`
   — convenience for `n = 1` (a single Beta(2,5) quantized into `k` cells);
   used by the analytic validation against the closed-form Gaussian Hellinger
   distance. Shares all internals with (4).
6. `shifted_gmm(gmm: GMM, delta: np.ndarray) -> GMM`
   — returns a copy with `means + delta` (weights/covariances unchanged);
   used to build a non-identical GMM so validation can assert distances `> 0`.

### `src/hellinger_gmm.py` (Program 1)

1. `gaussian_hellinger_block(mu_p, cov_p, mu_q, cov_q) -> np.ndarray`
   — vectorized full-covariance Gaussian Hellinger distance matrix
   `D[i,j] = sqrt(max(0, 1 - B_ij))` with
   `Sigma_ij = 0.5*(cov_p[i] + cov_q[j])` and
   `log B_ij = 0.25*logdet(cov_p[i]) + 0.25*logdet(cov_q[j])
   - 0.5*logdet(Sigma_ij) - 0.5*dmu^T Sigma_ij^{-1} dmu`, where
   `dmu = mu_p[i] - mu_q[j]`. Exactly §2 / Step 2 of
   `papers/Hellinger_algorithm.md`. Implemented with batched
   `np.linalg.slogdet` and `np.linalg.solve` on stacked `(C, M, n, n)` arrays;
   `B` is clipped to `[0, 1]` before the square root for numerical safety and
   the wrapper records how often clipping was active.
2. `component_hellinger_matrix(g1: GMM, g2: GMM, chunk: int = DEFAULT_CHUNK,
   return_stage_time: bool = False)`
   — loops over row chunks of `g1` (default 64) to bound memory at
   `chunk * K * n * n * 8` bytes (~4.6 MB per stacked operand at K=1000, n=3),
   assembling the `(K1, K2)` matrix; asserts `g1.dim == g2.dim`.
   Stage key: `"cost_matrix"`.
3. `transport_cost(cost: np.ndarray, weights_p: np.ndarray,
   weights_q: np.ndarray, solver: str = "pot") -> tuple[float, np.ndarray]`
   — solves the Mercurio–Singla LP `min sum_ij w_ij D_ij` s.t. `w >= 0`,
   row sums `= alpha`, column sums `= beta`. `"pot"` uses
   `ot.emd(weights_p, weights_q, cost)` (network simplex → the same exact
   optimum as a general LP, but viable at 1e6 variables) and returns
   `(np.sum(plan * cost), plan)`. Raises an `ImportError` carrying the exact
   `pip install POT` instruction when `ot` is missing. `"scipy"` switches to
   `scipy.optimize.linprog(method="highs")` with equality constraints assembled
   from `scipy.sparse`, documented as a slow cross-check for small K only.
4. `gmm_hellinger(g1: GMM, g2: GMM, chunk: int = DEFAULT_CHUNK,
   solver: str = "pot") -> HellingerResult`
   — the **timed distance function**: cost matrix → LP → distance, with
   `stage_times = {"cost_matrix": t1, "transport_lp": t2}` and
   `transport_sum_marginal_error` computed from the returned plan.
5. `main(argv: list[str] | None = None) -> int`
   — CLI: `python -m src.hellinger_gmm --k 1000 [--chunk 64]
   [--solver pot|scipy] [--gmm b|shifted] [--repeat 1]`. Builds the Beta
   product GMM, computes the self-distance by default, prints distance plus
   per-stage timings, and exits non-zero if `GMM.validate()` or the LP
   feasibility check fails.

### `src/airm_gmm.py` (Program 2)

1. `component_spd_blocks(g: GMM) -> np.ndarray`
   — for each component `k`: `A_k = Sigma_k + pi_k * mu_k mu_k^T`,
   `X_k = pi_k * mu_k`, `B_k = pi_k`; returns the stacked `(K, n+1, n+1)`
   array with `S_k = [[A_k, X_k], [X_k^T, pi_k]]` (Steps 1–2 and 12–22 of
   `papers/dAIRM_algorithm.md`).
2. `spd_embedding(g: GMM, allow_large: bool = False) -> np.ndarray`
   — dense `(K(n+1), K(n+1))` embedding `S = [[A, X],[X^T, B]]` assembled with
   `scipy.linalg.block_diag`; used by the dense reference path and by the
   block-diagonality validation. Raises `ValueError` (explicit memory guard)
   when `K*(n+1) > 1000` and `allow_large` is `False`, preventing an accidental
   4000×4000 allocation.
3. `airm_block(g1: GMM, g2: GMM) -> AIRMResult`
   — **default fast, exact path**. Batched per-component computation:
   `w1, V1 = np.linalg.eigh(S1)`; `S1^{-1/2} = V1 diag(w1^{-1/2}) V1^T`;
   `M_k = S1_k^{-1/2} S2_k S1_k^{-1/2}`; symmetrize `M_k ← 0.5*(M_k + M_k^T)`;
   `d = sqrt(sum_k sum_a log(lambda_{k,a})^2)` with
   `lambda = np.linalg.eigvalsh(M)`. No Python-level loop over components —
   every operation is a stacked `(K, n+1, n+1)` batched call.
   Asserts `g1.k == g2.k` and `g1.dim == g2.dim` (the reference method requires
   equal K). `stage_times = {"embedding": t1, "matrix_function": t2}`.
4. `airm_dense(g1: GMM, g2: GMM, use_logm: bool = True) -> AIRMResult`
   — naive reference path following the pseudocode literally:
   `M = S1^{-1/2} S2 S1^{-1/2}` with `S1^{-1/2}` from
   `scipy.linalg.fractional_matrix_power(S1, -0.5)`, then `scipy.linalg.logm(M)`
   and `np.linalg.norm(L, "fro")`. `use_logm=False` swaps `logm` for the
   eigendecomposition shortcut `sqrt(sum(log(eigvalsh(M))^2))`, isolating the
   `logm` cost. `stage_times = {"embedding": t1, "sqrtm": t2, "logm": t3}`.
5. `gmm_airm(g1: GMM, g2: GMM, method: str = "auto") -> AIRMResult`
   — dispatcher: `"auto"` → `airm_block` for `g1.k >= 100`, else `airm_dense`;
   `"block"`/`"dense"` force a path; `"both"` runs both and raises when they
   disagree beyond tolerance (used by `tests/validate_metrics.py`).
6. `main(argv: list[str] | None = None) -> int`
   — CLI: `python -m src.airm_gmm --k 1000 [--method block|dense|auto]
   [--gmm b|shifted] [--repeat 1]`. Prints distance, `method`, `spd_size`, and
   stage timings.

### `src/timing_utils.py`

1. `timed(func, *args, **kwargs) -> tuple[Any, float]`
   — wraps a call between two `time.perf_counter()` reads; returns
   `(result, elapsed_s)`. This is the single timing primitive used everywhere.
2. `write_csv(path: str | Path, rows: list[dict], stage_order: list[str]) -> None`
   — `csv.DictWriter` writer that creates parent directories; flattens
   `stage_times` into fixed `stage_1_name/stage_1_s/stage_2_name/stage_2_s`
   columns so the CSV schema is stable regardless of metric.
3. `summarize(rows: list[dict]) -> list[dict]`
   — groups by `(k, metric)` and computes `mean/std/min/median/max` of
   `total_s` (stdlib `statistics`), carrying the measured `distance`.
4. `format_table(rows: list[dict], columns: list[str]) -> str`
   — fixed-width console table with no third-party formatting dependency
   (`pandas` is installed but deliberately unused, keeping the dependency
   surface minimal).
5. `thread_env_report() -> dict[str, str]`
   — reads `OMP_NUM_THREADS`/`OPENBLAS_NUM_THREADS`/`MKL_NUM_THREADS` (read-only)
   so benchmark output records the threading configuration next to the timings.

### `benchmark_single_run.py` — timing experiment protocol 1

1. `run_single(k_list: Sequence[int], metrics: Sequence[str], chunk: int = DEFAULT_CHUNK,
   solver: str = "pot") -> list[dict]`
   — for each K: build the Beta product GMM **once** (untimed, followed by
   `GMM.validate()`), then for each metric perform exactly **one** timed call to
   `gmm_hellinger(gmm, gmm)` or `gmm_airm(gmm, gmm)`. No repeats and no warm-up
   run (a single cold run is the point of this protocol). Records
   `TimingRecord`s, then returns one flattened dict per `(k, metric)`.
2. `main(argv: list[str] | None = None) -> int`
   — flags `--k 100 200 500 1000`, `--metrics hellinger,airm`, `--chunk`,
   `--solver`, `--out results/benchmark_single.csv`. Prints the thread-env
   report, the results table, and writes the CSV.

### `benchmark_repeat_runs.py` — timing experiment protocol 2

1. `run_repeats(k_list: Sequence[int], metrics: Sequence[str],
   repeats: int = REPEATS, chunk: int = DEFAULT_CHUNK, solver: str = "pot")
   -> tuple[list[dict], list[dict]]`
   — same structure as protocol 1, but each metric is called `repeats` times
   (default exactly **5**, no extra warm-up) with `run_index = 1..repeats`;
   GMM construction still happens once per K and outside the timed region.
   Returns `(raw_rows, summary_rows)` where `summary_rows` come from
   `timing_utils.summarize`.
2. `main(argv) -> int`
   — flags `--k`, `--metrics`, `--repeats`, `--chunk`, `--solver`,
   `--out-raw results/benchmark_repeat.csv`,
   `--out-summary results/benchmark_repeat_summary.csv`. Prints per-run timings,
   the mean/std/min/median table, and a `ratio = airm_mean / hellinger_mean`
   column for each K.

### `tests/validate_metrics.py` — dependency-free validation

1. `test_beta_gmm() -> None`
   — asserts K ∈ {100,200,500,1000} yields exactly K components and 3 dims,
   weights > 0 and sum to 1, factorizations match `(4,5,5)/(5,5,8)/(5,10,10)/
   (10,10,10)`, and that component means stay inside `[0,1]^3`.
2. `test_component_hellinger() -> None`
   — `D_ii == 0` for identical components, `D ∈ [0,1]`, symmetry
   `D_ij == D_ji` for identical GMMs, and agreement with a hand-written scalar
   formula on 5 random SPD pairs (tolerance `1e-12`).
3. `test_hellinger_self_distance() -> None`
   — `gmm_hellinger(g, g).distance < 1e-10` and LP marginal error `< 1e-9`;
   then `> 0.05` for the mean-shifted copy.
4. `test_airm_self_distance() -> None`
   — `gmm_airm(g, g).distance < 1e-12` for the block path.
5. `test_airm_block_vs_dense() -> None`
   — `K = 100` and `K = 200`: relative difference `< 1e-8` between
   `airm_block` and `airm_dense` on both the self-distance and a mean-shifted
   pair; also asserts the deviation of the off-block entries of
   `spd_embedding(g)` from zero (the block-diagonal property that justifies the
   fast path).
6. `test_airm_metric_axioms() -> None`
   — symmetry `d(a,b) == d(b,a)` and triangle inequality
   `d(a,c) <= d(a,b) + d(b,c)` on three mean-shifted GMMs (tolerance `1e-9`).
7. `main() -> int` — runs all of the above, prints `PASS`/`FAIL` per check, and
   returns non-zero on any failure.

## [Classes]

* `GMM` (`src/gmm_types.py`) — `@dataclass(frozen=True)`; fields
  `weights: np.ndarray`, `means: np.ndarray`, `covariances: np.ndarray`;
  properties `k` and `dim`; method `validate(tol=1e-12) -> None`.
  No inheritance. Constructed only via `build_beta_product_gmm`,
  `build_beta_product_gmm_1d`, or `shifted_gmm`, all of which validate before
  returning. `__post_init__` performs cheap structural shape checks so a
  malformed instance cannot silently reach the distance code.
* `TimingRecord` (`src/gmm_types.py`) — plain `@dataclass`;
  `to_row(stage_order: list[str]) -> dict` flattens the record for CSV output.
* `HellingerResult` (`src/gmm_types.py`) — `@dataclass`; carries
  `distance`, `stage_times`, `cost_matrix_shape`,
  `transport_sum_marginal_error`, `plan` (the optimal coupling, retained for
  validation).
* `AIRMResult` (`src/gmm_types.py`) — `@dataclass`; carries `distance`,
  `stage_times`, `method`, `spd_size`.
* No classes are removed or replaced (greenfield repository). The metric
  implementations are intentionally module-level functions rather than classes,
  matching the two standalone "programs" (`src/hellinger_gmm.py`,
  `src/airm_gmm.py`) required by the task, so each metric can be timed as a
  single function call.

## [Dependencies]

New packages (`requirements.txt`):
1. `POT>=0.9.4` — provides `ot.emd` for the Hellinger transport LP. **Not
   currently installed**; install with `pip install POT` (Python 3.13 wheels are
   available). This is the only new dependency.
2. `numpy>=2.0` — already satisfied (2.5.2). Used for all vectorized algebra.
3. `scipy>=1.13` — already satisfied (1.18.0). Used for
   `scipy.stats.beta.pdf`, `scipy.linalg.block_diag`,
   `scipy.linalg.fractional_matrix_power`, `scipy.linalg.logm`, and
   `scipy.optimize.linprog(method="highs")` (optional cross-check solver only).

Explicitly not added:
* `pytest` — not installed; the validation script uses plain `assert`s so no new
  test dependency is required.
* `pandas`, `ot`'s optional solvers, `cvxopt`, `tabulate`, `matplotlib` — the
  task only needs timing numbers printed/tabulated, and formatting is done with
  stdlib `csv`/`statistics`.

Version-change requirements: none (no downgrades; nothing pinned conflicting with
the installed set).

Integration notes:
* `import ot` is performed lazily inside `transport_cost` so the metric module
  can be imported and the AIRM benchmark run even before POT is installed; the
  error message names the fix explicitly.
* Optionally record `ot.__version__` in the CSV header comment for provenance.

## [Testing]

Approach: one dependency-free assertion script, `tests/validate_metrics.py`,
runnable as `python tests/validate_metrics.py` from the repo root, exiting
non-zero on failure. It covers, in order:
1. **GMM construction** — exact component counts/dimensions for K ∈
   {100,200,500,1000}; weights positive and normalized; means inside `[0,1]^3`;
   covariance SPD; factorization values as planned.
2. **Component Hellinger matrix** — `D_ii == 0`, `D ∈ [0,1]`, symmetry, and
   agreement with an independently written scalar implementation of the
   closed-form Gaussian Hellinger distance (reference formula from
   `papers/Hellinger_algorithm.md`) on random SPD pairs, tolerance `1e-12`.
3. **Hellinger end-to-end** — self-distance `< 1e-10`; LP plan row/column sums
   match `alpha`/`beta` to `1e-9`; distance `> 0.05` for a mean-shifted copy;
   `scipy`-solver result matches the `ot.emd` result at `K = 100` (tolerance
   `1e-8`), which validates the transport formulation independently of POT.
4. **AIRM end-to-end** — self-distance `< 1e-12`; block path vs dense `logm`
   path agreement `< 1e-8` relative at `K = 100` and `K = 200`; the
   block-diagonality assertion on `spd_embedding(g)`; symmetry and triangle
   inequality on three mean-shifted GMMs.
5. **Analytic 1D check** — for `K = 1`, `build_beta_product_gmm_1d` reduces the
   Hellinger distance to the single-Gaussian closed form and the AIRM distance
   to the `4×4` SPD geodesic distance computed by hand.

Existing test files to modify: none (no tests exist).

Benchmark validation strategy:
* Both self-distances must be ≈ 0 in every reported row — a nonzero value
  signals a bug in the cost matrix, the LP, or the embedding, so the benchmark
  scripts fail loudly (`exit 1`) instead of writing misleading timings.
* The two protocols are cross-validated: the single-run time for a given `K`
  should lie within ~15% of the 5-run mean; the `results/*.csv` files are
  checked to contain 4 K values × 2 metrics (protocol 1) and 4 × 2 × 5 rows
  (protocol 2).
* Each timed call is asserted to return a finite, non-negative number, and the
  Hellinger plan feasibility metric is asserted `< 1e-9` for every run.
* Timing scope is enforced structurally: `perf_counter` is only ever started
  inside `gmm_hellinger`/`gmm_airm` and their benchmark wrappers, never around
  GMM construction — the plan documents this as the invariant to preserve.

Edge cases explicitly handled:
* `K = 1000` Hellinger memory (row-chunked cost matrix) and LP size
  (1e6 variables → POT network simplex rather than general LP).
* `K = 1000` AIRM dense allocation guard (4000×4000) with the block path as the
  default; the dense path is documented as impractical above `K = 200`, and that
  limitation is itself a reportable timing result.
* `K` values not factorizable into a near-cubic triple (e.g. a prime like 997)
  — `factorize_components` falls back to the best triple found, and the
  constructor raises a clear error if no exact triple exists, so the benchmark
  cannot silently produce the wrong component count.
* Missing POT → actionable `ImportError` rather than a silent fallback.
* Zero or extreme weights at Beta boundary cells → floored at `1e-300` and
  renormalized, avoiding POT/`linprog` degeneracy warnings.
* Non-convergent or infinite-timing runs → benchmark exits non-zero without
  overwriting a valid previous CSV (write to a temp file then rename).

## [Implementation Order]

1. **Environment setup** — create `requirements.txt` (numpy/scipy/POT) and
   `.gitignore`; run `pip install POT` and confirm
   `python -c "import ot; print(ot.__version__)"`.
2. **`src/gmm_types.py`** — `GMM`, `TimingRecord`, `HellingerResult`,
   `AIRMResult`, and the shared constants. Everything else imports from here, so
   it must land first.
3. **`src/beta_gmm.py`** — `beta_pdf`, `factorize_components`, `axis_grid`,
   `build_beta_product_gmm`, `build_beta_product_gmm_1d`, `shifted_gmm`.
   Verify immediately: for K ∈ {100,200,500,1000} print `k`, `dim`, weight sum,
   factorization, and min/max mean. This is the "create a Beta distribution →
   convert it into a K-component GMM" deliverable.
4. **`src/hellinger_gmm.py`** — component cost matrix (chunked, general
   covariance), then `transport_cost` with `ot.emd`, then `gmm_hellinger`, then
   `main`. Smoke test: `python -m src.hellinger_gmm --k 100` must print a
   self-distance ≈ 0 and stage timings.
5. **`src/airm_gmm.py`** — `component_spd_blocks`, `spd_embedding`
   (with the size guard), `airm_block`, `airm_dense`, `gmm_airm`, `main`.
   Smoke test: `python -m src.airm_gmm --k 100 --method both` must show
   agreement between block and dense paths and a self-distance ≈ 0.
6. **`src/timing_utils.py`** — `timed`, `write_csv`, `summarize`,
   `format_table`, `thread_env_report`. Kept after the metrics so the exact
   stage keys are already known.
7. **`benchmark_single_run.py`** — protocol 1 (one timed call per metric per K,
   build-once-untimed). Run over K = 100, 200, 500, 1000 and write
   `results/benchmark_single.csv`.
8. **`benchmark_repeat_runs.py`** — protocol 2 (5 repeats per metric per K,
   mean/std/min/median). Write `results/benchmark_repeat.csv` and
   `results/benchmark_repeat_summary.csv`.
9. **`tests/validate_metrics.py`** — all checks from the [Testing] section;
   run it and require a clean exit before trusting any benchmark number.
10. **`README.md`** — document the four entry points, the meaning of each timing
    column, the fact that only distance calls are timed, the AIRM block-path
    justification (exact decomposition of the block-diagonal SPD embedding),
    the `K ≤ 200` dense-path limitation, and the `OMP_NUM_THREADS` note.
11. **Final validation run** — `python tests/validate_metrics.py`,
    `python benchmark_single_run.py`, `python benchmark_repeat_runs.py`; report
    the resulting comparison table (timings for Hellinger vs AIRM at each K,
    both protocols) and confirm both self-distances are ≈ 0.

### Commands used to validate (PowerShell, repo root)

```powershell
pip install POT
python -c "import ot; print(ot.__version__)"
python -m src.hellinger_gmm --k 1000
python -m src.airm_gmm --k 1000 --method block
python tests/validate_metrics.py
python benchmark_single_run.py --k 100 200 500 1000
python benchmark_repeat_runs.py --k 100 200 500 1000 --repeats 5
```

### Explicitly out of scope

* No plots/figures (numbers and CSVs only) unless requested later.
* No changes to anything under `papers/` (reference material, read-only).
* No non-self-distance sweep beyond the shifted-GMM sanity cases used in
  validation; the task specifies the distance of the GMM from itself.
* No multiprocessing/GPU threading experiments; only the read-only thread-env
  report is included, and `OMP_NUM_THREADS=1` may be set externally for
  reproducibility.
