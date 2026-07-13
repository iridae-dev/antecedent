# Phase 5 deviations

Intentional deviations / notes for DESIGN.md §32 Phase 5 (PCMCI+ and full CI
framework) relative to `parity/tigramite.toml` (authoritative for `status`).

## 1. Weighted ParCorr via name string uses unit weights

`ci_from_name("weighted_parcorr")` (and the Python `ci=` kwarg) constructs a
unit-weight adapter. Custom observation weights remain available in Rust via
`WeightedPartialCorrelation::new(weights)`. Discovery does not accept a weight
vector across the Python FFI boundary in Phase 5.

## 2. Multivariate ParCorr is a first-PC approximation

`MultivariatePartialCorrelation` residualizes each multivariate block onto its
leading principal direction then applies scalar ParCorr. Full matrix-variate
partial correlation (Tigramite’s denser multivariate API) is not claimed.

## 3. kNN / mixed / symbolic CMI significance is coarse

Phase 5 CMI tests use a lightweight permutation / proxy p-value suitable for
wiring and calibration smoke tests. Analytic KSG degrees-of-freedom and full
null distributions may be tightened in a later stats pass without changing the
`ConditionalIndependence` surface.

## 4. GPDC is native RBF-GP + distance correlation (no torch)

Matches the Phase 5 locked decision. Numerical parity with Tigramite’s
torch-backed GPDC is not required; deviations are expected under the native
backend.

## 5. PCMCI+ conformance is clean-room Exact parents

`conformance/tigramite/pcmci_plus_lag0` pins a synthetic SCM with lagged +
contemporaneous parents. Black-box Tigramite PCMCI+ graph comparison is not
pinned in this fixture (`tigramite.available = false`).

## 6. Circle endpoints / LPCMCI remain Phase 8

Temporal CPDAG rejects `Endpoint::Circle` with a clear error. PAG / LPCMCI stay
out of Phase 5.
