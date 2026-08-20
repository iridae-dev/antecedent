# temporal response baselines

Criterion benches (run with `--test` in release / feature gates):

- `antecedent-estimate` bench `temporal_response`:
  `temporal_response_multi_horizon_n800`,
  `temporal_response_intervention_shift_n100000`

**Budgets (local regression, Apple M1 class):**

| Case | Soft latency budget |
|------|---------------------|
| temporal_response_multi_horizon_n800 | asserted gate **25 ms** (2× headroom for `--test` noise) |
| temporal_response_intervention_shift_n100000 | asserted gate **200 ms** (measured ~10.7 ms steady state; wide headroom is deliberate — see below) |

Allocation / memory contract: prepare-once identification + indexer are reused
across estimate clicks on the facade `PreparedStudy::estimate_series` path;
this bench pins the shared estimator hot path that those clicks call for a
multi-horizon dose grid without re-unfolding identification.

`temporal_response_intervention_shift_n100000` covers the `InterventionResponse`
additive-shift path specifically, which `temporal_response_multi_horizon_n800`
does not exercise at all (it only benches `MeanCurve`). This path was previously
O(n^2 * p) — it averaged g-computation over every observed treatment level — and
was collapsed to an O(n*p) closed form `mu_hat(Abar + delta)` (an O(p) evaluation
per horizon on top of the unavoidable O(n*p) design fit). n = 100,000 is large
enough that a reintroduced per-observation averaging loop would push wall time
from ~10 ms to well over the 200 ms budget, while the closed-form path stays
comfortably within it. This repo already suffered one silent O(n^2) response-curve
regression (0.5.2, 77× fix); this bench exists so that class of regression on the
temporal `InterventionResponse` path fails the gate instead of shipping unnoticed.
