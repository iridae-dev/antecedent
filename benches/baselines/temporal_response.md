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

## 1.2 posterior and sustained-window reference

Measured locally on macOS 26.5.2 arm64, 2026-09-07, optimized Criterion build,
10 samples, 1 s warm-up and 1 s measurement. Values are Criterion means and
95% bootstrap confidence intervals; these short local runs are regression
references, not portable latency guarantees.

| Case | Mean (95% interval) | Optional local mean gate |
|------|---------------------|--------------------------|
| sustained_window_n800_bootstrap20 | 443.4 µs (436.0–450.8 µs) | ≤ **5 ms** |
| sustained_window_n800_draws512 | 187.6 µs (173.9–210.9 µs) | ≤ **5 ms** |
| bayesian_temporal_response_n800_draws512_grid4 | 96.5 µs (93.7–100.9 µs) | ≤ **2 ms** |

The window fixture intervenes at both lagged treatment times. Its Frequentist
fit refits mechanisms on 20 shared moving-block samples; its Bayesian fit
retains 512 composed effects. The response fixture fits one identified horizon
and evaluates four doses. Identification/indexer construction is outside the
timed loops; fitting and posterior composition are inside. Output assertions
pin 512 effects or four surface means, without retaining a row-by-draw result.
They do not measure peak working allocation. The optional mean gates use
`GATE_CRITERION_MEANS=1 bash scripts/gate_hot_path_baselines.sh`; release smokes
always execute the cases and their shape assertions.
