# ADR 0021 — temporal response

- Status: Accepted
- Date: 2026-08-20

## Context

Invariant 5 is still half-true after 0.5–0.6: `PulseEffect` and
`SustainedEffect` are two-point temporal contrasts. Static
`ResponseCurve` / `InterventionResponse` already carry four result axes
(identification, support, uncertainty, assumptions) on a function-valued
payload, but they are typed-impossible on temporal graph classes
(`parity/support_n_a.toml`). Soft and sequenced interventions are refused
by the static g-computation path. `CausalState` registers only scalar ATE
queries. Artifact format 0.3 has no temporal-grid response functional.

0.7 makes time a response, not a contrast. It is one scientific expansion,
not a new identification theory.

## Decision

### Query spine

Temporal dose-over-horizon and policy-path queries live on
`CausalQuery::Response`, not a parallel query kind.

- `ResponseFunctional::MeanCurve` gains an optional temporal attachment:
  discrete horizons (steps after the policy origin), a
  [`TemporalPolicy`](../crates/antecedent-core/src/intervention.rs), and
  optional `max_history_lag`. When present, the public query name remains
  `ResponseCurve`.
- `ResponseFunctional::InterventionResponse` on a `TemporalDag` accepts
  Soft(`constant` / `additive_shift`) and a single-step `Sequence` policy
  under the licensed temporal path. Unsupported Soft families, multi-step
  `Sequence`, and nested sequences fail closed with a stable error; a
  multi-step sequence is refused, not silently collapsed to its last step.
- Positional `(treatment, outcome)` arguments are unchanged. Temporal
  fields are keyword-only on the Python dataclasses.

`PulseEffect` / `SustainedEffect` remain public names. On licensed
TemporalDag cells they dispatch through `execute_temporal` directly to
`TemporalLinearAdjustment`, the same lower-level estimator machinery the
dose × horizon surface (`TemporalResponseEstimator`) is built on — they are
not computed by slicing a surface the response estimator produced. The two
paths agree numerically as a two-point contrast on the shared known-truth
fixture (`conformance/response/temporal_dose_horizon`); this is observed
equivalence via shared adjustment machinery, not derivation by projection,
and it is not a second estimand family. Point-estimate agreement does not
extend to uncertainty: `TemporalResponseEstimator::new()` hardcodes zero
bootstrap replicates, while the Pulse/Sustained path carries whatever
`bootstrap_replicates` the `Study` was configured with, so standard errors
can diverge between the two paths even when point estimates match.

### Identification

Reuse [`TemporalBackdoorIdentifier`](../crates/antecedent-identify/src/temporal_backdoor.rs)
over a finite unfolding. No new identification algorithm. Prepare caches
identification once per unique requested horizon — `I(h)`, not a single
`I(max h)`. Estimate clicks must not re-identify. A union of per-horizon
adjustment sets is not treated as one shared `Z`; each cell is estimated
under the estimand identified for that horizon.

### Estimation

A temporal response estimator evaluates a dose × horizon surface (row-major
`ResponseValue::Surface` with `dimension = 2`: dose major, then horizon)
via linear g-computation on the unfolded backdoor design. Soft /
sequenced temporal policies that are licensed evaluate through the same
unfolded design with mechanism overlays; everything else refuses.

Result axes match static response: structural identification, empirical
support, uncertainty kind, assumptions. Empirical support is function-valued
on the same dose × horizon geometry as the estimate. Each cell is classified
against that horizon's lag-aligned treatment range. `SupportReport.status`
summarizes the cell grid (fully supported / partially extrapolative / no
cell supported) rather than the union of horizon ranges. Per-cell labels
live in `SupportReport.point_status`. Static curves keep worst-over-points.

Known-truth evidence is not a single DGP. `temporal_dose_horizon` pins the
unconfounded linear surface, bands, and licensed intervention overlays.
`temporal_confounded_pulse` pins identification under confounding:
adjustment set `{Z@-1}` at horizon 1, method `temporal.backdoor.unfolded`,
and the structural estimate (empty `Z` recovers the confounded association).
The same fixture's `multi_horizon` contract pins horizon-dependent
identification: `I(1)={Z@-1}` and `I(2)={}` on `horizons=[1,2]`. Reusing
the long-horizon empty set at the short cell is the confounded association.
`temporal_horizon_support` pins mixed
per-horizon empirical support.

### Support matrix (0.7 licensed subset)

License:

| query | graph_class | structure | inference | validation |
|-------|-------------|-----------|-----------|------------|
| `ResponseCurve` | `TemporalDag` | `explicit`, `accepted` | `Frequentist` | `none` |
| `InterventionResponse` | `TemporalDag` | `explicit`, `accepted` | `Frequentist` | `none` |
| `PulseEffect` | `TemporalDag` | `explicit`, `accepted` | `Frequentist` | `none` |
| `SustainedEffect` | `TemporalDag` | `explicit`, `accepted` | `Frequentist` | `none` |

`PulseEffect` / single-step `SustainedEffect` dispatch through
`execute_temporal` to `TemporalLinearAdjustment` directly and agree
numerically, as a two-point contrast, with the dose × horizon surface on the
shared fixture (`conformance/response/temporal_dose_horizon`) — shared
lower-level adjustment machinery, not derivation from the surface. Multi-step
Sustained remains estimator-refused. All other nearby allowlist rows stay
refused. Python query construction reads `TemporalResponseSpec::license()`
(horizon cap, allowed policies, default lag) rather than defining those
values.

Temporal CPDAG/PAG, graph-posterior response, Bayesian temporal surfaces,
and validation ≠ `none` remain refused / n/a / allowlisted as today.

`ResponseCurve` / `InterventionResponse` without a temporal attachment
remain n/a on temporal graph classes (static queries are not temporal
cells). With a temporal attachment they are the licensed cells above.

### CausalState

`CausalState` may register function-valued response queries. Append /
replace / intervention / assumption events invalidate cached curve
results. Recomputation is caller-driven via `refresh_results` only — never
an implicit rerun inside `apply`.

### Artifact format

Advance `STABLE_FORMAT` from `{ major: 0, minor: 3 }` to
`{ major: 0, minor: 4 }`. Format 0.4 adds temporal fields on response
functionals and documents dose × horizon surface layout. Formats 0.1–0.3
remain migration sources; 0.3 payloads without temporal fields decode as
static responses.

## Consequences

- `parity/support_n_a.toml` no longer blanket-blocks
  `ResponseCurve` / `InterventionResponse` on every temporal graph class;
  static (no temporal attachment) vs temporal cells are distinguished at
  classification time and by licensed/closed/n_a predicates.
- Provenance records, frozen conformance fixtures, staged-path tests,
  artifact round trips, and a hot-path bench are merge requirements for
  every newly licensed cell.
- Package version is 0.7.0.
