# Bring in external evidence and synthesize studies

Record which evidence each result used, combine studies without counting anything twice, and see what a revision downstream changes. Evidence that re-enters an analysis is not new evidence. The ledger records
which observations, experiments, priors, fitted models and external claims
contributed to each result, how they overlap, and what changed afterwards.
Overlap that is not declared stays unknown; independence is never assumed.

## Boundary

| Item | Supported | Refused |
| --- | --- | --- |
| Sources | observation sets, experiments, priors, fitted models, external claims, with row-level observation identities and a derivation lineage | observations that are not declared are *unknown*, not independent |
| Overlap | independent (known, disjoint), shared (counted once, with repeated slots reported), unknown | pooling shared or unknown sources as independent |
| Re-entry | duplicate, transform, subset, prior plus likelihood, external claim built from used data are detected from lineage and rows | silent double counting: `evidence_double_counted` |
| Combinations | statistical pooling, Bayesian borrowing, causal transport, evidence reuse, repeated observation use (five distinct operations) | an operation does not imply another |
| Transport | a transported claim carries its source populations, selection-diagram digest and assumptions | counting one source as native target data and as transported |
| Revision | add, update, replace, withdraw, supersede, revalidate-downstream, each with an identity and a downstream classification | using a withdrawn or superseded source, citing a stale result |
| Uncertainty | point values; an assumption range for unknown dependence; a 95% pooled-mean interval reported with its calibration status | a posterior, or an interval for a result that has no interval construction (a borrowing posterior, a dependence range) |

## Mass accounting

A result counts each observation once. For sources with `n_i` declared
observations and a union of `u`, the account reports `sum n_i` declared slots,
`u` distinct observations and `sum n_i - u` repeated slots. A duplicate adds
slots but no observations. A source with no declared observations cannot be
accounted and is listed as unknown.

## Combinations

| Operation | Needs | Supported when |
| --- | --- | --- |
| Statistical pooling, independent | two or more sources | every pair is known and disjoint |
| Statistical pooling, declared covariance | summaries and a full covariance | the covariance has cross terms (shared or unknown overlap), or the sources are known and disjoint |
| Statistical pooling, assumption range | a correlation interval inside `(-1/(k-1), 1)` | always; the result is an assumption range |
| Bayesian borrowing | priors (kind `prior`) and likelihood sources | likelihood sources pairwise independent and every prior's data origin disjoint from them |
| Causal transport | transported claims (with identity) and native evidence (without) | native and transported evidence share no observations |
| Evidence reuse | a valid existing result | the result is not stale, withdrawn or replaced |
| Repeated observation use | sources with known observations and a named stage | always; the result is flagged `data_reuse` |

A prior elicited without data is declared with an empty observation set; it
overlaps nothing, including undeclared evidence.

## Revision

Each operation is atomic: a refused request leaves the ledger unchanged. The
operation identity chains the previous operation, so the head identity binds
the whole history.

- `update` only grows a source; `replace` corrects it; `supersede` replaces a
  fitted model or an external claim with a newer one; `withdraw` retracts it.
- A result is classified `still_valid`, `recompute` (an input changed and a
  successor exists) or `withdrawn` (an input it cannot do without is gone).
  Dependence flows through source lineage, so a model built from a withdrawn
  source is lost with it.
- A recomputation names the result it replaces; the original is preserved and
  unusable.

## Summary-data numerics

The synthesis numerics compute the numbers the ledger supports:

- generalized least squares pooling with a full covariance, `1'S^-1 y / 1'S^-1 1`
  with variance `1 / 1'S^-1 1`; a non positive semidefinite covariance is
  refused, a singular one has no weights;
- random effects by DerSimonian-Laird moments or restricted maximum
  likelihood, with Cochran's `Q`, `tau^2` and `I^2`;
- normal-normal hierarchical borrowing with a declared finite prior on
  `tau`, evaluated as an exact finite mixture;
- an exact envelope of the pooled estimate and variance over a declared range
  of one common correlation. The extremes may be interior (the variance of two
  studies peaks at `rho = s_min / s_max`); the stationarity quadratic is
  solved exactly. The envelope is labelled `assumption_range`.

## Facade and lifecycle

`StudyBuilder::evidence_ledger` returns a session that holds the ledger and the
numbers computed over it. Each combination (`pool_summaries`,
`random_effects`, `hierarchical_borrowing`, `dependence_range`,
`transport_claim`, `reuse_result`, `repeat_observations`) is registered with
the ledger first, on a staged copy, so shared or unknown overlap and a retired
source are refused before any number; the numbers are then computed and the staged ledger is committed only when
both succeed. `revise_source` applies add, update, replace, supersede and
withdraw and returns the downstream classification. `synthesis` refuses a
result that is no longer valid, so stale numbers are never served, and
`pooled_interval` reports a 95% interval with its calibration status, which is
`calibrated` only when a current record covers its exact construction and scope.
Fixed-effect Wald and HKSJ intervals have separate calibration keys. An HKSJ
interval at a study count other than 20, 25 or 30 is annotated `not_measured`: a
strong-heterogeneity five- or fifteen-study record cannot attest an unknown study family.

For independent Gaussian study estimates sharing one marginal mean, use the
equal-study Gaussian workflow: Rust `pool_gaussian_studies(...)`, followed by
`pooled_interval(result)`. In Python, select
`random_effects(..., method="equal_study_gaussian")`, then
`ledger.pooled_interval(outcome.value.result)`. Study variances may be unequal and unknown;
independent Gaussian random effects satisfy the common marginal mean model.
Unequal fixed means and dependent study sources do not. Equal study weights,
empirical study-level variance and a Student reference form the distinct
`independent_gaussian_study_t` construction. Its coverage design spans
moderate-heterogeneity designs at 5/10/15/20/25/30 studies and
homogeneous and strong-heterogeneity designs at 5/15/30, at per-study sizes 75/100/200.
Its records are separate from the HKSJ records and do not touch the DL/REML numerical recipes.

Calling DL/REML `random_effects` without the Gaussian option selects the DL/REML
construction; there is no automatic fallback between constructions.
A result with no numbers refuses; a borrowing posterior or an assumption range has no
interval construction and refuses `route_not_supported`
(`evidence_ledger.interval_construction_unavailable`). Fewer than 5 or more than 30 studies,
or a construction without a current matching record, still reports the interval, annotated
`not_measured` or `outside_measured_scope`.


`refresh` with added or retired evidence:

| | |
|---|---|
| keeps | every result whose sources are untouched (`still_valid`) with its numbers; the operation history, chained onto the head identity |
| recomputes | nothing automatically: a result whose input changed is classified `recompute` and is refused for use until a combination over the replacing sources is registered with `replaces` set |
| refuses | the whole batch when one change is refused by the ledger (all-or-nothing); a withdrawn result's numbers; any use of a retired source |

The session a refresh is called on is untouched either way. A hierarchical
borrowing over prior sources is accounted in the ledger and computed in the
session, but the stored synthesis format carries one summary per source, so
the export lists such a result in `not_carried` instead of dropping it.

## Artifact

`evidence_ledger_v1` stores every operation with its declaration, identity and
effect, the head, the state digest and optional syntheses bound to ledger
results. A consumer replays the operations through the ledger, recomputes the
syntheses and refuses a stored identity, effect, value, covariance or range
that does not replay, including a consistently resealed edit.

## Python surface

`antecedent.evidence_ledger.EvidenceLedger` is the session: `add_source`
(a `SourceDeclaration` with `rows(domain, keys)` observation identities and a
`Lineage`), `account_observations`, `pool_summaries`, `random_effects`,
`hierarchical_borrowing`, `dependence_range`, `transport_claim`,
`reuse_result`, `repeat_observations`, `revise_source`, `synthesis`,
`refresh` (returns a new session and leaves this one untouched) and `export`.
A refused request raises `EvidenceLedgerRefusal` with its registered
`reason_code` and `evidence_ledger.*` `detail`, and leaves the session as it
was; `pooled_interval` returns the interval with its calibration slot (naming
the coverage record when one attests), and raises `EvidenceLedgerRefusal` for a
result with no interval construction.
`consume_evidence_ledger_artifact` replays the operations, recomputes the
syntheses and refuses another retained identity.

### Python persistence and composition

`ledger.export(...)` returns a typed `ExportedArtifact`: `bytes(artifact)` are the container,
`.expectation` the identity to retain, and carried/not-carried export receipts sit in immutable
`.metadata`; no summary silently becomes a replayable synthesis.
`ant.load(bytes(artifact), expected=artifact.expectation)` replays it through the ledger
consumer. Retain the expectation separately from the container.

A scalar `SynthesisOutcome` can use `outcome.as_mean_source(quantity=...)` for
decision or inverse-query composition. Explicitly declare the pooled quantity's units,
population and regime. Evidence and lineage remain attached; an assumption range without
a point, or a pooled interval without support, cannot become a mean or a law.

The ledger records which evidence a result used, not which methods produced it. Each
analysis result that enters the ledger carries the provenance of its own route
(`outcome.method_provenance`: the published sources and the documented deviations from
them), and `antecedent.citations.bibliography(*results)` lists the sources of a whole
study once. See [cite the methods behind a study](guides/citing-a-study.md).
