# Reach of measurement, composition, evidence and transport routes

A route is useful only where something can consume it. This page records, for
each route of the measurement, mechanism-composition, evidence-ledger and
counterfactual-transport packages, what inference mode it has and whether it
can feed each downstream consumer today. It describes the Rust facades in
`crates/antecedent/src/analysis/`; Python surfaces are not part of this table.

Every cell is exactly one state. "Not considered" is not a state.

| State | Meaning |
| --- | --- |
| **DIRECT** | The consumer takes the route's typed output with no translation; code and a test exist. |
| **ADAPTER** | An explicit adapter in code converts the output to what the consumer accepts, keeping the label. |
| **REFUSED** | The request is refused with a typed reason; the registered detail is named. |
| **NO ROUTE** | By design the request cannot be formed from this output, and the reason is stated; there is no typed refusal to emit. |
| **SCHEDULED** | Not connected today. The owner is the module that would hold the adapter; until then the consumer has no way to accept the output, and the value stays point-only. |

An ADAPTER or DIRECT cell never upgrades a label. A point stays a point, an
uncalibrated standard error stays labelled `uncalibrated_standard_error`, an
assumption range stays an assumption range, and no interval, posterior or
coverage statement exists for any row below: each interval route is closed
with `cell_not_licensed` until whole-method coverage is measured.

Consumers and their owning modules:

| Consumer | Owner of an adapter |
| --- | --- |
| decision | `analysis/decision_contract_facade.rs`, `analysis/decision_claims.rs` (antecedent-design decision contract) |
| inverse query | `analysis/inverse_query.rs` (finite-action, point-only) |
| sensitivity | `analysis/sensitivity_decision.rs` |
| design ranking | `analysis/design_ranking.rs` (antecedent-design) |
| counterfactual | `analysis/counterfactual_transport.rs`, `analysis/transported_counterfactual.rs` |
| transport | the transport routes under `analysis/` and the evidence catalog |
| temporal | the temporal facades under `analysis/` (`temporal_transport.rs`, temporal counterfactual) |
| composition | `analysis/composition.rs` and the composition bundle of antecedent-design |

## Measurement models

All five routes return a point with a delta-method standard error that
includes the uncertainty of the estimated measurement parameters. The
standard error is uncalibrated; `PreparedMeasurementModel::interval` and the
estimate's `interval` always refuse (`measurement_model.interval_closed`).
The route card (`antecedent_contract::measurement::route_card`) lists each
route's allowed and refused operations, and a refused operation reads
`cell_not_licensed` / `measurement_model.operation_not_licensed` with the
reason from the card.

| Route | Inference mode | Allowed downstream operation | Decision | Inverse query | Sensitivity | Design ranking | Counterfactual | Transport | Temporal | Composition |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| classical continuous error | point-only, uncalibrated SE | linear adjusted effect; linear functional of the corrected coefficients (`downstream_linear_functional`) | SCHEDULED | SCHEDULED | SCHEDULED | SCHEDULED | REFUSED (conditioning on the construct) | REFUSED (invariance premise) | REFUSED (time-varying error) | SCHEDULED |
| repeated measurements | point-only, uncalibrated SE | as above | SCHEDULED | SCHEDULED | SCHEDULED | SCHEDULED | REFUSED (conditioning on the construct) | REFUSED (invariance premise) | REFUSED (time-varying error) | SCHEDULED |
| binary misclassification | point-only, uncalibrated SE | latent prevalence; risk difference | SCHEDULED | SCHEDULED | SCHEDULED | SCHEDULED | REFUSED (per-unit conditioning) | REFUSED (invariance premise) | REFUSED (time-varying error) | SCHEDULED |
| probabilistic categorical | point-only, uncalibrated SE | latent prevalence; category outcome means | SCHEDULED | SCHEDULED | SCHEDULED | SCHEDULED | REFUSED (per-unit soft assignments) | REFUSED (invariance premise) | REFUSED (time-varying error) | SCHEDULED |
| external latent posterior | point-only (provider posterior in, Rubin or posterior-mean substitution out), uncalibrated SE | linear adjusted effect; linear functional | SCHEDULED | SCHEDULED | SCHEDULED | SCHEDULED | REFUSED (not an allowed operation) | REFUSED (bound to the posterior's population and regime) | REFUSED (time-varying error) | SCHEDULED |

Refusal details for this table: counterfactual and transport cells are
`measurement_model.operation_not_licensed` (the operations
`condition_on_construct` and `transport_to_other_population`); an
intervention on a representation's coordinates is
`measurement_model.operation_not_licensed` or
`measurement_model.learned_coordinates_not_intervenable`; error that differs
by population or time is `measurement_model.differential_axis_unsupported`.
Declared-invariance transport of a measurement model is not delivered; the
refusal names the invariance premise that would license it.

The artifact (`measurement_model_v1`) is consumed independently by
`consume_measurement_model_artifact`, whose `check_use` re-asks the contract
for the consumer's own population, regime, arm, representation and operation.
That is the only consumer connected today.

Scheduled owners: decision, inverse query, sensitivity and design ranking
accept the corrected functional only when an adapter keeps its
uncalibrated-standard-error label and its model identity; the modules in the
owner table above would hold it. The composition bundle needs a node kind for
a measurement result.

## Mechanism composition

The linear-Gaussian family is exact given the declared mechanisms; the
delta-method variance of a linear contrast is first-order, holds residual
scales fixed and is uncalibrated. Finite tables are exact enumeration with no
uncertainty propagation (`mechanism_composition.uncertainty_family_unsupported`
for a request to propagate it).

| Route | Inference mode | Decision | Inverse query | Sensitivity | Design ranking | Counterfactual | Transport | Temporal | Composition |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| interventional law (linear-Gaussian) | exact-given-declared (closed form) | SCHEDULED | SCHEDULED | SCHEDULED | SCHEDULED | REFUSED (conditioning is a separate query) | REFUSED unless admitted by a declared selection diagram or invariance | REFUSED (horizon mismatch, cycles) | SCHEDULED |
| mean contrast with variance | point-only + uncalibrated first-order variance | SCHEDULED | SCHEDULED | SCHEDULED | SCHEDULED | REFUSED (conditioning is a separate query) | REFUSED unless admitted | REFUSED (horizon mismatch, cycles) | SCHEDULED |
| structural samples | exact draws from the composed law, named noise streams | SCHEDULED | SCHEDULED | SCHEDULED | SCHEDULED | REFUSED (conditioning is a separate query) | REFUSED unless admitted | REFUSED (horizon mismatch, cycles) | SCHEDULED |
| finite intervention distribution | exact-finite (enumeration, truncated factorization) | SCHEDULED | SCHEDULED | SCHEDULED | SCHEDULED | REFUSED (conditioning is a separate query) | REFUSED unless admitted | REFUSED (horizon mismatch, cycles) | SCHEDULED |
| interval or posterior of a composed functional | closed | REFUSED | REFUSED | REFUSED | REFUSED | REFUSED | REFUSED | REFUSED | REFUSED |

Refusal details: a conditioning request or a query of the other family is
`mechanism_composition.request_unsupported`; a mechanism fitted in another
population or regime without a declared admission is
`mechanism_composition.population_undeclared_transport` or
`mechanism_composition.regime_undeclared_invariance`; a selection arrow into
an admitted variable is
`mechanism_composition.population_selection_arrow_into_mechanism`; a horizon
difference is `mechanism_composition.quantity_horizon_mismatch`; a directed
cycle is `mechanism_composition.cycle_directed_cycle`; an interval is
`mechanism_composition.interval_route_closed`.

Transport reach here is a declaration, not a derivation: the plan represents
the admission (and binds it into the plan identity) but does not execute a
transport derivation. Whether the composition's refusal and the exact
counterfactual transport obstruction agree on one fixture is not yet
established.

Connected consumers: `export` and `consume` of `mechanism_composition_v1`
replay the premises and one query to identical result bits.

Scheduled owners: as for measurement. The composition bundle's node kind for a
composed model is explicitly not wired. An inverse query over a composed
model's response needs a forward source with its own refusal.

## Evidence ledger

Every combination is point-only. The dependence range is an assumption
range and is labelled so. Nothing is an interval or posterior.

| Route | Inference mode | Decision | Inverse query | Sensitivity | Design ranking | Counterfactual | Transport | Temporal | Composition |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| statistical pooling (inverse variance, declared covariance) | point-only, model-based variance | SCHEDULED | SCHEDULED | SCHEDULED | SCHEDULED | NO ROUTE (a pooled summary carries no structural model) | SCHEDULED (a transported claim entering a pool with its identity is not composed) | SCHEDULED | SCHEDULED |
| random effects (DerSimonian-Laird, REML) | point-only, variance conditional on the estimated tau squared | SCHEDULED | SCHEDULED | SCHEDULED | SCHEDULED | NO ROUTE | SCHEDULED | SCHEDULED | SCHEDULED |
| hierarchical borrowing | point-only posterior mean and variance under the declared finite prior | SCHEDULED | SCHEDULED | SCHEDULED | SCHEDULED | NO ROUTE | SCHEDULED | SCHEDULED | SCHEDULED |
| dependence range | assumption range (not an interval) | SCHEDULED | SCHEDULED | SCHEDULED | SCHEDULED | NO ROUTE | SCHEDULED | SCHEDULED | SCHEDULED |
| transported claim registration | identity-bearing registration, never native data | SCHEDULED | SCHEDULED | SCHEDULED | SCHEDULED | NO ROUTE | DIRECT | SCHEDULED | SCHEDULED |
| evidence reuse, repeated observation use | accounting only (mass counted once, flagged `data_reuse`) | NO ROUTE (no numeric value) | NO ROUTE | NO ROUTE | NO ROUTE | NO ROUTE | NO ROUTE (accounting only) | NO ROUTE | SCHEDULED |
| pooled interval | closed | REFUSED | REFUSED | REFUSED | REFUSED | REFUSED | REFUSED | REFUSED | REFUSED |

The pooled interval refusal is `evidence_ledger.interval_not_licensed`. The
one connected consumer of the ledger is counted-table transport:
`overlap_declaration_from_ledger` derives the overlap declaration of count
tables from ledger sources, so that tables whose observations the ledger
knows are disjoint are independent, shared observations are shared units and
everything else is unknown. The transport identity of a transported claim is
supplied by the caller; the ledger refuses a source that would count as both
native and transported (`evidence_ledger.native_and_transported`).

Scheduled owners: a pooled value as a decision claim, a sensitivity input or a
design-ranking signal needs an adapter that carries the ledger result
identity and its `claim_class`; the composition bundle needs a ledger node
kind. A ledger source is not yet bound to a mechanism's evidence identity; the
mechanism composition holds evidence identity as a caller-supplied string.

## Counterfactual transport

The exact route is point-only on finite laws: exact finite-law evaluation with
no sampling uncertainty. Identification is sound and incomplete; a failure to
derive is `not_certified`, never a proof of non-transportability, and a
two-model witness that re-verifies by enumeration is the only obstruction.
The sampled route is a plug-in (or smoothed plug-in) point with an
uncalibrated delta-method standard error, or a structural range over a
declared unseen-mass family; the interval route is closed.

| Route | Inference mode | Decision | Inverse query | Sensitivity | Design ranking | Counterfactual | Transport | Temporal | Composition |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exact finite laws (identified) | exact-finite, point-only | SCHEDULED | SCHEDULED | SCHEDULED | SCHEDULED | DIRECT (nested and conditional counterfactual events, natural direct and indirect effects) | DIRECT (evidence catalog of population, regime, snapshot) | NO ROUTE (static graph class; temporal transport has its own rows) | SCHEDULED |
| proven obstruction | exact witness, no value | REFUSED | REFUSED | REFUSED | REFUSED | DIRECT (the witness re-verifies by enumeration) | DIRECT | REFUSED | SCHEDULED |
| sampled plug-in point | point-only, uncalibrated delta-method SE | SCHEDULED | SCHEDULED | SCHEDULED | SCHEDULED | DIRECT (evaluates the stored derivation unchanged) | DIRECT (evidence catalog of population, regime, snapshot) | NO ROUTE (static graph class; temporal transport has its own rows) | SCHEDULED |
| sampled unseen-mass envelope | structural range, no point | SCHEDULED | SCHEDULED | SCHEDULED | SCHEDULED | DIRECT | DIRECT | NO ROUTE (static graph class; temporal transport has its own rows) | SCHEDULED |
| sampled interval | closed | REFUSED | REFUSED | REFUSED | REFUSED | REFUSED | REFUSED | REFUSED | REFUSED |

Refusal details: counted, empirical or posterior laws given to the exact
route are `counterfactual_transport.sampled_inference_closed`; a cyclic graph
is `counterfactual_transport.graph_outside_contract`; a selected-sample law is
`counterfactual_transport.selected_sample_law`; an interval from the sampled
route is `counterfactual_transport_sampled.interval_not_licensed`; an
undeclared table relation is `counterfactual_transport_sampled.overlap_unknown`.

Scheduled owners: a transported response as a decision claim or as a forward
source for the inverse query, with a typed refusal when the value is
point-only and the request needs a law or interval; an assumption range for
selection-diagram violation through the sensitivity artifact surface; and a
composition bundle node kind. Temporal transport (time-period shift) has its
own rows and is not delivered by this route. The general path-specific
transport row stays carried forward (see
[the carryover disposition](https://github.com/iridae-dev/antecedent/blob/main/parity/carryover_disposition.md)).

## Reading the table

- A SCHEDULED cell is not a promise of a date. It records that the consumer
  has no way to accept this output today and which module owns the adapter.
- Frequentist and Bayesian modes are not licensed for any row: each would need
  a measured whole-method coverage record, and every such route is closed with
  a typed refusal until then.
- The refusals named above, and their remedies, are listed in
  [refusal and partial knowledge](refusal-and-partial-knowledge.md).
- The identities each route freezes are in
  [composition identities](architecture/composition-identities.md).
