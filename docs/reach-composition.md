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
A model may be carried to another population only through
`antecedent_estimate::measurement_transport::transport_measurement_model`
under an explicit invariance declaration naming each error parameter the route
uses: the premise is a declaration and is never tested, the correction stays
point-only with its uncalibrated standard error, a differential population
needs the target population's own model, and the replicate route and an
external posterior (bound to its source snapshot) have nothing to carry. Every
route above still refuses the transport operation without that declaration,
and the refusal names the premise that would license it.

The artifact (`measurement_model_v1`) is consumed independently by
`consume_measurement_model_artifact`, whose `check_use` re-asks the contract
for the consumer's own population, regime, arm, representation and operation.
That is the only consumer connected today.

Owners: decision, inverse query, sensitivity and design ranking
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

A selection arrow into a mechanism's variable (or into a latent cause it loads
on) is the statement that the mechanism differs, so it dominates a declared
invariance for the same variable; an invariance admits only what the diagram
is silent about. `antecedent_model::mechanism_transport::transport_linear_gaussian`
carries fitted additive-noise linear-Gaussian mechanisms of a source population
under that rule (a natively refitted target mechanism always wins and is never
averaged; finite tables are not carried and refuse with a pointer to the finite
counterfactual transport derivation). The composition refusal and the finite
counterfactual obstruction agree on the same fixtures (the obstruction
agreement test of antecedent-model), including a selection arrow into a shared
latent. The contrast variance of a carried composition is the first-order
source-fit variance only, labelled
`fitted_parameters_only_transport_assumption_not_propagated`; transport adds
no variance and no interval exists.

Connected consumers: `export` and `consume` of `mechanism_composition_v1`
replay the premises and one query to identical result bits.

A composed mechanism model reaches an inverse query through
`PreparedMechanismComposition::inverse_forward_source`: a linear-Gaussian
composition supplies a replayed mean per action (a quantile or probability is
refused `inverse_forward_source.composed_law_not_materialized`, an interval
`inverse_forward_source.interval_not_licensed`), a finite-table composition
supplies the exact law of each action under the caller's coding of outcome
levels, and a composition with admitted mechanisms is a transported claim: it
needs the carrying receipt and an input policy that accepts a transported
input (`inverse_forward_source.transported_not_accepted`,
`cell_not_licensed`, otherwise), and its support label is extrapolative at
best. A composition can also be discretized into an exact finite SCM with a
computed mean-contrast error bound
(`PreparedMechanismComposition::discretize`; point-only, `discretized_scm.point_only`
for an interval or posterior). Declared independence between mechanisms is
decided from ledger rows by
`antecedent_contract::check_declared_independence_against_ledger`
(`mechanism_composition.posterior_dependence_evidence_overlap_unknown` when a
row set is undeclared).

Owners: as for measurement. The composition bundle's node kind for a
composed model is not defined, so a bundle cannot carry one. A composed response as a decision
claim stays a declared `MeanSource`; no decision-side composed adapter exists.

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

Owners: a pooled value as a decision claim, a sensitivity input or a
design-ranking signal needs an adapter that carries the ledger result
identity and its `claim_class`; the composition bundle needs a ledger node
kind. A mechanism's evidence id is read as a ledger source identifier by
`check_declared_independence_against_ledger`; a non-ledger id is unknown
evidence, never independent.

Transport-then-synthesize is delivered by
`antecedent_design::transported_synthesis::synthesize_transported` over the
ledger: a native claim and claims transported into the same target population
pool (generalized least squares with each transport's declared error counted
once through its identity), borrow (hierarchical, a transport that serves
several claims refused) or serve as the prior of a native likelihood. The
result is a model variance under the declared transport variances, never an
interval; an undeclared transport variance is refused
`transported_synthesis.transport_variance_undeclared` and the same
observations as native and transported are refused by the ledger.

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

A transported response (a value with a receipt of route, populations,
transport identities, assumptions and an extrapolative support label) is a
decision claim through
`BoundDecisionContract::evaluate_transported` and a forward source for the
inverse query through `antecedent::analysis::inverse_sources`; a point answers
an affine utility or a target mean and refuses a law
(`transported_decision.law_not_supplied`,
`inverse_forward_source.point_only_law_requested`) and an interval
(`cell_not_licensed`), and the default policy refuses the label
(`transported_decision.transported_not_accepted`). A bounded violation of the
transport premise gives an assumption range through the sensitivity artifact
surface (`transport_sensitivity`; additive or proportional shift, an
assumption range on a declared bound, never composed with a sampling
interval), and a changed source, diagram, target or calibration record is
classified still valid, recompute or withdrawn by `transport_revalidation`
(a calibration established at another target is withdrawn, never inherited).

| Route | Inference mode | Decision | Inverse query | Sensitivity | Design ranking | Counterfactual | Transport | Temporal | Composition |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| transported response, point (cohort effect, policy, synthesis, declared) | point-only, extrapolative, receipt carried | DIRECT (`evaluate_transported`; law and interval REFUSED) | DIRECT under an accepting input policy; REFUSED `transported_not_accepted` otherwise | DIRECT (assumption range) | SCHEDULED | NO ROUTE | DIRECT (revalidation) | SCHEDULED | SCHEDULED |
| transported response, law | transported law, receipt carried | DIRECT (probability criterion; interval REFUSED) | DIRECT under an accepting input policy | REFUSED (`transport_sensitivity.law_response_not_scalar`) | SCHEDULED | NO ROUTE | DIRECT (revalidation) | SCHEDULED | SCHEDULED |
| cohort effect under a declared selection shift | point-only, labelled extrapolative | ADAPTER (declared mean source) | ADAPTER (`cohort_forward_source`; accepting policy) | DIRECT | SCHEDULED | NO ROUTE | DIRECT | SCHEDULED | SCHEDULED |
| transported policy value (initial-state law shifted, selection diagram declared) | point-only, exact finite enumeration | ADAPTER (value as a declared mean) | ADAPTER | REFUSED | SCHEDULED | NO ROUTE | DIRECT (library; Python `transported_constructors.from_policy`) | REFUSED (use the period route) | SCHEDULED |
| period shift of a policy value (fixed horizon, time-invariant mechanisms) | point-only | ADAPTER | ADAPTER | REFUSED | SCHEDULED | NO ROUTE | REFUSED (`temporal_policy_transport.mechanism_time_varying`) | DIRECT (library; Python `transported_constructors.from_period_shift`) | SCHEDULED |
| adaptive trial as a transport source (recorded-probability standardization) | point-only, uncalibrated, no variance | ADAPTER (declared mean contrast) | REFUSED | REFUSED | REFUSED | NO ROUTE | DIRECT (library, finite strata; Python `transported_constructors.from_adaptive_trial`) | REFUSED | SCHEDULED |
| discovered CPDAG class | structural envelope, never a point or interval | REFUSED | REFUSED | REFUSED (`transport_sensitivity.no_scalar_family`) | REFUSED | NO ROUTE | DIRECT (library and Python `transport.discovered.transport_across_class`; PAG refused `discovered_transport.pag_members_not_causal_graphs`) | NO ROUTE | SCHEDULED |
| provider output as source evidence | point-only or unmeasured, never native target data, never measured | not applicable (source evidence) | not applicable | not applicable | not applicable | NO ROUTE | DIRECT (admission; attested never relabelled verified) | NO ROUTE | SCHEDULED |
| discretized composition | point-only with a computed mean-contrast bound | REFUSED (`discretized_scm.point_only` for an interval) | REFUSED | REFUSED | REFUSED | NO ROUTE | NO ROUTE | NO ROUTE | SCHEDULED |
| whole-path cross-world spread (finite Markovian row) | uncalibrated spread, never added across estimands | REFUSED (`cross_world_execution.interval_route_closed`) | REFUSED | REFUSED | REFUSED | DIRECT | REFUSED (`cross_world_uncertainty.transported_path_closed`) | NO ROUTE | SCHEDULED |

Temporal transport of a policy value is delivered by
`temporal_policy_transport`, not by the counterfactual route above. None of
these outputs is a composition bundle node kind, so a bundle cannot carry them. The general path-specific transport row stays carried
forward (see
[the carryover disposition](https://github.com/iridae-dev/antecedent/blob/main/parity/carryover_disposition.md)).

## Derived uncertainty, transported vertical and discovery routes

These rows describe the library and Python entries of the derived-uncertainty,
transported-measurement, transported-mechanism, transported-cohort and
discovery routes. The first column after the inference mode names the own
consumer recorded in the Rust tables of the
[producer and consumer matrix](producer-consumer-matrix.md), with the executing
test that backs each cell. No row carries an interval or a coverage statement.

Minimum-DIRECT floor: every producer in those tables has at least one DIRECT
cell with an executing test, or a documented REFUSED cell where no consumer in
code takes the output. The transported spread meets the floor with a DIRECT
cell (the propagation consumer replays both the covariance and the aligned
spread to identical bits) and REFUSED cells for the decision-reliability,
mechanism-composition and inverse-query consumers. The repeated-measures report
is a terminal derived statement and meets the floor with a documented REFUSED
cell: it has no decision-reliability route.

| Route | Inference mode | Own consumer | Decision | Inverse query | Sensitivity | Design ranking | Counterfactual | Transport | Temporal | Composition |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| composed spread (linear moments, aligned functional) | uncalibrated spread, derived and unmeasured | DIRECT (`propagation_artifact`: rebuilt from its bytes under the retained identity) | NO ROUTE (a spread is not a utility input) | NO ROUTE | NO ROUTE | NO ROUTE | NO ROUTE | NO ROUTE | NO ROUTE | SCHEDULED |
| composed exceedance (probability, band, verdict; known-weight mixture) | uncalibrated spread, derived and unmeasured | DIRECT (`propagation_artifact`) | NO ROUTE (no utility reading) | NO ROUTE | NO ROUTE | NO ROUTE | NO ROUTE | NO ROUTE | NO ROUTE | SCHEDULED |
| transported spread (source sampling, target sampling, transport functional) | uncalibrated spread, all three pieces and their dependence named | DIRECT (`propagation_artifact`: replays to identical bits); REFUSED for decision-reliability, composition and inverse query | NO ROUTE | NO ROUTE | NO ROUTE | NO ROUTE | NO ROUTE | NO ROUTE | NO ROUTE | SCHEDULED |
| composed interval | closed until a measured record exists at the composed coordinate and construction | REFUSED (`published_route`: `cell_not_licensed`, with the mismatch named) | REFUSED | REFUSED | REFUSED | REFUSED | REFUSED | REFUSED | REFUSED | REFUSED |
| derived decision uncertainty (numerical, selection, rollout components) | derived and unmeasured, components never combined | DIRECT (`decision_reliability_derived`: re-derived from the carried draws); REFUSED when published | DIRECT (assessed on the bound decision's own draws) | NO ROUTE | NO ROUTE | NO ROUTE | NO ROUTE | NO ROUTE | NO ROUTE | SCHEDULED |
| repeated-measures or clustered procedure report | exact forms plus seeded simulation, derived and unmeasured | REFUSED (`decision_reliability_derived`: no route) | NO ROUTE | NO ROUTE | NO ROUTE | NO ROUTE | NO ROUTE | NO ROUTE | NO ROUTE | SCHEDULED |
| transported cohort effect (facade over the fitted-effect transport) | point-only, extrapolative, typed receipt | DIRECT (`decision`: the existing cohort-mean source) | DIRECT (interval and law REFUSED) | ADAPTER (`cohort_forward_source`; accepting policy) | DIRECT | NO ROUTE | NO ROUTE | DIRECT (revalidation) | NO ROUTE | SCHEDULED |
| transported measurement estimate (carried or target-established error rates) | point-only, uncalibrated, typed receipt | DIRECT (`measurement_artifact`: the existing artifact replays) | SCHEDULED | SCHEDULED | SCHEDULED | SCHEDULED | NO ROUTE (an error-rate invariance is not a counterfactual claim) | DIRECT (library) | NO ROUTE | SCHEDULED |
| transported mechanism composition (carried or refitted linear-Gaussian mechanisms) | point-only with a labelled parameter-only variance | DIRECT (`mechanism_composition_artifact`: replays to the same bits) | SCHEDULED | ADAPTER (`inverse_forward_source` of the composition) | SCHEDULED | SCHEDULED | REFUSED (conditioning is a separate query) | DIRECT (library; a selection arrow into a carried mechanism refused) | NO ROUTE | SCHEDULED |
| per-population discovery | structure only, nothing pooled, differences descriptive | DIRECT (`claim_set`: one claim per population) | NO ROUTE (a structure has no response) | NO ROUTE | NO ROUTE | NO ROUTE | NO ROUTE | REFUSED (`selective_discovery.differences_are_not_selection_targets`) | NO ROUTE | SCHEDULED |
| sample-split discovery | structure on the discovery part; analysis part bound by a receipt | DIRECT (`evidence_ledger`: two disjoint sources; reuse refused) | NO ROUTE | NO ROUTE | NO ROUTE | NO ROUTE | NO ROUTE | NO ROUTE | NO ROUTE | SCHEDULED |

## Reading the table

- A SCHEDULED cell is not a promise of a date. It records that the consumer
  has no way to accept this output today and which module owns the adapter.
- Frequentist and Bayesian modes are not licensed for any row: each would need
  a measured whole-method coverage record, and every such route is closed with
  a typed refusal until then. A derived, unmeasured statement (a composed
  spread, a decision-reliability numerical or selection interval, a
  whole-procedure adaptive report) is labelled `uncalibrated_spread` or
  `derived_unmeasured` and is never a license: publishing it refuses
  `cell_not_licensed`.
- Which paradigm each route supports, and the typed closed row for a missing
  one (`paradigm_parity.paradigm_closed`), is the parity table of
  `antecedent_model::discretized_scm::PARITY_TABLE`.
- The refusals named above, and their remedies, are listed in
  [refusal and partial knowledge](refusal-and-partial-knowledge.md).
- The identities each route freezes are in
  [composition identities](architecture/composition-identities.md).
