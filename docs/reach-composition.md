# Reach of measurement, composition, evidence and transport routes

A route is useful only where something can consume it. This page records, for
each route of the measurement, mechanism-composition, evidence-ledger and
counterfactual-transport packages, what inference mode it has and whether it
can feed each downstream consumer. It describes the Rust facades in
`crates/antecedent/src/analysis/`; the Python entry of each cell is named in the
[producer and consumer matrix](producer-consumer-matrix.md), whose cells these rows follow.

Every cell is exactly one state. "Not considered" is not a state.

| State | Meaning |
| --- | --- |
| **DIRECT** | The consumer takes the route's typed output with no translation; code and a test exist. |
| **ADAPTER** | An explicit adapter in code converts the output to what the consumer accepts, keeping the label. The mean and inverse adapters declare a `MeanSource` coordinate (quantity, units, population, regime, functional) carrying the point, with the producer identity as provider; they answer an affine utility or a target mean, carry no standard error, variance, law or sample into the decision, and refuse a quantile or probability criterion (`decision_evaluation.mean_source_insufficient`). |
| **REFUSED** | The request is refused with a typed reason; the registered detail is named. |
| **NO ROUTE** | By design the request cannot be formed from this output, and the reason is stated; there is no typed refusal to emit. |
| **NOT CONNECTED** | No adapter connects this output to the consumer, so the consumer has no way to accept it and the value stays point-only. The owner is the module an adapter would belong in. |

An ADAPTER or DIRECT cell never upgrades a label. A point stays a point, an
uncalibrated standard error stays labelled `uncalibrated_standard_error`, an
assumption range stays an assumption range, and no posterior or coverage
statement exists for any row below: an interval route is supported only with an
independent check of its construction, and without one it refuses (the code is
named in the row). Two rows report an interval: the equal-tailed Wald interval
of a measurement estimate, from the point and its delta-method standard error,
and the pooled interval of an evidence-ledger result. Each carries its
calibration status, which says whether a coverage record measured that
construction; no consumer below takes either interval.

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
standard error is uncalibrated. `MeasurementEstimate::interval(level)` forms the
equal-tailed Wald interval `point -/+ z * standard_error` of the corrected functional
(`antecedent_estimate::measurement::corrected_functional_interval`); it refuses a level
outside (0, 1) (`measurement_model.interval_level`) and a record without a derived
standard error (`route_not_supported` / `measurement_model.interval_requires_estimate`).
`PreparedMeasurementModel::interval` always refuses with
`measurement_model.interval_requires_estimate`: a prepared model holds no corrected value
to put an interval around, so the interval is requested from the estimate. The Wald
interval is a delta-method construction, not a measured coverage statement.
The route card (`antecedent_contract::measurement::route_card`) lists each
route's allowed and refused operations, and a refused operation reads
`route_not_supported` / `measurement_model.operation_not_supported` with the
reason from the card.

| Route | Inference mode | Allowed downstream operation | Decision | Inverse query | Sensitivity | Design ranking | Counterfactual | Transport | Temporal | Composition |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| classical continuous error | point-only, uncalibrated SE | linear adjusted effect; linear functional of the corrected coefficients (`downstream_linear_functional`) | ADAPTER (declared mean of the latent quantity) | ADAPTER (the same mean as a claim) | REFUSED (`CausalTypeError`: not a `SensitivityArtifact`) | REFUSED (`CausalTypeError`: a point is neither prior nor signal) | REFUSED (conditioning on the construct) | REFUSED (invariance premise) | REFUSED (time-varying error) | REFUSED (`CompositionBundleRefusal`: no node kind) |
| repeated measurements | point-only, uncalibrated SE | as above | ADAPTER (declared mean of the latent quantity) | ADAPTER (the same mean as a claim) | REFUSED (`CausalTypeError`: not a `SensitivityArtifact`) | REFUSED (`CausalTypeError`: a point is neither prior nor signal) | REFUSED (conditioning on the construct) | REFUSED (invariance premise) | REFUSED (time-varying error) | REFUSED (`CompositionBundleRefusal`: no node kind) |
| binary misclassification | point-only, uncalibrated SE | latent prevalence; risk difference | ADAPTER (declared mean of the latent quantity) | ADAPTER (the same mean as a claim) | REFUSED (`CausalTypeError`: not a `SensitivityArtifact`) | REFUSED (`CausalTypeError`: a point is neither prior nor signal) | REFUSED (per-unit conditioning) | REFUSED (invariance premise) | REFUSED (time-varying error) | REFUSED (`CompositionBundleRefusal`: no node kind) |
| probabilistic categorical | point-only, uncalibrated SE | latent prevalence; category outcome means | ADAPTER (declared mean of the latent quantity) | ADAPTER (the same mean as a claim) | REFUSED (`CausalTypeError`: not a `SensitivityArtifact`) | REFUSED (`CausalTypeError`: a point is neither prior nor signal) | REFUSED (per-unit soft assignments) | REFUSED (invariance premise) | REFUSED (time-varying error) | REFUSED (`CompositionBundleRefusal`: no node kind) |
| external latent posterior | point-only (provider posterior in, Rubin or posterior-mean substitution out), uncalibrated SE | linear adjusted effect; linear functional | ADAPTER (declared mean of the latent quantity) | ADAPTER (the same mean as a claim) | REFUSED (`CausalTypeError`: not a `SensitivityArtifact`) | REFUSED (`CausalTypeError`: a point is neither prior nor signal) | REFUSED (not an allowed operation) | REFUSED (bound to the posterior's population and regime) | REFUSED (time-varying error) | REFUSED (`CompositionBundleRefusal`: no node kind) |

Refusal details for this table: counterfactual and transport cells are
`measurement_model.operation_not_supported` (the operations
`condition_on_construct` and `transport_to_other_population`); an
intervention on a representation's coordinates is
`measurement_model.operation_not_supported` or
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
and the refusal names the premise that would support it.

The artifact (`measurement_model_v1`) is consumed independently by
`consume_measurement_model_artifact`, whose `check_use` re-asks the contract
for the consumer's own population, regime, arm, representation and operation.
That is the only artifact consumer of the model.

Decision and inverse query take the corrected point through the mean adapters
(`MeasurementEstimate` to a `MeanSource` coordinate; the inverse query takes the same mean as a
`MeanClaim`): the caller declares the latent quantity's units, population and regime, the
adapter retains the model identity as provider and the data identity as snapshot, the standard
error stays behind, and an affine utility or a target mean is answered. Sensitivity and design
ranking refuse the point (it is not a `SensitivityArtifact`, a prior or a signal). The
composition bundle has no node kind for a measurement result.

## Mechanism composition

The linear-Gaussian family is exact given the declared mechanisms; the
delta-method variance of a linear contrast is first-order, holds residual
scales fixed and is uncalibrated. Finite tables are exact enumeration with no
uncertainty propagation (`mechanism_composition.uncertainty_family_unsupported`
for a request to propagate it).

| Route | Inference mode | Decision | Inverse query | Sensitivity | Design ranking | Counterfactual | Transport | Temporal | Composition |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| interventional law (linear-Gaussian) | exact-given-declared (closed form) | ADAPTER (declared means under each action; the law is not carried) | ADAPTER (`inverse_forward_source`: replayed mean per action) | REFUSED (`CausalTypeError`: not a `SensitivityArtifact`) | REFUSED (`CausalTypeError`: not a `design.StatePrior`) | REFUSED (conditioning is a separate query) | REFUSED unless admitted by a declared selection diagram or invariance | REFUSED (horizon mismatch, cycles) | REFUSED (`CompositionBundleRefusal`: no node kind) |
| mean contrast with variance | point-only + uncalibrated first-order variance | ADAPTER (declared means; the variance is not carried) | ADAPTER (declared means as a `MeanClaim`) | REFUSED (`CausalTypeError`: not a `SensitivityArtifact`) | REFUSED (`CausalTypeError`: not a `design.StatePrior`) | REFUSED (conditioning is a separate query) | REFUSED unless admitted | REFUSED (horizon mismatch, cycles) | REFUSED (`CompositionBundleRefusal`: no node kind) |
| structural samples | exact draws from the composed law, named noise streams | NOT CONNECTED | NOT CONNECTED | NOT CONNECTED | NOT CONNECTED | REFUSED (conditioning is a separate query) | REFUSED unless admitted | REFUSED (horizon mismatch, cycles) | NOT CONNECTED |
| finite intervention distribution | exact-finite (enumeration, truncated factorization) | ADAPTER (declared means; the law is not carried) | ADAPTER (`inverse_forward_source`: exact law of each action under the caller's coding) | REFUSED (`CausalTypeError`: not a `SensitivityArtifact`) | REFUSED (`CausalTypeError`: not a `design.StatePrior`) | REFUSED (conditioning is a separate query) | REFUSED unless admitted | REFUSED (horizon mismatch, cycles) | REFUSED (`CompositionBundleRefusal`: no node kind) |
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
`antecedent_model::typed_mechanism_transport::transport_typed` extends the same
rule to nonlinear and non-Gaussian mechanisms: an invariant mechanism is carried,
a shifted one is used only from a target fit and refused without one, and a
declared invariance that target data reject is refused. See the
[transport guide](guides/mechanism-and-synthesis-transport.md#nonlinear-and-non-gaussian-mechanism-transport).

Connected consumers: `export` and `consume` of `mechanism_composition_v1`
replay the premises and one query to identical result bits.

A composed mechanism model reaches an inverse query through
`PreparedMechanismComposition::inverse_forward_source`: a linear-Gaussian
composition supplies a replayed mean per action (a quantile or probability is
refused `inverse_forward_source.composed_law_not_materialized`, an interval
`inverse_forward_source.interval_not_supported`), a finite-table composition
supplies the exact law of each action under the caller's coding of outcome
levels, and a composition with admitted mechanisms is a transported claim: it
needs the carrying receipt and an input policy that accepts a transported
input (`inverse_forward_source.transported_not_accepted`,
`not_supported`, otherwise), and its support label is extrapolative at
best. A composition can also be discretized into an exact finite SCM with a
computed mean-contrast error bound
(`PreparedMechanismComposition::discretize`; point-only, `discretized_scm.point_only`
for an interval or posterior). Declared independence between mechanisms is
decided from ledger rows by
`antecedent_contract::check_declared_independence_against_ledger`
(`mechanism_composition.posterior_dependence_evidence_overlap_unknown` when a
row set is undeclared).

Decision takes a composed response only as declared `MeanSource` coordinates (the outcome mean
under each action, with the plan identity as provider; `prepared.as_mean_source(quantity=...,
contrast=...)` requires the scientific coordinate and the selected contrast), so an affine
utility is answered and a quantile or probability criterion is refused; no decision-side adapter
carries the composed law. The composition bundle has no node kind for a composed model, so a
bundle cannot carry one. Structural samples reach no consumer beyond their own replay.

## Evidence ledger

Every combination is a point except the pooled interval, which is reported
with its calibration status. The dependence range is an assumption range and is
labelled so. Nothing is a posterior.

| Route | Inference mode | Decision | Inverse query | Sensitivity | Design ranking | Counterfactual | Transport | Temporal | Composition |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| statistical pooling (inverse variance, declared covariance) | point-only, model-based variance | ADAPTER (declared pooled quantity as a `MeanSource`) | ADAPTER (the same mean as a `MeanClaim`) | REFUSED (`CausalTypeError`: not a `SensitivityArtifact`) | REFUSED (`CausalTypeError`: not a `design.StatePrior`) | NO ROUTE (a pooled summary carries no structural model) | NOT CONNECTED (a transported claim entering a pool with its identity is not composed) | NOT CONNECTED | REFUSED (`CompositionBundleRefusal`: no node kind) |
| random effects (DerSimonian-Laird, REML) | point-only, variance conditional on the estimated tau squared | ADAPTER (declared pooled quantity as a `MeanSource`) | ADAPTER (the same mean as a `MeanClaim`) | REFUSED (`CausalTypeError`: not a `SensitivityArtifact`) | REFUSED (`CausalTypeError`: not a `design.StatePrior`) | NO ROUTE | NOT CONNECTED | NOT CONNECTED | REFUSED (`CompositionBundleRefusal`: no node kind) |
| hierarchical borrowing | point-only posterior mean and variance under the declared finite prior | ADAPTER (declared pooled quantity as a `MeanSource`) | ADAPTER (the same mean as a `MeanClaim`) | REFUSED (`CausalTypeError`: not a `SensitivityArtifact`) | REFUSED (`CausalTypeError`: not a `design.StatePrior`) | NO ROUTE | NOT CONNECTED | NOT CONNECTED | REFUSED (`CompositionBundleRefusal`: no node kind) |
| dependence range | assumption range (not an interval) | REFUSED (`CausalTypeError`: an assumption range has no point) | REFUSED (an envelope has no mean) | REFUSED (`CausalTypeError`: not a `SensitivityArtifact`) | REFUSED (`CausalTypeError`: not a `design.StatePrior`) | NO ROUTE | NOT CONNECTED | NOT CONNECTED | REFUSED (`CompositionBundleRefusal`: no node kind) |
| transported claim registration | identity-bearing registration, never native data | NOT CONNECTED | NOT CONNECTED | NOT CONNECTED | NOT CONNECTED | NO ROUTE | DIRECT | NOT CONNECTED | NOT CONNECTED |
| evidence reuse, repeated observation use | accounting only (mass counted once, flagged `data_reuse`) | NO ROUTE (no numeric value) | NO ROUTE | NO ROUTE | NO ROUTE | NO ROUTE | NO ROUTE (accounting only) | NO ROUTE | NOT CONNECTED |
| pooled interval (`EvidenceLedger.pooled_interval`: Wald `z` for fixed-effect and declared-covariance pooling, Hartung-Knapp-Sidik-Jonkman `t` for random effects) | interval with its calibration status; refused (`route_not_supported`) for hierarchical borrowing and a dependence range | NO ROUTE (the mean adapters carry the pooled point, not the interval) | NO ROUTE | NO ROUTE | NO ROUTE | NO ROUTE | NO ROUTE | NO ROUTE | NO ROUTE |

The pooled interval is `EvidenceLedger.pooled_interval`, reported with its calibration
status. The
one connected consumer of the ledger is counted-table transport:
`overlap_declaration_from_ledger` derives the overlap declaration of count
tables from ledger sources, so that tables whose observations the ledger
knows are disjoint are independent, shared observations are shared units and
everything else is unknown. The transport identity of a transported claim is
supplied by the caller; the ledger refuses a source that would count as both
native and transported (`evidence_ledger.native_and_transported`).

A `point_only` pooled value is a decision claim and an inverse-query claim through the mean
adapters: the caller declares the pooled quantity (the ledger carries no estimand, units or
population) as one `MeanSource` coordinate, with the result identity as provider and the ledger
head as snapshot; an `assumption_range` envelope has no point and is refused, and the pooled
interval is not carried into a decision. A sensitivity input or a
design-ranking signal takes no pooled value, and the composition bundle has no ledger node kind
(its export preserves the carried and not-carried receipts). A mechanism's evidence id is read as a ledger source identifier by
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
declared unseen-mass family; the row has no interval route.

| Route | Inference mode | Decision | Inverse query | Sensitivity | Design ranking | Counterfactual | Transport | Temporal | Composition |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exact finite laws (identified) | exact-finite, point-only | ADAPTER (declared means of the outcome indicator under each action; an affine utility only) | ADAPTER (the same means as a `MeanClaim`) | REFUSED (`CausalTypeError`: not a `SensitivityArtifact`) | REFUSED (`CausalTypeError`: not a `design.StatePrior`) | DIRECT (nested and conditional counterfactual events, natural direct and indirect effects) | DIRECT (evidence catalog of population, regime, snapshot) | NO ROUTE (static graph class; temporal transport has its own rows) | REFUSED (`CompositionBundleRefusal`: no node kind) |
| proven obstruction | exact witness, no value | REFUSED | REFUSED | REFUSED | REFUSED | DIRECT (the witness re-verifies by enumeration) | DIRECT | REFUSED | NOT CONNECTED |
| sampled plug-in point | point-only, uncalibrated delta-method SE | ADAPTER (the same declared means; the standard error is not carried) | ADAPTER (the same means as a `MeanClaim`) | REFUSED (`CausalTypeError`: not a `SensitivityArtifact`) | REFUSED (`CausalTypeError`: not a `design.StatePrior`) | DIRECT (evaluates the stored derivation unchanged) | DIRECT (evidence catalog of population, regime, snapshot) | NO ROUTE (static graph class; temporal transport has its own rows) | REFUSED (`CompositionBundleRefusal`: no node kind) |
| sampled unseen-mass envelope | structural range, no point | NOT CONNECTED | NOT CONNECTED | NOT CONNECTED | NOT CONNECTED | DIRECT | DIRECT | NO ROUTE (static graph class; temporal transport has its own rows) | NOT CONNECTED |
| sampled interval | closed | REFUSED | REFUSED | REFUSED | REFUSED | REFUSED | REFUSED | REFUSED | REFUSED |

Refusal details: counted, empirical or posterior laws given to the exact
route are `counterfactual_transport.no_sampled_inference`; a cyclic graph
is `counterfactual_transport.graph_outside_contract`; a selected-sample law is
`counterfactual_transport.selected_sample_law`; an
undeclared table relation is `counterfactual_transport_sampled.overlap_unknown`.

A transported response (a value with a receipt of route, populations,
transport identities, assumptions and an extrapolative support label) is a
decision claim through
`BoundDecisionContract::evaluate_transported` and a forward source for the
inverse query through `antecedent::analysis::inverse_sources`; a point answers
an affine utility or a target mean and refuses a law
(`transported_decision.law_not_supplied`,
`inverse_forward_source.point_only_law_requested`) and an interval
(`not_supported`), and the default policy refuses the label
(`transported_decision.transported_not_accepted`). A bounded violation of the
transport premise gives an assumption range through the sensitivity artifact
surface (`transport_sensitivity`; additive or proportional shift, an
assumption range on a declared bound, never composed with a sampling
interval), and a changed source, diagram, target or calibration record is
classified still valid, recompute or withdrawn by `transport_revalidation`
(a calibration established at another target is withdrawn, never inherited).

| Route | Inference mode | Decision | Inverse query | Sensitivity | Design ranking | Counterfactual | Transport | Temporal | Composition |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| transported response, point (cohort effect, policy, synthesis, declared) | point-only, extrapolative, receipt carried | DIRECT (`evaluate_transported`; law and interval REFUSED) | DIRECT under an accepting input policy; REFUSED `transported_not_accepted` otherwise | DIRECT (assumption range) | NOT CONNECTED | NO ROUTE | DIRECT (revalidation) | NOT CONNECTED | NOT CONNECTED |
| transported response, law | transported law, receipt carried | DIRECT (probability criterion; interval REFUSED) | DIRECT under an accepting input policy | REFUSED (`transport_sensitivity.law_response_not_scalar`) | NOT CONNECTED | NO ROUTE | DIRECT (revalidation) | NOT CONNECTED | NOT CONNECTED |
| cohort effect under a declared selection shift | point-only, labelled extrapolative | ADAPTER (declared mean source) | ADAPTER (`cohort_forward_source`; accepting policy) | DIRECT | NOT CONNECTED | NO ROUTE | DIRECT | NOT CONNECTED | NOT CONNECTED |
| transported policy value (initial-state law shifted, selection diagram declared) | point-only, exact finite enumeration | ADAPTER (value as a declared mean) | ADAPTER | REFUSED | NOT CONNECTED | NO ROUTE | DIRECT (library; Python `transported_constructors.from_policy`) | REFUSED (use the period route) | NOT CONNECTED |
| period shift of a policy value (fixed horizon, time-invariant mechanisms) | point-only | ADAPTER | ADAPTER | REFUSED | NOT CONNECTED | NO ROUTE | REFUSED (`temporal_policy_transport.mechanism_time_varying`) | DIRECT (library; Python `transported_constructors.from_period_shift`) | NOT CONNECTED |
| adaptive trial as a transport source (recorded-probability standardization) | point-only, uncalibrated, no variance | ADAPTER (declared mean contrast) | REFUSED | REFUSED | REFUSED | NO ROUTE | DIRECT (library, finite strata; Python `transported_constructors.from_adaptive_trial`) | REFUSED | NOT CONNECTED |
| discovered CPDAG class | structural envelope, never a point or interval | REFUSED | REFUSED | REFUSED (`transport_sensitivity.no_scalar_family`) | REFUSED | NO ROUTE | DIRECT (library and Python `transport.discovered.transport_across_class` for a CPDAG or PAG class, estimates through `transport.discovered.estimate_discovered_transport`; temporal classes at a declared horizon in the library) | NO ROUTE | NOT CONNECTED |
| provider output as source evidence | point-only or not_measured, never native target data, never measured | not applicable (source evidence) | not applicable | not applicable | not applicable | NO ROUTE | DIRECT (admission; attested never relabelled verified) | NO ROUTE | NOT CONNECTED |
| discretized composition | point-only with a computed mean-contrast bound | REFUSED (`discretized_scm.point_only` for an interval) | REFUSED | REFUSED | REFUSED | NO ROUTE | NO ROUTE | NO ROUTE | NOT CONNECTED |
| whole-path cross-world spread (finite Markovian row) | uncalibrated spread, never added across estimands | REFUSED (`cross_world_execution.no_interval_method`) | REFUSED | REFUSED | REFUSED | DIRECT | REFUSED (`cross_world_uncertainty.no_transported_path_derivation`) | NO ROUTE | NOT CONNECTED |

Temporal transport of a policy value is delivered by
`temporal_policy_transport`, not by the counterfactual route above. None of
these outputs is a composition bundle node kind, so a bundle cannot carry them. General
path-specific transport is not supported (see
`parity/capability_gaps.toml`).

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
| composed spread (linear moments, aligned functional) | uncalibrated spread, derived and not_measured | DIRECT (`propagation_artifact`: rebuilt from its bytes under the retained identity) | NO ROUTE (a spread is not a utility input) | NO ROUTE | NO ROUTE | NO ROUTE | NO ROUTE | NO ROUTE | NO ROUTE | NOT CONNECTED |
| composed exceedance (probability, band, verdict; known-weight mixture) | uncalibrated spread, derived and not_measured | DIRECT (`propagation_artifact`) | NO ROUTE (no utility reading) | NO ROUTE | NO ROUTE | NO ROUTE | NO ROUTE | NO ROUTE | NO ROUTE | NOT CONNECTED |
| transported spread (source sampling, target sampling, transport functional) | uncalibrated spread, all three pieces and their dependence named | DIRECT (`propagation_artifact`: replays to identical bits); REFUSED for decision-reliability, composition and inverse query | NO ROUTE | NO ROUTE | NO ROUTE | NO ROUTE | NO ROUTE | NO ROUTE | NO ROUTE | NOT CONNECTED |
| composed interval | no interval route; one needs an independent check of the composed construction | REFUSED (`published_route`: `not_supported`, with the mismatch named) | REFUSED | REFUSED | REFUSED | REFUSED | REFUSED | REFUSED | REFUSED | REFUSED |
| derived decision uncertainty (numerical, selection, rollout components) | derived and not_measured, components never combined | DIRECT (`decision_reliability_derived`: re-derived from the carried draws); REFUSED when published | DIRECT (assessed on the bound decision's own draws) | NO ROUTE | NO ROUTE | NO ROUTE | NO ROUTE | NO ROUTE | NO ROUTE | NOT CONNECTED |
| repeated-measures or clustered procedure report | exact forms plus seeded simulation, derived and not_measured | REFUSED (`decision_reliability_derived`: no route) | NO ROUTE | NO ROUTE | NO ROUTE | NO ROUTE | NO ROUTE | NO ROUTE | NO ROUTE | NOT CONNECTED |
| transported cohort effect (facade over the fitted-effect transport) | point-only, extrapolative, typed receipt | DIRECT (`decision`: the existing cohort-mean source) | DIRECT (interval and law REFUSED) | ADAPTER (`cohort_forward_source`; accepting policy) | DIRECT | NO ROUTE | NO ROUTE | DIRECT (revalidation) | NO ROUTE | NOT CONNECTED |
| transported measurement estimate (carried or target-established error rates) | point-only, uncalibrated, typed receipt | DIRECT (`measurement_artifact`: the existing artifact replays) | NOT CONNECTED | NOT CONNECTED | NOT CONNECTED | NOT CONNECTED | NO ROUTE (an error-rate invariance is not a counterfactual claim) | DIRECT (library) | NO ROUTE | NOT CONNECTED |
| transported mechanism composition (carried or refitted linear-Gaussian mechanisms) | point-only with a labelled parameter-only variance | DIRECT (`mechanism_composition_artifact`: replays to the same bits) | NOT CONNECTED | ADAPTER (`inverse_forward_source` of the composition) | NOT CONNECTED | NOT CONNECTED | REFUSED (conditioning is a separate query) | DIRECT (library; a selection arrow into a carried mechanism refused) | NO ROUTE | NOT CONNECTED |
| per-population discovery | structure only, nothing pooled, differences descriptive | DIRECT (`claim_set`: one claim per population) | NO ROUTE (a structure has no response) | NO ROUTE | NO ROUTE | NO ROUTE | NO ROUTE | REFUSED (`selective_discovery.differences_are_not_selection_targets`) | NO ROUTE | NOT CONNECTED |
| sample-split discovery | structure on the discovery part; analysis part bound by a receipt | DIRECT (`evidence_ledger`: two disjoint sources; reuse refused) | NO ROUTE | NO ROUTE | NO ROUTE | NO ROUTE | NO ROUTE | NO ROUTE | NO ROUTE | NOT CONNECTED |

## Reading the table

- A NOT CONNECTED cell records that the consumer has no way to accept this
  output and which module an adapter would belong in. It is not a promise.
- Apart from the two intervals below, frequentist and Bayesian modes are not
  supported for any row: each would need an independent check of its
  whole-method construction, and every such route refuses with a typed
  refusal. The Wald interval of a measurement
  estimate is the delta-method construction named in that section, and the
  pooled interval is the construction named in the evidence-ledger section; each
  is annotated with its calibration status, which says whether a coverage record
  measured it. A derived, not_measured statement (a composed
  spread, a decision-reliability numerical or selection interval, a
  whole-procedure adaptive report) is labelled `uncalibrated_spread` or
  `derived_not_measured` and is never support: publishing it refuses
  `not_supported`.
- Which paradigm each route supports, and the typed closed row for a missing
  one (`paradigm_parity.paradigm_not_supported`), is the parity table of
  `antecedent_model::discretized_scm::PARITY_TABLE`.
- The refusals named above, and their remedies, are listed in
  [refusal codes](reference/refusal-codes.md).
- The identities each route freezes are in
  [composition identities](architecture/composition-identities.md).
