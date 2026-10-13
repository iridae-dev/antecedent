# Refusal and partial knowledge

Sometimes the correct causal result is not a number. Antecedent distinguishes
states because they imply different next actions.

- **Not identified:** the declared structure and evidence do not determine the
  request; obtain different evidence or justify stronger assumptions.
- **Missing evidence/provider:** a formula may exist, but a required joint law
  or statistical implementation is absent.
- **Support failure:** the causal question may be identified while the actual
  data do not cover the requested intervention or population.
- **Not certified/licensed:** a meaningful request is outside the implemented
  theorem or evidence-backed execution surface.
- **Budget or numerical failure:** execution did not complete; this is not a
  scientific impossibility claim.

Structural transport makes this especially visible through typed outcomes such
as `ProvenNonTransportable`, `NotCertified`, `MissingEvidence`,
`MissingProvider`, `UnsupportedEvaluator`, `SupportFailure`,
`NumericalFailure`, and `BudgetCancel`. The principle applies everywhere:
information that is unavailable must remain unavailable, not become an empty
success object, `NaN`, or a generic error string.

## Structured refusals and their remedies

Scientific stage refusals share the structured
base, `antecedent.errors.StructuredRefusal` (itself a `CausalUnsupportedError`).
It exposes machine-readable fields, so you branch on fields rather than parse
a message. Input type, value and resource errors can instead raise ordinary
`CausalTypeError`, `CausalValueError` or `CausalResourceError` without a scientific
reason code.

| Field | Meaning |
| --- | --- |
| `code` (alias `reason_code`) | the registered refusal code, such as `cell_not_licensed`, `route_not_supported`, `transport_proven_non_transportable` or `invalid_argument` |
| `detail` | the namespaced `family.slot`, such as `msm_sensitivity.lambda_below_one` or `composition_bundle.node_not_found` |
| `offending` | the field, quantity, node or identity the refusal is about, or `None` |
| `remedy` | what the caller can change in order to proceed, or `None` |
| `stage` | the refusing stage |
| `message` | human-readable context |

```python
from antecedent import msm_sensitivity
from antecedent.errors import StructuredRefusal

try:
    msm_sensitivity.msm_ate_sensitivity(strata, 0.5)
except StructuredRefusal as refusal:
    if refusal.code == "invalid_argument":
        ...                      # a malformed argument: fix the call
    elif refusal.code == "cell_not_licensed":
        ...                      # a meaningful request outside the licensed surface
    print(refusal.detail, refusal.offending, refusal.remedy)
```

A refusal whose code is `invalid_argument` rejects a malformed argument rather than
a scientific route, and is also a `CausalValueError`, so `except CausalValueError`
catches it. A refusal is registered: its code is one of the reasons in the registry
and its detail is a literal under its route's namespace.

The families built on the base are: `DecisionRefusal` (and its
`ScenarioDecisionRefusal` and `CompositionRefusal` subtypes), `SensitivityRefusal`,
`MsmSensitivityRefusal`, `MechanismDiscrepancyRefusal`, `TransportedCounterfactualRefusal`
(with a two-model `witness` when it proves non-transportability),
`ScenarioInvarianceRefusal`, `RepairRefusal`, `RecalcRefusal` (with
`RecalcUnavailable`, `ScoreResumeRefusal` and `ScoreResumeUnavailable`),
`CompositionBundleRefusal` (with one subclass per stage, including
`NodeNotFoundRefusal`, `TamperedQuantityRefusal` and `ExpectedIdentityMismatchRefusal`),
and the earlier estimator refusals `VectorTreatmentRefusal`,
`CategoricalTreatmentRefusal` and `CompactExportRefusal`. Many of them are also importable
lazily from `antecedent.errors`. `ExternalRefusal` and `DesignRankingRefusal` (with
`SignalProviderRefusal`, `CostUnitsRefusal`, `SourceOverlapRefusal`) are
`CausalUnsupportedError` subclasses that carry the same `reason_code`, `remedy`,
`stage`, `detail` and `offending` fields. To handle any structured refusal generically,
catch `CausalUnsupportedError` and read those attributes.

Common remedies:

| Refusal | What to change |
| --- | --- |
| `cell_not_licensed` / `*.composition_not_licensed` | Report the sampling interval next to the assumption range; do not compose them. No composition method is licensed. |
| `cell_not_licensed` / `learned_joint_transport.route_frozen` | Use the checked `transport.advanced.learned_joint_transport` adapter inside its measured protocol. The separate `prepare_learned_continuous` IID route offers a point estimate or its scoped analytic interval via `prepared.interval()`; these methods have distinct contracts. |
| `cell_not_licensed` / `transported_counterfactual.nonadditive_mechanism` and its siblings | Declare the premise in `Premises` if you can defend it, or choose an assignment inside the affine-additive class. |
| `transport_proven_non_transportable` | Read `witness`: two models agree on the source and differ on the target. Remove the selection on that mechanism or supply target evidence for it. |
| `decision_contract_unsatisfied` / `native_claims.source_not_supplied` | A mean cannot answer a quantile or probability. Supply aligned joint draws. |
| `route_not_supported` / `decision_evaluation.mean_source_not_replayable` | `Decision.export()` requires aligned joint draws; `decision.replay` uses that same law. For a bound external mean claim, `composition_bundle.mean_decision` supplies a distinct point-only bundle artifact. Native mean workflows retain their own source-bound recalculation/replay contracts. |
| `design_cost_units_mismatch` | Supply a `CostMap` from the study cost unit to the decision's utility unit, or consume under the cost map the ranking used. |
| `score_table_unavailable` / `recalc.unavailable_data` | The request needs the data or a fit; supply the data and refit, or ask only for a same-row reweighting of the frozen scores. |
| `composition_bundle.node_not_found` | Add the node before connecting it (`NodeNotFoundRefusal.offending` is the missing id). |
| `invalid_argument` | Fix the argument named in `detail` and `offending`. |

See also the [lifecycle](lifecycle.md) for the refusal-handling pattern in
context.

## Measurement, composition, evidence and counterfactual-transport refusals

Each detail below is registered with its top-level code and a remedy in `parity/promotion.toml`. The remedy says what evidence, assumption or declaration would license the request; a refusal that has no licensing declaration yet says so. A refusal is never a verdict on the science when it is a cancellation, a budget stop or an invalid argument.

### Measurement models

`measurement_model.*`. A measured column is related to a latent construct only through a declared model; a refusal names the declaration or evidence that would license the request.

| Detail | Code | What would license the request |
| --- | --- | --- |
| `measurement_model.arm_mismatch` | `measurement_binding_mismatch` | Declare rates for this arm, or analyse an arm the differential model covers. |
| `measurement_model.artifact_invalid` | `invalid_argument` | Supply an intact measurement model artifact of the supported layout and size. |
| `measurement_model.artifact_refused` | `measurement_binding_mismatch` | Supply the artifact whose model identity equals the retained one and whose result recomputes from its stored inputs. |
| `measurement_model.blank_field` | `measurement_model_invalid` | Name every identity, unit, population and regime of the model. |
| `measurement_model.cancelled` | `cancelled_no_claim` | Rerun without cancellation; the cancellation is not a verdict on the data. |
| `measurement_model.confusion_ill_conditioned` | `measurement_not_identified` | Supply a better conditioned confusion matrix (a more accurate indicator, or validation units for every latent category). |
| `measurement_model.construct_mismatch` | `measurement_binding_mismatch` | Request the construct the model names, or declare a model for the construct asked about. |
| `measurement_model.design_singular` | `design_rank_deficient` | Remove collinear columns or add observations so the design has full rank. |
| `measurement_model.differential_axis_unsupported` | `differential_error_not_pooled` | Declare error that differs by arm; other axes (population, time) have no bounded route and stay refused. |
| `measurement_model.differential_declared` | `differential_error_not_pooled` | Analyse the model with its per-arm parameters, not through the non-differential route. |
| `measurement_model.differential_undeclared` | `differential_error_not_pooled` | Declare the differential structure with its own per-arm parameters. |
| `measurement_model.indicator_as_construct` | `proxy_offered_as_construct` | Request the construct by its identity; the measured column is its proxy. |
| `measurement_model.indicator_kind` | `measurement_model_invalid` | Offer an indicator of a kind the route accepts (see the route card). |
| `measurement_model.indicator_observed` | `measurement_model_invalid` | Declare the column noisy or misclassified, or use it directly as observed. |
| `measurement_model.input_invalid` | `invalid_argument` | Supply equal-length columns, distinct arms and weights of the declared width. |
| `measurement_model.insufficient_draws` | `measurement_not_identified` | Supply at least the number of joint draws the model declares. |
| `measurement_model.insufficient_replicates` | `measurement_not_identified` | Replicate the measurement on more units so the within-unit variance is identified. |
| `measurement_model.interval_closed` | `cell_not_licensed` | None available yet: the point and an uncalibrated standard error are the result; an interval needs a measured whole-method coverage record. |
| `measurement_model.intervention_meaning_not_carried` | `representation_mapping_required` | Declare evidence that interventions correspond across the mapping; a relabeling carries values only. |
| `measurement_model.invalid_model` | `measurement_model_invalid` | Declare a well formed model: named identities and parameters of the declared route within range. |
| `measurement_model.learned_coordinates_not_intervenable` | `representation_mapping_required` | Declare the intervention on the scientific construct; coordinates of a learned representation have no intervention meaning of their own. |
| `measurement_model.mapping_endpoints` | `representation_mapping_required` | Supply a mapping that runs exactly from the model's representation version to the requested one. |
| `measurement_model.operation_not_licensed` | `cell_not_licensed` | Request an operation in the route's allowed set (see its route card). |
| `measurement_model.parameter` | `measurement_model_invalid` | Supply parameters within their ranges (probabilities in [0, 1], positive variances). |
| `measurement_model.population_mismatch` | `measurement_binding_mismatch` | Analyse data of the population the model names, or declare a model for this population. |
| `measurement_model.posterior_capability` | `measurement_posterior_not_trusted` | Declare the sampling operation; the route consumes joint posterior draws. |
| `measurement_model.posterior_declaration` | `measurement_posterior_not_trusted` | Complete the provider declaration so it is consistent with the object. |
| `measurement_model.posterior_identity` | `measurement_posterior_not_trusted` | Supply the exact provider object the model names. |
| `measurement_model.posterior_kind` | `measurement_posterior_not_trusted` | Supply the latent variable as a parameter-posterior object. |
| `measurement_model.posterior_native_trust` | `measurement_posterior_not_trusted` | Attest or verify the provider object; a provider object is never natively licensed. |
| `measurement_model.posterior_quantity` | `measurement_posterior_not_trusted` | Have the posterior declare the construct in the model's population and regime. |
| `measurement_model.posterior_receipt` | `measurement_posterior_not_trusted` | Supply a verification receipt that covers this exact provider object. |
| `measurement_model.proxy_as_construct` | `proxy_offered_as_construct` | Name the latent scientific construct and relate the proxy to it by a measurement model. |
| `measurement_model.regime_mismatch` | `measurement_binding_mismatch` | Analyse data of the regime the model names, or declare a model for this regime. |
| `measurement_model.reliability_too_small` | `measurement_not_identified` | Use a more reliable measurement or averaged replicates; the error variance leaves too little latent variance. |
| `measurement_model.representation_unmapped` | `representation_mapping_required` | Supply an explicit mapping between the two representation versions; a changed coordinate system or version is a different quantity. |
| `measurement_model.route_mismatch` | `route_not_supported` | Call the entry point of the route the model declares, or declare a model for this route. |
| `measurement_model.route_parameters` | `measurement_model_invalid` | Supply parameters of the route the model declares. |
| `measurement_model.structure_parameters` | `differential_error_not_pooled` | Declare shared parameters as non-differential, or one parameter set per arm as differential by arm. |
| `measurement_model.unknown_level` | `invalid_argument` | Name only categories that are declared levels of the confusion model. |
| `measurement_model.unmodelled_proxy` | `proxy_offered_as_construct` | Declare a measurement model relating the measured quantity to the construct, or ask the question about the measured quantity itself. |
| `measurement_model.unrecognized` | `measurement_model_invalid` | Report the construction failure; no registered declaration would license it. |
| `measurement_model.variance_not_positive` | `measurement_not_identified` | Supply data and parameters under which the corrected variance is positive (more units or a less degenerate design). |
| `measurement_model.youden_too_small` | `measurement_not_identified` | Use a more accurate indicator or a validation study that shows a Youden index above the declared minimum. |

### Mechanism composition

`mechanism_composition.*`. Individually valid mechanisms compose only when the declarations establish one coherent joint law; every defect names the declaration that would license the composition.

| Detail | Code | What would license the request |
| --- | --- | --- |
| `mechanism_composition.artifact_invalid` | `invalid_argument` | Supply an intact artifact of the supported layout with well formed declarations. |
| `mechanism_composition.cancelled` | `cancelled_no_claim` | Rerun without cancellation; the cancellation is not a verdict on the mechanisms. |
| `mechanism_composition.conditioning_extra_conditioner` | `mechanism_composition_incoherent` | Refit without the extra conditioners, or declare them as parents. |
| `mechanism_composition.conditioning_parent_not_conditioned` | `mechanism_composition_incoherent` | Refit so the fit conditions on every declared parent. |
| `mechanism_composition.cycle_directed_cycle` | `graph_invalid` | Declare an acyclic graph; a feedback system needs its own equilibrium model outside this route. |
| `mechanism_composition.family_mixed` | `mechanism_composition_incoherent` | Compose one family at a time: all linear-Gaussian or all finite tables; refit within the family. |
| `mechanism_composition.identity_declaration_unknown_variable` | `quantity_semantics_mismatch` | Name only variables that have a mechanism in the composition. |
| `mechanism_composition.identity_duplicate_mechanism` | `quantity_semantics_mismatch` | Keep one fitted mechanism per variable, or compose the alternatives separately. |
| `mechanism_composition.identity_edge_unknown_variable` | `quantity_semantics_mismatch` | Declare edges only between variables that have a mechanism. |
| `mechanism_composition.identity_parent_without_mechanism` | `quantity_semantics_mismatch` | Supply a mechanism for the parent; a root needs its own marginal mechanism. |
| `mechanism_composition.interval_route_closed` | `cell_not_licensed` | None available yet: the point and the labelled first-order variance are the result; an interval needs a measured whole-method coverage record. |
| `mechanism_composition.latent_family_unsupported` | `mechanism_composition_incoherent` | Use linear-Gaussian mechanisms for a shared latent, or marginalize the latent explicitly in a model that supports it. |
| `mechanism_composition.latent_loading_undeclared` | `mechanism_composition_incoherent` | Declare the loading of the mechanism on the shared latent. |
| `mechanism_composition.latent_shared_with_absent_mechanism` | `mechanism_composition_incoherent` | Add the mechanism that shares the latent to the composition; a dropped latent is never treated as independent. |
| `mechanism_composition.latent_sharing_not_declared` | `mechanism_composition_incoherent` | List every mechanism that depends on the latent on each of them. |
| `mechanism_composition.latent_sharing_not_reciprocal` | `mechanism_composition_incoherent` | Have both mechanisms declare the shared latent. |
| `mechanism_composition.latent_too_many` | `mechanism_composition_incoherent` | Reduce the number of declared latent causes to the bound. |
| `mechanism_composition.numbers_unbound` | `invalid_argument` | Bind numbers to every planned mechanism, a fitted domain to every parent, and declare every latent a loading names. |
| `mechanism_composition.numerical_failure` | `invalid_argument` | Inspect the named numerical condition and supply well-conditioned declared parameters; the declarations themselves were accepted. |
| `mechanism_composition.parents_declared_graph_mismatch` | `graph_invalid` | Refit the mechanism on exactly the declared parents, or correct the declared graph. |
| `mechanism_composition.population_invariance_unstated` | `quantity_semantics_mismatch` | State the invariance assumption in words the artifact can carry. |
| `mechanism_composition.population_latent_blocks_selection_admissibility` | `quantity_semantics_mismatch` | Declare an explicit invariance assumption for the mechanism with its latent, or refit it in the target population; a selection diagram does not cover latent causes. |
| `mechanism_composition.population_parent_context_differs_from_fit` | `quantity_semantics_mismatch` | Declare the parent in the population the consuming mechanism was fitted in, or refit the consumer. |
| `mechanism_composition.population_selection_arrow_into_mechanism` | `quantity_semantics_mismatch` | Refit the mechanism in the target population; a selection arrow into it means it is not invariant. |
| `mechanism_composition.population_undeclared_transport` | `quantity_semantics_mismatch` | Declare a selection diagram with no arrow into this variable, or an explicit invariance assumption, or refit the mechanism in the target population. |
| `mechanism_composition.posterior_dependence_cross_block_duplicate` | `joint_law_required` | Give each pair of distinct mechanisms exactly one cross-covariance block. |
| `mechanism_composition.posterior_dependence_cross_covariance_missing` | `joint_law_required` | Give every pair of mechanisms an explicit covariance block; a zero block is a declared independence for that pair. |
| `mechanism_composition.posterior_dependence_independence_unjustified` | `joint_law_required` | State why the estimates are independent (disjoint, independently generated evidence). |
| `mechanism_composition.posterior_dependence_shared_evidence_declared_independent` | `joint_law_required` | Give the mechanisms distinct evidence identities, or declare the dependence between them. |
| `mechanism_composition.posterior_dependence_unknown` | `joint_law_required` | Declare an aligned joint parameter law, an explicit cross-mechanism covariance for every pair, or independence by design with a justification and distinct evidence identities. |
| `mechanism_composition.premises_mismatch` | `mechanism_composition_incoherent` | Supply the artifact that matches the retained identity, or regenerate it from the intended premises. |
| `mechanism_composition.provenance_declared_mechanism_unacknowledged` | `mechanism_composition_incoherent` | Acknowledge in the composition that the mechanism is declared rather than estimated, or supply a fitted mechanism. |
| `mechanism_composition.provenance_missing_artifact` | `mechanism_composition_incoherent` | Bind the mechanism to the identity of the fit or external binding that produced it. |
| `mechanism_composition.provenance_missing_evidence` | `mechanism_composition_incoherent` | Name the evidence identity the mechanism was estimated from so dependence can be judged. |
| `mechanism_composition.quantity_conditioned_event` | `quantity_semantics_mismatch` | Compose unconditioned structural mechanisms and ask the conditional question downstream. |
| `mechanism_composition.quantity_conditioning_mismatch` | `quantity_semantics_mismatch` | Make the parent and its producer conditioned on the same events. |
| `mechanism_composition.quantity_functional_mismatch` | `quantity_semantics_mismatch` | Make the parent and its producer the same functional of the variable. |
| `mechanism_composition.quantity_horizon_mismatch` | `quantity_semantics_mismatch` | Make the parent and its producer refer to the same time horizon. |
| `mechanism_composition.quantity_invalid_coordinate` | `quantity_semantics_mismatch` | Give every semantic coordinate a non-blank identity and canonical conditioning. |
| `mechanism_composition.quantity_not_a_realization` | `quantity_semantics_mismatch` | Declare the mechanism over realized values; a mean or other summary is not a parent value. |
| `mechanism_composition.quantity_role_mismatch` | `quantity_semantics_mismatch` | Make the parent and its producer name the same variable in the same causal role. |
| `mechanism_composition.regime_parent_context_differs_from_fit` | `quantity_semantics_mismatch` | Declare the parent in the regime the consuming mechanism was fitted in, or refit the consumer. |
| `mechanism_composition.regime_selection_diagram_not_applicable` | `quantity_semantics_mismatch` | Declare an explicit invariance assumption between the fitted and the target regime, or refit the mechanism in the target regime. |
| `mechanism_composition.regime_undeclared_invariance` | `quantity_semantics_mismatch` | Declare that the mechanism is invariant between the fitted and the target regime, or refit it in the target regime. |
| `mechanism_composition.request_unsupported` | `route_not_supported` | Restate the request inside the bounded family: known, non-repeated intervention variables, a sample size within the bound, and a query of the composition's family. |
| `mechanism_composition.result_mismatch` | `mechanism_composition_incoherent` | Regenerate the artifact from the premises and query; a stored result must be the replay of them. |
| `mechanism_composition.shape_blank_coordinate` | `invalid_argument` | Name the target population, regime and every declared coordinate. |
| `mechanism_composition.shape_blank_identity` | `invalid_argument` | Give every mechanism, parent and latent a stable non-blank identity. |
| `mechanism_composition.shape_coefficient_parent_mismatch` | `invalid_argument` | Give exactly one coefficient per declared parent. |
| `mechanism_composition.shape_duplicate_parent` | `invalid_argument` | List each parent of a mechanism once. |
| `mechanism_composition.shape_empty_composition` | `invalid_argument` | Declare at least one mechanism. |
| `mechanism_composition.shape_invalid_domain` | `invalid_argument` | Declare a well formed output support (a non-empty level set, or ordered finite bounds). |
| `mechanism_composition.shape_invalid_latents` | `invalid_argument` | Name each latent once with a non-blank identity. |
| `mechanism_composition.shape_invalid_table` | `invalid_argument` | Supply each finite mechanism as a normalized conditional table (every row a distribution) over declared levels. |
| `mechanism_composition.shape_mechanism_count` | `invalid_argument` | Declare between one mechanism and the bound. |
| `mechanism_composition.shape_negative_scale` | `invalid_argument` | Supply a non-negative residual standard deviation. |
| `mechanism_composition.shape_non_finite_parameter` | `invalid_argument` | Refit or correct the mechanism so every coefficient, intercept and scale is finite. |
| `mechanism_composition.shape_too_many_mechanisms` | `invalid_argument` | Split the composition below the declared bound. |
| `mechanism_composition.shape_too_many_parents` | `invalid_argument` | Reduce the parent set below the declared bound. |
| `mechanism_composition.slot_not_linear_gaussian` | `mechanism_composition_incoherent` | Offer a plain linear-Gaussian slot or a finite table; basis expansions and discrete, constant, vacant or pending slots are outside the bounded family. |
| `mechanism_composition.support_disjoint` | `mechanism_composition_incoherent` | Refit the consumer on the producer's range or restrict the producer; with no overlap the composition has no support. |
| `mechanism_composition.support_intervention_outside_domain` | `mechanism_composition_incoherent` | Intervene only within the domain every consumer of the variable was fitted on. |
| `mechanism_composition.support_kind_mismatch` | `mechanism_composition_incoherent` | Refit so a numeric output feeds a numeric input and a level output feeds a level input. |
| `mechanism_composition.support_missing_parent_configuration` | `mechanism_composition_incoherent` | Supply a table row for every combination of parent levels the producers can emit. |
| `mechanism_composition.support_not_contained` | `mechanism_composition_incoherent` | Restrict the producer to the consumer's fitted domain, refit the consumer on the producer's range, or declare global validity of the consumer's domain. |
| `mechanism_composition.transform_mismatch` | `quantity_semantics_mismatch` | Declare the transform the consumer was fitted on, or refit one side. |
| `mechanism_composition.uncertainty_assembled_not_psd` | `mechanism_composition_incoherent` | Supply marginals and cross-covariances that form one positive semidefinite joint covariance. |
| `mechanism_composition.uncertainty_cross_block_shape` | `mechanism_composition_incoherent` | Give each cross block the shape of exactly the two mechanisms' parameter sets, finite. |
| `mechanism_composition.uncertainty_cross_block_unknown_mechanism` | `mechanism_composition_incoherent` | Name only mechanisms that are in the composition. |
| `mechanism_composition.uncertainty_family_unsupported` | `mechanism_composition_incoherent` | Request the point result for finite tables; parameter uncertainty is propagated for linear-Gaussian mechanisms only. |
| `mechanism_composition.uncertainty_joint_parameter_set_mismatch` | `mechanism_composition_incoherent` | Make the joint law cover exactly every parameter of every mechanism once. |
| `mechanism_composition.uncertainty_marginal_inconsistent_with_joint` | `mechanism_composition_incoherent` | Use one posterior: make the joint law agree with each mechanism's own reported marginal. |
| `mechanism_composition.uncertainty_marginal_missing` | `mechanism_composition_incoherent` | Have the owning fit report its parameter covariance. |
| `mechanism_composition.uncertainty_meaning_mismatch` | `mechanism_composition_incoherent` | Supply parameter laws that all carry one meaning (all posterior or all sampling). |
| `mechanism_composition.uncertainty_not_psd` | `mechanism_composition_incoherent` | Supply a positive semidefinite covariance. |
| `mechanism_composition.uncertainty_not_symmetric` | `mechanism_composition_incoherent` | Supply a symmetric covariance. |
| `mechanism_composition.uncertainty_parameter_set_mismatch` | `mechanism_composition_incoherent` | Report the covariance over exactly the intercept and one coefficient per declared parent. |
| `mechanism_composition.uncertainty_shape` | `mechanism_composition_incoherent` | Supply a finite k by k covariance matching the parameter list. |
| `mechanism_composition.units_mismatch` | `quantity_semantics_mismatch` | Convert the units explicitly and refit one side; no unit conversion is inferred. |

### Evidence ledger

`evidence_ledger.*`. Overlap that is not declared stays unknown and is never treated as independent; a refusal names the declaration, covariance or recomputation that would license the combination.

| Detail | Code | What would license the request |
| --- | --- | --- |
| `evidence_ledger.cancelled` | `cancelled_no_claim` | Rerun without cancellation; the ledger is unchanged and no verdict was formed. |
| `evidence_ledger.diagonal_covariance_shared_rows` | `evidence_double_counted` | Supply the cross covariance for sources that share observations. |
| `evidence_ledger.duplicate_result` | `invalid_argument` | Cite the existing result rather than registering the combination twice. |
| `evidence_ledger.duplicate_row` | `evidence_double_counted` | Declare each observation once within a source. |
| `evidence_ledger.duplicate_source` | `evidence_double_counted` | Cite the existing source rather than adding the same evidence twice. |
| `evidence_ledger.identity_mismatch` | `route_not_supported` | Supply the artifact whose recomputed identity equals the retained identity, or regenerate it from the intended ledger history. |
| `evidence_ledger.interval_not_licensed` | `cell_not_licensed` | None available yet: the ledger returns point values and labelled assumption ranges; an interval needs measured whole-method coverage. |
| `evidence_ledger.invalid_declaration` | `invalid_argument` | Declare a non-empty bounded label and a well formed lineage, transport identity and assumption range. |
| `evidence_ledger.kind_mismatch` | `invalid_argument` | Keep the source kind unchanged, or register a new source of the other kind. |
| `evidence_ledger.limit_exceeded` | `invalid_argument` | Reduce the declaration below the stated ledger or combination bound. |
| `evidence_ledger.lineage_rows_mismatch` | `invalid_argument` | Make the declared observations agree with the declared lineage (a subset or duplicate holds exactly its parent's rows). |
| `evidence_ledger.native_and_transported` | `evidence_double_counted` | Count each source once: as native target data or as a transported claim, not both. |
| `evidence_ledger.operation_arity` | `invalid_argument` | Name at least two sources for pooling, or a named stage for a repeated use. |
| `evidence_ledger.overlap_unknown` | `evidence_overlap_unknown` | Declare the row-level observations behind each source, supply a covariance with cross terms, or request an assumption range. |
| `evidence_ledger.reentry_derived_from_used_data` | `evidence_double_counted` | Use the underlying data once, or derive the claim from observations disjoint from those already used. |
| `evidence_ledger.reentry_duplicate` | `evidence_double_counted` | Drop the copy, or combine only sources with distinct observations. |
| `evidence_ledger.reentry_prior_plus_likelihood` | `evidence_double_counted` | Build the prior from data disjoint from the likelihood, or elicit it without data (empty observation set). |
| `evidence_ledger.reentry_transform` | `evidence_double_counted` | Use one of the sources, or declare the transform's lineage and supply a covariance with cross terms. |
| `evidence_ledger.repeated_source` | `evidence_double_counted` | List each source once among the inputs of a combination or derived source. |
| `evidence_ledger.replay_mismatch` | `invalid_argument` | Regenerate the artifact from the ledger operations; a stored operation identity, effect, head or state digest must replay. |
| `evidence_ledger.result_not_valid` | `evidence_source_withdrawn` | Recompute the result over the replacing sources before citing it again. |
| `evidence_ledger.shared_observations` | `evidence_double_counted` | Declare the shared observations and supply a covariance with cross terms, or combine only sources known to be disjoint. |
| `evidence_ledger.source_superseded` | `evidence_source_withdrawn` | Use the successor of the superseded source, or revalidate the request over it. |
| `evidence_ledger.source_withdrawn` | `evidence_source_withdrawn` | Remove the withdrawn source or register evidence that does not rest on it. |
| `evidence_ledger.supersede_kind_mismatch` | `invalid_argument` | Supersede only a fitted model or external claim, by one of the same kind; replace or update other evidence. |
| `evidence_ledger.synthesis_mismatch` | `invalid_argument` | Regenerate the synthesis from its ledger result; the stored values must recompute from the stored inputs. |
| `evidence_ledger.transport_identity_missing` | `invalid_argument` | Attach the transport identity (source populations, selection diagram, assumptions) to the transported claim. |
| `evidence_ledger.transported_as_native` | `invalid_argument` | Register the claim as transported, or supply native target data instead. |
| `evidence_ledger.unknown_result` | `invalid_argument` | Register the result in the ledger first, or cite the result that replaced it. |
| `evidence_ledger.unknown_source` | `invalid_argument` | Add the source to the ledger before referring to it. |
| `evidence_ledger.update_drops_rows` | `invalid_argument` | Use replace to correct or drop observations; an update only adds them. |

### Counterfactual transport (exact finite laws)

`counterfactual_transport.*`. A refusal to derive is not a proof of non-transportability; only `checked_obstruction` carries a verified two-model witness.

| Detail | Code | What would license the request |
| --- | --- | --- |
| `counterfactual_transport.bounds_exceeded` | `route_not_supported` | Reduce the problem below the stated variable, level, world, atom, source, regime or search limit, or request smaller limits. |
| `counterfactual_transport.budget` | `transport_budget_cancel` | Raise the operation, depth or memory budget within the contract maxima, or reduce the problem. |
| `counterfactual_transport.checked_obstruction` | `transport_proven_non_transportable` | None within the declared regimes: the query is not transportable from them; add a regime that changes the obstruction (a new experiment or a smaller selection set). |
| `counterfactual_transport.coupling_unsupported` | `route_not_supported` | Request the shared latent exogenous coupling; other couplings have no route here. |
| `counterfactual_transport.evidence_incompatible` | `transport_numerical_failure` | Repair the supplied laws or the declared graph; the laws contradict the graph. |
| `counterfactual_transport.factor_not_derivable` | `transport_not_certified` | Declare more evidence regimes or fewer selection targets; the bounded search found neither a derivation nor a witness, which is not a proof of non-transportability. |
| `counterfactual_transport.graph_outside_contract` | `route_not_supported` | Declare an acyclic graph. |
| `counterfactual_transport.invalid_artifact` | `invalid_argument` | Supply an intact artifact whose premises and data digest match its contents and the expected identity; the refused slot is named. |
| `counterfactual_transport.invalid_catalog` | `invalid_argument` | Supply a valid evidence catalog whose regimes belong to the target or a declared source and name variables of the problem. |
| `counterfactual_transport.invalid_derivation` | `transport_not_certified` | Regenerate the derivation under its stored limits; a stored derivation must be reproduced by replay. |
| `counterfactual_transport.invalid_evidence` | `invalid_argument` | Supply laws whose axes, levels, intervention set and world match the regime the derivation cites. |
| `counterfactual_transport.invalid_query` | `invalid_argument` | Declare a well formed graph, worlds, routes, levels, event, selection diagram and controllable sets, and a conditioning event of positive probability. |
| `counterfactual_transport.invalid_witness` | `transport_not_certified` | Discard the stored witness; it does not verify by exact enumeration. |
| `counterfactual_transport.missing_law` | `transport_missing_provider` | Supply the law of the cited regime at the needed intervention level. |
| `counterfactual_transport.missing_regime` | `transport_missing_evidence` | Supply the named regimes the catalog lacks; the query is derivable with them. |
| `counterfactual_transport.sampled_inference_closed` | `cell_not_licensed` | Use the sampled counterfactual transport route for counted, empirical or posterior laws. |
| `counterfactual_transport.selected_sample_law` | `route_not_supported` | Supply laws of the population itself; this route has no sampling-selection model. |
| `counterfactual_transport.support_failure` | `transport_support_failure` | Supply laws with positive mass on every conditioning event the formula divides by. |
| `counterfactual_transport.zero_conditioning_mass` | `transport_support_failure` | Condition on an event with positive mass in the target population. |

### Counterfactual transport from counts

`counterfactual_transport_sampled.*`. Sampled evidence is point-only with an uncalibrated standard error; independence of tables is never assumed.

| Detail | Code | What would license the request |
| --- | --- | --- |
| `counterfactual_transport_sampled.bounds_exceeded` | `route_not_supported` | Reduce the stored table counts, table sizes or search and evaluation limits to the consumer's bounds, or consume the artifact with higher limits within the contract maxima. |
| `counterfactual_transport_sampled.cancelled` | `transport_budget_cancel` | Rerun without cancellation; no estimate was formed. |
| `counterfactual_transport_sampled.envelope_budget` | `transport_budget_cancel` | Declare a smaller unseen-mass family or raise the envelope evaluation limit within the bound. |
| `counterfactual_transport_sampled.envelope_undefined` | `transport_support_failure` | Declare an unseen-mass family in which some member has a defined value, or supply more observations. |
| `counterfactual_transport_sampled.interval_not_licensed` | `cell_not_licensed` | None available yet: the point estimate and an uncalibrated standard error are the result; an interval needs measured whole-method coverage. |
| `counterfactual_transport_sampled.invalid_artifact` | `invalid_argument` | Supply an intact sampled artifact whose stored estimate follows from its stored counts and which carries no interval. |
| `counterfactual_transport_sampled.invalid_counts` | `invalid_argument` | Supply well formed count tables: named axes and levels, a shape that matches the counts, observations, no structural zero with observations, and parameters in range. |
| `counterfactual_transport_sampled.invalid_overlap` | `invalid_argument` | Declare each pair of tables once with one relation, naming only supplied tables. |
| `counterfactual_transport_sampled.missing_table` | `transport_missing_provider` | Supply a count table for each cited regime. |
| `counterfactual_transport_sampled.overlap_declared_independent` | `evidence_double_counted` | Declare the tables as sharing the same observations, or supply independent samples. |
| `counterfactual_transport_sampled.overlap_unknown` | `sampling_dependence_unknown` | Declare whether each pair of tables is independent samples or the same observations; independence is never assumed. |
| `counterfactual_transport_sampled.sample_size_below_limit` | `transport_support_failure` | Collect more observations, or lower the declared minimum table total. |
| `counterfactual_transport_sampled.shared_units_unlinked` | `route_not_supported` | Supply the shared observations as one dataset so a covariance is recoverable; counts alone do not carry it. |
| `counterfactual_transport_sampled.sparsity_above_limit` | `transport_support_failure` | Collect more observations, or raise the declared sparsity limit knowingly. |
| `counterfactual_transport_sampled.table_not_cited` | `invalid_argument` | Supply tables only for the populations and regimes the stored derivation cites. |
| `counterfactual_transport_sampled.unseen_cell` | `transport_support_failure` | Choose a declared additive pseudo-count (smoothed, labelled) or an unseen-mass envelope rule; zero-filling is never done. |

ok
### Observation-conditioned queries

`observation_query.*`. A conditioning claim and an interventional claim are distinct, and a recorded indicator value is not the construct it measures; a refusal names the declaration or evidence that would license the request. See [observation-queries.md](observation-queries.md).

| Detail | Code | What would license the request |
| --- | --- | --- |
| `observation_query.arm_mismatch` | `measurement_binding_mismatch` | Declare a model for this arm, or ask about the arm the model covers. |
| `observation_query.artifact_invalid` | `invalid_argument` | Supply an intact observation_query_v1 artifact of the supported layout. |
| `observation_query.artifact_refused` | `invalid_argument` | Supply the artifact whose identity equals the retained one and whose result recomputes from its premises. |
| `observation_query.blank_field` | `invalid_argument` | Name the outcome variable and the variable and level of every slot. |
| `observation_query.calibrated_probability_closed` | `cell_not_licensed` | None available yet: a probability from a model fitted to a sample is a point value with no claim about how often it is right; calibration needs a measured whole-method record at the query's coordinate. |
| `observation_query.cancelled` | `cancelled_no_claim` | Rerun without cancellation; the cancellation is not a verdict on the model. |
| `observation_query.construct_mismatch` | `measurement_binding_mismatch` | Request the construct the model names, or declare a model for the construct asked about. |
| `observation_query.construct_observed_directly` | `invalid_argument` | Condition on the construct directly or on its measured indicator, not both. |
| `observation_query.differential_error_unsupported` | `differential_error_not_pooled` | Use a non-differential model; error that differs has no bounded conditioning route and is never pooled. |
| `observation_query.indicator_kind` | `invalid_argument` | Use a direct slot for an error-free recorded value, or declare the indicator's measurement kind. |
| `observation_query.indicator_mismatch` | `measurement_binding_mismatch` | Declare a model for this indicator. |
| `observation_query.interval_route_closed` | `cell_not_licensed` | None available yet: the exact value under the declared model, or the labelled point value of a fitted model, is the result; an interval needs a measured whole-method coverage record. |
| `observation_query.intervention_in_conditioning_slot` | `quantity_semantics_mismatch` | State an assigned value as an intervention query; a conditioning slot holds only recorded values, and the observed and interventional laws are separate claims. |
| `observation_query.invalid_declaration` | `invalid_argument` | Supply named levels and distributions that sum to one, with probabilities in range. |
| `observation_query.level_unreachable` | `evidence_cannot_reach_level` | State a level inside the reachable range, supply a signal with a larger likelihood ratio, or allow more observations; the refusal states the most extreme reachable value. |
| `observation_query.measured_as_construct` | `proxy_offered_as_construct` | Declare a measurement model (a confusion matrix or reliability) relating the indicator to the construct and condition through it, or ask the question about the recorded indicator itself. |
| `observation_query.measured_background` | `route_not_supported` | Hold recorded values fixed; condition on a measured signal in a conditional-outcome query. |
| `observation_query.measurement_route` | `route_not_supported` | Declare a binary-misclassification or categorical-confusion measurement model. |
| `observation_query.mixed_conditioning_and_intervention` | `route_not_supported` | Ask the conditional and the interventional claims separately; a distribution conditioning on observations after an intervention has no bounded route. |
| `observation_query.multiple_measured_signals` | `route_not_supported` | Condition on one measured signal per query; several indicators need a joint measurement model. |
| `observation_query.outcome_assigned` | `invalid_argument` | Ask about an outcome that is not assigned in the same query. |
| `observation_query.population_mismatch` | `measurement_binding_mismatch` | Establish the measurement model for this population, or declare the invariance premise under a transport contract. |
| `observation_query.positive_level` | `invalid_argument` | Declare which level of the construct is the positive class. |
| `observation_query.regime_mismatch` | `measurement_binding_mismatch` | Establish the measurement model for this regime, or declare the invariance premise under a transport contract. |
| `observation_query.repeated_slot` | `invalid_argument` | Assign each variable at most once. |
| `observation_query.request_unsupported` | `route_not_supported` | State the query within the bounded finite family. |
| `observation_query.unknown_level` | `invalid_argument` | Name a declared level in the variable's level coding. |
| `observation_query.unknown_variable` | `unknown_variable` | Name a variable of the composed model. |
| `observation_query.zero_probability_evidence` | `invalid_argument` | Condition on values the declared model gives positive probability, or revise the model's support. |

### Scientific bridges

`scientific_bridge.*`. A bridge states the assumptions under which one quantity answers a question about another; a refusal names the declaration, evidence or route that would license the request. A mean, predictive law, posterior, identified set and joint outcome law are never interchanged.

| Detail | Code | What would license the request |
| --- | --- | --- |
| `scientific_bridge.artifact_invalid` | `invalid_argument` | Supply an intact scientific_bridge_v1 artifact. |
| `scientific_bridge.association_as_intervention` | `bridge_substitution_refused` | Declare and name the identification under which the predictive quantity equals the causal quantity, or identify the causal quantity from the data. |
| `scientific_bridge.assumption_evidence_missing` | `bridge_assumption_missing` | Name the identity of the evidence the assumption rests on. |
| `scientific_bridge.assumption_missing` | `bridge_assumption_missing` | Declare the assumption with an identity and name the evidence it rests on. |
| `scientific_bridge.blank_field` | `scientific_bridge_invalid` | Name every identity and statement in at most the declared number of bytes. |
| `scientific_bridge.conditioning_mismatch` | `bridge_binding_mismatch` | Bridge quantities with the same conditioning. |
| `scientific_bridge.degenerate_validation` | `bridge_mapping_not_licensed` | Validate in trials that differ in their effects. |
| `scientific_bridge.differential_error` | `differential_error_not_pooled` | Bridge through a non-differential model, or analyse the arms with their own rates. |
| `scientific_bridge.duplicate_assumption` | `scientific_bridge_invalid` | Give every declared assumption its own identity. |
| `scientific_bridge.endpoint_classes` | `scientific_bridge_invalid` | Connect the class pair the bridge kind relates. |
| `scientific_bridge.functional_not_supported` | `bridge_mapping_not_licensed` | State the quantity as a prevalence or risk on the probability scale. |
| `scientific_bridge.horizon_mismatch` | `bridge_binding_mismatch` | Bridge quantities of one horizon. |
| `scientific_bridge.identity_mismatch` | `route_not_supported` | Supply the artifact whose recomputed identity equals the retained identity, or regenerate it from the intended bridge. |
| `scientific_bridge.interval_not_licensed` | `cell_not_licensed` | None available yet: a bridged claim is a point value; whole-method coverage of a bridge is not measured. |
| `scientific_bridge.invalid_quantity` | `scientific_bridge_invalid` | Name the variable, units, population, regime, functional and transform with canonical conditioning. |
| `scientific_bridge.malformed_mapping` | `scientific_bridge_invalid` | Declare the mapping within its range. |
| `scientific_bridge.mapping_not_allowed` | `scientific_bridge_invalid` | Carry the value by a mapping the bridge kind allows. |
| `scientific_bridge.mapping_scope_mismatch` | `bridge_binding_mismatch` | Establish the model for the bridge's scope, or declare the evidence scope the model was established for. |
| `scientific_bridge.measurement_model_mismatch` | `bridge_binding_mismatch` | Bridge the quantities the measurement model relates. |
| `scientific_bridge.measurement_route_not_supported` | `bridge_mapping_not_licensed` | Use a binary-misclassification model, or its own corrected-functional record. |
| `scientific_bridge.model_identity_mismatch` | `bridge_binding_mismatch` | Name the identity of the measurement model the mapping uses. |
| `scientific_bridge.model_score_as_probability` | `bridge_substitution_refused` | Use the model's calibrated predictive law where it supplies one; no calibration bridge is licensed for a score. |
| `scientific_bridge.non_finite_value` | `scientific_bridge_invalid` | Bridge finite point values with finite mapping numbers. |
| `scientific_bridge.outside_attainable_range` | `bridge_mapping_not_licensed` | Check the declared rates and the recorded value. |
| `scientific_bridge.outside_validated_range` | `bridge_mapping_not_licensed` | Validate trials that cover the effect; extrapolation is not licensed. |
| `scientific_bridge.population_mismatch` | `bridge_binding_mismatch` | Bridge quantities of one population; population transport is a separate declared route. |
| `scientific_bridge.predictive_importance_as_causal_effect` | `bridge_substitution_refused` | Ask the causal question directly: identify and estimate the effect of the feature under a declared graph. |
| `scientific_bridge.provenance` | `scientific_bridge_invalid` | Derive a claim whose identity differs from its source. |
| `scientific_bridge.proxy_as_construct` | `proxy_offered_as_construct` | State a measurement model that relates the measured quantity to the construct and bridge through its correction. |
| `scientific_bridge.quantity_mismatch` | `bridge_binding_mismatch` | Supply a claim about the contract's source quantity. |
| `scientific_bridge.regime_mismatch` | `bridge_binding_mismatch` | Bridge quantities of one regime. |
| `scientific_bridge.replay_mismatch` | `invalid_argument` | Regenerate the artifact from the contract and the source claim. |
| `scientific_bridge.scientific_endpoint_as_utility` | `bridge_substitution_refused` | Declare the utility function that maps the endpoint to decision utility and bridge through it. |
| `scientific_bridge.source_not_point` | `cell_not_licensed` | Bridge a claim that carries one finite point value. |
| `scientific_bridge.stated_preference_as_realized_behavior` | `bridge_substitution_refused` | Declare the behavioral linkage by which the stated preference determines the realized choice and name its evidence. |
| `scientific_bridge.too_many_assumptions` | `scientific_bridge_invalid` | Declare at most the bounded number of assumptions. |
| `scientific_bridge.transport_premise_mismatch` | `bridge_binding_mismatch` | Declare the premise for exactly the scope the bridge was established in and the scope the claim is about. |
| `scientific_bridge.transport_premise_missing` | `bridge_assumption_missing` | Declare the transportability premise with its identity and evidence, or establish the bridge in the claim's own population. |
| `scientific_bridge.transport_premise_unused` | `bridge_binding_mismatch` | Remove the premise, or bridge a claim about another scope. |
| `scientific_bridge.units_mismatch` | `bridge_binding_mismatch` | Supply the claim in the declared units; no implicit conversion is made. |
| `scientific_bridge.validation_r_squared_too_low` | `bridge_mapping_not_licensed` | Validate a better surrogate, or declare the coefficient of determination the evidence supports. |
| `scientific_bridge.validation_trials_too_few` | `bridge_mapping_not_licensed` | Validate the surrogate in at least the declared number of trials. |
| `scientific_bridge.value_kind_not_carried` | `cell_not_licensed` | Only a point mean is bridged; transforming a law, posterior, identified set or joint law needs its own licensed route. |
| `scientific_bridge.value_kind_substitution` | `bridge_substitution_refused` | Supply the requested object from its own route; these are not interchangeable. |
| `scientific_bridge.youden_too_small` | `measurement_not_identified` | A more accurate measurement or a larger declared minimum-evidence study is needed. |

## Remedies

Some refusals also name what the caller can change to proceed. The remedy is
a separate, optional text field, never part of the message, and it never adds
or changes a reason code: Rust reads it with `EstimationError::remedy()` (and
`CausalError::remedy()` at the facade), Python with the `remedy` attribute
every `CausalError` carries. A refusal that names no remedy reads `None`.
Estimation errors are not serialized into artifacts, so no artifact or wire
format changes with it.

| Refusal | Remedy |
| --- | --- |
| Additive-GAM non-convergence (`additive GAM target did not converge`, `additive GAM nuisance did not converge`, and the weighted nuisance form) | Standardize the treatment and adjustment columns, drop or coarsen near-collinear or near-constant adjustment covariates, or raise `nuisance_lambda` / lower `nuisance_basis` in the response options. |
| `plug-in response Jacobian supports at most two treatments` | Query the Jacobian over at most two treatments at a time (one `ResponseJacobian` per pair, with the remaining treatments in the adjustment set where the graph licenses it), or an `AverageDerivative` per treatment. |
| `plug-in directional derivative supports at most two treatments` | Restrict the direction to at most two treatments, or query a `ResponseJacobian` per pair and take the inner product with the direction. |
| `ConditionalLinearAdjustment currently supports one effect modifier` (and `conditional arm scores require AllObserved and exactly one modifier` when several modifiers were asked for) | The frequentist interaction regression and its calibrated interval cover one modifier. For several, use the Bayesian conditional estimator (`.estimator(EstimatorId::BayesianConditional).inference(InferenceMode::Bayesian(..))`), hand the identified adjustment set to an external CATE learner with `antecedent.handoff.econml(result, modifiers=[...])`, or fit one `ConditionalEffect` per modifier. |
| Python `ConditionalEffect takes one modifier column` (a sequence passed as `modifier`) | Use `antecedent.handoff.econml(result, modifiers=[...])` or one `ConditionalEffect` per modifier; the multi-modifier Bayesian conditional estimator is reachable from Rust (`EstimatorId::BayesianConditional`). |

```python
query = antecedent.ResponseJacobian(["a", "b", "c"], ["y"], at=[0.0, 0.0, 0.0])
try:
    antecedent.analyze(data, query=query, graph=graph)
except antecedent.CausalError as err:
    print(err)         # plug-in response Jacobian supports at most two treatments
    print(err.remedy)  # query the Jacobian over at most two treatments at a time ...
```
