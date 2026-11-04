# Refusal codes and remedies

The reference catalog of refusal families, codes and remedies. For how refusals reach your code and what each category means, read [refusals](../concepts/refusals.md). Each detail below is registered with its top-level code and a remedy in `parity/capability_records.toml`.

## Structured refusals and their remedies

Scientific stage refusals share the structured
base, `antecedent.errors.StructuredRefusal` (itself a `CausalUnsupportedError`).
It exposes machine-readable fields, so you branch on fields rather than parse
a message. Input type, value and resource errors can instead raise ordinary
`CausalTypeError`, `CausalValueError` or `CausalResourceError` without a scientific
reason code.

| Field | Meaning |
| --- | --- |
| `code` (alias `reason_code`) | the registered refusal code, such as `not_supported`, `route_not_supported`, `transport_proven_non_transportable` or `invalid_argument` |
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
    elif refusal.code == "not_supported":
        ...                      # a unit lacking evidence (release build); the detail names what is missing
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
| `route_not_supported` / `*.composition_not_supported` | Report the sampling interval next to the assumption range; do not compose them. No composition method is supported. |
| `route_not_supported` / `transported_counterfactual.nonadditive_mechanism` and its siblings | Declare the premise in `Premises` if you can defend it, or choose an assignment inside the affine-additive class. |
| `decision_contract_unsatisfied` / `*.transported_not_accepted` | The consumer was not declared to accept a transported input. Accept it explicitly (an inverse query's `accepts_transported=True`, a decision support policy that names `extrapolative`) and the result carries the transport receipt, or supply native target data. |
| `transport_support_failure` / `transported_decision.support_rule_unmet` | A support rule of the decision contract is stronger than the transported label can meet; weaken the declared rule or obtain target evidence. |
| `transport_proven_non_transportable` | Read `witness`: two models agree on the source and differ on the target. Remove the selection on that mechanism or supply target evidence for it. |
| `decision_contract_unsatisfied` / `native_claims.source_not_supplied` | A mean cannot answer a quantile or probability. Supply aligned joint draws. |
| `route_not_supported` / `decision_evaluation.mean_source_not_replayable` | `Decision.export()` requires aligned joint draws; `decision.replay` uses that same law. For a bound external mean claim, `composition_bundle.mean_decision` supplies a distinct point-only bundle artifact. Native mean workflows retain their own source-bound recalculation/replay contracts. |
| `design_cost_units_mismatch` | Supply a `CostMap` from the study cost unit to the decision's utility unit, or consume under the cost map the ranking used. |
| `score_table_unavailable` / `recalc.unavailable_data` | The request needs the data or a fit; supply the data and refit, or ask only for a same-row reweighting of the frozen scores. |
| `composition_bundle.node_not_found` | Add the node before connecting it (`NodeNotFoundRefusal.offending` is the missing id). |
| `invalid_argument` | Fix the argument named in `detail` and `offending`. |
| `positivity_violated` (`propensity.boundary_saturated`) | The propensity model gives some rows a probability within 1e-8 of 0 or 1 and no overlap policy is declared. Declare `Overlap(trim=...)` to estimate on the common-support population (`estimate.overlap_excluded_fraction` reports the exclusion), declare a clip that bounds the weights, restrict the population, or obtain a design that produces both treatments. `refusal_fields` carries `boundary_count` and `boundary_margin`. |

See also the [refusal recipes](../guides/refusal-recipes.md) for the refusal-handling pattern in
context.

## Measurement, composition, evidence and counterfactual-transport refusals

Each detail below is registered with its top-level code and a remedy in `parity/capability_records.toml`. The remedy says what evidence, assumption or declaration would support the request; a refusal that has no support declaration yet says so. A refusal is never a verdict on the science when it is a cancellation, a budget stop or an invalid argument.

### Measurement models

`measurement_model.*`. A measured column is related to a latent construct only through a declared model; a refusal names the declaration or evidence that would support the request.

| Detail | Code | What would support the request |
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
| `measurement_model.intervention_meaning_not_carried` | `representation_mapping_required` | Declare evidence that interventions correspond across the mapping; a relabeling carries values only. |
| `measurement_model.invalid_model` | `measurement_model_invalid` | Declare a well formed model: named identities and parameters of the declared route within range. |
| `measurement_model.learned_coordinates_not_intervenable` | `representation_mapping_required` | Declare the intervention on the scientific construct; coordinates of a learned representation have no intervention meaning of their own. |
| `measurement_model.mapping_endpoints` | `representation_mapping_required` | Supply a mapping that runs exactly from the model's representation version to the requested one. |
| `measurement_model.operation_not_supported` | `route_not_supported` | Request an operation in the route's allowed set (see its route card). |
| `measurement_model.parameter` | `measurement_model_invalid` | Supply parameters within their ranges (probabilities in [0, 1], positive variances). |
| `measurement_model.population_mismatch` | `measurement_binding_mismatch` | Analyse data of the population the model names, or declare a model for this population. |
| `measurement_model.posterior_capability` | `measurement_posterior_not_trusted` | Declare the sampling operation; the route consumes joint posterior draws. |
| `measurement_model.posterior_declaration` | `measurement_posterior_not_trusted` | Complete the provider declaration so it is consistent with the object. |
| `measurement_model.posterior_identity` | `measurement_posterior_not_trusted` | Supply the exact provider object the model names. |
| `measurement_model.posterior_kind` | `measurement_posterior_not_trusted` | Supply the latent variable as a parameter-posterior object. |
| `measurement_model.posterior_native_trust` | `measurement_posterior_not_trusted` | Attest or verify the provider object; a provider object is never natively supported. |
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
| `measurement_model.unrecognized` | `measurement_model_invalid` | Report the construction failure; no registered declaration would support it. |
| `measurement_model.variance_not_positive` | `measurement_not_identified` | Supply data and parameters under which the corrected variance is positive (more units or a less degenerate design). |
| `measurement_model.youden_too_small` | `measurement_not_identified` | Use a more accurate indicator or a validation study that shows a Youden index above the declared minimum. |

### Mechanism composition

`mechanism_composition.*`. Individually valid mechanisms compose only when the declarations establish one coherent joint law; every defect names the declaration that would support the composition.

| Detail | Code | What would support the request |
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

`evidence_ledger.*`. Overlap that is not declared stays unknown and is never treated as independent; a refusal names the declaration, covariance or recomputation that would support the combination.

| Detail | Code | What would support the request |
| --- | --- | --- |
| `evidence_ledger.cancelled` | `cancelled_no_claim` | Rerun without cancellation; the ledger is unchanged and no verdict was formed. |
| `evidence_ledger.diagonal_covariance_shared_rows` | `evidence_double_counted` | Supply the cross covariance for sources that share observations. |
| `evidence_ledger.duplicate_result` | `invalid_argument` | Cite the existing result rather than registering the combination twice. |
| `evidence_ledger.duplicate_row` | `evidence_double_counted` | Declare each observation once within a source. |
| `evidence_ledger.duplicate_source` | `evidence_double_counted` | Cite the existing source rather than adding the same evidence twice. |
| `evidence_ledger.identity_mismatch` | `route_not_supported` | Supply the artifact whose recomputed identity equals the retained identity, or regenerate it from the intended ledger history. |
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

| Detail | Code | What would support the request |
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
| `counterfactual_transport.no_sampled_inference` | `route_not_supported` | Use the sampled counterfactual transport route for counted, empirical or posterior laws. |
| `counterfactual_transport.selected_sample_law` | `route_not_supported` | Supply laws of the population itself; this route has no sampling-selection model. |
| `counterfactual_transport.support_failure` | `transport_support_failure` | Supply laws with positive mass on every conditioning event the formula divides by. |
| `counterfactual_transport.zero_conditioning_mass` | `transport_support_failure` | Condition on an event with positive mass in the target population. |

### Counterfactual transport from counts

`counterfactual_transport_sampled.*`. Sampled evidence is point-only with an uncalibrated standard error; independence of tables is never assumed.

| Detail | Code | What would support the request |
| --- | --- | --- |
| `counterfactual_transport_sampled.bounds_exceeded` | `route_not_supported` | Reduce the stored table counts, table sizes or search and evaluation limits to the consumer's bounds, or consume the artifact with higher limits within the contract maxima. |
| `counterfactual_transport_sampled.cancelled` | `transport_budget_cancel` | Rerun without cancellation; no estimate was formed. |
| `counterfactual_transport_sampled.envelope_budget` | `transport_budget_cancel` | Declare a smaller unseen-mass family or raise the envelope evaluation limit within the bound. |
| `counterfactual_transport_sampled.envelope_undefined` | `transport_support_failure` | Declare an unseen-mass family in which some member has a defined value, or supply more observations. |
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

`observation_query.*`. A conditioning claim and an interventional claim are distinct, and a recorded indicator value is not the construct it measures; a refusal names the declaration or evidence that would support the request. See [observation-queries.md](../observation-queries.md).

| Detail | Code | What would support the request |
| --- | --- | --- |
| `observation_query.arm_mismatch` | `measurement_binding_mismatch` | Declare a model for this arm, or ask about the arm the model covers. |
| `observation_query.artifact_invalid` | `invalid_argument` | Supply an intact observation_query_v1 artifact of the supported layout. |
| `observation_query.artifact_refused` | `invalid_argument` | Supply the artifact whose identity equals the retained one and whose result recomputes from its premises. |
| `observation_query.blank_field` | `invalid_argument` | Name the outcome variable and the variable and level of every slot. |
| `observation_query.no_calibrated_probability` | `uncertainty_representation_insufficient` | A probability from a model fitted to a sample is a point value with no claim about how often it is right. |
| `observation_query.cancelled` | `cancelled_no_claim` | Rerun without cancellation; the cancellation is not a verdict on the model. |
| `observation_query.construct_mismatch` | `measurement_binding_mismatch` | Request the construct the model names, or declare a model for the construct asked about. |
| `observation_query.construct_observed_directly` | `invalid_argument` | Condition on the construct directly or on its measured indicator, not both. |
| `observation_query.differential_error_unsupported` | `differential_error_not_pooled` | Use a non-differential model; error that differs has no bounded conditioning route and is never pooled. |
| `observation_query.indicator_kind` | `invalid_argument` | Use a direct slot for an error-free recorded value, or declare the indicator's measurement kind. |
| `observation_query.indicator_mismatch` | `measurement_binding_mismatch` | Declare a model for this indicator. |
| `observation_query.declared_model_has_no_sampling_interval` | `uncertainty_representation_insufficient` | A declared model's posterior probability is an exact number with no sampling distribution, so it has no sampling interval; for a probability computed from tables fitted to counts, request the count-fitted interval, which reports posterior quantiles with a calibration status. |
| `observation_query.intervention_in_conditioning_slot` | `quantity_semantics_mismatch` | State an assigned value as an intervention query; a conditioning slot holds only recorded values, and the observed and interventional laws are separate claims. |
| `observation_query.invalid_declaration` | `invalid_argument` | Supply named levels and distributions that sum to one, with probabilities in range. |
| `observation_query.level_unreachable` | `evidence_cannot_reach_level` | State a level inside the reachable range, supply a signal with a larger likelihood ratio, or allow more observations; the refusal states the most extreme reachable value. |
| `observation_query.measured_as_construct` | `proxy_offered_as_construct` | Declare a measurement model (a confusion matrix or reliability) relating the indicator to the construct and condition through it, or ask the question about the recorded indicator itself. |
| `observation_query.measured_background` | `route_not_supported` | Hold recorded values fixed; condition on a measured signal in a conditional-outcome query. |
| `observation_query.measurement_route` | `route_not_supported` | Declare a binary-misclassification or categorical-confusion measurement model. |
| `observation_query.mixed_conditioning_and_intervention` | `route_not_supported` | Condition on recorded variables that are not intervened on (a descendant of the intervention carries information about the unobserved causes of the outcome), or ask the interventional claim alone; a variable whose mechanism an intervention removes is set, not observed. |
| `observation_query.levels_for_continuous_model` | `invalid_argument` | State the evidence and the interventions of a linear-Gaussian composition as typed real values. |
| `observation_query.real_values_for_finite_model` | `invalid_argument` | State the evidence and the interventions of a finite composition as levels of the variable. |
| `observation_query.value_not_finite` | `invalid_argument` | Supply a finite real value. |
| `observation_query.evidence_covariance_singular` | `invalid_argument` | Observe values whose joint covariance under the model is positive definite: drop an observed variable that is a deterministic function of the others, or give the mechanism a positive residual scale. |
| `observation_query.measured_signal_not_redrawn` | `route_not_supported` | Ask the point claim of the measured query, or condition on directly recorded values; a measured signal's error rates are declared, not fitted from retained data. |
| `observation_query.no_parameter_draws` | `invalid_argument` | Ask for parameter uncertainty with a retained-data redraw. |
| `observation_query.unknown_quantity` | `invalid_argument` | Name one of the quantities of the claim's parameter draws. |
| `observation_query.invalid_interval` | `invalid_argument` | Supply finite draws and a level strictly between zero and one. |
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

`scientific_bridge.*`. A bridge states the assumptions under which one quantity answers a question about another; a refusal names the declaration, evidence or route that would support the request. A mean, predictive law, posterior, identified set and joint outcome law are never interchanged.

| Detail | Code | What would support the request |
| --- | --- | --- |
| `scientific_bridge.artifact_invalid` | `invalid_argument` | Supply an intact scientific_bridge_v1 artifact. |
| `scientific_bridge.association_as_intervention` | `bridge_substitution_refused` | Declare and name the identification under which the predictive quantity equals the causal quantity, or identify the causal quantity from the data. |
| `scientific_bridge.assumption_evidence_missing` | `bridge_assumption_missing` | Name the identity of the evidence the assumption rests on. |
| `scientific_bridge.assumption_missing` | `bridge_assumption_missing` | Declare the assumption with an identity and name the evidence it rests on. |
| `scientific_bridge.blank_field` | `scientific_bridge_invalid` | Name every identity and statement in at most the declared number of bytes. |
| `scientific_bridge.chain_stage_not_affine` | `route_not_supported` | Derive the bridged draws and supply them to the chain as a source; a mapping with no chain expression is not a stage. |
| `scientific_bridge.conditioning_mismatch` | `bridge_binding_mismatch` | Bridge quantities with the same conditioning. |
| `scientific_bridge.degenerate_validation` | `bridge_mapping_not_supported` | Validate in trials that differ in their effects. |
| `scientific_bridge.dependence_not_a_covariance` | `bridge_mapping_not_supported` | State a covariance whose square is at most the product of the slope variance and the source variance. |
| `scientific_bridge.differential_error` | `differential_error_not_pooled` | Bridge through a non-differential model, or analyse the arms with their own rates. |
| `scientific_bridge.draws_meaning_mismatch` | `distribution_meaning_mismatch` | Declare the value kind the draws are: a posterior or a predictive law; a bootstrap, sampling or empirical law is neither. |
| `scientific_bridge.draws_not_aligned` | `joint_law_required` | Supply the draws as columns of one aligned joint draw table. |
| `scientific_bridge.draws_outside_domain` | `bridge_mapping_not_supported` | The bridge neither truncates nor extrapolates a law; validate trials that cover the draws, or check the declared rates. |
| `scientific_bridge.draws_unusable` | `scientific_bridge_invalid` | Supply finite draws that vary and a target quantity the table does not already carry. |
| `scientific_bridge.duplicate_assumption` | `scientific_bridge_invalid` | Give every declared assumption its own identity. |
| `scientific_bridge.endpoint_classes` | `scientific_bridge_invalid` | Connect the class pair the bridge kind relates. |
| `scientific_bridge.functional_not_supported` | `bridge_mapping_not_supported` | State the quantity as a prevalence or risk on the probability scale. |
| `scientific_bridge.horizon_mismatch` | `bridge_binding_mismatch` | Bridge quantities of one horizon. |
| `scientific_bridge.identity_mismatch` | `route_not_supported` | Supply the artifact whose recomputed identity equals the retained identity, or regenerate it from the intended bridge. |
| `scientific_bridge.interval_not_supported` | `route_not_supported` | None available yet: a bridged claim is a point value; whole-method coverage of a bridge is not measured. |
| `scientific_bridge.invalid_quantity` | `scientific_bridge_invalid` | Name the variable, units, population, regime, functional and transform with canonical conditioning. |
| `scientific_bridge.malformed_mapping` | `scientific_bridge_invalid` | Declare the mapping within its range. |
| `scientific_bridge.mapping_not_allowed` | `scientific_bridge_invalid` | Carry the value by a mapping the bridge kind allows. |
| `scientific_bridge.mapping_scope_mismatch` | `bridge_binding_mismatch` | Establish the model for the bridge's scope, or declare the evidence scope the model was established for. |
| `scientific_bridge.mapping_uncertainty_invalid` | `scientific_bridge_invalid` | Declare the mapping's uncertainty once: a trial-level line and a correction draw it from their own evidence. |
| `scientific_bridge.measurement_model_mismatch` | `bridge_binding_mismatch` | Bridge the quantities the measurement model relates. |
| `scientific_bridge.measurement_route_not_supported` | `bridge_mapping_not_supported` | Use a binary-misclassification model, or its own corrected-functional record. |
| `scientific_bridge.model_identity_mismatch` | `bridge_binding_mismatch` | Name the identity of the measurement model the mapping uses. |
| `scientific_bridge.model_score_as_probability` | `bridge_substitution_refused` | Use the model's calibrated predictive law where it supplies one; no calibration bridge is supported for a score. |
| `scientific_bridge.non_finite_value` | `scientific_bridge_invalid` | Bridge finite point values with finite mapping numbers. |
| `scientific_bridge.outside_attainable_range` | `bridge_mapping_not_supported` | Check the declared rates and the recorded value. |
| `scientific_bridge.outside_validated_range` | `bridge_mapping_not_supported` | Validate trials that cover the effect; extrapolation is not supported. |
| `scientific_bridge.population_mismatch` | `bridge_binding_mismatch` | Bridge quantities of one population; population transport is a separate declared route. |
| `scientific_bridge.predictive_importance_as_causal_effect` | `bridge_substitution_refused` | Ask the causal question directly: identify and estimate the effect of the feature under a declared graph. |
| `scientific_bridge.provenance` | `scientific_bridge_invalid` | Derive a claim whose identity differs from its source. |
| `scientific_bridge.proxy_as_construct` | `proxy_offered_as_construct` | State a measurement model that relates the measured quantity to the construct and bridge through its correction. |
| `scientific_bridge.quantity_mismatch` | `bridge_binding_mismatch` | Supply a claim about the contract's source quantity. |
| `scientific_bridge.regime_mismatch` | `bridge_binding_mismatch` | Bridge quantities of one regime. |
| `scientific_bridge.replay_mismatch` | `invalid_argument` | Regenerate the artifact from the contract and the source claim. |
| `scientific_bridge.scientific_endpoint_as_utility` | `bridge_substitution_refused` | Declare the utility function that maps the endpoint to decision utility and bridge through it. |
| `scientific_bridge.source_dependence_missing` | `sampling_dependence_unknown` | Declare that the source and the mapping are independent, or declare their covariance. |
| `scientific_bridge.source_not_draws` | `bridge_mapping_not_supported` | Bridge the aligned draws of a posterior or predictive law; an associational claim is never a bridge source. |
| `scientific_bridge.source_not_point` | `bridge_mapping_not_supported` | Bridge a claim that carries one finite point value. |
| `scientific_bridge.stated_preference_as_realized_behavior` | `bridge_substitution_refused` | Declare the behavioral linkage by which the stated preference determines the realized choice and name its evidence. |
| `scientific_bridge.transport_premise_mismatch` | `bridge_binding_mismatch` | Declare the premise for exactly the scope the bridge was established in and the scope the claim is about. |
| `scientific_bridge.transport_premise_missing` | `bridge_assumption_missing` | Declare the transportability premise with its identity and evidence, or establish the bridge in the claim's own population. |
| `scientific_bridge.transport_premise_unused` | `bridge_binding_mismatch` | Remove the premise, or bridge a claim about another scope. |
| `scientific_bridge.units_mismatch` | `bridge_binding_mismatch` | Supply the claim in the declared units; no implicit conversion is made. |
| `scientific_bridge.validation_r_squared_too_low` | `bridge_mapping_not_supported` | Validate a better surrogate, or declare the coefficient of determination the evidence supports. |
| `scientific_bridge.validation_trials_too_few` | `bridge_mapping_not_supported` | Validate the surrogate in at least the declared number of trials. |
| `scientific_bridge.value_kind_not_carried` | `bridge_mapping_not_supported` | A point mean takes the point route and a posterior or predictive law takes the draws route; an identified set or joint law has no executed bridge. |
| `scientific_bridge.value_kind_substitution` | `bridge_substitution_refused` | Supply the requested object from its own route; these are not interchangeable. |
| `scientific_bridge.youden_too_small` | `measurement_not_identified` | A more accurate measurement or a larger declared minimum-evidence study is needed. |

## Transport, uncertainty and reach refusals

Each detail below is registered with its top-level code and a remedy in `parity/capability_records.toml`, like the sections above. These are the refusals of the transport, decision-reach and uncertainty routes. Every one is a statement that a premise or declaration was missing; none is a failure of the data.

### Transported cohort effect and measurement transport

`transported_effect.*`, `measurement_transport.*`. A retained effect model or measurement model is carried to a declared target; the value is point-only. See [the guide](../guides/transported-decisions-and-measurement.md).

| Detail | Code | What would support the request |
| --- | --- | --- |
| `transported_effect.cohort_invalid` | `invalid_argument` | Declare a target cohort with named finite feature rows and normalized weights. |
| `transported_effect.weights_invalid` | `invalid_argument` | Supply finite nonnegative weights. |
| `transported_effect.weights_not_normalized` | `invalid_argument` | Normalize the weights to sum to one. |
| `transported_effect.feature_schema_mismatch` | `schema_mismatch` | Supply exactly the model's named features: none missing and none extra. |
| `transported_effect.population_not_different` | `quantity_semantics_mismatch` | Decide it as a within-population cohort with no shift declaration. |
| `transported_effect.shift_declaration_required` | `transport_missing_evidence` | Declare the populations, the selection node, the variables it points into and the effect modifiers. |
| `transported_effect.declaration_invalid` | `invalid_argument` | Name the populations, the selection node and the variables it points into. |
| `transported_effect.shift_declaration_mismatch` | `transport_not_certified` | Declare the model's population as source and the cohort's population as target. |
| `transported_effect.selection_enters_mechanism` | `transport_not_certified` | Let the selection node point only into declared effect modifiers; otherwise refit in the target population. |
| `transported_effect.modifier_set_mismatch` | `transport_not_certified` | Declare exactly the model's features as the effect modifiers. |
| `transported_effect.model_invalid` | `invalid_argument` | Refit or re-export the model and name every retained feature once. |
| `transported_effect.link_not_identity` | `quantity_semantics_mismatch` | Retain the identity-scale conditional-contrast map. |
| `transported_effect.support_contract_absent` | `invalid_argument` | Refit with a producer that retains the fitted-row support contract. |
| `transported_effect.no_overlap` | `transport_support_failure` | Supply a model fitted on rows that cover the target population. |
| `transported_effect.out_of_support` | `transport_support_failure` | Restrict the cohort (a new cohort) or fit on rows that cover the target. |
| `transported_effect.prediction_refused` | `invalid_argument` | Check the cohort features against the model. |
| `transported_effect.non_finite_result` | `invalid_argument` | Check the model and the cohort. |
| `transported_effect.interval_not_supported` | `route_not_supported` | None available: the value is point-only under a declared premise until whole-method coverage including the transport and support steps is measured. |
| `transported_effect.law_not_supplied` | `decision_contract_unsatisfied` | Supply aligned outcome draws under a declared transport model. |
| `measurement_model.no_model` | `measurement_model_invalid` | Supply the source model the premise names. |
| `measurement_model.invariance_premise_required` | `transport_missing_evidence` | Declare which error parameters are invariant, declare them differential, or validate in the target population. |
| `measurement_model.differential_population_unestablished` | `differential_error_not_pooled` | Obtain a validation sample in the target population. |
| `measurement_model.invariance_target_mismatch` | `transport_not_certified` | Redeclare the premise for the data's population. |
| `measurement_model.invariance_source_unknown` | `transport_not_certified` | Supply the source model. |
| `measurement_model.invariance_not_declared` | `transport_not_certified` | Declare each parameter named in the message. |
| `measurement_model.nothing_to_transport` | `route_not_supported` | Use the target replicates directly. |
| `measurement_model.posterior_bound_to_snapshot` | `route_not_supported` | Request a new provider posterior for the target snapshot. |

### Mechanism transport, synthesis and bridge premise

`mechanism_transport.*`, `typed_mechanism_transport.*`, `transported_synthesis.*`, `scientific_bridge.*`. See [the guide](../guides/mechanism-and-synthesis-transport.md).

| Detail | Code | What would support the request |
| --- | --- | --- |
| `mechanism_transport.origin_not_fitted` | `mechanism_composition_incoherent` | Carry only mechanisms estimated from retained evidence; an external or declared mechanism needs its own admission. |
| `mechanism_transport.family_unsupported` | `mechanism_composition_incoherent` | Decide a finite table with the finite counterfactual transport derivation; refit other families in the target. |
| `mechanism_transport.diagram_source_mismatch` | `quantity_semantics_mismatch` | Declare the diagram between that population and the target. |
| `mechanism_transport.diagram_target_mismatch` | `quantity_semantics_mismatch` | Declare the diagram between the source population and this target. |
| `mechanism_transport.target_mechanism_not_native` | `quantity_semantics_mismatch` | Supply as a target mechanism only one fitted in the target; carry the others as source mechanisms. |
| `mechanism_transport.selection_arrow_into_latent` | `quantity_semantics_mismatch` | Refit the mechanism in the target population. |
| `mechanism_transport.duplicate_mechanism` | `quantity_semantics_mismatch` | Give each variable exactly one mechanism per role. |
| `typed_mechanism_transport.shifted_without_target_evidence` | `transport_missing_evidence` | A selection node points into this variable, so its mechanism differs in the target; fit it on target data, it is never carried. |
| `typed_mechanism_transport.fit_refused` | `mechanism_composition_incoherent` | Check the specification and the data's support. |
| `typed_mechanism_transport.fit_not_converged` | `mechanism_fit_not_converged` | Coarsen or drop parents that separate the response, or fix the singular design. |
| `typed_mechanism_transport.invariance_contradicted` | `quantity_semantics_mismatch` | Target data reject the declared invariance of this mechanism; declare a selection node into it and refit it in the target. |
| `typed_mechanism_transport.target_specification_differs` | `quantity_semantics_mismatch` | Refit the target mechanism with the source mechanism's family, mean and parents. |
| `typed_mechanism_transport.design_not_shared` | `quantity_semantics_mismatch` | Refit the target mechanism on the source mechanism's design with `refit_in_target`. |
| `typed_mechanism_transport.parents_changed` | `quantity_semantics_mismatch` | A shifted mechanism's target fit keeps the declared parent set. |
| `typed_mechanism_transport.regime_not_declared_invariant` | `quantity_semantics_mismatch` | Declare the invariance of this mechanism across the two regimes explicitly. |
| `typed_mechanism_transport.selection_target_unknown_variable` | `quantity_semantics_mismatch` | A selection node points into a variable no supplied mechanism produces. |
| `typed_mechanism_transport.parent_without_mechanism` | `mechanism_composition_incoherent` | Supply a source or target mechanism for every parent. |
| `typed_mechanism_transport.cycle` | `mechanism_composition_incoherent` | A typed transported composition is acyclic. |
| `typed_mechanism_transport.invariance_level_invalid` | `mechanism_composition_incoherent` | Test declared invariances at a level strictly between 0 and 0.5. |
| `typed_mechanism_transport.unknown_variable` | `mechanism_composition_incoherent` | Name only variables of the transported model in a contrast. |
| `typed_mechanism_transport.dependence_invalid` | `uncertainty_representation_insufficient` | Declare a justified independence, an aligned positive semidefinite joint covariance, or well-shaped cross-covariance blocks. |
| `typed_mechanism_transport.parameter_dependence_unknown` | `uncertainty_representation_insufficient` | Declare independence by design, a joint covariance or cross-covariance blocks. |
| `typed_mechanism_transport.parameter_dependence_incomplete` | `uncertainty_representation_insufficient` | Declare a cross-covariance block for every pair of involved mechanisms, or a joint covariance listing them all. |
| `typed_mechanism_transport.transport_identity_mismatch` | `quantity_semantics_mismatch` | Re-derive the transport from the retained fits and declaration. |
| `transported_synthesis.transport_variance_undeclared` | `transport_missing_evidence` | Declare the transport variance; an undeclared transport is not a zero-variance one. |
| `transported_synthesis.transport_variance_on_native_source` | `invalid_argument` | Drop it, or declare the transport identity in the ledger. |
| `transported_synthesis.source_listed_twice` | `invalid_argument` | Contribute each source once. |
| `transported_synthesis.unknown_source` | `invalid_argument` | Add the source to the ledger first. |
| `transported_synthesis.no_contributions` | `invalid_argument` | Contribute at least one ledger claim. |
| `transported_synthesis.overlap_not_modelled` | `invalid_argument` | Supply a sampling covariance modelling the overlap, or contribute sources with disjoint declared observations. |
| `transported_synthesis.target_population_mismatch` | `quantity_semantics_mismatch` | Transport every claim to the one target population the native evidence is drawn from. |
| `transported_synthesis.transport_identity_incomplete` | `transport_missing_evidence` | State the assumptions, the source populations and a distinct target. |
| `transported_synthesis.native_population_blank` | `invalid_argument` | Name the population of the native evidence. |
| `transported_synthesis.numeric_refused` | `invalid_argument` | Supply finite positive sampling variances and a finite nonnegative, consistent transport variance. |
| `transported_synthesis.shared_transport_needs_covariance` | `joint_law_required` | Pool them with the shared covariance. |
| `transported_synthesis.prior_role_mismatch` | `invalid_argument` | Name one transported prior; pool or borrow to combine several transported claims. |
| `scientific_bridge.transport_premise_contradicted_by_selection_diagram` | `bridge_assumption_missing` | Validate the surrogate in the claim's population, or declare a diagram without a selection node into the terminal outcome and its modifiers. |
| `scientific_bridge.transport_diagram_scope_mismatch` | `bridge_binding_mismatch` | Declare the diagram between those two populations. |

### Transported policy, adaptive trial source and period shift

`transported_policy.*`, `adaptive_transport.*`, `temporal_policy_transport.*`. See [the guide](../guides/transported-decisions-and-measurement.md).

| Detail | Code | What would support the request |
| --- | --- | --- |
| `transported_policy.selection_diagram_missing` | `transport_missing_evidence` | Supply a selection diagram that marks every mechanism of the model invariant or shifted between the source and target populations. |
| `transported_policy.target_initial_law_missing` | `transport_missing_evidence` | Supply the target population's initial law over the model's latent states. |
| `transported_policy.diagram_binding_mismatch` | `transport_missing_evidence` | Declare the diagram for the model's population and the requested target. |
| `transported_policy.mechanism_status_missing` | `transport_missing_evidence` | Declare every mechanism invariant or shifted; invariance is never assumed. |
| `transported_policy.mechanism_not_invariant` | `transport_missing_evidence` | Establish it for the target population, or restrict the shift to the initial-state law. |
| `transported_policy.target_support_outside_source` | `transport_support_failure` | Restrict the target law to the source's support or establish the mechanisms there. |
| `transported_policy.initial_law_declared_invariant` | `invalid_argument` | Mark the initial-state law shifted or supply the source's law. |
| `transported_policy.target_initial_law` | `invalid_argument` | Supply one probability per state summing to one. |
| `transported_policy.target_population` | `invalid_argument` | Name the target population. |
| `transported_policy.target_is_source` | `invalid_argument` | Use the sequential g-formula; there is nothing to transport. |
| `transported_policy.mechanism_unknown` | `invalid_argument` | Declare the model's own mechanisms only. |
| `transported_policy.mechanism_declared_twice` | `invalid_argument` | Declare each mechanism once. |
| `transported_policy.value_mismatch` | `invalid_argument` | Regenerate the evaluation from the request. |
| `temporal_policy_transport.time_invariance_missing` | `transport_missing_evidence` | Declare every mechanism invariant across the periods or shifted, over a stated fixed horizon. |
| `temporal_policy_transport.target_initial_law_missing` | `transport_missing_evidence` | Supply the target period's initial-state law. |
| `temporal_policy_transport.invariance_binding_mismatch` | `transport_missing_evidence` | Declare the invariance for the model's period and the target period. |
| `temporal_policy_transport.mechanism_status_missing` | `transport_missing_evidence` | Declare every mechanism; time invariance is never assumed. |
| `temporal_policy_transport.mechanism_time_varying` | `transport_missing_evidence` | Establish it for the target period, or declare it invariant if that is the premise. |
| `temporal_policy_transport.history_alphabet_incompatible` | `transport_missing_evidence` | Establish the same actions and symbols with the same meanings; the policy table is not re-keyed implicitly. |
| `temporal_policy_transport.horizon_mismatch` | `route_not_supported` | State one fixed horizon; a horizon change is not a bounded route. |
| `temporal_policy_transport.target_support_outside_source` | `transport_support_failure` | Restrict the law to the source's support. |
| `temporal_policy_transport.initial_law_declared_invariant` | `invalid_argument` | Mark the initial-state law shifted or supply the source's law. |
| `temporal_policy_transport.target_initial_law` | `invalid_argument` | Supply one probability per state summing to one. |
| `temporal_policy_transport.target_period` | `invalid_argument` | Name the target period. |
| `temporal_policy_transport.target_is_source` | `invalid_argument` | Use the sequential g-formula. |
| `temporal_policy_transport.mechanism_unknown` | `invalid_argument` | Declare the model's own mechanisms only. |
| `temporal_policy_transport.mechanism_declared_twice` | `invalid_argument` | Declare each mechanism once. |
| `temporal_policy_transport.value_mismatch` | `invalid_argument` | Regenerate the evaluation from the request. |
| `adaptive_transport.assignment_monitoring_contract_missing` | `adaptive_contract_incomplete` | Supply the adaptive design contract: policy history, realized assignment probabilities, exposure history, outcome availability and delay, interim analyses, stopping rule. |
| `adaptive_transport.outcome_missing` | `adaptive_contract_incomplete` | Retain the outcome of every unit whose availability is declared. |
| `adaptive_transport.target_stratum_law_missing` | `transport_missing_evidence` | Supply the target population's law over the strata. |
| `adaptive_transport.selection_diagram_missing` | `transport_missing_evidence` | Supply a diagram marking the stratum law and outcome mechanism. |
| `adaptive_transport.mechanism_status_missing` | `transport_missing_evidence` | Declare both; invariance is never assumed. |
| `adaptive_transport.outcome_mechanism_not_invariant` | `transport_missing_evidence` | Establish it in the target (a target trial) or restrict the shift to the stratum law. |
| `adaptive_transport.diagram_binding_mismatch` | `transport_missing_evidence` | Declare the diagram between the trial population and the target. |
| `adaptive_transport.design_not_adaptive` | `route_not_supported` | Use the fixed-design transport routes. |
| `adaptive_transport.stratum_arm_unobserved` | `transport_support_failure` | Restrict the target law, coarsen strata or collect units there. |
| `adaptive_transport.no_analysed_units` | `transport_support_failure` | The trial holds no observed outcome to transport. |
| `adaptive_transport.assignment_below_floor` | `adaptive_assumption_violated` | Restrict the compared arms or justify a lower floor; nothing is clipped. |
| `adaptive_transport.strata` | `invalid_argument` | Name at least one distinct, non-blank stratum. |
| `adaptive_transport.unit_strata` | `invalid_argument` | Give every enrolled unit a declared stratum. |
| `adaptive_transport.target_stratum_law` | `invalid_argument` | Supply one probability per stratum summing to one. |
| `adaptive_transport.arms` | `invalid_argument` | Name two distinct arms. |
| `adaptive_transport.mechanism_declared_twice` | `invalid_argument` | Declare each once. |
| `adaptive_transport.population_label` | `invalid_argument` | Name both populations. |
| `adaptive_transport.target_is_source` | `invalid_argument` | Use the within-trial estimate; there is nothing to transport. |
| `adaptive_transport.stratum_law_declared_invariant` | `invalid_argument` | Mark the stratum law shifted or supply the trial's law. |
| `adaptive_transport.value_mismatch` | `invalid_argument` | Regenerate from the request. |

### Discovered structure and per-population discovery

`discovered_transport.*`, `selective_discovery.*`. See [the guide](../guides/discovered-structure-transport.md).

| Detail | Code | What would support the request |
| --- | --- | --- |
| `discovered_transport.search_incomplete` | `discovered_structure_unresolved` | Complete the search or declare the graph. |
| `discovered_transport.orientation_conflict` | `discovered_structure_unresolved` | Resolve the conflict or declare the graph. |
| `discovered_transport.selection_target_unknown` | `discovered_structure_unresolved` | Declare selection targets that are variables of the class. |
| `discovered_transport.member_bound_zero` | `discovered_structure_unresolved` | Allow at least one member. |
| `discovered_transport.class_too_large` | `discovered_structure_unresolved` | Restrict the structure or declare a graph. |
| `discovered_transport.no_members` | `discovered_structure_unresolved` | Check the discovered structure. |
| `discovered_transport.pag_class_ambiguous` | `discovered_structure_unresolved` | Rerun discovery or declare the graph. |
| `discovered_transport.pag_selection_bias_edge` | `discovered_structure_unresolved` | Remove the edge with background knowledge or declare the graph. |
| `discovered_transport.temporal_structure_unsupported` | `discovered_structure_unresolved` | Request transport through the temporal entry with the treatment, outcome and horizon. |
| `discovered_transport.temporal_query_invalid` | `discovered_structure_unresolved` | State a mean response curve of one outcome under the declared treatment among the structure's variables. |
| `discovered_transport.class_not_identified` | `discovered_structure_unresolved` | Resolve the marks that separate the members or declare the graph. |
| `discovered_transport.member_not_estimable` | `discovered_structure_unresolved` | Supply evidence that covers every factor or declare the graph. |
| `discovered_transport.class_enumeration_partial` | `discovered_structure_unresolved` | Raise the member bound and the completion budget or declare the graph. |
| `discovered_transport.evidence_invalid` | `invalid_argument` | Supply complete integer-coded tables; no interval for a temporal request. |
| `selective_discovery.pooled_population_search` | `route_not_supported` | Run one population at a time; cross-population assumptions belong to the transport contract. |
| `selective_discovery.populations_missing` | `invalid_argument` | Supply at least two populations or use the single-population route. |
| `selective_discovery.population_missing` | `invalid_argument` | Name every population. |
| `selective_discovery.population_repeated` | `invalid_argument` | Use each population label once. |
| `selective_discovery.pooled_population_data` | `invalid_argument` | Supply each population's own rows. |
| `selective_discovery.nothing_retained` | `invalid_argument` | Run discovery before reporting. |
| `selective_discovery.split_rows_repeated` | `invalid_argument` | List each discovery row once. |
| `selective_discovery.split_row_out_of_range` | `invalid_argument` | Use row indices inside the table. |
| `selective_discovery.split_fraction_invalid` | `invalid_argument` | Choose a fraction in (0, 1). |
| `selective_discovery.split_part_empty` | `invalid_argument` | Leave rows in both parts. |
| `selective_discovery.split_not_disjoint` | `invalid_argument` | Re-split the table. |
| `selective_discovery.split_receipt_mismatch` | `invalid_argument` | Regenerate the receipt from the rule. |
| `selective_discovery.analysis_rows_not_the_split_part` | `invalid_argument` | Analyse the analysis part the receipt names. |
| `selective_discovery.split_unsupported_rows` | `route_not_supported` | Split a float64 table. |
| `selective_discovery.split_row_excluded` | `invalid_argument` | Name only eligible rows (inside the mask, complete in the searched variables). |
| `selective_discovery.split_rows_mismatch` | `invalid_argument` | Re-split the table. |
| `selective_discovery.split_mask_mismatch` | `invalid_argument` | Replay the split on the table it was made from. |
| `selective_discovery.split_weights_mismatch` | `invalid_argument` | Replay the split on the table it was made from. |
| `selective_discovery.split_source_mismatch` | `invalid_argument` | Replay the split on the table it was made from. |
| `selective_discovery.split_part_weight_invalid` | `invalid_argument` | Give the rule rows that carry weight. |
| `selective_discovery.weights_not_read_by_test` | `option_not_applicable` | Declare `weighted_parcorr`, or split an unweighted table. |
| `selective_discovery.weighted_test_without_weights` | `option_not_applicable` | Supply observation weights, or declare an unweighted test. |
| `selective_discovery.background_knowledge_contradictory` | `invalid_argument` | Remove the conflicting statement. |
| `selective_discovery.background_knowledge_unknown_variable` | `unknown_variable` | Name only searched variables. |
| `selective_discovery.background_knowledge_not_applicable` | `option_not_applicable` | Use an algorithm that enforces the statement, or drop it. |

### Provider-sourced transport and handoff

`provider_transport.*`, `transport_handoff.*`. See [the guide](../guides/transported-decisions-and-measurement.md).

| Detail | Code | What would support the request |
| --- | --- | --- |
| `provider_transport.blank_field` | `interface_contract_invalid` | Name every identity. |
| `provider_transport.population_missing` | `interface_contract_invalid` | Declare the population it was established in. |
| `provider_transport.regime_missing` | `interface_contract_invalid` | Declare the regime (observational is explicit). |
| `provider_transport.quantity_invalid` | `interface_contract_invalid` | Name variable, units, functional, transform and a canonical conditioning. |
| `provider_transport.object_unsupported` | `interface_contract_invalid` | Offer a supported object. |
| `provider_transport.object_invalid` | `interface_contract_invalid` | Declare a complete identity, distinct quantities and operations. |
| `provider_transport.assumptions_missing` | `interface_contract_invalid` | Name them. |
| `provider_transport.native_trust_claimed` | `interface_contract_invalid` | Declare attested or verified. |
| `provider_transport.trust_incomplete` | `interface_contract_invalid` | Name the attestor. |
| `provider_transport.quantity_not_declared` | `external_binding_mismatch` | Offer a declared quantity. |
| `provider_transport.verification_required` | `external_verification_failed` | Supply an exact-request receipt covering the object. |
| `provider_transport.receipt_does_not_cover` | `external_verification_failed` | Supply the receipt for this exact object. |
| `provider_transport.population_not_a_source` | `interface_contract_quantity_mismatch` | Use a provider from a declared source population. |
| `provider_transport.assumptions_not_carried` | `interface_contract_assumption_mismatch` | Carry it in the transport identity. |
| `provider_transport.transport_identity_incomplete` | `interface_contract_invalid` | Name them; transport only to a different population. |
| `provider_transport.evidence_missing` | `interface_contract_invalid` | Name the sources it derives from. |
| `provider_transport.claim_not_in_target` | `interface_contract_quantity_mismatch` | State it in the target population. |
| `provider_transport.native_target_required` | `interface_contract_assumption_mismatch` | Supply native data or declare the input as accepting a transported claim. |
| `provider_transport.input_not_external` | `interface_contract_assumption_mismatch` | Hand the claim to an externally supplied input. |
| `provider_transport.identity_mismatch` | `route_not_supported` | Re-seal from the declarations. |
| `provider_transport.artifact_invalid` | `invalid_argument` | Supply an artifact written by this library. |

### Transported decisions and inverse forward sources

`transported_decision.*`, `transported_decision_artifact.*`, `inverse_forward_source.*`. A consumer that does not accept a transported input refuses it as `decision_contract_unsatisfied` with `transported_not_accepted`, whichever source (cohort, composed, response, decision) carried it. See [the guide](../guides/inverse-outcome.md#inverse-queries-on-composed-cohort-and-transported-sources).

| Detail | Code | What would support the request |
| --- | --- | --- |
| `transported_decision.transport_identity_incomplete` | `transport_missing_evidence` | Name them; transport only to a different population. |
| `transported_decision.transport_identity_not_digest` | `transport_missing_evidence` | Pass the producing result's digest identity. |
| `transported_decision.not_transported` | `invalid_argument` | Decide it as a within-population claim. |
| `transported_decision.route_needs_typed_source` | `invalid_argument` | Use the typed constructor so populations and identities are taken from the producer. |
| `transported_decision.invalid_value` | `invalid_argument` | Supply finite values for distinct coordinates. |
| `transported_decision.coordinate_not_in_target` | `quantity_semantics_mismatch` | State every coordinate in the target population. |
| `transported_decision.support_label_inflated` | `transport_not_certified` | Declare extrapolative or weaker. |
| `transported_decision.calibration_label_inflated` | `transport_not_certified` | Declare point-only or unmeasured. |
| `transported_decision.native_label_on_transported_value` | `transport_not_certified` | A transported value is never native. |
| `transported_decision.mixed_transports` | `transport_not_certified` | Decide each transport's value on its own. |
| `transported_decision.law_not_supplied` | `decision_contract_unsatisfied` | Supply a transported law. |
| `transported_decision.interval_not_supported` | `uncertainty_representation_insufficient` | None: the value is point-only. |
| `transported_decision.transported_not_accepted` | `decision_contract_unsatisfied` | Accept a transported claim explicitly in the support policy. |
| `transported_decision.uncertainty_requirement` | `decision_contract_unsatisfied` | Supply a transported law or drop the requirement. |
| `transported_decision.exclusions_leave_no_comparison` | `decision_contract_unsatisfied` | Declare fewer exclusions. |
| `transported_decision.support_rule_unmet` | `transport_support_failure` | Weaken the support rule for this input. |
| `transported_decision_artifact.law_not_exportable` | `route_not_supported` | Export a point, or decide the law in process. |
| `transported_decision_artifact.bounds_exceeded` | `invalid_argument` | Export fewer coordinates or actions. |
| `transported_decision_artifact.replay_mismatch` | `invalid_argument` | Re-export from the producer. |
| `transported_decision_artifact.identity_expected` | `decision_contract_unsatisfied` | Supply the artifact whose identity was retained. |
| `transported_decision_artifact.invalid` | `invalid_argument` | Supply an artifact written by this library. |
| `inverse_forward_source.transported_not_accepted` | `decision_contract_unsatisfied` | Accept a transported input explicitly with its receipt, or supply native target data. |
| `inverse_forward_source.point_only_law_requested` | `decision_contract_unsatisfied` | Supply a source that carries a law. |
| `inverse_forward_source.interval_not_supported` | `uncertainty_representation_insufficient` | None: the source is point-only; use a mean or a law constraint. |
| `inverse_forward_source.origin_invalid` | `invalid_argument` | Re-seal the query from its source. |
| `inverse_forward_source.composed_response_invalid` | `mechanism_composition_incoherent` | Supply the composition's response with its carrying receipt. |
| `inverse_forward_source.composed_query_result_mismatch` | `invalid_argument` | Re-export the composition. |
| `inverse_forward_source.composed_family_mismatch` | `mechanism_composition_incoherent` | Query the composition in its own family. |
| `inverse_forward_source.composed_level_coding` | `mechanism_composition_incoherent` | Supply a numeric coding of every outcome level exactly once. |
| `inverse_forward_source.composed_coordinate_is_outcome_law` | `invalid_argument` | Name the mean coordinate. |
| `inverse_forward_source.composed_law_not_materialized` | `decision_contract_unsatisfied` | Draw the intervened law with the model sampler and supply aligned draws. |
| `inverse_forward_source.per_action_not_sealable` | `invalid_argument` | Seal one artifact per action. |
| `inverse_observational_scenarios.invalid_scenarios` | `invalid_argument` | Declare distinct, named scenario cells. |
| `inverse_observational_scenarios.invalid_parameter` | `invalid_argument` | Correct the declaration. |
| `inverse_observational_scenarios.condition_on_outcome` | `quantity_semantics_mismatch` | Condition on a covariate coordinate. |
| `inverse_observational_scenarios.coordinate_missing` | `quantity_semantics_mismatch` | Name a coordinate of the law. |
| `inverse_observational_scenarios.coordinate_unsupported` | `quantity_semantics_mismatch` | Use an unmasked coordinate. |
| `inverse_observational_scenarios.joint_law_required` | `joint_law_required` | Supply an aligned joint observational law. |
| `inverse_observational_scenarios.meaning_not_observational` | `distribution_meaning_mismatch` | Supply an observational law. |
| `inverse_observational_scenarios.event_has_no_mass` | `invalid_argument` | Choose an event with mass. |
| `inverse_observational_scenarios.non_finite_draw` | `invalid_argument` | Supply finite draws. |
| `inverse_observational_scenarios.result_mismatch` | `invalid_argument` | Regenerate the result from the law. |

### Assumption range and revalidation

`transport_sensitivity.*`, `transport_revalidation.*`. See [the guide](../guides/assumption-range-and-revalidation.md).

| Detail | Code | What would support the request |
| --- | --- | --- |
| `transport_sensitivity.no_scalar_family` | `route_not_supported` | State the sensitivity on the transported effect estimated with the structure and declare the shift bound there. |
| `transport_sensitivity.non_finite_point` | `invalid_argument` | Supply the finite point value the route returned. |
| `transport_sensitivity.invalid_bound` | `invalid_argument` | Declare a finite, positive bound for the shift. |
| `transport_sensitivity.invalid_grid` | `invalid_argument` | Request at least two and at most the maximum number of grid points. |
| `transport_sensitivity.proportional_at_zero` | `invalid_argument` | Declare an additive shift in the quantity's units instead. |
| `transport_sensitivity.not_transported` | `invalid_argument` | Name the target population the claim is transported to. |
| `transport_sensitivity.quantity_not_in_target` | `quantity_semantics_mismatch` | State the quantity in the target population the route transports to. |
| `transport_sensitivity.blank_identity` | `invalid_argument` | Supply the identity field. |
| `transport_sensitivity.surface_refused` | `invalid_argument` | Declare a reference quantity in the same population and units as the transported quantity. |
| `transport_sensitivity.interval_not_composed` | `route_not_supported` | None: the range stays an assumption range; sampling is withheld. |
| `transport_sensitivity.law_response_not_scalar` | `route_not_supported` | Reduce the law to the scalar functional the decision uses and transport that. |
| `transport_sensitivity.route_undeclared` | `route_not_supported` | Declare the transport route the premise belongs to. |
| `transport_sensitivity.coordinate_out_of_range` | `invalid_argument` | Name a coordinate the response has. |
| `transport_revalidation.scene_invalid` | `invalid_argument` | Declare a scene whose claims, sources and decisions form a valid graph. |
| `transport_revalidation.claim_target_mismatch` | `quantity_semantics_mismatch` | State every claim in the scene's target population. |
| `transport_revalidation.source_undeclared` | `transport_missing_evidence` | Declare the source in the scene. |
| `transport_revalidation.duplicate_claim` | `invalid_argument` | Give each claim a unique key. |
| `transport_revalidation.unknown_claim` | `invalid_argument` | Name a claim of the scene. |
| `transport_revalidation.value_digest` | `invalid_argument` | Supply a finite value. |
| `transport_revalidation.identity_mismatch` | `route_not_supported` | Re-seal from the scenes. |
| `transport_revalidation.replay_mismatch` | `invalid_argument` | Re-seal from the scenes. |

### Composed spread, whole path, evidence overlap and interface contracts

`joint_law_propagation.*`, `cross_world_uncertainty.*`, `interface_contract.*`. See [composed uncertainty](../guides/composed-uncertainty.md) and [interface-contract adapters](../guides/build/scientific-interfaces.md#adapt-declarations-you-already-have).

| Detail | Code | What would support the request |
| --- | --- | --- |
| `joint_law_propagation.spread_needs_supported_dependence` | `uncertainty_representation_insufficient` | Supply a covariance, an aligned joint law, known weights or a declared independence of distinct evidence; a correlation range or marginals alone give only the labelled envelope. |
| `joint_law_propagation.evidence_overlap_unknown` | `sampling_dependence_unknown` | Declare the observations of each source in the ledger. |
| `joint_law_propagation.mixture_weights_unknown` | `invalid_argument` | Supply the mixture weights or ask for the dependence envelope. |
| `cross_world_uncertainty.unit_sampling_undeclared` | `sampling_dependence_unknown` | Declare fixed units or an independent sample of a population. |
| `cross_world_uncertainty.sampled_path_not_supported` | `estimator_inference_mismatch` | Use exact enumeration; sampled-path coverage is not measured. |
| `cross_world_uncertainty.no_transported_path_derivation` | `transport_not_certified` | Derive in the population the mechanisms were declared for; no transported abduction derivation exists. |
| `cross_world_uncertainty.incompatible_history` | `route_not_supported` | Declare mechanisms for the history of the evidence. |
| `cross_world_uncertainty.bounds_exceeded` | `invalid_argument` | Reduce units or states below the bound. |
| `cross_world_uncertainty.result_mismatch` | `invalid_argument` | Pass the result of executing this request. |
| `cross_world_uncertainty.account_mismatch` | `invalid_argument` | Re-derive the account from the execution. |
| `interface_contract.population_mismatch` | `interface_contract_quantity_mismatch` | Produce the quantity for the consumer's population or declare a transport of evidence. |
| `interface_contract.operation_unavailable` | `interface_contract_operation_unavailable` | Supply a producer that offers the operation, or lower the consumer's requirement to what it uses. |
| `interface_contract.measurement_assumption_mismatch` | `interface_contract_assumption_mismatch` | Produce under the consumer's models or declare the consumer under the producer's. |
| `interface_contract.missing_operations` | `interface_contract_invalid` | Declare at least one operation the port offers or requires. |
| `interface_contract.shared_evidence` | `interface_contract_assumption_mismatch` | Feed disjoint evidence or declare the dependence the combination uses. |
| `interface_contract.dependence_unknown` | `interface_contract_assumption_mismatch` | Declare the evidence sources (their evidence-ledger identities). |
| `mechanism_composition.posterior_dependence_evidence_overlap_unknown` | `joint_law_required` | Declare the observations of each mechanism's evidence in the ledger. |
| `joint_law_propagation.shared_evidence_declared_independent` | `evidence_double_counted` | Treat the inputs as dependent and supply a covariance or aligned joint law. |
| `joint_law_propagation.alignment_mismatch` | `joint_law_required` | Supply columns of one joint law; rows of different joint laws are not paired. |
| `joint_law_propagation.index_pairing_refused` | `joint_law_required` | Supply the draws of one joint law with a shared alignment identity, a covariance for a linear operation, or ask for the dependence envelope. |

### Decision reliability and adaptive procedure uncertainty

`decision_reliability.*`, `adaptive_inference.*`. See [decision reliability](../decision-reliability.md#derived-intervals-and-confidence-sets) and [adaptive procedures](../guides/composed-uncertainty.md#uncertainty-of-an-adaptive-procedure).

| Detail | Code | What would support the request |
| --- | --- | --- |
| `adaptive_inference.policy_history_missing` | `adaptive_contract_incomplete` | Declare the assignment rule that governed every unit, as a function of the observed history. |
| `adaptive_inference.interim_analyses_missing` | `adaptive_contract_incomplete` | Declare the interim looks: the units enrolled between consecutive looks. |
| `adaptive_inference.outcome_availability_missing` | `adaptive_contract_incomplete` | Declare when outcomes become observable and how loss relates to the outcome value. |
| `adaptive_inference.exposure_history_missing` | `adaptive_contract_incomplete` | Declare the exposure model: the probability that an assigned arm is not delivered and what is delivered instead. |
| `adaptive_inference.stopping_rule_missing` | `adaptive_contract_incomplete` | Declare the stopping rule, its parameters and what it decides at each look. |
| `adaptive_inference.policy_selection_missing` | `adaptive_contract_incomplete` | Declare how the procedure selects an arm when no stopping rule decided, or declare no selection. |
| `adaptive_inference.arms_invalid` | `invalid_argument` | Correct the declared element. |
| `adaptive_inference.contexts_invalid` | `invalid_argument` | Declare 1 to 4 contexts with known probabilities summing to one. |
| `adaptive_inference.rule_mismatch` | `invalid_argument` | Correct the declared element. |
| `adaptive_inference.paradigm_rule_mismatch` | `adaptive_assumption_violated` | Declare the stopping rule the paradigm presumes: boundaries over the looks for group-sequential inference, a posterior expected-loss or futility rule for decision-theoretic stopping. |
| `adaptive_inference.looks_not_ordered` | `invalid_argument` | Declare a schedule whose first look enrols at least one unit. |
| `adaptive_inference.declared_units_exceeded` | `invalid_argument` | Declare at most 65,535 units; split a larger study into separately declared stages. |
| `adaptive_inference.declared_looks_exceeded` | `invalid_argument` | Declare fewer looks (24 single-outcome, 64 participant-level) or enrol larger batches between looks. |
| `adaptive_inference.exact_law_states_exceeded` | `invalid_argument` | Simulate the whole procedure instead of enumerating it, or reduce the units, arms, contexts or delay. |
| `adaptive_inference.exact_law_work_exceeded` | `invalid_argument` | Simulate the whole procedure instead of enumerating it, or reduce the units, arms, contexts or delay. |
| `adaptive_inference.simulation_replications_exceeded` | `invalid_argument` | Request between 1 and 1,000,000 replications, or run several seeded requests. |
| `adaptive_inference.simulation_work_exceeded` | `invalid_argument` | Reduce replications, units or measurements per unit so that their product is within the work budget. |
| `adaptive_inference.simulation_memory_exceeded` | `invalid_argument` | Reduce the replications or the units, or raise the memory budget of the execution context. |
| `adaptive_inference.retained_record_exceeded` | `invalid_argument` | Retain fewer looks or participants in the exported record. |
| `adaptive_inference.artifact_size_exceeded` | `route_not_supported` | Retain fewer looks or participants, or raise the export budget; the run is complete and stays in memory. |
| `adaptive_inference.historical_borrowing_missing` | `adaptive_contract_incomplete` | Declare the prior of every arm, with its historical sources and discounts, before recruitment. |
| `adaptive_inference.anytime_settings_missing` | `adaptive_contract_incomplete` | Declare the miscoverage level and tuning sample size of the confidence sequences. |
| `adaptive_inference.historical_prior_invalid` | `invalid_argument` | Declare one prior per arm with valid sources: successes within trials, a discount in [0, 1], a label. |
| `adaptive_inference.anytime_settings_invalid` | `invalid_argument` | Declare a miscoverage level in (0, 1) and a positive tuning sample size. |
| `adaptive_inference.borrowing_fixed_after_first_look` | `adaptive_assumption_violated` | Declare the borrowing before recruitment and keep it; a different plan is a new declaration. |
| `adaptive_inference.assignment_not_replayed` | `adaptive_assumption_violated` | Retain the allocation probabilities the declared procedure computed, or declare the procedure that ran. |
| `adaptive_inference.stopping_not_replayed` | `adaptive_assumption_violated` | Retain the looks the declared decisions performed, or declare the decisions that ran. |
| `adaptive_inference.batch_incomplete` | `adaptive_assumption_violated` | Retain the responses of every participant of a batch whose follow-up has elapsed, or end the history at the last batch whose participants all completed. |
| `adaptive_inference.outcome_availability_invalid` | `invalid_argument` | Correct the declared element. |
| `adaptive_inference.exposure_invalid` | `invalid_argument` | Correct the declared element. |
| `adaptive_inference.posterior_premise_outcome_dependent_delay` | `adaptive_assumption_violated` | Declare ignorable loss and delay, or analyse a paradigm that models the loss. |
| `adaptive_inference.anytime_premise_contexts` | `adaptive_assumption_violated` | Analyse one context at a time or declare a single context. |
| `adaptive_inference.outcome_dependent_delay` | `adaptive_assumption_violated` | Declare ignorable loss or use a paradigm that models the loss. |
| `adaptive_inference.fixed_sample_design_adaptive` | `adaptive_assumption_violated` | Use the paradigm supported for the adaptive cell. |
| `adaptive_inference.fixed_sample_data_dependent_stopping` | `adaptive_assumption_violated` | Use the group-sequential or anytime-valid paradigm. |
| `adaptive_inference.group_sequential_assignment_adaptive` | `adaptive_assumption_violated` | Use fixed randomization or an anytime-valid paradigm. |
| `adaptive_inference.inversion_premise_contexts` | `adaptive_assumption_violated` | Invert one context at a time. |
| `decision_reliability.effective_sample_size_too_small` | `route_not_supported` | Draw more or reduce serial dependence. |
| `decision_reliability.dependence_unknown` | `sampling_dependence_unknown` | Declare shared simulation, independent streams or an aligned joint law. |
| `decision_reliability.search_record_required` | `candidate_search_record_required` | Supply the candidate search record (considered and retained counts). |
| `decision_reliability.validation_not_independent` | `validation_stream_not_independent` | Validate on an independent stream or a disjoint split. |
| `decision_reliability.producer_point_only` | `joint_law_required` | Supply aligned per-draw utilities from one joint law, or retain per-rollout utilities. |
| `decision_reliability.producer_mean_disagrees` | `distribution_meaning_mismatch` | Supply draws of the same quantity or drop the agreement check. |

### Finite-SCM discretization

`discretized_scm.*`, `discretized_scm_artifact.*`, `paradigm_parity.*`. See [the guide](../cross-world-execution.md#discretize-a-gaussian-composition-into-a-finite-model).

| Detail | Code | What would support the request |
| --- | --- | --- |
| `discretized_scm.levels_undeclared` | `invalid_argument` | Declare a level count for every variable. |
| `discretized_scm.levels_out_of_range` | `invalid_argument` | Declare between 2 and 64 levels. |
| `discretized_scm.unknown_variable` | `invalid_argument` | Declare levels for the composition's own variables. |
| `discretized_scm.state_space_too_large` | `invalid_argument` | Declare fewer levels or fewer variables. |
| `discretized_scm.non_gaussian_mechanism` | `route_not_supported` | Discretize a linear-Gaussian composition. |
| `discretized_scm.support_not_real_line` | `route_not_supported` | Declare an unbounded real support. |
| `discretized_scm.latent_confounding` | `route_not_supported` | Use a Markovian composition; a latent-confounded finite SCM needs the exogenous-term construction. |
| `discretized_scm.degenerate_marginal` | `route_not_supported` | Declare a positive finite noise scale. |
| `discretized_scm.composition_incoherent` | `mechanism_composition_incoherent` | Repair the composition findings. |
| `discretized_scm.intervention_invalid` | `invalid_argument` | Name declared variables and levels. |
| `discretized_scm.point_only` | `uncertainty_representation_insufficient` | None: the value is point-only with a computed mean-error bound. |
| `discretized_scm.cancelled` | `cancelled_no_claim` | Re-run without cancellation. |
| `discretized_scm.numerical` | `invalid_argument` | Check the composition parameters. |
| `discretized_scm_artifact.malformed` | `invalid_argument` | Supply bytes written by this library. |
| `discretized_scm_artifact.version_unsupported` | `schema_mismatch` | Re-export with this library version. |
| `discretized_scm_artifact.not_exportable` | `route_not_supported` | Re-export from a linear-Gaussian composition. |
| `discretized_scm_artifact.rederivation_refused` | `invalid_argument` | Re-export from the producing composition. |
| `discretized_scm_artifact.identity_mismatch` | `invalid_argument` | Supply the artifact whose identity was retained. |
| `discretized_scm_artifact.claimed_numbers_differ` | `invalid_argument` | Re-export from the producing composition. |
| `paradigm_parity.paradigm_not_supported` | `route_not_supported` | Use the paradigm the route supports; the closed reason is returned as the remedy. |

### Evidence overlap of design studies

`evsi.*`, `design_ranking.*`, `prior_signal.*`. Whether a study's data overlap the prior's is read from the evidence ledger; a lineage the ledger cannot decide is unknown, and unknown is never disjoint.

| Detail | Code | What would support the request |
| --- | --- | --- |
| `evsi.source_overlap_unknown` | `evidence_overlap_unknown` | Declare the study's observations (or its lineage) in the ledger. |
| `design_ranking.source_overlap_unknown` | `evidence_overlap_unknown` | Declare the observations of the study and prior sources in the ledger. |
| `prior_signal.observation_overlap_unknown` | `evidence_overlap_unknown` | Declare the observation ids of every source. |

### Evidence that would resolve a not-identified result

A not-identified refusal says what cannot be claimed. When the contract holds
the not-identified product (after `prepare`; cheap `inspect` identifies nothing and
so derives nothing), it also says what evidence would change that, derived from
the graph and never executed:

| Next action id | Derived from | Meaning |
| --- | --- | --- |
| `measure_variable:<name>` | A joint-law obligation of the back-door, front-door or sensitivity route | Measure that variable (an unmeasured common cause of a bidirected edge is named `common_cause_of_<a>_<b>`) jointly with the treatment and outcome. |
| `run_experiment:<target>` | An `intervene` obligation of a hedge | Run an experiment that sets the target: the treatment, or a hedge node that intercepts every directed path while its response to the treatment is identified. |
| `find_instrument` | An instrument obligation | Find a variable that moves the treatment without reaching the outcome except through it. |
| `collect_panel` | The panel alternative, when back-door adjustment needs unmeasured variables | Collect repeated measurements of the same units. |

Each action traces to an `EvidenceObligation` that keeps its proof step
(`backdoor.adjustment_set:<ids>`, `frontdoor.mediator_set:<ids>`,
`iv.instrument:<id>`, `hedge:f=[..];f_prime=[..]`, `hedge.surrogate:<id>`,
`sensitivity.<method>:strength=<s>`). An instrument's relevance and exclusion,
the effect restriction of the Wald ratio and the time-invariance a panel relies on
are `establish_assumption` obligations: no study satisfies them, so a repair report
over an instrument contract never certifies on data alone and an assumption yields no
action of its own. A family whose observed law already identifies the effect refuses
with `identification_repair.not_a_failure`; a search stopped by its budget keeps its
unevaluated subsets and is never reported as impossibility.

The capability report also carries the supported neighbors and the unmet obligations:
the host record (`operation_record`) emits `neighbors` and `unmet_obligations` beside
`next_actions`, and Python exposes `next_actions`, `neighbors`, `unmet_obligations` and
`evidence_obligations` on `inspect()` reports. A neighbor is an explicit alternative
that changes named premises, never an automatic fallback; the descriptive fallback
below is likewise opt-in and returned beside the refusal.

### Ranked evidence plans

`design.evidence_plan` turns the evidence a failed identification owes into the options
that would close it: each declared study is checked by the theorem of every route, and the
plan is ordered by the identification gate and then by value net of a declared cost map.
It never converts a refusal into a verdict about the world:

| Detail | Code | What it says |
| --- | --- | --- |
| `evidence_plan.budget` | `transport_budget_cancel` | A declared search budget stopped with nothing certified; the unexamined subsets are unknown, not impossible. |
| `evidence_plan.none_certified` | `transport_not_certified` | Nothing declared was certified; that does not show that no study can close the gap. |
| `evidence_obligations.wrong_contract` | `transport_missing_evidence` | The only verified-but-uncertified options rest on assumptions (an instrument's validity) that no study establishes. |
| `evidence_plan.cost_units_mismatch` | `design_cost_units_mismatch` | The declared studies, or the cost map, do not share one cost unit. |
| `evidence_plan.invalid_request` | `invalid_argument` | No route report, no declared study, or a report naming an undeclared study. |
| `study_candidate.declarations_required` | `design_signal_invalid` | A template was used as a study: cost, timing, recruitment, unit rules, feasibility and sample size are never defaulted. |
| `study_candidate.sample_below_obligation` | `design_signal_invalid` | The declared sample size is below what an addressed obligation needs. |
| `confounder_value.range_surface` | `decision_contract_unsatisfied` | An assumption range is not a probability law; only a point surface can be valued under a declared prior. |
| `evidence_plan_artifact.replay_mismatch` | `transport_not_certified` | A stored option, outcome or identity differs from the replay of its routes and value ranking. |

A plan is hypothetical identification, not an estimate: its inference claim is `none`, the
ordering is `point_only` and its calibration `unmeasured`.

### An opt-in description beside the refusal

A not-identified refusal is the answer to the causal question and is never replaced. A
caller who also wants to know what the data do say, and what evidence would change the
answer, opts in: Python `analyze(..., on_refusal="describe")`, Rust
`Study::analyze_with_fallback`. Without the option nothing changes and the refusal is raised.
With it, an identification-class refusal (`effect_not_identified`, `cross_world_not_identified`;
never a search a budget stopped, a cancellation, a resource limit, an invalid argument or a
data or estimation error) returns an `AnalysisWithFallback`:

| Part | What it is |
| --- | --- |
| `refusal` | The original refusal object, unchanged: same class, reason code, remedy and report. |
| `associational` | The compiled associational route over the same outcome, treatment and adjustment variables (the measured parents of the treatment and the declared modifiers), labelled noncausal, with the refusal's reason code retained in its identification layer. Identification is not claimed; it is not the effect of an intervention and it ranks no action. |
| `evidence_plan` | The verified evidence obligations, the evidence-gathering next actions and the candidate templates; with declared candidate studies, the ranked plan of the options that would close the gap. |

Each of `associational` and `evidence_plan` is either present or a typed
`FallbackUnavailable` that says why not:

| Detail | Code | What it says |
| --- | --- | --- |
| `fallback.no_association_form` | `route_not_supported` | The query kind relates no single outcome to a single treatment (a counterfactual, a distribution, a mediation), so there is nothing to describe; state the associational question directly. |
| `fallback.variable_not_measured` | `invalid_argument` | A variable the query names is not a measured column. |
| `fallback.modality_not_described` | `route_not_supported` | The data are a panel, events or several environments; describe one table or one series. |
| `fallback.graph_class_not_read` | `route_not_supported` | The evidence routes are read from a declared directed or bidirected graph. |
| `identification_repair.not_a_failure` | `invalid_argument` | The declared graph identifies the effect, so the refusal came from the estimator or the data and no structural evidence gap is owed. |
| `fallback.association_refused` | the refusing route's code | The compiled associational route itself refused (for example an option it does not take); the message carries the next attempt. |

The returned object states plainly that the associational result is not an answer to the
causal question, and the host record (`antecedent.host.fallback_response`,
`antecedent::host::fallback_response`) shows `causal_status: unsupported` beside the status of
the noncausal claim. See the [fallback guide](../guides/fallback.md).

## Other registered refusal codes

Every code below is registered in `parity/reason_codes.toml`. Each is a scientific, input or resource refusal with its own code; none is `not_supported`, which is reserved for a unit that lacks provenance or an independent scientific check.

| Code | Meaning |
| --- | --- |
| `data_modality_not_supported` | The query is not supported on this data modality (tabular, series, event, panel, multi-environment). |
| `construction_not_supported` | The coordinate is supported for one named construction (TransportQuery: a mean ResponseCurve transported by Direct or S-admissible trial-to-target IPW; InterferenceQuery: NeighborCount exposure under Bernoulli assignment). Other constructions refuse on the supported estimator; a separately identified off-axis estimator may execute without inheriting its support or calibration. |
| `validators_not_applicable` | Custom validators refute one scalar average effect; this estimand has no such refutation problem. |
| `population_not_estimable` | The query has no target population to redeclare, or this route estimates the all-observed population only. |
| `prior_transfer_not_hydrated` | This route fits under the isotropic prior scale and does not hydrate a transferred prior. |
| `prior_dimension_mismatch` | A banked posterior's coefficient count does not match the target design's, so its coefficients cannot be mapped one-to-one onto a prior. |
| `likelihood_not_supported` | The Bayesian route fits a Gaussian identity-link model only, or cannot carry a transferred prior onto a non-Gaussian link; a Bernoulli or Poisson likelihood is refused rather than fitted as Gaussian. |
| `posterior_draws_below_supported` | A posterior draw count below the supported minimum of the backend was requested under draw controls. The refusal names the minimum; the count is never rewritten. |
| `posterior_draws_not_converged` | Under draw controls the Monte Carlo standard error of an interval endpoint exceeds the accepted fraction of the interval width, because the draws (too few, or autocorrelated by a sampler that mixes poorly) rather than the posterior set the endpoints. The refusal names the endpoint errors, the effective draws and the tolerance. |
| `stage_stream_unavailable` | The route fits its point and uncertainty together, or mixes identifications, so it has no progressive stages. |
| `not_executed` | The study has not been executed; there is no claim or calibration slot. |
| `row_weights_bound_to_snapshot` | A RowWeights population may re-execute only on its own data snapshot. |
| `mechanism_declaration_invalid` | A registered external mechanism's declaration is malformed or incomplete: a blank implementation id or version, a content digest that is not 64 lowercase hex characters, a duplicate dependency, input or parameter, a finite-difference step outside (0, 1], Lipschitz bounds that do not match the inputs, parameter values that do not match the declared schema, an analytic-derivative declaration whose implementation supplies no gradient, or an implementation whose content does not hash to its declared digest. The refusal names the declaration that would make the mechanism well formed. |
| `mechanism_resolver_missing` | A portable record names an external mechanism but the caller supplied no trusted resolver, so independent replay is refused and nothing is run. The record carries identities and digests only, never code; the remedy is to supply a resolver that maps the recorded implementation id, version and content digest to the implementation. |
| `mechanism_resolver_failed` | The caller's trusted resolver itself failed while resolving a recorded external mechanism: it raised, or returned an object that is not an implementation. This is a resolver fault, distinct from an absent implementation (mechanism_implementation_unavailable); replay is refused and nothing is run. The remedy is to repair the resolver; the refusal carries its failure text. |
| `mechanism_implementation_unavailable` | The caller's trusted resolver has no implementation for a recorded external mechanism (its implementation id, version and content digest), so independent replay is refused. The remedy is to register exactly the recorded implementation; a nearby version is a different mechanism. |
| `mechanism_implementation_changed` | The implementation a resolver returned differs from the recorded identity in its implementation id, version, content digest, declared dependencies, or declared signature, parameter schema, derivative declaration or Lipschitz bounds. Independent replay is refused and the implementation is not run; a changed implementation is a different mechanism and its results are not the recorded ones. |
| `attested_not_reverifiable` | Caller-attested evidence (custom validators or an external estimate) is not re-verified on load. |
| `treatment_support_too_discrete` | A local-polynomial response in the treatment needs more distinct treatment values carrying kernel weight at an evaluation point than the treatment has there; a binary or few-level treatment is answered by AverageEffect or InterventionResponse instead. |
| `rd_assignment_not_sharp` | A sharp regression-discontinuity design declares treatment = 1{running variable >= cutoff}; the treatment column breaks that rule (or is not 0/1), so the outcome jump is not the effect of the treatment and is not reported as one. |
| `estimator_inference_mismatch` | The selected estimator does not implement the requested inference mode on this query, so it is refused rather than run under the other mode. |
| `refutation_not_applicable` | The requested refutation suite has no state to refute for this estimand. |
| `development_artifact` | The artifact was sealed by a development build (`build: development`), which runs routes a release build refuses; a release build does not consume it. |
| `cell_not_applicable` | The support-matrix cell does not denote for this query, structure, inference and validation. |
| `dynamics_not_identified` | The trajectory design does not identify the requested continuous-time dynamics: no unit is observed at two or more times, or the observed information of the requested coordinates is singular at the converged optimum (a coordinate the observations cannot determine, such as the dynamics of an unobserved state, or an unidentified diffusion factor). The refusal names the evidence and the observation or coordinate change that would identify the dynamics. |
| `external_capability_missing` | An external scientific object lacks the exact operation required by the request; approximation requires a separately supported error and replicate receipt. |
| `inverse_functional_unsupported` | An inverse decision request needs a forward law, functional, finite action support or joint alignment that the supplied claim cannot provide. |
| `unknown_strategy` | The estimator or identifier id is not in the closed strategy table. |
| `strategy_incompatible` | The identifier and estimator (or query and estimator) do not form a compatible pair. |
| `required_option_missing` | The selected route needs a companion option the call did not supply. |
| `transport_unsupported_evaluator` | The transport formula is identified but this evaluator does not run it. |
| `transport_policy_required` | A prior transport was requested for differing source and target populations without a TransportPolicy; the library will not assume the populations are exchangeable. |
| `transport_source_count_mismatch` | The number of transported source priors does not match the number of source populations supplied. |
| `transport_invalid_weights` | Transport weights are not finite, non-negative and normalisable. |
| `transport_unknown_policy` | A TransportPolicy wire name is not one the library defines. |
| `transport_coef_index_out_of_range` | A coefficient index passed to a transport reweighting is outside the prior's dimension. |
| `sampling_weights_invalid` | Declared sampling weights are not a valid weighted population: a weight is negative, NaN or infinite, every weight is zero, no analysed row has a positive weight, or the declaration does not cover the table's rows. The refusal names the first offending weight. |
| `scenario_aggregate_not_supported` | An inferential statement across structural scenarios that share data was requested; only per-scenario results and structural envelopes are supported. |
| `glm_fit_refused` | A generalized linear fit (a propensity or an outcome model) did not return a usable fit: it separated the arms or outcome classes, did not converge, or, for a model other than a propensity, saturated at the probability boundary. A propensity that saturates refuses as positivity_violated instead. The refusal names which defect occurred and the iteration count; drop or coarsen the separating or near-deterministic columns, standardize near-collinear columns, or (AIPW only) declare a penalized propensity with propensity_penalty. |
| `target_weights_not_supported` | Custom target weights are not supported: their declared dependence includes the treatment, an intervened coordinate or a descendant of either, is not a function of the certified adjustment set, is empty while the weights vary, or cannot be checked without a directed graph. The refusal names the failing condition; declare depends_on over adjustment-set columns only, or pass constant weights. |
| `rank_drop_not_supported` | An opt-in rank-deficiency drop would remove a column the query needs (a treatment, outcome or effect modifier) or the declared column priority does not cover the design; nothing is dropped. |
| `stratum_too_sparse` | A retained propensity stratum has fewer than two treated or two control rows, so its within-stratum variance and the analytic interval are undefined. The refusal names the stratum and its counts; use fewer strata, declare an overlap trim that removes the sparse region, or request a bootstrap interval. |
| `arm_not_populated` | A treatment arm or joint cell has no complete-case rows, so no contrast against it is estimable; the refusal names the arm and its counts. |
| `penalized_interval_not_supported` | A factorized joint-cell propensity with penalized conditional models publishes cell points and a score table only; its nuisance remainder and interval coverage have not been supported, so a requested interval is refused. The separately calibrated ridge/lasso AIPW route has its own interval contract. |
| `selection_inference_not_supported` | The nuisance learner selects variables from the data (lasso), which changes the inference contract; an ordinary influence-function interval is not valid after selection, so the option is closed. |
| `nuisance_fallback_not_supported` | A declared fallback from a failed GLM nuisance fit to a flexible learner has no support or uncertainty support on this route; the failed fit is reported and no fallback result is produced. |
| `joint_cell_unsupported` | One discrete joint treatment cell cannot be evaluated (a conditional factor or outcome model is unsupported, or the cell's weights have too small an effective sample); the cell is refused individually with its reason and the rest of the family is kept. |
| `derived_treatment_invalid` | A derived-treatment declaration is inconsistent, a constituent column or forbidden descendant sits in the adjustment set without a declared causal exclusion rule, or an observed value is outside the declared legal values; nothing is dropped to repair it. |
| `ml_nuisance_not_supported` | A machine-learning nuisance was declared for joint cells in a way no record supports: an interval or joint covariance over learner-supplied nuisances (which need their own remainder and coverage support), or a machine-learning name that is not a declared joint-cell learner; refused rather than silently replaced by the ridge-logistic factors. |
| `cluster_interval_not_supported` | Cluster-DML AIPW refuses a requested row bootstrap or alternative analytic SE kind: the row bootstrap would split clusters, and neither requested construction has coverage support. The default one-way cluster t interval has its own measured coordinate. |
| `too_few_clusters` | The number of independent clusters is below the declared minimum, so a sandwich over cluster sums and a whole-cluster cross-fit are not formed; the refusal states the count found and the minimum declared. |
| `dyadic_dependence_not_supported` | Rows share endpoints (dyads) in a structure the two-way cluster-DML construction does not cover: an entity used as both a first and a second endpoint, or one connected component of the endpoint graph holding more than a fold's share of the rows, so whole-component folds and the two-way variance are not supported. |
| `diagnostic_not_available` | A diagnostic is requested that the executed route's own design does not carry (no propensity score, an E-value that embeds an interval endpoint, a result with no single effect); it is reported absent with this code and a detail, never fabricated and never refit from a different design. |
| `mechanism_candidates_invalid` | A mechanism candidate list, per-node family constraint or validation-split declaration cannot be honoured: an unknown family or alias, an empty list, a variable outside the model, two constraints on one variable, a family both required and excluded or outside the allowed list, a required family of the wrong outcome type, or time-ordered or clustered splits the rows cannot support. Raised before any family is fit; the message names the node and the defect. |
| `mechanism_family_unavailable` | A family a node requires could not be scored or fit on that node (a tree ensemble on a root or on fewer rows than it needs, a provider that is not compiled in, a fit that failed on every fold). The message names the node, the family and its failure; the other candidates' scores are not used to substitute another family. |
| `decision_evidence_scope_mismatch` | A request reaches past what the experiment established: a population, randomized contrast or horizon the evidence does not cover, or a deployment-policy quantity with no declared structural bridge from the experimental contrast. Nothing is carried across; the refusal names the evidence's scope and the bridge or evidence that would support the request. |
| `decision_evidence_design_unsupported` | The assignment design, the pretreatment adjustment or the exposure assumption has no joint route: an assignment design other than complete randomization of units or clusters, a learned coefficient that would change the unit of analysis, between-unit exposure left unspecified, or more unknown dimensions than one evaluation enumerates. The refusal names the admissible alternatives. |
| `decision_evidence_invalid` | A joint evidence, derived quantity or decision declaration is not well formed: misaligned or non-finite arrays, a covariance that is not symmetric with a nonnegative diagonal, an arm with fewer than two assigned units, an unknown component, or a frame that does not name exactly two actions. The refusal names the field. |
| `candidate_search_record_required` | A ranking, optimality or selection-aware statement was requested over candidates that an adaptive search produced, and no search record names the generator, its budget, the number of candidates evaluated and the selection rule. Retained candidates alone would silently condition the claim on a search it does not describe; the refusal states the record that would support the request. |
| `validation_stream_not_independent` | A held-out or split re-evaluation of a selected decision uses draws that overlap, or are not separated from, the draws that selected it (the same stream coordinates, or an overlapping or too-close range of one stream), so it cannot correct selection bias. The refusal names both segments and the independent stream or gap that would support the request. |
| `uncertainty_components_not_combined` | One combined uncertainty number was requested for a sampled or selected decision. Source-distribution spread, Monte Carlo error and selection-induced uncertainty are reported separately and never summed; a combined statement is a whole-procedure claim that needs a calibration at the coordinate (candidate-set size, Monte Carlo budget, selection rule). |
| `history_support_violated` | A supplied policy takes an action, with positive probability, in a reachable history where the action has no support (a positivity violation): the model declares no support for it there, or the logging policy that generated the logged data has probability zero or no declared decision there, or no logged trajectory is consistent with the target policy so every importance weight is zero. The value of the policy is not identified by extrapolating across the gap; the refusal names the history, the action and the support declaration or logged coverage that would support the evaluation. |
| `adaptive_design_ignored` | An analysis that presumes a design fixed in advance (fixed sample size, fixed assignment, no stopping rule) is requested for data collected under an adaptive assignment policy, interim monitoring or a stopping rule. The refusal names the adaptive-design routes that account for the design. |
| `adaptive_policy_unobserved_feature` | A history-dependent intervention policy reads a variable that is not observed when the policy acts (for example a quantity realised after the action). The refusal names the variable; a rule may read only information available at decision time. |
| `adaptive_policy_identification_unsupported` | An identified policy effect was requested for an optimised history-dependent policy but the response model's identification does not support a causal policy-effect reading. The result can be requested as a fitted-model optimisation, which is conditional on the fitted model and its assumptions. |
| `adaptive_policy_heldout_unsupported` | A held-out evaluation of the selected policy was required but the study design supplies neither independent simulated draws nor held-out units, so the selected policy's in-sample value would be optimistic and unevaluated. |
| `interface_contract_uncertainty_mismatch` | The producer's interval, posterior, law or draws does not mean what the consumer requires: no uncertainty meaning is declared, the paradigm differs (a posterior is not a sampling distribution, and neither is a bootstrap or an empirical distribution), or the object differs within a paradigm (a posterior over parameters is not a posterior over a causal functional or an outcome). The refusal names the required and declared meanings. |
| `interface_contract_cyclic_composition` | The connections of a composition form a cycle: feedback between one component's output and another's input. Cyclic scientific composition is not supported; the loop is refused before any component runs and the refusal names the cycle path. Compose the components feed-forward, or model the loop inside one component's own declared contract. |
| `claim_revalidation_required` | A claim rests on an upstream object (data, evidence, a model or representation version, or a structural derivation) that changed after the claim was produced, so its numbers, fit and uncertainty carry no authority until it is recomputed. The refusal names the changed origin; the support for the claim is not withdrawn. |
| `claim_support_withdrawn` | A claim rests on an upstream object whose support no longer holds: an input that was retired, replaced or removed, coordinates outside the support, a provider whose trust was withdrawn, or a stage that cannot be executed. This is distinct from stale numbers: recomputation alone does not restore the support. The refusal names the origin and the declaration that would restore it. |
| `callback_protocol_invalid` | A callback declaration is not well formed: a blank provider, object, version, snapshot or request identity, an operation the callback kind does not answer, a meaning that does not belong to the kind, a posterior update without its candidate, prior and observation or without the observation law, a seeded callback without a semantic stream identity, an idempotent retry without an attempt bound, a nested callback with no callback input, or a trust level metadata cannot confer (native support) or identification evidence a declaration cannot mint. The refusal names the declaration that would make the callback well formed. |
| `callback_reuse_ineligible` | A stored callback result is not eligible for reuse, or a request cannot be answered without running the callback: a dependency (data, model, file, remote service, environment, closure state) is opaque or undeclared, a stateful callback exposes no state identity through the protocol, or a side-effecting callback has no idempotent retry contract. Without a stable identity the callback is recomputed or the request is refused, never silently reused. The refusal names the declaration that would make reuse sound. |
| `callback_side_effect_forbidden` | A callback with effects outside the process would be invoked where the protocol forbids it: during planning or verification, without an explicit opt-in in the execution policy, or as a retry outside its declared idempotent contract. The callback is not run. The refusal names the policy or contract that would allow the execution. |
| `callback_output_invalid` | A callback returned a value the protocol does not accept as an answer of the declared quantity: the wrong shape for the query, a non-finite value, a value outside the declared support, a distribution function outside [0, 1] or decreasing in the threshold, a quantile function decreasing in the level, or a query that is itself malformed. Nothing is repaired, clipped or sorted; the whole answer is refused. |
| `callback_provider_unavailable` | No compatible provider can answer a callback request in this process: none was supplied for a fresh-process resume, or the provider failed. The historical result stays inspectable and the stage is recorded as unavailable; no claim is issued. The refusal names the provider object, version, snapshot and request, or the declared portable runtime with resolved dependencies, that would support execution. |
| `callback_limit_exceeded` | A callback request exceeds a declared resource bound: the invocation count, the output size, or the wall-clock bound. The in-process executor enforces the wall-clock bound only after the call returns and discards the late result; a hard preemptible limit needs a declared isolated runtime. The refusal names the bound and the policy field to change. |
| `discovery_statistics_invalid` | Retained discovery statistics cannot be used: the raw data they were computed from are absent or differ, the contract differs, a receipt is not a usable statistic or does not cover a requested test, or replaying them does not reproduce the recorded result. Statistics are bound to the content they were computed from; the remedy is to rerun discovery on the raw data. |
| `near_collinear_column` | An adjustment column is not an exact linear function of higher-priority columns but leaves almost no residual after projecting them out (uncentered); fits are ill conditioned and nothing is dropped. Review finding from design preflight. |
| `column_separates_arms` | A single adjustment column separates the treatment arms completely or quasi-completely, so a propensity model saturates; a positivity review finding for human review, not an identification verdict. |
| `adjustment_duplicates_query_variable` | An adjustment column is an exact copy of a treatment or the outcome; review for a post-treatment or label-leaking column. |
| `duplicate_adjustment_columns` | Adjustment columns are exact copies of one another, so the design is rank deficient by exactly this duplication. |
| `duplicate_query_columns` | The treatment and outcome columns are exact copies. |
| `treatment_near_determined_by_adjustment` | A treatment column is near-deterministically predicted by the adjustment columns (linear R^2 above threshold); review finding. |
| `outcome_near_determined_by_adjustment` | The outcome column is near-deterministically predicted by the adjustment columns; review finding. |
| `adjustment_column_tracks_treatment` | A single adjustment column near-deterministically predicts a treatment column; review finding. |
| `adjustment_column_tracks_outcome` | A single adjustment column near-deterministically predicts the outcome; review finding. |
| `treatment_level_undeclared` | The treatment column has rows at a level that is none of the levels the estimate declares (the active and control levels of a pairwise contrast, or the declared level set of a full-population generalized-propensity estimate). Dropping those rows would restrict the target population, so the estimate refuses unless the restriction to the rows at the declared levels is declared. |
| `policy_learned_on_evaluation_rows` | A treatment policy declared as learned on the rows that would score it. Its value on those rows is estimated too high because the policy was chosen to fit them; the refusal names the two honest evaluations, independent held-out rows and cross-fitted learning. |
| `missing_policy_required` | A route that has no default for missing values found one and no missing-data policy was declared. Randomized, panel, temporal and NOTEARS routes never delete, weight or restrict rows on their own. The refusal names the first missing cell (column and row) and the policies the route can execute. |
| `missing_policy_not_applicable` | A declared missing-data policy cannot be executed validly on this route or design: deleting rows would unbalance a panel or compress a time axis, the design (cluster, stratified, factorial, multi-arm, switchback, receipt-adjusted) has no responder-weighted estimator, or the covariates of the observation model are themselves incomplete. The refusal names the policies the route supports. |
| `policy_dependency_circular` | A value program reads a variable that depends on the node it assigns, directly, through the graph's edges or through other programs' reads, so no evaluation order computes the inputs before the node. The refusal names the cycle and the policies in it. |
| `policy_dependency_unavailable` | A value program reads a variable it cannot see: a variable absent from the compiled model, a read it did not declare exactly, or the generated value of a node whose mechanism the program replaces (an assignment). The refusal names the policy and the missing or undeclared read. |
| `policy_directive_conflict` | Two directives on one node have no well-defined composition: two different value programs, a program with a hard set or an additive shift, or an assignment with a stochastic policy or soft mechanism. The refusal names the node and both directives. |
| `policy_output_type_invalid` | A value program produces a type its node cannot hold: a boolean written to a numeric node without an explicit indicator, or a number that is not a level of a categorical node's support. The refusal names the node and the expression's type. |
| `policy_history_unavailable` | A value program reads temporal history that the route cannot supply: a lagged read on a static model with no declared history window, a look-back deeper than the route's declared window, a look-back that reaches before the first available step of a temporal rollout, or a look-back of zero steps (a future value does not exist when the policy applies). The refusal names the offending reference and, for a rollout, the step. |
| `policy_route_not_supported` | A policy is supported for some execution routes but not the one requested. Simulation and counterfactual execution accept any valid program; observational identification accepts only programs that reduce to a hard level (a set, or an assignment that folds to a constant). The refusal names the route and the routes the policy is supported for. |
| `policy_callback_failed` | A caller-supplied batch callback in a value program raised while it ran (a Python exception, for instance). The refusal carries the callback's own error message and the callback's name; a failure is never turned into a missing value or a default. |
| `policy_callback_output_invalid` | A batch callback returned an output that violates its declaration: the wrong number of values, a dtype that is not real-valued, a non-finite value, or a value outside the declared support or levels. The refusal names the callback, the first offending row and the declaration. |
| `policy_callback_not_portable` | A value program contains a callback that is a runtime binding (an in-process function with no verifiable recoverable implementation), so no artifact, identity or wire encoding can carry it. A callback is exportable only when declared with a content digest of its self-contained source and pinned dependency versions that the consumer re-verifies; a dependency declared without a pinned version leaves nothing to verify and is refused the same way. The refusal names the callback. |
| `policy_callback_unrecoverable` | A recovery record for a callback does not verify against the implementation supplied for replay: the source digest, entry point, input labels, output declaration or a pinned dependency version differs, a pinned dependency is not installed, or no implementation was supplied for a record the artifact carries. The refusal names the callback and the first mismatching field. |
| `route_not_lowered` | Execution was handed a physical plan that carries no lowered program for its route, or whose lowered program was compiled for a different route. Only `Study::compile` lowers a route; a plan built from a logical record alone cannot be dispatched. |
| `plan_view_unavailable` | The handle executes no lowered study program (a transport stage or a policy-program handle), so it has no plan view of compiler layers to read. |
| `public_builder_reentry` | Compile, prepare or one-shot run was called with a context that is already inside the dispatch of a lowered program. A dispatched route executes its retained program and does not build and run another public study. |
| `solver_model_order_budget` | The model order (variables) exceeds the admitted bound of the dense disturbance covariance (4096), of an exact affine-moment score (1024), or of a compositional expression's gradient storage; nothing is truncated. Not an impossibility claim. |
| `solver_jacobian_nnz_budget` | The free Jacobian of an equilibrium solve has more structural nonzeros than the policy admits (max_nnz); refused before any iteration. |
| `solver_block_size_budget` | A strongly connected feedback block is larger than the policy admits (max_block_size); refused before any iteration. |
| `solver_memory_budget` | The estimated solver memory (covariance, sparse Jacobian copies, factor fill, expression tapes) exceeds the policy admission (max_bytes). |
| `solver_factor_fill_budget` | Sparse LU fill-in of a Newton step exceeded the policy bound (max_factor_fill); the block is neither solved nor declared singular. |
| `solver_iteration_budget` | The iterations summed over all blocks exceeded the policy bound (max_total_iterations). Per-block non-convergence is a retained outcome, not this refusal. |
| `solver_wall_clock_budget` | The wall-clock deadline of the solve policy elapsed (max_wall_clock_ms); the partial solve is discarded, not reported. |
| `hmc_work_budget` | The total requested HMC sampling and diagnostic work exceeds the caller's admitted work limit; the sampling schedule is not shortened. |
| `execution_cancelled` | The caller cancelled execution before a complete result was available; this is an operational outcome, not a scientific refusal. |
| `moment_matched_table_work_budget` | The requested joint normal table exceeds its admitted work or table-size limit; the requested row count and covariance are not changed. |
| `execution_resource_limit` | The execution could not obtain the shared worker or memory resources needed to run; no scientific conclusion follows from the refusal. |
| `choice_evidence_scope_mismatch` | Retained choice evidence does not describe the target: its ordered option set, the meaning of its choices (stated or observed) or its population differs, and no transfer with a discount was declared. Nothing is carried across; the refusal names the source. |
| `choice_evidence_overlap` | The same observation identity would be counted more than once, as two sources or as a source and the observations being conditioned on. The refusal names the identity. |
| `choice_evidence_conflict` | A source of choice evidence conflicts with the remaining evidence beyond the screening threshold and the policy refuses conflict. The diagnostic is a screen with unmeasured calibration. |
| `choice_evidence_invalid` | A choice-evidence declaration is not well formed, or the inference family refused its shape or size: fewer than two distinct options, a Dirichlet shape below one, a discount outside [0, 1], misaligned counts, or a study beyond the supported size. The refusal names the field. |

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
| `plug-in response Jacobian supports at most two treatments` | Query the Jacobian over at most two treatments at a time (one `ResponseJacobian` per pair, with the remaining treatments in the adjustment set where the graph supports it), or an `AverageDerivative` per treatment. |
| `plug-in directional derivative supports at most two treatments` | Restrict the direction to at most two treatments, or query a `ResponseJacobian` per pair and take the inner product with the direction. |

```python
query = antecedent.ResponseJacobian(["a", "b", "c"], ["y"], at=[0.0, 0.0, 0.0])
try:
    antecedent.analyze(data, query=query, graph=graph)
except antecedent.CausalError as err:
    print(err)         # plug-in response Jacobian supports at most two treatments
    print(err.remedy)  # query the Jacobian over at most two treatments at a time ...
```
