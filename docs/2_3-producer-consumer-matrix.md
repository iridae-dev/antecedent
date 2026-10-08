# Producer and consumer matrix (2.3, milestone C1)

Which object can be given to which decision-side consumer, by which route, and what is
refused. The rule behind every cell: **a consumer never infers a joint law, a posterior or a
causal response from a neighbouring object.** A mean is not an outcome distribution, a point
estimate is not a response, a posterior summary is not a prior signal, and a decision result is
not a source. Every route here is point-only; nothing on this page is an interval, a coverage
statement or a probability license.

The executable source of truth is `python/tests/test_c1_matrix.py`. Its `STATUS` table holds
the 88 cells asserted below; `test_c1_matrix_document_agrees_with_the_executable_table` fails
if this page and the table drift apart. Each producer row is `test_c1_matrix_<row>_row`,
parametrized over the eight consumers.

## Reading a cell

- **DIRECT**: a licensed entry point takes the producer (Python and Rust entry named).
- **ADAPTER**: a named step takes it: a different entry point, or a declaration the caller
  writes. An adapter is a caller assertion, not a verification of the producer.
- **REFUSED**: the consumer must not take it. The refusal code and detail are the ones the code
  emits, or, where a consumer does not type its input, the rejection is stated as untyped.

A REFUSED cell is checked by feeding the producer to **every** entry point of the consumer and
requiring an exception, with the exact type where the consumer types its input
(`CausalTypeError`, `ScenarioDecisionRefusal`, `CompositionBundleRefusal`).

## Producers (rows)

| Row key | Object | Built from |
| --- | --- | --- |
| `native_response` | a native response claim | `program_claims.native_claim(view, program)` (`NativeClaim`) |
| `fitted_effect_model` | a fitted effect model | `compact_export.CompactExport.build(...)` (coefficients, covariance, support, mask) |
| `retargeted_batch` | a retargeted batch result | `PreparedBatch.retarget(...)` (`BatchRetarget`); `recalc.RecalcSession.execute(...)` with `TargetWeights` |
| `external_claim` | an external bound claim | `external.response(...).bind(response)` (`BoundExternalClaim`) |
| `joint_draws` | aligned joint draws | `joint_distribution.JointDistributionArtifact` |
| `bayesian_prior` | Bayesian prior inputs | `inference.PosteriorArtifact` (learned posterior), `priors.PriorCatalog`, `priors.ComposedPrior` |
| `identified_scenarios` | identified sets and scenarios | a prepared transport scenario stage, CPDAG completions, `inverse_query.IdentifiedSet`, `decision_robust.IdentifiedUtility` |
| `sensitivity_artifact` | a sensitivity artifact | `sensitivity_decision.SensitivityArtifact` |
| `decision` | a decision result | `decision.Decision` (`Contract.evaluate(...)`) |
| `study_ranking` | a study ranking | `design_ranking.DesignRankingResult` (`rank_designs`) |
| `failed_contract` | a failed identification contract (added: the only producer `repair` takes) | `repair.BackdoorContract` or `TransportContract` |

`failed_contract` is not in the TODO list of producers; it is added so that the `repair`
column has a DIRECT route. "External bound claim" is read as a bound external claim
(`BoundExternalClaim`); a partial-identification interval is a separate field
(`IdentifiedBound`) of a sensitivity artifact, not a producer here.

## Consumers (columns) and their entry points

| Column key | Python entry | Rust entry |
| --- | --- | --- |
| `decision` | `Contract.evaluate`, `composition.evaluate_with_support`, `decision_robust.finite_scenarios`, `decision_robust.identified_sets` | `decision_eval::evaluate_contract`, `evaluate_contract_on_means`, `composition_boundary`, `decision_robustness::evaluate_robust`, `decision_adapters::evaluate_identified_sets` |
| `inverse_query` | `InverseQuery.evaluate(point=, interval_region=, identified_set=, scenarios=)` | `inverse_query::evaluate_inverse_query` |
| `design_ranking` | `design_ranking.rank_designs`, `evsi`, `consume` | `evsi::evaluate_evsi`, `design_ranking_artifact`, `prior_signal::adapt_prior_to_signal` |
| `sensitivity_decision` | `sensitivity_decision.decide` | `sensitivity_decision::evaluate_sensitivity_decision` |
| `scenario_decision` | `decide_from_scenarios`, `decide_from_cpdag_completions` | `analysis/decision_claims.rs`, `decision_adapters` |
| `repair` | `repair.obligations`, `repair.repair` | `repair::repair`, `obligation_adapters` |
| `bundle_node` | `composition_bundle.Bundle.builder().add_artifact` | `composition_bundle::CompositionBundle`, `composition_verifiers::standard_consumer` |
| `recalc_plan` | `recalc.plan_recalculation`, `RecalcSession.execute` | `analysis/recalc_cell.rs`, `analysis/recalc_receipt.rs` |

## The matrix

| Producer | `decision` | `inverse_query` | `design_ranking` | `sensitivity_decision` | `scenario_decision` | `repair` | `bundle_node` | `recalc_plan` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `native_response` | DIRECT: `NativeClaim.as_decision_source(contract).source` into `Contract.evaluate` (A 3, B 3.5); Rust `evaluate_contract_on_means`; a quantile or probability contract is REFUSED `native_claims.source_not_supplied` (`decision_contract_unsatisfied`) | ADAPTER: `NativeDecisionSource.mean_claim()` into `InverseQuery.evaluate`; the claim itself is rejected (`CausalTypeError`); quantile or probability REFUSED `decision_evaluation.mean_source_insufficient` | REFUSED: no entry takes it; as a prior it is rejected untyped | REFUSED: `CausalTypeError`, artifact must be a `SensitivityArtifact` | REFUSED: `CausalTypeError`; the CPDAG entry refuses `scenario_decision.no_native_run` (`not_executed`) | REFUSED: `CausalTypeError` from `obligations` and `repair` | REFUSED: no export; `CausalTypeError` (an artifact is its container bytes) | ADAPTER: the caller declares `stage_identity(Stage.QUERY, program_identity)`; the plan reads the digest only |
| `fitted_effect_model` | REFUSED: `Contract.evaluate` rejects it untyped; `evaluate_with_support` raises `CausalTypeError`; a point prediction is not a decision source | REFUSED: `CausalTypeError` (not a forward claim) | REFUSED: rejected untyped as a prior | REFUSED: `CausalTypeError` | REFUSED: `CausalTypeError`; CPDAG `scenario_decision.no_native_run` | REFUSED: `CausalTypeError` | REFUSED: its container is not a bundle kind; `CompositionBundleRefusal` | ADAPTER: `stage_identity(Stage.SCORE_ARTIFACT, export.identity)` declared by the caller |
| `retargeted_batch` | REFUSED: untyped rejection from `Contract.evaluate`; `CausalTypeError` from `evaluate_with_support`; no law or population is inferred from retargeted points | REFUSED: `CausalTypeError` | REFUSED: rejected untyped as a prior | REFUSED: `CausalTypeError` | REFUSED: `CausalTypeError`; CPDAG `scenario_decision.no_native_run` | REFUSED: `CausalTypeError` | REFUSED: no export; `CausalTypeError` | DIRECT: `RecalcSession.execute` with `TargetWeights` is the retargeting route itself (scores reused, 0 fold fits, 1 reweight, 1 decision); a `BatchRetarget` object is not an input there |
| `external_claim` | DIRECT: `Contract.evaluate(claim)` (wait 1, treat 9; no EVPI; `export` REFUSED `decision_evaluation.mean_source_not_replayable`), `DecisionInput.from_claim`; Rust `evaluate_contract_on_means`; probability or quantile REFUSED `composition_boundary.mean_is_not_a_distribution` | DIRECT: `InverseQuery.evaluate(claim)` (target mean 5 keeps `treat`); quantile REFUSED `decision_evaluation.mean_source_insufficient` | REFUSED: rejected untyped as a prior | REFUSED: `CausalTypeError` | REFUSED: `CausalTypeError`; CPDAG `scenario_decision.no_native_run` | REFUSED: `CausalTypeError` | DIRECT: `add_artifact("external_claim", claim)` with `composition_bundle.mean_decision`, labelled `point_only_attested`; a nonlinear utility REFUSED `composition_bundle.unsupported_law` | ADAPTER: `stage_identity(Stage.external_study(0), provider_fingerprint)` declared by the caller |
| `joint_draws` | DIRECT: `Contract.evaluate(law)` (risky 2, safe 3, EVPI 1/2), `DecisionInput.from_distribution`; Rust `evaluate_contract`; the only producer that answers `P(utility >= t)` here (1/2) | DIRECT: `InverseQuery.evaluate(law)` (means 2 and 3; target 2.5 keeps `safe`) | REFUSED: rejected untyped as a prior (a joint law is not a scalar-state prior) | REFUSED: `CausalTypeError` | REFUSED: `CausalTypeError`; CPDAG `scenario_decision.no_native_run` | REFUSED: `CausalTypeError` | DIRECT: `add_artifact("distribution", law)` beneath a contract and result, labelled `joint_draw`; Rust `composition_verifiers` | ADAPTER: `stage_identity(Stage.LAW, source_digest)` declared by the caller |
| `bayesian_prior` | REFUSED: untyped rejection from `Contract.evaluate`; `CausalTypeError` from `DecisionInput.from_distribution` and `evaluate_with_support` | REFUSED: `CausalTypeError` | ADAPTER: the caller summarises the posterior into `design_ranking.Prior.draws(...)` (prior EU 0.05, EVPI 1/60); the checked Rust `prior_signal::adapt_prior_to_signal` is not exported to Python; the artifact as a prior is rejected untyped | REFUSED: `CausalTypeError` | REFUSED: `CausalTypeError`; CPDAG `scenario_decision.no_native_run` | REFUSED: `CausalTypeError` | REFUSED: not a bundle kind; `CompositionBundleRefusal` (shared priors are declared with `relate(..., "shared_prior")`, not embedded) | ADAPTER: `stage_identity(Stage.prior(0), ...)` declared by the caller |
| `identified_scenarios` | ADAPTER: `decision_robust.finite_scenarios` over per-structure laws and `identified_sets` over intervals (invariant best A; `bayes_over_structures` over a set REFUSED `decision_adapters.bayes_over_identified_set`); the stage itself is not a source | ADAPTER: `IdentifiedSet` and `Scenario.evaluated` built by the caller from each structure's law (exhaustive set: `safe` feasible; not exhaustive: unevaluated) | REFUSED: rejected untyped as a prior | REFUSED: `CausalTypeError` | DIRECT: `decide_from_scenarios(contract, stage, policy, outcomes=...)` (direct leader `treat`, standardize leader `hold`); Bayes without declared weights REFUSED `decision_claims.probabilities_not_declared` | REFUSED: `CausalTypeError` | REFUSED: the stage export is not a bundle kind; `CompositionBundleRefusal` | ADAPTER: `stage_identity(Stage.GRAPH, digest of the stage export)` declared by the caller |
| `sensitivity_artifact` | REFUSED: untyped rejection from `Contract.evaluate`; `CausalTypeError` from `evaluate_with_support` | REFUSED: `CausalTypeError` | REFUSED: rejected untyped as a prior | DIRECT: `sensitivity_decision.decide(artifact.contract(), artifact)` (assumption-dependent switch, exact tie at 1); a composed sampling interval REFUSED `cell_not_licensed` | REFUSED: `CausalTypeError`; CPDAG `scenario_decision.no_native_run` | REFUSED: `CausalTypeError` | DIRECT: `add_artifact("sensitivity", artifact.export())`; node identity is the artifact digest | ADAPTER: `stage_identity(Stage.EVIDENCE, identity digest)` declared by the caller |
| `decision` | REFUSED: a result is not a source; untyped rejection from `Contract.evaluate`, `CausalTypeError` from `evaluate_with_support` | REFUSED: `CausalTypeError` | ADAPTER: the contract (a `Contract`, whose action set is checked, or its identity string, which binds only the identity) frames `design_ranking.Decision`; EVSI 1/4, net 3/20 | REFUSED: `CausalTypeError` | REFUSED: `CausalTypeError`; CPDAG `scenario_decision.no_native_run` | REFUSED: `CausalTypeError` | DIRECT: `result.export()` as a `decision_result` node beneath its law and contract (risky 2, safe 3); a mean-source result is exported only by `composition_bundle.mean_decision` | ADAPTER: `stage_identity(Stage.DECISION, contract_identity + source_digest)` declared by the caller |
| `study_ranking` | REFUSED: untyped rejection from `Contract.evaluate`; `CausalTypeError` from `evaluate_with_support` | REFUSED: `CausalTypeError` | DIRECT: `design_ranking.consume(ranking.export(), expected=ranking.expectation())` (EVSI 1/4 and 1/8, net 3/20 and 1/40; external signals never natively replayed); a changed cost map REFUSED `design_ranking.cost_units_mismatch` (`design_cost_units_mismatch`) | REFUSED: `CausalTypeError` | REFUSED: `CausalTypeError`; CPDAG `scenario_decision.no_native_run` | REFUSED: `CausalTypeError` | DIRECT: `add_artifact("study_ranking", ranking.export())`; it cannot stand in for a `decision_result` (`CompositionBundleRefusal`) | ADAPTER: `stage_identity(Stage.UTILITY, ranking.identity)` declared by the caller; the stage model has no ranking stage |
| `failed_contract` | REFUSED: untyped rejection from `Contract.evaluate`; `CausalTypeError` from `evaluate_with_support` | REFUSED: `CausalTypeError` | REFUSED: rejected untyped as a prior | REFUSED: `CausalTypeError` | REFUSED: `CausalTypeError`; CPDAG `scenario_decision.no_native_run` | DIRECT: `repair.obligations(contract)` (one joint-law obligation over t, y, z1, z2) and `repair.repair(contract, candidates)` (only the cohort that observes all four repairs it); Rust `repair::repair` | REFUSED: no export; `CausalTypeError` | ADAPTER: `stage_identity(Stage.QUERY, contract_id)` declared by the caller |

Counts: 88 cells, **15 DIRECT**, **15 ADAPTER**, **58 REFUSED**, **0 UNVERIFIED**. Every cell is
executed by `test_c1_matrix_<row>_row[<consumer>]`.

## Refusals the matrix relies on

| Code | Detail | Where |
| --- | --- | --- |
| `decision_contract_unsatisfied` | `native_claims.source_not_supplied` (expected `joint_draws,marginal_draws`, supplied `mean`) | `NativeClaim.as_decision_source` for a quantile, probability or nonlinear contract |
| `route_not_supported` | `native_claims.unsupported_estimand` (expected `mean_curve`) | `native_claim` for an estimand that is not a static response curve |
| `decision_contract_unsatisfied` | `decision_evaluation.mean_source_insufficient` | `InverseQuery.evaluate` and `Contract.evaluate` asking a mean for a quantile, probability, constraint or nonlinear utility |
| `route_not_supported` | `decision_evaluation.mean_source_not_replayable` | `Decision.export` over a mean source |
| `decision_contract_unsatisfied` | `composition_boundary.mean_is_not_a_distribution` | `evaluate_functional` over a mean or scalar input |
| `joint_law_required` | `composition_boundary.paired_draws_across_sources` | draws paired across a mean source or two sources |
| `decision_contract_unsatisfied` | `decision_adapters.bayes_over_identified_set` | `identified_sets` under `bayes_over_structures` |
| `decision_contract_unsatisfied` | `decision_claims.probabilities_not_declared` | `decide_from_scenarios` under `bayes_over_structures` without declared weights |
| `not_executed` | `scenario_decision.no_native_run` | `decide_from_cpdag_completions` over anything that is not a native CPDAG run |
| `cell_not_licensed` | `sensitivity_decision_composition.composition_not_licensed` | a sampling interval composed with an assumption range |
| `joint_law_required` | `composition_bundle.unsupported_law` | a mean decision whose functional needs a joint law, or a joint-law result over a mean-only claim |
| `design_cost_units_mismatch` | `design_ranking.cost_units_mismatch` | `design_ranking.consume` under a different cost map |

## Findings

- **No neighbour is accepted.** In no REFUSED cell does a consumer accept a neighbouring
  object; no joint law, posterior or causal response is reconstructed from a mean, a point
  estimate, a retargeted point, a posterior summary or a decision result.
- **Untyped rejection (a structured-refusal gap, not an acceptance).** `Contract.evaluate`
  reads `source._native` and `design_ranking.Decision` calls `prior._wire()`, so a neighbouring
  object fails with an `AttributeError` or a native `TypeError` rather than a
  `CausalTypeError`. `composition_bundle.add_artifact` calls `obj.export(artifact_id=...)`, so an
  object whose `export` takes no such argument (for example `CompactExport`) also fails with a
  plain `TypeError`; passing its bytes is refused as a `CompositionBundleRefusal`.
  `test_c1_matrix_untyped_rejections_are_a_tracked_defect` pins the set so it can only shrink.
- **Identity-only binding.** `design_ranking.Decision(contract="<identity>", actions=...)`
  binds a contract by identity string only and does not check the action set against it; a
  `Contract` object does check it.
- **No checked prior adapter in Python.** `prior_signal::adapt_prior_to_signal` (shared-source
  overlap and transport-policy checks) exists only in Rust; the Python prior-to-EVSI step is the
  caller's.
- **Recalc adapters are declarations.** The plan compares caller-supplied digests and never
  reads the producer, so it cannot notice a different object that reuses the same identity
  string.
- **Native Python claims are mean-only.** The response view keeps no draws, so a joint law from a
  native Python response needs draws retained there; Rust supplies one from credible draws.

## Tests that cite this page

`test_c1_matrix_declares_every_cell_once_and_counts_them`,
`test_c1_matrix_document_agrees_with_the_executable_table`,
`test_c1_matrix_native_response_row`, `test_c1_matrix_fitted_effect_model_row`,
`test_c1_matrix_retargeted_batch_row`, `test_c1_matrix_external_claim_row`,
`test_c1_matrix_joint_draws_row`, `test_c1_matrix_bayesian_prior_row`,
`test_c1_matrix_identified_scenarios_row`, `test_c1_matrix_sensitivity_artifact_row`,
`test_c1_matrix_decision_row`, `test_c1_matrix_study_ranking_row`,
`test_c1_matrix_failed_contract_row`,
`test_c1_matrix_untyped_rejections_are_a_tracked_defect` and
`test_c1_matrix_a_mean_never_answers_a_probability_a_quantile_or_a_law`, all in
`python/tests/test_c1_matrix.py`.
