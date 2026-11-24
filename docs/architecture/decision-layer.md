# Decision layer

This page states what the decision layer is for, how its parts fit, and the invariants a change
must keep. It is written for a contributor who changes decision contracts, the trade-off engine,
the compiled decision route or the Python `antecedent.decision` module. How to use the layer is
in the guides: [Decisions](../decisions.md), [Trade-off ranges in depth](../preference-decisions.md)
and [Learn trade-off ranges](../preference-learning.md).

## Purpose and scope

The layer answers a causal decision question: which action to take, given what an analysis
established about each action's consequences and what the decision-maker values. It keeps three
uncertainties apart and reports each in its own terms:

- **causal**: what each action does, carried by the source (a mean, aligned joint draws, a set
  of causal structures);
- **preference**: how outcomes are valued against each other, declared as fixed numbers or as
  ranges of admissible trade-offs;
- **decision**: whether the chosen action changes across admissible structures and trade-offs.

A trade-off range is a declared assumption, never a probability. The admissible combinations are
every pairing of an admissible structure with an admissible trade-off; a trade-off is never
linked to a structure.

Declared trade-offs are the primary capability. Learning trade-offs from recorded choices
(`antecedent.decision_learning`) is optional and structurally separate: it produces
`AcceptedPreferences`, which a contract reads as its value exactly as it reads a declared one.
Learning never decides, and deciding never loads or reads learning code.

Deliberately excluded:

- partial pooling of preferences across decision-makers (sets are learned per decision-maker,
  compared, or pooled under one named assumption);
- chains of decisions in which one choice changes the options of a later one;
- proposing which question to put to a stakeholder next.

## One contract, two forms

The trade-off machinery extends the ordinary decision contract; it is not a second decision
system. In Rust, `PreferenceContract` (`antecedent_design::preference_contract`) reuses
`DecisionCriterion`, `HardConstraint`, `StructuralPolicy` and `ActionKind`, and
`PreferenceContract::bind(point)` lowers it to an ordinary `DecisionContract` at any trade-off
point. In Python, `decision.Contract` takes either form:

- **ordinary**: each action has its own `utility=` over positional inputs read by `decision.x(k)`;
- **value**: one `value=decision.Value(...)` over named attributes, each action naming its inputs
  by attribute.

`Contract.form`, `Contract.route` and `Contract.trade_offs` say which form and route apply. The
engine is chosen in Rust: describing a value contract reports its ordinary contract when one
exists, and `decided_across_preferences` when the value names a trade-off or the ordinary utility
language cannot state it. Python reads that flag and raises any other refusal of the ordinary form.

**A plain-number trade-off is the ordinary contract.**
`PreferenceContract::ordinary_contract` returns the bound ordinary contract exactly when the value
names no trade-off, and refuses with `preference_contract.no_ordinary_form` otherwise. A value
contract with an ordinary form is that contract in every respect:

- the same contract identity as the contract written by hand with `decision.x(k)`;
- the same result type (`Decision`), exact ties, `evpi` and regret amounts;
- the same export bytes and exact replay.

Two limits come from the ordinary utility language. `exp` or `ln` of an attribute has no ordinary
form, so such a value is decided by the trade-off engine. A plain number supplied for an attribute
that a hard constraint reads splits that constraint per action (`{id}@{action}`) inside the bound
contract and its identity; every result reports the exclusion under the declared id.

**A named fixed trade-off stays on the trade-off engine.** `decision.tradeoff(id, value=...)`
carries an assumption, so the result reports it. Its verdict, admissible actions, exclusions and
criterion values equal the hand-written contract's under every criterion, constraint set and
structural policy. The differences are deliberate and listed in
[fixed trade-offs and the tie tolerance](../preference-decisions.md#fixed-trade-offs-and-the-tie-tolerance):
tie rule, verdict words, summation order (`sum of E[term]` against `E[sum]`), no regret amounts
or `evpi`, structure exclusion on the exact route, and replay tolerance.

**Shared parts.** One set of each serves both forms:

| Part | Home |
| --- | --- |
| Identity encoders (utility, criterion, action kind, structural policy, constraint) | `antecedent_design::decision_contract` |
| Artifact wires (expressions, criteria, constraints, exclusions) | `antecedent_design::decision_wire` |
| Criterion kernel and weighted left inverse | `antecedent_design::decision_eval` |
| Structure masses and evidence gates | `antecedent_design::structure_aggregator` |
| Standing and statement of every result | `antecedent_design::decision_standing` (`Standing`, `DecisionOutcome`) |
| Claim and wire parsing in the Python bridge | `python/src/decision_claims.rs`, the design crate's decision wires |
| Compile goal | `CompileGoal::Decision(DecisionProgram)` |

Per-structure scores stay route-specific: the trade-off routes combine affine functions of the
trade-offs, the ordinary route combines scalars.

## Routes and what their verdicts mean

| Route | Used when | Verdict words |
| --- | --- | --- |
| Ordinary | the contract has an ordinary form | `uniquely_optimal`, `indistinguishable`, `no_admissible_action`; across structures, the structural and robust verdicts |
| Exact | value affine in its trade-offs, expectation criterion, no value relations | `necessarily_best`, `never_beaten`, `tied`, `preference_dependent`, the policy choices; per action: possibly best, dominated by, value range, worst-case regret; switches with thresholds |
| Certified | any other value or criterion, or a set with value relations | the exact route's words, plus `undetermined` with the undecided share |
| Sampled class | a declared uniform class prior is sampled above its exact limit | `best_on_sampled_measure`, `undetermined_on_sampled_measure`, `best_on_every_sampled_member`, `per_sampled_member` |

`Decision`, `TradeoffDecision` and `RobustDecision` share `DecisionResult`
(`python/antecedent/_decision_result.py`): `choice`, `kind`, `standing`, `statement`,
`contract_identity`, `contract`, `effective_tie_tolerance`, `receipt`. `standing` is computed in
Rust by an exhaustive match over each verdict and sent in every reply; Python keeps no verdict
table. `RobustDecision.kind` is the claim type it adapted; its verdict word is `verdict.kind`.
The reading of every standing is in [read any decision result](../decisions.md#read-any-decision-result).

The exact route answers every question by checked linear programs over the admissible set. The
certified route bounds each criterion on sub-boxes of the declared ranges with outward-rounded
interval arithmetic. Both are statements about the supplied law; the sampling error of that law
is not part of the ranges. See
[two routes and what each guarantees](../preference-decisions.md#two-routes-and-what-each-guarantees).

## Compiled route

`Contract.compile(source)` makes the decision part of the program of the analysis that estimates
its quantities (`crates/antecedent/src/analysis/execute/compile_preference.rs`). The source is a
study or prepared study, several studies, or a compiled joint design. An ordinary contract and a
value contract compile through one path (`CompiledContract::{Ordinary, Tradeoff}`):

1. **Bind.** Every quantity an action reads must be a coordinate the study's query targets:
   outcome, regime on the dose grid, population, horizon, mean on the identity scale, units. A
   refusal names the action and the attribute or input position.
2. **Demand.** The contract's source requirement sets the representation: means when they
   answer it, aligned joint draws otherwise. A hard constraint is a probability, so it demands
   draws.
3. **Lower.** The decision enters layer four of the study's program as a `DecisionProgram` whose
   utility is `UtilityForm::PreferenceValue`, naming every quantity read and whether means
   suffice. An ordinary contract states the same form with no trade-off parameters. The carrier
   is checked against the statistical realization, and a quantity no structure identifies is
   refused, before anything is fitted.
4. **Execute.** Each prepared structure runs the decision program its plan states. An ordinary
   contract is then evaluated by `evaluate_contract` or `evaluate_contract_on_means`, so its
   result and contract identity are those of the contract on the same source.

The receipt (`PreferenceDecisionReceipt`, Python `CompiledReceipt`) binds the bound and contract
identities, the causal contract, every compiled program and, aligned with them, the program each
study actually executed, plus the identities of the measure and options the execution read. An
executed program that differs from the compiled one refuses
(`preference_compile.executed_program_mismatch`).

Other decision programs keep exactly the plan facts they state without this route.

## Evidence overlap, design law and structure classes

**Several studies.** Separately estimated outcomes meet as one mean source when means suffice,
and otherwise through a propagation chain declared, checked and compiled at compile time; its
family table is bound to each study's fitted moments at execution. Before joining, the evidence
ledger judges which studies observe the same units:

- studies of one table share its rows, identified by row content and position, independent of
  any analysis mask or weights; projections and row selections carry their origin;
- studies of different tables are compared through a `UnitKey` in a named identifier space;
  keys that cannot be compared leave the overlap unknown and refuse;
- studies that share observations need a declared cross-study covariance block; it becomes the
  cross block of the chain's covariance.

**Randomized designs.** Randomization identifies arm means and a conservative bound for linear
combinations, not the cross-arm dependence of the arm-mean estimators. A value that needs a joint
law of a cluster-randomized, matched-pair or switchback design therefore needs a declared
sampling premise (`sampling=`), under which an independent-unit bootstrap estimates the law. The
premise, refits and seed enter the decision program identity and the result's assumptions.

**Structure classes.** A graph posterior's structures compile one by one; a CPDAG or PAG atom
compiles member by member. A count of members is never a probability. A `ClassPrior` declared
beside the posterior is the measure over members (uniform under a named assumption, or keyed by
member fingerprint), enumerated up to an exact limit and sampled uniformly above it when a uniform
prior declares a sample. Only a compiled decision reads a class prior; an ordinary estimate of the
same study refuses it (`structure_posterior.class_prior_not_read`). PAG atoms that differ only in
circle marks are different classes. Unidentified and unevaluated mass is reported and never
renormalized. An ordinary contract over a graph posterior refuses at compile
(`preference_compile.ordinary_structures_not_compiled`); `decision.Structures` decides it across
structures in hand.

The user-facing rules are in [which studies share evidence](../preference-decisions.md#which-studies-share-evidence),
[a joint law of a randomized design](../preference-decisions.md#a-joint-law-of-a-randomized-design)
and [weigh the members of an equivalence class](../preference-decisions.md#weigh-the-members-of-an-equivalence-class).

## Honesty rules

- **Refuse, never silently ignore.** An option the chosen engine or source does not read refuses
  with `option_not_applicable`: `policy=` or `measure=` on an ordinary contract, `admissibility=`
  on a trade-off contract, a compile argument the source does not read, a class prior an ordinary
  estimate does not read.
- **Never exact from sampled.** A sampled class gives a Monte Carlo verdict on the sampled
  measure or each sampled member's verdict, never a verdict over every member, and never an exact
  verdict word.
- **Witnessed point.** The certified route makes a choosing verdict only when a kept box lies
  wholly inside every relation or a point is proven inside with no tolerance. Without a witness the
  verdict is `undetermined`, neither a choice nor an empty set. A lead that is pure rounding never
  counts.
- **Effective tie tolerance.** Every result reports the tolerance its verdict applied: `0.0` for
  the ordinary route, `max(tie_tolerance, numerical bound of the margins)` for the trade-off
  routes. A record stores it only when the numerical bound widened it.
- **Ranges are not probabilities.** A share is reported only under a supplied or estimated
  measure, and statements never call a range or a volume fraction a probability.
- **Refusal stage.** A refusal keeps the stage of the check that raised it through every route:
  an ordinary evaluator refusal is reported exactly as the decision contract reports it, a
  structure check stays at `structural`.
- **Analysis errors are the analysis's.** A cancelled compile or execution is a
  `PreferenceRefusal` (`preference_compile.cancelled`); any other error of the study or design
  behind a compiled decision raises that analysis's own exception and reason code.
- **One refusal hierarchy.** `PreferenceRefusal` is a `DecisionRefusal`.

## Statistical choices

| Choice | Reason |
| --- | --- |
| Joined laws use the sample covariance, divisor `n - 1` | the family table's rows whiten with `rows - 1`, so the joined law reproduces each study's own covariance |
| Clustered choice-model covariance is the sandwich times `C/(C-1)` | the finite-cluster correction for maximum-likelihood estimators (Cameron and Miller, 2015) |
| Clustering needs `C >= 2q + 1` decision-makers, else refuses | the sandwich has at most `C - 1` degrees of freedom; the model covariance is never substituted, since it treats dependent records as independent |
| The shape-parameter fit refuses a saddle or a point away from a maximum | a profile interval around a non-maximum states nothing |
| Accepted-preference constructors relearn from the records and must reproduce what they are given | a learned object supplied by a caller is unverified until the records reproduce it; relations must come from recorded choices |
| Sampled class masses carry exact Clopper–Pearson intervals | binomial coverage of a sampled share without a normal approximation |
| Stated and recorded choices are never fitted together | they do not share a noise scale |
| An unavailable option teaches nothing | it was not a choice the decision-maker could make |

## Module map

| Concern | Rust | Python |
| --- | --- | --- |
| Ordinary contract, evaluation, structures | `antecedent_design::{decision_contract, decision_eval, decision_structural, structure_aggregator}` | `antecedent.decision` (`Contract`, `Action`, `Constraint`, `Criterion`, `Expr`, `x`, `Decision`), `antecedent.decision_robust` (`RobustDecision`, admissibility) |
| Value and trade-offs | `antecedent_design::{preference_model, preference_contract}` | `antecedent.decision` (`attribute`, `tradeoff`, `Tradeoff`, `ValueExpr`, `Value`, the shapes, `Curve`, `relation`, `Measure`, `TradeoffPolicy`, `Structures`) in `_decision_value.py` |
| Exact and certified routes | `preference_decision{,/region,/structural,/statements,/measure}`, `preference_certified{,/scoring,/subdivision}`, `preference_enclosure`, `preference_route_choice` (as `preference_evaluation`), `preference_sampled` | `_decision_tradeoff.py`, `TradeoffDecision` in `_decision_tradeoff_results.py` |
| Standing, refusals, artifacts | `decision_standing`, `preference_refusal`, `decision_artifact`, `preference_decision_artifact` | `DecisionResult`, `Standing`, `CompiledReceipt` in `_decision_result.py`; `DecisionRefusal`, `PreferenceRefusal`, `replay` in `decision.py` |
| Compiled route | `antecedent::analysis::execute::compile_preference{,/bind,/atoms,/structures,/outcomes,/joint,/evidence,/design,/facts}`; `antecedent::analysis::preference_decision` | `Contract.compile`, `CompiledDecision` in `_decision_compile.py`; sources in `_decision_sources.py`; `UnitKey` in `_row_evidence.py` |
| Learning | `antecedent_design::{choice_record, preference_set_learning, preference_choice_model, accepted_preferences}`; `antecedent_stats::{linear_program, glm::conditional_logit, glm::nonlinear_conditional_logit}` | `antecedent.decision_learning` |
| Supporting kernels | `antecedent_stats::binomial_interval`, `antecedent_graph::{mec_uniform, pag_uniform}`, `antecedent_estimate::joint_design::unit_bootstrap`, `antecedent_data` row origin | |

Rust compiled entry points: `Study::compile_for_preference_decision`,
`Study::compile_for_contract_decision` (and the same on `PreparedStudy`),
`compile_preference_decision_across_studies`, `compile_contract_decision_across_studies`,
`compile_preference_decision_with_shared_evidence`, the `_across_prepared_studies` forms,
`PreparedNativeJointDesign::compile_for_preference_decision` and `::compile_for_contract_decision`,
returning `CompiledDecision` (`CompiledPreferenceDecision`).

## Invariants

1. A plain-number value has the hand-written contract's identity and returns a `Decision`: `python/tests/test_decision_value_contract.py::test_a_plain_number_trade_off_is_the_ordinary_contract`, `crates/antecedent-design/tests/preference_contract.rs::an_ordinary_contract_exists_exactly_when_no_trade_off_is_named`.
2. A plain-number value exports the hand-written contract's bytes and replays exactly: `python/tests/test_decision_value_contract.py::test_fixed_value_export_bytes_equal_hand_written_and_replay_exactly`.
3. A plain-number value ties by exact equality; a trade-off contract within its effective tolerance: `python/tests/test_decision_value_contract.py::test_fixed_value_ties_exactly`, `crates/antecedent-design/tests/preference_evaluation.rs::near_ties_follow_the_documented_tie_rules`, `::certified_near_ties_follow_the_documented_tie_rules`.
4. A named fixed trade-off, or `exp`/`ln` of an attribute, is decided by the trade-off engine: `python/tests/test_decision_value_contract.py::test_named_fixed_tradeoff_is_a_trade_off_decision`, `::test_exp_of_an_attribute_without_trade_offs_uses_the_trade_off_engine`.
5. A fixed trade-off on the trade-off engine gives the hand-written contract's verdict, admissibility, exclusions and values under every criterion, constraint set and structural policy: `crates/antecedent-design/tests/preference_evaluation.rs::fixed_trade_offs_decide_as_a_hand_written_contract_under_every_criterion`, `::fixed_trade_offs_decide_across_structures_as_a_hand_written_contract`, `::structure_exclusions_follow_the_documented_rules`.
6. Exclusions are reported under the declared constraint id: `crates/antecedent-design/tests/preference_evaluation.rs::exclusions_are_reported_under_the_declared_constraint_ids`, `python/tests/test_decision_value_contract.py::test_a_constraint_bound_per_action_is_reported_under_its_declared_id`.
7. A contract with a trade-off range has no ordinary form, and ordinary-only consumers refuse it: `python/tests/test_decision_value_contract.py::test_ranged_contract_has_no_ordinary_form`.
8. Contract identities and artifact bytes are pinned: `crates/antecedent-design/tests/preference_contract.rs::identities_are_pinned`, `crates/antecedent-design/tests/decision_artifact.rs::contract_and_result_bytes_are_pinned`, `crates/antecedent-design/tests/preference_artifact.rs::decision_artifact_bytes_are_pinned`.
9. Every result reads `choice`, `kind`, `standing`, `statement`, with standing from Rust: `python/tests/test_decision_result_family.py::test_every_result_has_choice_statement_kind_standing`, `::test_standing_of_every_native_kind`, `crates/antecedent-design/src/decision_standing.rs::tests::exact_preference_verdicts_map_to_standing_and_choice`.
10. `PreferenceRefusal` is a `DecisionRefusal`; an ordinary refusal is not a `PreferenceRefusal`: `python/tests/test_decision_result_family.py::test_refusal_hierarchy`.
11. A refusal keeps its stage through every route: `crates/antecedent-design/src/preference_refusal.rs::tests::an_ordinary_evaluator_refusal_keeps_its_own_stage_through_every_preference_route`.
12. A value action takes its kind third: `python/tests/test_decision_value_contract.py::test_a_value_action_takes_its_kind_third`.
13. Deciding reads and loads no learning code: `crates/antecedent-design/tests/preference_evaluation.rs::the_explicit_route_and_its_record_read_no_preference_learning_code`, `::declared_ranges_decide_without_records_accepted_preferences_or_a_measure`, `python/tests/test_decision_value_contract.py::test_deciding_with_stated_trade_offs_loads_no_learning`, `::test_importing_a_deciding_module_loads_no_learning`.
14. An option the engine or source does not read refuses with `option_not_applicable`: `python/tests/test_decision_value_contract.py::test_value_form_validation`, `python/tests/test_preferences.py::test_compile_refuses_an_argument_its_source_does_not_read`, `crates/antecedent/tests/ordinary_contract_compiled.rs::a_measure_over_preferences_is_refused_for_an_ordinary_contract`.
15. The plan states quantities, representation and carrier in layer four before anything is fitted: `crates/antecedent/tests/preference_decision_compiled.rs::the_plan_names_the_decisions_quantities_and_carrier_before_anything_is_fitted`, `python/tests/test_preferences.py::test_compile_shows_the_plan_before_anything_is_fitted`.
16. An ordinary contract's plan states the value form with no trade-off parameters: `crates/antecedent/tests/ordinary_contract_compiled.rs::the_plan_names_an_ordinary_contracts_quantities_and_carrier_before_anything_is_fitted`.
17. Other decision programs keep the plan facts they state without this route: `crates/antecedent/tests/preference_decision_compiled_plan_facts.rs::an_ordinary_decision_plan_states_exactly_the_facts_it_always_did`.
18. A quantity no structure identifies refuses at compile: `crates/antecedent/tests/preference_decision_compiled.rs::a_quantity_not_identified_under_the_structure_is_refused_at_compile`, `python/tests/test_preferences.py::test_compile_refuses_a_quantity_no_structure_identifies`.
19. A compiled ordinary contract gives the contract's own evaluation on the produced source: `crates/antecedent/tests/ordinary_contract_compiled.rs::means_answer_an_ordinary_contract_exactly_as_the_contract_evaluates_them`, `::joint_draws_answer_a_quantile_exactly_as_the_contract_evaluates_them`, `python/tests/test_decision_compile.py::test_an_ordinary_contract_compiles_with_a_prepared_study`.
20. Each study executes the program its plan states, and the receipt names it with the measure and options read: `crates/antecedent/tests/preference_decision_compiled_program_identity.rs::one_study_executes_the_decision_program_its_receipt_names`, `::the_receipt_binds_the_options_and_measure_the_execution_read`, `crates/antecedent/tests/ordinary_contract_compiled.rs::the_receipt_names_the_decision_program_that_ran`.
21. Only compiled results carry a receipt: `python/tests/test_decision_result_family.py::test_receipt_only_on_compiled_results`.
22. An ordinary contract over a graph posterior refuses at compile: `crates/antecedent/tests/ordinary_contract_compiled.rs::an_ordinary_contract_over_a_graph_posterior_is_refused_before_anything_is_fitted`, `python/tests/test_decision_compile.py::test_ordinary_compile_over_graph_posterior`.
23. A hard constraint demands draws at compile: `python/tests/test_preferences.py::test_a_hard_constraint_demands_draws_at_compile`.
24. The chain joining separate studies is compiled before fitting and is the program executed: `crates/antecedent/tests/preference_decision_compiled_chain_binding.rs::the_chain_is_compiled_before_anything_is_fitted_and_is_the_program_executed`.
25. Studies that share observations refuse without a declared cross-study covariance: `crates/antecedent/tests/preference_decision_compiled_joint_law.rs::studies_that_share_observations_need_a_declared_cross_study_covariance`.
26. Row identity ignores analysis mask and weights: `crates/antecedent/tests/preference_decision_compiled_evidence_basis.rs::a_mask_or_weights_do_not_make_one_tables_rows_other_observations`.
27. Projections share rows; keyed studies are judged by shared units; incomparable keys leave overlap unknown: `crates/antecedent/tests/preference_decision_compiled_unit_keys.rs::projections_of_one_table_share_its_rows`, `::keyed_studies_of_different_tables_are_judged_by_the_units_they_share`, `::keys_that_cannot_be_compared_leave_the_overlap_unknown`.
28. The joined law reproduces each study's sample covariance: `crates/antecedent/tests/preference_decision_compiled_evidence_basis.rs::the_joined_law_reproduces_each_studys_own_covariance`.
29. A randomized design's joint law needs a declared sampling premise: `crates/antecedent/tests/preference_decision_compiled_design.rs::a_joint_law_of_a_randomized_design_needs_a_declared_sampling_premise`, `python/tests/test_preferences.py::test_a_cluster_design_reads_its_independent_unit_bootstrap_under_a_declared_sampling_premise`.
30. A count of class members is never a probability: `crates/antecedent/tests/preference_decision_compiled_atoms.rs::a_class_atom_compiles_completion_by_completion_without_making_counts_probabilities`, `python/tests/test_preference_structures.py::test_bayes_over_structures_without_a_measure_still_refuses`.
31. A sampled class is never given an exact verdict: `crates/antecedent/tests/preference_decision_class_weighting.rs::a_class_above_the_exact_limit_is_sampled_and_never_called_best_everywhere`, `crates/antecedent/tests/preference_decision_compiled_class_marks.rs::a_sampled_mixture_is_never_read_as_the_class`, `python/tests/test_preference_structures.py::test_a_sampled_result_exposes_no_exact_verdict`.
32. PAG atoms that differ only in circle marks are different classes: `crates/antecedent/tests/preference_decision_compiled_class_marks.rs::pag_atoms_that_differ_only_in_circle_marks_are_different_classes`.
33. An ordinary graph-posterior estimate refuses a class prior it does not read: `crates/antecedent/tests/preference_decision_class_weighting.rs::the_graph_posterior_mixture_refuses_a_class_prior_it_does_not_read`, `python/tests/test_preference_structures.py::test_the_graph_posterior_estimate_refuses_a_class_prior_it_does_not_read`.
34. Unidentified and unevaluated mass is never renormalized: `crates/antecedent-design/tests/preference_decision_structural.rs::weighting_over_structures_mixes_coefficients_without_renormalizing`, `crates/antecedent/tests/preference_decision_compiled_structures.rs::bayes_over_compiled_structures_keeps_the_unidentified_mass`.
35. A certified choice needs a witnessed point: `crates/antecedent-design/tests/preference_certified.rs::a_gap_below_the_feasibility_tolerance_is_undetermined_not_a_choice`, `::value_relations_that_leave_no_point_choose_nothing`, `::a_thin_but_non_empty_set_keeps_its_choice`.
36. A lead that is pure rounding is a tie: `crates/antecedent-design/tests/preference_certified.rs::a_lead_that_is_pure_rounding_does_not_count_toward_a_measure_share`, `crates/antecedent-design/tests/preference_decision.rs::a_margin_below_the_rounding_of_large_expected_values_is_a_tie`.
37. The effective tie tolerance is reported, and stored only when widened: `crates/antecedent-design/tests/preference_artifact.rs::the_effective_tie_tolerance_is_stored_only_when_a_numerical_bound_widens_it`, `python/tests/test_preferences.py::test_the_effective_tie_tolerance_is_reported`.
38. The certified route never contradicts the exact route on affine problems: `crates/antecedent-design/tests/preference_evaluation.rs::the_certified_route_never_contradicts_the_exact_route_on_affine_problems`.
39. A mean source answers only the exact route: `crates/antecedent-design/tests/preference_evaluation.rs::a_mean_source_answers_only_the_exact_route`.
40. No statement calls a range or a share a probability: `crates/antecedent-design/tests/preference_decision.rs::statements_never_call_a_range_or_share_a_probability`.
41. A record re-exports byte for byte with every stored option: `python/tests/test_preferences.py::test_a_record_written_with_every_option_re_exports_byte_identically`.
42. A cancelled run is a refusal, never a verdict, on both engines: `python/tests/test_preferences.py::test_a_cancelled_compile_or_execution_is_a_preference_refusal`, `python/tests/test_decision_value_contract.py::test_a_cancelled_execution_refuses_the_ordinary_engine`.
43. The clustered covariance carries `C/(C-1)`: `crates/antecedent-stats/tests/conditional_logit.rs::cluster_sandwich_matches_the_hand_formula_in_the_two_option_case`.
44. Clustering needs `C >= 2q + 1` decision-makers: `crates/antecedent-design/tests/preference_choice_model.rs::the_covariance_is_clustered_only_with_enough_deciders`.
45. The shape fit refuses a non-maximum: `crates/antecedent-stats/src/glm/nonlinear_conditional_logit.rs::tests::a_saddle_in_the_shape_directions_is_refused`, `crates/antecedent-stats/tests/nonlinear_conditional_logit.rs::a_polish_that_cannot_reach_a_stationary_point_is_refused`.
46. Accepted preferences are relearned and must reproduce what was supplied, keeping every declaration: `crates/antecedent-design/tests/accepted_preferences.rs::edited_learned_content_is_refused_at_acceptance`, `::a_learned_set_that_drops_a_declaration_is_refused`.
47. Sampled class masses use exact Clopper–Pearson bounds: `crates/antecedent/tests/preference_decision_class_weighting.rs::clopper_pearson_bounds_match_their_closed_forms`.
48. An unavailable option teaches nothing; stated and recorded choices are never fitted together: `crates/antecedent-design/tests/preference_set_learning.rs::an_unavailable_option_teaches_nothing`, `crates/antecedent-design/tests/preference_choice_model.rs::stated_and_recorded_choices_are_never_fitted_together`.
49. A non-cancellation error of the study or design behind a compiled decision is that analysis's own exception and reason code, never a decision refusal: `python/tests/test_decision_compile.py::test_an_analysis_error_behind_a_compiled_decision_is_the_analysis_own`.
