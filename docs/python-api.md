# Python API

New to Antecedent? Start with the [Python quickstart](start/first-analysis.md). For configuration details, use the [workflow reference](python-options.md).

<!-- pdoc adds this sibling directory after MkDocs builds the narrative. -->
<strong><a href="../python/antecedent.html">Browse the Python API</a></strong>

Read the Docs builds this reference from the matching Antecedent release (and
falls back to the checked-out source only before that release reaches PyPI).
The link stays within the documentation version you are reading.

`discovery.GraphPosterior.from_graphs()` requires atoms of one graph class and
matching variable order. Its packed ADMG representation supports bow-free
graphs: a pair may carry a directed edge or a bidirected edge, but cannot carry
both. A bow raises an argument error rather than dropping its directed edge or
changing the posterior's scientific identity. Use an explicit `Admg` with
`identify(graph=..., query=...)` or `prepare(data, graph=..., query=...)` to
retain such a graph; identification can still refuse an unidentifiable effect.
This encoding limit does not prohibit bows in the native ADMG graph model.

To generate a local reference for the installed package, run:

```bash
python -m pip install antecedent pdoc
python -m pdoc antecedent -o site/python
```

## Go directly to a public symbol

| Task | Reference |
| --- | --- |
| Analyze or prepare | <a href="../python/antecedent.html#analyze">analyze</a>, <a href="../python/antecedent.html#prepare">prepare</a> |
| Persist and consume | <a href="../python/antecedent.html#load">load</a>, <a href="../python/antecedent/artifacts.html">artifacts</a> |
| Interpret a result | <a href="../python/antecedent/results.html">results</a> |
| Correct measurement | <a href="../python/antecedent/measurement_models.html">measurement_models</a> |
| Track evidence overlap | <a href="../python/antecedent/evidence_ledger.html">evidence_ledger</a> |
| Ask a derivative or cross-world question over time | [TemporalQuery](causal-responses.md#ask-a-derivative-or-cross-world-question-over-time) |
| Declare adaptive procedures | <a href="../python/antecedent/adaptive_inference.html">adaptive_inference</a> |
| Bind a callback contract | <a href="../python/antecedent/provider_callbacks.html">provider_callbacks</a> |

## Task guides

The Python surface is documented by task in these guides:

- [Notebook to system](guides/notebook-to-system.md): retained identification, guarded design updates, explicit repreparation, checked mean composition and independent artifact consumption.
- [Migrate from 2.4 to 3.0](migration.md): immutable preparation, explicit execution controls, outcomes, typed artifacts and cyclic exploration.
- [Evaluate actions with a decision contract](decisions.md): decide with `Contract.evaluate(result)` or on a bound external claim.
- [Population, time and uncertainty](population-time-uncertainty.md): exact measured adapter scopes and `inference.MeasuredInference.load` with retained source identity; every scalar needs its own current record.
- [Design ranking](design-ranking.md): the shared `design.rank_designs` entry point for identification, structural sufficiency, information objectives and decision value; `DesignDecision.from_contract`, `design.adapt_prior_to_signal` and `design.consume`.
- [Sensitivity and robustness](sensitivity-and-robustness.md): `mnar_sensitivity.mnar_ate_sensitivity`, `msm_sensitivity.msm_ate_sensitivity`, `sensitivity_decision`, `decision_robust` and `mechanism_discrepancy.diagnose_mechanism_discrepancy`.
- [Transport and counterfactuals](transport-counterfactuals.md): the affine-additive transported path-specific effect, `MechanismSelection`, the scenario invariance report and the source-bound learned joint measured adapter.
- [Prepared recalculation](recalculation-capabilities.md): checked family sessions, selective recalculation, bounded external mean callbacks, frozen scores and portable resume.
- [Transported decisions and measurement](guides/transported-decisions-and-measurement.md) and [inverse queries and decisions](guides/inverse-outcome.md#inverse-queries-on-composed-cohort-and-transported-sources): `transported_response` (`TransportedResponse`, `TransportDeclaration`, `evaluate_transported`) and `inverse_sources` (`composed_forward_source`, `cohort_forward_source`, `transported_forward_source`, `observational_scenarios`).
- [Assumption range and revalidation](guides/assumption-range-and-revalidation.md): `transport.sensitivity.transport_sensitivity`, `transport.revalidation` (`TransportScene`, `revalidate`, `TransportClaimState`).
- [Provider-sourced transport](guides/transported-decisions-and-measurement.md): `transport_handoff.hand_to_callback` and `consume_for_callback`.
- [Finite-SCM discretization](cross-world-execution.md#discretize-a-gaussian-composition-into-a-finite-model): `discretized_composition.discretize`, `PreparedMechanismComposition.discretize`, `consume_discretized_composition`.
- [Composed uncertainty](guides/composed-uncertainty.md): `CrossWorldExecution.whole_path_uncertainty`.
- [What the transport and uncertainty routes do not do](user-facing-limits.md).
- [Refusals](reference/refusal-codes.md#structured-refusals-and-their-remedies): the shared `StructuredRefusal` base with `.code`, `.detail`, `.offending` and `.remedy`.
- [Composition and bundles](composition.md): `composition_bundle.Bundle`, `add_artifact` and `consume_bundle`.

## Every public module

Each module below is importable as `antecedent.<name>` without a separate import. The last column is the page that covers it by task.

| Module | What it does | Guide |
| --- | --- | --- |
| `accepted_graph` | Versioned accepted-graph session handle for artifact-first interactive UX | [selective-discovery.md](selective-discovery.md) |
| `adaptive_inference` | Inference from adaptive experiments | [adaptive-inference.md](adaptive-inference.md) |
| `artifacts` | Durable format-0.5 artifacts for causal queries and results | [artifacts.md](artifacts.md) |
| `attribution` | Attribution helpers | [guides/intrinsic-influence.md](guides/intrinsic-influence.md) |
| `categorical_treatment` | Categorical treatments: ordered and unordered regimes with a declared reference | [guides/treatment-levels.md](guides/treatment-levels.md) |
| `cohort_decision` | A retained fitted effect model decides for a declared cohort | [decision-reliability.md](decision-reliability.md) |
| `cohort_pushdown` | Cohort-checked column projection and filter pushdown over a table or a table file | [guides/large-tables.md](guides/large-tables.md) |
| `compact_export` | Compact runtime export of a fitted linear-in-coefficients effect model  | [guides/build/artifacts-and-handoff.md](guides/build/artifacts-and-handoff.md) |
| `composition_bundle` | Portable composed result: build, export and independently consume a bundle | [composition.md](composition.md) |
| `composition_verification` | End-to-end composition verification: which downstream claims survive a change upstream | [composition-verification.md](composition-verification.md) |
| `composition` | The composition boundary: what a decision input carries and what may be combined | [composition.md](composition.md) |
| `counterfactual_id` | The effect of treatment on the treated on a bounded ADMG | [guides/transport-scope.md](guides/transport-scope.md) |
| `counterfactual` | GCM counterfactual helpers | [guides/gcm-estimands.md](guides/gcm-estimands.md) |
| `cross_world_execution` | General cross-world selective execution over a fully declared finite SCM | [cross-world-execution.md](cross-world-execution.md) |
| `cross_world` | Path-specific edge interventions on a fixed Markovian DAG | [guides/transport-scope.md](guides/transport-scope.md) |
| `cyclic` | Nonlinear cyclic equilibrium SCMs with retained Rust execution and replay | [tutorials/feedback-three-questions.md](tutorials/feedback-three-questions.md) |
| `data` | Owned immutable data constructors and explicit conversion probes | [start/first-analysis.md](start/first-analysis.md) |
| `decision_reliability` | Reliability of sampled and adaptively selected decisions | [decision-reliability.md](decision-reliability.md) |
| `decision_robust` | Admissibility rules and robustness for decisions under structural uncertainty | [sensitivity-and-robustness.md](sensitivity-and-robustness.md) |
| `decision` | Declare a decision problem and evaluate it on aligned joint draws | [decisions.md](decisions.md) |
| `derived` | Derived treatments and factorized joint cells (point only) | [guides/joint-cells.md](guides/joint-cells.md) |
| `descriptive` | Descriptive raw-versus-adjusted comparison and declared reporting-scale transforms | [guides/descriptive-comparison.md](guides/descriptive-comparison.md) |
| `design_models` | Declared likelihoods and priors of the randomized-design Bayesian estimators | [guides/designs/randomized-experiments.md](guides/designs/randomized-experiments.md) |
| `design` | Rank studies, plan evidence and consume plans (a subpackage) | [design-ranking.md](design-ranking.md) |
| `discovery` | Explicit discovery, algorithm configurations, and structural review helpers | [selective-discovery.md](selective-discovery.md) |
| `discretized_composition` | Discretize a linear-Gaussian composition into an executable finite model | [cross-world-execution.md](cross-world-execution.md) |
| `dose_grid` | Dose-grid functional row: a named functional of a randomized-dose response curve | [decisions.md](decisions.md) |
| `drift` | Staleness and drift diagnostics: is a stored plan or claim still valid for new data? | [guides/staleness.md](guides/staleness.md) |
| `effect_constancy_review` | Downstream advisory consumers of independently verified effect-constancy evidence | [population-time-uncertainty.md](population-time-uncertainty.md) |
| `errors` | The exception surface: `CausalError` and its concrete subclasses | [reference/refusal-codes.md](reference/refusal-codes.md) |
| `estimation` | High-level estimation entry points | [python-options.md](python-options.md) |
| `estimators` | Typed front-end over the `estimator_config=` dict kwarg | [python-options.md](python-options.md) |
| `evidence_ledger` | The evidence ledger: which evidence contributed to a result, how it overlaps, and what changed | [evidence-ledger.md](evidence-ledger.md) |
| `execution_attempt` | Diagnostic work observations around one original synchronous native execution | [recalculation-capabilities.md](recalculation-capabilities.md) |
| `execution` | Reusable execution policies with shared native admission accounting | [python-options.md](python-options.md) |
| `experiment` | First-class randomized experiment designs and intention-to-treat queries | [guides/designs/randomized-experiments.md](guides/designs/randomized-experiments.md) |
| `extensibility` | Typing Protocols for slow-path Python callbacks | [guides/build/scientific-interfaces.md](guides/build/scientific-interfaces.md) |
| `external` | Bind a foreign scientific result to an identified causal contract | [guides/build/scientific-interfaces.md](guides/build/scientific-interfaces.md) |
| `external_mechanisms` | Registered external cyclic mechanisms: portable fit, artifact and trusted-resolver replay | [external-cyclic-mechanisms.md](external-cyclic-mechanisms.md) |
| `factorial` | Native-backed utilities for two-factor randomized 2×2 experiments | [guides/designs/randomized-experiments.md](guides/designs/randomized-experiments.md) |
| `fallback` | The opt-in descriptive fallback: what can be said, and what evidence would help, when the | [guides/fallback.md](guides/fallback.md) |
| `finite_discrete` | The adjustment contrast of a finite-discrete frequency table with its posterior draws | [guides/build/scientific-interfaces.md](guides/build/scientific-interfaces.md) |
| `functional_source` | Original finite-law functional requests and independently replayed derivation | [composition.md](composition.md) |
| `gcm` | Discover-then-fit GCM composition helpers | [guides/gcm-estimands.md](guides/gcm-estimands.md) |
| `graph_falsification` | Graph permutation falsification: a DAG against data, with no fitted mechanisms | [guides/graph-falsification.md](guides/graph-falsification.md) |
| `graph` | Graph types and interchange helpers | [start/first-analysis.md](start/first-analysis.md) |
| `handoff` | Handoffs to external estimators that consume an adjustment set | [guides/build/interoperability.md](guides/build/interoperability.md) |
| `horizon_composition` | Composition across the horizons of one temporal claim | [causal-responses.md](causal-responses.md) |
| `host` | Host-integration contract for software built on Antecedent | [guides/build/embedding.md](guides/build/embedding.md) |
| `identify` | Data-free identification followed by immutable preparation and execution | [start/first-analysis.md](start/first-analysis.md) |
| `ids` | Typed identifier / estimator / latency / refute wire ids (Pythonic enums) | [python-options.md](python-options.md) |
| `inference` | Inference-mode configuration for `antecedent.analyze` | [python-options.md](python-options.md) |
| `interface_contract` | Scientific interface contracts | [guides/build/scientific-interfaces.md](guides/build/scientific-interfaces.md) |
| `interference` | Randomization designs and exposure mappings for interference queries | [guides/designs/interference-and-network-exposure.md](guides/designs/interference-and-network-exposure.md) |
| `intervention_callbacks` | Batch callbacks for intervention programs | [guides/intervention-programs.md](guides/intervention-programs.md) |
| `intervention_programs` | Closed-form intervention programs for fitted graphical causal models | [guides/intervention-programs.md](guides/intervention-programs.md) |
| `intervention` | Typed intervention specifications for :class:`InterventionResponse` | [causal-responses.md](causal-responses.md) |
| `intrinsic_influence` | Intrinsic causal influence: how much of a target's variance each noise term explains | [guides/intrinsic-influence.md](guides/intrinsic-influence.md) |
| `inverse_query` | Generalized finite inverse decision query | [guides/inverse-outcome.md](guides/inverse-outcome.md) |
| `inverse_sources` | Forward sources of an inverse query, and the associational observational route | [guides/inverse-outcome.md](guides/inverse-outcome.md) |
| `inverse` | Finite-action inverse-outcome query: which enumerated interventions reach a target mean? | [guides/inverse-outcome.md](guides/inverse-outcome.md) |
| `joint_distribution` | Typed, bounded aligned distribution artifacts | [joint-distributions.md](joint-distributions.md) |
| `joint_law_propagation` | Composition-aware uncertainty: propagate uncertain inputs under the dependence they need | [joint-law-propagation.md](joint-law-propagation.md) |
| `joint_panel` | Retained multichannel panel execution through the Rust causal compiler | [guides/joint-intervention-responses.md](guides/joint-intervention-responses.md) |
| `latent_class` | Latent-class (finite mixture) regime effects | [guides/mediation-and-latent-classes.md](guides/mediation-and-latent-classes.md) |
| `learners` | Typed prediction models shared by causal estimators and transport providers | [python-options.md](python-options.md) |
| `matched` | Conditional odds ratio from matched case-control sets (point only) | [guides/matched-case-control.md](guides/matched-case-control.md) |
| `measurement_models` | Measurement models: corrected functionals of a latent construct measured with error | [observation-queries.md](observation-queries.md) |
| `mechanism_composition` | Composing independently fitted mechanisms into one executable joint model | [cross-world-execution.md](cross-world-execution.md) |
| `mechanism_discrepancy` | Source-target mechanism discrepancy diagnostic | [sensitivity-and-robustness.md](sensitivity-and-robustness.md) |
| `missing` | Declared handling of missing values | [guides/missing-data.md](guides/missing-data.md) |
| `mixed_decision` | One decision from a cluster trial, a temporal period effect, a cost model and an uncertain bridge | [guides/mixed-decisions.md](guides/mixed-decisions.md) |
| `model` | Interventional sampling helpers, plus one associational convenience predictor | [observation-queries.md](observation-queries.md) |
| `mnar_sensitivity` | Missing-not-at-random departure grid: effect as a function of a bounded parameter, tipping points, decision changes | [sensitivity-and-robustness.md](sensitivity-and-robustness.md) |
| `msm_sensitivity` | Marginal sensitivity model (MSM) bounds, tipping point and sensitivity artifact | [sensitivity-and-robustness.md](sensitivity-and-robustness.md) |
| `multi_level_treatment` | Observational contrasts between levels of a multi-level treatment | [guides/treatment-levels.md](guides/treatment-levels.md) |
| `nonlinear_mediation` | Nonlinear continuous-mediator mediation: natural direct, indirect and total effects | [guides/mediation-and-latent-classes.md](guides/mediation-and-latent-classes.md) |
| `nuisance_scores` | Nuisance scores of a prepared study, retained as artifacts that are refit on load | [guides/dml-scores.md](guides/dml-scores.md) |
| `observation_query` | Observation-conditioned scientific queries | [reference/refusal-codes.md](reference/refusal-codes.md) |
| `observation` | Observation-mechanism specifications for latent scientific outcomes | [observation-queries.md](observation-queries.md) |
| `observational_policy` | Observational policy value under a declared treatment support | [guides/heterogeneous-effects-and-policies.md](guides/heterogeneous-effects-and-policies.md) |
| `observational_survival` | Observational survival effects at a declared horizon | [guides/designs/survival-outcomes.md](guides/designs/survival-outcomes.md) |
| `outcomes` | One retained outcome for scientific answers and scientific refusals | [start/reading-a-result.md](start/reading-a-result.md) |
| `policy` | Binary treatment policies and held-out randomized evaluation | [guides/heterogeneous-effects-and-policies.md](guides/heterogeneous-effects-and-policies.md) |
| `population` | Named predicates and custom target-distribution weights for analyze() | [python-options.md](python-options.md) |
| `prediction` | Portable fitted CATE prediction, bound to an independently verified parent claim | [decisions.md](decisions.md) |
| `predictive_assessment` | Assessment of predictive laws: proper scores, calibration diagnostics and held-out checks | [predictive-assessment.md](predictive-assessment.md) |
| `preflight` | Preflight diagnostics, rank-drop plans and cost counts | [guides/preflight-diagnostics.md](guides/preflight-diagnostics.md) |
| `prepared` | Immutable prepared studies and the common exploration lifecycle | [guides/plan-view.md](guides/plan-view.md) |
| `priors` | External prior bank: catalog metadata and compatibility filtering | [priors.md](priors.md) |
| `program_claims` | Bind claims to the identified program and request they answer | [composition.md](composition.md) |
| `propagation_chain` | Uncertainty carried through a chain of transformations | [scientific-bridges.md](scientific-bridges.md) |
| `proposal_arrival` | Actual finite transport proposal arrivals; structural checks stay in proposals | [design-ranking.md](design-ranking.md) |
| `proposals` | Per-proposal receipts linking identification repair and design ranking | [design-ranking.md](design-ranking.md) |
| `provider_callbacks` | Arbitrary law, posterior and nested callbacks | [provider-callbacks.md](provider-callbacks.md) |
| `quasi` | Point estimation utilities for explicitly scoped quasi-experimental designs | [guides/designs/panel-designs.md](guides/designs/panel-designs.md) |
| `query` | Typed causal queries for the Python facade | [causal-responses.md](causal-responses.md) |
| `recalc_adjusted` | Selective adjusted regression using the shared native planner and receipt | [recalculation-capabilities.md](recalculation-capabilities.md) |
| `recalc_bayesian` | Native-issued ordinary Gaussian posterior recalculation, with measured work | [recalculation-capabilities.md](recalculation-capabilities.md) |
| `recalc_capabilities` | Operation-level inventory of prepared and attested callback recalculation families | [recalculation-capabilities.md](recalculation-capabilities.md) |
| `recalc_cell` | Selective recalculation on the cell-AIPW route, with portable score resume | [recalculation-capabilities.md](recalculation-capabilities.md) |
| `recalc_composite` | One shared workflow for a retained native response and two attested mean callbacks | [recalculation-capabilities.md](recalculation-capabilities.md) |
| `recalc_design` | Shared selective execution of checked IV, sharp RD and linear front-door designs | [recalculation-capabilities.md](recalculation-capabilities.md) |
| `recalc_dr` | Selective DML AIPW and DR-Learner execution using the shared native stage planner | [recalculation-capabilities.md](recalculation-capabilities.md) |
| `recalc_external` | Executing bounded foreign mean callbacks under exact declared dependencies | [recalculation-capabilities.md](recalculation-capabilities.md) |
| `recalc_static` | Selective finite ADMG response and multi-source exact-law transport | [recalculation-capabilities.md](recalculation-capabilities.md) |
| `recalc_temporal` | Checked empirical binary two-step histories; point-only temporal recalculation | [recalculation-capabilities.md](recalculation-capabilities.md) |
| `recalc` | Selective recalculation with a visible receipt | [recalculation-capabilities.md](recalculation-capabilities.md) |
| `recipe_glm` | A logistic g-computation run through admitted recipes | [guides/build/scientific-interfaces.md](guides/build/scientific-interfaces.md) |
| `recovery_chain` | Ordered-response observation recovery: a response chain `R -> R` | [guides/missing-data.md](guides/missing-data.md) |
| `regimes` | Point evaluation of prespecified longitudinal treatment regimes | [guides/designs/sequential-regimes.md](guides/designs/sequential-regimes.md) |
| `release_migration` | Consume artifacts written by a previous release | [migration.md](migration.md) |
| `repair` | Evidence obligations, durable study candidates and identification repair | [design-ranking.md](design-ranking.md) |
| `results` | The result and analysis types returned by `analyze` and `estimate` (a subpackage) | [start/reading-a-result.md](start/reading-a-result.md) |
| `scenario_decision` | Decide over the claims of a real scenario set or CPDAG-completion result | [sensitivity-and-robustness.md](sensitivity-and-robustness.md) |
| `scenario_invariance` | Selection differences and invariances behind each transport-scenario answer | [transport-counterfactuals.md](transport-counterfactuals.md) |
| `scientific_bridge` | Typed bridges between scientific quantities | [scientific-bridges.md](scientific-bridges.md) |
| `selective_discovery` | Selective discovery: reuse conditional-independence statistics across discovery requests | [selective-discovery.md](selective-discovery.md) |
| `sensitivity_decision` | Consume an assumption-sensitivity result in a durable decision | [sensitivity-and-robustness.md](sensitivity-and-robustness.md) |
| `sequential_evaluation` | Sequential evaluation of a caller-supplied contingent policy | [sequential-evaluation.md](sequential-evaluation.md) |
| `source_evidence` | Original scoped diagnostics and independently consumed source lineage | [composition.md](composition.md) |
| `source_projection` | Portable original-source projections; numerical replay supplies no native authority | [composition.md](composition.md) |
| `state` | Incremental causal state helpers | [guides/retained-refresh.md](guides/retained-refresh.md) |
| `survival` | Survival and competing-risk outcomes for randomized two-arm studies | [guides/designs/survival-outcomes.md](guides/designs/survival-outcomes.md) |
| `temporal_counterfactual` | Fixed-population temporal counterfactual and the closed transported route | [population-time-uncertainty.md](population-time-uncertainty.md) |
| `temporal_designs` | Sequentially randomized unit-period designs and regime survival | [guides/temporal-designs.md](guides/temporal-designs.md) |
| `temporal_model` | Standalone queries on the fitted dynamics of a `TemporalDag` | [guides/temporal-designs.md](guides/temporal-designs.md) |
| `temporal` | Effect constancy across a declared time or region partition | [population-time-uncertainty.md](population-time-uncertainty.md) |
| `tier_diagnostics` | Tier-aware overlap and E-value diagnostics of an executed tiered result | [guides/tier-diagnostics.md](guides/tier-diagnostics.md) |
| `transport_handoff` | Hand a transported claim to a callback input, and consume the record of the connection | [guides/transported-decisions-and-measurement.md](guides/transported-decisions-and-measurement.md) |
| `transport` | Transport identification and evaluation routes (a subpackage) | [guides/transport-scope.md](guides/transport-scope.md) |
| `transported_constructors` | Typed constructors of a transported response from the library transport routes | [guides/transported-decisions-and-measurement.md](guides/transported-decisions-and-measurement.md) |
| `transported_counterfactual` | The transported static path-specific counterfactual, narrow class | [transport-counterfactuals.md](transport-counterfactuals.md) |
| `transported_response` | Transported responses: a value carried to a target population, and the decision made on one | [guides/transported-decisions-and-measurement.md](guides/transported-decisions-and-measurement.md) |
| `validation` | Validation: discovery stability, target-native refutation and sensitivity ranges | [guides/target-native-validation.md](guides/target-native-validation.md) |
| `vector_treatment` | Joint vector-treatment coefficients | [guides/treatment-levels.md](guides/treatment-levels.md) |

Every stage module is also reachable lazily from the root, for example `antecedent.design`, without a separate import.

For Rust, use `cargo doc -p antecedent --open` for this checkout, or visit the [published crate reference](https://docs.rs/antecedent/3.0.0/antecedent/).
