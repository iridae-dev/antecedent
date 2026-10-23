# Examples

## Python environment

Use Python 3.11 or later. Released-package setup:

<!-- generated:release:python-pin:begin -->
```bash
python -m pip install "antecedent>=2.4.1,<2.5" pandas matplotlib ipykernel jupyterlab
```
<!-- generated:release:python-pin:end -->

For a source build, follow [installation](../docs/installation.md#build-from-source). Select the notebook kernel for your installed environment and restart after upgrading an imported native extension.

```bash
python examples/python/analysis_workflow.py
python examples/python/notebook_to_system.py
jupyter lab examples/notebooks
cargo run -p antecedent --example ate_quickstart
```

Follow [the notebook-to-system tutorial](../docs/guides/notebook-to-system.md), [bring your own data](../docs/guides/your-data.md), or [measurement to decision](../docs/tutorials/measurement-to-decision.md). The published site includes executed notebook previews; the source notebooks remain free of saved outputs.

<!-- generated:example-catalog:begin -->

All datasets are simulated. Runtime estimates assume a warm installed environment and exclude installation and Rust's first compilation. First native imports and notebook-kernel setup can add substantial time; machine, workload and optional dependencies affect elapsed time. Assertions check these examples, not your causal assumptions.

## Start and reuse

| Example and question | Level / estimated runtime | Dependencies beyond Antecedent | Run / open |
| --- | --- | --- | --- |
| **First analysis** — Estimate, inspect, refresh and reload one synthetic average effect. <details><summary>Claim scope</summary>Average-effect execution, immutable refresh history, and saved-result acceptance; inspect interval calibration.</details> | beginner / up to 1 min | none | [python](python/analysis_workflow.py) |
| **Artifact consumer** — Check independent consumption, wrong identity and altered bytes. <details><summary>Claim scope</summary>Identity-bound consumption, wrong-identity and malformed-container refusals; loading does not recreate a live study.</details> | beginner / up to 1 min | none | [python](python/artifact_consumer.py) |
| **Reference host** — Compile, read the status, tell a supported claim from a refusal and load an artifact, with no agent logic. <details><summary>Claim scope</summary>Status, coded refusal and identity-bound artifact acceptance read through the host-integration contract; the host plans nothing and certifies no assumption.</details> | intermediate / up to 1 min | none | [python](python/reference_host.py) · [rust](rust/reference_host.rs) |
| **Artifact producer** — Store bytes and a separately retainable producer expectation. <details><summary>Claim scope</summary>Separately retainable scientific identity and bytes; saving does not establish real-world causal validity.</details> | beginner / up to 1 min | none | [python](python/artifact_producer.py) |
| **CausalState workflow** — Update an analysis as data arrive and identify outdated results <details><summary>Claim scope</summary>State updates and explicit stale-result handling as observations arrive.</details> | beginner / up to 1 min | none | [python](python/causal_state_workflow.py) · [rust](rust/causal_state_workflow.rs) |
| **Notebook to system** — Retain identification, refute an execution, guard design updates, compose a mean and consume bytes in a fresh process. <details><summary>Claim scope</summary>Retained average effect, randomized ITT, declared mean composition, and independent identity-bound consumption.</details> | beginner / up to 5 min | pandas, matplotlib, ipykernel | [python](python/notebook_to_system.py) · [notebook](notebooks/notebook_to_system.ipynb) · [Colab](https://colab.research.google.com/github/iridae-dev/antecedent/blob/v2.4.1/examples/notebooks/notebook_to_system.ipynb) |
| **From a Parquet file** — Analyze a Parquet file read in bounded chunks, then stream a summary through a pool smaller than the table. <details><summary>Claim scope</summary>The average effect from a Parquet file equals the one from the same table in memory; only the needed columns are decoded, every read is charged to the admission pool, and the content identity tracks the values. Calibration above 100k rows is not assessed.</details> | beginner / up to 1 min | pyarrow | [python](python/parquet_e2e.py) |
| **Refusal and repair** — Refuse changed design assignments and an unlicensed interval, then use valid neighboring routes. <details><summary>Claim scope</summary>Changed randomized-design bindings and an unlicensed composed interval; explicit repreparation preserves the preceding result.</details> | beginner / up to 1 min | none | [python](python/refusal_recipes.py) |
| **Bring your own table** — Read a temporary CSV, check numeric columns, prepare and inspect an average effect. <details><summary>Claim scope</summary>Reviewed numeric table and declared DAG average effect; missing values require an explicit observation policy.</details> | beginner / up to 1 min | pandas | [python](python/your_data.py) |

## Bayesian evidence

| Example and question | Level / estimated runtime | Dependencies beyond Antecedent | Run / open |
| --- | --- | --- | --- |
| **Prior bank surveys** — Select and combine evidence from earlier surveys <details><summary>Claim scope</summary>Declared historical-evidence selection and pooling with source and target compatibility checks.</details> | intermediate / up to 5 min | none | [python](python/prior_bank_surveys.py) · [rust](rust/prior_bank_surveys.rs) |
| **Sequential Bayes** — Use one batch’s results as prior evidence for the next <details><summary>Claim scope</summary>Posterior evidence transferred between compatible analyses under their declared models and evidence policy.</details> | intermediate / up to 5 min | none | [python](python/sequential_bayes.py) · [rust](rust/sequential_bayes.rs) |
| **Can last quarter's campaign inform this quarter's sales analysis?** — Transfer historical campaign evidence and update it with current observations. <details><summary>Claim scope</summary>Historical treatment-effect posterior transfer with target compatibility and new-data updating.</details> | intermediate / up to 5 min | pandas, matplotlib, ipykernel | [notebook](notebooks/sales_campaign_prior_transfer.ipynb) · [Colab](https://colab.research.google.com/github/iridae-dev/antecedent/blob/v2.4.1/examples/notebooks/sales_campaign_prior_transfer.ipynb) |

## Cyclic systems

| Example and question | Level / estimated runtime | Dependencies beyond Antecedent | Run / open |
| --- | --- | --- | --- |
| **Independent cyclic paper protocols** — Reproduce applicable nonlinear and stochastic feedback synthetic experiments with held-out joint interventions and fresh CPU artifact replay. <details><summary>Claim scope</summary>Specified synthetic likelihood experiments with explicit deviations, optimizer diagnostics and independent numerical references; no calibration or latent recovery claim</details> | advanced / paper-size runs vary with feature and sample counts | numpy, scipy | [python](python/cyclic_paper_experiments.py) |

## Decisions and composition

| Example and question | Level / estimated runtime | Dependencies beyond Antecedent | Run / open |
| --- | --- | --- | --- |
| **Declared adaptive procedure** — Enumerate a tiny procedure under supplied truths, keeping undefined contrast mass visible. <details><summary>Claim scope</summary>Derived procedure report conditional on declared design and truths</details> | intermediate / up to 1 min | none | [python](python/adaptive_procedure.py) |
| **Composed product spread** — Compute a product from one aligned law and distinguish its spread from interval coverage. <details><summary>Claim scope</summary>Point and uncalibrated spread under one aligned joint law</details> | intermediate / up to 1 min | none | [python](python/composed_uncertainty.py) |
| **Composition bundle** — Export a composed decision and consume it under a retained identity <details><summary>Claim scope</summary>Portable composition with independently retained identities and checked downstream consumption.</details> | intermediate / up to 1 min | none | [python](python/composition_bundle.py) |
| **Decision lifecycle** — Analyze, bind external evidence, inspect, decide, then rank the next study <details><summary>Claim scope</summary>Checked decision contracts consume native or bound external claims while retaining their different standing.</details> | intermediate / up to 1 min | none | [python](python/decision_lifecycle.py) |
| **Measurement to decision** — Correct exposure prevalence, pool disjoint evidence and revise a finite-model decision. <details><summary>Claim scope</summary>Declared misclassification correction, disjoint fixed-effect pooling, finite mechanisms and affine point decisions.</details> | intermediate / up to 1 min | none | [python](python/measurement_to_decision.py) |
| **Rank designs** — Rank candidate experiments by identification probability <details><summary>Claim scope</summary>Identification-oriented ranking of declared studies; inspect objective, budgets and Monte Carlo readings.</details> | intermediate / up to 5 min | none | [python](python/rank_designs.py) · [rust](rust/rank_designs.rs) |
| **Expected value of a study** — Rank candidate studies by expected value of sample information <details><summary>Claim scope</summary>Expected-value study ranking under the supplied planning model and utility declarations.</details> | intermediate / up to 5 min | none | [python](python/rank_designs_evsi.py) |
| **Recalculation with frozen scores** — Recompute a decision with zero refits; resume from exported scores <details><summary>Claim scope</summary>Retargeting retained scores and portable resume under the original supported data and score identities.</details> | intermediate / up to 1 min | none | [python](python/recalc_cell_resume.py) |
| **Repair and evidence obligations** — State the evidence a contract owes and which study would repair it <details><summary>Claim scope</summary>Evidence obligations and declared study repairs; a repair plan is not new observed evidence.</details> | intermediate / up to 1 min | none | [python](python/repair_obligations.py) |
| **Which marketing experiment should we run next?** — Rank investments that resolve a marketing identification problem. <details><summary>Claim scope</summary>Budgeted study ranking under the declared marketing causal question and study costs.</details> | intermediate / up to 5 min | pandas, matplotlib, ipykernel | [notebook](notebooks/marketing_experiment_design.ipynb) · [Colab](https://colab.research.google.com/github/iridae-dev/antecedent/blob/v2.4.1/examples/notebooks/marketing_experiment_design.ipynb) |

## Discovery and identification

| Example and question | Level / estimated runtime | Dependencies beyond Antecedent | Run / open |
| --- | --- | --- | --- |
| **Uncertain graph directions** — Estimate a range of effects when some edge directions are unknown <details><summary>Claim scope</summary>Class-preserving bounds or partial answer; a reviewed orientation changes the structural declaration.</details> | intermediate / up to 1 min | none | [python](python/class_preserving_cpdag.py) · [rust](rust/class_preserving_cpdag.rs) |
| **Discover then estimate** — Discover once, accept a DAG, re-estimate many times <details><summary>Claim scope</summary>Estimation conditional on an explicitly accepted discovered DAG; acceptance does not make discovery certain.</details> | intermediate / up to 5 min | none | [python](python/discover_then_estimate.py) · [rust](rust/discover_then_estimate.rs) |
| **Identify only** — Identification without fitting <details><summary>Claim scope</summary>Identification and derivation without numerical fitting.</details> | intermediate / up to 1 min | none | [rust](rust/identify_only.rs) |
| **Should we increase paid-search spend?** — Compare naive marketing attribution with effects under uncertain graph directions. <details><summary>Claim scope</summary>Attribution conditional on graph completions versus a reviewed graph; preserve unidentified structural mass.</details> | intermediate / up to 5 min | pandas, matplotlib, ipykernel | [notebook](notebooks/marketing_channel_structural_uncertainty.ipynb) · [Colab](https://colab.research.google.com/github/iridae-dev/antecedent/blob/v2.4.1/examples/notebooks/marketing_channel_structural_uncertainty.ipynb) |

## Estimation and designs

| Example and question | Level / estimated runtime | Dependencies beyond Antecedent | Run / open |
| --- | --- | --- | --- |
| **Randomized experiment** — Estimate a randomized ITT effect; design inputs named by data column <details><summary>Claim scope</summary>Randomized ITT under declared assignment probabilities and units; inspect graphless support and interval standing.</details> | beginner / up to 1 min | none | [python](python/experiment_randomized_effect.py) |
| **Factorial cell means** — 2x2 Bernoulli factorial main effects and interaction (point utility) <details><summary>Claim scope</summary>Declared Bernoulli factorial cell means, main effects and interaction.</details> | intermediate / up to 1 min | none | [python](python/factorial_cell_means.py) |
| **IHDP-style propensity / AIPW** — Synthetic IHDP-like covariates → assumed DAG → IPW + AIPW + refute <details><summary>Claim scope</summary>Synthetic IHDP-like covariates under an assumed DAG; IPW and AIPW effects and refutation.</details> | intermediate / up to 1 min | none | [python](python/ihdp_propensity_e2e.py) |
| **MSM sensitivity** — Marginal-sensitivity bounds and tipping point for a stratified ATE <details><summary>Claim scope</summary>Assumption-sensitivity range and tipping point for a stratified average effect; not a confidence interval.</details> | intermediate / up to 1 min | none | [python](python/msm_sensitivity.py) |
| **Policy value** — Held-out value of a fixed binary policy under known propensities <details><summary>Claim scope</summary>Held-out value of a fixed binary policy under known assignment propensities.</details> | intermediate / up to 1 min | none | [python](python/policy_value_evaluation.py) |
| **Propensity weighting** — Adjust for treatment selection using inverse-probability weights <details><summary>Claim scope</summary>Average effect via propensity weighting under a reviewed DAG; inspect overlap and interval standing.</details> | beginner / up to 1 min | none | [python](python/propensity_weighting.py) · [rust](rust/propensity_weighting.rs) |
| **Difference-in-differences** — 2x2 and balanced-panel DiD point utilities <details><summary>Claim scope</summary>Two-period and balanced-panel difference-in-differences under their declared design assumptions.</details> | intermediate / up to 1 min | none | [python](python/quasi_difference_in_differences.py) |
| **Sales analysis** — Explore average, mediated, individual, and delayed effects <details><summary>Claim scope</summary>Simulated average, mediated, individual and temporal effects, each under its own declared question.</details> | intermediate / up to 5 min | none | [python](python/sales_spreadsheet_e2e.py) · [rust](rust/sales_spreadsheet_e2e.rs) |
| **Mediation and individual effects** — Separate direct and mediated effects; estimate individual effects <details><summary>Claim scope</summary>Average, mediated and individual effects under the declared static model and supported query-specific routes.</details> | intermediate / up to 1 min | none | [python](python/staged_static_kinds.py) · [rust](rust/staged_static_kinds.rs) |
| **Survival RMST** — Randomized survival curve and RMST by arm <details><summary>Claim scope</summary>Randomized survival and restricted mean survival time under declared observation and assignment assumptions.</details> | intermediate / up to 1 min | none | [python](python/survival_rmst.py) |
| **ATE quickstart** — Build and run an average-effect analysis <details><summary>Claim scope</summary>Parametric linear average-effect point with bootstrap and refutation disabled.</details> | beginner / up to 1 min | none | [rust](rust/ate_quickstart.rs) |
| **GCM do** — Fit a GCM and sample under `do(·)` <details><summary>Claim scope</summary>Interventional samples under fitted, declared causal mechanisms.</details> | intermediate / up to 1 min | none | [rust](rust/gcm_do.rs) |
| **How does the outcome change with dose?** — Inspect supported doses, response values and derivatives. <details><summary>Claim scope</summary>Supported response and derivative coordinates under an explicit dose-response model.</details> | intermediate / up to 5 min | pandas, matplotlib, ipykernel | [notebook](notebooks/continuous_causal_response.ipynb) · [Colab](https://colab.research.google.com/github/iridae-dev/antecedent/blob/v2.4.1/examples/notebooks/continuous_causal_response.ipynb) |
| **Pricing, availability, and latent demand** — Distinguish inventory-limited sales from latent demand. <details><summary>Claim scope</summary>Demand response under an explicit censoring premise; unsupported derivative requests are refused.</details> | intermediate / up to 5 min | pandas, matplotlib, ipykernel | [notebook](notebooks/pricing_availability_latent_demand.ipynb) · [Colab](https://colab.research.google.com/github/iridae-dev/antecedent/blob/v2.4.1/examples/notebooks/pricing_availability_latent_demand.ipynb) |
| **Shared hidden cause counterfactual** — Compute a counterfactual through an exogenous term shared by two mechanisms <details><summary>Claim scope</summary>Exact enumeration of a fully declared finite SCM whose mechanisms share an exogenous term; undeclared latent confounding is refused as not identified.</details> | intermediate / up to 1 min | none | [python](python/shared_exogenous_counterfactual.py) · [rust](rust/shared_exogenous_counterfactual.rs) |

## Graphical causal models

| Example and question | Level / estimated runtime | Dependencies beyond Antecedent | Run / open |
| --- | --- | --- | --- |
| **Graph falsification** — Compare a DAG with random relabellings of itself through the independencies it asserts <details><summary>Claim scope</summary>Permutation comparison of declared graph independencies under the named conditional-independence test; not rejected does not show the graph correct, and nothing is fitted.</details> | intermediate / up to 1 min | none | [python](python/graph_falsification.py) · [rust](rust/graph_falsification.rs) |
| **Intervention programs** — Apply a price floor to a fitted model, replay it on factual units, save it and test its callback form <details><summary>Claim scope</summary>Typed value programs and batch callbacks executed on declared mechanisms; routes are licensed separately and a runtime callback is not portable.</details> | intermediate / up to 1 min | none | [python](python/intervention_programs.py) · [rust](rust/intervention_programs.rs) |
| **Intrinsic causal influence** — Split a target's variance among the noise terms of its ancestors <details><summary>Claim scope</summary>Variance shares of independent, declared noise terms under fitted mechanisms; each interval names the one uncertainty component it covers and none is calibrated.</details> | intermediate / up to 1 min | none | [python](python/intrinsic_influence.py) · [rust](rust/intrinsic_influence.rs) |
| **Mechanism selection** — Score linear, spline and tree families on shared held-out folds and keep the audit trail <details><summary>Claim scope</summary>Held-out comparison of declared candidate mechanisms with constraints and split schemes; the choice is a model-fit decision, not a causal one.</details> | intermediate / up to 1 min | none | [python](python/mechanism_selection.py) · [rust](rust/mechanism_selection.rs) |
| **Noise replay** — Keep a model's exogenous noise, fix or redraw coordinates and evaluate again <details><summary>Claim scope</summary>Replay of a compiled model on retained noise with fixed, redrawn and coalition coordinates; closed-form variances on a linear Gaussian chain.</details> | intermediate / up to 1 min | none | [python](python/noise_replay.py) · [rust](rust/noise_replay.rs) |

## Interoperability

| Example and question | Level / estimated runtime | Dependencies beyond Antecedent | Run / open |
| --- | --- | --- | --- |
| **DoWhy handoff** — Exchange a reviewed backdoor graph with DoWhy (optional dependency) <details><summary>Claim scope</summary>Reviewed backdoor graph exchange; each library owns its own fit and scientific interpretation.</details> | intermediate / up to 1 min | pandas, dowhy (optional) | [python](python/dowhy_handoff.py) |
| **EconML handoff** — Compare adjusted average effects with EconML CATE predictions (optional dependency) <details><summary>Claim scope</summary>Antecedent supplies adjustment columns; the optional EconML fit owns its conditional-effect output.</details> | intermediate / up to 1 min | econml (optional) | [python](python/econml_cate_handoff.py) |

## Mechanisms and counterfactuals

| Example and question | Level / estimated runtime | Dependencies beyond Antecedent | Run / open |
| --- | --- | --- | --- |
| **General cyclic systems (3.0 development)** — Fit overlapping nonlinear feedback loops, query a held-out joint intervention and replay a shared-noise counterfactual artifact. <details><summary>Claim scope</summary>Specified-structure/covariance nonlinear equilibrium likelihood and shared-disturbance counterfactuals; calibration unmeasured.</details> | advanced / up to 1 min | none | [python](python/general_cyclic_systems.py) |

## Temporal and longitudinal

| Example and question | Level / estimated runtime | Dependencies beyond Antecedent | Run / open |
| --- | --- | --- | --- |
| **Manufacturing temporal** — Estimate how a pressure change affects later defects <details><summary>Claim scope</summary>Lagged intervention effect under an explicit temporal graph; horizons and support remain part of the claim.</details> | intermediate / up to 1 min | none | [python](python/manufacturing_temporal.py) · [rust](rust/manufacturing_temporal.rs) |
| **PCMCI pulse discovery** — Discover a lagged graph and estimate a pulse effect in simulated data <details><summary>Claim scope</summary>Pulse effect conditional on an accepted lagged graph discovered from the simulated series.</details> | intermediate / up to 5 min | none | [python](python/pcmci_pulse_benchmark.py) |
| **Longitudinal regime value** — Value of a static treatment rule by inverse-probability g-formula <details><summary>Claim scope</summary>Static longitudinal-regime value by inverse-probability g-formula under declared histories and probabilities.</details> | intermediate / up to 1 min | none | [python](python/regime_value.py) |
| **Temporal response curve** — Compare pressure levels and their effects over time <details><summary>Claim scope</summary>Temporal response grid under declared lag structure; inspect each requested coordinate and uncertainty.</details> | intermediate / up to 1 min | none | [python](python/temporal_response_curve.py) · [rust](rust/temporal_response_curve.rs) |

## Transport

| Example and question | Level / estimated runtime | Dependencies beyond Antecedent | Run / open |
| --- | --- | --- | --- |
| **ML and transport lifecycle** — Prepare, estimate, inspect and reload fitted models and transport results <details><summary>Claim scope</summary>Supported learner-backed estimation and transport with retained model and artifact identity.</details> | intermediate / up to 1 min | none | [python](python/cohesive_ml_transport.py) |
| **Mechanism discrepancy** — Test whether a node's mechanism differs between source and target <details><summary>Claim scope</summary>Diagnostic comparison of source and target mechanisms; a diagnostic does not prove invariance.</details> | intermediate / up to 1 min | none | [python](python/mechanism_discrepancy.py) |
| **Provider transport handoff** — Seal a declared transported offer to a callback input and refuse a native-only consumer. <details><summary>Claim scope</summary>Transported external claim; handoff checks do not prove numerical transport</details> | intermediate / up to 1 min | none | [python](python/provider_transport.py) |
| **Scenario invariance** — Why each transport scenario answers as it does: selection and invariances <details><summary>Claim scope</summary>Inspection of explicitly declared transport scenarios and their invariance assumptions.</details> | intermediate / up to 1 min | none | [python](python/scenario_invariance.py) |
| **Transport an exact law** — Single-source exact-law transport of a response grid <details><summary>Claim scope</summary>Transported response from supplied exact finite laws; no sampling interval is inferred.</details> | intermediate / up to 1 min | none | [python](python/transport_exact.py) · [rust](rust/transport_exact.rs) |
| **Complementary-source grid** — Combined sources identify a target curve; one source does not <details><summary>Claim scope</summary>Complementary finite source laws identify a grid; missing laws retain unsupported coordinates.</details> | intermediate / up to 1 min | none | [python](python/transport_meta_grid.py) · [rust](rust/transport_meta_grid.rs) |
| **Transport a source table** — Single-source empirical transport of a response grid <details><summary>Claim scope</summary>Empirical-table transported grid; inspect local support and bootstrap calibration at this coordinate.</details> | intermediate / up to 1 min | none | [python](python/transport_statistical.py) · [rust](rust/transport_statistical.rs) |
| **Transported counterfactual** — Transport direct and indirect effects under a shared additive model; inspect refusals <details><summary>Claim scope</summary>Static transported path-specific contrast under the supported shared affine-additive model.</details> | intermediate / up to 1 min | none | [python](python/transported_counterfactual.py) |
| **Transported measurement** — Correct target prevalence under explicitly invariant source sensitivity and specificity. <details><summary>Claim scope</summary>Conditional point and uncalibrated standard error under declared invariance</details> | intermediate / up to 1 min | none | [python](python/transported_measurement.py) |

## Performance

| Example and question | Level / estimated runtime | Dependencies beyond Antecedent | Run / open |
| --- | --- | --- | --- |
| **Python ingest performance** — Measure Python-to-native ingest overhead with repeatable workload inputs. <details><summary>Claim scope</summary>Numerical timing; no scientific claim</details> | advanced / up to 5 min | pandas | [python](python/bench_python_overhead.py) |

For exact claim settings, inspect each result and check the support matrix. Optional handoff dependencies are needed to exercise their external half; an absent dependency prints a skip.

<!-- generated:example-catalog:end -->
