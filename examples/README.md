# Examples

## Python environment

Use Python 3.11 or later. Released-package setup:

<!-- generated:release:python-pin:begin -->
```bash
python -m pip install "antecedent>=3.0.0,<3.1" pandas matplotlib ipykernel jupyterlab
```
<!-- generated:release:python-pin:end -->

For a source build, follow [installation](../docs/start/installation.md#build-from-source). Select the notebook kernel for your installed environment and restart after upgrading an imported native extension.

## Run an example

```bash
python examples/python/feedback_three_questions.py
cargo run -p antecedent --example feedback_three_questions
jupyter lab examples/notebooks
```

The tutorials are the front door: [one feedback system and three questions](../docs/tutorials/feedback-three-questions.md), [a joint intervention program](../docs/tutorials/joint-interventions.md), [an experiment and the next study](../docs/tutorials/experiment-to-next-study.md), [deciding under uncertainty](../docs/tutorials/decide-under-uncertainty.md), [replay and current decisions](../docs/tutorials/replay-and-current-decisions.md), [an application component](../docs/tutorials/application-component.md) and [an observational study](../docs/tutorials/observational-study.md).

## Datasets

Most examples simulate their data from a declared mechanism so the known answer can be checked. Examples that read a real public dataset describe its origin, licence, version and checksums in [`data/`](data/README.md), and run offline on a labelled synthetic stand-in. The published site includes executed notebook previews; the source notebooks remain free of saved outputs.

<!-- generated:example-catalog:begin -->

Examples run on simulated data with a declared mechanism, so the known truth can be asserted; a simulated truth is a fixture truth of that stated model, not a claim about a real system. One tutorial reads a real public dataset on request and falls back to a synthetic stand-in offline, and its output says which table it used. Runtime estimates assume a warm installed environment and exclude installation and Rust's first compilation. First native imports and notebook-kernel setup can add substantial time; machine, workload and optional dependencies affect elapsed time. Assertions check each example against its stated truth, not your causal assumptions.

## Start here

| Example and question | Level / estimated runtime | Dependencies beyond Antecedent | Run / open |
| --- | --- | --- | --- |
| **Run a first analysis and branch it onto new data** — Estimate, inspect, rebind and reload one synthetic average effect. <details><summary>Claim scope</summary>Average-effect execution, immutable branch history, and saved-result acceptance; inspect interval calibration.</details> | beginner / up to 1 min | none | [python](python/analysis_workflow.py) |
| **Read a refusal and change the question** — Refuse changed design assignments and an unlicensed interval, then use valid neighboring routes. <details><summary>Claim scope</summary>Changed randomized-design bindings and an unlicensed composed interval; explicit repreparation preserves the preceding result.</details> | beginner / up to 1 min | none | [python](python/refusal_recipes.py) |
| **Estimate an average effect in Rust** — Build and run an average-effect analysis <details><summary>Claim scope</summary>Parametric linear average-effect point with bootstrap and refutation disabled.</details> | beginner / up to 1 min | none | [rust](rust/ate_quickstart.rs) |
| **Carry a notebook analysis into an application** — Retain identification, refute an execution, guard design updates, compose a mean and consume bytes in a fresh process. <details><summary>Claim scope</summary>Retained average effect, randomized ITT, declared mean composition, and independent identity-bound consumption.</details> | beginner / up to 5 min | pandas, matplotlib, ipykernel | [python](python/notebook_to_system.py) · [notebook](notebooks/notebook_to_system.ipynb) · [Colab](https://colab.research.google.com/github/iridae-dev/antecedent/blob/v3.0.0/examples/notebooks/notebook_to_system.ipynb) |

## Tutorials

| Example and question | Level / estimated runtime | Dependencies beyond Antecedent | Run / open |
| --- | --- | --- | --- |
| **Compare an observation, a clamp and a shock** — Ask three different questions of one feedback loop, then keep an intervention with no unique solution as failed mass. <details><summary>Claim scope</summary>Exact Gaussian equilibrium answers of the stated model (observation 0.776, clamp 0.5, shock 0.625); no data are simulated, so these are fixture truths of the equations, not claims about a real system. The unsolvable intervention is reported as retained failed mass rather than an answer.</details> | intermediate / up to 1 min | none | [python](python/feedback_three_questions.py) · [rust](rust/feedback_three_questions.rs) |
| **Compare a coordinated program with its baseline** — Set two channels for a 24-week horizon on a simulated panel, read the joint contrast and its shared uncertainty, and ask what the assignment design establishes. <details><summary>Claim scope</summary>Model-conditional contrast of a whole program over its baseline, checked against the declared mechanism (terminal-week effect 4); uncertainty comes from draws every world shares. Per-period randomization does not by itself identify a held-out whole-program effect. Read result.inspect().calibration for interval calibration status.</details> | advanced / up to 1 min | none | [python](python/joint_intervention_program.py) |
| **Update a decision after an experiment, then rank the next studies** — Incorporate a later experiment into a retained response law through an explicit estimand mapping, read the changed decision, and rank further studies by information value net of cost. <details><summary>Claim scope</summary>Frozen importance conditioning of retained draws on a declared experimental mean; the update does not refit the model and does not promote the law to an identified causal result. The experiment mean is declared, not drawn from the simulated world.</details> | advanced / up to 1 min | none | [python](python/experiment_to_next_study.py) |
| **Stop, continue or run longer after a geo experiment** — Analyze a randomized regional advertising experiment, decide on its mean, ask a risk question the mean cannot answer, and value another study against its cost. <details><summary>Claim scope</summary>Randomized-experiment effect, a decision under a stated utility and belief, and expected value of information, all against a simulated truth of 1,000 incremental orders per region. The decision depends on the declared utility and belief, not only on the data.</details> | intermediate / up to 1 min | none | [python](python/decide_under_uncertainty.py) |
| **Withdraw a source and separate replay from current use** — Build one decision from three evidence routes, change its utility without refitting, export it, then withdraw a source. <details><summary>Claim scope</summary>After withdrawal the exported numbers still replay exactly while operations asking for current permission refuse. Scoped to this mixed-evidence workflow; it is not a general revocation mechanism, and the numbers are fixture numbers of the declared model.</details> | advanced / up to 1 min | none | [python](python/replay_and_current_decisions.py) |
| **Use a causal analysis as a component in an application** — A host submits a prepared request, tells a supported result from a typed refusal, applies its own acceptance policy, branches onto new data and consumes an artifact. <details><summary>Claim scope</summary>Randomized reminder effect of 2 sessions in a simulated fixture; the host reads support and acceptance fields and applies its own policy. Loading an artifact does not recreate a live study.</details> | intermediate / up to 1 min | none | [python](python/application_component.py) |
| **Estimate a training programme's effect from observational data** — Adjust for declared confounders on a job-training study with a comparison group, using a public dataset on request and a synthetic stand-in offline. <details><summary>Claim scope</summary>Effect on the treated under a declared no-unmeasured-confounding graph. Offline runs use a synthetic stand-in with a known truth; real data are evidence about whether the adjustment works here, never known causal truth. See examples/data/nsw_psid.toml for provenance and licence.</details> | intermediate / up to 1 min | numpy | [python](python/observational_study.py) |

## Data and inputs

| Example and question | Level / estimated runtime | Dependencies beyond Antecedent | Run / open |
| --- | --- | --- | --- |
| **Analyze a table of your own** — Read a temporary CSV, check numeric columns, prepare and inspect an average effect. <details><summary>Claim scope</summary>Reviewed numeric table and declared DAG average effect; missing values require an explicit observation policy.</details> | beginner / up to 1 min | pandas | [python](python/your_data.py) |
| **Analyze a Parquet file in bounded chunks** — Analyze a Parquet file read in bounded chunks, then stream a summary through a pool smaller than the table. <details><summary>Claim scope</summary>The average effect from a Parquet file equals the one from the same table in memory; only the needed columns are decoded, every read is charged to the admission pool, and the content identity tracks the values. Calibration above 100k rows is not assessed.</details> | beginner / up to 1 min | pyarrow | [python](python/parquet_e2e.py) |

## Estimate effects

| Example and question | Level / estimated runtime | Dependencies beyond Antecedent | Run / open |
| --- | --- | --- | --- |
| **Adjust for a confounder by propensity weighting** — Adjust for treatment selection using inverse-probability weights <details><summary>Claim scope</summary>Average effect via propensity weighting under a reviewed DAG; inspect overlap and interval standing.</details> | beginner / up to 1 min | none | [python](python/propensity_weighting.py) · [rust](rust/propensity_weighting.rs) |
| **Compare propensity weighting and AIPW on a simulated confounded benchmark** — Synthetic IHDP-like covariates → assumed DAG → IPW + AIPW + refute <details><summary>Claim scope</summary>Synthetic IHDP-like covariates under an assumed DAG; IPW and AIPW effects and refutation.</details> | intermediate / up to 1 min | none | [python](python/ihdp_propensity_e2e.py) |
| **Split an effect into direct and mediated parts** — Separate direct and mediated effects; estimate individual effects <details><summary>Claim scope</summary>Average, mediated and individual effects under the declared static model and supported query-specific routes.</details> | intermediate / up to 1 min | none | [python](python/staged_static_kinds.py) · [rust](rust/staged_static_kinds.rs) |
| **Ask several causal questions of one sales dataset** — Explore average, mediated, individual, and delayed effects <details><summary>Claim scope</summary>Simulated average, mediated, individual and temporal effects, each under its own declared question.</details> | intermediate / up to 5 min | none | [python](python/sales_spreadsheet_e2e.py) · [rust](rust/sales_spreadsheet_e2e.rs) |
| **How does the outcome change with dose?** — Inspect supported doses, response values and derivatives. <details><summary>Claim scope</summary>Supported response and derivative coordinates under an explicit dose-response model.</details> | intermediate / up to 5 min | pandas, matplotlib, ipykernel | [notebook](notebooks/continuous_causal_response.ipynb) · [Colab](https://colab.research.google.com/github/iridae-dev/antecedent/blob/v3.0.0/examples/notebooks/continuous_causal_response.ipynb) |
| **Pricing, availability, and latent demand** — Distinguish inventory-limited sales from latent demand. <details><summary>Claim scope</summary>Demand response under an explicit censoring premise; unsupported derivative requests are refused.</details> | intermediate / up to 5 min | pandas, matplotlib, ipykernel | [notebook](notebooks/pricing_availability_latent_demand.ipynb) · [Colab](https://colab.research.google.com/github/iridae-dev/antecedent/blob/v3.0.0/examples/notebooks/pricing_availability_latent_demand.ipynb) |
| **Bound an effect under unmeasured confounding** — Marginal-sensitivity bounds and tipping point for a stratified ATE <details><summary>Claim scope</summary>Assumption-sensitivity range and tipping point for a stratified average effect; not a confidence interval.</details> | intermediate / up to 1 min | none | [python](python/msm_sensitivity.py) |
| **Score a fixed treatment policy on held-out rows** — Held-out value of a fixed binary policy under known propensities <details><summary>Claim scope</summary>Held-out value of a fixed binary policy under known assignment propensities.</details> | intermediate / up to 1 min | none | [python](python/policy_value_evaluation.py) |
| **Value a treatment regime over two time points** — Value of a static treatment rule by inverse-probability g-formula <details><summary>Claim scope</summary>Static longitudinal-regime value by inverse-probability g-formula under declared histories and probabilities.</details> | intermediate / up to 1 min | none | [python](python/regime_value.py) |
| **Estimate a difference-in-differences effect** — 2x2 and balanced-panel DiD point utilities <details><summary>Claim scope</summary>Two-period and balanced-panel difference-in-differences under their declared design assumptions.</details> | intermediate / up to 1 min | none | [python](python/quasi_difference_in_differences.py) |
| **Compare restricted mean survival time between arms** — Randomized survival curve and RMST by arm <details><summary>Claim scope</summary>Randomized survival and restricted mean survival time under declared observation and assignment assumptions.</details> | intermediate / up to 1 min | none | [python](python/survival_rmst.py) |
| **Hand an adjustment set to DoWhy** — Exchange a reviewed backdoor graph with DoWhy (optional dependency) <details><summary>Claim scope</summary>Reviewed backdoor graph exchange; each library owns its own fit and scientific interpretation.</details> | intermediate / up to 1 min | pandas, dowhy (optional) | [python](python/dowhy_handoff.py) |
| **Hand an adjustment set to EconML** — Compare adjusted average effects with EconML CATE predictions (optional dependency) <details><summary>Claim scope</summary>Antecedent supplies adjustment columns; the optional EconML fit owns its conditional-effect output.</details> | intermediate / up to 1 min | econml (optional) | [python](python/econml_cate_handoff.py) |

## Experiments and study designs

| Example and question | Level / estimated runtime | Dependencies beyond Antecedent | Run / open |
| --- | --- | --- | --- |
| **Estimate a randomized effect and reuse its design** — Estimate a randomized ITT effect; design inputs named by data column <details><summary>Claim scope</summary>Randomized ITT under declared assignment probabilities and units; inspect graphless support and interval standing.</details> | beginner / up to 1 min | none | [python](python/experiment_randomized_effect.py) |
| **Read main effects and an interaction from a 2x2 experiment** — 2x2 Bernoulli factorial main effects and interaction (point utility) <details><summary>Claim scope</summary>Declared Bernoulli factorial cell means, main effects and interaction.</details> | intermediate / up to 1 min | none | [python](python/factorial_cell_means.py) |
| **Enumerate the outcomes of an adaptive procedure** — Enumerate a tiny procedure under supplied truths, keeping undefined contrast mass visible. <details><summary>Claim scope</summary>Derived procedure report conditional on declared design and truths</details> | intermediate / up to 1 min | none | [python](python/adaptive_procedure.py) |
| **Estimate a delayed pulse effect on defects** — Estimate how a pressure change affects later defects <details><summary>Claim scope</summary>Lagged intervention effect under an explicit temporal graph; horizons and support remain part of the claim.</details> | intermediate / up to 1 min | none | [python](python/manufacturing_temporal.py) · [rust](rust/manufacturing_temporal.rs) |
| **Trace a delayed response across dose and horizon** — Compare pressure levels and their effects over time <details><summary>Claim scope</summary>Temporal response grid under declared lag structure; inspect each requested coordinate and uncertainty.</details> | intermediate / up to 1 min | none | [python](python/temporal_response_curve.py) · [rust](rust/temporal_response_curve.rs) |
| **Discover a lagged parent, then estimate its pulse effect** — Discover a lagged graph and estimate a pulse effect in simulated data <details><summary>Claim scope</summary>Pulse effect conditional on an accepted lagged graph discovered from the simulated series.</details> | intermediate / up to 5 min | none | [python](python/pcmci_pulse_benchmark.py) |

## Structure and discovery

| Example and question | Level / estimated runtime | Dependencies beyond Antecedent | Run / open |
| --- | --- | --- | --- |
| **Ask what is identified when an edge direction is unknown** — Estimate a range of effects when some edge directions are unknown <details><summary>Claim scope</summary>Class-preserving bounds or partial answer; a reviewed orientation changes the structural declaration.</details> | intermediate / up to 1 min | none | [python](python/class_preserving_cpdag.py) · [rust](rust/class_preserving_cpdag.rs) |
| **Discover a graph, accept it and estimate** — Discover once, accept a DAG, re-estimate many times <details><summary>Claim scope</summary>Estimation conditional on an explicitly accepted discovered DAG; acceptance does not make discovery certain.</details> | intermediate / up to 5 min | none | [python](python/discover_then_estimate.py) · [rust](rust/discover_then_estimate.rs) |
| **Ask whether an effect is identified before fitting** — Identification without fitting <details><summary>Claim scope</summary>Identification and derivation without numerical fitting.</details> | intermediate / up to 1 min | none | [rust](rust/identify_only.rs) |
| **Should we increase paid-search spend?** — Compare naive marketing attribution with effects under uncertain graph directions. <details><summary>Claim scope</summary>Attribution conditional on graph completions versus a reviewed graph; preserve unidentified structural mass.</details> | intermediate / up to 5 min | pandas, matplotlib, ipykernel | [notebook](notebooks/marketing_channel_structural_uncertainty.ipynb) · [Colab](https://colab.research.google.com/github/iridae-dev/antecedent/blob/v3.0.0/examples/notebooks/marketing_channel_structural_uncertainty.ipynb) |
| **Test a graph against random relabellings** — Compare a DAG with random relabellings of itself through the independencies it asserts <details><summary>Claim scope</summary>Permutation comparison of declared graph independencies under the named conditional-independence test; not rejected does not show the graph correct, and nothing is fitted.</details> | intermediate / up to 1 min | none | [python](python/graph_falsification.py) · [rust](rust/graph_falsification.rs) |
| **Select a mechanism family under constraints** — Score linear, spline and tree families on shared held-out folds and keep the audit trail <details><summary>Claim scope</summary>Held-out comparison of declared candidate mechanisms with constraints and split schemes; the choice is a model-fit decision, not a causal one.</details> | intermediate / up to 1 min | none | [python](python/mechanism_selection.py) · [rust](rust/mechanism_selection.rs) |

## Structural models and feedback

| Example and question | Level / estimated runtime | Dependencies beyond Antecedent | Run / open |
| --- | --- | --- | --- |
| **Run a price-floor intervention program** — Apply a price floor to a fitted model, replay it on factual units, save it and test its callback form <details><summary>Claim scope</summary>Typed value programs and batch callbacks executed on declared mechanisms; routes are licensed separately and a runtime callback is not portable.</details> | intermediate / up to 1 min | none | [python](python/intervention_programs.py) · [rust](rust/intervention_programs.rs) |
| **Attribute a variance to its upstream causes** — Split a target's variance among the noise terms of its ancestors <details><summary>Claim scope</summary>Variance shares of independent, declared noise terms under fitted mechanisms; each interval names the one uncertainty component it covers and none is calibrated.</details> | intermediate / up to 1 min | none | [python](python/intrinsic_influence.py) · [rust](rust/intrinsic_influence.rs) |
| **Replay exogenous noise and fix one coordinate** — Keep a model's exogenous noise, fix or redraw coordinates and evaluate again <details><summary>Claim scope</summary>Replay of a compiled model on retained noise with fixed, redrawn and coalition coordinates; closed-form variances on a linear Gaussian chain.</details> | intermediate / up to 1 min | none | [python](python/noise_replay.py) · [rust](rust/noise_replay.rs) |
| **Sample from a fitted model under an intervention** — Fit a GCM and sample under `do(·)` <details><summary>Claim scope</summary>Interventional samples under fitted, declared causal mechanisms.</details> | intermediate / up to 1 min | none | [rust](rust/gcm_do.rs) |
| **Ask a counterfactual through a shared hidden cause** — Compute a counterfactual through an exogenous term shared by two mechanisms <details><summary>Claim scope</summary>Exact enumeration of a fully declared finite SCM whose mechanisms share an exogenous term; undeclared latent confounding is refused as not identified.</details> | intermediate / up to 1 min | none | [python](python/shared_exogenous_counterfactual.py) · [rust](rust/shared_exogenous_counterfactual.rs) |
| **Fit overlapping feedback loops and replay an artifact** — Fit overlapping nonlinear feedback loops, query a held-out joint intervention and replay a shared-noise counterfactual artifact. <details><summary>Claim scope</summary>Specified-structure/covariance nonlinear equilibrium likelihood and shared-disturbance counterfactuals; read result.inspect().calibration for the calibration status of any interval.</details> | advanced / up to 1 min | none | [python](python/general_cyclic_systems.py) |
| **Learn a feedback law from clamp experiments** — Learn a linear feedback law from a passive sample and two randomized clamp experiments, branch intervention targets and reuse a consumed law. <details><summary>Claim scope</summary>Linear feedback recovered under a declared recipe from independent clamp experiments; sampling draws describe fit uncertainty, not a parameter posterior or calibrated coverage.</details> | advanced / up to 1 min | none | [python](python/recovered_feedback_law.py) |
| **Reproduce the published cyclic-system protocols** — Reproduce applicable nonlinear and stochastic feedback synthetic experiments with held-out joint interventions and fresh CPU artifact replay. <details><summary>Claim scope</summary>Specified synthetic likelihood experiments with explicit deviations, optimizer diagnostics and independent numerical references; no calibration or latent recovery claim</details> | advanced / paper-size runs vary with feature and sample counts | numpy, scipy | [python](python/cyclic_paper_experiments.py) |

## Joint interventions and search

| Example and question | Level / estimated runtime | Dependencies beyond Antecedent | Run / open |
| --- | --- | --- | --- |
| **Answer a joint intervention question on a panel in Rust** — Compile a joint intervention-response question on an independent panel fixture, then fit it with frequentist and Bayesian inference. <details><summary>Claim scope</summary>Model-conditional joint responses under a declared period randomization; compilation performs no fit and the report carries numerical diagnostics, not calibrated coverage.</details> | advanced / up to 1 min | none | [rust](rust/joint_panel_workflow.rs) |

## Transport

| Example and question | Level / estimated runtime | Dependencies beyond Antecedent | Run / open |
| --- | --- | --- | --- |
| **Transport a response from a known source law** — Single-source exact-law transport of a response grid <details><summary>Claim scope</summary>Transported response from supplied exact finite laws; no sampling interval is inferred.</details> | intermediate / up to 1 min | none | [python](python/transport_exact.py) · [rust](rust/transport_exact.rs) |
| **Transport a response from a source table** — Single-source empirical transport of a response grid <details><summary>Claim scope</summary>Empirical-table transported grid; inspect local support and bootstrap calibration at this coordinate.</details> | intermediate / up to 1 min | none | [python](python/transport_statistical.py) · [rust](rust/transport_statistical.rs) |
| **Transport from complementary sources over a grid** — Combined sources identify a target curve; one source does not <details><summary>Claim scope</summary>Complementary finite source laws identify a grid; missing laws retain unsupported coordinates.</details> | intermediate / up to 1 min | none | [python](python/transport_meta_grid.py) · [rust](rust/transport_meta_grid.rs) |
| **Transport a mediated effect and see where it refuses** — Transport direct and indirect effects under a shared additive model; inspect refusals <details><summary>Claim scope</summary>Static transported path-specific contrast under the supported shared affine-additive model.</details> | intermediate / up to 1 min | none | [python](python/transported_counterfactual.py) |
| **Carry a measurement correction to another clinic** — Correct target prevalence under explicitly invariant source sensitivity and specificity. <details><summary>Claim scope</summary>Conditional point and uncalibrated standard error under declared invariance</details> | intermediate / up to 1 min | none | [python](python/transported_measurement.py) |
| **Decide which factors carry over to a new scenario** — Why each transport scenario answers as it does: selection and invariances <details><summary>Claim scope</summary>Inspection of explicitly declared transport scenarios and their invariance assumptions.</details> | intermediate / up to 1 min | none | [python](python/scenario_invariance.py) |
| **Test whether a mechanism is invariant across sources** — Test whether a node's mechanism differs between source and target <details><summary>Claim scope</summary>Diagnostic comparison of source and target mechanisms; a diagnostic does not prove invariance.</details> | intermediate / up to 1 min | none | [python](python/mechanism_discrepancy.py) |
| **Check a transported claim against a provider input** — Seal a declared transported offer to a callback input and refuse a native-only consumer. <details><summary>Claim scope</summary>Transported external claim; handoff checks do not prove numerical transport</details> | intermediate / up to 1 min | none | [python](python/provider_transport.py) |
| **Reload a learner and a transported estimate** — Prepare, estimate, inspect and reload fitted models and transport results <details><summary>Claim scope</summary>Supported learner-backed estimation and transport with retained model and artifact identity.</details> | intermediate / up to 1 min | none | [python](python/cohesive_ml_transport.py) |

## Decisions and study planning

| Example and question | Level / estimated runtime | Dependencies beyond Antecedent | Run / open |
| --- | --- | --- | --- |
| **Rank candidate studies by identification** — Rank candidate experiments by identification probability <details><summary>Claim scope</summary>Identification-oriented ranking of declared studies; inspect objective, budgets and Monte Carlo readings.</details> | intermediate / up to 5 min | none | [python](python/rank_designs.py) · [rust](rust/rank_designs.rs) |
| **Rank candidate studies by expected value net of cost** — Rank candidate studies by expected value of sample information <details><summary>Claim scope</summary>Expected-value study ranking under the supplied planning model and utility declarations.</details> | intermediate / up to 5 min | none | [python](python/rank_designs_evsi.py) |
| **Which marketing experiment should we run next?** — Rank investments that resolve a marketing identification problem. <details><summary>Claim scope</summary>Budgeted study ranking under the declared marketing causal question and study costs.</details> | intermediate / up to 5 min | pandas, matplotlib, ipykernel | [notebook](notebooks/marketing_experiment_design.ipynb) · [Colab](https://colab.research.google.com/github/iridae-dev/antecedent/blob/v3.0.0/examples/notebooks/marketing_experiment_design.ipynb) |
| **Find which study repairs a missing identification** — State the evidence a contract owes and which study would repair it <details><summary>Claim scope</summary>Evidence obligations and declared study repairs; a repair plan is not new observed evidence.</details> | intermediate / up to 1 min | none | [python](python/repair_obligations.py) |
| **Update a decision without refitting** — Recompute a decision with zero refits; resume from exported scores <details><summary>Claim scope</summary>Retargeting retained scores and portable resume under the original supported data and score identities.</details> | intermediate / up to 1 min | none | [python](python/recalc_cell_resume.py) |

## Evidence and composition

| Example and question | Level / estimated runtime | Dependencies beyond Antecedent | Run / open |
| --- | --- | --- | --- |
| **Compose two quantities and read the spread** — Compute a product from one aligned law and distinguish its spread from interval coverage. <details><summary>Claim scope</summary>Point and uncalibrated spread under one aligned joint law</details> | intermediate / up to 1 min | none | [python](python/composed_uncertainty.py) |
| **Hand a composed decision to another consumer** — Export a composed decision and consume it under a retained identity <details><summary>Claim scope</summary>Portable composition with independently retained identities and checked downstream consumption.</details> | intermediate / up to 1 min | none | [python](python/composition_bundle.py) |
| **Combine earlier surveys as prior evidence** — Select and combine evidence from earlier surveys <details><summary>Claim scope</summary>Declared historical-evidence selection and pooling with source and target compatibility checks.</details> | intermediate / up to 5 min | none | [python](python/prior_bank_surveys.py) · [rust](rust/prior_bank_surveys.rs) |
| **Carry one batch's posterior into the next** — Use one batch’s results as prior evidence for the next <details><summary>Claim scope</summary>Posterior evidence transferred between compatible analyses under their declared models and evidence policy.</details> | intermediate / up to 5 min | none | [python](python/sequential_bayes.py) · [rust](rust/sequential_bayes.rs) |
| **Can last quarter's campaign inform this quarter's sales analysis?** — Transfer historical campaign evidence and update it with current observations. <details><summary>Claim scope</summary>Historical treatment-effect posterior transfer with target compatibility and new-data updating.</details> | intermediate / up to 5 min | pandas, matplotlib, ipykernel | [notebook](notebooks/sales_campaign_prior_transfer.ipynb) · [Colab](https://colab.research.google.com/github/iridae-dev/antecedent/blob/v3.0.0/examples/notebooks/sales_campaign_prior_transfer.ipynb) |

## Prepare, branch and reuse

| Example and question | Level / estimated runtime | Dependencies beyond Antecedent | Run / open |
| --- | --- | --- | --- |
| **Export a result with a separately retained identity** — Store bytes and a separately retainable producer expectation. <details><summary>Claim scope</summary>Separately retainable scientific identity and bytes; saving does not establish real-world causal validity.</details> | beginner / up to 1 min | none | [python](python/artifact_producer.py) |
| **Consume an artifact against its retained identity** — Check independent consumption, wrong identity and altered bytes. <details><summary>Claim scope</summary>Identity-bound consumption, wrong-identity and malformed-container refusals; loading does not recreate a live study.</details> | beginner / up to 1 min | none | [python](python/artifact_consumer.py) |
| **Track which queries a data append makes stale** — Update an analysis as data arrive and identify outdated results <details><summary>Claim scope</summary>State updates and explicit stale-result handling as observations arrive.</details> | beginner / up to 1 min | none | [python](python/causal_state_workflow.py) · [rust](rust/causal_state_workflow.rs) |

## Build with Antecedent

| Example and question | Level / estimated runtime | Dependencies beyond Antecedent | Run / open |
| --- | --- | --- | --- |
| **Read status, claims and refusals as a host** — Compile, read the status, tell a supported claim from a refusal and load an artifact, with no agent logic. <details><summary>Claim scope</summary>Status, coded refusal and identity-bound artifact acceptance read through the host-integration contract; the host plans nothing and certifies no assumption.</details> | intermediate / up to 1 min | none | [python](python/reference_host.py) · [rust](rust/reference_host.rs) |
| **Measure Python ingest overhead** — Measure Python-to-native ingest overhead with repeatable workload inputs. <details><summary>Claim scope</summary>Numerical timing; no scientific claim</details> | advanced / up to 5 min | pandas | [python](python/bench_python_overhead.py) |

For exact claim settings, inspect each result and check the support matrix. Optional handoff dependencies are needed to exercise their external half; an absent dependency prints a skip.

<!-- generated:example-catalog:end -->
