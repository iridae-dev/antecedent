# Choose an example

Start with the [Python quickstart](python-workflow.md) or [Rust quickstart](rust-quickstart.md). For notebooks, follow the [notebook setup](https://github.com/iridae-dev/antecedent/blob/v2.3.0/examples/README.md#python-environment). All examples use simulated data.

## Notebooks

See Antecedent on a real decision. Open a notebook locally or in Google Colab after setup. Run all cells to generate the tables, plots, and reports in your environment.

### [Paid-search attribution](https://github.com/iridae-dev/antecedent/blob/v2.3.0/examples/notebooks/marketing_channel_structural_uncertainty.ipynb)

See how a naive marketing dashboard can materially overstate paid-search impact by crediting the campaign for demand that would have existed anyway. Compare results under uncertain graph directions with an estimate conditional on a reviewed graph.

### [Campaign evidence transfer](https://github.com/iridae-dev/antecedent/blob/v2.3.0/examples/notebooks/sales_campaign_prior_transfer.ipynb)

Use evidence from a previous sales campaign without assuming the new campaign is identical. Antecedent transfers the historical treatment-effect posterior into a different target model, then lets current data update—or contradict—it.

### [Marketing experiment design](https://github.com/iridae-dev/antecedent/blob/v2.3.0/examples/notebooks/marketing_experiment_design.ipynb)

Compare a holdout experiment, better intent data and additional CRM records to determine which investment actually resolves the causal question. Antecedent identifies the best feasible action under a £40,000 budget and shows why collecting more of the same data would not fix the attribution problem.

### [Continuous causal response](https://github.com/iridae-dev/antecedent/blob/v2.3.0/examples/notebooks/continuous_causal_response.ipynb)

Ask how the outcome changes with dose, how steep the curve is, and how that slope changes across the data. Check where the data support those answers.

### [Pricing, availability and latent demand](https://github.com/iridae-dev/antecedent/blob/v2.3.0/examples/notebooks/pricing_availability_latent_demand.ipynb)

Show why inventory-limited sales are not demand. Estimate a demand curve under an explicit censoring assumption, then see why the same method refuses a demand derivative.

## Python scripts

Choose Python or Rust for each workflow.

For moving a reviewed backdoor graph between Antecedent and DoWhy without wrapping either library, see the [DoWhy interoperability cookbook](interop_dowhy.md) and [`dowhy_handoff.py`](https://github.com/iridae-dev/antecedent/blob/v2.3.0/examples/python/dowhy_handoff.py) (`dowhy` optional).

[The Python analysis workflow](https://github.com/iridae-dev/antecedent/blob/v2.3.0/examples/python/analysis_workflow.py) demonstrates the one-call API, retained study, inspection, refresh, and verified artifact loading.

```bash
# Python (from repo root, with your environment activated)
python examples/python/<name>.py

# Rust
cargo run -p antecedent --example <name>
```

| Example | Description | Python | Rust |
| ------- | ----------- | ------ | ---- |
| Propensity weighting | Adjust for treatment selection using inverse-probability weights | [python](https://github.com/iridae-dev/antecedent/blob/v2.3.0/examples/python/propensity_weighting.py) | [rust](https://github.com/iridae-dev/antecedent/blob/v2.3.0/examples/rust/propensity_weighting.rs) |
| IHDP-style propensity / AIPW | Synthetic IHDP-like covariates → assumed DAG → IPW + AIPW + refute | [python](https://github.com/iridae-dev/antecedent/blob/v2.3.0/examples/python/ihdp_propensity_e2e.py) | — |
| Mediation and individual effects | Separate direct and mediated effects; estimate individual effects | [python](https://github.com/iridae-dev/antecedent/blob/v2.3.0/examples/python/staged_static_kinds.py) | [rust](https://github.com/iridae-dev/antecedent/blob/v2.3.0/examples/rust/staged_static_kinds.rs) |
| Uncertain graph directions | Estimate a range of effects when some edge directions are unknown | [python](https://github.com/iridae-dev/antecedent/blob/v2.3.0/examples/python/class_preserving_cpdag.py) | [rust](https://github.com/iridae-dev/antecedent/blob/v2.3.0/examples/rust/class_preserving_cpdag.rs) |
| Manufacturing temporal | Estimate how a pressure change affects later defects | [python](https://github.com/iridae-dev/antecedent/blob/v2.3.0/examples/python/manufacturing_temporal.py) | [rust](https://github.com/iridae-dev/antecedent/blob/v2.3.0/examples/rust/manufacturing_temporal.rs) |
| Temporal response curve | Compare pressure levels and their effects over time | [python](https://github.com/iridae-dev/antecedent/blob/v2.3.0/examples/python/temporal_response_curve.py) | [rust](https://github.com/iridae-dev/antecedent/blob/v2.3.0/examples/rust/temporal_response_curve.rs) |
| Discover then estimate | Discover once, accept a DAG, re-estimate many times | [python](https://github.com/iridae-dev/antecedent/blob/v2.3.0/examples/python/discover_then_estimate.py) | [rust](https://github.com/iridae-dev/antecedent/blob/v2.3.0/examples/rust/discover_then_estimate.rs) |
| Sequential Bayes | Use one batch’s results as prior evidence for the next | [python](https://github.com/iridae-dev/antecedent/blob/v2.3.0/examples/python/sequential_bayes.py) | [rust](https://github.com/iridae-dev/antecedent/blob/v2.3.0/examples/rust/sequential_bayes.rs) |
| Prior bank surveys | Select and combine evidence from earlier surveys | [python](https://github.com/iridae-dev/antecedent/blob/v2.3.0/examples/python/prior_bank_surveys.py) | [rust](https://github.com/iridae-dev/antecedent/blob/v2.3.0/examples/rust/prior_bank_surveys.rs) |
| Rank designs | Rank candidate experiments by identification probability | [python](https://github.com/iridae-dev/antecedent/blob/v2.3.0/examples/python/rank_designs.py) | [rust](https://github.com/iridae-dev/antecedent/blob/v2.3.0/examples/rust/rank_designs.rs) |
| CausalState workflow | Update an analysis as data arrive and identify outdated results | [python](https://github.com/iridae-dev/antecedent/blob/v2.3.0/examples/python/causal_state_workflow.py) | [rust](https://github.com/iridae-dev/antecedent/blob/v2.3.0/examples/rust/causal_state_workflow.rs) |
| Sales analysis | Explore average, mediated, individual, and delayed effects | [python](https://github.com/iridae-dev/antecedent/blob/v2.3.0/examples/python/sales_spreadsheet_e2e.py) | [rust](https://github.com/iridae-dev/antecedent/blob/v2.3.0/examples/rust/sales_spreadsheet_e2e.rs) |
| Transport a source table | Single-source empirical transport of a response grid | [python](https://github.com/iridae-dev/antecedent/blob/v2.3.0/examples/python/transport_statistical.py) | [rust](https://github.com/iridae-dev/antecedent/blob/v2.3.0/examples/rust/transport_statistical.rs) |
| Transport an exact law | Single-source exact-law transport of a response grid | [python](https://github.com/iridae-dev/antecedent/blob/v2.3.0/examples/python/transport_exact.py) | [rust](https://github.com/iridae-dev/antecedent/blob/v2.3.0/examples/rust/transport_exact.rs) |
| Complementary-source grid | Combined sources identify a target curve; one source does not | [python](https://github.com/iridae-dev/antecedent/blob/v2.3.0/examples/python/transport_meta_grid.py) | [rust](https://github.com/iridae-dev/antecedent/blob/v2.3.0/examples/rust/transport_meta_grid.rs) |
| Randomized experiment | Estimate a randomized ITT effect; design inputs named by data column | [python](https://github.com/iridae-dev/antecedent/blob/v2.3.0/examples/python/experiment_randomized_effect.py) | — |
| Factorial cell means | 2x2 Bernoulli factorial main effects and interaction (point utility) | [python](https://github.com/iridae-dev/antecedent/blob/v2.3.0/examples/python/factorial_cell_means.py) | — |
| Policy value | Held-out value of a fixed binary policy under known propensities | [python](https://github.com/iridae-dev/antecedent/blob/v2.3.0/examples/python/policy_value_evaluation.py) | — |
| Difference-in-differences | 2x2 and balanced-panel DiD point utilities | [python](https://github.com/iridae-dev/antecedent/blob/v2.3.0/examples/python/quasi_difference_in_differences.py) | — |
| Longitudinal regime value | Value of a static treatment rule by inverse-probability g-formula | [python](https://github.com/iridae-dev/antecedent/blob/v2.3.0/examples/python/regime_value.py) | — |
| Survival RMST | Randomized survival curve and RMST by arm | [python](https://github.com/iridae-dev/antecedent/blob/v2.3.0/examples/python/survival_rmst.py) | — |
| Decision lifecycle | Analyze, bind external evidence, inspect, decide, then rank the next study | [python](https://github.com/iridae-dev/antecedent/blob/v2.3.0/examples/python/decision_lifecycle.py) | — |
| MSM sensitivity | Marginal-sensitivity bounds and tipping point for a stratified ATE | [python](https://github.com/iridae-dev/antecedent/blob/v2.3.0/examples/python/msm_sensitivity.py) | — |
| Mechanism discrepancy | Test whether a node's mechanism differs between source and target | [python](https://github.com/iridae-dev/antecedent/blob/v2.3.0/examples/python/mechanism_discrepancy.py) | — |
| Scenario invariance | Why each transport scenario answers as it does: selection and invariances | [python](https://github.com/iridae-dev/antecedent/blob/v2.3.0/examples/python/scenario_invariance.py) | — |
| Transported counterfactual | Transport direct and indirect effects under a shared additive model; inspect refusals | [python](https://github.com/iridae-dev/antecedent/blob/v2.3.0/examples/python/transported_counterfactual.py) | — |
| Recalculation with frozen scores | Recompute a decision with zero refits; resume from exported scores | [python](https://github.com/iridae-dev/antecedent/blob/v2.3.0/examples/python/recalc_cell_resume.py) | — |
| Composition bundle | Export a composed decision and consume it under a retained identity | [python](https://github.com/iridae-dev/antecedent/blob/v2.3.0/examples/python/composition_bundle.py) | — |
| Repair and evidence obligations | State the evidence a contract owes and which study would repair it | [python](https://github.com/iridae-dev/antecedent/blob/v2.3.0/examples/python/repair_obligations.py) | — |
| Expected value of a study | Rank candidate studies by expected value of sample information | [python](https://github.com/iridae-dev/antecedent/blob/v2.3.0/examples/python/rank_designs_evsi.py) | — |
| ML and transport lifecycle | Prepare, estimate, inspect and reload fitted models and transport results | [python](https://github.com/iridae-dev/antecedent/blob/v2.3.0/examples/python/cohesive_ml_transport.py) | — |
| DoWhy handoff | Exchange a reviewed backdoor graph with DoWhy (optional dependency) | [python](https://github.com/iridae-dev/antecedent/blob/v2.3.0/examples/python/dowhy_handoff.py) | — |
| EconML handoff | Compare adjusted average effects with EconML CATE predictions (optional dependency) | [python](https://github.com/iridae-dev/antecedent/blob/v2.3.0/examples/python/econml_cate_handoff.py) | — |
| PCMCI pulse discovery | Discover a lagged graph and estimate a pulse effect in simulated data | [python](https://github.com/iridae-dev/antecedent/blob/v2.3.0/examples/python/pcmci_pulse_benchmark.py) | — |
| ATE quickstart | Build and run an average-effect analysis | — | [rust](https://github.com/iridae-dev/antecedent/blob/v2.3.0/examples/rust/ate_quickstart.rs) |
| Identify only | Identification without fitting | — | [rust](https://github.com/iridae-dev/antecedent/blob/v2.3.0/examples/rust/identify_only.rs) |
| GCM do | Fit a GCM and sample under `do(·)` | — | [rust](https://github.com/iridae-dev/antecedent/blob/v2.3.0/examples/rust/gcm_do.rs) |
