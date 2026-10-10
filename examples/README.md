# Examples

## Python environment

These examples use Antecedent from PyPI and Python 3.11 or later:

```bash
python -m pip install "antecedent>=2.3.0,<2.4" pandas matplotlib ipykernel jupyterlab
```

Download an example notebook and open it with `jupyter lab`. If you have the repository locally, run `jupyter lab examples/notebooks` from its root. Choose the kernel for the Python environment where you installed Antecedent.

For Google Colab, **Runtime → Run all**. If you upgrade after importing Antecedent, restart the kernel or Colab session.

## Start with one analysis

Run the [first analysis](python/analysis_workflow.py):

```bash
python examples/python/analysis_workflow.py
```

It creates a simulated experiment with a treatment effect of 2, estimates that effect, and then repeats the analysis on updated data. It also shows how to save and reload the result. Read the [step-by-step guide](../docs/python-workflow.md) for installation, expected output, and explanations.

Choose an example below by the question you want to answer. All datasets are simulated. The assertions check the examples; they do not validate assumptions for your own data.

## Notebooks

See Antecedent on a real decision. Open a notebook locally or in Google Colab after the setup above. Run all cells to generate the tables, plots, and reports in your environment.

### [Paid-search attribution](notebooks/marketing_channel_structural_uncertainty.ipynb)

See how a naive marketing dashboard can materially overstate paid-search impact by crediting the campaign for demand that would have existed anyway. Compare results under uncertain graph directions with an estimate conditional on a reviewed graph.

### [Campaign evidence transfer](notebooks/sales_campaign_prior_transfer.ipynb)

Use evidence from a previous sales campaign without assuming the new campaign is identical. Antecedent transfers the historical treatment-effect posterior into a different target model, then lets current data update—or contradict—it.

### [Marketing experiment design](notebooks/marketing_experiment_design.ipynb)

Compare a holdout experiment, better intent data and additional CRM records to determine which investment actually resolves the causal question. Antecedent identifies the best feasible action under a £40,000 budget and shows why collecting more of the same data would not fix the attribution problem.

### [Continuous causal response](notebooks/continuous_causal_response.ipynb)

Ask how the outcome changes with dose, how steep the curve is, and how that
slope changes across the data. Check where the data support those answers.

### [Pricing, availability and latent demand](notebooks/pricing_availability_latent_demand.ipynb)

Show why inventory-limited sales are not demand. Estimate a demand curve under an explicit censoring assumption, then see why
the same method refuses a demand derivative.

## Scripts

Choose Python or Rust for each workflow.

[The Python analysis workflow](python/analysis_workflow.py) demonstrates the one-call
API, retained study, inspection, refresh, and verified artifact loading.

```bash
# Python (from repo root, with the environment above activated)
python examples/python/<name>.py

# Rust
cargo run -p antecedent --example <name>
```

| Example | Description | Python | Rust |
| ------- | ----------- | ------ | ---- |
| Propensity weighting | Adjust for treatment selection using inverse-probability weights | [python](python/propensity_weighting.py) | [rust](rust/propensity_weighting.rs) |
| IHDP-style propensity / AIPW | Synthetic IHDP-like covariates → assumed DAG → IPW + AIPW + refute | [python](python/ihdp_propensity_e2e.py) | — |
| Mediation and individual effects | Separate direct and mediated effects; estimate individual effects | [python](python/staged_static_kinds.py) | [rust](rust/staged_static_kinds.rs) |
| Uncertain graph directions | Estimate a range of effects when some edge directions are unknown | [python](python/class_preserving_cpdag.py) | [rust](rust/class_preserving_cpdag.rs) |
| Manufacturing temporal | Estimate how a pressure change affects later defects | [python](python/manufacturing_temporal.py) | [rust](rust/manufacturing_temporal.rs) |
| Temporal response curve | Compare pressure levels and their effects over time | [python](python/temporal_response_curve.py) | [rust](rust/temporal_response_curve.rs) |
| Discover then estimate | Discover once, accept a DAG, re-estimate many times | [python](python/discover_then_estimate.py) | [rust](rust/discover_then_estimate.rs) |
| Sequential Bayes | Use one batch’s results as prior evidence for the next | [python](python/sequential_bayes.py) | [rust](rust/sequential_bayes.rs) |
| Prior bank surveys | Select and combine evidence from earlier surveys | [python](python/prior_bank_surveys.py) | [rust](rust/prior_bank_surveys.rs) |
| Rank designs | Rank candidate experiments by identification probability | [python](python/rank_designs.py) | [rust](rust/rank_designs.rs) |
| CausalState workflow | Update an analysis as data arrive and identify outdated results | [python](python/causal_state_workflow.py) | [rust](rust/causal_state_workflow.rs) |
| Sales analysis | Explore average, mediated, individual, and delayed effects | [python](python/sales_spreadsheet_e2e.py) | [rust](rust/sales_spreadsheet_e2e.rs) |
| Transport a source table | Single-source empirical transport of a response grid | [python](python/transport_statistical.py) | [rust](rust/transport_statistical.rs) |
| Transport an exact law | Single-source exact-law transport of a response grid | [python](python/transport_exact.py) | [rust](rust/transport_exact.rs) |
| Complementary-source grid | Combined sources identify a target curve; one source does not | [python](python/transport_meta_grid.py) | [rust](rust/transport_meta_grid.rs) |
| Randomized experiment | Estimate a randomized ITT effect; design inputs named by data column | [python](python/experiment_randomized_effect.py) | — |
| Factorial cell means | 2x2 Bernoulli factorial main effects and interaction (point utility) | [python](python/factorial_cell_means.py) | — |
| Policy value | Held-out value of a fixed binary policy under known propensities | [python](python/policy_value_evaluation.py) | — |
| Difference-in-differences | 2x2 and balanced-panel DiD point utilities | [python](python/quasi_difference_in_differences.py) | — |
| Longitudinal regime value | Value of a static treatment rule by inverse-probability g-formula | [python](python/regime_value.py) | — |
| Survival RMST | Randomized survival curve and RMST by arm | [python](python/survival_rmst.py) | — |
| Decision lifecycle | Analyze, bind external evidence, inspect, decide, then rank the next study | [python](python/decision_lifecycle.py) | — |
| MSM sensitivity | Marginal-sensitivity bounds and tipping point for a stratified ATE | [python](python/msm_sensitivity.py) | — |
| Mechanism discrepancy | Test whether a node's mechanism differs between source and target | [python](python/mechanism_discrepancy.py) | — |
| Scenario invariance | Why each transport scenario answers as it does: selection and invariances | [python](python/scenario_invariance.py) | — |
| Transported counterfactual | Transport direct and indirect effects under a shared additive model; inspect refusals | [python](python/transported_counterfactual.py) | — |
| Recalculation with frozen scores | Recompute a decision with zero refits; resume from exported scores | [python](python/recalc_cell_resume.py) | — |
| Composition bundle | Export a composed decision and consume it under a retained identity | [python](python/composition_bundle.py) | — |
| Repair and evidence obligations | State the evidence a contract owes and which study would repair it | [python](python/repair_obligations.py) | — |
| Expected value of a study | Rank candidate studies by expected value of sample information | [python](python/rank_designs_evsi.py) | — |
| ML and transport lifecycle | Prepare, estimate, inspect and reload fitted models and transport results | [python](python/cohesive_ml_transport.py) | — |
| DoWhy handoff | Exchange a reviewed backdoor graph with DoWhy (optional dependency) | [python](python/dowhy_handoff.py) | — |
| EconML handoff | Compare adjusted average effects with EconML CATE predictions (optional dependency) | [python](python/econml_cate_handoff.py) | — |
| PCMCI pulse discovery | Discover a lagged graph and estimate a pulse effect in simulated data | [python](python/pcmci_pulse_benchmark.py) | — |
| ATE quickstart | Build and run an average-effect analysis | — | [rust](rust/ate_quickstart.rs) |
| Identify only | Identification without fitting | — | [rust](rust/identify_only.rs) |
| GCM do | Fit a GCM and sample under `do(·)` | — | [rust](rust/gcm_do.rs) |

The `transport_statistical`, `transport_exact` and `transport_meta_grid` scripts use `antecedent.transport.Transport`, which is absent from the 1.x releases (it arrived in 2.0). Callers moving 1.11 names should read the [transport migration](../docs/migrations/2.0-transport-day1.md). The scripts share one workflow and differ only in evidence. Rust uses the native `StudyBuilder` stage path; Python uses `analyze(Transport(...))`. Exact and complementary-source examples make no sampling-coverage claim.
