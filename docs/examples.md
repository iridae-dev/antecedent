# Choose an example

Start with the [Python quickstart](python-workflow.md) or [Rust quickstart](rust-quickstart.md).
For notebooks, follow the [notebook setup](https://github.com/iridae-dev/antecedent/blob/1.11/examples/README.md#python-environment).
All examples use simulated data.

## Notebooks

See Antecedent on a real decision. Open a notebook locally or in Google Colab
after setup. Run all cells to generate the tables, plots, and reports in your environment.

### [Paid-search attribution](https://github.com/iridae-dev/antecedent/blob/1.11/examples/notebooks/marketing_channel_structural_uncertainty.ipynb)

See how a naive marketing dashboard can materially overstate paid-search impact by crediting the campaign for demand that would have existed anyway. Compare results under uncertain graph directions with an estimate conditional on a reviewed graph.

### [Campaign evidence transfer](https://github.com/iridae-dev/antecedent/blob/1.11/examples/notebooks/sales_campaign_prior_transfer.ipynb)

Use evidence from a previous sales campaign without assuming the new campaign is identical. Antecedent transfers the historical treatment-effect posterior into a different target model, then lets current data update—or contradict—it.

### [Marketing experiment design](https://github.com/iridae-dev/antecedent/blob/1.11/examples/notebooks/marketing_experiment_design.ipynb)

Compare a holdout experiment, better intent data and additional CRM records to determine which investment actually resolves the causal question. Antecedent identifies the best feasible action under a £40,000 budget and shows why collecting more of the same data would not fix the attribution problem.

### [Continuous causal response](https://github.com/iridae-dev/antecedent/blob/1.11/examples/notebooks/continuous_causal_response.ipynb)

Ask how the outcome changes with dose, how steep the curve is, and how that
slope changes across the data. Check where the data support those answers.

### [Pricing, availability and latent demand](https://github.com/iridae-dev/antecedent/blob/1.11/examples/notebooks/pricing_availability_latent_demand.ipynb)

Show why inventory-limited sales are not demand. Estimate a demand curve under an explicit censoring assumption, then see why
the same method refuses a demand derivative.

## Python scripts

Choose Python or Rust for each workflow.

[The Python analysis workflow](https://github.com/iridae-dev/antecedent/blob/1.11/examples/python/analysis_workflow.py) demonstrates the one-call
API, retained study, inspection, refresh, and verified artifact loading.

```bash
# Python (from repo root, with your environment activated)
python examples/python/<name>.py

# Rust
cargo run -p antecedent --example <name>
```

| Example | Description | Python | Rust |
| ------- | ----------- | ------ | ---- |
| Propensity weighting | Adjust for treatment selection using inverse-probability weights | [python](https://github.com/iridae-dev/antecedent/blob/1.11/examples/python/propensity_weighting.py) | [rust](https://github.com/iridae-dev/antecedent/blob/1.11/examples/rust/propensity_weighting.rs) |
| Mediation and individual effects | Separate direct and mediated effects; estimate individual effects | [python](https://github.com/iridae-dev/antecedent/blob/1.11/examples/python/staged_static_kinds.py) | [rust](https://github.com/iridae-dev/antecedent/blob/1.11/examples/rust/staged_static_kinds.rs) |
| Uncertain graph directions | Estimate a range of effects when some edge directions are unknown | [python](https://github.com/iridae-dev/antecedent/blob/1.11/examples/python/class_preserving_cpdag.py) | [rust](https://github.com/iridae-dev/antecedent/blob/1.11/examples/rust/class_preserving_cpdag.rs) |
| Manufacturing temporal | Estimate how a pressure change affects later defects | [python](https://github.com/iridae-dev/antecedent/blob/1.11/examples/python/manufacturing_temporal.py) | [rust](https://github.com/iridae-dev/antecedent/blob/1.11/examples/rust/manufacturing_temporal.rs) |
| Temporal response curve | Compare pressure levels and their effects over time | [python](https://github.com/iridae-dev/antecedent/blob/1.11/examples/python/temporal_response_curve.py) | [rust](https://github.com/iridae-dev/antecedent/blob/1.11/examples/rust/temporal_response_curve.rs) |
| Discover then estimate | Discover once, accept a DAG, re-estimate many times | [python](https://github.com/iridae-dev/antecedent/blob/1.11/examples/python/discover_then_estimate.py) | [rust](https://github.com/iridae-dev/antecedent/blob/1.11/examples/rust/discover_then_estimate.rs) |
| Sequential Bayes | Use one batch’s results as prior evidence for the next | [python](https://github.com/iridae-dev/antecedent/blob/1.11/examples/python/sequential_bayes.py) | [rust](https://github.com/iridae-dev/antecedent/blob/1.11/examples/rust/sequential_bayes.rs) |
| Prior bank surveys | Select and combine evidence from earlier surveys | [python](https://github.com/iridae-dev/antecedent/blob/1.11/examples/python/prior_bank_surveys.py) | [rust](https://github.com/iridae-dev/antecedent/blob/1.11/examples/rust/prior_bank_surveys.rs) |
| Rank designs | Rank candidate experiments by identification probability | [python](https://github.com/iridae-dev/antecedent/blob/1.11/examples/python/rank_designs.py) | [rust](https://github.com/iridae-dev/antecedent/blob/1.11/examples/rust/rank_designs.rs) |
| CausalState workflow | Update an analysis as data arrive and identify outdated results | [python](https://github.com/iridae-dev/antecedent/blob/1.11/examples/python/causal_state_workflow.py) | [rust](https://github.com/iridae-dev/antecedent/blob/1.11/examples/rust/causal_state_workflow.rs) |
| Sales analysis | Explore average, mediated, individual, and delayed effects | [python](https://github.com/iridae-dev/antecedent/blob/1.11/examples/python/sales_spreadsheet_e2e.py) | [rust](https://github.com/iridae-dev/antecedent/blob/1.11/examples/rust/sales_spreadsheet_e2e.rs) |
| Transport a source table | Single-source empirical transport of a response grid | [python](https://github.com/iridae-dev/antecedent/blob/1.11/examples/python/transport_statistical.py) | [rust](https://github.com/iridae-dev/antecedent/blob/1.11/examples/rust/transport_statistical.rs) |
| Transport an exact law | Single-source exact-law transport of a response grid | [python](https://github.com/iridae-dev/antecedent/blob/1.11/examples/python/transport_exact.py) | [rust](https://github.com/iridae-dev/antecedent/blob/1.11/examples/rust/transport_exact.rs) |
| Complementary-source grid | Combined sources identify a target curve; one source does not | [python](https://github.com/iridae-dev/antecedent/blob/1.11/examples/python/transport_meta_grid.py) | [rust](https://github.com/iridae-dev/antecedent/blob/1.11/examples/rust/transport_meta_grid.rs) |
| ATE quickstart | Build and run an average-effect analysis | — | [rust](https://github.com/iridae-dev/antecedent/blob/1.11/examples/rust/ate_quickstart.rs) |
| Identify only | Identification without fitting | — | [rust](https://github.com/iridae-dev/antecedent/blob/1.11/examples/rust/identify_only.rs) |
| GCM do | Fit a GCM and sample under `do(·)` | — | [rust](https://github.com/iridae-dev/antecedent/blob/1.11/examples/rust/gcm_do.rs) |
