# Choose an example

Start with the [Python quickstart](python-workflow.md) or [Rust quickstart](rust-quickstart.md). For notebooks, follow the [notebook setup](../examples/README.md#python-environment). All examples use simulated data.

## Notebooks

See Antecedent on a real decision. Open a notebook locally or in Google Colab after setup. Run all cells to generate the tables, plots, and reports in your environment.

### [Paid-search attribution](../examples/notebooks/marketing_channel_structural_uncertainty.ipynb)

See how a naive marketing dashboard can materially overstate paid-search impact by crediting the campaign for demand that would have existed anyway. Compare results under uncertain graph directions with an estimate conditional on a reviewed graph.

### [Campaign evidence transfer](../examples/notebooks/sales_campaign_prior_transfer.ipynb)

Use evidence from a previous sales campaign without assuming the new campaign is identical. Antecedent transfers the historical treatment-effect posterior into a different target model, then lets current data update—or contradict—it.

### [Marketing experiment design](../examples/notebooks/marketing_experiment_design.ipynb)

Compare a holdout experiment, better intent data and additional CRM records to determine which investment actually resolves the causal question. Antecedent identifies the best feasible action under a £40,000 budget and shows why collecting more of the same data would not fix the attribution problem.

### [Continuous causal response](../examples/notebooks/continuous_causal_response.ipynb)

Ask how the outcome changes with dose, how steep the curve is, and how that slope changes across the data. Check where the data support those answers.

### [Pricing, availability and latent demand](../examples/notebooks/pricing_availability_latent_demand.ipynb)

Show why inventory-limited sales are not demand. Estimate a demand curve under an explicit censoring assumption, then see why the same method refuses a demand derivative.

## Python scripts

Choose Python or Rust for each workflow.

For moving a reviewed backdoor graph between Antecedent and DoWhy without wrapping either library, see the [DoWhy interoperability cookbook](interop_dowhy.md) and [`dowhy_handoff.py`](../examples/python/dowhy_handoff.py) (`dowhy` optional).

[The Python analysis workflow](../examples/python/analysis_workflow.py) demonstrates the one-call API, retained study, inspection, refresh, and verified artifact loading.

```bash
# Python (from repo root, with your environment activated)
python examples/python/<name>.py

# Rust
cargo run -p antecedent --example <name>
```

| Example | Description | Python | Rust |
| ------- | ----------- | ------ | ---- |
| Propensity weighting | Adjust for treatment selection using inverse-probability weights | [python](../examples/python/propensity_weighting.py) | [rust](../examples/rust/propensity_weighting.rs) |
| Mediation and individual effects | Separate direct and mediated effects; estimate individual effects | [python](../examples/python/staged_static_kinds.py) | [rust](../examples/rust/staged_static_kinds.rs) |
| Uncertain graph directions | Estimate a range of effects when some edge directions are unknown | [python](../examples/python/class_preserving_cpdag.py) | [rust](../examples/rust/class_preserving_cpdag.rs) |
| Manufacturing temporal | Estimate how a pressure change affects later defects | [python](../examples/python/manufacturing_temporal.py) | [rust](../examples/rust/manufacturing_temporal.rs) |
| Temporal response curve | Compare pressure levels and their effects over time | [python](../examples/python/temporal_response_curve.py) | [rust](../examples/rust/temporal_response_curve.rs) |
| Discover then estimate | Discover once, accept a DAG, re-estimate many times | [python](../examples/python/discover_then_estimate.py) | [rust](../examples/rust/discover_then_estimate.rs) |
| Sequential Bayes | Use one batch’s results as prior evidence for the next | [python](../examples/python/sequential_bayes.py) | [rust](../examples/rust/sequential_bayes.rs) |
| Prior bank surveys | Select and combine evidence from earlier surveys | [python](../examples/python/prior_bank_surveys.py) | [rust](../examples/rust/prior_bank_surveys.rs) |
| Rank designs | Rank candidate experiments by identification probability | [python](../examples/python/rank_designs.py) | [rust](../examples/rust/rank_designs.rs) |
| CausalState workflow | Update an analysis as data arrive and identify outdated results | [python](../examples/python/causal_state_workflow.py) | [rust](../examples/rust/causal_state_workflow.rs) |
| Sales analysis | Explore average, mediated, individual, and delayed effects | [python](../examples/python/sales_spreadsheet_e2e.py) | [rust](../examples/rust/sales_spreadsheet_e2e.rs) |
| Transport a source table | Single-source empirical transport of a response grid | [python](../examples/python/transport_statistical.py) | [rust](../examples/rust/transport_statistical.rs) |
| Transport an exact law | Single-source exact-law transport of a response grid | [python](../examples/python/transport_exact.py) | [rust](../examples/rust/transport_exact.rs) |
| Complementary-source grid | Combined sources identify a target curve; one source does not | [python](../examples/python/transport_meta_grid.py) | [rust](../examples/rust/transport_meta_grid.rs) |
| ATE quickstart | Build and run an average-effect analysis | — | [rust](../examples/rust/ate_quickstart.rs) |
| Identify only | Identification without fitting | — | [rust](../examples/rust/identify_only.rs) |
| GCM do | Fit a GCM and sample under `do(·)` | — | [rust](../examples/rust/gcm_do.rs) |
