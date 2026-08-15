# Examples

## Notebooks

The fastest way to see Antecedent on a real decision. View them on GitHub, or
open in Google Colab from the instructions inside each notebook — no local
install required.

### [Paid-search attribution](notebooks/marketing_channel_structural_uncertainty.ipynb)

See how a naive marketing dashboard can materially overstate paid-search impact by crediting the campaign for demand that would have existed anyway. Antecedent adjusts for market demand and produces a decision-ready estimate of incremental pipeline.

### [Campaign evidence transfer](notebooks/sales_campaign_prior_transfer.ipynb)

Use evidence from a previous sales campaign without assuming the new campaign is identical. Antecedent transfers the historical treatment-effect posterior into a different target model, then lets current data update—or contradict—it.

### [Marketing experiment design](notebooks/marketing_experiment_design.ipynb)

Compare a holdout experiment, better intent data and additional CRM records to determine which investment actually resolves the causal question. Antecedent identifies the best feasible action under a £40,000 budget and shows why collecting more of the same data would not fix the attribution problem.

### [Continuous causal response](notebooks/continuous_causal_response.ipynb)

Estimate a nonlinear dose-response curve, local derivative, elasticity and
observed-law average derivative. Read structural identification, empirical
support and uncertainty as separate result axes.

### [Pricing, availability and latent demand](notebooks/pricing_availability_latent_demand.ipynb)

Show why inventory-limited sales are not demand. The notebook compares the
naive observed-sales response with an explicit censoring mechanism and
independence assumption, and demonstrates the current fail-closed boundary for
observation-aware response execution.

## Scripts

Paired Python and Rust demos for the same workflows.

```bash
# Python (from repo root; requires `maturin develop` in python/)
python examples/python/<name>.py

# Rust
cargo run -p antecedent --example <name>
```

| Example | Description | Python | Rust |
| ------- | ----------- | ------ | ---- |
| Propensity weighting | IPW ATE on confounded data with overlap diagnostics | [python](python/propensity_weighting.py) | [rust](rust/propensity_weighting.rs) |
| Manufacturing temporal | Pulse effect of pressure → defect on a temporal DAG | [python](python/manufacturing_temporal.py) | [rust](rust/manufacturing_temporal.rs) |
| Discover then estimate | Discover once, accept a DAG, re-estimate many times | [python](python/discover_then_estimate.py) | [rust](rust/discover_then_estimate.rs) |
| Sequential Bayes | Transfer a posterior artifact from batch A as batch B’s prior | [python](python/sequential_bayes.py) | [rust](rust/sequential_bayes.rs) |
| Prior bank surveys | Catalog → rank → compose external priors → target analysis | [python](python/prior_bank_surveys.py) | [rust](rust/prior_bank_surveys.rs) |
| Rank designs | Rank candidate experiments by identification probability | [python](python/rank_designs.py) | [rust](rust/rank_designs.rs) |
| CausalState workflow | Online append / stale queries / incremental OLS (ADR 0016) | [python](python/causal_state_workflow.py) | [rust](rust/causal_state_workflow.rs) |
| Sales spreadsheet E2E | Bayesian ATE + path decompose + ITE + temporal pulse | [python](python/sales_spreadsheet_e2e.py) | [rust](rust/sales_spreadsheet_e2e.rs) |
| ATE quickstart | Minimal static ATE builder → run | — | [rust](rust/ate_quickstart.rs) |
| Identify only | Identification without fitting | — | [rust](rust/identify_only.rs) |
| GCM do | Fit a GCM and sample under `do(·)` | — | [rust](rust/gcm_do.rs) |
