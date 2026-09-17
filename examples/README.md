# Examples

## Python environment

These examples use Antecedent 1.10 and Python 3.11 or later.
Install the package and notebook tools from PyPI:

```bash
python -m pip install antecedent pandas matplotlib ipykernel jupyterlab
```

Download an example notebook and open it with `jupyter lab`. If you have the
repository locally, run `jupyter lab examples/notebooks` from its root.
Choose the kernel for the Python environment where you installed Antecedent.

For Google Colab, run this in a setup cell:

```python
%pip install antecedent pandas matplotlib
```

If you upgrade Antecedent after importing it, restart the kernel or Colab session.

## Start with one analysis

Run the [first analysis](python/analysis_workflow.py):

```bash
python examples/python/analysis_workflow.py
```

It creates a simulated experiment with a treatment effect of 2, estimates that
effect, and then repeats the analysis on updated data. It also shows how to save
and reload the result. Read the [step-by-step guide](../docs/python-workflow.md)
for installation, expected output, and explanations.

Choose an example below by the question you want to answer. All datasets are
simulated. The assertions check the examples; they do not validate assumptions
for your own data.

## Notebooks

See Antecedent on a real decision. Open a notebook locally or in Google Colab
after the setup above. Run all cells to generate the tables, plots, and reports in your environment.

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

[The Python analysis workflow](python/analysis_workflow.py) demonstrates the 1.10 one-call
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
| ATE quickstart | Build and run an average-effect analysis | — | [rust](rust/ate_quickstart.rs) |
| Identify only | Identification without fitting | — | [rust](rust/identify_only.rs) |
| GCM do | Fit a GCM and sample under `do(·)` | — | [rust](rust/gcm_do.rs) |
