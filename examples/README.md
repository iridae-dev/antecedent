# Examples

## Python environment (1.10 branch)

These Python scripts and all five notebooks use the **1.10 Python API**.
Build the `1.10.0` checkout; older published wheels do not have `prepare`,
`load`, or `result.study`. The notebook setup checks the API and will not
silently install an older release over a development build.

With the repository's Rust toolchain installed, from the repository root:

```bash
python -m venv python/.venv
source python/.venv/bin/activate
python -m pip install maturin numpy pandas matplotlib ipykernel jupyterlab
maturin develop --release --manifest-path python/Cargo.toml
python -m ipykernel install --sys-prefix --name antecedent --display-name "Antecedent 1.10"
jupyter lab examples/notebooks
```

Choose the **Antecedent 1.10** kernel. For Google Colab, first install a Rust
toolchain that supports this repository (see [development](../docs/development.md)),
then install the branch from source in a setup cell:

```python
%pip install "git+https://github.com/iridae-dev/antecedent.git@1.10.0#subdirectory=python" numpy pandas matplotlib
```

Restart the Colab session after installation if Antecedent was already imported.
The branch reference follows development; record the git commit or pin the source
URL to a commit for a reproducible run. Building from source can take several minutes.

## Shared Python workflow

```python
import antecedent as ant

result = ant.analyze(data, graph=graph, query=query)
study = result.study
report = result.inspect().to_dict()
updated = study.refresh(new_data)
loaded = ant.load(result.export())
```

Use `ant.prepare(...)` followed by `study.estimate()` when you want to stop
before estimation. Keep ordinary notebook analyses as a single `analyze(...)`
call. Reports preserve answer shape, uncertainty, assumptions, support and
calibration availability. See the [workflow guide](../docs/python-workflow.md)
for prepared-route boundaries and descriptive refusal reports.

The Bayesian transfer examples use `result.study.export_artifact()` to obtain
the posterior payload required by `Bayesian(prior_from=...)`. Full execution
archives use `result.export()` and `ant.load(...)`; they serve a different purpose.
Design ranking and incremental `CausalState` keep their stage APIs.

## Notebooks

See Antecedent on a real decision. Open a notebook locally or in Google Colab
after the source-build setup above. The saved outputs were regenerated with this branch; run all cells to reproduce
the tables, plots and execution reports in your environment.

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
naive observed-sales slope with a demand `ResponseCurve` under Cox IPCW, then
shows that an observation-adjusted demand derivative still fails closed.

## Scripts

Paired Python and Rust demos for the same workflows.

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
| Propensity weighting | IPW ATE on confounded data with overlap diagnostics | [python](python/propensity_weighting.py) | [rust](rust/propensity_weighting.rs) |
| Staged static kinds | Natural mediation and unit ITE on a confounded linear SCM | [python](python/staged_static_kinds.py) | [rust](rust/staged_static_kinds.rs) |
| Class-preserving CPDAG | MEC-envelope ATE on a supplied `Cpdag`; oriented Cpdag handoff | [python](python/class_preserving_cpdag.py) | [rust](rust/class_preserving_cpdag.rs) |
| Manufacturing temporal | Pulse effect of pressure → defect on a temporal DAG | [python](python/manufacturing_temporal.py) | [rust](rust/manufacturing_temporal.rs) |
| Temporal response curve | Dose × horizon ``ResponseCurve`` and intervention path on a temporal DAG | [python](python/temporal_response_curve.py) | [rust](rust/temporal_response_curve.rs) |
| Discover then estimate | Discover once, accept a DAG, re-estimate many times | [python](python/discover_then_estimate.py) | [rust](rust/discover_then_estimate.rs) |
| Sequential Bayes | Transfer a posterior artifact from batch A as batch B’s prior | [python](python/sequential_bayes.py) | [rust](rust/sequential_bayes.rs) |
| Prior bank surveys | Catalog → rank → compose external priors → target analysis | [python](python/prior_bank_surveys.py) | [rust](rust/prior_bank_surveys.rs) |
| Rank designs | Rank candidate experiments by identification probability | [python](python/rank_designs.py) | [rust](rust/rank_designs.rs) |
| CausalState workflow | Online append / stale queries / incremental OLS (ADR 0016) | [python](python/causal_state_workflow.py) | [rust](rust/causal_state_workflow.rs) |
| Sales spreadsheet E2E | Bayesian ATE + path decompose + ITE + temporal pulse | [python](python/sales_spreadsheet_e2e.py) | [rust](rust/sales_spreadsheet_e2e.rs) |
| ATE quickstart | Minimal static ATE builder → run | — | [rust](rust/ate_quickstart.rs) |
| Identify only | Identification without fitting | — | [rust](rust/identify_only.rs) |
| GCM do | Fit a GCM and sample under `do(·)` | — | [rust](rust/gcm_do.rs) |
