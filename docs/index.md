# Antecedent

Antecedent is an identification-first causal inference engine for **Python** and
**Rust**. It takes an analysis from causal structure through estimation,
diagnostics, interventions, and counterfactuals — without silently treating
discovered graphs as ground truth.

Three rules are enforced throughout:

* identification is evaluated before estimation;
* priors and parametric assumptions do not upgrade nonparametric identification;
* uncertainty about causal structure is retained rather than silently resolved.

## What you would use it for

* **Estimate an effect you can defend.** `analyze()` checks identification
  first, reports the strategy and adjustment set it used, and runs refuters
  against the estimate by default.
* **Work with discovered structure honestly.** Discovery returns equivalence
  classes and graph posteriors, not a single guessed DAG. Estimation refuses to
  run on an unreviewed partial graph; the Bayesian path reports how much
  posterior mass sits on structures where the effect is unidentified.
* **Analyze temporal systems.** Temporal graphs with lagged edges, PCMCI-family
  discovery, pulse and sustained interventions, temporal mediation, and
  incremental `CausalState` for online workflows.
* **Go past effect estimates.** Interventional sampling, counterfactuals,
  root-cause and distribution-change attribution, sensitivity analysis, and
  experimental-design ranking, all in the same engine.

## Getting started

`pip install antecedent`, then start from the runnable examples in the
[project README](https://github.com/iridae-dev/antecedent#readme) — each is a
single paste-and-run block with its real output shown. The Rust entry point is
`Study::tabular()` (or `::series` / `::series_multi` / `::panel` / `::events`) in the
[`antecedent` crate](https://docs.rs/antecedent).

## Guides

| Doc | Contents |
|-----|----------|
| [Causal responses](causal-responses.md) | Curves, derivatives, support, uncertainty, observation mechanisms |
| [Transport and interference](transport-interference.md) | Structural transport, trial generalization, randomized network exposure |
| [Capabilities](capabilities.md) | Full inventory: graphs, discovery, identification, estimation, validation, design |
| [Comparison](comparison.md) | Antecedent vs. DoWhy, EconML, Tigramite, causal-learn — and when to use each |
| [Architecture](architecture.md) | Invariants, crates, analysis pipeline, execution model |
| [Development](development.md) | CI vs local gates, tests, performance rules, versions |
| [Artifacts](artifacts.md) | Wire format, migration, graph interchange |
| [Prior bank](priors.md) | External prior catalog, compose, conflict, transport |
| [API naming](api_naming.md) | Rust ↔ Python capability dictionary |
| [Hot paths](hot_paths.md) | Benches, baselines, allocation contracts |
| [Conformance](conformance/README.md) | Generated from `conformance/` fixtures |
| [Security review](security_review.md) | Unsafe, deps, licensing evidence |

## API reference

- **Python:** [Python API](python-api.md) on this site (`/python/` via pdoc; no download)
- **Rust:** [docs.rs/antecedent](https://docs.rs/antecedent); locally `cargo doc -p antecedent --open`

Decisions: see `adr/` in the repository.

Regenerate conformance pages:

```bash
python3 scripts/generate_conformance_docs.py
```
