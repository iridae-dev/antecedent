# antecedent

Antecedent is an identification-first causal inference engine. This crate is
the Rust facade: it takes an analysis from causal structure through
identification, estimation, and refutation — for static and temporal queries,
with logical/physical planning — without silently treating discovered graphs
as ground truth.

Use it when you need a causal effect you can defend: identification is
evaluated before anything is estimated, refuters run against the estimate by
default, and uncertainty about the causal graph (equivalence classes, graph
posteriors) is carried through to the effect instead of being resolved by fiat.

```toml
antecedent = "0.7"
```

The entry point is `Study::tabular()` (or `::series` / `::series_multi` /
`::panel` / `::events` for temporal data):

```rust
use antecedent::prelude::*;

let result = Study::tabular(data)
    .graph(dag)
    .query(query)
    .build()?
    .run(&ctx)?;

// result.effect(), plus identification status, assumptions,
// refutation outcomes, and provenance.
```

A complete compiling example (schema → data → DAG → query → run) is in the
[crate docs](https://docs.rs/antecedent); runnable examples live in
[`examples/`](https://github.com/iridae-dev/antecedent/tree/main/examples)
(`cargo run -p antecedent --example <name>`).
The same engine powers the Python package
([`pip install antecedent`](https://pypi.org/project/antecedent/)).

Beyond effect estimation, the facade covers causal discovery (PC, GES, LiNGAM,
NOTEARS, FCI/RFCI, the PCMCI family, Bayesian graph posteriors), continuous
causal responses (mean curves, derivatives, elasticities), interventional
sampling and counterfactuals via structural causal models, attribution and
root-cause analysis, sensitivity analysis, experimental-design ranking, and
incremental causal state for online workflows. Observation mechanisms,
transport, and interference are explicit stage APIs, not `analyze` flags.

Supporting libraries are **`antecedent-*`** crates on crates.io and are
**public dependencies** of this facade. The Python extension
(`antecedent-py`) is not published to crates.io.

See the [workspace root README](https://github.com/iridae-dev/antecedent#readme),
[docs/capabilities.md](https://github.com/iridae-dev/antecedent/blob/main/docs/capabilities.md),
and `CHANGELOG.md`. Pre-1.0: minor versions may introduce breaking changes
(see [ROADMAP.md](https://github.com/iridae-dev/antecedent/blob/main/ROADMAP.md)).
