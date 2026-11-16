# `antecedent`

Antecedent is a compiler and evidence-aware runtime for causal science. This crate is its
Rust facade: the one dependency a Rust program needs.

You declare a causal question, the structure or study design you assume, and the data.
Antecedent checks whether those declarations identify the quantity you asked for, compiles an
analysis you can inspect before anything is fitted, and executes it. The answer comes back
with its identification, assumptions, uncertainty, calibration status and the published
methods behind it. A question the declarations cannot answer returns a typed refusal that
names what is missing, not a number.

<!-- generated:release:crate-dep:begin -->
```toml
[dependencies]
antecedent = "3.0"
```
<!-- generated:release:crate-dep:end -->

## A first study

```rust,ignore
use antecedent::prelude::*;

let schema = CausalSchemaBuilder::new()
    .continuous("t").treatment()
    .continuous("y").outcome()
    .continuous("z").context()
    .build()?;
// `t`, `y` and `z` are `&[f64]` columns of equal length.
let data = TabularData::from_f64_columns([("t", t), ("y", y), ("z", z)])?;
let dag = Dag::from_named_edges(&schema, &[("z", "t"), ("z", "y"), ("t", "y")])?;
let query = AverageEffectQuery::binary_ate(schema.id_of("t")?, schema.id_of("y")?);

let result = Study::tabular(data)
    .graph(dag)
    .query(query)
    .build()?
    .run(&ExecutionContext::production_default(1))?;
println!("{}", result.effect());
```

`Study::run` decides identification from the graph and the query before it fits anything,
then estimates; `.refute(...)` on the builder chooses the refuters run against the estimate.
`Study::prepare` returns a `PreparedStudy` that re-estimates on new rows without repeating
identification, and `StudyBuilder::identify_structure` decides identification alone.
The crate-level documentation walks through the lifecycle and lists every module.

## What a result carries

A route runs when it has provenance and an independent scientific check; otherwise a release
build returns a typed `not_supported` refusal that names the missing evidence. Every reported
interval carries its calibration status, and every result names the provenance records behind
the route it executed (`StudyResult::method_provenance()`, `antecedent::bibliography`), so
the published methods and their documented deviations travel with the answer.

## Features

The default `file-source` feature reads Parquet and Arrow IPC tables. `ml-gbdt`, `ml-forest`,
`ml-full` and `ml-neural` enable optional learner backends, and `ml-neural-wgpu` adds a WGPU
accelerator to the neural learner; they add estimators and grant no new scientific claims.

## Documentation

- [Documentation](https://antecedent.readthedocs.io/en/stable/): tutorials, guides, the
  support matrix and the refusal-code reference.
- [API reference](https://docs.rs/antecedent/) for this crate.
- The component crates (`antecedent-graph`, `antecedent-identify`, `antecedent-estimate` and
  others) are re-exported through this crate's modules; depend on them directly only for
  stage-specific work.

## License

MIT OR Apache-2.0.
