# Start with Rust

From the source checkout, run the smallest effect-estimation example:

```bash
cargo run -p antecedent --example ate_quickstart
```

The repository selects Rust 1.85 through `rust-toolchain.toml`.
The example creates a table, declares the causal graph, and asks for the average
treatment effect. Its simulated outcome increases by **2** when treatment changes
from 0 to 1, holding the baseline variable fixed. Expect `effect = 2.0000`.

Read the [complete source](https://github.com/iridae-dev/antecedent/blob/v2.0.0/examples/rust/ate_quickstart.rs)
to see how `Study::tabular` connects the data, graph, and question.
This small example disables refutation and bootstrap intervals and uses a
parametric model; it does not establish overlap for a real dataset.

Build the API reference from the same checkout:

```bash
cargo doc -p antecedent --open
```

The [published Rust reference](https://docs.rs/antecedent) describes the published
crate. Select the crate version that matches this release. Next, [choose a worked example](examples.md)
or read about [supported analyses](supported-analyses.md).
