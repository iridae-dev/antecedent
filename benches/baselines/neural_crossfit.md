# X7 cross-fitted neural workload, 2.1.0

Run `cargo run --release -q -p antecedent-learn --features ml-neural --example neural_crossfit_benchmark`.
The benchmark fixes 4,000 rows, eight covariates, three folds, and two nuisance
models (binary propensity and continuous outcome), each with 16 hidden units
and 20 epochs. It times fold assignment and each complete fit-and-predict pass,
including the Burn adapter's fold materialization and prediction transfer back
to physical row order. The synthetic design and seed are deterministic.

On the local arm64 host with Rust 1.85.1, one release-mode run measured:

| Stage | Time (ms) |
| --- | ---: |
| Fold assignment | 0.042 |
| Propensity, all folds | 85.163 |
| Outcome, all folds | 65.592 |
| End to end | 150.797 |

The shipping adapter is Burn `NdArray<f32>` on CPU. There is no GPU provider or
host-to-device transfer in 2.1.0, so the benchmark does not establish an
accelerator speedup. A future opt-in backend must beat this full workload,
including fold scheduling and transfer, before its distribution and replay
gates can be considered. This is a timing baseline, not a causal coverage test.
