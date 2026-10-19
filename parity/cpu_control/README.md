# G01 CPU control

This directory records the CPU control for the 3.0 neural-crossfit candidate. Status is `measured`. The record is `protocol.toml`.

The measured commit is `420bab39471f7881abf2696c80540b9ccb576c07` on an Apple M1 Max (arm64, macOS 26.5.2, 10 threads). The backend is `burn_ndarray_cpu`. The design and the predictions are `f64`. Training and inference run on `NdArray<f32>`.

Cold is the first execution after the release build: fold planning 0.061 ms, propensity fit-and-predict 72.613 ms, outcome fit-and-predict 67.373 ms, total 140.046 ms. Warm is a later execution of that binary: 0.019 ms, 75.390 ms, 67.293 ms, total 142.703 ms. Prediction and restoration to physical row order are inside the two fit-and-predict timers. The historical 2.1 timing in `benches/baselines/neural_crossfit.md` is not this control.

## Command

From the repository root:

```bash
cargo run --release -q -p antecedent-learn --features ml-neural --example neural_crossfit_benchmark
```

A comparison repeats that command on the exact host and revision used for the comparison.

## G03 threshold

The material-gain threshold is 2× complete-workload gain. It is frozen from this CPU record, before any GPU result. A GPU result cannot move it.
