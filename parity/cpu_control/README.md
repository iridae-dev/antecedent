# G01 CPU control

This directory registers the CPU control for the current 3.0 neural-crossfit candidate. `protocol.toml` is the plan. It is not a measurement. Status is `registered_not_measured`. No host, commit, backend observation, dtype, stage time, or cold/warm state is stored here.

The workload is fold planning, propensity fit, outcome fit, prediction, and restoration to physical row order. A finished record fills host, commit, backend, dtype, stage times, and cold/warm from the run. The historical 2.1 timing in `benches/baselines/neural_crossfit.md` is not this control.

A comparison repeats the same command on the exact host and revision used for that comparison.

## Command

From the repository root:

```bash
cargo run --release -q -p antecedent-learn --features ml-neural --example neural_crossfit_benchmark
```

The example is `crates/antecedent-learn/examples/neural_crossfit_benchmark.rs`. It prints `fold_plan_ms`, `propensity_fit_predict_ms`, `outcome_fit_predict_ms`, and `total_ms`. Prediction and restoration to physical row order run inside the two fit-and-predict passes. The example prints `backend=burn_ndarray_cpu`. It does not print host, commit, dtype, or cold/warm; those stay part of the record the operator fills.

## G03 threshold

The proposed G03 material-gain threshold is 2× complete-workload gain. That figure is a proposal. It stays unfrozen until a CPU record of this control exists, and it is frozen only after that characterization and before any GPU result is examined. A GPU result cannot freeze or move the threshold.
