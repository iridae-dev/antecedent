# 2.3 C0 X7 accelerator decision

X7 remains deferred. No accelerator row is licensed, and the CPU wheel remains the available backend. `antecedent-learn`'s `ml-neural` feature selects only the Burn `NdArray<f32>` CPU adapter; the repository has no GPU/WGPU backend or host-device transfer path to compare against it. A material whole-workload GPU gain has therefore not been measured. This is an unmet start condition, not a negative GPU performance result.

On 2026-10-06, at `4fb13e25` plus documentation-only C0 changes, the representative existing benchmark ran with:

```text
cargo run --release -q -p antecedent-learn --features ml-neural --example neural_crossfit_benchmark
rows=4000 cols=8 folds=3 hidden=16 epochs=20 fold_plan_ms=0.127 propensity_fit_predict_ms=120.727 outcome_fit_predict_ms=265.609 total_ms=386.463
backend=burn_ndarray_cpu gpu_transfer_ms=unmeasured gpu_end_to_end_gain=unmeasured
```

The run includes fold planning and complete fold fit/predict passes in the CPU adapter. It is a single development-host timing, not a controlled comparison with the 2.1 baseline's 150.797 ms run, whose measured commit and exact chip were not recorded. The current host reports an Apple M1 Max with a Metal-capable GPU, but Antecedent has no backend using that device. X7 may start only after a candidate backend measures end-to-end gain including transfer, fold orchestration, prediction ordering and repeatability against a same-host CPU run, followed by its separate distribution and replay gates. No other 2.3 cell depends on opening X7.
