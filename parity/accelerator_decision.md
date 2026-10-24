# Accelerator decision

Status: the device route is implemented and instrumented. Promotion is decided by the
same-host staged run below, not by this text. No timing, speedup, or coverage figure is
recorded here until that run has produced `parity/accelerator_stage_timings.toml`.

## Candidates

Two backends are compiled behind opt-in features and neither is in a CPU wheel.

- `wgpu` links Burn 0.16's WGPU backend (Metal on Apple). The fit runs on the device, copies a
  scalar loss to the host every epoch, copies the fitted weights back, and predicts on the
  host from portable weights.
- `accelerate` keeps the `NdArray<f32>` backend and routes its matrix products through Apple
  Accelerate. It has no transfer and is the fallback approach when the device route fails the
  gain requirement.

## Instrumentation

`antecedent_learn_burn::staged` runs one fit, or a cross-fitted workload, on either backend
and reads a clock around each stage:

| Stage | What is timed |
| --- | --- |
| prepare | validate, standardize, narrow to `f32`, gather fold rows |
| upload | training tensors placed on the device, device drained |
| fit | module build, forward, backward, Adam step; device drained after each |
| synchronize | per-epoch scalar loss copied to the host |
| download | fitted weights copied to the host |
| predict | portable host forward on the held-out rows |
| output | held-out values written to physical row slots as `f64` |

The device is drained after each fit step, so queued kernels are billed to fit and the
synchronize stage is the host copy alone. The CPU control in `parity/cpu_control/protocol.toml`
is not repeated here.

## Harness

```text
cargo run --release -p antecedent-learn-burn --features wgpu \
  --example accelerator_stage_timing -- --repeats 5 --sweep \
  --out parity/accelerator_stage_timings.toml

cargo run --release -p antecedent-learn-burn --features accelerate \
  --example accelerator_stage_timing -- --repeats 5 --workloads small,control \
  --out parity/accelerator_stage_timings_accelerate.toml
```

Each `[[measurement]]` reports cold (first run, runtime and shader setup included) and warm
(per-stage median) clocks. Each `[[comparison]]` reports the end-to-end warm speedup of the
device over the CPU backend of the same build, the largest out-of-fold prediction gap in
target standard deviations, and the transfer and synchronize share. `[crossover]` names the
smallest workload where the device wins.

## Promotion rule

The device is promoted only if, on the same host and workload as the CPU control, the warm
end-to-end device time (transfer included) is at most half the CPU time
(`meets_gain_threshold = true`), the out-of-fold gap stays under the device tolerance in
`tests/accelerator_acceptance.rs`, and the downstream causal estimate and its standard error
agree (`scripts/distribution_acceptance.py all`). If the device fails the gain requirement the
`accelerate` run is measured against the same rule. A failure is recorded with its numbers and
the next approach is implemented; it is not recorded as a feasibility deferral.

## Public policy

`NeuralNet(device="cpu" | "gpu" | "auto")` (Rust `NeuralSpec::device`). `cpu` is the default,
never probes a device, and keeps the plain `burn` provenance label. `gpu` is refused with a
coded reason when the dtype is not `f32`, sample weights are present, the build does not link
the device runtime (`runtime_not_compiled`), no adapter is usable (`no_device`), or the device
account cannot hold the fit (`device_memory`). `auto` takes the device only when every check
passes and the fit is above the crossover (`policy::DEFAULT_MIN_DEVICE_WORK`, set from
`[crossover]`), otherwise it runs on the CPU and records the code as `fallback=` in the
learner provenance. The label names the library, device, dtype, and requested policy of the
route that ran, so CPU work is never labelled as device work. `ANTECEDENT_DISABLE_DEVICE=1`
hides the device from a device-capable build.

Seeded replay: CPU is bitwise on one host and build. Device is within the declared tolerance
on one adapter and driver, and not bitwise against the CPU.

## Resource and quality acceptance

`antecedent_learn_burn::admission` sizes a fit's host, scratch, and device bytes and admits it
against the shared CPU/RAM pool and the device pool in the order CPU then device; a refused
device request returns the CPU hold. Cancellation and unwinding return every hold. One CPU slot
is one concurrent fold. The acceptance tests are in
`crates/antecedent-learn-burn/tests/accelerator_acceptance.rs`: concurrent folds against the
slot limit, cancellation mid-fit, device out-of-memory refusal with CPU fallback, physical
prediction order, the linear-fit tolerance, the CPU-favoured small workload, and (with
`wgpu`) the measured small-workload CPU win and device agreement.

## Distribution

The CPU wheel is the standard wheel and links no device library. The accelerator distribution
is the same package built with `accelerator-wgpu` and labelled `+wgpu`.
`scripts/distribution_acceptance.py` builds both, checks the labels and linked libraries,
installs each in a clean virtual environment, checks the missing-runtime and hidden-device
refusals, provenance, the downstream effect and standard error, and that each wheel's exported
model is consumed by the other.
