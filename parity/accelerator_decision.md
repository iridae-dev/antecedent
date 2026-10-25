# Accelerator decision

Status: decided from the same-host staged run recorded in
`parity/accelerator_stage_timings.toml` (device against the `NdArray<f32>` CPU backend) and
`parity/accelerator_stage_timings_accelerate.toml` (Accelerate CPU backend). Host: Apple M1
Max, Metal, five repeats per workload (one cold, four warm, median). The host was not idle:
the 1-minute load average was about 23 during the timed runs (a long-running test binary and
desktop applications), while the CPU was about 65-78 percent idle, so absolute times carry
that background noise; the device was not contended by other GPU work.

## Decision

Promote the device for fits whose work is at or above `policy::DEFAULT_MIN_DEVICE_WORK`
= 37,094,400,000 `fit_flops` (per fit: about 20,000 training rows, 32 columns, hidden width
128, 15 epochs). `auto` runs the CPU below that. `gpu` is always honoured. The threshold is the
per-fit work of the smallest measured workload whose warm end-to-end device time (transfers
included) is at most half the CPU time with the out-of-fold gap inside tolerance.

The threshold is the lowest measured point that clears the rule, not an interpolation: the
hidden-width sweep reaches 1.91x at 1.7e10 per-fit flops and 1.73x at 2.4e10, and the
`large` shape is the first to clear 2x.

## Measured results

Out-of-fold gap is the largest CPU-versus-device difference in target standard deviations; the
tolerance is 0.05.

| Workload | Total flops | CPU ms | Device ms | Speedup | Gap (SD) | 2x rule |
| --- | --- | --- | --- | --- | --- | --- |
| small | 1.28e6 | 3.0 | 82.4 | 0.04 | 0 | no |
| control | 3.84e8 | 59.4 | 669.3 | 0.09 | 1e-6 | no |
| sweep_h008 | 2.40e8 | 65.6 | 268.8 | 0.24 | 0 | no |
| sweep_h016 | 6.34e8 | 83.1 | 263.7 | 0.32 | 0 | no |
| sweep_h032 | 1.88e9 | 115.1 | 274.3 | 0.42 | 1e-6 | no |
| sweep_h064 | 6.22e9 | 185.9 | 283.9 | 0.65 | 0.012 | no |
| sweep_h128 | 2.23e10 | 377.8 | 253.9 | 1.49 | 0.009 | no |
| sweep_h160 | 3.40e10 | 515.1 | 270.2 | 1.91 | 1e-6 | no |
| sweep_h192 | 4.82e10 | 626.1 | 362.7 | 1.73 | 0.015 | no |
| large | 7.42e10 | 1301.7 | 575.3 | 2.26 | 7.5e-5 | yes |
| sweep_h256 | 8.4e10 | 921.0 | 413.8 | 2.23 | 0.029 | yes |

(The `large` row is 2.60x in a first run and 2.26x in the second, h256 2.15x and 2.23x: the
2x rule is met in both, with run-to-run spread of about 15 percent under background load.)

The device pays a fixed cost of roughly 250 ms per workload (two folds) regardless of size
that the CPU does not, so it loses everywhere below about 2e10 flops and never wins on the
CPU control workload.

## What was wrong with the earlier fits

An earlier run recorded a 0.8-1.7 SD gap between CPU and device fits. That was not float noise:
each backend initialised the weights from its own RNG (`B::seed` is backend-specific), so the
two fits trained different networks. The weights are now drawn on the host from the seed
(SplitMix64, uniform on plus or minus 1/sqrt(fan_in), the same distribution as Burn's default
initializer) and uploaded, and the fit is full-batch so there is no sample order to differ.
The gap is 1e-6 SD on the small and control workloads and at most 0.029 SD (hidden 256) on
the largest, from f32 accumulation order. `tests/device_cpu_agreement.rs` asserts this on the
device. Changing the initial weights changes the values of seeded CPU fits relative to
earlier releases; they remain bitwise reproducible on one host and build.

## Evaluated next backend: Accelerate

`NdArray<f32>` with Apple Accelerate BLAS has no transfer cost but gains little. Warm wall
times against the plain ndarray backend: small 3.34 ms (3.0 plain), control 54.6 ms (59.4
plain, 1.09x), large 973.5 ms (1301.7 plain, 1.34x). It never reaches 2x over the CPU
control, so it is not promoted as a route; it is the evaluated alternative for CPU-only
accelerator builds and the device is 1.7x faster than it on the `large` workload.

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
smallest workload where the device wins and the smallest that meets the gain threshold with
the gap inside tolerance.

## Promotion rule

The device is promoted only if, on the same host and workload as the CPU control, the warm
end-to-end device time (transfer included) is at most half the CPU time
(`meets_gain_threshold = true`), the out-of-fold gap stays under the device tolerance in
`tests/accelerator_acceptance.rs`, and the downstream causal estimate and its standard error
agree (`scripts/distribution_acceptance.py all`). If the device fails the gain requirement the
`accelerate` run is measured against the same rule. A failure is recorded with its numbers and
the next approach is implemented; it is not recorded as a feasibility deferral. The device
met the rule at the workloads above, so the fused-step and half-precision follow-ups were not
needed.

## Public policy

`NeuralNet(device="cpu" | "gpu" | "auto")` (Rust `NeuralSpec::device`). `cpu` is the default,
never probes a device, and keeps the plain `burn` provenance label. `gpu` is refused with a
coded reason when the dtype is not `f32`, sample weights are present, the build does not link
the device runtime (`runtime_not_compiled`), no adapter is usable (`no_device`), or the device
account cannot hold the fit (`device_memory`). `auto` takes the device only when every check
passes and the fit is at or above the crossover (`policy::DEFAULT_MIN_DEVICE_WORK`, set from
`smallest_gain_threshold_per_fit_flops` in `[crossover]`), otherwise it runs on the CPU and records the code as `fallback=` in the
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
