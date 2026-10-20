# Accelerator decision

G03 stays deferred. The opt-in `wgpu` feature compiles Burn's WGPU path beside the default `NdArray<f32>` wheel, and it is not promoted. `probe_wgpu_small_fold` returns `NotLinked::StageBoundariesPrivate` and does not read a clock: `fit_portable_wgpu` fuses prepare, upload, fit, synchronize, and download, so those stages are not separately public. No same-host staged comparison against the frozen threshold was taken. This is a feasibility deferral, not a negative GPU result. The 2× threshold was not applied. A release small-fold CPU probe is not that comparison.

G04 closes on that failed promotion. A public `cpu` / `gpu` / `auto` policy ships only after a positive G03 gain decision, and that decision did not happen. No measurement closed this gate. There is no public GPU backend and no policy that can select an accelerator. `auto` is not a selector, so it cannot label CPU work as GPU work.

G05 is the refusal already in the tree, and it is not a promotion. `select_compute(ComputeSelection::Accelerator)` returns `DeviceReservationError::Unpromoted` and reserves no device bytes. The test is `unpromoted_accelerator_is_refused` in `crates/antecedent-core/src/device_reservation/cpu_admission.rs`. That is the honest test of accelerator selection, so it is named here and not copied. `select_compute(ComputeSelection::Cpu)` returns `CpuAdmission`, a value with no device label. `small_serial_job_reserves_no_device` in the same module shows that CPU admission leaves the device account unchanged. A CPU result on this path is not recorded as GPU work.

G06 closes the public GPU route and retains the CPU path. Promotion required G01–G06 to pass. G03 did not produce a gain decision, so the route stays closed on that failed promotion. The public refusal remains `DeviceReservationError::Unpromoted`. The opt-in `wgpu` feature stays unpromoted. This record does not change the CPU control, the small-fold clocks below, or any core claim, and it supplies no GPU timing, speedup, or coverage rate.

The frozen threshold is 2× the G01 CPU control in `parity/cpu_control/protocol.toml`: commit `420bab394`, Apple M1 Max, backend `burn_ndarray_cpu`, cold total 140.046 ms, warm total 142.703 ms. This decision does not repeat that control.

One release-profile run of `probe_small_fold` on this host timed a different workload: 32 rows, 4 columns, 24 training rows, hidden width 8, 4 epochs, seed 11, `NdArray<f32>`. Those clocks do not enter the 2× comparison.

```text
prepare_ns=13625
upload_ns=79625
fit_ns=1687418
predict_ns=2208
synchronize_ns=15167
download_ns=18167
assemble_ns=125
```
