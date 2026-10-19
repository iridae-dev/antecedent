# Accelerator decision

G03 stays deferred. The opt-in `wgpu` feature compiles Burn's WGPU path beside the default `NdArray<f32>` wheel, and it is not promoted. `probe_wgpu_small_fold` returns `NotLinked::StageBoundariesPrivate` and does not read a clock: `fit_portable_wgpu` fuses prepare, upload, fit, synchronize, and download, so those stages are not separately public. No same-host staged comparison against the frozen threshold was taken. This is a feasibility deferral, not a negative GPU result. G04–G06 stay closed.

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
