# Measurement protocol

This directory is the measurement protocol for the four B08 lanes. The protocol is not a measurement. `protocol.toml` names the plan and stores no timings, speedups, or byte counts. Every lane has status `registered_not_measured`.

The predecessor is published **v2.4.1**. Controls, missing entry points, and RAM/time/disk envelopes stay in `parity/performance_lanes/lanes.toml`. Historical timings in `benches/baselines/` are a different record.

## Hardware

Reference classes are Apple Silicon (`apple_silicon`) and x86-64 Linux (`x86_64_linux`). Thread counts are 1, 4, and 16. Builds are `portable` and `runtime_dispatched`. Each combination is its own cell. Device work stays on the host CPU; larger-server and device runs are outside this grid.

Held fixed inside a cell: the frozen v2.4.1 control on that same host, the hardware class, the thread count, the build, the control id and its seed, the timed section (uncertainty and diagnostics included), encoded width on the fixed-width lanes, the result and retention contract, and the control's uncertainty budget. The row tier does not shrink replicates, draws, or diagnostic passes.

## Sample size

The sample-size and row tiers are `1e4`, `1e5`, `1e6`, and `1e7`. Those names are the plan, not results. A finished smaller tier does not stand in for a larger tier that missed its lane goal.

## Noise and retry

Reference-host cells are the protocol. A shared-runner smoke check is not one of them.

The Criterion comparison fails when the relative-change lower bound exceeds +25% and warns when the point estimate exceeds +10% with a lower bound above zero. A lower bound at or below zero is noise on that comparison, not a failure. Every repeat stays in the record and is reduced to min, median, and max. An even count uses the floor midpoint of the two central values. Repeats that miss, run out of memory, or cancel are not dropped, and the run does not stop early in order to drop them.

A retry repeats the same cell and appends a record. It does not delete the earlier attempt, and it does not change the tier, the thread count, the uncertainty budget, or the diagnostic passes. A control with no entry point is recorded as unsuccessful. The nearest example is not the control.

## Small-workload ceiling

The small-workload tier is `1e4`. When a resource attempt carries a relative comparison, the ceiling uses the same bounds as the Criterion comparison: a lower bound above +25% is beyond the ceiling, and a point estimate above +10% with a lower bound above zero warns. That report is not the script's exit status. A missing comparison is not filled in.

## Precision and output

Counters are non-negative integers. Zero means none reported. Peak RSS is `peak_rss_bytes` or a non-empty `peak_rss_unsupported` reason. Stage times are integer nanoseconds; a stage missing from a repeat is omitted. Variability is count, min, median, and max. The scientific output is the control's uncertainty and diagnostics. A point estimate by itself does not complete a cell.

Native fields match `scripts/bench_regression.py`: `peak_rss_bytes`, `allocation_bytes`, `copy_bytes`, `borrow_bytes`, `source_passes`, `fit_count`, `test_count`, `retained_outputs`, `spill_bytes`, `threads`, `peak_rss_unsupported`, `device`, `stage_times_ns`, `cancelled`, `failed`, and `variability`. Keys from a Python allocator trace (`python_`…) are ignored.

Each attempt also names `lane`, `control`, `hardware_class`, `build`, `threads`, `row_tier`, `outcome` (`completed`, `unsuccessful`, or `oom`), and `target_met`.

## Unsuccessful runs and missed targets

Unsuccessful and OOM attempts are written and printed. They are not dropped. Cancellation stays on the attempt (`cancelled` true, outcome `unsuccessful`). A refusal, a timeout, and a missing entry point are unsuccessful records.

A missed lane target stays on the record (`target_met` false). The next attempt changes the plan that was supposed to reach that end-state. It does not substitute an easier workload: a smaller tier, a narrower control, a point estimate without its uncertainty, a dropped diagnostic pass, or the nearest example. Completing `1e6` does not replace a missed 10× row goal or a missed 100× reduction goal.

## Lanes

`first_answer`. Proposed 2–5× faster first answer on the lane's selected complete workloads, versus frozen same-host v2.4.1. The timed section is the complete first answer.

`reuse_throughput`. Proposed 10× throughput for compatible query, retarget, and refresh, against both the independent v2.4.1 solo call and the existing prepared/batch API. Dropping either baseline is not the cell.

`row_capacity`. Proposed 10× rows on named fixed-width lanes, under that control's RAM envelope and declared time envelope, with the complete result. Encoded width stays at the control.

`exact_capacity`. Proposed exact 100× on one streaming reduction (fixed-width weighted Gaussian and OLS summaries), with declared diagnostic passes still in the timed section, under a fixed RAM ceiling and declared time and disk. Nonlinear likelihoods are not this lane.

## Criterion path

`scripts/bench_regression.py` keeps the Criterion relative-change comparison as its exit status (`FAIL_LOWER_BOUND` 0.25, `WARN_POINT` 0.10). End-to-end attempts are a second directory of `resource.json` files, printed beside that comparison. Omitting the directory, or pointing it at a directory with no `resource.json`, does not fail the Criterion comparison.

```bash
python3 scripts/bench_regression.py [CRITERION_DIR] [RESOURCE_DIR]
```
