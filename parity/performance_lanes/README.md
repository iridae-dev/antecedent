# Performance lanes

`lanes.toml` registers the four release lanes before optimization. Proposals are goals. Nothing in this directory is a measurement. The predecessor is published **v2.4.1**. Historical timings in `benches/baselines/` stay out of these lanes.

Each lane names the control ids it may use (`parity/controls/workloads.toml`), which of those are `missing_entry_point` (none now), the measurement-record fields that must be filled, and a RAM, time, and disk envelope. Envelopes are requirements. Every lane has status `registered_not_measured`.

The timed section includes the uncertainty and diagnostics that control already computes. A point estimate by itself does not complete the timed section.

Measurement fields, from the native measurement record: `peak_rss`, `copy_bytes`, `source_passes`, `fit_count`, `test_count`, `retained_outputs`, `spill_bytes`, `threads`, `stage_times_ns`.

## first_answer

Proposed 2–5× faster first answer on selected complete workloads, versus frozen same-host v2.4.1.

Controls: `adjustment`, `glm`, `aipw`, `bootstrap`, `gaussian_bayes`, `nonlinear_bayes`, `pc`, `fci`, `pcmci`, `panel`, `survival`, `policy`, `transport`, `counterfactual`, `decision`.

Missing entry point: none.

Envelope: same host as that frozen control; the control's own timed section; spill and retention recorded, with no added disk budget.

## reuse_throughput

Proposed 10× throughput for compatible query, retarget, and refresh. The comparison names both baselines: an independent v2.4.1 solo call, and the existing v2.4.1 prepared/batch API.

Solo controls: `adjustment`, `aipw`. Prepared control: `prepared_query_families`. Also `refresh` and `decision`.

Missing entry point: none.

Envelope: resident set of one independent call plus declared prepared state; repeated query, retarget, and refresh against both baselines; declared retention of prepared state.

## row_capacity

Proposed 10× rows on named fixed-width lanes, under the same RAM envelope and the declared time envelope, with the complete v2.4.1 result.

Controls: `adjustment`, `glm`, `aipw`, `bootstrap`, `gaussian_bayes`, `panel`, `transport`, `survival`, `policy`.

Missing entry point: none.

Envelope: the named control's RAM envelope, its declared time envelope, and its retention requirements.

## exact_capacity

One streaming reduction. Proposed exact 100×, with a fixed RAM ceiling and declared time and disk. The lane starts from fixed-width weighted Gaussian and OLS sufficient statistics (`X'WX` style summaries). Declared diagnostic passes (robust covariance, leverage, residual checks) are part of the timed section. Result and retention requirements match the v2.4.1 control. Gaussian lowering applies to those weighted Gaussian and OLS likelihoods. Arbitrary nonlinear likelihoods keep their own scans.

Controls: `glm`, `gaussian_bayes`. Missing entry point: none.

Envelope: a fixed RAM ceiling for the reduction and diagnostic state; a declared time envelope covering both; declared disk for the scan, spill, and outputs.

## Entry points

Each control has one or more `[[entry_point]]` rows in `lanes.toml`: a `frozen_command` against the published v2.4.1 build and a `candidate_command` against the current build, on the same host, with identical data, seeds, requested uncertainty and diagnostics inside the timed section. Controls that were already defined run their `workloads.toml` command in both builds. The controls that had no entry point have scripts in `entry_points/`:

| Control | Python | Rust |
| --- | --- | --- |
| `nonlinear_bayes` | `entry_points/nonlinear_bayes.py` (Bernoulli-logit HMC, ESS and R-hat gates, measured draw control) | `crates/antecedent-prob/benches/hmc_logistic.rs` |
| `fci` | `entry_points/fci.py` | `crates/antecedent-discovery/benches/fci_search.rs` |
| `survival` | `entry_points/survival.py` (399 bootstrap replicates, pointwise intervals) | |
| `policy` | `entry_points/policy.py` (doubly robust values with intervals) | |
| `prepared_query_families` | `entry_points/prepared_query_families.py` (prepare, estimate, eight simultaneous-interval retargets) | |

```bash
python parity/performance_lanes/entry_points/survival.py --role control    # published v2.4.1 environment
python parity/performance_lanes/entry_points/survival.py --role candidate  # current build
```

`--rows` sets the row tier, `--threads` the thread count and `--repeats` the repeat count. Every repeat is printed, with a digest of the input columns so a pair can be checked for identical data. The scripts refuse to run under the wrong build. The Rust benches run unchanged in the v2.4.1 worktree: `stage_frozen.py` copies the bench or example file the tag lacks and appends its `[[bench]]` stanza, so nothing is edited by hand (see Run).

## Check

From the repository root:

```bash
python3 parity/performance_lanes/check_lanes.py
```

The checker reads this table and the control ids. It does not import antecedent and does not invoke cargo. It fails on any lane control that lacks a frozen command, a candidate command, or an existing entry file, and on a control still marked `missing_entry_point`. Its tests:

```bash
python3 parity/performance_lanes/test_check_lanes.py
```

## Run

One command stages the frozen build, verifies every entry point, runs the four lanes and writes the report:

```bash
python3 parity/performance_lanes/run_lanes.py                    # all four lanes, whole grid
python3 parity/performance_lanes/run_lanes.py plan               # print every command, run nothing
python3 parity/performance_lanes/run_lanes.py --lanes first_answer --threads 1 --repeats 5
```

The steps can be run alone (`prepare`, `verify`, `run`, `report`). The harness lives in this directory:

| File | Job |
| --- | --- |
| `stage_frozen.py` | A detached worktree of `v2.4.1`; the bench, example and script files the tag lacks, copied from the current tree with their `[[bench]]` stanzas appended in the frozen crate manifests (recorded in `.staged_entry_points.json`); one environment with the published `antecedent==2.4.1` wheel (or the tag built from source with `--frozen-from-source`) and one with this tree built by maturin. `--dry-run` prints the plan. |
| `verify_entry_points.py` | `--level static` checks every lane control for a frozen and a candidate command, existing and compiling files, registered benches in both trees and the `--role` pairing. `--level import` also runs each Python entry point's `--help` under the interpreter of its role and checks the build. `--level build` also compiles every Rust bench and example in both trees without running them. |
| `run_lanes.py` | The grid and the run. One OS process per repeat and arm, interleaved. Arms: `frozen`, `candidate`, `frozen_solo` (the reuse lane's independent-call baseline for the prepared family), and `scalar_direct`, the current tree with the scalar reference kernels (registered once in the `[scalar_direct]` table of `lanes.toml` and read by `KernelPolicy::production_policy`; the prepared query family control also runs as independent calls). |
| `measure.py` | `wait4` resource usage: peak resident set (native allocations included), CPU time, faults, wall time, timeouts and out-of-memory classification. |
| `lane_verdicts.py` | The goal tests. A goal is met only when both arms completed on the same data with the same result accounting; a repeat is never dropped. |
| `lanes_common.py` | Command parsing, tiers, the row-count mechanism of each command. |
| `entry_points/run_example.py` | Runs a repository example whole inside the timed section, in either build. |

Native resource measurement sits inside the timed procedure. Python entry points read the process peak resident set, CPU time and page faults immediately before and after the timed section (`getrusage` counts native allocations, not Python's). Every process is also measured from outside by the operating system, which is the only reading available for a Criterion bench. The candidate's engine counters (scans, copies, copied bytes, model fits, spill, borrowed bytes) are recorded whenever the result exposes `execution_facts` or `performance`; a counter the engine did not report is absent, never zero.

Lane tests, by lane:

- `first_answer`: median section time of `frozen` over `candidate` is at least the registered goal, same data, same result accounting.
- `reuse_throughput`: the candidate's prepared family against both `frozen_solo` and the frozen prepared API; other controls against the frozen call.
- `row_capacity`: the candidate completes the tier one decade above the largest tier the frozen build completes, inside the frozen build's peak resident set and `--time-factor` times its time.
- `exact_capacity`: the same, two decades above.

Every control of the two capacity lanes takes a row count: `--rows` for the Python entry points (`adjustment.py`, `aipw.py`, `panel.py`, `survival.py`, `policy.py`), `USER_PATH_N` for the Gaussian Bayes bench, and `ANTECEDENT_LANE_ROWS` for the GLM bench, the bootstrap bench and the transport example (unset, each keeps its registered size; staging replaces the frozen tree's copy of those three files with the current one, which differs only in reading that variable). The first-answer and reuse lanes run each control once at its registered size under tier `control_default`. Cells above a tier that every arm failed are recorded as not attempted. Out-of-memory, timed-out and failed repeats are written, never dropped.

Output, under `target/performance_lanes/runs/<timestamp>/` (`--work-dir`, `--run-dir`): `plan.txt`, `raw/` (every process's stdout and stderr), `cells.json` (every repeat), `resource/<lane>/<cell>/resource.json` (the format `scripts/bench_regression.py <criterion-dir> <run>/resource` prints), `summary.json` and `summary.md`. The summary also carries the wall-clock before and after of the bootstrap control, from the frozen build and this tree. Native counters for the Rust controls come from measured companions: `laplace_glm_measured`, `propensity_bootstrap_measured` and `transport_measured` repeat the control's work under `antecedent_core::measurement_record::measure` with the counting allocator (`antecedent_kernels::native_probe`) and write `measurement.json`; the runner runs each once per candidate and scalar cell and merges allocation bytes and calls, live-byte peak, copied bytes, scans, fits, tests, retained and spill bytes. Counters the run did not cross are listed as unobserved, not zero. The Criterion benches themselves do not install the allocator, so their timing is not perturbed.

Builds. `--build portable` (default) is the ordinary compile. `--build runtime_dispatched` compiles the same source with `--dispatch-rustflags` (default `-C target-cpu=native`) into its own environment and target directory. `--build both` runs the whole grid twice, once per candidate build against the same frozen build, under `<run>/portable` and `<run>/runtime_dispatched`.

The harness tests:

```bash
python3 parity/performance_lanes/test_lane_harness.py
```

## Lifecycle counter comparison

`benchmarks/performance/compare_lifecycle.py` replays fixed workloads and records exact counters in `lifecycle_counters.json` (sorted keys, no timestamps, byte-identical across runs). Workloads: prepare, estimate, rebind and estimate branches; batch shared fit; nuisance fit, repeat and artifact load; trial transport estimate, snapshot branch and export; cyclic family and source branching; retained refresh append, window, evict, replace, summary-only and spill; ingestion copy controls.

Recorded per step: attempted, completed, failed and unfinished native operations from `observe_native_attempts`; `native_fits` (attempted operations whose kind ends in `_fit`); execution-fact fits, copies, retained bytes and peak admitted bytes where a result carries them (cyclic); retained frame bytes (refresh); artifact bytes; Python-side ingestion copies. Counters a result does not measure stay `null`, never zero.

Limits. The seven workloads above are current tree only. A before/after comparison covers three of them, under the `baseline_comparison` key of the same file, written by `benchmarks/performance/compare_lifecycle_baseline.py` (`run --side baseline|current`, then `merge`; a rerun of `compare_lifecycle.py` rewrites the file and drops that key until `merge` is run again). The predecessor side runs the predecessor checkout (`python-api-baseline`, commit `844c3ac5`) with its own built extension through `PYTHONPATH`, same data seeds, graph, estimator, explicit seeds (3; 13 for the nuisance workload) and call order; only call names change (`estimate(data)`/`refresh` on a mutable study versus `rebind` and `estimate`; `PreparedBatch.prepare`/`estimate(data)` versus `prepare_many`/`estimate_many`; the study method versus the module function for the nuisance score).

Counters compared are attempted native operations by kind from `observe_native_attempts`, which both sides expose. Calls are aligned in phases because the APIs divide work differently: the predecessor `refresh` replaces data and estimates in one call, compared with the current `rebind` plus `estimate`; the predecessor `PreparedBatch.prepare` fits eagerly, the current `prepare_many` only identifies. Per-phase verdicts therefore move work between phases, so read the workload `total` verdicts. Verdicts: `equal`, `improved` (current attempts fewer), `WORSE` (current attempts more), `not_comparable`. Answers (point estimate and analytic standard error per query, nuisance score values) agree to 1e-9 on all three workloads.

Honest limits. The predecessor does not instrument its nuisance-score fit, repeat or refit on artifact load, so those phases show no operations there; they are `not_comparable` and excluded from the totals rather than read as zero work. Prepare-time fits in the predecessor batch are counted (its `prepare` and each `estimate(data)` fit separately), so the batch fit reduction is from not fitting at prepare, not from sharing across queries: the shared-propensity fit count per estimate is equal (5 fits for two queries on both sides). The batch workload is `WORSE` in program compilation (2 to 4) and score construction (8 to 10): the current branch compiles its plans on first execution and again on the rebound data, where the predecessor compiled once at prepare and reused the plan for new rows. Counters are exact for these workloads and host-independent; they are not a wall-clock claim. Wall-clock medians are produced only under `--timing` with at least five repeats and a host load average below 3; the recorded run was skipped for load, so the file holds no timings. Nothing here is a reference-host performance, capacity or calibration claim.
