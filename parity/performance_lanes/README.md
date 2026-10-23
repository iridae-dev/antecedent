# Performance lanes

`lanes.toml` registers the four release lanes before optimization. Proposals are goals. Nothing in this directory is a measurement. The predecessor is published **v2.4.1**. Historical timings in `benches/baselines/` stay out of these lanes.

Each lane names the B06 control ids it may use (`parity/controls/workloads.toml`), which of those are `missing_entry_point` (none now), the measurement-record fields that must be filled, and a RAM, time, and disk envelope. Envelopes are requirements. Every lane has status `registered_not_measured`.

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

`--rows` sets the row tier, `--threads` the thread count and `--repeats` the repeat count. Every repeat is printed, with a digest of the input columns so a pair can be checked for identical data. The scripts refuse to run under the wrong build. The Rust benches run unchanged in a v2.4.1 worktree once the bench file and its Cargo.toml entry are added.

## Check

From the repository root:

```bash
python3 parity/performance_lanes/check_lanes.py
```

The checker reads this table and the B06 control ids. It does not import antecedent and does not invoke cargo. It fails on any lane control that lacks a frozen command, a candidate command, or an existing entry file, and on a control still marked `missing_entry_point`. Its tests:

```bash
python3 parity/performance_lanes/test_check_lanes.py
```
