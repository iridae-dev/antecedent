# Workload controls

`workloads.toml` names one end-to-end control for each family in the 3.0 workload suite: adjustment, GLM, AIPW, prepared query families, bootstrap, Gaussian Bayes, nonlinear Bayes, PC, FCI, PCMCI, panel, survival, policy, transport, counterfactual, decision, refresh, and neural crossfit.

The predecessor is the published tag **v2.4.1**. Commands are the entry points already in this repository. Historical wall times in `benches/baselines/` are not these controls, and this directory does not store measurements.

## Run one control

From the repository root, copy one `command` from `workloads.toml` and run it. A `defined` row is specified and not yet measured. Time the section described in `timed_work`. That section includes the uncertainty and diagnostics the command already computes, not a point estimate alone.

```bash
python examples/python/propensity_weighting.py
```

A row may also carry `candidate_command`: the command for the current build when it differs from the frozen `command` only in the build role (see `parity/performance_lanes/entry_points/`).

`missing_entry_point` rows have no such command. `nearest` is the closest existing file. Do not treat that file as the control.

Each defined row also names the `uncertainty` it requests, the `diagnostics` it computes and the `resource_scope` of its native measurement (the process peak resident set, CPU time and faults, read around the timed section for the Python controls and from the operating system for the benches). `python3 parity/performance_lanes/run_lanes.py` runs every control in both builds with those measurements.

Check the table without running the workloads:

```bash
python3 parity/controls/check_workloads.py
```
