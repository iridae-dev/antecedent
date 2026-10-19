# Workload controls

`workloads.toml` names one end-to-end control for each family in the 3.0 workload suite: adjustment, GLM, AIPW, prepared query families, bootstrap, Gaussian Bayes, nonlinear Bayes, PC, FCI, PCMCI, panel, survival, policy, transport, counterfactual, decision, refresh, and neural crossfit.

The predecessor is the published tag **v2.4.1**. Commands are the entry points already in this repository. Historical wall times in `benches/baselines/` are not these controls, and this directory does not store measurements.

## Run one control

From the repository root, copy one `command` from `workloads.toml` and run it. A `defined` row is specified and not yet measured. Time the section described in `timed_work`. That section includes the uncertainty and diagnostics the command already computes, not a point estimate alone.

```bash
python examples/python/propensity_weighting.py
```

`missing_entry_point` rows have no such command. `nearest` is the closest existing file. Do not treat that file as the control.

Check the table without running the workloads:

```bash
python3 parity/controls/check_workloads.py
```
