# Manufacturing temporal effect conformance fixture

**Suite path:** `conformance/manufacturing/temporal_pressure_defect`

#
# Synthetic lagged SCM: defect_t = 0.9 * pressure_{t-1}.
# Used by Rust `manufacturing_temporal` test and Python examples.

## Expected summary

Top-level keys: `edges, expected_ate, horizon_steps, n, note, outcome, pulse_active_level, sampling_interval_ns, scm, tolerance_class, treatment, treatment_lag, true_effect_per_unit` (13 fields).
