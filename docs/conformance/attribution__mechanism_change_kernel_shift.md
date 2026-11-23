# Mechanism-change detection (kernel two-sample)

**Suite path:** `conformance/attribution/mechanism_change_kernel_shift`

Same synthetic periods as `mechanism_change_detect`. Detection should flag
`y` as changed under `kernel_two_sample` (MMD on residuals).

## Expected summary

Top-level keys: `changed, evidence_kind, evidence_proposition, evidence_roles, evidence_source, limitations, method, oracle, significance_level, targets` (10 fields).
