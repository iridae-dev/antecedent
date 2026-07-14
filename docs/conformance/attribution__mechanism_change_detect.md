# Mechanism-change detection

**Suite path:** `conformance/attribution/mechanism_change_detect`

Same synthetic periods as distribution_change_y_shift. Detection should flag
`y` as changed under mean_diff; attribution remains a separate call.

## Expected summary

Top-level keys: `changed, method, significance_level, targets` (4 fields).
