# Masked MCI lag-1 conformance fixture

**Suite path:** `conformance/tigramite/masked_mci_lag1`

Synthetic Exact edge-set pin for `tigramite.data.masks`.
Analysis mask hides every 7th row; PCMCI must still recover `x@1 → y`.

Comparison class: Exact (recovered lagged edge set).

## Expected summary

Top-level keys: `alpha, fdr, generation, mask, max_lag, n, scm, tolerance_class, true_parents` (9 fields).
