# Masked MCI lag-1 conformance

**Suite path:** `conformance/discovery/masked_mci_lag1`

Analysis mask hides every 7th row; PCMCI must still recover `x@1 → y`.
Comparison class: Exact. Oracle: see `expected.json` → `reference`.

## Expected summary

Top-level keys: `alpha, fdr, generation, historical_generation, historical_oracle, historical_reference, mask, max_lag, n, oracle, reference, scm, tolerance_class, true_parents` (14 fields).
