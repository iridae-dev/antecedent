# Observation primitive calibration

**Suite path:** `conformance/response/observation_primitives`

This fixture freezes small paper-equation calculations for selected-outcome
AIPW pseudo-values, marginal Kaplan–Meier inverse-censoring weights (with and
without delayed entry), and exact/censored/truncated Gaussian likelihood terms.
It is an independent numerical calibration fixture, not an external package
parity claim.

## Expected summary

Top-level keys: `contract, fixture_id, gaussian_observation_likelihood, kaplan_meier_ipcw, selected_outcome, tolerance` (6 fields).
